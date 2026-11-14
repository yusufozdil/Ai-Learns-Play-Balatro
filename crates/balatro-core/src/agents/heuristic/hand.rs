//! Hand expert: Play vs Discard, which cards, consumables during the round.
//!
//! 1. Preview every selection; if one beats the blind, play the cheapest-side-effect winner
//!    (fewest cards under The Tooth, not The Ox's hand, otherwise the highest score).
//! 2. Otherwise generate a few structured discards (keep the best hand's scoring cards, chase a
//!    flush, a straight, pairs; cycle the lowest cards), estimate each by Monte Carlo (random
//!    draws from the draw pile, best previewed play of the resulting hand), and discard when the
//!    estimate beats playing the current best hand by a margin that shrinks as hands run out.

use super::consumables::{self, Ctx};
use super::jokers;
use super::valuer::{draw_random, sort_like_hand};
use super::{HeuristicAgent, HeuristicConfig};
use crate::actions::Action;
use crate::card::Card;
use crate::content::ids::BossId;
use crate::enums::{Enhancement, HandType, Seal, Suit};
use crate::game::Game;
use crate::hand_eval;
use crate::state::GameState;

/// A previewed selection.
#[derive(Debug, Clone, Copy)]
struct Play {
    mask: u32,
    score: f64,
    ht: HandType,
    cards: u32,
}

/// Value of keeping a card (rough chips plus enhancement bonuses); debuffed cards are junk.
fn card_value(c: &Card) -> f64 {
    if c.debuffed {
        return -5.0;
    }
    let mut v = f64::from(c.base_chips()) + f64::from(c.perma_bonus) / 5.0;
    v += match c.enhancement {
        Enhancement::None => 0.0,
        Enhancement::Stone => 2.0,
        _ => 6.0,
    };
    if c.edition != crate::enums::Edition::Base {
        v += 6.0;
    }
    if c.seal != Seal::None {
        v += 5.0;
    }
    v
}

fn scoring_positions(s: &GameState, mask: u32) -> u32 {
    let hand = &s.deck.hand;
    let sel: Vec<Card> = hand
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < 16 && (mask >> i) & 1 == 1)
        .map(|(_, c)| *c)
        .collect();
    let ev = hand_eval::evaluate(&sel, &s.mods.rules);
    let mut out = 0u32;
    let mut j = 0;
    for i in 0..hand.len().min(16) {
        if (mask >> i) & 1 == 1 {
            if ev.scores(j) {
                out |= 1 << i;
            }
            j += 1;
        }
    }
    out
}

/// Mask of one card per present rank inside the best 5-rank window (Ace low or high), and how
/// many ranks the window holds.
fn straight_window(hand: &[Card]) -> (usize, u32) {
    let mut pos = [None::<usize>; 15];
    for (i, c) in hand.iter().enumerate().take(16) {
        if c.is_stone() || c.debuffed {
            continue;
        }
        let v = c.rank as usize;
        if pos[v].is_none() {
            pos[v] = Some(i);
        }
        if v == 14 && pos[1].is_none() {
            pos[1] = Some(i);
        }
    }
    let mut best = (0usize, 0u32);
    for start in 1..=10usize {
        let mut k = 0;
        let mut mask = 0u32;
        for p in &pos[start..start + 5] {
            if let Some(i) = *p
                && mask & (1 << i) == 0
            {
                k += 1;
                mask |= 1 << i;
            }
        }
        if k > best.0 {
            best = (k, mask);
        }
    }
    best
}

/// Candidate discard masks (1..=5 cards each), deduplicated, ascending.
fn discard_candidates(s: &GameState, best_mask: u32, forced: Option<usize>) -> Vec<u32> {
    let hand = &s.deck.hand;
    let n = hand.len().min(16);
    let rules = &s.mods.rules;
    let all: Vec<usize> = (0..n).collect();
    let mut out: Vec<u32> = Vec::new();

    let from_keep = |keep: u32| -> Option<u32> {
        let mut d: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&i| (keep >> i) & 1 == 0)
            .collect();
        d.sort_by(|&a, &b| {
            card_value(&hand[a])
                .partial_cmp(&card_value(&hand[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        d.truncate(5);
        let m = d.iter().fold(0u32, |m, &i| m | (1 << i));
        (m != 0).then_some(m)
    };

    let mut keeps: Vec<u32> = Vec::new();
    keeps.push(scoring_positions(s, best_mask));
    // Flush chase.
    for suit in Suit::ALL {
        let mut m = 0u32;
        let mut k = 0;
        for (i, c) in hand.iter().enumerate().take(16) {
            if !c.debuffed && c.has_suit(suit, rules) {
                m |= 1 << i;
                k += 1;
            }
        }
        if k >= 3 {
            keeps.push(m);
        }
    }
    // Straight chase.
    let (k, m) = straight_window(hand);
    if k >= 3 {
        keeps.push(m);
    }
    // Multiples: keep every card whose rank occurs at least twice.
    let mut counts = [0u8; 15];
    for c in hand.iter().take(16) {
        if !c.is_stone() && !c.debuffed {
            counts[c.rank as usize] += 1;
        }
    }
    let mut grp = 0u32;
    for (i, c) in hand.iter().enumerate().take(16) {
        if !c.is_stone() && !c.debuffed && counts[c.rank as usize] >= 2 {
            grp |= 1 << i;
        }
    }
    if grp != 0 {
        keeps.push(grp);
    }
    for keep in keeps {
        if let Some(m) = from_keep(keep) {
            out.push(m);
        }
    }
    // Cycle the three lowest cards.
    if let Some(m) = from_keep(0) {
        let low3 = (0..16)
            .filter(|i| (m >> i) & 1 == 1)
            .take(3)
            .fold(0u32, |a, i| a | (1 << i));
        out.push(low3);
        out.push(m);
    }
    if let Some(p) = forced {
        for m in &mut out {
            if (*m >> p) & 1 == 0 {
                if m.count_ones() >= 5 {
                    // Drop the highest-position bit that is not the forced card.
                    if let Some(drop) = (0..16).rev().find(|&i| i != p && (*m >> i) & 1 == 1) {
                        *m &= !(1 << drop);
                    }
                }
                *m |= 1 << p;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl HeuristicAgent {
    /// All valid selections with score, exact (every selection, no pruning).
    fn all_plays(&mut self, game: &Game) -> Vec<Play> {
        let n = game.state().deck.hand.len().min(16);
        let mut v = Vec::new();
        for mask in 1u32..(1u32 << n) {
            if mask.count_ones() > 5 {
                continue;
            }
            if let Ok(r) = game.preview_play_with(&mut self.scorer, mask) {
                v.push(Play {
                    mask,
                    score: if r.invalid { 0.0 } else { r.score },
                    ht: r.hand_type,
                    cards: mask.count_ones(),
                });
            }
        }
        v
    }

    /// Monte-Carlo estimate for each candidate discard: `(mean best score after, mean of
    /// min(best, need))` over random draws from the draw pile.
    fn mc_discards(&mut self, game: &Game, cands: &[u32], need: f64) -> Vec<(f64, f64)> {
        let s = game.state();
        let mut sim = Game::from_state(s.clone());
        // After the discard: one discard fewer (Banner, Mystic Summit read the count).
        sim.state_mut().discards_left = s.discards_left.saturating_sub(1);
        sim.state_mut().discards_used_round += 1;
        let pile = s.deck.draw_pile.clone();
        let mut buf = Vec::with_capacity(pile.len());
        let draws: Vec<Vec<Card>> = (0..self.cfg.mc_samples)
            .map(|_| draw_random(&mut self.rng, &pile, 5, &mut buf))
            .collect();
        let mut res = Vec::with_capacity(cands.len());
        for &m in cands {
            let k = m.count_ones() as usize;
            let kept: Vec<Card> = s
                .deck
                .hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i >= 16 || (m >> i) & 1 == 0)
                .map(|(_, c)| *c)
                .collect();
            let (mut sum_best, mut sum_min) = (0.0, 0.0);
            for d in &draws {
                let mut hand = kept.clone();
                hand.extend(d.iter().take(k).copied());
                sort_like_hand(&mut hand);
                let (sc, _, _) = self.valuer.best_play(&mut sim, &hand);
                sum_best += sc;
                sum_min += sc.min(need);
            }
            let n = draws.len().max(1) as f64;
            res.push((sum_best / n, sum_min / n));
        }
        res
    }

    pub(super) fn play_action(&mut self, game: &Game) -> Action {
        let s = game.state();
        let step = self.tick(s);
        let boss_on = s.boss_in_effect();

        // Joker order (free action, only before the first hand of a round).
        if s.hands_played_round == 0
            && s.discards_used_round == 0
            && step < 12
            && let Some(a) = self.arrange_action(game)
        {
            return a;
        }

        // Verdant Leaf debuffs every card until a joker is sold: sell the weakest one.
        if boss_on
            && s.boss == Some(BossId::VerdantLeaf)
            && step < 3
            && s.deck.hand.iter().all(|c| c.debuffed)
            && let Some((i, _)) = s
                .jokers
                .iter()
                .enumerate()
                .filter(|(_, j)| !j.stickers.eternal)
                .min_by(|(_, a), (_, b)| {
                    let va = jokers::prior(a.id) + f64::from(a.sell_value) * 0.01;
                    let vb = jokers::prior(b.id) + f64::from(b.sell_value) * 0.01;
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
            && let Ok(i) = u8::try_from(i)
        {
            return Action::SellJoker(i);
        }

        self.valuer.full = boss_on && s.boss == Some(BossId::ThePsychic);
        let plays = self.all_plays(game);
        let Some(best) = plays.iter().copied().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return Self::fallback(game);
        };

        // Consumables first (planets, tarots on the cards about to be played, ...).
        if step < 10 {
            let ctx = Ctx {
                dominant: self.strat.dominant,
                best_mask: Some(best.mask),
            };
            if let Some(a) = consumables::best_held_action(game, &ctx) {
                return a;
            }
        }

        let need = (s.target - s.round_score).max(0.0);
        if best.score >= need {
            return Action::Play(self.cheapest_winner(game, &plays, need).mask);
        }
        if s.discards_left == 0 {
            return Action::Play(best.mask);
        }

        if s.hands_left > 1 && best.score >= self.cfg.pace * need / f64::from(s.hands_left) {
            return Action::Play(best.mask);
        }
        let forced = crate::boss::forced_position(s);
        let cands: Vec<u32> = discard_candidates(s, best.mask, forced)
            .into_iter()
            .filter(|&m| game.validate(Action::Discard(m)).is_ok())
            .collect();
        if cands.is_empty() {
            return Action::Play(best.mask);
        }
        let est = self.mc_discards(game, &cands, need);
        let h = f64::from(s.hands_left);
        let d = f64::from(s.discards_left);
        let (idx, gain) = if s.hands_left <= 1 {
            let (i, e) = est
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, e)| (i, e.1))
                .unwrap_or((0, 0.0));
            (i, e / best.score.max(1.0))
        } else {
            let (i, e) = est
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.0
                        .partial_cmp(&b.1.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, e)| (i, e.0))
                .unwrap_or((0, 0.0));
            (i, e / best.score.max(1.0))
        };
        let thr = discard_threshold(&self.cfg, h, d);
        if gain > thr {
            Action::Discard(cands[idx])
        } else {
            Action::Play(best.mask)
        }
    }

    /// Among the plays that beat `need`: the one with the fewest side effects.
    fn cheapest_winner(&self, game: &Game, plays: &[Play], need: f64) -> Play {
        let s = game.state();
        let tooth = s.boss_in_effect() && s.boss == Some(BossId::TheTooth);
        let ox = if s.boss_in_effect() && s.boss == Some(BossId::TheOx) {
            s.boss_state.hand
        } else {
            None
        };
        plays
            .iter()
            .copied()
            .filter(|p| p.score >= need)
            .min_by(|a, b| {
                let ka = (
                    Some(a.ht) == ox,
                    if tooth { a.cards } else { 0 },
                    std::cmp::Reverse((a.score * 1000.0) as u64),
                );
                let kb = (
                    Some(b.ht) == ox,
                    if tooth { b.cards } else { 0 },
                    std::cmp::Reverse((b.score * 1000.0) as u64),
                );
                ka.cmp(&kb)
            })
            .unwrap_or(plays[0])
    }
}

/// Required relative gain of a discard over playing now: low when hands or discards are about to
/// run out, higher when plenty are left (the discards are better saved for later hands).
fn discard_threshold(cfg: &HeuristicConfig, hands_left: f64, discards_left: f64) -> f64 {
    if hands_left <= 1.0 {
        cfg.discard_last
    } else {
        cfg.discard_base + cfg.discard_slope * (hands_left - 1.0) / discards_left.max(1.0)
    }
}
