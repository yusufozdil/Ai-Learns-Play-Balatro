//! Greedy agent: plays the highest-scoring hand, discards toward a flush/straight or keeps the
//! scoring cards of its best hand when the blind cannot be beaten this turn.

use super::Agent;
use crate::actions::Action;
use crate::card::Card;
use crate::enums::{HandType, Phase, Suit};
use crate::game::Game;
use crate::hand_eval;
use crate::scoring::Scorer;

/// Deterministic (no RNG) one-ply greedy agent.
#[derive(Debug, Default)]
pub struct GreedyAgent {
    scorer: Scorer,
}

impl GreedyAgent {
    /// New greedy agent.
    pub fn new() -> Self {
        Self::default()
    }

    /// Best `Play` mask by previewed score (lowest mask wins ties), its score and hand type.
    fn best_play(&mut self, game: &Game) -> Option<(u16, f64, HandType)> {
        let n = game.state().deck.hand.len().min(16);
        let mut best: Option<(u16, f64, HandType)> = None;
        for mask in 1u32..(1u32 << n) {
            if mask.count_ones() > 5 {
                continue;
            }
            let mask = mask as u16;
            if let Ok(r) = game.preview_play_with(&mut self.scorer, mask) {
                let score = if r.invalid { 0.0 } else { r.score };
                if best.is_none_or(|(_, s, _)| score > s) {
                    best = Some((mask, score, r.hand_type));
                }
            }
        }
        best
    }

    /// Discard mask chosen by the heuristic, or `None` to play instead.
    fn discard_plan(game: &Game, best_mask: u16, best_type: HandType) -> Option<u16> {
        let s = game.state();
        let hand = &s.deck.hand;
        let rules = &s.mods.rules;
        if best_type >= HandType::Flush {
            return None;
        }
        // Flush chase: most common suit.
        let mut flush_k = 0usize;
        let mut flush_mask = 0u16;
        for suit in Suit::ALL {
            let mut m = 0u16;
            let mut k = 0usize;
            for (i, c) in hand.iter().enumerate().take(16) {
                if c.has_suit(suit, rules) {
                    m |= 1 << i;
                    k += 1;
                }
            }
            if k > flush_k {
                flush_k = k;
                flush_mask = m;
            }
        }
        // Straight chase: best window of 5 consecutive ranks (Ace low or high).
        let (straight_k, straight_keep) = best_straight_window(hand);

        let all = if hand.len() >= 16 {
            u16::MAX
        } else {
            (1u16 << hand.len()) - 1
        };
        // Limit to 5 cards (lowest positions first); `None` if nothing to discard.
        let cap = |m: u16| -> Option<u16> {
            let mut out = 0u16;
            let mut cnt = 0;
            for i in 0..16 {
                if m & (1 << i) != 0 && cnt < 5 {
                    out |= 1 << i;
                    cnt += 1;
                }
            }
            (out != 0).then_some(out)
        };
        if flush_k >= 4 || (flush_k >= 3 && best_type <= HandType::HighCard) {
            return cap(all & !flush_mask);
        }
        if straight_k >= 4 && best_type < HandType::Straight {
            return cap(all & !straight_keep);
        }
        if best_type >= HandType::Straight {
            return None;
        }
        // Keep the scoring cards of the best hand, throw away the rest.
        let sel: Vec<Card> = hand
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < 16 && best_mask & (1 << *i) != 0)
            .map(|(_, c)| *c)
            .collect();
        let eval = hand_eval::evaluate(&sel, rules);
        let mut keep = 0u16;
        let mut j = 0;
        for i in 0..hand.len().min(16) {
            if best_mask & (1 << i) != 0 {
                if eval.scores(j) {
                    keep |= 1 << i;
                }
                j += 1;
            }
        }
        cap(all & !keep)
    }
}

/// `(cards present in the best 5-rank window, mask of one card per present rank in it)`.
fn best_straight_window(hand: &[Card]) -> (usize, u16) {
    // Index 1..=14, Ace counts at 1 and 14.
    let mut pos = [None::<usize>; 15];
    for (i, c) in hand.iter().enumerate().take(16) {
        if c.is_stone() {
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
    let mut best = (0usize, 0u16);
    for start in 1..=10usize {
        let mut k = 0;
        let mut mask = 0u16;
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

impl Agent for GreedyAgent {
    fn act(&mut self, game: &Game) -> Action {
        let s = game.state();
        match s.phase {
            Phase::BlindSelect => Action::SelectBlind,
            Phase::Play => {
                let Some((mask, score, ht)) = self.best_play(game) else {
                    // No playable hand: take any legal action.
                    return game
                        .legal_actions()
                        .into_iter()
                        .next()
                        .unwrap_or(Action::EndShop);
                };
                let need = s.target - s.round_score;
                if score >= need || s.discards_left == 0 {
                    return Action::Play(mask);
                }
                // A boss may forbid the planned discard (Cerulean Bell's forced card).
                match Self::discard_plan(game, mask, ht) {
                    Some(d) if game.validate(Action::Discard(d)).is_ok() => Action::Discard(d),
                    _ => Action::Play(mask),
                }
            }
            Phase::Shop => Action::EndShop,
            _ => game
                .legal_actions()
                .into_iter()
                .next()
                .unwrap_or(Action::EndShop),
        }
    }
}
