//! Simulation-based valuation: previews sampled hands on a cloned game to measure how much a
//! change (a joker added/removed/moved, a hand level-up) changes the score the agent could make.
//!
//! The unit is `log2(best hand score)` averaged over sampled hands, so "+1.0" means "the hands
//! the build can make score twice as much". Samples are drawn once per evaluation context and
//! shared by every candidate (common random numbers), so differences are far less noisy than the
//! absolute numbers.

use crate::blinds::BlindKind;
use crate::card::{Card, RuleFlags};
use crate::enums::{HandType, Phase};
use crate::game::Game;
use crate::hand_eval::{self, HandEval};
use crate::joker::JokerInstance;
use crate::rng::{GameRng, RngKey};
use crate::scoring::Scorer;

/// Sort a hand like `Deck::sort_hand` (rank descending, suit, uid, Stone last).
pub fn sort_like_hand(cards: &mut [Card]) {
    cards.sort_by_key(|c| {
        (
            c.is_stone(),
            std::cmp::Reverse(c.rank as u8),
            c.suit.index(),
            c.uid,
        )
    });
}

/// Draw `k` distinct cards uniformly from `pool` with a partial Fisher-Yates shuffle of `buf`.
pub fn draw_random(rng: &mut GameRng, pool: &[Card], k: usize, buf: &mut Vec<Card>) -> Vec<Card> {
    buf.clear();
    buf.extend_from_slice(pool);
    let k = k.min(buf.len());
    for i in 0..k {
        let j = rng.range(RngKey::Misc, i, buf.len());
        buf.swap(i, j);
    }
    buf[..k].to_vec()
}

/// Selections worth previewing: one per distinct scoring set (the smallest selection that scores
/// exactly those cards). Kickers add no chips, so the others only matter through jokers that
/// look at the number of played cards or at held cards; the exact play uses all selections.
pub fn pruned_masks(
    hand: &[Card],
    rules: &RuleFlags,
    scratch: &mut Vec<(u16, HandEval)>,
    keyed: &mut Vec<(u16, u8, u16)>,
    out: &mut Vec<u16>,
    full: bool,
) {
    hand_eval::evaluate_all_subsets(hand, rules, scratch);
    out.clear();
    if full {
        out.extend(scratch.iter().map(|e| e.0));
        return;
    }
    keyed.clear();
    for &(mask, ev) in scratch.iter() {
        let mut key = 0u16;
        let mut j = 0usize;
        for i in 0..hand.len().min(16) {
            if (mask >> i) & 1 == 1 {
                if ev.scores(j) {
                    key |= 1 << i;
                }
                j += 1;
            }
        }
        keyed.push((key, mask.count_ones() as u8, mask));
    }
    keyed.sort_unstable();
    keyed.dedup_by_key(|e| e.0);
    out.extend(keyed.iter().map(|e| e.2));
}

/// Reusable scratch + scorer for simulations.
#[derive(Debug, Default)]
pub struct Valuer {
    scorer: Scorer,
    subsets: Vec<(u16, HandEval)>,
    keyed: Vec<(u16, u8, u16)>,
    masks: Vec<u16>,
    /// Preview every selection (needed while a boss demands exactly five cards).
    pub full: bool,
}

impl Valuer {
    /// New valuer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A game cloned from `game` and put in a neutral `Play` position (no boss, full hands and
    /// discards, nothing scored, empty hand) for previewing sampled hands.
    pub fn sim_game(game: &Game) -> Game {
        let mut g = Game::from_state(game.state().clone());
        let s = g.state_mut();
        s.phase = Phase::Play;
        s.blind = BlindKind::Small;
        s.hands_left = s.round_hands().max(1);
        s.discards_left = s.round_discards();
        s.hands_played_round = 0;
        s.discards_used_round = 0;
        s.round_score = 0.0;
        s.hand_plays_round = [0; HandType::COUNT];
        s.round_hand_size = 0;
        s.deck.hand.clear();
        g
    }

    /// Neutral simulation copy of an existing simulation game (clone only).
    pub fn sim_game_from(sim: &Game) -> Game {
        Game::from_state(sim.state().clone())
    }

    /// `n` random hands of `size` cards from the whole deck of `game`.
    pub fn sample_hands(game: &Game, rng: &mut GameRng, n: usize, size: usize) -> Vec<Vec<Card>> {
        let pool: Vec<Card> = game
            .state()
            .deck
            .iter_all()
            .map(|c| {
                let mut c = *c;
                c.debuffed = false;
                c.face_down = false;
                c
            })
            .collect();
        let mut buf = Vec::with_capacity(pool.len());
        (0..n)
            .map(|_| {
                let mut h = draw_random(rng, &pool, size, &mut buf);
                sort_like_hand(&mut h);
                h
            })
            .collect()
    }

    /// Best previewed score over the (pruned) selections of `hand` on `g`, with the mask and the
    /// hand type. `g` must be in `Play`; its hand is overwritten.
    pub fn best_play(&mut self, g: &mut Game, hand: &[Card]) -> (f64, u16, HandType) {
        {
            let h = &mut g.state_mut().deck.hand;
            h.clear();
            h.extend_from_slice(hand);
        }
        let rules = g.state().mods.rules;
        pruned_masks(
            hand,
            &rules,
            &mut self.subsets,
            &mut self.keyed,
            &mut self.masks,
            self.full,
        );
        let mut best = (0.0f64, 0u16, HandType::HighCard);
        for &m in &self.masks {
            if let Ok(r) = g.preview_play_with(&mut self.scorer, m) {
                let sc = if r.invalid { 0.0 } else { r.score };
                if sc > best.0 || best.1 == 0 {
                    best = (sc, m, r.hand_type);
                }
            }
        }
        best
    }

    /// Previewed score of playing exactly `cards` (all of them) on `g`; 0 if invalid.
    pub fn score_cards(&mut self, g: &mut Game, cards: &[Card]) -> f64 {
        {
            let h = &mut g.state_mut().deck.hand;
            h.clear();
            h.extend_from_slice(cards);
        }
        let mask = ((1u32 << cards.len()) - 1) as u16;
        match g.preview_play_with(&mut self.scorer, mask) {
            Ok(r) if !r.invalid => r.score,
            _ => 0.0,
        }
    }

    /// Mean of `log2(max(best score, 1))` over `hands`.
    pub fn mean_log2(&mut self, g: &mut Game, hands: &[Vec<Card>]) -> f64 {
        if hands.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for h in hands {
            sum += self.best_play(g, h).0.max(1.0).log2();
        }
        sum / hands.len() as f64
    }

    /// `(mean log2 best, histogram of the best hand type)` over `hands`.
    pub fn base_and_histogram(
        &mut self,
        g: &mut Game,
        hands: &[Vec<Card>],
    ) -> (f64, [f64; HandType::COUNT]) {
        let mut hist = [0.0; HandType::COUNT];
        let mut sum = 0.0;
        for h in hands {
            let (sc, _, ht) = self.best_play(g, h);
            sum += sc.max(1.0).log2();
            hist[ht.index()] += 1.0;
        }
        (sum / hands.len().max(1) as f64, hist)
    }
}

/// `sim` with `joker` inserted at `pos` (clamped), modifiers refreshed (no `on_acquire`).
pub fn with_joker(sim: &Game, joker: JokerInstance, pos: usize) -> Game {
    let mut g = Game::from_state(sim.state().clone());
    let s = g.state_mut();
    let pos = pos.min(s.jokers.len());
    s.jokers.insert(pos, joker);
    s.refresh_modifiers();
    g
}

/// `sim` without the joker at `idx`.
pub fn without_joker(sim: &Game, idx: usize) -> Game {
    let mut g = Game::from_state(sim.state().clone());
    let s = g.state_mut();
    if idx < s.jokers.len() {
        s.jokers.remove(idx);
    }
    s.refresh_modifiers();
    g
}

/// `sim` with the joker at `from` moved to `to` (the others shift), like `Action::MoveJoker`.
pub fn with_move(sim: &Game, from: usize, to: usize) -> Game {
    let mut g = Game::from_state(sim.state().clone());
    let s = g.state_mut();
    if from < s.jokers.len() && to < s.jokers.len() {
        let j = s.jokers.remove(from);
        s.jokers.insert(to, j);
    }
    s.refresh_modifiers();
    g
}

/// `sim` with `hand` leveled up `levels` times (every hand when `None`).
pub fn with_levels(sim: &Game, hand: Option<HandType>, levels: u32) -> Game {
    let mut g = Game::from_state(sim.state().clone());
    let s = g.state_mut();
    match hand {
        Some(h) => s.hand_levels[h.index()] += levels,
        None => {
            for l in &mut s.hand_levels {
                *l += levels;
            }
        }
    }
    g
}
