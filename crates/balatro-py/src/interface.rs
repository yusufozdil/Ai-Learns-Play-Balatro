//! TEMPORARY observation / reward / done interface (Phase 7B).
//!
//! Everything in this module is a placeholder that Phase 8 (observation + fixed-size action codec)
//! and Phase 10 (reward design) replace. The rest of the crate only talks to the env through
//! the functions here, so swapping them out does not touch the threading / buffer plumbing:
//!
//! * [`OBS_DIM`] + [`write_obs`]: a small fixed summary vector (Phase 8 replaces both, and adds a
//!   fixed-size action mask next to `legal_count`).
//! * [`step_reward`]: the placeholder reward (Phase 10 replaces it).
//! * `DONE_*` codes: kept (they are plumbing, not model design).

use balatro_core::blinds::BlindKind;
use balatro_core::enums::Phase;
use balatro_core::game::{Game, StepInfo};
use balatro_core::state::Outcome;

/// Observation vector length (float32).
pub const OBS_DIM: usize = 24;

/// Not done.
pub const DONE_NO: u8 = 0;
/// The run was lost.
pub const DONE_LOSS: u8 = 1;
/// The run was won (ante-8 boss beaten).
pub const DONE_WIN: u8 = 2;
/// The episode hit `max_episode_steps` and was cut off (neither win nor loss).
pub const DONE_TRUNCATED: u8 = 3;

/// Placeholder reward: +1 for every blind cleared ...
pub const REWARD_BLIND_CLEARED: f32 = 1.0;
/// ... and +10 more when the run is won; 0 otherwise.
pub const REWARD_WIN: f32 = 10.0;

/// Reward of one transition (PLACEHOLDER, Phase 10 designs the real signal).
pub fn step_reward(info: &StepInfo, game: &Game) -> f32 {
    let mut r = 0.0;
    if info.round.is_some_and(|rr| rr.won) {
        r += REWARD_BLIND_CLEARED;
    }
    if game.outcome() == Some(Outcome::Win) {
        r += REWARD_WIN;
    }
    r
}

/// Done code of the game right now (before the step-limit truncation check).
pub fn done_code(game: &Game) -> u8 {
    match game.outcome() {
        Some(Outcome::Win) => DONE_WIN,
        Some(Outcome::Loss) => DONE_LOSS,
        None if game.is_terminal() => DONE_LOSS,
        None => DONE_NO,
    }
}

fn log10p(x: f64) -> f32 {
    // Targets can be +inf at very high antes; keep the feature finite.
    (1.0 + x.clamp(0.0, 1e300)).log10() as f32
}

/// Write the placeholder observation of `game` into `out` (`out.len() == OBS_DIM`).
///
/// Layout: `[0..6)` phase one-hot (BlindSelect, Play, Shop, PackOpen, GameOver, Won); then ante,
/// round, blind kind (0 small / 1 big / 2 boss), money, log10(1+target), round_score/target
/// (0 when the target is infinite), log10(1+round_score), hands left, discards left, hand size,
/// cards in hand, jokers, joker slots, consumables, consumable slots, draw pile size, legal
/// action count, boss effect active.
pub fn write_obs(game: &Game, legal_count: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), OBS_DIM);
    let s = game.state();
    out.fill(0.0);
    let phase = match s.phase {
        Phase::BlindSelect => 0,
        Phase::Play => 1,
        Phase::Shop => 2,
        Phase::PackOpen => 3,
        Phase::GameOver => 4,
        Phase::Won => 5,
    };
    out[phase] = 1.0;
    out[6] = s.ante as f32;
    out[7] = s.round as f32;
    out[8] = match s.blind {
        BlindKind::Small => 0.0,
        BlindKind::Big => 1.0,
        BlindKind::Boss => 2.0,
    };
    out[9] = s.money as f32;
    out[10] = log10p(s.target);
    out[11] = if s.target.is_finite() && s.target > 0.0 {
        (s.round_score / s.target).clamp(0.0, 1e3) as f32
    } else {
        0.0
    };
    out[12] = log10p(s.round_score);
    out[13] = s.hands_left as f32;
    out[14] = s.discards_left as f32;
    out[15] = s.hand_size() as f32;
    out[16] = s.deck.hand.len() as f32;
    out[17] = s.jokers.len() as f32;
    out[18] = s.joker_slots as f32;
    out[19] = s.consumables.len() as f32;
    out[20] = s.consumable_slots as f32;
    out[21] = s.deck.draw_pile.len() as f32;
    out[22] = legal_count as f32;
    out[23] = if s.boss_in_effect() { 1.0 } else { 0.0 };
}
