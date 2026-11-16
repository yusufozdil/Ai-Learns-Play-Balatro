//! Build-power meter (idea G1): how well does the *current build* (jokers, deck, hand levels,
//! hands/discards, vouchers) fare against this ante's boss blind?
//!
//! `power(state, k, cfg)` clones the state, turns it into "the next blind is this ante's boss"
//! and plays that single round `k` times with the [`GreedyAgent`] play/discard policy, each time
//! on a freshly shuffled deck. `phi` is the mean over the `k` rounds of
//! `ln(clamp(round score / target, e^-3, e^3))`.
//!
//! * **Pure function of the state.** The input is never mutated. The `k` shuffle/RNG seeds come
//!   from a digest of the build (never from `state.rng`, never from the real draw order: the deck
//!   is re-shuffled from its canonical `uid` order), mixed with a caller-provided salt. The same
//!   state and salt always give the same report, and the game's own RNG streams are untouched.
//! * **v1 (`boss_aware = false`)**: the plain boss target (x2 base chips) with no boss effect.
//!   **v2 (`boss_aware = true`)**: this ante's actual boss, its multiplier and rules (the boss is
//!   chosen at the start of the ante and is already part of the state).
//! * Consumables are neither used nor removed, no shop, no packs; the round starts like a real
//!   `SelectBlind` (so round-start jokers/boss set-up behave as in the game).
//! * `play_on` (default): the round keeps going after the target is reached (the game's
//!   target is lifted out of the way), so a strong build gets a ratio well above 1 instead of the
//!   tiny overshoot of the hand that happened to cross the line. The policy still decides
//!   play/discard against the real target.

use super::greedy::GreedyAgent;
use crate::actions::Action;
use crate::blinds::BlindKind;
use crate::enums::Phase;
use crate::game::Game;
use crate::rng::GameRng;
use crate::state::GameState;
use serde::Serialize;

/// Lower clamp of `ln(score / target)`.
pub const LN_CLAMP: f64 = 3.0;
/// Action cap of one simulated round (a round has at most a few dozen plays and discards).
const MAX_ACTIONS: usize = 64;

/// Options of [`power`].
#[derive(Debug, Clone, Copy)]
pub struct PowerCfg {
    /// Measure against this ante's actual boss and its rules (v2) instead of a plain boss (v1).
    pub boss_aware: bool,
    /// Keep playing after the target is reached (see the module docs).
    pub play_on: bool,
    /// Mixed into the seeds (0 = none). Different salts give independent Monte-Carlo samples.
    pub salt: u64,
}

impl Default for PowerCfg {
    fn default() -> Self {
        PowerCfg {
            boss_aware: true,
            play_on: true,
            salt: 0,
        }
    }
}

/// Result of [`power`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerReport {
    /// Mean of the clamped `ln(score / target)` over the `k` simulated rounds.
    pub phi: f64,
    /// Standard deviation of the per-round values (population, 0 for `k = 1`).
    pub std: f64,
    /// Fraction of simulated rounds that reached the target ("would beat the boss").
    pub frac_beat: f64,
    /// Chip target of the simulated boss round.
    pub target: f64,
    /// Number of simulated rounds.
    pub k: usize,
}

/// FNV-1a over a byte stream.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// What the build is, without anything hidden: no RNG, no pile order, no shop.
#[derive(Serialize)]
struct Build<'a> {
    config: &'a crate::state::GameConfig,
    ante: i32,
    money: i32,
    base: [u32; 5],
    boss: &'a Option<crate::content::ids::BossId>,
    jokers: &'a [crate::joker::JokerInstance],
    levels: &'a [u32],
    vouchers: &'a [crate::content::ids::VoucherId],
    consumables: &'a [crate::consumable::ConsumableInstance],
    cards: Vec<&'a crate::card::Card>,
}

/// Digest of the build (deterministic, order-independent for the deck).
pub fn state_digest(s: &GameState) -> u64 {
    let mut cards: Vec<&crate::card::Card> = s
        .deck
        .draw_pile
        .iter()
        .chain(&s.deck.hand)
        .chain(&s.deck.played)
        .chain(&s.deck.discard_pile)
        .collect();
    cards.sort_unstable_by_key(|c| c.uid);
    let b = Build {
        config: &s.config,
        ante: s.ante,
        money: s.money,
        base: [
            s.base_hands,
            s.base_discards,
            s.base_hand_size,
            s.joker_slots,
            s.consumable_slots,
        ],
        boss: &s.boss,
        jokers: &s.jokers,
        levels: &s.hand_levels,
        vouchers: &s.vouchers,
        consumables: &s.consumables,
        cards,
    };
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    if let Ok(bytes) = serde_json::to_vec(&b) {
        fnv(&mut h, &bytes);
    }
    h
}

/// One simulated boss round; `(round score, target)`.
fn simulate(base: &GameState, cfg: &PowerCfg, seed: u64, agent: &mut GreedyAgent) -> (f64, f64) {
    let mut st = base.clone();
    st.phase = Phase::BlindSelect;
    st.blind = BlindKind::Boss;
    st.pack = None;
    st.boss_disabled = false;
    if !cfg.boss_aware {
        st.boss = None;
    }
    st.rng = GameRng::new(seed);
    let mut g = Game::from_state(st);
    if g.step(Action::SelectBlind).is_err() {
        return (0.0, 1.0);
    }
    let real_target = g.state().target;
    for _ in 0..MAX_ACTIONS {
        if g.state().phase != Phase::Play {
            break;
        }
        if cfg.play_on {
            g.state_mut().target = f64::INFINITY;
        }
        let a = agent.act_play(&g, real_target);
        if g.step(a).is_err() {
            break;
        }
    }
    (g.state().round_score, real_target)
}

/// Build power of `state` against this ante's boss, from `k` simulated rounds (`k >= 1`).
pub fn power(state: &GameState, k: usize, cfg: &PowerCfg) -> PowerReport {
    let k = k.max(1);
    let mut z = state_digest(state) ^ cfg.salt.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let mut agent = GreedyAgent::new();
    let (mut sum, mut sum2, mut beat, mut target) = (0.0, 0.0, 0usize, 1.0);
    for _ in 0..k {
        let seed = splitmix(&mut z);
        let (score, t) = simulate(state, cfg, seed, &mut agent);
        target = t;
        let v = if score <= 0.0 {
            -LN_CLAMP
        } else {
            (score / t).ln().clamp(-LN_CLAMP, LN_CLAMP)
        };
        sum += v;
        sum2 += v * v;
        beat += usize::from(score >= t);
    }
    let n = k as f64;
    let mean = sum / n;
    PowerReport {
        phi: mean,
        std: (sum2 / n - mean * mean).max(0.0).sqrt(),
        frac_beat: beat as f64 / n,
        target,
        k,
    }
}
