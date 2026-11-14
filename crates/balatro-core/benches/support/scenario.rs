//! Shared scenario code for the Phase 7A benchmarks (`benches/game.rs`, `examples/steps.rs`, `examples/breakdown.rs`):
//! game set-ups and the uniform / market-weighted agents.

#![allow(dead_code)]

use balatro_core::actions::Action;
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::enums::{Edition, Phase};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

/// How an agent picks among the legal actions.
#[derive(Clone, Copy)]
pub enum Policy {
    /// Uniform over legal actions.
    Uniform,
    /// Favours market, packs, consumables and joker moves (same weights as the fuzz test).
    Weighted,
}

/// Game set-up for a scenario.
#[derive(Clone, Copy)]
pub struct Scenario {
    pub policy: Policy,
    /// Start with 5 random jokers.
    pub jokers: bool,
    /// Percent chance that a `Play` first lifts the round score to `target - 1`, so that games
    /// reach deep antes (shops, packs, many jokers) instead of dying in the first rounds.
    pub shortcut_pct: usize,
}

pub fn joker_ids() -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs().map(|d| d.id).collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

pub fn new_game(seed: u64, sc: &Scenario, ids: &[JokerId]) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    if sc.jokers {
        let mut p = GameRng::new(seed ^ 0x0060_1DE4);
        for _ in 0..5 {
            let mut j = JokerInstance::new(0, ids[p.range(RngKey::Misc, 0, ids.len())]);
            j.edition = match p.range(RngKey::Misc, 0, 8) {
                0 => Edition::Foil,
                1 => Edition::Holographic,
                2 => Edition::Polychrome,
                3 => Edition::Negative,
                _ => Edition::Base,
            };
            let _ = g.add_joker(j);
        }
    }
    g
}

/// One agent decision: legal actions into `buf`, pick, and (optionally) a score shortcut.
pub fn pick(
    g: &mut Game,
    rng: &mut GameRng,
    sc: &Scenario,
    buf: &mut Vec<Action>,
    w: &mut Vec<f64>,
) -> Option<Action> {
    g.legal_actions_into(buf);
    if buf.is_empty() {
        return None;
    }
    Some(pick_from(g, rng, sc, buf, w))
}

/// The choice part of [`pick`], given the legal actions.
pub fn pick_from(
    g: &mut Game,
    rng: &mut GameRng,
    sc: &Scenario,
    buf: &[Action],
    w: &mut Vec<f64>,
) -> Action {
    let i = match sc.policy {
        Policy::Uniform => rng.range(RngKey::Misc, 0, buf.len()),
        Policy::Weighted => {
            w.clear();
            w.extend(buf.iter().map(|a| match a {
                Action::Play(_) => 2.0,
                Action::Discard(_) => 1.0,
                Action::EndShop => 0.2,
                Action::SkipPack => 0.4,
                Action::SelectBlind => 1.0,
                Action::SkipBlind => 1.5,
                Action::SellJoker(_) | Action::SellConsumable(_) => 1.0,
                // n*(n-1) of them are legal; at weight 5 they would be ~60% of all steps.
                Action::MoveJoker { .. } => 0.1,
                _ => 5.0,
            }));
            rng.weighted_index(RngKey::Misc, w).unwrap_or(0)
        }
    };
    let a = buf[i];
    if sc.shortcut_pct > 0
        && matches!(a, Action::Play(_))
        && rng.range(RngKey::Misc, 0, 100) < sc.shortcut_pct
    {
        let st = g.state_mut();
        st.round_score = st.round_score.max(st.target - 1.0);
    }
    a
}

/// Play one full game; returns the number of steps taken.
pub fn play_game(
    seed: u64,
    sc: &Scenario,
    ids: &[JokerId],
    buf: &mut Vec<Action>,
    w: &mut Vec<f64>,
) -> u64 {
    let mut g = new_game(seed, sc, ids);
    let mut rng = GameRng::new(seed ^ 0x5EED);
    let mut steps = 0;
    while !g.is_terminal() && steps < 4000 {
        let Some(a) = pick(&mut g, &mut rng, sc, buf, w) else {
            break;
        };
        if g.step(a).is_err() {
            break;
        }
        steps += 1;
    }
    steps
}

/// Advance a scenario game until `phase` is reached `skip` times; returns the game.
pub fn reach(phase: Phase, sc: &Scenario, ids: &[JokerId], skip: u32) -> Game {
    let mut buf = Vec::new();
    let mut w = Vec::new();
    for seed in 0..500u64 {
        let mut g = new_game(seed, sc, ids);
        let mut rng = GameRng::new(seed ^ 0x5EED);
        let mut seen = 0;
        for _ in 0..4000 {
            if g.is_terminal() {
                break;
            }
            if g.state().phase == phase {
                if seen == skip {
                    return g;
                }
                seen += 1;
            }
            let Some(a) = pick(&mut g, &mut rng, sc, &mut buf, &mut w) else {
                break;
            };
            if g.step(a).is_err() {
                break;
            }
        }
    }
    panic!("phase {phase:?} not reached");
}

pub const JOKERLESS: Scenario = Scenario {
    policy: Policy::Uniform,
    jokers: false,
    shortcut_pct: 0,
};
pub const JOKERS_MARKET: Scenario = Scenario {
    policy: Policy::Weighted,
    jokers: true,
    shortcut_pct: 30,
};
pub const JOKERS_MARKET_NO_SHORTCUT: Scenario = Scenario {
    policy: Policy::Weighted,
    jokers: true,
    shortcut_pct: 0,
};
