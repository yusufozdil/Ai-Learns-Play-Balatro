//! Phase 7A: golden determinism test. 200 seeded games (random and market-weighted agents, 3/4 of
//! them with 5 random jokers) are played to the end; the final states (JSON), the action streams
//! and the step counts are folded into one FNV-1a hash that must equal [`GOLDEN`].
//!
//! The constant was produced with the code BEFORE the Phase 7 performance work, so a mismatch
//! means an optimisation changed behaviour. If a deliberate rule change alters it, regenerate
//! with `GOLDEN_PRINT=1 cargo test --release -p balatro-core --test golden -- --nocapture` and
//! explain the change in the commit.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, RandomAgent};
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::enums::Edition;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey, fnv1a64};
use balatro_core::serialize::state_to_json;
use balatro_core::state::GameConfig;

const GAMES: u64 = 200;
const STEP_CAP: u64 = 4000;
// Regenerated after the real-game float parity fixes: xmult is `m + m * (x - 1)`, Ramen
// is raw double arithmetic (no rounding), Plasma floors. Same 18278 steps, other final states.
const GOLDEN: u64 = 0xa565_19c4_0ca5_54d3;

/// Favours market, packs, consumables and joker moves (same weights as the fuzz test).
struct WeightedAgent {
    rng: GameRng,
    legal: Vec<Action>,
    w: Vec<f64>,
}

impl WeightedAgent {
    fn new(seed: u64) -> Self {
        WeightedAgent {
            rng: GameRng::new(seed ^ 0x57E1_6417),
            legal: Vec::new(),
            w: Vec::new(),
        }
    }
}

impl Agent for WeightedAgent {
    fn act(&mut self, game: &Game) -> Action {
        game.legal_actions_into(&mut self.legal);
        if self.legal.is_empty() {
            return Action::EndShop;
        }
        self.w.clear();
        self.w.extend(self.legal.iter().map(|a| match a {
            Action::Play(_) => 2.0,
            Action::Discard(_) => 1.0,
            Action::EndShop => 0.2,
            Action::SkipPack => 0.4,
            Action::SelectBlind => 1.0,
            Action::SkipBlind => 1.5,
            Action::SellJoker(_) | Action::SellConsumable(_) => 1.0,
            _ => 5.0,
        }));
        let i = self.rng.weighted_index(RngKey::Misc, &self.w).unwrap_or(0);
        self.legal[i]
    }
}

fn joker_ids() -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs().map(|d| d.id).collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

/// Play game `seed` to the end (or the step cap); returns (hash, steps).
fn play(seed: u64, ids: &[JokerId]) -> (u64, u64) {
    let mut p = GameRng::new(seed ^ 0x0060_1DE4);
    let mut g = Game::new(seed, GameConfig::default());
    if !seed.is_multiple_of(4) {
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
    let mut agent: Box<dyn Agent> = match seed % 4 {
        0 => Box::new(RandomAgent::with_play_weight(seed, 4.0)),
        2 => Box::new(GreedyAgent::new()),
        _ => Box::new(WeightedAgent::new(seed)),
    };
    // Every third game takes score shortcuts so that the runs reach deep antes (boss rotation,
    // shops with many jokers, showdowns) instead of dying in the first rounds.
    let shortcuts = seed.is_multiple_of(3);
    let mut log = String::new();
    let mut steps = 0;
    while !g.is_terminal() && steps < STEP_CAP {
        let a = agent.act(&g);
        if shortcuts && matches!(a, Action::Play(_)) && p.range(RngKey::Misc, 0, 100) < 60 {
            let st = g.state_mut();
            st.round_score = st.round_score.max(st.target - 1.0);
        }
        log.push_str(&format!("{a:?};"));
        if g.step(a).is_err() {
            log.push('!');
            break;
        }
        steps += 1;
    }
    let mut bytes = state_to_json(g.state(), false).into_bytes();
    bytes.extend_from_slice(log.as_bytes());
    bytes.extend_from_slice(&steps.to_le_bytes());
    (fnv1a64(&bytes), steps)
}

#[test]
fn golden_final_states_are_unchanged() {
    let ids = joker_ids();
    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    let mut total = 0;
    for seed in 0..GAMES {
        let (h, n) = play(seed, &ids);
        total += n;
        acc = fnv1a64(&[acc.to_le_bytes(), h.to_le_bytes()].concat());
    }
    if std::env::var("GOLDEN_PRINT").is_ok() {
        println!("GOLDEN hash = {acc:#018x} ({total} steps)");
        return;
    }
    assert_eq!(acc, GOLDEN, "golden hash changed ({total} steps)");
}
