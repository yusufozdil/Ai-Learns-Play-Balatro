//! Phase 5E smoke test: random and greedy agents over every (deck, stake) combination (15 x 8 =
//! 120), invariants checked after every step, no panics, every game terminates.
//!
//! `SMOKE_GAMES=<n>` sets the games per combination and agent (default 12 so the debug
//! `cargo test` stays fast; release runs use 1000+):
//! `SMOKE_GAMES=1000 cargo test --release -p balatro-core --test deck_stake_smoke -- --nocapture`

use balatro_core::agents::{Agent, GreedyAgent, HeuristicAgent, RandomAgent};
use balatro_core::game::Game;
use balatro_core::state::{DeckKind, GameConfig, StakeKind};

const STEP_CAP: u32 = 20_000;

fn games_per_combo() -> u64 {
    std::env::var("SMOKE_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

/// Play one game to the end; returns (steps, final ante, won).
fn run(mut game: Game, agent: &mut dyn Agent, label: &str) -> (u32, i32, bool) {
    let mut steps = 0;
    while !game.is_terminal() {
        assert!(
            steps < STEP_CAP,
            "{label}: no termination in {STEP_CAP} steps"
        );
        let a = agent.act(&game);
        game.step(a)
            .unwrap_or_else(|e| panic!("{label}: agent action {a:?} rejected: {e}"));
        game.state()
            .check_invariants()
            .unwrap_or_else(|e| panic!("{label}: invariant after {a:?}: {e}"));
        steps += 1;
    }
    (steps, game.state().ante, game.outcome().is_some())
}

#[test]
fn every_deck_and_stake_survives_random_and_greedy_agents() {
    let n = games_per_combo();
    let mut total_steps = 0u64;
    for deck in DeckKind::ALL {
        for stake in StakeKind::ALL {
            let config = GameConfig {
                deck,
                stake,
                endless: false,
            };
            for i in 0..n {
                let seed = 1_000 * (deck.index() as u64 * 8 + stake.index() as u64) + i;
                let label = format!("{deck:?}/{stake:?} seed {seed}");
                let (s, _, _) = run(
                    Game::new(seed, config),
                    &mut RandomAgent::new(seed),
                    &format!("{label} random"),
                );
                total_steps += u64::from(s);
                let (s, _, _) = run(
                    Game::new(seed, config),
                    &mut GreedyAgent::new(),
                    &format!("{label} greedy"),
                );
                total_steps += u64::from(s);
            }
        }
    }
    eprintln!("smoke: 120 combos x {n} games x 2 agents, {total_steps} steps");
}

#[test]
fn heuristic_agent_plays_every_combination() {
    // The market agent prices stickered jokers, rent and starting consumables too.
    let n = games_per_combo().div_ceil(6);
    for deck in DeckKind::ALL {
        for stake in StakeKind::ALL {
            let config = GameConfig {
                deck,
                stake,
                endless: false,
            };
            for i in 0..n {
                let seed = 77_000 + 100 * (deck.index() as u64 * 8 + stake.index() as u64) + i;
                run(
                    Game::new(seed, config),
                    &mut HeuristicAgent::new(seed),
                    &format!("{deck:?}/{stake:?} seed {seed} heuristic"),
                );
            }
        }
    }
}

#[test]
fn harder_stakes_are_harder_for_the_heuristic_agent() {
    // Sanity of the whole ladder: mean ante on Red Deck drops from White to Gold.
    let games = 40u64;
    let mean = |stake: StakeKind| -> f64 {
        let mut sum = 0;
        for seed in 0..games {
            let config = GameConfig {
                deck: DeckKind::Red,
                stake,
                endless: false,
            };
            let (_, ante, _) = run(
                Game::new(500 + seed, config),
                &mut HeuristicAgent::new(seed),
                "ladder",
            );
            sum += i64::from(ante);
        }
        sum as f64 / games as f64
    };
    let (white, gold) = (mean(StakeKind::White), mean(StakeKind::Gold));
    eprintln!("heuristic mean ante: White {white:.2}, Gold {gold:.2}");
    assert!(
        gold < white,
        "Gold {gold} should be harder than White {white}"
    );
}
