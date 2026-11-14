//! Robustness (Phase 2D): mass random games, determinism, clone-continuation and invalid-action
//! fuzzing. The 100k run is `#[ignore]`: `cargo test --release -- --ignored`.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, RandomAgent};
use balatro_core::content::jokers::JokerId;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::{GameConfig, GameState};

/// Far above any real run.
const MAX_STEPS: usize = 5000;

fn new_game(seed: u64, with_jokers: bool) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    if with_jokers {
        let mut p = GameRng::new(seed ^ 0x70C3);
        for _ in 0..p.range(RngKey::Misc, 1, 6) {
            let id = JokerId::ALL[p.range(RngKey::Misc, 0, JokerId::COUNT)];
            let _ = g.add_joker(JokerInstance::new(0, id));
        }
    }
    g
}

/// Plays `games` seeded games from `first_seed`; returns total steps.
fn mass_run(first_seed: u64, games: u64) -> u64 {
    let mut total = 0u64;
    for seed in first_seed..first_seed + games {
        let mut g = new_game(seed, seed % 2 == 1);
        // Mix: mostly uniform random, some Play-biased (reaches deeper antes).
        let mut agent = match seed % 4 {
            0 | 1 => RandomAgent::new(seed),
            2 => RandomAgent::with_play_weight(seed, 6.0),
            _ => RandomAgent::with_play_weight(seed, 30.0),
        };
        let mut steps = 0usize;
        while !g.is_terminal() {
            assert!(steps < MAX_STEPS, "seed {seed}: no termination");
            let a = agent.act(&g);
            // Every 8th game: nudge the round score so blinds are beaten and later antes,
            // shops and the win path are exercised too (random play almost never gets there).
            if seed % 8 == 7 && matches!(a, Action::Play(_)) {
                let t = g.state().target;
                g.state_mut().round_score = t - 1.0;
            }
            if let Err(e) = g.step(a) {
                panic!("seed {seed}: legal-looking {a:?} rejected: {e}");
            }
            g.state()
                .check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
            steps += 1;
        }
        assert!(g.outcome().is_some(), "seed {seed}");
        total += steps as u64;
    }
    total
}

#[test]
fn ten_thousand_random_games() {
    let steps = mass_run(0, 10_000);
    assert!(steps > 40_000);
}

#[test]
#[ignore = "100k games; run with `cargo test --release -- --ignored`"]
fn hundred_thousand_random_games() {
    let steps = mass_run(1_000_000, 100_000);
    eprintln!("100k games: {steps} steps");
    assert!(steps > 400_000);
}

fn run_recorded(seed: u64, agent_seed: u64, jokers: bool) -> (Vec<Action>, GameState) {
    let mut g = new_game(seed, jokers);
    let mut agent = RandomAgent::with_play_weight(agent_seed, 4.0);
    let mut log = Vec::new();
    while !g.is_terminal() && log.len() < MAX_STEPS {
        let a = agent.act(&g);
        g.step(a).unwrap();
        log.push(a);
    }
    (log, g.into_state())
}

#[test]
fn same_seeds_same_trajectory_and_state() {
    for seed in 0..100u64 {
        let jokers = seed % 2 == 0;
        let (l1, s1) = run_recorded(seed, seed + 7, jokers);
        let (l2, s2) = run_recorded(seed, seed + 7, jokers);
        assert_eq!(l1, l2, "seed {seed}");
        assert_eq!(s1, s2, "seed {seed}");
    }
}

#[test]
fn greedy_is_deterministic() {
    for seed in 0..30u64 {
        let run = || {
            let mut g = new_game(seed, seed % 2 == 1);
            let mut a = GreedyAgent::new();
            let mut log = Vec::new();
            while !g.is_terminal() && log.len() < MAX_STEPS {
                let act = a.act(&g);
                g.step(act).unwrap();
                log.push(act);
            }
            (log, g.into_state())
        };
        assert_eq!(run(), run(), "seed {seed}");
    }
}

#[test]
fn clone_mid_game_continues_identically() {
    for seed in 0..200u64 {
        let mut g = new_game(seed, seed % 2 == 0);
        let mut agent = RandomAgent::with_play_weight(seed, 4.0);
        let cut = (seed % 17) as usize + 1;
        for _ in 0..cut {
            if g.is_terminal() {
                break;
            }
            let a = agent.act(&g);
            g.step(a).unwrap();
        }
        let mut c1 = g.clone();
        let mut c2 = g.clone();
        assert_eq!(c1.state(), c2.state());
        let mut agent2 = agent.clone();
        let mut steps = 0;
        while !c1.is_terminal() && steps < MAX_STEPS {
            let a = agent.act(&c1);
            let b = agent2.act(&c2);
            assert_eq!(a, b);
            c1.step(a).unwrap();
            c2.step(b).unwrap();
            assert_eq!(c1.state(), c2.state(), "seed {seed} step {steps}");
            steps += 1;
        }
    }
}

#[test]
fn invalid_actions_leave_state_untouched() {
    let mut rejected = 0u64;
    for seed in 0..300u64 {
        let mut g = new_game(seed, seed % 2 == 0);
        let mut fuzz = GameRng::new(seed ^ 0xF00D);
        let mut good = RandomAgent::with_play_weight(seed, 4.0);
        let mut steps = 0;
        while !g.is_terminal() && steps < MAX_STEPS {
            // A few arbitrary (mostly invalid) actions before each real one.
            for _ in 0..3 {
                let a = random_action(&mut fuzz);
                let before = g.state().clone();
                if g.step(a).is_err() {
                    rejected += 1;
                    assert_eq!(
                        *g.state(),
                        before,
                        "seed {seed}: {a:?} changed state on Err"
                    );
                } else {
                    g.state()
                        .check_invariants()
                        .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
                }
                if g.is_terminal() {
                    break;
                }
            }
            if g.is_terminal() {
                break;
            }
            let a = good.act(&g);
            g.step(a).unwrap();
            steps += 1;
        }
        // Terminal game rejects everything without change.
        let before = g.state().clone();
        assert!(g.step(Action::EndShop).is_err());
        assert_eq!(*g.state(), before);
    }
    assert!(rejected > 1000);
}

fn random_action(r: &mut GameRng) -> Action {
    let m = |r: &mut GameRng| r.range(RngKey::Misc, 0, 1 << 16) as u16;
    let b = |r: &mut GameRng| r.range(RngKey::Misc, 0, 8) as u8;
    match r.range(RngKey::Misc, 0, 16) {
        0 => Action::SelectBlind,
        1 => Action::SkipBlind,
        2 | 3 => Action::Play(m(r) & 0xFF),
        4 => Action::Play(m(r)),
        5 | 6 => Action::Discard(m(r) & 0xFF),
        7 => Action::Discard(m(r)),
        8 => Action::SellJoker(b(r)),
        9 => Action::EndShop,
        10 => Action::UseConsumable {
            slot: b(r),
            targets: m(r),
        },
        11 => Action::SellConsumable(b(r)),
        12 => Action::Buy(b(r)),
        13 => Action::Reroll,
        14 => Action::MoveJoker {
            from: b(r),
            to: b(r),
        },
        _ => Action::PickPackItem {
            index: b(r),
            targets: m(r),
        },
    }
}
