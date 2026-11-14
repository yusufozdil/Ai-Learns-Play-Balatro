//! Smoke test (Phase 2A): 1000 seeded games with a uniform random legal-action policy, half of
//! them starting with random prototype jokers. Every legal action must step `Ok`, invariants
//! hold after every step (`debug_assert` inside `step`), and every game terminates.
//! The 100k-game version is Phase 2D.

use balatro_core::actions::Action;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

const GAMES: u64 = 1000;
/// Far above any real run (a run is at most ~24 rounds x ~10 actions + shops).
const MAX_STEPS: usize = 5000;

#[test]
fn random_games_terminate_without_panics() {
    let mut legal = Vec::new();
    let mut total_steps = 0usize;
    let mut rounds_won = 0u32;
    for seed in 0..GAMES {
        let mut g = Game::new(seed, GameConfig::default());
        let mut policy = GameRng::new(seed ^ 0x5EED);
        if seed % 2 == 1 {
            for _ in 0..policy.range(RngKey::Misc, 1, 6) {
                let id = JokerId::ALL[policy.range(RngKey::Misc, 0, JokerId::COUNT)];
                let _ = g.add_joker(JokerInstance::new(0, id));
            }
        }
        let mut steps = 0;
        while !g.is_terminal() {
            assert!(steps < MAX_STEPS, "seed {seed}: no termination");
            g.legal_actions_into(&mut legal);
            assert!(
                !legal.is_empty(),
                "seed {seed}: no legal action in {:?}",
                g.state().phase
            );
            let a = legal[policy.range(RngKey::Misc, 0, legal.len())];
            if let Err(e) = g.step(a) {
                panic!("seed {seed}: legal {a:?} rejected: {e}");
            }
            steps += 1;
        }
        assert!(matches!(g.state().phase, Phase::GameOver | Phase::Won));
        assert!(g.outcome().is_some());
        assert!(g.state().check_invariants().is_ok());
        total_steps += steps;
        rounds_won += g.state().stats.rounds_won;
    }
    // Random play rarely beats a blind, but the loop must be exercised.
    assert!(total_steps > GAMES as usize * 4);
    eprintln!("{GAMES} random games: {total_steps} steps, {rounds_won} rounds won");
}

#[test]
fn round_winning_policy_reaches_later_antes() {
    // A policy that always beats the blind (round score nudged to target - 1) exercises the full
    // ante loop, shops and the win; with Golden Joker money and skips mixed in.
    let mut legal = Vec::new();
    let mut wins = 0;
    for seed in 0..50u64 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut policy = GameRng::new(seed);
        let _ = g.add_joker(JokerInstance::new(0, JokerId::GoldenJoker));
        let mut steps = 0;
        while !g.is_terminal() && steps < MAX_STEPS {
            g.legal_actions_into(&mut legal);
            let mut a = legal[policy.range(RngKey::Misc, 0, legal.len())];
            // A hand the boss rejects (The Psychic) scores 0: play a valid one instead.
            let valid = |a: &Action| match *a {
                Action::Play(m) => g.preview_play(m).is_ok_and(|r| !r.invalid),
                _ => false,
            };
            if matches!(a, Action::Play(_))
                && !valid(&a)
                && let Some(&b) = legal.iter().find(|x| valid(x))
            {
                a = b;
            }
            if g.state().phase == Phase::Play && matches!(a, Action::Play(_)) {
                let t = g.state().target;
                g.state_mut().round_score = t - 1.0;
            }
            g.step(a).expect("legal action");
            steps += 1;
        }
        assert!(g.is_terminal(), "seed {seed}");
        if g.state().phase == Phase::Won {
            wins += 1;
            assert_eq!(g.state().ante, 8);
        }
    }
    assert_eq!(wins, 50);
}
