//! G1 build-power meter: determinism, purity, sanity.

use balatro_core::actions::Action;
use balatro_core::agents::power::{PowerCfg, power, state_digest};
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::state::GameConfig;

fn at_ante(seed: u64, ante: i32) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.state_mut().ante = ante;
    g
}

#[test]
fn deterministic_and_pure() {
    for seed in 0..4u64 {
        let mut g = at_ante(seed, 5);
        g.add_joker(JokerInstance::new(0, JokerId::Joker)).unwrap();
        let before = g.state().clone();
        for boss_aware in [false, true] {
            let cfg = PowerCfg {
                boss_aware,
                ..PowerCfg::default()
            };
            let a = power(g.state(), 8, &cfg);
            let b = power(g.state(), 8, &cfg);
            assert_eq!(a, b, "same state, same report");
            assert_eq!(*g.state(), before, "input state untouched (incl. rng)");
            // The real draw order and RNG streams are not inputs.
            let mut h = g.clone_state_game();
            h.state_mut().rng = balatro_core::rng::GameRng::new(999);
            h.state_mut().deck.draw_pile.reverse();
            assert_eq!(a, power(h.state(), 8, &cfg));
        }
        // Salt gives an independent sample; a different build gives a different digest.
        let s1 = power(
            g.state(),
            8,
            &PowerCfg {
                salt: 1,
                ..PowerCfg::default()
            },
        );
        let s2 = power(
            g.state(),
            8,
            &PowerCfg {
                salt: 2,
                ..PowerCfg::default()
            },
        );
        assert!(s1.phi != s2.phi || s1.std != s2.std);
        let mut h = g.clone_state_game();
        h.state_mut().money += 1;
        assert_ne!(state_digest(g.state()), state_digest(h.state()));
    }
}

trait CloneGame {
    fn clone_state_game(&self) -> Game;
}
impl CloneGame for Game {
    fn clone_state_game(&self) -> Game {
        Game::from_state(self.state().clone())
    }
}

#[test]
fn works_from_any_phase_and_after_real_play() {
    let mut g = Game::new(3, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    let r = power(g.state(), 4, &PowerCfg::default());
    assert!(r.phi.is_finite() && r.phi >= -3.0 && r.phi <= 3.0);
    assert!((0.0..=1.0).contains(&r.frac_beat));
}

#[test]
fn bare_deck_at_ante_8_is_hopeless() {
    let mut lo = 0.0;
    for seed in 0..6u64 {
        let g = at_ante(seed, 8);
        let r = power(g.state(), 8, &PowerCfg::default());
        assert_eq!(r.frac_beat, 0.0);
        lo += r.phi;
    }
    assert!(lo / 6.0 < -2.0, "mean phi {}", lo / 6.0);
}

#[test]
fn strong_xmult_joker_does_not_lower_power() {
    let (mut base, mut with, mut n) = (0.0, 0.0, 0.0);
    for seed in 0..12u64 {
        for ante in [3, 5] {
            let g = at_ante(seed, ante);
            base += power(g.state(), 8, &PowerCfg::default()).phi;
            let mut h = at_ante(seed, ante);
            h.add_joker(JokerInstance::new(0, JokerId::Cavendish))
                .unwrap();
            with += power(h.state(), 8, &PowerCfg::default()).phi;
            n += 1.0;
        }
    }
    assert!(
        with / n > base / n + 0.2,
        "base {} with {}",
        base / n,
        with / n
    );
}
