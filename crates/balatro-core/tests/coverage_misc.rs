//! Phase 6A coverage gaps: behaviors found untested by `cargo llvm-cov` (tuning knobs, agent
//! accessors/reset, joker state reset, endless blind curve, illegal-action rejection).

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, HeuristicAgent, HeuristicConfig};
use balatro_core::blinds::{StakeScaling, base_chips};
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::HandType;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::state::GameConfig;

#[test]
fn heuristic_config_set_known_unknown_and_unparsable() {
    let mut c = HeuristicConfig::default();
    assert!(c.set("pace", "1.5"));
    assert_eq!(c.pace, 1.5);
    assert!(c.set("shop_samples", "7"));
    assert_eq!(c.shop_samples, 7);
    assert!(c.set("boss_reroll_reserve", "12"));
    assert_eq!(c.boss_reroll_reserve, 12);
    // Unknown key and bad value leave the config untouched.
    assert!(!c.set("no_such_knob", "1"));
    assert!(!c.set("pace", "abc"));
    assert_eq!(c.pace, 1.5);
}

#[test]
fn heuristic_agent_accessors_and_reset() {
    let mut g = Game::new(3, GameConfig::default());
    let mut a = HeuristicAgent::new(3);
    for _ in 0..20 {
        if g.is_terminal() {
            break;
        }
        let act = a.act(&g);
        g.step(act).unwrap();
    }
    assert!(a.pressure().is_finite());
    let _ = a.strategy().top_target();
    a.reset();
    assert_eq!(a.strategy().top_target(), HandType::ALL[0]);
}

#[test]
fn joker_reset_state_restores_definition_defaults() {
    let mut j = JokerInstance::new(1, JokerId::IceCream)
        .with_value(3.0)
        .with_counter(9);
    j.aux = 5;
    j.reset_state();
    let d = j.def().expect("def");
    assert_eq!(j.value, d.init_value);
    assert_eq!(j.counter, d.init_counter);
    assert_eq!(j.aux, 0);
}

#[test]
fn endless_base_chips_grow_and_overflow_to_infinity() {
    assert_eq!(
        base_chips(0, StakeScaling::White),
        base_chips(-3, StakeScaling::White)
    );
    let mut prev = base_chips(8, StakeScaling::White);
    for ante in 9..=38 {
        let v = base_chips(ante, StakeScaling::White);
        assert!(v.is_finite() && v > prev, "ante {ante}");
        prev = v;
    }
    assert!(base_chips(39, StakeScaling::White).is_infinite());
}

#[test]
fn invalid_actions_are_rejected_not_applied() {
    let mut g = Game::new(1, GameConfig::default());
    let before = format!("{:?}", g.state());
    assert!(g.step(Action::EndShop).is_err());
    assert!(g.step(Action::SkipPack).is_err());
    assert_eq!(format!("{:?}", g.state()), before);
}
