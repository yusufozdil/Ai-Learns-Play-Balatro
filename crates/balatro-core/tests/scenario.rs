//! Scenario generator (`balatro_core::scenario`): valid, deterministic, plausible, playable.

use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::enums::{Phase, Rarity};
use balatro_core::scenario::{random_scenario, try_random_scenario};
use balatro_core::serialize::state_to_json;
use balatro_core::state::{DeckKind, GameConfig, StakeKind};

fn cfg(i: u64) -> GameConfig {
    GameConfig {
        deck: DeckKind::ALL[(i as usize) % DeckKind::COUNT],
        stake: StakeKind::ALL[(i as usize / 3) % StakeKind::COUNT],
        endless: i % 11 == 0,
    }
}

#[test]
fn scenarios_are_valid_deterministic_and_plausible() {
    let (mut sticker_seen, mut legendary, mut ante_hist) = ([0u32; 3], 0u32, [0u32; 9]);
    for seed in 0..3000u64 {
        let c = cfg(seed);
        let g = try_random_scenario(seed, c).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let s = g.state();
        assert_eq!(s.phase, Phase::BlindSelect);
        assert!((1..=8).contains(&s.ante));
        ante_hist[s.ante as usize] += 1;
        for j in &s.jokers {
            let st = j.stickers;
            if st.eternal {
                assert!(c.stake.eternal_jokers(), "eternal below Black");
                sticker_seen[0] += 1;
            }
            if st.perishable {
                assert!(c.stake.perishable_jokers(), "perishable below Orange");
                sticker_seen[1] += 1;
            }
            if st.rental {
                assert!(c.stake.rental_jokers(), "rental below Gold");
                sticker_seen[2] += 1;
            }
            if j.def().is_some_and(|d| d.rarity == Rarity::Legendary) {
                legendary += 1;
            }
        }
        // Pure function of (seed, config).
        let again = random_scenario(seed, c);
        assert_eq!(state_to_json(s, false), state_to_json(again.state(), false));
    }
    assert!(
        sticker_seen.iter().all(|&n| n > 0),
        "stickers {sticker_seen:?}"
    );
    assert!(legendary > 0, "no legendary joker in 3000 scenarios");
    assert!(
        ante_hist[1..].iter().all(|&n| n > 200),
        "ante spread {ante_hist:?}"
    );
}

#[test]
fn random_policy_from_scenarios_keeps_invariants() {
    let mut steps = 0u64;
    let mut seed = 0u64;
    while steps < 120_000 {
        let c = cfg(seed);
        let mut g = random_scenario(seed, c);
        let mut agent = RandomAgent::with_play_weight(seed, 4.0);
        let mut n = 0;
        while !g.is_terminal() && n < 3000 {
            let a = agent.act(&g);
            g.step(a)
                .unwrap_or_else(|e| panic!("seed {seed} step {n}: {a:?}: {e}"));
            g.state()
                .check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} step {n}: {e}"));
            n += 1;
        }
        steps += n as u64;
        seed += 1;
    }
    assert!(seed > 20);
}
