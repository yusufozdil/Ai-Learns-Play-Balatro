//! Phase 2C: `legal_actions` is exactly the set of actions `validate` accepts.
//! Listed => `validate` Ok; sampled unlisted candidates => `validate` Err. Deterministic RNG.

use balatro_core::actions::{Action, legal_actions, validate};
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::{GameConfig, GameState};
use std::collections::HashSet;

fn rand_mask(rng: &mut GameRng) -> u16 {
    let r = rng.next_u64(RngKey::Misc);
    // Mix of dense, sparse and tiny masks.
    match r & 3 {
        0 => (r >> 8) as u16,
        1 => ((r >> 8) as u16) & ((r >> 24) as u16),
        2 => ((r >> 8) as u16) & 0xFF,
        _ => 1 << ((r >> 8) % 16),
    }
}

/// Candidate actions: every phase-independent action shape, with random/out-of-range operands.
fn candidates(rng: &mut GameRng, n: usize) -> Vec<Action> {
    let mut v = vec![
        Action::SelectBlind,
        Action::SkipBlind,
        Action::EndShop,
        Action::Reroll,
        Action::SkipPack,
        Action::Play(0),
        Action::Discard(0),
        Action::Play(u16::MAX),
        Action::Discard(u16::MAX),
    ];
    for i in 0..8u8 {
        v.push(Action::SellJoker(i));
    }
    v.push(Action::SellJoker(255));
    for _ in 0..n {
        let m = rand_mask(rng);
        let s = (rng.next_u64(RngKey::Misc) % 12) as u8;
        v.push(Action::Play(m));
        v.push(Action::Discard(m));
        v.push(Action::SellJoker(s));
        v.push(Action::SellConsumable(s));
        v.push(Action::Buy(s));
        v.push(Action::OpenPack(s));
        v.push(Action::UseConsumable {
            slot: s,
            targets: m,
        });
        v.push(Action::MoveJoker { from: s, to: 0 });
        v.push(Action::PickPackItem {
            index: s,
            targets: m,
        });
    }
    v
}

fn check_state(state: &GameState, rng: &mut GameRng) {
    let mut listed = Vec::new();
    legal_actions(state, &mut listed);
    let set: HashSet<Action> = listed.iter().copied().collect();
    assert_eq!(set.len(), listed.len(), "duplicates in legal list");
    for &a in &listed {
        assert_eq!(
            validate(state, a),
            Ok(()),
            "listed {a:?} in {:?}",
            state.phase
        );
    }
    if state.is_terminal() {
        assert!(listed.is_empty());
    }
    for a in candidates(rng, 60) {
        if !set.contains(&a) {
            assert!(
                validate(state, a).is_err(),
                "unlisted {a:?} validates in {:?} (hand {})",
                state.phase,
                state.deck.hand.len()
            );
        }
    }
}

/// Exhaustive over all 65536 masks for one state.
fn check_all_masks(state: &GameState) {
    let mut listed = Vec::new();
    legal_actions(state, &mut listed);
    let set: HashSet<Action> = listed.iter().copied().collect();
    for m in 0..=u16::MAX {
        for a in [Action::Play(m), Action::Discard(m)] {
            assert_eq!(validate(state, a).is_ok(), set.contains(&a), "{a:?}");
        }
    }
}

#[test]
fn legal_matches_validate_over_random_games() {
    for seed in 0..60u64 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut rng = GameRng::new(seed.wrapping_mul(7919) + 1);
        // Start some games with jokers (some eternal) so sell actions get exercised.
        if seed % 2 == 0 {
            for k in 0..(seed % 5) {
                let mut j = JokerInstance::new(0, JokerId::Joker);
                j.stickers.eternal = k % 2 == 1;
                g.add_joker(j);
            }
        }
        let mut buf = Vec::new();
        for step in 0..300 {
            check_state(g.state(), &mut rng);
            if step % 40 == 3 {
                check_all_masks(g.state());
            }
            g.legal_actions_into(&mut buf);
            let Some(i) = rng.choose_index(RngKey::Misc, buf.len()) else {
                break;
            };
            g.step(buf[i]).expect("listed action accepted");
        }
        check_state(g.state(), &mut rng);
    }
}

#[test]
fn legal_covers_all_phases_and_edge_cases() {
    let mut g = Game::new(11, GameConfig::default());
    let mut rng = GameRng::new(5);
    // BlindSelect (Small): select + skip.
    assert_eq!(g.legal_actions().len(), 2);
    check_state(g.state(), &mut rng);
    g.step(Action::SelectBlind).unwrap();
    // 8 cards, 4 hands, 4 discards: 218 masks each.
    assert_eq!(g.state().phase, Phase::Play);
    let n = g.legal_actions().len();
    assert_eq!(n, 218 * 2);
    check_all_masks(g.state());
    // No hands left: only discards.
    g.state_mut().hands_left = 0;
    assert!(
        g.legal_actions()
            .iter()
            .all(|a| matches!(a, Action::Discard(_)))
    );
    check_state(g.state(), &mut rng);
    // No discards either: nothing to list (a state the engine never rests in, but it is consistent).
    g.state_mut().discards_left = 0;
    assert!(g.legal_actions().is_empty());
    check_state(g.state(), &mut rng);
    // Boss blind cannot be skipped.
    let mut g = Game::new(12, GameConfig::default());
    g.state_mut().blind = balatro_core::blinds::BlindKind::Boss;
    assert_eq!(g.legal_actions(), vec![Action::SelectBlind]);
    check_state(g.state(), &mut rng);
}
