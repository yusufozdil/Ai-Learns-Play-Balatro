//! Phase 8A: card-selection masks are `u32`. Hands longer than 16 cards (Certificate, DNA,
//! Familiar & co. add cards on top of the hand size) can select every position up to 31; the
//! legal-action tables stay exact for small hands and are generated lazily for big ones.

use balatro_core::actions::{Action, ActionError, MaskIter, validate};
use balatro_core::blinds::BlindKind;
use balatro_core::consumable::ConsumableInstance;
use balatro_core::content::{BossId, ConsumableId};
use balatro_core::deck::Pile;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use std::collections::HashSet;

/// A game in `Phase::Play` whose hand holds exactly `n` cards (extra cards come from the draw
/// pile, so every invariant holds).
fn game_with_hand(seed: u64, n: usize) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    let d = &mut g.state_mut().deck;
    while d.hand.len() < n {
        let c = d.draw_pile.pop().expect("deck has enough cards");
        d.add_card(c, Pile::Hand);
    }
    d.sort_hand();
    assert_eq!(d.hand.len(), n);
    g.state().check_invariants().expect("invariants");
    g
}

/// C(n, 1) + ... + C(n, 5).
fn selections(n: u64) -> u64 {
    let mut total = 0;
    let mut c = 1;
    for k in 1..=5u64.min(n) {
        c = c * (n + 1 - k) / k;
        total += c;
    }
    total
}

#[test]
fn seventeenth_card_can_be_played() {
    let mut g = game_with_hand(3, 17);
    let a = Action::Play(1 << 16);
    assert_eq!(g.validate(a), Ok(()));
    assert!(g.legal_actions().contains(&a));
    let last = g.state().deck.hand[16];
    let info = g.step(a).expect("play the 17th card");
    assert!(info.hand.is_some());
    // The played card left the hand (it is gone from the hand or went to the discard pile).
    assert!(!g.state().deck.hand.iter().any(|c| c.uid == last.uid));
    // Position 17 does not exist in a 17-card hand.
    let g = game_with_hand(3, 17);
    assert!(matches!(
        validate(g.state(), Action::Play(1 << 17)),
        Err(ActionError::SelectionOutOfRange { .. })
    ));
    // A mask mixing low and high positions is fine up to five cards, and six is rejected.
    assert_eq!(g.validate(Action::Discard(1 | 1 << 8 | 1 << 16)), Ok(()));
    assert!(matches!(
        g.validate(Action::Discard(0b11_1111 << 11)),
        Err(ActionError::TooManyCards { .. })
    ));
}

#[test]
fn legal_actions_are_complete_for_long_hands() {
    for n in [16usize, 17, 20] {
        let g = game_with_hand(5, n);
        let legal = g.legal_actions();
        let expected = 2 * selections(n as u64);
        let (plays, discards) = (
            legal
                .iter()
                .filter(|a| matches!(a, Action::Play(_)))
                .count(),
            legal
                .iter()
                .filter(|a| matches!(a, Action::Discard(_)))
                .count(),
        );
        assert_eq!((plays + discards) as u64, expected, "hand of {n}");
        let mut scratch = Vec::new();
        assert_eq!(g.legal_count(&mut scratch), legal.len());
        assert_eq!(g.selection_space().len() as u64, expected);
        // Every listed Play validates; ascending order; no duplicates.
        let masks: Vec<u32> = legal
            .iter()
            .filter_map(|a| match a {
                Action::Play(m) => Some(*m),
                _ => None,
            })
            .collect();
        assert!(masks.windows(2).all(|w| w[0] < w[1]));
        for &m in masks.iter().step_by(37) {
            assert_eq!(g.validate(Action::Play(m)), Ok(()));
        }
        // The highest mask of a 17+ card hand reaches the last position.
        let top = *masks.last().unwrap();
        assert!(top >> (n - 1) & 1 == 1);
    }
}

#[test]
fn thirty_two_card_hand_is_generated_lazily() {
    let g = game_with_hand(7, 32);
    let sp = g.selection_space();
    assert!(sp.table().is_none());
    assert_eq!(sp.mask_count() as u64, selections(32));
    assert_eq!(sp.iter().count() as u64, selections(32));
    // Highest selection: the last five positions.
    assert_eq!(sp.iter().last(), Some(0b11111u32 << 27));
    assert_eq!(g.validate(Action::Play(1 << 31)), Ok(()));
    let mut g = g;
    g.step(Action::Play(1 << 31 | 1 << 30)).expect("play");
}

#[test]
fn generated_masks_equal_the_tables_for_small_hands() {
    // The 16-card table holds every smaller hand's masks as a prefix.
    let g = game_with_hand(11, 16);
    let table = g.selection_space().table().expect("table for <= 16 cards");
    assert_eq!(table.len() as u64, selections(16));
    for n in 1..=16usize {
        let prefix: Vec<u32> = table.iter().copied().filter(|&m| m < (1u32 << n)).collect();
        let generated: Vec<u32> = MaskIter::new(n, 5, 0).collect();
        assert_eq!(generated, prefix, "hand of {n}");
        assert_eq!(prefix.len() as u64, selections(n as u64));
    }
    // A required card filters without changing the order.
    let with_req: Vec<u32> = MaskIter::new(10, 5, 1 << 4).collect();
    let expect: Vec<u32> = MaskIter::new(10, 5, 0)
        .filter(|m| m >> 4 & 1 == 1)
        .collect();
    assert_eq!(with_req, expect);
}

#[test]
fn forced_card_past_position_16_is_respected() {
    let mut g = game_with_hand(13, 19);
    let uid = g.state().deck.hand[18].uid;
    {
        let s = g.state_mut();
        s.boss = Some(BossId::CeruleanBell);
        s.blind = BlindKind::Boss;
        s.boss_disabled = false;
        s.boss_state.card_uid = Some(uid);
    }
    let sp = g.selection_space();
    assert_eq!(sp.forced, Some(18));
    let legal = g.legal_actions();
    let plays: Vec<u32> = legal
        .iter()
        .filter_map(|a| match a {
            Action::Play(m) => Some(*m),
            _ => None,
        })
        .collect();
    assert!(plays.iter().all(|m| m >> 18 & 1 == 1));
    // C(18, 0..=4): the other cards fill 0..=4 places.
    let expected: u64 = (0..=4u64).map(|k| binom(18, k)).sum();
    assert_eq!(plays.len() as u64, expected);
    assert_eq!(sp.mask_count() as u64, expected);
    assert_eq!(
        g.validate(Action::Play(1)),
        Err(ActionError::ForcedCardMissing { position: 18 })
    );
    assert_eq!(g.validate(Action::Play(1 << 18)), Ok(()));
}

fn binom(n: u64, k: u64) -> u64 {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

#[test]
fn consumable_targets_reach_high_positions() {
    let mut g = game_with_hand(17, 20);
    g.add_consumable(ConsumableInstance::new(ConsumableId::TheHangedMan))
        .expect("room");
    let a = Action::UseConsumable {
        slot: 0,
        targets: 1 << 19 | 1 << 17,
    };
    assert_eq!(g.validate(a), Ok(()));
    assert!(g.legal_actions().contains(&a));
    // Out of range for the 20-card hand.
    assert!(matches!(
        g.validate(Action::UseConsumable {
            slot: 0,
            targets: 1 << 20
        }),
        Err(ActionError::SelectionOutOfRange { .. })
    ));
    // The Hanged Man takes 1..=2 cards: C(20,1) + C(20,2) legal uses.
    let uses = g
        .legal_actions()
        .into_iter()
        .filter(|a| matches!(a, Action::UseConsumable { .. }))
        .count();
    assert_eq!(uses as u64, 20 + 190);
    let before = g.state().deck.hand.len();
    g.step(a).expect("use");
    assert_eq!(g.state().deck.hand.len(), before - 2);
    let seen: HashSet<u32> = g.state().deck.hand.iter().map(|c| c.uid).collect();
    assert_eq!(seen.len(), g.state().deck.hand.len());
}
