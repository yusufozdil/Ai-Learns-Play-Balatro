//! Phase 8A: the structured action space (`[kind, arg0, arg1, cards]`) is consistent with
//! `legal_actions` / `validate`.
//!
//! * completeness: every legal action is `permits`-ed by the masks;
//! * soundness: every (kind, arg0, arg1, card set) the masks allow passes `validate` (exhaustive
//!   over the card sets of small hands, sampled through the sequential-selection helpers for
//!   big ones), and for small hands the allowed set equals the legal set exactly;
//! * sequential selection (`next_candidates` / `can_finish`) never dead-ends and only produces
//!   valid sets;
//! * `Action::parts` / `from_parts` round-trip.

use balatro_core::action_mask::{ActionKind, CardSelect, SlotSet, card_select, structured_mask};
use balatro_core::actions::{Action, legal_actions, validate};
use balatro_core::blinds::BlindKind;
use balatro_core::consumable::ConsumableInstance;
use balatro_core::content::{BossId, ConsumableId};
use balatro_core::deck::Pile;
use balatro_core::enums::{Edition, Phase};
use balatro_core::game::Game;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::{GameConfig, GameState};
use std::collections::HashSet;

/// Random walk over the sequential selection; returns the finished set (or `None` if it chose
/// to stop without one, which the caller treats as a dead end).
fn walk(sel: &CardSelect, rng: &mut GameRng) -> u32 {
    let mut partial = 0u32;
    loop {
        let can_stop = sel.can_finish(partial);
        let next = sel.next_candidates(partial);
        assert!(
            can_stop || next != 0,
            "dead end: partial {partial:#b} of {sel:?}"
        );
        if next == 0 || (can_stop && rng.range(RngKey::Misc, 0, 3) == 0) {
            return partial;
        }
        let k = rng.range(RngKey::Misc, 0, next.count_ones() as usize);
        let bit = (0..32).filter(|i| next >> i & 1 == 1).nth(k).unwrap();
        assert_eq!(partial >> bit & 1, 0, "offered a selected card");
        assert!(sel.candidates >> bit & 1 == 1);
        partial |= 1 << bit;
    }
}

fn check_state(state: &GameState, rng: &mut GameRng) {
    let m = structured_mask(state);
    let mut legal = Vec::new();
    legal_actions(state, &mut legal);
    let legal_set: HashSet<Action> = legal.iter().copied().collect();

    // Completeness + round trip.
    for &a in &legal {
        assert!(
            m.permits(state, a),
            "legal {a:?} not permitted in {:?} (mask {m:?})",
            state.phase
        );
        let (k, a0, a1, c) = a.parts();
        assert_eq!(Action::from_parts(k, a0, a1, c), a);
        assert_eq!(k, a.kind());
    }
    if state.is_terminal() {
        assert!(m.kinds.is_empty());
        return;
    }
    // Every legal kind is in the mask, and the other way round for operand-free kinds.
    for k in ActionKind::ALL {
        let any = legal.iter().any(|a| a.kind() == k);
        assert_eq!(m.allows_kind(k), any, "kind {k:?}");
    }

    // Soundness (+ equality for fully enumerated states).
    let mut permitted: Vec<Action> = Vec::new();
    let mut sampled = false;
    for kind in m.kinds.iter() {
        let arg0s = m.arg0(kind);
        assert!(!arg0s.is_empty(), "{kind:?} allowed with no arg0");
        if !kind.has_arg0() {
            assert_eq!(arg0s, SlotSet::zero());
        }
        for a0 in arg0s.iter() {
            let a1s: Vec<u8> = if kind.has_arg1() {
                let to = m.move_to(a0);
                assert!(!to.contains(a0));
                to.iter().collect()
            } else {
                vec![0]
            };
            for a1 in a1s {
                match card_select(state, kind, a0) {
                    None => permitted.push(Action::from_parts(kind, a0, a1, 0)),
                    Some(sel) => {
                        assert!(kind.may_take_cards());
                        assert!(sel.min >= 1 && sel.min <= sel.max, "{sel:?}");
                        assert_eq!(sel.required & !sel.candidates, 0);
                        let n = 32 - sel.candidates.leading_zeros();
                        if n <= 10 {
                            let sets: Vec<u32> = sel.iter_sets().collect();
                            assert!(!sets.is_empty(), "{kind:?} {a0}: no card set");
                            for s in sets {
                                permitted.push(Action::from_parts(kind, a0, a1, s));
                            }
                        } else {
                            sampled = true;
                            for _ in 0..24 {
                                let s = walk(&sel, rng);
                                assert!(sel.is_valid_set(s));
                                permitted.push(Action::from_parts(kind, a0, a1, s));
                            }
                        }
                    }
                }
            }
        }
    }
    for &a in &permitted {
        assert_eq!(
            validate(state, a),
            Ok(()),
            "permitted {a:?} fails validate in {:?}",
            state.phase
        );
        assert!(m.permits(state, a));
    }
    if !sampled {
        let permitted_set: HashSet<Action> = permitted.iter().copied().collect();
        assert_eq!(permitted_set.len(), permitted.len(), "duplicate permitted");
        assert_eq!(
            permitted_set, legal_set,
            "mask and legal_actions disagree in {:?}",
            state.phase
        );
    }
    // Pack/Use items with card selections always have a legal set (tightness of arg0).
    for kind in [ActionKind::UseConsumable, ActionKind::PickPackItem] {
        for a0 in m.arg0(kind).iter() {
            if let Some(sel) = card_select(state, kind, a0) {
                let s = walk(&sel, rng);
                let a = Action::from_parts(kind, a0, 0, s);
                assert_eq!(validate(state, a), Ok(()), "{a:?}");
            }
        }
    }
}

fn weighted_pick(legal: &[Action], rng: &mut GameRng) -> Action {
    let w: Vec<f64> = legal
        .iter()
        .map(|a| match a {
            Action::Play(_) => 2.0,
            Action::Discard(_) => 1.0,
            Action::EndShop => 0.3,
            Action::SkipPack => 0.4,
            Action::SelectBlind => 1.0,
            Action::SellJoker(_) | Action::SellConsumable(_) => 0.5,
            _ => 5.0,
        })
        .collect();
    legal[rng.weighted_index(RngKey::Misc, &w).unwrap_or(0)]
}

/// Perturb the position a bit so rare structures show up: consumables of every kind, editions on
/// hand cards (Aura), a forced card (Cerulean Bell), a hand longer than 16 cards.
fn perturb(g: &mut Game, rng: &mut GameRng) {
    if g.state().phase == Phase::Play && rng.range(RngKey::Misc, 0, 6) == 0 {
        let s = g.state_mut();
        for c in s.deck.hand.iter_mut() {
            if rng.range(RngKey::Misc, 0, 3) == 0 {
                c.edition = Edition::Foil;
            }
        }
    }
    if rng.range(RngKey::Misc, 0, 5) == 0 {
        let id = ConsumableId::ALL[rng.range(RngKey::Misc, 0, ConsumableId::COUNT)];
        let _ = g.add_consumable(ConsumableInstance::new(id));
    }
    if g.state().phase == Phase::Play && rng.range(RngKey::Misc, 0, 25) == 0 {
        let n = 17 + rng.range(RngKey::Misc, 0, 4);
        let d = &mut g.state_mut().deck;
        while d.hand.len() < n {
            let Some(c) = d.draw_pile.pop() else { break };
            d.add_card(c, Pile::Hand);
        }
        d.sort_hand();
    }
}

#[test]
fn structured_masks_agree_with_legal_actions() {
    let mut rng = GameRng::new(0xA11CE);
    let mut states = 0u64;
    let mut targeted = 0u64;
    let mut long_hands = 0u64;
    let mut pick_targets = 0u64;
    let mut shops = 0u64;
    for seed in 0..150u64 {
        let mut g = Game::new(seed, GameConfig::default());
        for _ in 0..260 {
            if g.is_terminal() {
                break;
            }
            perturb(&mut g, &mut rng);
            // Forced card (Cerulean Bell) on a copy.
            if g.state().phase == Phase::Play && rng.range(RngKey::Misc, 0, 5) == 0 {
                let mut probe = g.clone();
                let h = probe.state().deck.hand.len();
                let uid = probe.state().deck.hand[rng.range(RngKey::Misc, 0, h)].uid;
                let s = probe.state_mut();
                s.boss = Some(BossId::CeruleanBell);
                s.blind = BlindKind::Boss;
                s.boss_disabled = false;
                s.boss_state.card_uid = Some(uid);
                check_state(probe.state(), &mut rng);
            }
            check_state(g.state(), &mut rng);
            states += 1;
            if g.state().deck.hand.len() > 16 {
                long_hands += 1;
            }
            if g.state().phase == Phase::Shop {
                shops += 1;
            }
            if g.state().phase == Phase::PackOpen {
                let m = structured_mask(g.state());
                pick_targets += m
                    .pick_pack_item
                    .iter()
                    .filter(|&i| card_select(g.state(), ActionKind::PickPackItem, i).is_some())
                    .count() as u64;
            }
            targeted += g
                .state()
                .consumables
                .iter()
                .filter(|c| {
                    balatro_core::content::try_consumable_def(c.id)
                        .is_some_and(|d| d.max_targets > 0)
                })
                .count() as u64;
            let mut legal = Vec::new();
            legal_actions(g.state(), &mut legal);
            let a = weighted_pick(&legal, &mut rng);
            // Score shortcut: reach the shops, packs and deep antes instead of dying early.
            if matches!(a, Action::Play(_)) && rng.range(RngKey::Misc, 0, 100) < 55 {
                let st = g.state_mut();
                st.round_score = st.round_score.max(st.target - 1.0);
            }
            g.step(a).expect("legal action");
        }
        check_state(g.state(), &mut rng);
    }
    assert!(states > 3000, "only {states} states");
    assert!(targeted > 100, "only {targeted} targeted consumables seen");
    assert!(long_hands > 20, "only {long_hands} long-hand states");
    assert!(shops > 200, "only {shops} shop states");
    assert!(pick_targets > 20, "only {pick_targets} targeted pack picks");
}

#[test]
fn aura_candidates_exclude_editioned_cards() {
    let mut g = Game::new(21, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    {
        let h = &mut g.state_mut().deck.hand;
        h[0].edition = Edition::Negative;
        h[3].edition = Edition::Foil;
    }
    g.add_consumable(ConsumableInstance::new(ConsumableId::Aura))
        .expect("room");
    let sel = g
        .card_select(ActionKind::UseConsumable, 0)
        .expect("Aura takes a card");
    assert_eq!((sel.min, sel.max), (1, 1));
    assert_eq!(sel.candidates, 0b1111_0110);
    let m = g.structured_mask();
    assert!(m.use_consumable.contains(0));
    check_state(g.state(), &mut GameRng::new(1));
    // No eligible card at all: the slot is not offered.
    for c in g.state_mut().deck.hand.iter_mut() {
        c.edition = Edition::Foil;
    }
    assert!(g.card_select(ActionKind::UseConsumable, 0).is_none());
    assert!(!g.structured_mask().use_consumable.contains(0));
    assert!(g.structured_mask().sell_consumable.contains(0));
}

#[test]
fn cerulean_bell_makes_the_forced_card_mandatory() {
    let mut g = Game::new(5, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    let uid = g.state().deck.hand[6].uid;
    {
        let s = g.state_mut();
        s.boss = Some(BossId::CeruleanBell);
        s.blind = BlindKind::Boss;
        s.boss_disabled = false;
        s.boss_state.card_uid = Some(uid);
    }
    let sel = g.card_select(ActionKind::Play, 0).unwrap();
    assert_eq!(sel.required, 1 << 6);
    assert_eq!((sel.min, sel.max), (1, 5));
    // Only the forced card may be added once four others are chosen.
    let partial = 0b1111u32;
    assert_eq!(sel.next_candidates(partial), 1 << 6);
    assert!(!sel.can_finish(partial));
    assert!(sel.can_finish(partial | 1 << 6));
    assert_eq!(sel.next_candidates(partial | 1 << 6), 0);
    check_state(g.state(), &mut GameRng::new(2));
}

#[test]
fn sequential_selection_never_dead_ends() {
    let mut rng = GameRng::new(77);
    for (cand, req, min, max) in [
        (0xFFu32, 0u32, 1u8, 5u8),
        (0x1_FFFF, 1 << 16, 1, 5),
        (0xF0F0, 0, 2, 3),
        (0b1011, 0, 3, 3),
        (0xFFFF_FFFF, 1 << 31, 1, 5),
    ] {
        let sel = CardSelect {
            candidates: cand,
            required: req,
            min,
            max,
        };
        for _ in 0..300 {
            let s = walk(&sel, &mut rng);
            assert!(sel.is_valid_set(s), "{sel:?} {s:#b}");
        }
    }
}

#[test]
fn kind_indices_are_stable() {
    for (i, k) in ActionKind::ALL.into_iter().enumerate() {
        assert_eq!(k.index(), i);
        assert_eq!(ActionKind::from_index(i), Some(k));
    }
    assert_eq!(ActionKind::from_index(ActionKind::COUNT), None);
    assert_eq!(ActionKind::COUNT, 17);
}
