//! Phase 4B: the 18 Spectral cards through `Game::step` (use, target-count errors, `can_use`,
//! effects, seed determinism, card-count invariants, joker reactions). Guide:
//! docs/design/consumables_bosses.md; rules docs/mechanics/consumables.md §3, notes.md §4.1b.

use balatro_core::actions::{Action, ActionError};
use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::card::Card;
use balatro_core::consumable::ConsumableInstance;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{
    ConsumableId as C, consumable_def_count, spectrals, try_consumable_def,
};
use balatro_core::data::consumables::ADDED_CARD_ENHANCEMENTS;
use balatro_core::enums::{Edition, Enhancement, Phase, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

// ----- helpers ------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

/// Small blind selected: Play phase with 8 cards.
fn in_play(seed: u64) -> Game {
    let mut g = new_game(seed);
    g.step(Action::SelectBlind).expect("select");
    g
}

/// Beat the Small blind: Shop phase.
fn in_shop(seed: u64) -> Game {
    let mut g = in_play(seed);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("winning play");
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

fn give(g: &mut Game, id: C) -> usize {
    g.add_consumable(ConsumableInstance::new(id)).expect("room")
}

fn use_c(g: &mut Game, slot: u8, targets: u16) -> Result<StepInfo, ActionError> {
    g.step(Action::UseConsumable { slot, targets })
}

fn add_joker(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("joker room")
}

fn add_joker_ed(g: &mut Game, id: JokerId, e: Edition) -> usize {
    g.add_joker(JokerInstance::new(0, id).with_edition(e))
        .expect("joker room")
}

/// Replace hand card `pos` with a plain card (keeps its uid).
fn set_card(g: &mut Game, pos: usize, rank: Rank, suit: Suit) {
    let c = &mut g.state_mut().deck.hand[pos];
    c.rank = rank;
    c.suit = suit;
    c.enhancement = Enhancement::None;
    c.edition = Edition::Base;
    c.seal = Seal::None;
    c.perma_bonus = 0;
}

/// Make every hand card a plain King of Spades.
fn all_kings(g: &mut Game) {
    for pos in 0..g.state().deck.hand.len() {
        set_card(g, pos, Rank::King, Suit::Spades);
    }
}

/// Put all but `n` hand cards back on the draw pile (cards stay conserved).
fn shrink_hand(g: &mut Game, n: usize) {
    let d = &mut g.state_mut().deck;
    let rest: Vec<Card> = d.hand.drain(n..).collect();
    d.draw_pile.extend(rest);
}

fn hand_uids(g: &Game) -> Vec<u32> {
    g.state().deck.hand.iter().map(|c| c.uid).collect()
}

fn joker_value(g: &Game, id: JokerId) -> f64 {
    g.state()
        .jokers
        .iter()
        .find(|j| j.id == id)
        .expect("joker owned")
        .value
}

fn invariants(g: &Game) {
    g.state().check_invariants().expect("invariants");
}

const UNTARGETED: [C; 11] = [
    C::Familiar,
    C::Grim,
    C::Incantation,
    C::Wraith,
    C::Sigil,
    C::Ouija,
    C::Ectoplasm,
    C::Immolate,
    C::Ankh,
    C::Hex,
    C::TheSoul,
];
const TARGETED: [C; 6] = [
    C::Talisman,
    C::Aura,
    C::DejaVu,
    C::Trance,
    C::Medium,
    C::Cryptid,
];

// ----- registry and validation -------------------------------------------------------------

#[test]
fn all_eighteen_spectrals_are_defined_with_the_right_shape() {
    assert_eq!(spectrals::DEFS.len(), 18);
    for &id in C::SPECTRALS {
        assert_eq!(consumable_def_count(id), 1, "{id:?}");
        let d = try_consumable_def(id).expect("defined");
        if TARGETED.contains(&id) {
            assert_eq!((d.min_targets, d.max_targets), (1, 1), "{}", d.name);
        } else {
            assert_eq!((d.min_targets, d.max_targets), (0, 0), "{}", d.name);
        }
    }
    for id in [
        C::Familiar,
        C::Grim,
        C::Incantation,
        C::Sigil,
        C::Ouija,
        C::Immolate,
    ] {
        assert!(try_consumable_def(id).expect("defined").needs_hand);
    }
    for id in [
        C::Wraith,
        C::Ectoplasm,
        C::Ankh,
        C::Hex,
        C::TheSoul,
        C::BlackHole,
    ] {
        assert!(!try_consumable_def(id).expect("defined").requires_hand());
    }
}

#[test]
fn target_counts_are_validated_for_every_spectral() {
    for id in TARGETED {
        let mut g = in_play(1);
        give(&mut g, id);
        for (mask, selected) in [(0u16, 0u32), (0b11, 2)] {
            let before = g.state().clone();
            assert_eq!(
                use_c(&mut g, 0, mask),
                Err(ActionError::TargetCount {
                    selected,
                    min: 1,
                    max: 1
                }),
                "{id:?} {mask:b}"
            );
            assert_eq!(g.state(), &before);
        }
    }
    for id in UNTARGETED {
        let mut g = in_play(2);
        give(&mut g, id);
        assert_eq!(
            use_c(&mut g, 0, 1),
            Err(ActionError::TargetCount {
                selected: 1,
                min: 0,
                max: 0
            }),
            "{id:?}"
        );
    }
}

#[test]
fn hand_cards_need_a_drawn_hand_but_joker_cards_work_in_the_shop() {
    for id in [
        C::Familiar,
        C::Grim,
        C::Incantation,
        C::Sigil,
        C::Ouija,
        C::Immolate,
    ] {
        let mut g = in_shop(3);
        give(&mut g, id);
        assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::NeedsHand), "{id:?}");
    }
    for id in TARGETED {
        let mut g = in_shop(3);
        give(&mut g, id);
        assert_eq!(use_c(&mut g, 0, 1), Err(ActionError::NeedsHand), "{id:?}");
    }
    // Ectoplasm, Hex, Wraith, The Soul and Ankh need no hand.
    let mut g = in_shop(4);
    add_joker(&mut g, JokerId::Joker);
    give(&mut g, C::Ectoplasm);
    use_c(&mut g, 0, 0).expect("Ectoplasm in the shop");
    assert_eq!(g.state().jokers[0].edition, Edition::Negative);
    give(&mut g, C::Wraith);
    use_c(&mut g, 0, 0).expect("Wraith in the shop");
    assert_eq!(g.state().money, 0);
}

#[test]
fn hand_rewriting_cards_need_two_hand_cards() {
    for id in [
        C::Familiar,
        C::Grim,
        C::Incantation,
        C::Sigil,
        C::Ouija,
        C::Immolate,
    ] {
        let mut g = in_play(5);
        shrink_hand(&mut g, 1);
        give(&mut g, id);
        assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "{id:?}");
        // Two cards are enough.
        let mut g = in_play(5);
        shrink_hand(&mut g, 2);
        give(&mut g, id);
        use_c(&mut g, 0, 0).unwrap_or_else(|e| panic!("{id:?}: {e}"));
        invariants(&g);
    }
}

// ----- Familiar / Grim / Incantation ----------------------------------------------------------

/// Use `id` in a fresh Play game; returns (game, uids before).
fn add_cards_case(id: C, seed: u64) -> (Game, Vec<u32>) {
    let mut g = in_play(seed);
    let before = hand_uids(&g);
    give(&mut g, id);
    use_c(&mut g, 0, 0).expect("use");
    (g, before)
}

#[test]
fn familiar_grim_incantation_destroy_one_and_add_enhanced_cards() {
    // (card, added count, allowed ranks)
    let cases: [(C, usize, Vec<Rank>); 3] = [
        (C::Familiar, 3, vec![Rank::Jack, Rank::Queen, Rank::King]),
        (C::Grim, 2, vec![Rank::Ace]),
        (
            C::Incantation,
            4,
            (2u8..=10)
                .map(|v| Rank::from_value(v).expect("rank"))
                .collect(),
        ),
    ];
    for (id, n, ranks) in cases {
        for seed in 0..10 {
            let (g, before) = add_cards_case(id, 100 + seed);
            let s = g.state();
            let after = hand_uids(&g);
            let gone: Vec<u32> = before
                .iter()
                .copied()
                .filter(|u| !after.contains(u))
                .collect();
            let fresh: Vec<&Card> = s
                .deck
                .hand
                .iter()
                .filter(|c| !before.contains(&c.uid))
                .collect();
            assert_eq!(gone.len(), 1, "{id:?}: exactly one card destroyed");
            assert_eq!(fresh.len(), n, "{id:?}");
            assert_eq!(s.deck.hand.len(), 8 - 1 + n);
            for c in fresh {
                assert!(ranks.contains(&c.rank), "{id:?} {:?}", c.rank);
                assert!(ADDED_CARD_ENHANCEMENTS.contains(&c.enhancement), "{id:?}");
                assert_eq!((c.edition, c.seal), (Edition::Base, Seal::None));
            }
            assert_eq!(
                (s.stats.cards_destroyed, s.stats.cards_added),
                (1, n as u32)
            );
            assert_eq!(s.deck.len(), 52 - 1 + n);
            assert_eq!(s.stats.run.tarots_used, 0, "a Spectral, not a Tarot");
            assert_eq!(s.last_tarot_planet, None);
            invariants(&g);
        }
    }
}

#[test]
fn added_cards_cover_every_suit_enhancement_and_rank_over_many_seeds() {
    let mut suits = [false; 4];
    let mut enhs = vec![false; ADDED_CARD_ENHANCEMENTS.len()];
    let mut ranks = [false; 15];
    for seed in 0..60 {
        let (g, before) = add_cards_case(C::Incantation, 200 + seed);
        for c in g
            .state()
            .deck
            .hand
            .iter()
            .filter(|c| !before.contains(&c.uid))
        {
            suits[c.suit.index()] = true;
            let i = ADDED_CARD_ENHANCEMENTS
                .iter()
                .position(|&e| e == c.enhancement)
                .expect("allowed enhancement");
            enhs[i] = true;
            ranks[c.rank as usize] = true;
        }
    }
    assert!(suits.iter().all(|&b| b), "suits {suits:?}");
    assert!(enhs.iter().all(|&b| b), "enhancements {enhs:?}");
    assert!((2..=10).all(|r| ranks[r]), "ranks {ranks:?}");
    assert!(!ranks[11] && !ranks[12] && !ranks[13] && !ranks[14]);
}

#[test]
fn random_spectral_effects_are_deterministic_per_seed() {
    for id in [
        C::Familiar,
        C::Grim,
        C::Incantation,
        C::Sigil,
        C::Ouija,
        C::Immolate,
    ] {
        let run = |seed: u64| {
            let mut g = in_play(seed);
            give(&mut g, id);
            use_c(&mut g, 0, 0).expect("use");
            g.state().clone()
        };
        assert_eq!(run(7), run(7), "{id:?}");
        // ... and the result depends on the seed.
        assert!(
            (0..8).any(|s| run(s).deck.hand != run(s + 1).deck.hand),
            "{id:?}"
        );
    }
    for id in [C::Aura, C::Ectoplasm, C::Hex, C::Wraith, C::Ankh] {
        let run = |seed: u64| {
            let mut g = in_play(seed);
            for j in [JokerId::Joker, JokerId::Baron, JokerId::GreenJoker] {
                add_joker(&mut g, j);
            }
            give(&mut g, id);
            let t = u16::from(id == C::Aura);
            use_c(&mut g, 0, t).expect("use");
            g.state().clone()
        };
        assert_eq!(run(9), run(9), "{id:?}");
    }
}

#[test]
fn destroying_cards_triggers_canio_glass_joker_and_hologram() {
    // Familiar on a hand of Kings: one face card destroyed (Canio +1), 3 added (Hologram +0.75).
    let mut g = in_play(11);
    all_kings(&mut g);
    add_joker(&mut g, JokerId::Canio);
    add_joker(&mut g, JokerId::Hologram);
    give(&mut g, C::Familiar);
    use_c(&mut g, 0, 0).expect("Familiar");
    assert!((joker_value(&g, JokerId::Canio) - 2.0).abs() < 1e-9);
    assert!((joker_value(&g, JokerId::Hologram) - 1.75).abs() < 1e-9);
    invariants(&g);

    // Immolate on Glass Kings: five destroyed face cards, five Glass breaks.
    let mut g = in_play(12);
    all_kings(&mut g);
    for c in &mut g.state_mut().deck.hand {
        c.enhancement = Enhancement::Glass;
    }
    add_joker(&mut g, JokerId::Canio);
    add_joker(&mut g, JokerId::GlassJoker);
    give(&mut g, C::Immolate);
    use_c(&mut g, 0, 0).expect("Immolate");
    assert!((joker_value(&g, JokerId::Canio) - 6.0).abs() < 1e-9);
    assert!((joker_value(&g, JokerId::GlassJoker) - (1.0 + 0.75 * 5.0)).abs() < 1e-9);
    invariants(&g);

    // Cryptid: two copies added -> Hologram +0.5.
    let mut g = in_play(13);
    add_joker(&mut g, JokerId::Hologram);
    give(&mut g, C::Cryptid);
    use_c(&mut g, 0, 1).expect("Cryptid");
    assert!((joker_value(&g, JokerId::Hologram) - 1.5).abs() < 1e-9);
}

#[test]
fn added_cards_can_be_played_and_the_run_continues() {
    let mut g = in_play(14);
    give(&mut g, C::Incantation);
    use_c(&mut g, 0, 0).expect("Incantation");
    assert_eq!(g.state().deck.hand.len(), 11);
    g.step(Action::Play(0b11111)).expect("play");
    invariants(&g);
    assert_eq!(g.state().deck.len(), 52 - 1 + 4);
}

// ----- seals ------------------------------------------------------------------------------------

#[test]
fn seal_spectrals_seal_exactly_the_target_and_replace_old_seals() {
    for (id, seal) in [
        (C::Talisman, Seal::Gold),
        (C::DejaVu, Seal::Red),
        (C::Trance, Seal::Blue),
        (C::Medium, Seal::Purple),
    ] {
        let mut g = in_play(20);
        g.state_mut().deck.hand[3].seal = Seal::Red; // replaced (or kept for Deja Vu)
        g.state_mut().deck.hand[3].enhancement = Enhancement::Glass;
        let target = g.state().deck.hand[3].uid;
        give(&mut g, id);
        use_c(&mut g, 0, 1 << 3).expect("use");
        for c in &g.state().deck.hand {
            let want = if c.uid == target { seal } else { Seal::None };
            assert_eq!(c.seal, want, "{id:?}");
            if c.uid == target {
                assert_eq!(c.enhancement, Enhancement::Glass, "other fields kept");
            }
        }
        assert_eq!(g.state().deck.len(), 52);
        invariants(&g);
    }
}

// ----- Aura -----------------------------------------------------------------------------------------

#[test]
fn aura_gives_foil_holo_or_polychrome_to_an_editionless_card() {
    let mut seen = [0u32; 5];
    for seed in 0..120 {
        let mut g = in_play(300 + seed);
        let target = g.state().deck.hand[2].uid;
        give(&mut g, C::Aura);
        use_c(&mut g, 0, 1 << 2).expect("Aura");
        for c in &g.state().deck.hand {
            if c.uid == target {
                seen[c.edition.index()] += 1;
            } else {
                assert_eq!(c.edition, Edition::Base);
            }
        }
        invariants(&g);
    }
    assert_eq!(seen[Edition::Base.index()], 0);
    assert_eq!(seen[Edition::Negative.index()], 0, "never Negative");
    // Weights 50 / 35 / 15 -> Foil most often, Polychrome least (120 draws: wide margins).
    let (f, h, p) = (
        seen[Edition::Foil.index()],
        seen[Edition::Holographic.index()],
        seen[Edition::Polychrome.index()],
    );
    assert!(f > 0 && h > 0 && p > 0, "{seen:?}");
    assert!(f > p && h > p / 2, "{seen:?}");
}

#[test]
fn aura_refuses_a_card_that_already_has_an_edition() {
    let mut g = in_play(21);
    g.state_mut().deck.hand[0].edition = Edition::Foil;
    give(&mut g, C::Aura);
    let before = g.state().clone();
    assert_eq!(use_c(&mut g, 0, 1), Err(ActionError::CannotUse));
    assert_eq!(g.state(), &before);
    use_c(&mut g, 0, 0b10).expect("an edition-less card works");
    // legal_actions agrees: no mask selecting only card 0.
    let mut g = in_play(21);
    g.state_mut().deck.hand[0].edition = Edition::Negative;
    give(&mut g, C::Aura);
    let legal = g.legal_actions();
    let aura_uses: Vec<u16> = legal
        .iter()
        .filter_map(|a| match a {
            Action::UseConsumable { slot: 0, targets } => Some(*targets),
            _ => None,
        })
        .collect();
    assert_eq!(aura_uses.len(), 7, "{aura_uses:?}");
    assert!(!aura_uses.contains(&1));
}

// ----- Wraith / The Soul -------------------------------------------------------------------------------

#[test]
fn wraith_sets_money_to_zero_and_queues_a_rare_joker() {
    for money in [37, 0, -3] {
        let mut g = in_play(30);
        g.state_mut().money = money;
        give(&mut g, C::Wraith);
        use_c(&mut g, 0, 0).expect("Wraith");
        assert_eq!(g.state().money, 0, "money {money}");
        // Joker creation is dropped until Phase 5.
        assert_eq!(g.state().stats.effects_dropped, 1);
        assert!(g.state().jokers.is_empty());
        invariants(&g);
    }
}

#[test]
fn wraith_and_the_soul_need_a_free_joker_slot() {
    for id in [C::Wraith, C::TheSoul] {
        let mut g = in_play(31);
        for _ in 0..5 {
            add_joker(&mut g, JokerId::Joker);
        }
        give(&mut g, id);
        g.state_mut().money = 9;
        assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "{id:?}");
        assert_eq!(g.state().money, 9);
        // A Negative joker opens a sixth slot.
        let mut g = in_play(31);
        for _ in 0..4 {
            add_joker(&mut g, JokerId::Joker);
        }
        add_joker_ed(&mut g, JokerId::Joker, Edition::Negative);
        give(&mut g, id);
        use_c(&mut g, 0, 0).unwrap_or_else(|e| panic!("{id:?}: {e}"));
    }
}

#[test]
fn the_soul_queues_a_legendary_joker_and_changes_nothing_else() {
    let mut g = in_play(32);
    g.state_mut().money = 12;
    give(&mut g, C::TheSoul);
    let hand = g.state().deck.hand.clone();
    use_c(&mut g, 0, 0).expect("The Soul");
    assert_eq!(g.state().stats.effects_dropped, 1);
    assert_eq!(g.state().money, 12);
    assert_eq!(g.state().deck.hand, hand);
    assert!(g.state().consumables.is_empty());
    assert_eq!(g.state().consumable_uses(C::TheSoul), 1);
    invariants(&g);
}

// ----- Sigil / Ouija -------------------------------------------------------------------------------------

#[test]
fn sigil_gives_the_whole_hand_one_suit_and_skips_stone() {
    let mut seen = [false; 4];
    for seed in 0..40 {
        let mut g = in_play(400 + seed);
        g.state_mut().deck.hand[0].enhancement = Enhancement::Stone;
        let stone = g.state().deck.hand[0];
        let ranks: Vec<(u32, Rank)> = g
            .state()
            .deck
            .hand
            .iter()
            .map(|c| (c.uid, c.rank))
            .collect();
        give(&mut g, C::Sigil);
        use_c(&mut g, 0, 0).expect("Sigil");
        let mut suit = None;
        for c in &g.state().deck.hand {
            if c.uid == stone.uid {
                assert_eq!(
                    (c.suit, c.rank),
                    (stone.suit, stone.rank),
                    "Stone untouched"
                );
                continue;
            }
            assert_eq!(*suit.get_or_insert(c.suit), c.suit, "one suit");
            assert!(ranks.contains(&(c.uid, c.rank)), "ranks unchanged");
        }
        seen[suit.expect("a non-Stone card").index()] = true;
        assert_eq!(g.state().deck.hand.len(), 8);
        invariants(&g);
    }
    assert!(seen.iter().all(|&b| b), "all four suits appear: {seen:?}");
}

#[test]
fn ouija_gives_the_whole_hand_one_rank_and_costs_a_flat_hand_size() {
    let mut seen = [false; 15];
    for seed in 0..80 {
        let mut g = in_play(500 + seed);
        g.state_mut().deck.hand[7].enhancement = Enhancement::Stone;
        let stone = g.state().deck.hand[7];
        let suits: Vec<(u32, Suit)> = g
            .state()
            .deck
            .hand
            .iter()
            .map(|c| (c.uid, c.suit))
            .collect();
        give(&mut g, C::Ouija);
        use_c(&mut g, 0, 0).expect("Ouija");
        let mut rank = None;
        for c in &g.state().deck.hand {
            if c.uid == stone.uid {
                assert_eq!(c.rank, stone.rank, "Stone untouched");
                continue;
            }
            assert_eq!(*rank.get_or_insert(c.rank), c.rank, "one rank");
            assert!(suits.contains(&(c.uid, c.suit)), "suits unchanged");
        }
        seen[rank.expect("a non-Stone card") as usize] = true;
        assert_eq!(g.state().hand_size(), 7);
        invariants(&g);
    }
    assert!((2..=14).all(|r| seen[r]), "every rank appears: {seen:?}");
    // The penalty is flat: two uses -> -2.
    let mut g = in_play(501);
    give(&mut g, C::Ouija);
    give(&mut g, C::Ouija);
    use_c(&mut g, 0, 0).expect("first");
    use_c(&mut g, 0, 0).expect("second");
    assert_eq!(g.state().base_hand_size, 6);
}

// ----- Ectoplasm ---------------------------------------------------------------------------------------------

#[test]
fn ectoplasm_makes_an_editionless_joker_negative_and_the_penalty_grows() {
    let mut g = in_play(40);
    add_joker_ed(&mut g, JokerId::Joker, Edition::Foil);
    for j in [JokerId::Baron, JokerId::GreenJoker, JokerId::Egg] {
        add_joker(&mut g, j);
    }
    assert_eq!(g.state().base_hand_size, 8);
    let mut expect_size = 8;
    for (n, penalty) in [(1u32, 1u32), (2, 2), (3, 3)] {
        give(&mut g, C::Ectoplasm);
        use_c(&mut g, 0, 0).expect("Ectoplasm");
        expect_size -= penalty;
        assert_eq!(g.state().base_hand_size, expect_size, "use {n}");
        let negatives = g
            .state()
            .jokers
            .iter()
            .filter(|j| j.edition == Edition::Negative)
            .count();
        assert_eq!(negatives as u32, n);
        // The Foil joker is never chosen.
        assert_eq!(g.state().jokers[0].edition, Edition::Foil);
        invariants(&g);
    }
    assert_eq!(expect_size, 2, "8 - (1 + 2 + 3)");
    // No edition-less joker left.
    give(&mut g, C::Ectoplasm);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
}

#[test]
fn ectoplasm_needs_an_editionless_joker_and_picks_every_candidate_over_seeds() {
    let mut g = in_play(41);
    give(&mut g, C::Ectoplasm);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "no joker");
    let mut g = in_play(41);
    add_joker_ed(&mut g, JokerId::Joker, Edition::Polychrome);
    add_joker_ed(&mut g, JokerId::Baron, Edition::Negative);
    give(&mut g, C::Ectoplasm);
    assert_eq!(
        use_c(&mut g, 0, 0),
        Err(ActionError::CannotUse),
        "all have editions"
    );
    let mut hit = [false; 3];
    for seed in 0..40 {
        let mut g = in_play(600 + seed);
        for j in [JokerId::Joker, JokerId::Baron, JokerId::Egg] {
            add_joker(&mut g, j);
        }
        give(&mut g, C::Ectoplasm);
        use_c(&mut g, 0, 0).expect("Ectoplasm");
        let slot = g
            .state()
            .jokers
            .iter()
            .position(|j| j.edition == Edition::Negative)
            .expect("one Negative");
        hit[slot] = true;
    }
    assert!(hit.iter().all(|&b| b), "{hit:?}");
}

// ----- Immolate -------------------------------------------------------------------------------------------------

#[test]
fn immolate_destroys_five_distinct_cards_and_pays_twenty() {
    for seed in 0..10 {
        let mut g = in_play(700 + seed);
        let before = hand_uids(&g);
        g.state_mut().money = 3;
        give(&mut g, C::Immolate);
        use_c(&mut g, 0, 0).expect("Immolate");
        let s = g.state();
        let kept: Vec<u32> = s.deck.hand.iter().map(|c| c.uid).collect();
        assert_eq!(kept.len(), 3);
        assert!(kept.iter().all(|u| before.contains(u)));
        assert_eq!(s.stats.cards_destroyed, 5);
        assert_eq!(s.deck.len(), 47);
        assert_eq!(s.money, 23);
        invariants(&g);
    }
}

#[test]
fn immolate_with_a_short_hand_destroys_it_all_and_the_hand_refills() {
    let mut g = in_play(71);
    shrink_hand(&mut g, 3);
    give(&mut g, C::Immolate);
    use_c(&mut g, 0, 0).expect("Immolate");
    assert_eq!(g.state().stats.cards_destroyed, 3);
    assert_eq!(g.state().money, 4 + 20);
    assert_eq!(g.state().phase, Phase::Play);
    assert_eq!(g.state().deck.hand.len(), 8, "refilled from the draw pile");
    invariants(&g);
}

// ----- Hex ---------------------------------------------------------------------------------------------------------

#[test]
fn hex_polychromes_one_joker_and_destroys_the_others_except_eternal() {
    let mut both = [false; 2];
    for seed in 0..30 {
        let mut g = in_play(800 + seed);
        add_joker(&mut g, JokerId::Joker);
        let e = add_joker(&mut g, JokerId::GreenJoker);
        g.state_mut().jokers[e].stickers.eternal = true;
        add_joker(&mut g, JokerId::Baron);
        add_joker_ed(&mut g, JokerId::Egg, Edition::Foil); // not eligible, destroyed
        give(&mut g, C::Hex);
        use_c(&mut g, 0, 0).expect("Hex");
        let js = &g.state().jokers;
        assert!(
            js.iter().any(|j| j.id == JokerId::GreenJoker),
            "Eternal survives"
        );
        assert!(
            !js.iter().any(|j| j.id == JokerId::Egg),
            "the Foil joker is destroyed"
        );
        let poly: Vec<_> = js
            .iter()
            .filter(|j| j.edition == Edition::Polychrome)
            .collect();
        assert_eq!(poly.len(), 1);
        let eternal_is_poly = poly[0].id == JokerId::GreenJoker;
        assert_eq!(js.len(), if eternal_is_poly { 1 } else { 2 });
        both[usize::from(eternal_is_poly)] = true;
        invariants(&g);
    }
    assert!(both.iter().all(|&b| b), "both outcomes occur: {both:?}");
}

#[test]
fn hex_needs_an_editionless_joker() {
    let mut g = in_play(81);
    give(&mut g, C::Hex);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "no joker");
    add_joker_ed(&mut g, JokerId::Joker, Edition::Foil);
    assert_eq!(
        use_c(&mut g, 0, 0),
        Err(ActionError::CannotUse),
        "only an edition"
    );
    let mut g = in_play(81);
    add_joker(&mut g, JokerId::Joker);
    give(&mut g, C::Hex);
    use_c(&mut g, 0, 0).expect("one joker");
    assert_eq!(g.state().jokers.len(), 1);
    assert_eq!(g.state().jokers[0].edition, Edition::Polychrome);
}

#[test]
fn ankh_keeps_an_eternal_joker_and_hex_works_with_a_full_slot_row() {
    // Ankh is covered in tests/consumables.rs; here: Eternal survives when it is not the target
    // and everything else is destroyed (queue-before-copy), for many seeds.
    for seed in 0..15 {
        let mut g = in_play(900 + seed);
        let e = add_joker(&mut g, JokerId::Joker);
        g.state_mut().jokers[e].stickers.eternal = true;
        for _ in 0..3 {
            add_joker(&mut g, JokerId::Baron);
        }
        give(&mut g, C::Ankh);
        use_c(&mut g, 0, 0).expect("Ankh");
        assert!(g.state().jokers.iter().any(|j| j.stickers.eternal));
        invariants(&g);
    }
    // Hex with a full row of five.
    let mut g = in_play(950);
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    give(&mut g, C::Hex);
    use_c(&mut g, 0, 0).expect("Hex");
    assert_eq!(g.state().jokers.len(), 1);
}

// ----- Cryptid ----------------------------------------------------------------------------------------------------------

#[test]
fn cryptid_adds_two_exact_copies_of_the_target() {
    let mut g = in_play(60);
    {
        let c = &mut g.state_mut().deck.hand[4];
        c.rank = Rank::Seven;
        c.suit = Suit::Hearts;
        c.enhancement = Enhancement::Glass;
        c.edition = Edition::Foil;
        c.seal = Seal::Red;
        c.perma_bonus = 10;
    }
    let target = g.state().deck.hand[4];
    give(&mut g, C::Cryptid);
    use_c(&mut g, 0, 1 << 4).expect("Cryptid");
    let s = g.state();
    assert_eq!(s.deck.hand.len(), 10);
    let same: Vec<&Card> = s
        .deck
        .hand
        .iter()
        .filter(|c| {
            (
                c.rank,
                c.suit,
                c.enhancement,
                c.edition,
                c.seal,
                c.perma_bonus,
            ) == (
                Rank::Seven,
                Suit::Hearts,
                Enhancement::Glass,
                Edition::Foil,
                Seal::Red,
                10,
            )
        })
        .collect();
    assert_eq!(same.len(), 3, "the original and two copies");
    assert!(same.iter().any(|c| c.uid == target.uid), "original kept");
    let mut uids: Vec<u32> = same.iter().map(|c| c.uid).collect();
    uids.sort_unstable();
    uids.dedup();
    assert_eq!(uids.len(), 3, "copies have fresh uids");
    assert_eq!((s.stats.cards_added, s.stats.cards_destroyed), (2, 0));
    assert_eq!(s.deck.len(), 54);
    invariants(&g);
}

#[test]
fn cryptid_copies_stone_cards_too() {
    let mut g = in_play(61);
    g.state_mut().deck.hand[0].enhancement = Enhancement::Stone;
    give(&mut g, C::Cryptid);
    use_c(&mut g, 0, 1).expect("Cryptid");
    let stones = g
        .state()
        .deck
        .hand
        .iter()
        .filter(|c| c.enhancement == Enhancement::Stone)
        .count();
    assert_eq!(stones, 3);
    invariants(&g);
}

// ----- legal actions, robustness -------------------------------------------------------------------------------------------

#[test]
fn untargeted_spectrals_appear_exactly_once_in_legal_actions() {
    let mut g = in_play(70);
    add_joker(&mut g, JokerId::Joker);
    give(&mut g, C::Familiar);
    give(&mut g, C::Ectoplasm);
    let legal = g.legal_actions();
    for slot in 0..2u8 {
        let n = legal
            .iter()
            .filter(|a| matches!(a, Action::UseConsumable { slot: s, targets: 0 } if *s == slot))
            .count();
        assert_eq!(n, 1, "slot {slot}");
    }
    for slot in 0..2u8 {
        for targets in 0u16..256 {
            let a = Action::UseConsumable { slot, targets };
            assert_eq!(legal.contains(&a), g.validate(a).is_ok(), "{a:?}");
        }
    }
}

/// Random games where every blind starts with two random Spectrals and often some jokers.
fn spectral_game(seed: u64) -> Game {
    let mut g = new_game(seed);
    let mut agent = RandomAgent::with_play_weight(seed, 3.0);
    let mut pick = GameRng::new(seed ^ 0x5EC);
    let mut steps = 0;
    while !g.is_terminal() && steps < 400 {
        if g.state().phase == Phase::BlindSelect {
            for _ in 0..2 {
                let id = C::SPECTRALS[pick.range(RngKey::Misc, 0, C::SPECTRALS.len())];
                let _ = g.add_consumable(ConsumableInstance::new(id));
            }
            for _ in 0..pick.range(RngKey::Misc, 0, 4) {
                let j = JokerId::ALL[pick.range(RngKey::Misc, 0, JokerId::COUNT)];
                let _ = g.add_joker(JokerInstance::new(0, j));
            }
        }
        let a = agent.act(&g);
        g.step(a).expect("agent actions are legal");
        invariants(&g);
        steps += 1;
    }
    g
}

#[test]
fn random_games_full_of_spectrals_keep_the_invariants_and_are_deterministic() {
    for seed in 0..60 {
        let g = spectral_game(seed);
        if seed % 20 == 0 {
            assert_eq!(g.state(), spectral_game(seed).state(), "seed {seed}");
        }
    }
}
