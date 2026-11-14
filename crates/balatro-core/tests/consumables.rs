//! Phase 4.0 consumable contract: use/sell flow through `Game::step`, target validation, slots
//! (Negative), run counters, consumable side effects (creation pools, Perkeo, Gift Card, seals),
//! legal actions, registry audit, and the prototypes (Pluto, Mercury, The Fool, The Magician,
//! The Hermit, Death, Black Hole, Ankh). Guide: docs/design/consumables_bosses.md.

use balatro_core::actions::{self, Action, ActionError};
use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::card::Card;
use balatro_core::consumable::{
    ConsumableInstance, ConsumableKind, MAX_CONSUMABLE_TARGETS, fallback, in_random_pool,
    random_consumable,
};
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{
    BossId, ConsumableId as C, consumable_def_count, planets, spectrals, tarots, try_consumable_def,
};
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo, apply_side_effects};
use balatro_core::hooks::{SellTarget, SideEffect};
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

/// Replace hand card `pos` (keeps its uid).
fn set_card(g: &mut Game, pos: usize, rank: Rank, suit: Suit) {
    let c = &mut g.state_mut().deck.hand[pos];
    c.rank = rank;
    c.suit = suit;
    c.enhancement = Enhancement::None;
    c.edition = Edition::Base;
    c.seal = Seal::None;
    c.perma_bonus = 0;
}

// ----- flow and counters ---------------------------------------------------------------------

#[test]
fn planet_use_levels_the_hand_and_updates_the_run_counters() {
    let mut g = in_play(1);
    give(&mut g, C::Pluto);
    give(&mut g, C::Pluto);
    let info = use_c(&mut g, 0, 0).expect("use Pluto");
    assert_eq!(info.used, Some(C::Pluto));
    let s = g.state();
    assert_eq!(s.hand_levels[HandType::HighCard.index()], 2);
    assert_eq!(s.consumables.len(), 1);
    assert_eq!(
        (s.stats.run.planets_used, s.stats.run.unique_planets_used),
        (1, 1)
    );
    assert_eq!(s.stats.run.tarots_used, 0);
    assert_eq!(s.last_tarot_planet, Some(C::Pluto));
    assert_eq!(s.consumable_uses(C::Pluto), 1);
    assert_eq!(s.phase, Phase::Play);
    // The same Planet again: not a new unique Planet.
    use_c(&mut g, 0, 0).expect("use Pluto again");
    let s = g.state();
    assert_eq!(s.hand_levels[HandType::HighCard.index()], 3);
    assert_eq!(
        (s.stats.run.planets_used, s.stats.run.unique_planets_used),
        (2, 1)
    );
}

#[test]
fn planets_work_in_the_shop_and_level_their_own_hand() {
    let mut g = in_shop(2);
    give(&mut g, C::Mercury);
    use_c(&mut g, 0, 0).expect("Mercury in the shop");
    assert_eq!(g.state().hand_levels[HandType::Pair.index()], 2);
    assert_eq!(g.state().phase, Phase::Shop);
    // Every Planet levels exactly `planet_hand()`.
    for &id in C::PLANETS {
        let mut g = in_shop(3);
        give(&mut g, id);
        use_c(&mut g, 0, 0).expect("planet");
        let ht = id.planet_hand().expect("planet hand");
        for h in HandType::ALL {
            let want = if h == ht { 2 } else { 1 };
            assert_eq!(g.state().hand_levels[h.index()], want, "{id:?} {h:?}");
        }
    }
}

#[test]
fn leveled_hand_scores_with_the_new_base() {
    // Pluto: High Card 5x1 -> 15x2 (hands.md). One Ace: (15 + 11) x 2 = 52.
    let mut g = in_play(4);
    set_card(&mut g, 0, Rank::Ace, Suit::Spades);
    give(&mut g, C::Pluto);
    use_c(&mut g, 0, 0).expect("Pluto");
    let pos = g
        .state()
        .deck
        .hand
        .iter()
        .position(|c| c.rank == Rank::Ace && c.suit == Suit::Spades)
        .expect("ace");
    let h = g
        .step(Action::Play(1 << pos))
        .expect("play")
        .hand
        .expect("hand");
    assert_eq!((h.level, h.score), (2, 52.0));
}

#[test]
fn tarot_use_counts_for_fortune_teller_and_constellation_sees_planets() {
    let mut g = in_play(5);
    add_joker(&mut g, JokerId::Constellation);
    give(&mut g, C::TheHermit);
    give(&mut g, C::Pluto);
    use_c(&mut g, 0, 0).expect("Hermit");
    assert_eq!(g.state().stats.run.tarots_used, 1);
    assert_eq!(g.state().jokers[0].value, 1.0, "Hermit is not a Planet");
    use_c(&mut g, 0, 0).expect("Pluto");
    // Constellation: +0.1 per Planet used (fired after the Planet's own effect).
    assert!((g.state().jokers[0].value - 1.1).abs() < 1e-9);
}

// ----- validation ----------------------------------------------------------------------------

#[test]
fn target_validation_errors_leave_the_state_untouched() {
    let mut g = in_play(6);
    give(&mut g, C::TheMagician); // 1-2 targets
    give(&mut g, C::Pluto); // untargeted
    let cases = [
        (
            0,
            0,
            ActionError::TargetCount {
                selected: 0,
                min: 1,
                max: 2,
            },
        ),
        (
            0,
            0b111,
            ActionError::TargetCount {
                selected: 3,
                min: 1,
                max: 2,
            },
        ),
        (
            0,
            1 << 9,
            ActionError::SelectionOutOfRange {
                mask: 1 << 9,
                hand_len: 8,
            },
        ),
        (
            1,
            1,
            ActionError::TargetCount {
                selected: 1,
                min: 0,
                max: 0,
            },
        ),
        (2, 0, ActionError::InvalidConsumable { index: 2, len: 2 }),
    ];
    for (slot, targets, err) in cases {
        let before = g.state().clone();
        assert_eq!(use_c(&mut g, slot, targets), Err(err), "{slot} {targets:b}");
        assert_eq!(g.state(), &before);
    }
    assert_eq!(
        g.step(Action::SellConsumable(5)),
        Err(ActionError::InvalidConsumable { index: 5, len: 2 })
    );
}

#[test]
fn hand_targeting_cards_need_a_drawn_hand() {
    let mut g = in_shop(7);
    give(&mut g, C::TheMagician);
    give(&mut g, C::Death);
    assert_eq!(use_c(&mut g, 0, 1), Err(ActionError::NeedsHand));
    assert_eq!(use_c(&mut g, 1, 0b11), Err(ActionError::NeedsHand));
    // Not in BlindSelect at all.
    g.step(Action::EndShop).expect("end shop");
    assert!(matches!(
        use_c(&mut g, 0, 1),
        Err(ActionError::WrongPhase { .. })
    ));
}

#[test]
fn undefined_consumables_can_be_held_and_sold_but_not_used() {
    let Some(&id) = C::ALL.iter().find(|&&id| try_consumable_def(id).is_none()) else {
        return; // Phase 4 complete: every id is defined
    };
    let mut g = in_play(8);
    give(&mut g, id);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::UnknownConsumable));
    assert!(g.legal_actions().contains(&Action::SellConsumable(0)));
    assert!(
        !g.legal_actions()
            .iter()
            .any(|a| matches!(a, Action::UseConsumable { .. }))
    );
    g.step(Action::SellConsumable(0)).expect("sell");
}

// ----- selling and slots ---------------------------------------------------------------------

#[test]
fn selling_pays_the_sell_value_and_frees_the_slot() {
    let mut g = in_play(9);
    give(&mut g, C::TheHermit); // $3 -> $1
    give(&mut g, C::BlackHole); // Spectral $4 -> $2
    let m0 = g.state().money;
    let info = g.step(Action::SellConsumable(1)).expect("sell");
    assert_eq!(info.sold_for, Some(2));
    assert_eq!(g.state().money, m0 + 2);
    assert_eq!(g.state().consumables.len(), 1);
    assert_eq!(g.state().consumables[0].id, C::TheHermit);
    // In the shop too; Gift Card's bonus is part of the value.
    let mut g = in_shop(9);
    give(&mut g, C::Pluto);
    g.state_mut().consumables[0].sell_value += 3;
    let m0 = g.state().money;
    assert_eq!(
        g.step(Action::SellConsumable(0)).expect("sell").sold_for,
        Some(4)
    );
    assert_eq!(g.state().money, m0 + 4);
}

#[test]
fn full_slots_drop_creations_and_negative_cards_add_a_slot() {
    let mut g = in_play(10);
    give(&mut g, C::Pluto);
    give(&mut g, C::Mercury);
    assert!(
        g.add_consumable(ConsumableInstance::new(C::Uranus))
            .is_none()
    );
    let s = g.state_mut();
    let d0 = s.stats.effects_dropped;
    apply_side_effects(
        s,
        &[
            SideEffect::CreateConsumable {
                kind: ConsumableKind::Tarot,
                negative: false,
            },
            SideEffect::CreatePlanetFor {
                hand: HandType::Flush,
            },
        ],
    );
    assert_eq!(s.consumables.len(), 2);
    assert_eq!(s.stats.effects_dropped, d0 + 2);
    // A Negative card always fits and raises the capacity by one.
    apply_side_effects(
        s,
        &[SideEffect::CreateConsumable {
            kind: ConsumableKind::Tarot,
            negative: true,
        }],
    );
    assert_eq!(s.consumables.len(), 3);
    assert!(s.consumables[2].negative);
    assert_eq!(s.consumable_capacity(), 3);
    assert!(!s.has_consumable_room());
    assert!(s.check_invariants().is_ok());
}

#[test]
fn creation_side_effects_fill_free_slots() {
    let mut g = in_play(11);
    let s = g.state_mut();
    apply_side_effects(
        s,
        &[
            SideEffect::CreatePlanetFor {
                hand: HandType::FullHouse,
            },
            SideEffect::AddConsumable {
                id: C::Death,
                negative: false,
            },
        ],
    );
    let ids: Vec<C> = s.consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![C::Earth, C::Death]);
    // Perkeo-style copy: Negative, needs no slot.
    apply_side_effects(
        s,
        &[
            SideEffect::CopyConsumable {
                index: 1,
                negative: true,
            },
            SideEffect::AddSellValue {
                target: SellTarget::AllConsumables,
                amount: 2,
            },
        ],
    );
    assert_eq!(s.consumables.len(), 3);
    assert_eq!(s.consumables[2].id, C::Death);
    assert!(s.consumables[2].negative);
    assert!(s.consumables.iter().all(|c| c.sell_value == 3));
    assert_eq!(s.stats.effects_dropped, 0);
}

#[test]
fn perkeo_copies_a_held_consumable_as_negative_at_shop_exit() {
    let mut g = in_shop(12);
    add_joker(&mut g, JokerId::Perkeo);
    give(&mut g, C::TheHermit);
    g.step(Action::EndShop).expect("end shop");
    let s = g.state();
    assert_eq!(s.consumables.len(), 2);
    assert_eq!(s.consumables[1].id, C::TheHermit);
    assert!(s.consumables[1].negative);
}

#[test]
fn gift_card_raises_consumable_sell_values_at_round_end() {
    let mut g = in_play(13);
    add_joker(&mut g, JokerId::GiftCard);
    give(&mut g, C::Pluto);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().consumables[0].sell_value, 2);
    assert_eq!(g.state().stats.effects_dropped, 0);
}

#[test]
fn purple_seal_discard_creates_a_tarot_and_blue_seal_a_planet() {
    let mut g = in_play(14);
    g.state_mut().deck.hand[0].seal = Seal::Purple;
    g.step(Action::Discard(1)).expect("discard");
    assert_eq!(g.state().consumables.len(), 1);
    assert_eq!(g.state().consumables[0].kind(), ConsumableKind::Tarot);
    // Blue Seal held at round end: the Planet of the last hand played (High Card -> Pluto).
    let mut g = in_play(15);
    let n = g.state().deck.hand.len();
    g.state_mut().deck.hand[n - 1].seal = Seal::Blue;
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    let ids: Vec<C> = g.state().consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![C::Pluto]);
}

// ----- generation pool -----------------------------------------------------------------------

#[test]
fn random_pool_rules() {
    let plays0 = [0u32; HandType::COUNT];
    assert!(!in_random_pool(C::TheSoul, &plays0));
    assert!(!in_random_pool(C::BlackHole, &plays0));
    assert!(!in_random_pool(C::PlanetX, &plays0));
    let mut plays = plays0;
    plays[HandType::FiveOfAKind.index()] = 1;
    assert!(in_random_pool(C::PlanetX, &plays));
    assert!(in_random_pool(C::Pluto, &plays0));

    let mut rng = GameRng::new(1);
    let mut counts = [0u32; C::COUNT];
    for _ in 0..22_000 {
        let id = random_consumable(ConsumableKind::Tarot, &[], &plays0, false, &mut rng);
        assert_eq!(id.kind(), ConsumableKind::Tarot);
        counts[id.index()] += 1;
    }
    // Uniform over 22 Tarots: each about 1000.
    for &t in C::TAROTS {
        let n = counts[t.index()];
        assert!((850..1150).contains(&n), "{t:?}: {n}");
    }
    for _ in 0..2_000 {
        let p = random_consumable(ConsumableKind::Planet, &[], &plays0, false, &mut rng);
        assert!(p.index() < C::PlanetX.index(), "no secret planet: {p:?}");
        let s = random_consumable(ConsumableKind::Spectral, &[], &plays0, false, &mut rng);
        assert!(!matches!(s, C::TheSoul | C::BlackHole));
    }
    // Held cards are excluded unless duplicates are allowed (Showman); fallback when empty.
    let held: Vec<ConsumableInstance> = C::PLANETS[..9]
        .iter()
        .map(|&id| ConsumableInstance::new(id))
        .collect();
    assert_eq!(
        random_consumable(ConsumableKind::Planet, &held, &plays0, false, &mut rng),
        fallback(ConsumableKind::Planet)
    );
    let held1 = [ConsumableInstance::new(C::Pluto)];
    for _ in 0..200 {
        assert_ne!(
            random_consumable(ConsumableKind::Planet, &held1, &plays0, false, &mut rng),
            C::Pluto
        );
    }
    assert!((0..400).any(|_| {
        random_consumable(ConsumableKind::Planet, &held1, &plays0, true, &mut rng) == C::Pluto
    }));
}

// ----- prototypes ----------------------------------------------------------------------------

#[test]
fn the_magician_enhances_up_to_two_targets_to_lucky() {
    let mut g = in_play(20);
    give(&mut g, C::TheMagician);
    let uids: Vec<u32> = g.state().deck.hand.iter().map(|c| c.uid).collect();
    use_c(&mut g, 0, 0b101).expect("Magician");
    let hand = &g.state().deck.hand;
    for c in hand {
        let targeted = c.uid == uids[0] || c.uid == uids[2];
        let want = if targeted {
            Enhancement::Lucky
        } else {
            Enhancement::None
        };
        assert_eq!(c.enhancement, want, "uid {}", c.uid);
    }
    assert_eq!(g.state().stats.run.tarots_used, 1);
    assert_eq!(g.state().last_tarot_planet, Some(C::TheMagician));
}

#[test]
fn the_hermit_doubles_money_up_to_twenty() {
    for (money, want) in [(4, 8), (30, 50), (0, 0), (-3, -3), (20, 40)] {
        let mut g = in_play(21);
        g.state_mut().money = money;
        give(&mut g, C::TheHermit);
        use_c(&mut g, 0, 0).expect("Hermit");
        assert_eq!(g.state().money, want, "from {money}");
    }
}

#[test]
fn death_turns_the_left_card_into_a_copy_of_the_right_one() {
    let mut g = in_play(22);
    give(&mut g, C::Death);
    set_card(&mut g, 0, Rank::King, Suit::Hearts);
    set_card(&mut g, 3, Rank::Two, Suit::Clubs);
    {
        let r = &mut g.state_mut().deck.hand[3];
        r.enhancement = Enhancement::Glass;
        r.edition = Edition::Polychrome;
        r.seal = Seal::Red;
        r.perma_bonus = 7;
    }
    let (left_uid, right) = {
        let h = &g.state().deck.hand;
        (h[0].uid, h[3])
    };
    use_c(&mut g, 0, 0b1001).expect("Death");
    let h = &g.state().deck.hand;
    let left = h
        .iter()
        .find(|c| c.uid == left_uid)
        .expect("left kept its uid");
    assert_eq!(
        (
            left.rank,
            left.suit,
            left.enhancement,
            left.edition,
            left.seal,
            left.perma_bonus
        ),
        (
            Rank::Two,
            Suit::Clubs,
            Enhancement::Glass,
            Edition::Polychrome,
            Seal::Red,
            7
        )
    );
    assert_eq!(h.iter().find(|c| c.uid == right.uid), Some(&right));
    assert!(g.state().check_invariants().is_ok());
    // Exactly two targets.
    give(&mut g, C::Death);
    assert!(matches!(
        use_c(&mut g, 0, 1),
        Err(ActionError::TargetCount { selected: 1, .. })
    ));
}

#[test]
fn the_fool_copies_the_last_tarot_or_planet() {
    let mut g = in_play(23);
    give(&mut g, C::TheFool);
    // Nothing used yet.
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
    give(&mut g, C::Mercury);
    use_c(&mut g, 1, 0).expect("Mercury");
    // Spectrals never count as "last used".
    give(&mut g, C::BlackHole);
    use_c(&mut g, 1, 0).expect("Black Hole");
    assert_eq!(g.state().last_tarot_planet, Some(C::Mercury));
    // Full area (Fool + Hermit): the Fool's own slot frees up.
    give(&mut g, C::TheHermit);
    assert!(!g.state().has_consumable_room());
    use_c(&mut g, 0, 0).expect("Fool");
    let ids: Vec<C> = g.state().consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![C::TheHermit, C::Mercury]);
    // The Fool itself is never recorded.
    assert_eq!(g.state().last_tarot_planet, Some(C::Mercury));
    assert_eq!(g.state().stats.run.tarots_used, 1);
}

#[test]
fn the_fool_needs_a_free_slot_when_used_from_a_negative_copy() {
    // Two normal cards + a Negative Fool: capacity 3, and using the Fool removes its bonus slot.
    let mut g = in_play(24);
    g.state_mut().last_tarot_planet = Some(C::Pluto);
    give(&mut g, C::Pluto);
    give(&mut g, C::Mercury);
    g.add_consumable(ConsumableInstance::new(C::TheFool).with_negative(true))
        .expect("negative fits");
    assert_eq!(use_c(&mut g, 2, 0), Err(ActionError::CannotUse));
}

#[test]
fn black_hole_levels_every_hand() {
    let mut g = in_shop(25);
    give(&mut g, C::BlackHole);
    use_c(&mut g, 0, 0).expect("Black Hole");
    let s = g.state();
    assert!(s.hand_levels.iter().all(|&l| l == 2));
    assert_eq!(s.stats.run.planets_used, 0, "a Spectral, not a Planet");
    assert_eq!(s.last_tarot_planet, None);
}

#[test]
fn ankh_copies_one_joker_and_destroys_the_other_non_eternal_ones() {
    for seed in 0..12 {
        let mut g = in_play(30 + seed);
        add_joker(&mut g, JokerId::Joker);
        let e = add_joker(&mut g, JokerId::GreenJoker);
        g.state_mut().jokers[e].stickers.eternal = true;
        add_joker(&mut g, JokerId::Baron);
        give(&mut g, C::Ankh);
        use_c(&mut g, 0, 0).expect("Ankh");
        let js = &g.state().jokers;
        let eternal_kept = js.iter().any(|j| j.id == JokerId::GreenJoker);
        assert!(eternal_kept, "Eternal survives");
        // Kept: the copy target + its copy (+ the Eternal one if it was not the target).
        let ids: Vec<JokerId> = js.iter().map(|j| j.id).collect();
        let target = *ids.last().expect("copy");
        assert_eq!(ids.iter().filter(|&&i| i == target).count(), 2, "{ids:?}");
        let want_len = if target == JokerId::GreenJoker { 2 } else { 3 };
        assert_eq!(js.len(), want_len, "{ids:?}");
        assert!(g.state().check_invariants().is_ok());
    }
}

#[test]
fn ankh_needs_a_joker_and_a_free_slot_and_strips_negative() {
    let mut g = in_play(40);
    give(&mut g, C::Ankh);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "no joker");
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse), "no room");
    // One Negative joker: the copy is not Negative.
    let mut g = in_play(41);
    give(&mut g, C::Ankh);
    g.add_joker(JokerInstance::new(0, JokerId::Joker).with_edition(Edition::Negative))
        .expect("negative");
    use_c(&mut g, 0, 0).expect("Ankh");
    let eds: Vec<Edition> = g.state().jokers.iter().map(|j| j.edition).collect();
    assert_eq!(eds, vec![Edition::Negative, Edition::Base]);
}

// ----- legal actions -------------------------------------------------------------------------

#[test]
fn legal_actions_list_every_valid_use_and_sale() {
    let mut g = in_play(50);
    give(&mut g, C::TheMagician);
    give(&mut g, C::TheFool); // not usable yet
    let legal = g.legal_actions();
    let uses: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::UseConsumable { .. }))
        .collect();
    // Magician: C(8,1) + C(8,2) = 36 selections; the Fool has nothing to copy.
    assert_eq!(uses.len(), 36);
    assert!(uses.iter().all(|&a| g.validate(a).is_ok()));
    assert!(legal.contains(&Action::SellConsumable(0)));
    assert!(legal.contains(&Action::SellConsumable(1)));
    // Every candidate use is legal iff it validates.
    for slot in 0..3u8 {
        for targets in 0u16..256 {
            let a = Action::UseConsumable { slot, targets };
            assert_eq!(legal.contains(&a), g.validate(a).is_ok(), "{a:?}");
        }
    }
    // Shop: no hand, so only the untargeted card would be usable; both are sellable.
    let mut g = in_shop(50);
    give(&mut g, C::TheMagician);
    give(&mut g, C::Pluto);
    let legal = g.legal_actions();
    assert!(legal.contains(&Action::UseConsumable {
        slot: 1,
        targets: 0
    }));
    assert!(
        !legal
            .iter()
            .any(|a| matches!(a, Action::UseConsumable { slot: 0, .. }))
    );
    assert!(legal.contains(&Action::SellConsumable(0)));
}

// ----- registry audit ------------------------------------------------------------------------

#[test]
fn consumable_registry_is_consistent() {
    for (table, kind) in [
        (planets::DEFS, ConsumableKind::Planet),
        (tarots::DEFS, ConsumableKind::Tarot),
        (spectrals::DEFS, ConsumableKind::Spectral),
    ] {
        for d in table {
            assert_eq!(d.id.kind(), kind, "{} is in the wrong file", d.name);
            assert_eq!(consumable_def_count(d.id), 1, "{} defined twice", d.name);
            assert!(d.max_targets <= MAX_CONSUMABLE_TARGETS, "{}", d.name);
            assert!(d.min_targets <= d.max_targets, "{}", d.name);
            if d.max_targets > 0 {
                assert!(d.min_targets >= 1, "{}: targeted cards need >= 1", d.name);
            }
            assert_eq!(try_consumable_def(d.id).expect("registered").name, d.name);
        }
    }
    assert_eq!(C::COUNT, 52);
    assert_eq!(
        (C::PLANETS.len(), C::TAROTS.len(), C::SPECTRALS.len()),
        (12, 22, 18)
    );
    for (i, &h) in HandType::ALL.iter().enumerate() {
        assert_eq!(C::planet_for(h).planet_hand(), Some(h));
        assert_eq!(C::PLANETS[i].index(), i);
    }
    assert_eq!(C::TheFool.kind(), ConsumableKind::Tarot);
    assert_eq!(C::TheWorld.kind(), ConsumableKind::Tarot);
    assert_eq!(C::Familiar.kind(), ConsumableKind::Spectral);
    assert_eq!(C::BlackHole.planet_hand(), None);
}

// ----- robustness, determinism, replay ------------------------------------------------------

/// Random games where every round starts with two random defined consumables.
fn random_consumable_game(seed: u64) -> Game {
    let defined: Vec<C> = C::ALL
        .into_iter()
        .filter(|&id| try_consumable_def(id).is_some())
        .collect();
    let mut g = new_game(seed);
    let mut agent = RandomAgent::with_play_weight(seed, 3.0);
    let mut pick = GameRng::new(seed ^ 0xC0);
    let mut steps = 0;
    while !g.is_terminal() && steps < 400 {
        if g.state().phase == Phase::BlindSelect {
            for _ in 0..2 {
                let id = defined[pick.range(RngKey::Misc, 0, defined.len())];
                let neg = pick.range(RngKey::Misc, 0, 4) == 0;
                let _ = g.add_consumable(ConsumableInstance::new(id).with_negative(neg));
            }
            if pick.range(RngKey::Misc, 0, 3) == 0 {
                let j = JokerId::ALL[pick.range(RngKey::Misc, 0, JokerId::COUNT)];
                let _ = g.add_joker(JokerInstance::new(0, j));
            }
        }
        let a = agent.act(&g);
        g.step(a).expect("agent actions are legal");
        assert!(g.state().check_invariants().is_ok());
        steps += 1;
    }
    g
}

#[test]
fn random_games_with_consumables_keep_the_invariants() {
    let mut used = 0;
    for seed in 0..150 {
        let g = random_consumable_game(seed);
        used += g.state().consumable_use_counts.iter().sum::<u32>();
    }
    // The random agent rarely picks a use among ~450 actions, but some happen.
    assert!(used >= 10, "consumables were used: {used}");
}

#[test]
fn consumable_games_are_deterministic_and_replayable() {
    for seed in [3u64, 77] {
        let a = random_consumable_game(seed);
        let b = random_consumable_game(seed);
        assert_eq!(a.state(), b.state());
    }
    // A recorded action list replays to the same state.
    let mut g = in_play(60);
    let mut actions = vec![Action::SelectBlind];
    give(&mut g, C::TheMagician);
    give(&mut g, C::Pluto);
    for a in [
        Action::UseConsumable {
            slot: 0,
            targets: 0b11,
        },
        Action::UseConsumable {
            slot: 0,
            targets: 0,
        },
        Action::Play(0b11),
    ] {
        g.step(a).expect("legal");
        actions.push(a);
    }
    let mut h = new_game(60);
    h.step(actions[0]).expect("select");
    give(&mut h, C::TheMagician);
    give(&mut h, C::Pluto);
    for &a in &actions[1..] {
        h.step(a).expect("replay");
    }
    assert_eq!(g.state(), h.state());
    let _ = actions::validate(h.state(), Action::EndShop);
}

#[test]
fn boss_view_ids_are_typed() {
    // Every run has a typed boss from ante 1 on (the registry has bosses).
    let g = new_game(1);
    let b = g.state().boss.expect("a boss is chosen at ante start");
    assert!(!b.is_showdown());
    assert_eq!(g.state().run_view().boss.id, Some(b));
    let _: BossId = b;
}

#[test]
fn card_struct_stays_small() {
    assert!(std::mem::size_of::<Card>() <= 20);
    assert!(std::mem::size_of::<ConsumableInstance>() <= 8);
}
