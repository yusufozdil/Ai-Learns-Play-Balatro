//! Phase 5.0 booster packs: all 15 packs (contents, price, picks), opening/picking/skipping
//! through `Game::step`, the dealt hand of Arcana/Spectral packs, pack events (Red Card,
//! Hallucination), Telescope / Omen Globe / Showman, pack tags and invariants.
//! Guide: docs/design/shop_packs_vouchers_tags.md.

use balatro_core::actions::{Action, ActionError};
use balatro_core::consumable::{ConsumableInstance, ConsumableKind};
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{ConsumableId as C, TagId};
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rarity, Seal};
use balatro_core::game::Game;
use balatro_core::hooks::PackKind;
use balatro_core::joker::JokerInstance;
use balatro_core::packs::{self, OpenPack, PackItem, PackOrigin, PackSize, PackSpec};
use balatro_core::shop::ShopPack;
use balatro_core::state::{GameConfig, GameState};

// ----- helpers ---------------------------------------------------------------------------------

fn in_shop(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().phase, Phase::Shop);
    g.state_mut().money = 100;
    g
}

fn spec(kind: PackKind, size: PackSize) -> PackSpec {
    PackSpec { kind, size }
}

/// Put `spec` into pack slot 1 and open it.
fn open(g: &mut Game, spec: PackSpec) {
    g.state_mut().shop.packs[1] = Some(ShopPack { spec, free: false });
    g.step(Action::OpenPack(1)).expect("open pack");
    assert_eq!(g.state().phase, Phase::PackOpen);
}

fn pack(g: &Game) -> &OpenPack {
    g.state().pack.as_ref().expect("open pack")
}

fn set_items(g: &mut Game, items: Vec<PackItem>) {
    let p = g.state_mut().pack.as_mut().expect("open pack");
    p.items = items.into_iter().map(Some).collect();
}

fn pick(index: u8, targets: u32) -> Action {
    Action::PickPackItem { index, targets }
}

fn add_joker(g: &mut Game, id: JokerId) {
    g.add_joker(JokerInstance::new(0, id)).expect("room");
}

fn joker(id: JokerId) -> PackItem {
    PackItem::Joker(JokerInstance::new(0, id))
}

// ----- tables ----------------------------------------------------------------------------------

#[test]
fn all_fifteen_packs_have_the_documented_size_picks_and_price() {
    let cases = [
        (PackKind::Arcana, [3, 5, 5]),
        (PackKind::Celestial, [3, 5, 5]),
        (PackKind::Standard, [3, 5, 5]),
        (PackKind::Buffoon, [2, 4, 4]),
        (PackKind::Spectral, [2, 4, 4]),
    ];
    let mut indices = Vec::new();
    for (kind, cards) in cases {
        for (size, n) in PackSize::ALL.into_iter().zip(cards) {
            let sp = spec(kind, size);
            assert_eq!(sp.cards(), n, "{kind:?} {size:?}");
            assert_eq!(sp.picks(), if size == PackSize::Mega { 2 } else { 1 });
            indices.push(sp.index());
        }
    }
    indices.sort_unstable();
    assert_eq!(indices, (0..15).collect::<Vec<_>>());
}

#[test]
fn every_pack_opens_pays_its_price_and_shows_the_right_number_of_items() {
    for kind in PackKind::ALL {
        for (size, price) in PackSize::ALL.into_iter().zip([4, 6, 8]) {
            let mut g = in_shop(7);
            let before = g.state().money;
            g.state_mut().shop.packs[1] = Some(ShopPack {
                spec: spec(kind, size),
                free: false,
            });
            let info = g.step(Action::OpenPack(1)).expect("open");
            assert_eq!(info.spent, Some(price), "{kind:?} {size:?}");
            assert_eq!(g.state().money, before - price);
            let p = pack(&g);
            assert_eq!(
                p.items.iter().flatten().count(),
                usize::from(spec(kind, size).cards())
            );
            assert_eq!(p.picks_left, spec(kind, size).picks());
            assert_eq!(p.origin, PackOrigin::Shop);
            assert!(g.state().shop.packs[1].is_none());
            g.state().check_invariants().expect("invariants");
            // The kinds of the items match the pack.
            for item in p.items.iter().flatten() {
                match (kind, item) {
                    (PackKind::Arcana, PackItem::Consumable(id)) => {
                        assert!(matches!(id.kind(), ConsumableKind::Tarot) || *id == C::TheSoul);
                    }
                    (PackKind::Celestial, PackItem::Consumable(id)) => {
                        assert!(id.kind() == ConsumableKind::Planet);
                    }
                    (PackKind::Spectral, PackItem::Consumable(id)) => {
                        assert!(id.kind() == ConsumableKind::Spectral);
                    }
                    (PackKind::Standard, PackItem::Card(_))
                    | (PackKind::Buffoon, PackItem::Joker(_)) => {}
                    other => panic!("{kind:?} holds {other:?}"),
                }
            }
        }
    }
}

#[test]
fn only_arcana_and_spectral_packs_deal_a_hand() {
    for kind in PackKind::ALL {
        let mut g = in_shop(8);
        open(&mut g, spec(kind, PackSize::Normal));
        let hand = g.state().deck.hand.len();
        if matches!(kind, PackKind::Arcana | PackKind::Spectral) {
            assert_eq!(hand, 8, "{kind:?}");
        } else {
            assert_eq!(hand, 0, "{kind:?}");
        }
        // The deck is whole: hand + draw pile.
        assert_eq!(g.state().deck.len(), 52);
    }
}

// ----- picking ---------------------------------------------------------------------------------

#[test]
fn a_normal_pack_closes_after_one_pick_and_returns_to_the_shop() {
    let mut g = in_shop(9);
    open(&mut g, spec(PackKind::Arcana, PackSize::Normal));
    set_items(
        &mut g,
        vec![
            PackItem::Consumable(C::TheMagician),
            PackItem::Consumable(C::TheHermit),
        ],
    );
    let uid = g.state().deck.hand[2].uid;
    g.step(pick(0, 0b100))
        .expect("pick the Magician on hand card 2");
    assert_eq!(g.state().phase, Phase::Shop);
    assert!(g.state().pack.is_none());
    assert!(g.state().deck.hand.is_empty());
    let c = g
        .state()
        .deck
        .iter_all()
        .find(|c| c.uid == uid)
        .expect("card");
    assert_eq!(c.enhancement, Enhancement::Lucky, "the change is permanent");
    assert_eq!(g.state().stats.run.tarots_used, 1);
    assert!(
        g.state().consumables.is_empty(),
        "picked cards never take a slot"
    );
    assert_eq!(g.state().deck.len(), 52);
    g.state().check_invariants().expect("invariants");
}

#[test]
fn a_mega_pack_allows_two_picks() {
    let mut g = in_shop(10);
    open(&mut g, spec(PackKind::Celestial, PackSize::Mega));
    set_items(
        &mut g,
        vec![
            PackItem::Consumable(C::Mercury),
            PackItem::Consumable(C::Venus),
            PackItem::Consumable(C::Earth),
        ],
    );
    g.step(pick(0, 0)).expect("first pick");
    assert_eq!(g.state().phase, Phase::PackOpen);
    assert_eq!(pack(&g).picks_left, 1);
    assert!(pack(&g).items[0].is_none(), "positions do not shift");
    assert_eq!(
        g.step(pick(0, 0)),
        Err(ActionError::InvalidPackItem { index: 0 })
    );
    g.step(pick(2, 0)).expect("second pick");
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().hand_levels[HandType::Pair.index()], 2);
    assert_eq!(g.state().hand_levels[HandType::FullHouse.index()], 2);
    assert_eq!(g.state().hand_levels[HandType::ThreeOfAKind.index()], 1);
    assert_eq!(g.state().stats.run.planets_used, 2);
}

#[test]
fn targeted_tarots_need_the_right_number_of_hand_cards() {
    let mut g = in_shop(11);
    open(&mut g, spec(PackKind::Arcana, PackSize::Normal));
    set_items(&mut g, vec![PackItem::Consumable(C::TheLovers)]); // exactly 1 target
    assert!(matches!(
        g.step(pick(0, 0)),
        Err(ActionError::TargetCount { selected: 0, .. })
    ));
    assert!(matches!(
        g.step(pick(0, 0b11)),
        Err(ActionError::TargetCount { selected: 2, .. })
    ));
    assert!(matches!(
        g.step(pick(0, 1 << 10)),
        Err(ActionError::SelectionOutOfRange { .. })
    ));
    // Nothing happened.
    assert_eq!(g.state().phase, Phase::PackOpen);
    assert_eq!(pack(&g).picks_left, 1);
    // Legal picks: one per hand card.
    let picks: Vec<Action> = g
        .legal_actions()
        .into_iter()
        .filter(|a| matches!(a, Action::PickPackItem { .. }))
        .collect();
    assert_eq!(picks.len(), 8);
    assert!(g.legal_actions().contains(&Action::SkipPack));
    g.step(picks[3]).expect("legal pick");
    assert_eq!(g.state().phase, Phase::Shop);
}

#[test]
fn spectral_pack_cards_seal_and_create_jokers() {
    let mut g = in_shop(12);
    open(&mut g, spec(PackKind::Spectral, PackSize::Normal));
    set_items(
        &mut g,
        vec![
            PackItem::Consumable(C::Talisman),
            PackItem::Consumable(C::Wraith),
        ],
    );
    let uid = g.state().deck.hand[0].uid;
    g.state_mut().money = 33;
    // Wraith needs no hand cards and a free joker slot: legal, but only one pick is allowed.
    g.step(pick(0, 0b1)).expect("Talisman");
    let c = g
        .state()
        .deck
        .iter_all()
        .find(|c| c.uid == uid)
        .expect("card");
    assert_eq!(c.seal, Seal::Gold);
    assert_eq!(g.state().phase, Phase::Shop);

    let mut g = in_shop(12);
    open(&mut g, spec(PackKind::Spectral, PackSize::Mega));
    set_items(
        &mut g,
        vec![
            PackItem::Consumable(C::Wraith),
            PackItem::Consumable(C::TheSoul),
        ],
    );
    g.step(pick(0, 0)).expect("Wraith");
    assert_eq!(g.state().money, 0);
    assert_eq!(g.state().jokers[0].def().expect("def").rarity, Rarity::Rare);
    g.step(pick(1, 0)).expect("The Soul");
    assert_eq!(
        g.state().jokers[1].def().expect("def").rarity,
        Rarity::Legendary
    );
    assert_eq!(g.state().effects_no_room_total(), 0);
}

trait Counters {
    fn effects_no_room_total(&self) -> u32;
}
impl Counters for GameState {
    fn effects_no_room_total(&self) -> u32 {
        self.stats.effects_no_room + self.stats.effects_unimplemented
    }
}

#[test]
fn the_soul_needs_a_free_joker_slot_to_be_picked() {
    let mut g = in_shop(13);
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    open(&mut g, spec(PackKind::Arcana, PackSize::Normal));
    set_items(&mut g, vec![PackItem::Consumable(C::TheSoul)]);
    assert_eq!(g.step(pick(0, 0)), Err(ActionError::CannotUse));
    assert_eq!(
        g.legal_actions()
            .iter()
            .filter(|a| matches!(a, Action::PickPackItem { .. }))
            .count(),
        0
    );
    // Sell a joker while the pack is open, then it works.
    g.step(Action::SellJoker(0)).expect("sell during a pack");
    g.step(pick(0, 0)).expect("now it fits");
    assert_eq!(g.state().jokers.len(), 5);
}

#[test]
fn a_standard_card_is_shuffled_into_the_deck() {
    let mut g = in_shop(14);
    open(&mut g, spec(PackKind::Standard, PackSize::Jumbo));
    let PackItem::Card(card) = pack(&g).items[2].expect("item") else {
        panic!("standard packs hold cards");
    };
    g.step(pick(2, 0)).expect("pick");
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().deck.len(), 53);
    assert_eq!(g.state().stats.cards_added, 1);
    let new = g
        .state()
        .deck
        .iter_all()
        .find(|c| c.uid == 52)
        .expect("new card");
    assert_eq!(
        (new.rank, new.suit, new.enhancement, new.edition, new.seal),
        (
            card.rank,
            card.suit,
            card.enhancement,
            card.edition,
            card.seal
        )
    );
    g.state().check_invariants().expect("invariants");
}

#[test]
fn a_buffoon_joker_needs_room_unless_negative() {
    let mut g = in_shop(15);
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    open(&mut g, spec(PackKind::Buffoon, PackSize::Mega));
    set_items(
        &mut g,
        vec![
            joker(JokerId::GreedyJoker),
            PackItem::Joker(
                JokerInstance::new(0, JokerId::LustyJoker).with_edition(Edition::Negative),
            ),
        ],
    );
    assert_eq!(g.step(pick(0, 0)), Err(ActionError::NoJokerRoom));
    g.step(pick(1, 0)).expect("negative fits");
    assert_eq!(g.state().jokers.len(), 6);
    assert_eq!(g.state().phase, Phase::PackOpen, "Mega: one pick left");
    // The pack closes when nothing is left to pick? One item is left, so it stays open.
    g.step(Action::SellJoker(0)).expect("sell");
    g.step(Action::SellJoker(0)).expect("sell");
    g.step(pick(0, 0)).expect("room now");
    assert_eq!(g.state().phase, Phase::Shop);
    g.state().check_invariants().expect("invariants");
}

// ----- skipping and events -------------------------------------------------------------------------

#[test]
fn skipping_returns_to_the_shop_and_fires_one_skip_event_per_pack() {
    let mut g = in_shop(16);
    add_joker(&mut g, JokerId::RedCard);
    open(&mut g, spec(PackKind::Standard, PackSize::Normal));
    g.step(Action::SkipPack).expect("skip");
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().jokers[0].value, 3.0);
    // A Mega pack skipped after one pick counts once more.
    open(&mut g, spec(PackKind::Celestial, PackSize::Mega));
    g.step(pick(0, 0)).expect("one pick");
    g.step(Action::SkipPack).expect("skip the rest");
    assert_eq!(g.state().jokers[0].value, 6.0);
    // Finishing the picks is not a skip.
    open(&mut g, spec(PackKind::Celestial, PackSize::Normal));
    g.step(pick(0, 0)).expect("pick");
    assert_eq!(g.state().jokers[0].value, 6.0);
    g.state().check_invariants().expect("invariants");
}

#[test]
fn skipping_gives_the_hand_back_and_the_deck_is_intact() {
    let mut g = in_shop(17);
    open(&mut g, spec(PackKind::Arcana, PackSize::Jumbo));
    assert_eq!(g.state().deck.hand.len(), 8);
    g.step(Action::SkipPack).expect("skip");
    assert!(g.state().deck.hand.is_empty());
    assert_eq!(g.state().deck.draw_pile.len(), 52);
    g.state().check_invariants().expect("invariants");
    // Every pack kind can be skipped from the middle of a run.
    for kind in PackKind::ALL {
        let mut g = in_shop(18);
        open(&mut g, spec(kind, PackSize::Normal));
        g.step(Action::SkipPack).expect("skip");
        assert_eq!(g.state().phase, Phase::Shop);
    }
}

#[test]
fn hallucination_may_create_a_tarot_when_any_pack_opens() {
    let mut created = 0;
    let mut none = 0;
    for seed in 0..60 {
        let mut g = in_shop(seed);
        add_joker(&mut g, JokerId::Hallucination);
        open(&mut g, spec(PackKind::Standard, PackSize::Normal));
        match g.state().consumables.len() {
            0 => none += 1,
            1 => created += 1,
            n => panic!("{n} tarots"),
        }
        assert_eq!(g.state().stats.effects_unimplemented, 0);
    }
    assert!(created > 10 && none > 10, "{created} / {none}");
}

#[test]
fn consumables_in_slots_can_be_used_on_the_dealt_hand_of_an_open_pack() {
    let mut g = in_shop(19);
    g.add_consumable(ConsumableInstance::new(C::TheEmpress))
        .expect("room");
    // No hand yet in a Celestial pack: the targeted card cannot be used.
    open(&mut g, spec(PackKind::Celestial, PackSize::Normal));
    assert_eq!(
        g.step(Action::UseConsumable {
            slot: 0,
            targets: 1
        }),
        Err(ActionError::NeedsHand)
    );
    g.step(Action::SkipPack).expect("skip");
    open(&mut g, spec(PackKind::Arcana, PackSize::Normal));
    let uid = g.state().deck.hand[0].uid;
    g.step(Action::UseConsumable {
        slot: 0,
        targets: 1,
    })
    .expect("use on the dealt hand");
    assert_eq!(g.state().phase, Phase::PackOpen);
    assert_eq!(
        g.state()
            .deck
            .hand
            .iter()
            .find(|c| c.uid == uid)
            .expect("card")
            .enhancement,
        Enhancement::Mult
    );
}

// ----- generation rules ---------------------------------------------------------------------------

fn packs_state(seed: u64) -> GameState {
    let mut s = GameState::new(seed, GameConfig::default());
    s.phase = Phase::Shop;
    s
}

#[test]
fn no_duplicates_inside_a_pack_unless_showman() {
    let mut s = packs_state(21);
    for kind in [
        PackKind::Arcana,
        PackKind::Celestial,
        PackKind::Spectral,
        PackKind::Buffoon,
    ] {
        for _ in 0..300 {
            let items = packs::generate_items(&mut s, spec(kind, PackSize::Jumbo));
            let mut ids: Vec<String> = items
                .iter()
                .flatten()
                .map(|i| match i {
                    PackItem::Consumable(id) => format!("{id:?}"),
                    PackItem::Joker(j) => format!("{:?}", j.id),
                    PackItem::Card(_) => unreachable!(),
                })
                .collect();
            let n = ids.len();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), n, "{kind:?}");
        }
    }
    // With Showman duplicates are possible (tiny pools make it certain).
    let mut j = JokerInstance::new(0, JokerId::Showman);
    j.uid = s.alloc_joker_uid();
    s.jokers.push(j);
    s.refresh_modifiers();
    let dup = (0..400).any(|_| {
        let items = packs::generate_items(&mut s, spec(PackKind::Celestial, PackSize::Mega));
        let ids: Vec<C> = items
            .iter()
            .flatten()
            .filter_map(|i| {
                if let PackItem::Consumable(id) = i {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        (0..ids.len()).any(|i| ids[..i].contains(&ids[i]))
    });
    assert!(dup);
}

#[test]
fn telescope_puts_the_most_played_hands_planet_first() {
    let mut s = packs_state(22);
    s.shop_mods.telescope = true;
    s.hand_plays_run[HandType::Flush.index()] = 4;
    s.hand_plays_run[HandType::Pair.index()] = 9;
    s.hand_plays_run[HandType::TwoPair.index()] = 9; // tie: the higher tier wins
    for _ in 0..50 {
        let items = packs::generate_items(&mut s, spec(PackKind::Celestial, PackSize::Normal));
        assert_eq!(items[0], Some(PackItem::Consumable(C::Uranus)));
    }
    assert_eq!(
        packs::most_played_hand(&s.hand_plays_run),
        Some(HandType::TwoPair)
    );
    // Nothing played: a normal random pack.
    let s2 = packs_state(22);
    assert_eq!(packs::most_played_hand(&s2.hand_plays_run), None);
    // Without Telescope the first item is not forced.
    s.shop_mods.telescope = false;
    let not_always = (0..200).any(|_| {
        packs::generate_items(&mut s, spec(PackKind::Celestial, PackSize::Normal))[0]
            != Some(PackItem::Consumable(C::Uranus))
    });
    assert!(not_always);
}

#[test]
fn omen_globe_lets_arcana_packs_hold_spectrals() {
    let mut s = packs_state(23);
    let count = |s: &mut GameState| -> (u32, u32) {
        let (mut spec_n, mut total) = (0, 0);
        for _ in 0..2000 {
            for i in packs::generate_items(s, spec(PackKind::Arcana, PackSize::Normal))
                .iter()
                .flatten()
            {
                total += 1;
                if let PackItem::Consumable(id) = i
                    && id.kind() == ConsumableKind::Spectral
                    && *id != C::TheSoul
                {
                    spec_n += 1;
                }
            }
        }
        (spec_n, total)
    };
    assert_eq!(count(&mut s).0, 0);
    s.shop_mods.omen_globe = true;
    let (n, total) = count(&mut s);
    let frac = f64::from(n) / f64::from(total);
    assert!((frac - 0.2).abs() < 0.03, "{frac}");
}

#[test]
fn soul_and_black_hole_odds_are_about_a_third_of_a_percent() {
    let mut s = packs_state(24);
    let mut soul = 0u32;
    let mut slots = 0u32;
    for _ in 0..40_000 {
        for i in packs::generate_items(&mut s, spec(PackKind::Arcana, PackSize::Jumbo))
            .iter()
            .flatten()
        {
            slots += 1;
            if *i == PackItem::Consumable(C::TheSoul) {
                soul += 1;
            }
        }
    }
    let frac = f64::from(soul) / f64::from(slots);
    assert!((frac - 0.003).abs() < 0.0008, "{frac}");
    let mut bh = 0u32;
    let mut slots = 0u32;
    for _ in 0..40_000 {
        for i in packs::generate_items(&mut s, spec(PackKind::Celestial, PackSize::Jumbo))
            .iter()
            .flatten()
        {
            slots += 1;
            if *i == PackItem::Consumable(C::BlackHole) {
                bh += 1;
            }
        }
    }
    let frac = f64::from(bh) / f64::from(slots);
    assert!((frac - 0.003).abs() < 0.0008, "{frac}");
}

#[test]
fn standard_cards_follow_the_enhancement_seal_and_edition_odds() {
    let mut s = packs_state(25);
    let (mut enh, mut seal, mut ed, mut n) = (0u32, 0u32, 0u32, 0u32);
    for _ in 0..20_000 {
        for i in packs::generate_items(&mut s, spec(PackKind::Standard, PackSize::Jumbo))
            .iter()
            .flatten()
        {
            let PackItem::Card(c) = i else { panic!() };
            n += 1;
            enh += u32::from(c.enhancement != Enhancement::None);
            seal += u32::from(c.seal != Seal::None);
            ed += u32::from(c.edition != Edition::Base);
            assert_ne!(c.edition, Edition::Negative);
        }
    }
    let f = |x: u32| f64::from(x) / f64::from(n);
    assert!((f(enh) - 0.4).abs() < 0.01, "{}", f(enh));
    assert!((f(seal) - 0.2).abs() < 0.01, "{}", f(seal));
    assert!((f(ed) - 0.08).abs() < 0.005, "{}", f(ed));
}

// ----- pack tags --------------------------------------------------------------------------------------

#[test]
fn a_charm_tag_opens_a_mega_arcana_pack_on_skip_and_returns_to_blind_select() {
    let mut g = Game::new(26, GameConfig::default());
    g.state_mut().blind_tags[0] = TagId::Charm;
    g.step(Action::SkipBlind).expect("skip");
    assert_eq!(g.state().phase, Phase::PackOpen);
    let p = pack(&g);
    assert_eq!(p.spec, spec(PackKind::Arcana, PackSize::Mega));
    assert_eq!(p.origin, PackOrigin::Tag);
    assert_eq!(g.state().money, 4, "the pack is free");
    assert!(g.state().tags.is_empty(), "the tag is used up");
    assert_eq!(
        g.state().blind,
        balatro_core::blinds::BlindKind::Big,
        "the skip already advanced"
    );
    assert_eq!(g.state().deck.hand.len(), 8);
    g.step(Action::SkipPack).expect("skip pack");
    assert_eq!(g.state().phase, Phase::BlindSelect);
    assert!(g.state().deck.hand.is_empty());
    g.state().check_invariants().expect("invariants");
}

#[test]
fn a_double_tag_queues_two_packs_that_open_one_after_the_other() {
    let mut g = Game::new(27, GameConfig::default());
    g.state_mut().tags.push(TagId::Double);
    g.state_mut().blind_tags[0] = TagId::Charm;
    g.step(Action::SkipBlind).expect("skip");
    assert_eq!(g.state().phase, Phase::PackOpen);
    assert_eq!(g.state().pending_packs.len(), 1);
    g.step(Action::SkipPack).expect("first pack");
    assert_eq!(
        g.state().phase,
        Phase::PackOpen,
        "the second pack opens at once"
    );
    assert!(g.state().pending_packs.is_empty());
    g.step(Action::SkipPack).expect("second pack");
    assert_eq!(g.state().phase, Phase::BlindSelect);
    assert!(g.state().tags.is_empty());
}

// ----- replay / determinism ------------------------------------------------------------------------------

#[test]
fn pack_games_are_deterministic_and_replayable() {
    use balatro_core::agents::{Agent, GreedyAgent};
    use balatro_core::serialize::Replay;
    let mut game = Game::new(28, GameConfig::default());
    let mut replay = Replay::record(28, GameConfig::default());
    let mut agent = GreedyAgent::new();
    let mut opened = 0;
    while !game.is_terminal() && replay.actions.len() < 600 {
        let mut a = agent.act(&game);
        // Buy a pack whenever the greedy agent would leave the shop with money to spare.
        if a == Action::EndShop
            && game.state().phase == Phase::Shop
            && let Some(open) = game
                .legal_actions()
                .into_iter()
                .find(|x| matches!(x, Action::OpenPack(_)))
        {
            a = open;
            opened += 1;
        }
        game.step(a).expect("legal");
        replay.push(a);
    }
    assert!(opened > 0, "the run must open a pack");
    let again = replay.play().expect("replay");
    assert_eq!(again.state(), game.state());
}
