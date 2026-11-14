//! Phase 5B skip tags: all 24 tags through `Game::step` (skip, gain, timing of the effect),
//! Double Tag copying, the min-ante rule, pack tags (`PackOpen` and back), determinism.
//! Guide: docs/design/shop_packs_vouchers_tags.md; rules: docs/mechanics/tags.md.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{ConsumableId, TagId, try_tag_def};
use balatro_core::enums::{Edition, HandType, Phase, Rarity};
use balatro_core::game::Game;
use balatro_core::hooks::{PackKind, SideEffect};
use balatro_core::joker::JokerInstance;
use balatro_core::packs::{PackOrigin, PackSize};
use balatro_core::shop::{self, ShopCard, ShopCardKind};
use balatro_core::state::GameConfig;
use balatro_core::tag::TagCtx;

// ----- helpers ---------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn win(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b11111)).expect("winning play");
}

fn skip_with(g: &mut Game, tag: TagId) {
    g.state_mut().blind_tags = [tag, tag];
    g.step(Action::SkipBlind).expect("skip");
}

/// Beat the Small blind with `tag` held: the Shop phase (the tag's shop hook has run).
fn shop_with(seed: u64, tag: TagId) -> Game {
    let mut g = new_game(seed);
    g.state_mut().tags.push(tag);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

/// Run one `on_shop_enter` hook by hand on the current (edited) shop.
fn run_shop_hook(g: &mut Game, tag: TagId) -> bool {
    let f = try_tag_def(tag)
        .and_then(|d| d.on_shop_enter)
        .expect("shop hook");
    let mut fx: Vec<SideEffect> = Vec::new();
    f(&mut TagCtx::new(g.state_mut(), &mut fx, None))
}

fn joker_card(edition: Edition) -> Option<ShopCard> {
    let mut j = JokerInstance::new(0, JokerId::Joker);
    j.edition = edition;
    Some(ShopCard {
        kind: ShopCardKind::Joker(j),
        free: false,
    })
}

fn card_edition(c: &Option<ShopCard>) -> Option<(Edition, bool)> {
    match c {
        Some(ShopCard {
            kind: ShopCardKind::Joker(j),
            free,
        }) => Some((j.edition, *free)),
        _ => None,
    }
}

// ----- registry ----------------------------------------------------------------------------------

#[test]
fn all_24_tags_are_defined_with_the_wiki_min_ante() {
    use TagId as T;
    let min2 = [
        T::Negative,
        T::Standard,
        T::Meteor,
        T::Buffoon,
        T::Handy,
        T::Garbage,
        T::Ethereal,
        T::TopUp,
        T::Orbital,
    ];
    assert_eq!(TagId::COUNT, 24);
    for id in TagId::ALL {
        let d = try_tag_def(id).unwrap_or_else(|| panic!("{id:?} undefined"));
        assert_eq!(d.id, id);
        assert_eq!(d.min_ante, if min2.contains(&id) { 2 } else { 1 }, "{id:?}");
    }
}

#[test]
fn tags_below_their_min_ante_are_never_offered() {
    // Ante 1: exactly the 15 tags with min ante 1 appear, over many seeds.
    let mut seen = [false; TagId::COUNT];
    for seed in 0..400 {
        let g = new_game(seed);
        for t in g.state().blind_tags {
            assert_eq!(g.state().ante, 1);
            assert!(try_tag_def(t).expect("def").min_ante <= 1, "{t:?}");
            seen[t.index()] = true;
        }
    }
    assert_eq!(seen.iter().filter(|&&b| b).count(), 15);
    // Ante 2: all 24 appear over many seeds, and Orbital carries a hand.
    let mut seen2 = [false; TagId::COUNT];
    for seed in 0..200 {
        let mut g = new_game(seed);
        for _ in 0..2 {
            g.step(Action::SelectBlind).expect("select");
            win(&mut g);
            g.step(Action::EndShop).expect("end");
        }
        g.step(Action::SelectBlind).expect("boss");
        win(&mut g);
        if g.state().phase != Phase::Shop {
            continue;
        }
        g.step(Action::EndShop).expect("end");
        if g.state().ante != 2 {
            continue;
        }
        for (i, t) in g.state().blind_tags.into_iter().enumerate() {
            seen2[t.index()] = true;
            assert_eq!(
                g.state().blind_tag_hands[i].is_some(),
                t == TagId::Orbital,
                "{t:?}"
            );
        }
    }
    assert!(
        seen2.iter().all(|&b| b),
        "ante 2 offers every tag: {seen2:?}"
    );
}

// ----- immediate money / level tags -----------------------------------------------------------------

#[test]
fn handy_pays_one_dollar_per_hand_played_this_run() {
    let mut g = new_game(1);
    g.state_mut().stats.hands_played = 7;
    skip_with(&mut g, TagId::Handy);
    assert_eq!(g.state().money, 4 + 7);
    assert!(g.state().tags.is_empty());
    let mut g = new_game(1);
    skip_with(&mut g, TagId::Handy);
    assert_eq!(g.state().money, 4, "no hands played: $0");
}

#[test]
fn garbage_pays_one_dollar_per_unused_discard_this_run() {
    let mut g = new_game(2);
    g.state_mut().stats.unused_discards = 5;
    skip_with(&mut g, TagId::Garbage);
    assert_eq!(g.state().money, 4 + 5);
    assert!(g.state().tags.is_empty());
}

#[test]
fn speed_pays_five_per_blind_skipped_including_its_own_skip() {
    let mut g = new_game(3);
    skip_with(&mut g, TagId::Speed);
    assert_eq!(g.state().money, 4 + 5);
    skip_with(&mut g, TagId::Speed);
    assert_eq!(g.state().money, 4 + 5 + 10);
    assert!(g.state().tags.is_empty());
}

#[test]
fn orbital_levels_up_the_offered_hand_by_three() {
    let mut g = new_game(4);
    g.state_mut().blind_tags = [TagId::Orbital, TagId::Orbital];
    g.state_mut().blind_tag_hands = [Some(HandType::Pair), Some(HandType::Flush)];
    let before = g.state().hand_levels;
    g.step(Action::SkipBlind).expect("skip");
    let after = g.state().hand_levels;
    assert_eq!(
        after[HandType::Pair.index()],
        before[HandType::Pair.index()] + 3
    );
    assert_eq!(
        after[HandType::Flush.index()],
        before[HandType::Flush.index()]
    );
    assert!(g.state().tags.is_empty());
    // The Big blind's tag carries its own hand.
    g.step(Action::SkipBlind).expect("skip big");
    assert_eq!(
        g.state().hand_levels[HandType::Flush.index()],
        before[HandType::Flush.index()] + 3
    );
}

#[test]
fn top_up_creates_up_to_two_common_jokers_as_room_allows() {
    let mut g = new_game(5);
    skip_with(&mut g, TagId::TopUp);
    assert_eq!(g.state().jokers.len(), 2);
    for j in &g.state().jokers {
        assert_eq!(j.def().expect("def").rarity, Rarity::Common);
    }
    assert!(g.state().tags.is_empty());
    // Only one slot left: one joker is created, the other has no room (not an error).
    let mut g = new_game(5);
    for _ in 0..4 {
        g.add_joker(JokerInstance::new(0, JokerId::Joker))
            .expect("room");
    }
    skip_with(&mut g, TagId::TopUp);
    assert_eq!(g.state().jokers.len(), 5);
    g.state().check_invariants().expect("invariants");
}

// ----- pack tags ------------------------------------------------------------------------------------------

#[test]
fn pack_tags_open_their_pack_at_once_free_and_return_to_blind_select() {
    let cases = [
        (TagId::Standard, PackKind::Standard, PackSize::Mega, false),
        (TagId::Meteor, PackKind::Celestial, PackSize::Mega, false),
        (TagId::Buffoon, PackKind::Buffoon, PackSize::Mega, false),
        (TagId::Charm, PackKind::Arcana, PackSize::Mega, true),
        (TagId::Ethereal, PackKind::Spectral, PackSize::Normal, true),
    ];
    for (tag, kind, size, deals_hand) in cases {
        let mut g = new_game(6);
        skip_with(&mut g, tag);
        assert_eq!(g.state().phase, Phase::PackOpen, "{tag:?}");
        let p = g.state().pack.as_ref().expect("pack");
        assert_eq!((p.spec.kind, p.spec.size), (kind, size), "{tag:?}");
        assert_eq!(p.origin, PackOrigin::Tag);
        assert_eq!(p.items.len(), usize::from(p.spec.cards()));
        assert_eq!(g.state().money, 4, "{tag:?}: the pack is free");
        assert!(g.state().tags.is_empty(), "{tag:?} is used up");
        assert_eq!(!g.state().deck.hand.is_empty(), deals_hand, "{tag:?}");
        assert_eq!(
            g.state().blind,
            BlindKind::Big,
            "the skip advanced the blind"
        );
        g.step(Action::SkipPack).expect("skip pack");
        assert_eq!(g.state().phase, Phase::BlindSelect, "{tag:?}");
        assert!(g.state().pack.is_none());
        assert!(g.state().deck.hand.is_empty());
        g.state().check_invariants().expect("invariants");
    }
}

#[test]
fn a_pack_tag_pack_can_be_picked_from_and_the_run_goes_on() {
    let mut g = new_game(7);
    skip_with(&mut g, TagId::Meteor);
    let levels = g.state().hand_levels;
    g.step(Action::PickPackItem {
        index: 0,
        targets: 0,
    })
    .expect("pick");
    assert_eq!(g.state().phase, Phase::PackOpen, "Mega: a second pick");
    g.step(Action::PickPackItem {
        index: 1,
        targets: 0,
    })
    .expect("pick 2");
    assert_eq!(g.state().phase, Phase::BlindSelect);
    assert!(g.state().hand_levels.iter().sum::<u32>() > levels.iter().sum::<u32>());
    g.step(Action::SelectBlind).expect("play on");
}

// ----- Double Tag ----------------------------------------------------------------------------------------------

#[test]
fn double_copies_the_next_tag_including_pack_tags_but_never_double() {
    let mut g = new_game(8);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Speed);
    assert_eq!(g.state().money, 4 + 5 + 5, "Speed ran twice (1 skip each)");
    assert!(g.state().tags.is_empty());

    let mut g = new_game(8);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Meteor);
    g.step(Action::SkipPack).expect("first");
    assert_eq!(g.state().phase, Phase::PackOpen);
    g.step(Action::SkipPack).expect("second");
    assert_eq!(g.state().phase, Phase::BlindSelect);

    let mut g = new_game(8);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Foil);
    assert_eq!(g.state().tags, vec![TagId::Foil, TagId::Foil]);
    // Double does not copy Double: both Doubles stay.
    let mut g = new_game(8);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Double);
    assert_eq!(g.state().tags, vec![TagId::Double, TagId::Double]);
}

// ----- shop tags --------------------------------------------------------------------------------------------------

#[test]
fn uncommon_and_rare_tags_add_a_free_joker_of_that_rarity_to_the_next_shop() {
    for (tag, rarity) in [
        (TagId::Uncommon, Rarity::Uncommon),
        (TagId::Rare, Rarity::Rare),
    ] {
        for seed in 0..10 {
            let mut g = new_game(seed);
            skip_with(&mut g, tag);
            assert_eq!(g.state().tags, vec![tag], "held until a shop");
            g.step(Action::SelectBlind).expect("select");
            win(&mut g);
            let s = g.state();
            assert!(
                s.tags.len() <= 1,
                "the second tag (big blind) was not skipped"
            );
            assert!(s.tags.is_empty(), "{tag:?} used up");
            let extra = s
                .shop
                .cards
                .last()
                .copied()
                .flatten()
                .expect("tag joker appended");
            assert!(extra.free);
            assert_eq!(shop::card_price(s, &extra), 0);
            let ShopCardKind::Joker(j) = extra.kind else {
                panic!("a joker")
            };
            assert_eq!(j.def().expect("def").rarity, rarity);
            // Tags join the held list once per skip: the next shop has none left.
            g.step(Action::EndShop).expect("end");
        }
    }
}

#[test]
fn edition_tags_modify_the_first_base_edition_joker_and_make_it_free() {
    for (tag, ed) in [
        (TagId::Negative, Edition::Negative),
        (TagId::Foil, Edition::Foil),
        (TagId::Holographic, Edition::Holographic),
        (TagId::Polychrome, Edition::Polychrome),
    ] {
        let mut g = shop_with(9, TagId::Coupon); // any shop state; cards are replaced below
        g.state_mut().shop.cards = vec![
            Some(ShopCard {
                kind: ShopCardKind::Consumable(ConsumableId::planet_for(HandType::Pair)),
                free: false,
            }),
            joker_card(Edition::Foil),
            joker_card(Edition::Base),
            joker_card(Edition::Base),
        ];
        assert!(run_shop_hook(&mut g, tag), "{tag:?} applies");
        let c = &g.state().shop.cards;
        assert_eq!(
            card_edition(&c[1]),
            Some((Edition::Foil, false)),
            "skips non-base"
        );
        assert_eq!(card_edition(&c[2]), Some((ed, true)), "{tag:?}");
        assert_eq!(
            card_edition(&c[3]),
            Some((Edition::Base, false)),
            "only one"
        );
    }
}

#[test]
fn an_edition_tag_without_a_base_joker_stays_held_for_the_next_shop() {
    let mut g = shop_with(9, TagId::Coupon);
    g.state_mut().shop.cards = vec![joker_card(Edition::Foil), None];
    assert!(!run_shop_hook(&mut g, TagId::Polychrome), "not consumed");
    assert_eq!(
        card_edition(&g.state().shop.cards[0]),
        Some((Edition::Foil, false))
    );
    // End to end: consumed iff the shop shows a free joker with that edition.
    for seed in 0..40 {
        let mut g = shop_with(seed, TagId::Holographic);
        let has = g
            .state()
            .shop
            .cards
            .iter()
            .any(|c| card_edition(c) == Some((Edition::Holographic, true)));
        assert_eq!(g.state().tags.is_empty(), has, "seed {seed}");
        if !has {
            // Still held: the next shop retries.
            assert_eq!(g.state().tags, vec![TagId::Holographic]);
            g.step(Action::EndShop).expect("end");
        }
    }
}

#[test]
fn voucher_tag_adds_a_second_distinct_voucher_to_the_next_shop() {
    for seed in 0..10 {
        let g = shop_with(seed, TagId::Voucher);
        let v = &g.state().shop.vouchers;
        assert_eq!(v.len(), 2, "seed {seed}");
        let (a, b) = (v[0].expect("ante"), v[1].expect("tag"));
        assert_ne!(a.id, b.id);
        assert!(a.from_ante && !b.from_ante);
        assert!(g.state().tags.is_empty());
        g.state().check_invariants().expect("invariants");
    }
}

#[test]
fn d6_makes_the_first_reroll_free_in_the_next_shop_only() {
    let mut g = new_game(10);
    skip_with(&mut g, TagId::D6);
    assert_eq!(g.state().tags, vec![TagId::D6]);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert!(g.state().tags.is_empty());
    assert_eq!(shop::reroll_price(g.state()), 0);
    g.step(Action::Reroll).expect("reroll");
    assert_eq!(
        shop::reroll_price(g.state()),
        1,
        "base $0 + one paid reroll"
    );
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select big");
    win(&mut g);
    assert_eq!(shop::reroll_price(g.state()), 5, "the next shop is normal");
}

#[test]
fn juggle_adds_three_hand_size_for_the_next_round_only() {
    let mut g = new_game(11);
    skip_with(&mut g, TagId::Juggle);
    assert_eq!(
        g.state().tags,
        vec![TagId::Juggle],
        "held until a round starts"
    );
    assert_eq!(g.state().hand_size(), 8);
    g.step(Action::SelectBlind).expect("select");
    assert!(g.state().tags.is_empty());
    assert_eq!(g.state().hand_size(), 11);
    assert_eq!(g.state().deck.hand.len(), 11);
    win(&mut g);
    assert_eq!(g.state().hand_size(), 8, "cleared when the round ends");
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("next round");
    assert_eq!(g.state().deck.hand.len(), 8);
}

// ----- prototypes (Phase 5.0), one-line coverage so all 24 are tested here -------------------------------

#[test]
fn prototype_tags_economy_investment_coupon_boss_still_work() {
    let mut g = new_game(12);
    g.state_mut().money = 30;
    skip_with(&mut g, TagId::Economy);
    assert_eq!(g.state().money, 60);

    let g = shop_with(12, TagId::Coupon);
    assert!(g.state().tags.is_empty());
    assert!(g.state().shop.packs.iter().flatten().all(|p| p.free));

    let mut g = new_game(12);
    skip_with(&mut g, TagId::Boss);
    assert!(g.state().tags.is_empty());

    let mut g = new_game(12);
    g.state_mut().tags.push(TagId::Investment);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert_eq!(
        g.state().tags,
        vec![TagId::Investment],
        "only a boss pays it"
    );
}

// ----- all tags, smoke + determinism ----------------------------------------------------------------------------

fn play_through_tag(tag: TagId) -> Game {
    let mut g = new_game(13);
    skip_with(&mut g, tag);
    for _ in 0..40 {
        match g.state().phase {
            Phase::PackOpen => {
                g.step(Action::SkipPack).expect("skip pack");
            }
            Phase::BlindSelect => {
                if g.state().blind == BlindKind::Boss {
                    break;
                }
                g.step(Action::SelectBlind).expect("select");
            }
            Phase::Play => win(&mut g),
            Phase::Shop => {
                g.step(Action::EndShop).expect("end");
            }
            _ => break,
        }
        g.state().check_invariants().expect("invariants");
    }
    g
}

#[test]
fn every_tag_runs_through_a_round_trip_and_is_deterministic() {
    for id in TagId::ALL {
        let a = play_through_tag(id);
        let b = play_through_tag(id);
        assert_eq!(a.state(), b.state(), "{id:?}");
    }
}
