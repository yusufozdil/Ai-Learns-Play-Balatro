//! Phase 5E: the eight stakes (docs/mechanics/stakes.md). One test per effect; every expected
//! number is derived in a comment.

use balatro_core::actions::{Action, ActionError};
use balatro_core::blinds::{self, BlindKind};
use balatro_core::content::ConsumableId;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::jokers::try_def;
use balatro_core::enums::{Phase, Rarity};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::RngKey;
use balatro_core::shop::{self, ShopCard, ShopCardKind};
use balatro_core::state::{DeckKind, GameConfig, StakeKind};

fn game(stake: StakeKind, seed: u64) -> Game {
    Game::new(
        seed,
        GameConfig {
            stake,
            ..GameConfig::default()
        },
    )
}

/// Beat the current blind with one hand and, if the shop opened, leave it.
fn beat(g: &mut Game) -> balatro_core::game::StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn beat_and_leave_shop(g: &mut Game) {
    g.step(Action::SelectBlind).expect("select");
    beat(g);
    if g.state().phase == Phase::Shop {
        g.step(Action::EndShop).expect("end shop");
    }
}

/// Roll `n` shop jokers and return their sticker sets.
fn rolled(stake: StakeKind, n: usize) -> Vec<JokerInstance> {
    let mut g = game(stake, 7);
    let mut v = Vec::new();
    for _ in 0..n {
        let j = shop::roll_joker(g.state_mut(), RngKey::Shop, None, &[]).expect("a joker");
        v.push(j);
    }
    v
}

fn pct(count: usize, n: usize) -> f64 {
    100.0 * count as f64 / n as f64
}

#[test]
fn stake_ladder_flags_are_cumulative() {
    use StakeKind::*;
    // (stake, no small reward, eternal, discard penalty, perishable, rental)
    let table = [
        (White, false, false, 0, false, false),
        (Red, true, false, 0, false, false),
        (Green, true, false, 0, false, false),
        (Black, true, true, 0, false, false),
        (Blue, true, true, 1, false, false),
        (Purple, true, true, 1, false, false),
        (Orange, true, true, 1, true, false),
        (Gold, true, true, 1, true, true),
    ];
    assert_eq!(StakeKind::ALL.len(), 8);
    for (st, small, eternal, disc, perish, rental) in table {
        assert_eq!(st.no_small_reward(), small, "{st:?}");
        assert_eq!(st.eternal_jokers(), eternal, "{st:?}");
        assert_eq!(st.discard_penalty(), disc, "{st:?}");
        assert_eq!(st.perishable_jokers(), perish, "{st:?}");
        assert_eq!(st.rental_jokers(), rental, "{st:?}");
        assert_eq!(StakeKind::from_name(st.name()), Some(st));
    }
    assert_eq!(StakeKind::from_name("PURPLE"), Some(Purple));
    assert_eq!(StakeKind::from_name("rainbow"), None);
}

#[test]
fn stake_chip_tables_per_scaling_level() {
    use StakeKind::*;
    // blinds.md §2: White and Red use the base table, Green/Black/Blue the second, Purple/Orange/
    // Gold the third.
    let white = [
        300.0, 800.0, 2_000.0, 5_000.0, 11_000.0, 20_000.0, 35_000.0, 50_000.0,
    ];
    let green = [
        300.0, 900.0, 2_600.0, 8_000.0, 20_000.0, 36_000.0, 60_000.0, 100_000.0,
    ];
    let purple = [
        300.0, 1_000.0, 3_200.0, 9_000.0, 25_000.0, 60_000.0, 110_000.0, 200_000.0,
    ];
    for (stakes, table) in [
        (vec![White, Red], white),
        (vec![Green, Black, Blue], green),
        (vec![Purple, Orange, Gold], purple),
    ] {
        for st in stakes {
            for (i, &base) in table.iter().enumerate() {
                let ante = i as i32 + 1;
                assert_eq!(
                    blinds::base_chips(ante, st.scaling()),
                    base,
                    "{st:?} ante {ante}"
                );
            }
        }
    }
}

#[test]
fn green_stake_raises_the_requirement_in_a_run() {
    // Ante 1 is identical (300), ante 2 Small is 900 on Green against 800 on White.
    for (stake, expect) in [(StakeKind::White, 800.0), (StakeKind::Green, 900.0)] {
        let mut g = game(stake, 3);
        assert_eq!(g.state().target, 300.0);
        for _ in 0..3 {
            beat_and_leave_shop(&mut g);
        }
        let s = g.state();
        assert_eq!((s.ante, s.blind), (2, BlindKind::Small));
        assert_eq!(s.target, expect, "{stake:?}");
    }
    // Purple: 1,000.
    let mut g = game(StakeKind::Purple, 3);
    for _ in 0..3 {
        beat_and_leave_shop(&mut g);
    }
    assert_eq!(g.state().target, 1_000.0);
}

#[test]
fn red_stake_small_blind_pays_nothing() {
    for (stake, small) in [(StakeKind::White, 3), (StakeKind::Red, 0)] {
        let mut g = game(stake, 1);
        g.step(Action::SelectBlind).unwrap();
        let co = beat(&mut g).round.unwrap().cash_out.unwrap();
        assert_eq!(co.blind_reward, small, "{stake:?}");
    }
    // The Big blind still pays $4 on Red.
    let mut g = game(StakeKind::Red, 1);
    beat_and_leave_shop(&mut g);
    g.step(Action::SelectBlind).unwrap();
    let co = beat(&mut g).round.unwrap().cash_out.unwrap();
    assert_eq!(co.blind_reward, 4);
}

#[test]
fn blue_stake_takes_a_discard() {
    // Red Deck: 3 + 1 = 4 discards on White; Blue Stake: 3.
    let g = game(StakeKind::White, 1);
    assert_eq!(g.state().round_discards(), 4);
    let mut g = game(StakeKind::Blue, 1);
    assert_eq!(g.state().round_discards(), 3);
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().discards_left, 3);
    // Cumulative: Gold has it too, and a deck without extra discards drops to 2.
    let g = Game::new(
        1,
        GameConfig {
            deck: DeckKind::Blue,
            stake: StakeKind::Gold,
            endless: false,
        },
    );
    assert_eq!(g.state().round_discards(), 2);
}

#[test]
fn black_stake_makes_about_30_percent_eternal() {
    let n = 4_000;
    let v = rolled(StakeKind::Black, n);
    let eternal = v.iter().filter(|j| j.stickers.eternal).count();
    let p = pct(eternal, n);
    assert!((26.0..34.0).contains(&p), "eternal {p:.1}%");
    // Nothing else below Orange / Gold.
    assert!(
        v.iter()
            .all(|j| !j.stickers.perishable && !j.stickers.rental)
    );
    // Blue keeps it (cumulative); Green rolls no sticker at all.
    assert!(
        rolled(StakeKind::Blue, 200)
            .iter()
            .any(|j| j.stickers.eternal)
    );
    assert!(
        rolled(StakeKind::Green, 500)
            .iter()
            .all(|j| j.stickers == Default::default())
    );
}

#[test]
fn eternal_never_lands_on_incompatible_jokers() {
    let v = rolled(StakeKind::Black, 6_000);
    let mut seen_incompatible = false;
    for j in &v {
        let d = j.def().unwrap();
        if !d.eternal_compatible {
            seen_incompatible = true;
            assert!(!j.stickers.eternal, "{:?} cannot be eternal", j.id);
        }
    }
    assert!(
        seen_incompatible,
        "sample should include incompatible jokers"
    );
    // The 11 self-destroying jokers of the wiki (stakes.md §2).
    for id in [
        JokerId::GrosMichel,
        JokerId::IceCream,
        JokerId::Cavendish,
        JokerId::Luchador,
        JokerId::TurtleBean,
        JokerId::DietCola,
        JokerId::Popcorn,
        JokerId::Ramen,
        JokerId::Seltzer,
        JokerId::MrBones,
        JokerId::InvisibleJoker,
    ] {
        assert!(!try_def(id).unwrap().eternal_compatible, "{id:?}");
    }
}

#[test]
fn eternal_joker_cannot_be_sold() {
    let mut g = game(StakeKind::Black, 1);
    let mut j = JokerInstance::new(0, JokerId::Joker);
    j.stickers.eternal = true;
    g.add_joker(j).unwrap();
    assert_eq!(g.step(Action::SellJoker(0)), Err(ActionError::EternalJoker));
}

#[test]
fn orange_stake_makes_perishable_exclusive_with_eternal() {
    let n = 6_000;
    let v = rolled(StakeKind::Orange, n);
    let eternal = v.iter().filter(|j| j.stickers.eternal).count();
    let perishable = v.iter().filter(|j| j.stickers.perishable).count();
    // 30% each of the compatible jokers (a few incompatible ones lose their band).
    assert!((24.0..34.0).contains(&pct(eternal, n)), "eternal {eternal}");
    assert!(
        (24.0..34.0).contains(&pct(perishable, n)),
        "perishable {perishable}"
    );
    assert!(
        v.iter()
            .all(|j| !(j.stickers.eternal && j.stickers.perishable)),
        "Eternal and Perishable never combine"
    );
    assert!(
        v.iter()
            .filter(|j| j.stickers.perishable)
            .all(|j| j.stickers.perish_rounds_left == 5)
    );
    for j in &v {
        if !j.def().unwrap().perishable_compatible {
            assert!(!j.stickers.perishable, "{:?}", j.id);
        }
    }
    // Black Stake has no perishable ones.
    assert!(
        rolled(StakeKind::Purple, 1_000)
            .iter()
            .all(|j| !j.stickers.perishable)
    );
}

#[test]
fn perishable_joker_is_debuffed_after_five_won_rounds() {
    let mut g = game(StakeKind::Orange, 5);
    let mut j = JokerInstance::new(0, JokerId::Joker);
    j.stickers.perishable = true;
    j.stickers.perish_rounds_left = 5;
    g.add_joker(j).unwrap();
    for round in 1..=5u8 {
        assert!(!g.state().jokers[0].debuffed, "round {round} start");
        assert_eq!(g.state().jokers[0].stickers.perish_rounds_left, 6 - round);
        beat_and_leave_shop(&mut g);
    }
    let j = g.state().jokers[0];
    assert_eq!(j.stickers.perish_rounds_left, 0);
    assert!(j.debuffed, "debuffed after the 5th won round");
    // A debuffed joker adds nothing: Joker's +4 mult is gone.
    assert_eq!(
        g.state().mods,
        balatro_core::hooks::collect_modifiers(&g.state().jokers)
    );
    g.state().check_invariants().unwrap();
}

#[test]
fn perishable_golden_joker_pays_four_times() {
    // Wiki: the debuff lands before the cash-out, so a Perishable Golden Joker ($4) pays for
    // rounds 1-4 only.
    let mut g = game(StakeKind::Orange, 5);
    let mut j = JokerInstance::new(0, JokerId::GoldenJoker);
    j.stickers.perishable = true;
    j.stickers.perish_rounds_left = 5;
    g.add_joker(j).unwrap();
    let mut paid = Vec::new();
    for _ in 0..5 {
        g.step(Action::SelectBlind).unwrap();
        let co = beat(&mut g).round.unwrap().cash_out.unwrap();
        paid.push(co.jokers);
        if g.state().phase == Phase::Shop {
            g.step(Action::EndShop).unwrap();
        }
    }
    assert_eq!(paid, vec![4, 4, 4, 4, 0]);
}

#[test]
fn gold_stake_rolls_rental_independently() {
    let n = 5_000;
    let v = rolled(StakeKind::Gold, n);
    let rental = v.iter().filter(|j| j.stickers.rental).count();
    let p = pct(rental, n);
    assert!((26.0..34.0).contains(&p), "rental {p:.1}%");
    // Independent of Eternal / Perishable: they co-occur.
    assert!(v.iter().any(|j| j.stickers.rental && j.stickers.eternal));
    assert!(v.iter().any(|j| j.stickers.rental && j.stickers.perishable));
    // Rental sell value is $1.
    assert!(
        v.iter()
            .filter(|j| j.stickers.rental)
            .all(|j| j.sell_value == 1)
    );
    // Orange has none.
    assert!(
        rolled(StakeKind::Orange, 1_000)
            .iter()
            .all(|j| !j.stickers.rental)
    );
}

#[test]
fn rental_joker_costs_one_dollar_in_the_shop() {
    let mut g = game(StakeKind::Gold, 1);
    let mut j = JokerInstance::new(0, JokerId::Blueprint); // base $10
    j.stickers.rental = true;
    let card = ShopCard {
        kind: ShopCardKind::Joker(j),
        free: false,
    };
    assert_eq!(shop::card_price(g.state(), &card), 1);
    let mut plain = card;
    if let ShopCardKind::Joker(ref mut p) = plain.kind {
        p.stickers.rental = false;
    }
    assert_eq!(shop::card_price(g.state(), &plain), 10);
    // Buying it through the shop debits $1 and leaves a $1 sell value.
    g.state_mut().phase = Phase::Shop;
    g.state_mut().shop.cards = vec![Some(card)];
    g.state_mut().money = 4;
    g.step(Action::Buy(0)).unwrap();
    assert_eq!(g.state().money, 3);
    assert_eq!(g.state().jokers[0].sell_value, 1);
    assert!(g.state().jokers[0].stickers.rental);
}

#[test]
fn rental_rent_is_charged_before_the_payout() {
    // Gold Stake: Small pays $0. Money 10, one rental joker, one hand played (3 hands left).
    // Rent -3 -> 7; interest on 7 = $1; blind $0; hands $3 -> 11.
    let mut g = game(StakeKind::Gold, 2);
    let mut j = JokerInstance::new(0, JokerId::Joker);
    j.stickers.rental = true;
    g.add_joker(j).unwrap();
    g.state_mut().money = 10;
    g.step(Action::SelectBlind).unwrap();
    let co = beat(&mut g).round.unwrap().cash_out.unwrap();
    assert_eq!((co.interest_base, co.interest, co.hands), (7, 1, 3));
    assert_eq!(g.state().money, 11);
    // Without the sticker: 10 + interest 2 + hands 3 = 15.
    let mut g = game(StakeKind::Gold, 2);
    g.add_joker(JokerInstance::new(0, JokerId::Joker)).unwrap();
    g.state_mut().money = 10;
    g.step(Action::SelectBlind).unwrap();
    beat(&mut g);
    assert_eq!(g.state().money, 15);
}

#[test]
fn rent_goes_into_debt_and_stacks_per_joker() {
    let mut g = game(StakeKind::Gold, 2);
    for _ in 0..2 {
        let mut j = JokerInstance::new(0, JokerId::Joker);
        j.stickers.rental = true;
        g.add_joker(j).unwrap();
    }
    g.state_mut().money = 0;
    g.step(Action::SelectBlind).unwrap();
    beat(&mut g);
    // 0 - 6 = -6, no interest, hands $3 (blind pays 0) = -3.
    assert_eq!(g.state().money, -3);
}

#[test]
fn legendary_and_non_shop_jokers_never_carry_stickers() {
    let mut g = game(StakeKind::Gold, 4);
    let mut soul_rolls = 0;
    for _ in 0..300 {
        let j = shop::roll_joker(g.state_mut(), RngKey::Jokers, Some(Rarity::Legendary), &[])
            .expect("legendary");
        assert_eq!(j.stickers, Default::default());
        soul_rolls += 1;
        // The plain roll (Judgement, Wraith, Riff-raff, Top-up) never rolls stickers either.
        let p = shop::roll_joker_plain(g.state_mut(), RngKey::Jokers, None, &[]).unwrap();
        assert_eq!(p.stickers, Default::default());
    }
    assert!(soul_rolls > 0);
    // Judgement in a Gold Stake game: the created joker is clean.
    g.state_mut().consumables.clear();
    let mut c = balatro_core::consumable::ConsumableInstance::new(ConsumableId::Judgement);
    c.negative = true;
    g.add_consumable(c).unwrap();
    g.step(Action::SelectBlind).unwrap();
    g.step(Action::UseConsumable {
        slot: 0,
        targets: 0,
    })
    .unwrap();
    assert_eq!(g.state().jokers.len(), 1);
    assert_eq!(g.state().jokers[0].stickers, Default::default());
}

#[test]
fn buffoon_pack_jokers_carry_stickers() {
    // Pack and shop jokers use the same roll: a Gold Stake Buffoon pack yields stickered jokers.
    let mut stickered = 0;
    for seed in 0..60u64 {
        let mut g = game(StakeKind::Gold, seed);
        g.state_mut().phase = Phase::Shop;
        g.state_mut().money = 100;
        balatro_core::packs::open(
            g.state_mut(),
            balatro_core::packs::PackSpec {
                kind: balatro_core::hooks::PackKind::Buffoon,
                size: balatro_core::packs::PackSize::Mega,
            },
            balatro_core::packs::PackOrigin::Shop,
        );
        for it in g.state().pack.as_ref().unwrap().items.iter().flatten() {
            if let balatro_core::packs::PackItem::Joker(j) = it
                && j.stickers != Default::default()
            {
                stickered += 1;
            }
        }
    }
    assert!(stickered > 20, "{stickered}");
}

#[test]
fn white_and_red_stake_consume_no_sticker_rolls() {
    // Bit-identical RNG use below Black: the stream is untouched by the sticker roll.
    let mut a = game(StakeKind::White, 9);
    let mut b = game(StakeKind::Red, 9);
    for _ in 0..50 {
        let ja = shop::roll_joker(a.state_mut(), RngKey::Shop, None, &[]).unwrap();
        let jb = shop::roll_joker(b.state_mut(), RngKey::Shop, None, &[]).unwrap();
        assert_eq!(ja, jb);
    }
}

#[test]
fn sticker_compatibility_flags_match_the_wiki_lists() {
    use balatro_core::content::jokers::all_defs;
    let norm = |s: &str| s.to_ascii_lowercase().replace([' ', '.', '\''], "");
    let mut eternal_no: Vec<String> = all_defs()
        .filter(|d| !d.eternal_compatible)
        .map(|d| norm(d.name))
        .collect();
    let mut perish_no: Vec<String> = all_defs()
        .filter(|d| !d.perishable_compatible)
        .map(|d| norm(d.name))
        .collect();
    let list = |names: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = names.iter().map(|n| norm(n)).collect();
        v.sort();
        v
    };
    eternal_no.sort();
    perish_no.sort();
    // stakes.md §2 (wiki Eternal / Perishable pages).
    assert_eq!(
        eternal_no,
        list(&[
            "Gros Michel",
            "Ice Cream",
            "Cavendish",
            "Luchador",
            "Turtle Bean",
            "Diet Cola",
            "Popcorn",
            "Ramen",
            "Seltzer",
            "Mr. Bones",
            "Invisible Joker",
        ])
    );
    assert_eq!(
        perish_no,
        list(&[
            "Ceremonial Dagger",
            "Ride the Bus",
            "Runner",
            "Constellation",
            "Green Joker",
            "Red Card",
            "Madness",
            "Square Joker",
            "Vampire",
            "Hologram",
            "Rocket",
            "Obelisk",
            "Lucky Cat",
            "Flash Card",
            "Spare Trousers",
            "Castle",
            "Glass Joker",
            "Wee Joker",
        ])
    );
}
