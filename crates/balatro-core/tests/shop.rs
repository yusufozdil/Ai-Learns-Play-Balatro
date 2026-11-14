//! Phase 5.0 market: generation, prices, rerolls, buying/selling, Buy & Use, joker order,
//! voucher slot life cycle, joker pools, distributions (100k; the 1M version is `#[ignore]`) and
//! determinism. Guide: docs/design/shop_packs_vouchers_tags.md.

use balatro_core::actions::{Action, ActionError};
use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::consumable::{ConsumableInstance, ConsumableKind};
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::content::{ConsumableId as C, VoucherId};
use balatro_core::data::shop_weights::pack_weight;
use balatro_core::economy::{self, Discount};
use balatro_core::enums::{Edition, HandType, Phase, Rarity};
use balatro_core::game::Game;
use balatro_core::hooks::{PackKind, RunFlag};
use balatro_core::joker::JokerInstance;
use balatro_core::packs::{PackSize, PackSpec};
use balatro_core::rng::RngKey;
use balatro_core::shop::{self, ShopCard, ShopCardKind, ShopVoucher};
use balatro_core::state::{GameConfig, GameState};

// ----- helpers ---------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn in_play(seed: u64) -> Game {
    let mut g = new_game(seed);
    g.step(Action::SelectBlind).expect("select");
    g
}

/// Beat the Small blind: Shop phase (first shop of the run).
fn in_shop(seed: u64) -> Game {
    let mut g = in_play(seed);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("winning play");
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

fn add_joker(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("joker room")
}

/// A shop-state fixture: fresh state in the Shop phase (not the first shop).
fn shop_state(seed: u64) -> GameState {
    let mut s = GameState::new(seed, GameConfig::default());
    s.phase = Phase::Shop;
    s.stats.shops_entered = 1;
    s
}

/// Replace card slot `slot` of the shop.
fn put_card(g: &mut Game, slot: usize, kind: ShopCardKind, free: bool) {
    g.state_mut().shop.cards[slot] = Some(ShopCard { kind, free });
}

fn joker_kind(g: &Game, id: JokerId, ed: Edition) -> ShopCardKind {
    ShopCardKind::Joker(shop::joker_instance(g.state(), id, ed))
}

fn put_joker(g: &mut Game, slot: usize, id: JokerId, ed: Edition, free: bool) {
    let kind = joker_kind(g, id, ed);
    put_card(g, slot, kind, free);
}

// ----- layout ----------------------------------------------------------------------------------

#[test]
fn the_first_shop_has_two_cards_two_packs_one_voucher_and_opens_with_a_buffoon_pack() {
    let g = in_shop(1);
    let sh = &g.state().shop;
    assert_eq!(sh.cards.iter().flatten().count(), 2);
    assert_eq!(sh.packs.len(), 2);
    assert_eq!(
        sh.packs[0].expect("pack").spec,
        PackSpec {
            kind: PackKind::Buffoon,
            size: PackSize::Normal
        }
    );
    assert_eq!(sh.vouchers.len(), 1);
    assert_eq!(sh.rerolls, 0);
    assert_eq!(g.state().stats.shops_entered, 1);
    // Only the first shop of the run: the next one rolls both packs.
    let mut g = g;
    g.step(Action::EndShop).expect("end shop");
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().stats.shops_entered, 2);
}

#[test]
fn shops_never_hold_the_soul_black_hole_spectrals_or_unplayed_secret_planets() {
    let mut s = shop_state(2);
    for _ in 0..20_000 {
        let c = shop::roll_card(&mut s);
        if let ShopCardKind::Consumable(id) = c.kind {
            assert_ne!(id.kind(), ConsumableKind::Spectral, "{id:?}");
            assert!(!matches!(
                id,
                C::PlanetX | C::Ceres | C::Eris | C::TheSoul | C::BlackHole
            ));
        }
    }
    // After the secret hand was played its Planet may appear.
    s.hand_plays_run[HandType::FiveOfAKind.index()] = 1;
    let seen = (0..20_000).any(|_| {
        matches!(
            shop::roll_card(&mut s).kind,
            ShopCardKind::Consumable(C::PlanetX)
        )
    });
    assert!(seen);
}

#[test]
fn market_generation_is_deterministic_and_seed_dependent() {
    let render = |seed| {
        let g = in_shop(seed);
        format!("{:?}", g.state().shop)
    };
    assert_eq!(render(5), render(5));
    let differs = (6..30).any(|s| render(s) != render(5));
    assert!(differs);
}

#[test]
fn market_rolls_do_not_touch_the_other_streams() {
    // The shop uses RngKey::Shop only: the deck order of the next round is the same whether or
    // not a reroll happened.
    let mut a = in_shop(9);
    let mut b = in_shop(9);
    a.state_mut().money = 50;
    a.step(Action::Reroll).expect("reroll");
    for g in [&mut a, &mut b] {
        g.step(Action::EndShop).expect("end");
        g.step(Action::SelectBlind).expect("select");
    }
    assert_eq!(a.state().deck.hand, b.state().deck.hand);
}

// ----- prices ----------------------------------------------------------------------------------

#[test]
fn prices_include_edition_surcharges_discount_and_free_flags() {
    let mut g = in_shop(3);
    // Common joker base cost 2 (Joker): plain $2, Foil $4, Holographic $5, Polychrome $7.
    let j = |ed| ShopCard {
        kind: joker_kind(&g, JokerId::Joker, ed),
        free: false,
    };
    let p = |g: &Game, c: &ShopCard| shop::card_price(g.state(), c);
    assert_eq!(p(&g, &j(Edition::Base)), 2);
    assert_eq!(p(&g, &j(Edition::Foil)), 4);
    assert_eq!(p(&g, &j(Edition::Holographic)), 5);
    assert_eq!(p(&g, &j(Edition::Polychrome)), 7);
    assert_eq!(p(&g, &j(Edition::Negative)), 7);
    // Flagged free.
    let mut free = j(Edition::Polychrome);
    free.free = true;
    assert_eq!(p(&g, &free), 0);
    // Discount: floor((2x+1)(100-d)/200), at least $1: $6 -> 4 at 25%, 3 at 50%; $2 -> 2 / 1.
    let six = ShopCard {
        kind: ShopCardKind::Joker(shop::joker_instance(
            g.state(),
            JokerId::Joker,
            Edition::Foil,
        )),
        free: false,
    };
    assert_eq!(economy::shop_price(6, 0, Discount::Clearance), 4);
    assert_eq!(economy::shop_price(6, 0, Discount::Liquidation), 3);
    g.state_mut().shop_mods.discount = Discount::Clearance;
    assert_eq!(p(&g, &six), 3); // $4 * 0.75 = 3
    // Consumables: Tarot/Planet $3, Spectral $4.
    let tarot = ShopCard {
        kind: ShopCardKind::Consumable(C::TheFool),
        free: false,
    };
    g.state_mut().shop_mods.discount = Discount::None;
    assert_eq!(p(&g, &tarot), 3);
    // Astronomer: Planets (and Celestial packs) cost $0.
    add_joker(&mut g, JokerId::Astronomer);
    let planet = ShopCard {
        kind: ShopCardKind::Consumable(C::Pluto),
        free: false,
    };
    assert_eq!(p(&g, &planet), 0);
    assert_eq!(p(&g, &tarot), 3);
    let cel = balatro_core::shop::ShopPack {
        spec: PackSpec {
            kind: PackKind::Celestial,
            size: PackSize::Mega,
        },
        free: false,
    };
    assert_eq!(shop::pack_price_of(g.state(), &cel), 0);
    let arc = balatro_core::shop::ShopPack {
        spec: PackSpec {
            kind: PackKind::Arcana,
            size: PackSize::Mega,
        },
        free: false,
    };
    assert_eq!(shop::pack_price_of(g.state(), &arc), 8);
}

#[test]
fn pack_and_voucher_prices() {
    let g = in_shop(3);
    let price = |kind, size| {
        shop::pack_price_of(
            g.state(),
            &balatro_core::shop::ShopPack {
                spec: PackSpec { kind, size },
                free: false,
            },
        )
    };
    assert_eq!(price(PackKind::Standard, PackSize::Normal), 4);
    assert_eq!(price(PackKind::Buffoon, PackSize::Jumbo), 6);
    assert_eq!(price(PackKind::Arcana, PackSize::Mega), 8);
    assert_eq!(shop::voucher_price(g.state(), VoucherId::Overstock), 10);
}

#[test]
fn bought_items_sell_for_half_the_price_paid() {
    let mut g = in_shop(4);
    g.state_mut().money = 50;
    put_joker(&mut g, 0, JokerId::Joker, Edition::Foil, false); // $4
    put_joker(&mut g, 1, JokerId::Joker, Edition::Base, true); // $0
    g.step(Action::Buy(0)).expect("buy foil");
    assert_eq!(g.state().jokers[0].sell_value, 2); // floor(4/2)
    g.step(Action::Buy(1)).expect("buy free");
    assert_eq!(g.state().jokers[1].sell_value, 1); // a free item still sells for $1
    assert_eq!(g.state().money, 46);
}

// ----- reroll ----------------------------------------------------------------------------------

#[test]
fn reroll_costs_five_then_six_and_resets_in_the_next_shop() {
    let mut g = in_shop(7);
    g.state_mut().money = 100;
    let first_cards = g.state().shop.cards.clone();
    for (n, price) in [5, 6, 7].into_iter().enumerate() {
        assert_eq!(shop::reroll_price(g.state()), price, "reroll {n}");
        let before = g.state().money;
        let info = g.step(Action::Reroll).expect("reroll");
        assert_eq!(info.spent, Some(price));
        assert_eq!(g.state().money, before - price);
    }
    assert_eq!(g.state().shop.rerolls, 3);
    assert_ne!(g.state().shop.cards, first_cards);
    // Packs and the voucher are not touched by a reroll.
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().shop.rerolls, 0);
    assert_eq!(shop::reroll_price(g.state()), 5);
}

#[test]
fn reroll_keeps_packs_and_vouchers_and_needs_money() {
    let mut g = in_shop(8);
    g.state_mut().money = 4;
    assert_eq!(
        g.step(Action::Reroll),
        Err(ActionError::CannotAfford { cost: 5, money: 4 })
    );
    g.state_mut().money = 5;
    let packs = g.state().shop.packs.clone();
    let vouchers = g.state().shop.vouchers.clone();
    g.step(Action::Reroll).expect("reroll");
    assert_eq!(g.state().shop.packs, packs);
    assert_eq!(g.state().shop.vouchers, vouchers);
    assert_eq!(g.state().money, 0);
}

#[test]
fn reroll_base_follows_vouchers_and_the_d6_override() {
    let g = in_shop(8);
    let mut s = g.state().clone();
    s.shop_mods.reroll_base = Some(3);
    assert_eq!(shop::reroll_price(&s), 3);
    s.shop.rerolls = 2;
    assert_eq!(shop::reroll_price(&s), 5);
    s.shop.reroll_base = Some(0); // D6 Tag wins
    assert_eq!(shop::reroll_price(&s), 2);
}

#[test]
fn chaos_the_clown_gives_one_free_reroll_that_does_not_raise_the_price() {
    let mut g = in_shop(10);
    add_joker(&mut g, JokerId::ChaosTheClown);
    g.state_mut().money = 20;
    assert_eq!(shop::reroll_price(g.state()), 0);
    g.step(Action::Reroll).expect("free reroll");
    assert_eq!(g.state().money, 20);
    assert_eq!(g.state().shop.rerolls, 0);
    assert_eq!(shop::reroll_price(g.state()), 5); // the counter did not advance
    g.step(Action::Reroll).expect("paid reroll");
    assert_eq!(g.state().money, 15);
    assert_eq!(shop::reroll_price(g.state()), 6);
}

#[test]
fn a_free_reroll_is_legal_without_money_and_flash_card_grows_on_every_reroll() {
    let mut g = in_shop(11);
    add_joker(&mut g, JokerId::ChaosTheClown);
    let fc = add_joker(&mut g, JokerId::FlashCard);
    g.state_mut().money = 0;
    assert!(g.legal_actions().contains(&Action::Reroll));
    g.step(Action::Reroll).expect("free");
    assert_eq!(g.state().jokers[fc].value, 2.0);
    assert!(!g.legal_actions().contains(&Action::Reroll));
    g.state_mut().money = 9;
    g.step(Action::Reroll).expect("paid");
    assert_eq!(g.state().jokers[fc].value, 4.0);
}

// ----- buying ----------------------------------------------------------------------------------

#[test]
fn buying_a_joker_pays_fills_the_slot_and_leaves_the_slot_empty() {
    let mut g = in_shop(12);
    g.state_mut().money = 20;
    put_joker(&mut g, 0, JokerId::GreedyJoker, Edition::Base, false);
    let n = g.state().shop.cards.len();
    let info = g.step(Action::Buy(0)).expect("buy");
    assert_eq!(info.spent, Some(5));
    assert_eq!(g.state().money, 15);
    assert_eq!(g.state().jokers.len(), 1);
    assert_eq!(g.state().jokers[0].id, JokerId::GreedyJoker);
    assert_eq!(g.state().jokers[0].sell_value, 2);
    assert!(g.state().shop.cards[0].is_none());
    assert_eq!(g.state().shop.cards.len(), n, "slots do not shift");
    assert_eq!(
        g.step(Action::Buy(0)),
        Err(ActionError::InvalidShopSlot { index: 0 })
    );
    assert!(g.state().check_invariants().is_ok());
}

#[test]
fn buying_needs_money_a_free_joker_slot_and_a_consumable_slot() {
    let mut g = in_shop(13);
    put_joker(&mut g, 0, JokerId::GreedyJoker, Edition::Base, false);
    g.state_mut().money = 4;
    assert_eq!(
        g.step(Action::Buy(0)),
        Err(ActionError::CannotAfford { cost: 5, money: 4 })
    );
    g.state_mut().money = 100;
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    assert_eq!(g.step(Action::Buy(0)), Err(ActionError::NoJokerRoom));
    // A Negative joker needs no slot.
    put_joker(&mut g, 0, JokerId::GreedyJoker, Edition::Negative, false);
    g.step(Action::Buy(0)).expect("negative fits");
    // Consumable slots (2).
    put_card(&mut g, 1, ShopCardKind::Consumable(C::Pluto), false);
    for _ in 0..2 {
        g.add_consumable(ConsumableInstance::new(C::Mercury))
            .expect("room");
    }
    assert_eq!(g.step(Action::Buy(1)), Err(ActionError::NoConsumableRoom));
    g.step(Action::SellConsumable(0)).expect("sell");
    g.step(Action::Buy(1)).expect("buy planet");
    assert_eq!(g.state().consumables.len(), 2);
    assert_eq!(g.state().consumables[1].id, C::Pluto);
}

#[test]
fn credit_card_lets_a_purchase_go_into_debt() {
    let mut g = in_shop(14);
    add_joker(&mut g, JokerId::CreditCard);
    put_card(&mut g, 0, ShopCardKind::Consumable(C::Pluto), false);
    g.state_mut().money = 0;
    g.step(Action::Buy(0))
        .expect("within the $20 debt allowance");
    assert_eq!(g.state().money, -3);
    // Below -$20 nothing is affordable.
    g.state_mut().money = -19;
    put_card(&mut g, 1, ShopCardKind::Consumable(C::Mercury), false);
    assert!(matches!(
        g.step(Action::Buy(1)),
        Err(ActionError::CannotAfford { .. })
    ));
}

#[test]
fn buy_and_use_levels_a_planet_without_taking_a_slot() {
    let mut g = in_shop(15);
    g.state_mut().money = 10;
    put_card(&mut g, 0, ShopCardKind::Consumable(C::Mercury), false); // Pair
    for _ in 0..2 {
        g.add_consumable(ConsumableInstance::new(C::Pluto))
            .expect("room");
    }
    g.step(Action::BuyAndUse(0))
        .expect("buy and use with full slots");
    assert_eq!(g.state().hand_levels[HandType::Pair.index()], 2);
    assert_eq!(g.state().consumables.len(), 2);
    assert_eq!(g.state().money, 7);
    assert_eq!(g.state().stats.run.planets_used, 1);
    assert!(g.state().shop.cards[0].is_none());
}

#[test]
fn buy_and_use_refuses_cards_that_need_a_hand_or_cannot_be_used() {
    let mut g = in_shop(16);
    g.state_mut().money = 30;
    put_card(&mut g, 0, ShopCardKind::Consumable(C::TheMagician), false); // targets
    put_joker(&mut g, 1, JokerId::Joker, Edition::Base, false);
    assert_eq!(
        g.step(Action::BuyAndUse(0)),
        Err(ActionError::NotUsableOnBuy)
    );
    assert_eq!(
        g.step(Action::BuyAndUse(1)),
        Err(ActionError::NotUsableOnBuy)
    );
    // Plain Buy of the Magician is fine.
    g.step(Action::Buy(0)).expect("buy");
    assert_eq!(g.state().money, 27);
    // The Hermit works on the spot: it doubles money up to $20.
    put_card(&mut g, 0, ShopCardKind::Consumable(C::TheHermit), false);
    g.step(Action::BuyAndUse(0)).expect("hermit");
    assert_eq!(g.state().money, 27 - 3 + 20);
}

#[test]
fn a_playing_card_bought_with_magic_trick_joins_the_deck() {
    let mut g = in_shop(17);
    g.state_mut().money = 10;
    let card = balatro_core::card::Card::new(
        0,
        balatro_core::enums::Rank::Ace,
        balatro_core::enums::Suit::Spades,
    );
    put_card(&mut g, 0, ShopCardKind::PlayingCard(card), false);
    let n = g.state().deck.len();
    g.step(Action::Buy(0)).expect("buy card");
    assert_eq!(g.state().deck.len(), n + 1);
    assert_eq!(g.state().stats.cards_added, 1);
    assert_eq!(g.state().money, 9);
    assert!(g.state().check_invariants().is_ok());
}

// ----- selling and ordering ----------------------------------------------------------------------

#[test]
fn selling_is_allowed_in_every_phase_of_a_live_run() {
    let mut g = new_game(18);
    add_joker(&mut g, JokerId::Joker);
    add_joker(&mut g, JokerId::GreedyJoker);
    let m = g.state().money;
    g.step(Action::SellJoker(0)).expect("sell in BlindSelect");
    assert!(g.state().money > m);
    g.step(Action::SelectBlind).expect("select");
    g.step(Action::SellJoker(0)).expect("sell in Play");
    assert!(g.state().jokers.is_empty());
}

#[test]
fn campfire_grows_on_a_real_sale() {
    let mut g = in_shop(19);
    let cf = add_joker(&mut g, JokerId::Campfire);
    add_joker(&mut g, JokerId::Joker);
    let before = g.state().jokers[cf].value;
    g.step(Action::SellJoker(1)).expect("sell");
    assert!(g.state().jokers[cf].value > before);
    g.add_consumable(ConsumableInstance::new(C::Pluto))
        .expect("room");
    let v = g.state().jokers[cf].value;
    g.step(Action::SellConsumable(0)).expect("sell consumable");
    assert!(g.state().jokers[cf].value > v);
}

#[test]
fn move_joker_reorders_and_the_order_changes_the_score() {
    let mut g = in_play(20);
    add_joker(&mut g, JokerId::Joker); // +4 mult
    add_joker(&mut g, JokerId::Cavendish); // x3 mult
    let hi = (0..g.state().deck.hand.len())
        .map(|i| 1u32 << i)
        .find(|&m| g.preview_play(m).is_ok())
        .expect("a play");
    let plus_first = g.preview_play(hi).expect("preview").score;
    g.step(Action::MoveJoker { from: 0, to: 1 }).expect("move");
    assert_eq!(g.state().jokers[0].id, JokerId::Cavendish);
    assert_eq!(g.state().jokers[1].id, JokerId::Joker);
    let times_first = g.preview_play(hi).expect("preview").score;
    assert!(
        plus_first > times_first,
        "(m+4)*3 > m*3+4: {plus_first} vs {times_first}"
    );
    assert!(g.state().check_invariants().is_ok());
}

#[test]
fn move_joker_validation_and_phases() {
    let mut g = new_game(21);
    for id in [JokerId::Joker, JokerId::GreedyJoker, JokerId::LustyJoker] {
        add_joker(&mut g, id);
    }
    // Works in BlindSelect.
    g.step(Action::MoveJoker { from: 2, to: 0 }).expect("move");
    assert_eq!(g.state().jokers[0].id, JokerId::LustyJoker);
    assert_eq!(g.state().jokers[1].id, JokerId::Joker);
    let bad = [(1, 1), (0, 3), (3, 0)];
    for (from, to) in bad {
        assert_eq!(
            g.step(Action::MoveJoker { from, to }),
            Err(ActionError::InvalidJokerMove { from, to, len: 3 })
        );
    }
    // 3 jokers -> 6 ordered pairs, all legal.
    let moves = g
        .legal_actions()
        .into_iter()
        .filter(|a| matches!(a, Action::MoveJoker { .. }))
        .count();
    assert_eq!(moves, 6);
}

// ----- vouchers in the shop ------------------------------------------------------------------------

fn offer(g: &mut Game, id: VoucherId) {
    g.state_mut().ante_voucher = Some(id);
    g.state_mut().shop.vouchers = vec![Some(ShopVoucher {
        id,
        from_ante: true,
    })];
}

#[test]
fn buying_overstock_fills_the_new_slot_at_once() {
    let mut g = in_shop(22);
    g.state_mut().money = 30;
    offer(&mut g, VoucherId::Overstock);
    g.step(Action::BuyVoucher(0)).expect("buy voucher");
    assert_eq!(g.state().money, 20);
    assert_eq!(g.state().vouchers, vec![VoucherId::Overstock]);
    assert_eq!(g.state().shop.cards.iter().flatten().count(), 3);
    assert!(g.state().shop.vouchers[0].is_none());
    assert_eq!(g.state().ante_voucher, None);
    assert!(g.state().check_invariants().is_ok());
}

#[test]
fn the_ante_voucher_stays_until_bought_and_is_rolled_again_next_ante() {
    let mut g = in_shop(23);
    let v = g.state().ante_voucher.expect("a voucher is offered");
    assert_eq!(g.state().shop.vouchers[0].map(|s| s.id), Some(v));
    // Not bought: the same voucher is in the next shop of the ante.
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select big");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().shop.vouchers[0].map(|s| s.id), Some(v));
    // Bought: the slot is empty for the rest of the ante.
    g.state_mut().money = 50;
    g.step(Action::BuyVoucher(0)).expect("buy");
    assert_eq!(g.state().ante_voucher, None);
    g.step(Action::EndShop).expect("end");
    assert!(g.state().vouchers.contains(&v));
    // Beat the boss: next ante rolls a voucher that is not owned.
    g.step(Action::SelectBlind).expect("select boss");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b11111)).expect("win boss");
    assert_eq!(g.state().ante, 2);
    let nv = g.state().ante_voucher.expect("new voucher");
    assert_ne!(nv, v);
    assert!(
        nv.requires()
            .is_none_or(|b| g.state().vouchers.contains(&b))
    );
}

#[test]
fn voucher_picks_respect_ownership_and_the_base_requirement() {
    let mut s = shop_state(24);
    for _ in 0..200 {
        let v = shop::pick_voucher(&mut s, &[]).expect("eligible");
        assert!(v.requires().is_none(), "upgrades need their base: {v:?}");
    }
    s.vouchers.push(VoucherId::Overstock);
    s.shop_mods = balatro_core::state::fold_shop_mods(&s.vouchers);
    let picks: Vec<VoucherId> = (0..300)
        .map(|_| shop::pick_voucher(&mut s, &[]).expect("eligible"))
        .collect();
    assert!(!picks.contains(&VoucherId::Overstock));
    // Overstock Plus is only eligible when it is defined (Phase 5C); never before its base.
    let mut s2 = shop_state(24);
    for _ in 0..200 {
        assert_ne!(
            shop::pick_voucher(&mut s2, &[]),
            Some(VoucherId::OverstockPlus)
        );
    }
}

// ----- joker pools -------------------------------------------------------------------------------

#[test]
fn cavendish_needs_the_extinction_flag_and_gros_michel_leaves_after_it() {
    let mut s = shop_state(25);
    let ids = |s: &mut GameState| -> Vec<JokerId> {
        (0..4000)
            .filter_map(|_| shop::roll_joker(s, RngKey::Shop, Some(Rarity::Common), &[]))
            .map(|j| j.id)
            .collect()
    };
    let before = ids(&mut s);
    assert!(before.contains(&JokerId::GrosMichel));
    assert!(!before.contains(&JokerId::Cavendish));
    s.run_flags.push(RunFlag::GrosMichelExtinct);
    let after = ids(&mut s);
    assert!(after.contains(&JokerId::Cavendish));
    assert!(!after.contains(&JokerId::GrosMichel));
}

#[test]
fn owned_jokers_are_not_offered_unless_showman() {
    let mut s = shop_state(26);
    let commons: Vec<JokerId> = all_defs()
        .filter(|d| d.rarity == Rarity::Common)
        .map(|d| d.id)
        .collect();
    let keep = commons[7];
    for &id in commons.iter().filter(|&&id| id != keep) {
        if id == JokerId::Cavendish {
            continue;
        }
        let mut j = JokerInstance::new(0, id);
        j.uid = s.alloc_joker_uid();
        s.jokers.push(j);
    }
    s.joker_slots = 100;
    s.refresh_modifiers();
    for _ in 0..50 {
        let j =
            shop::roll_joker(&mut s, RngKey::Shop, Some(Rarity::Common), &[]).expect("one left");
        assert_eq!(j.id, keep);
    }
    // Items already in the shop count too.
    assert!(shop::roll_joker(&mut s, RngKey::Shop, Some(Rarity::Common), &[keep]).is_none());
    // Showman lifts the rule.
    let mut j = JokerInstance::new(0, JokerId::Showman);
    j.uid = s.alloc_joker_uid();
    s.jokers.push(j);
    s.refresh_modifiers();
    let seen: Vec<JokerId> = (0..400)
        .filter_map(|_| shop::roll_joker(&mut s, RngKey::Shop, Some(Rarity::Common), &[keep]))
        .map(|j| j.id)
        .collect();
    assert!(seen.iter().any(|&id| id != keep));
}

#[test]
fn legendary_jokers_come_only_from_an_explicit_request() {
    let mut s = shop_state(27);
    for _ in 0..3000 {
        let c = shop::roll_card(&mut s);
        if let ShopCardKind::Joker(j) = c.kind {
            assert_ne!(j.def().expect("def").rarity, Rarity::Legendary);
        }
    }
    let j =
        shop::roll_joker(&mut s, RngKey::Jokers, Some(Rarity::Legendary), &[]).expect("legendary");
    assert_eq!(j.def().expect("def").rarity, Rarity::Legendary);
}

// ----- distributions -------------------------------------------------------------------------------

/// Counts over `markets` generated markets: (type counts [joker, tarot, planet, other], rarity
/// counts [common, uncommon, rare], edition counts [base, foil, holo, poly, negative], pack
/// counts by spec index, cards, jokers, packs).
struct Tally {
    types: [u64; 4],
    rarity: [u64; 3],
    edition: [u64; 5],
    packs: [u64; 15],
    cards: u64,
    jokers: u64,
    packs_total: u64,
}

fn tally(markets: u64, seed: u64) -> Tally {
    let mut t = Tally {
        types: [0; 4],
        rarity: [0; 3],
        edition: [0; 5],
        packs: [0; 15],
        cards: 0,
        jokers: 0,
        packs_total: 0,
    };
    let mut s = shop_state(seed);
    for _ in 0..markets {
        let _ = shop::enter_shop(&mut s);
        for c in s.shop.cards.iter().flatten() {
            t.cards += 1;
            match c.kind {
                ShopCardKind::Joker(j) => {
                    t.types[0] += 1;
                    t.jokers += 1;
                    let r = j.def().expect("def").rarity;
                    t.rarity[r as usize] += 1;
                    t.edition[j.edition as usize] += 1;
                }
                ShopCardKind::Consumable(id) => match id.kind() {
                    ConsumableKind::Tarot => t.types[1] += 1,
                    ConsumableKind::Planet => t.types[2] += 1,
                    ConsumableKind::Spectral => t.types[3] += 1,
                },
                ShopCardKind::PlayingCard(_) => t.types[3] += 1,
            }
        }
        for p in s.shop.packs.iter().flatten() {
            t.packs[p.spec.index()] += 1;
            t.packs_total += 1;
        }
    }
    t
}

/// Every share is within `tol` percentage points of its expectation.
fn check(t: &Tally, tol: f64) {
    let pct = |n: u64, d: u64| 100.0 * n as f64 / d as f64;
    let near = |name: &str, got: f64, want: f64| {
        assert!(
            (got - want).abs() <= tol,
            "{name}: {got:.3}% vs expected {want:.3}% (tolerance {tol})"
        );
    };
    // Card types: Joker 20 / Tarot 4 / Planet 4.
    near("joker", pct(t.types[0], t.cards), 100.0 * 20.0 / 28.0);
    near("tarot", pct(t.types[1], t.cards), 100.0 * 4.0 / 28.0);
    near("planet", pct(t.types[2], t.cards), 100.0 * 4.0 / 28.0);
    assert_eq!(
        t.types[3], 0,
        "no spectrals or playing cards without their conditions"
    );
    // Rarity: 70 / 25 / 5.
    near("common", pct(t.rarity[0], t.jokers), 70.0);
    near("uncommon", pct(t.rarity[1], t.jokers), 25.0);
    near("rare", pct(t.rarity[2], t.jokers), 5.0);
    assert_eq!(t.rarity.iter().sum::<u64>(), t.jokers);
    // Editions: Negative 0.3, Poly 0.3, Holo 1.4, Foil 2.0.
    near("negative", pct(t.edition[4], t.jokers), 0.3);
    near("polychrome", pct(t.edition[3], t.jokers), 0.3);
    near("holographic", pct(t.edition[2], t.jokers), 1.4);
    near("foil", pct(t.edition[1], t.jokers), 2.0);
    // Packs: 22.42 total weight.
    for kind in PackKind::ALL {
        for size in PackSize::ALL {
            let spec = PackSpec { kind, size };
            let want = 100.0 * pack_weight(kind, size) / 22.42;
            near(
                &format!("{kind:?} {size:?}"),
                pct(t.packs[spec.index()], t.packs_total),
                want,
            );
        }
    }
}

#[test]
fn market_distribution_100k() {
    let t = tally(100_000, 31);
    assert_eq!(t.cards, 200_000);
    assert_eq!(t.packs_total, 200_000);
    check(&t, 0.6);
}

#[test]
#[ignore = "1M markets; run with `cargo test --release -p balatro-core --test shop -- --ignored`"]
fn market_distribution_1m() {
    let t = tally(1_000_000, 32);
    check(&t, 0.15);
}

#[test]
fn hone_and_glow_up_scale_the_edition_odds_but_not_negative() {
    let mut s = shop_state(33);
    let n = 200_000u64;
    for (m, foil, holo, poly) in [(2u32, 4.0, 2.8, 0.9), (4, 8.0, 5.6, 2.1)] {
        s.shop_mods.edition_mult = m;
        let mut c = [0u64; 5];
        for _ in 0..n {
            let j = shop::roll_joker(&mut s, RngKey::Shop, None, &[]).expect("joker");
            c[j.edition as usize] += 1;
        }
        let pct = |i: usize| 100.0 * c[i] as f64 / n as f64;
        for (name, got, want) in [
            ("foil", pct(1), foil),
            ("holo", pct(2), holo),
            ("poly", pct(3), poly),
            ("negative", pct(4), 0.3),
        ] {
            assert!((got - want).abs() < 0.35, "x{m} {name}: {got} vs {want}");
        }
    }
}

#[test]
fn tarot_and_planet_vouchers_shift_the_type_weights() {
    let mut s = shop_state(34);
    s.shop_mods.tarot_weight = Some(32.0);
    let w = shop::card_type_weights(&s);
    assert_eq!(w, [20.0, 32.0, 4.0, 0.0, 0.0]);
    s.shop_mods.planet_weight = Some(9.6);
    s.shop_mods.playing_cards = true;
    assert_eq!(shop::card_type_weights(&s), [20.0, 32.0, 9.6, 4.0, 0.0]);
    let n = 60_000;
    let tarots = (0..n)
        .filter(|_| {
            matches!(
                shop::roll_card(&mut s).kind,
                ShopCardKind::Consumable(id) if id.kind() == ConsumableKind::Tarot
            )
        })
        .count();
    let want = 100.0 * 32.0 / 65.6;
    let got = 100.0 * tarots as f64 / f64::from(n);
    assert!((got - want).abs() < 1.0, "{got} vs {want}");
}

#[test]
fn magic_trick_and_illusion_decide_what_a_shop_playing_card_looks_like() {
    let mut s = shop_state(35);
    s.shop_mods.playing_cards = true;
    let mut plain = 0;
    let mut total = 0;
    for _ in 0..5000 {
        if let ShopCardKind::PlayingCard(c) = shop::roll_card(&mut s).kind {
            total += 1;
            if c.enhancement == balatro_core::enums::Enhancement::None
                && c.seal == balatro_core::enums::Seal::None
                && c.edition == Edition::Base
            {
                plain += 1;
            }
        }
    }
    assert!(total > 300);
    assert_eq!(
        plain, total,
        "without Illusion shop playing cards are plain"
    );
    s.shop_mods.illusion = true;
    let mut enhanced = 0;
    let mut total = 0;
    for _ in 0..8000 {
        if let ShopCardKind::PlayingCard(c) = shop::roll_card(&mut s).kind {
            total += 1;
            if c.enhancement != balatro_core::enums::Enhancement::None {
                enhanced += 1;
            }
        }
    }
    let frac = f64::from(enhanced) / f64::from(total);
    assert!((frac - 0.4).abs() < 0.06, "{frac}");
}

// ----- whole-run behaviour ---------------------------------------------------------------------------

#[test]
fn every_legal_shop_action_steps_ok_and_keeps_the_invariants() {
    for seed in 0..40 {
        let mut g = in_shop(seed);
        g.state_mut().money = 60;
        for _ in 0..4 {
            let legal = g.legal_actions();
            assert!(!legal.is_empty());
            for a in &legal {
                let mut c = g.clone();
                c.step(*a)
                    .unwrap_or_else(|e| panic!("seed {seed}: {a:?}: {e}"));
                c.state().check_invariants().expect("invariants");
            }
            // Take the first non-EndShop action to move the shop along.
            let next = legal.iter().copied().find(|a| {
                *a != Action::EndShop
                    && !matches!(
                        a,
                        Action::SellJoker(_) | Action::MoveJoker { .. } | Action::SellConsumable(_)
                    )
            });
            match next {
                Some(a) => {
                    g.step(a).expect("step");
                }
                None => break,
            }
            if g.state().phase == Phase::PackOpen {
                g.step(Action::SkipPack).expect("skip");
            }
        }
    }
}

#[test]
fn random_agents_play_the_shop_and_stay_deterministic() {
    for seed in 0..60u64 {
        let run = |seed| {
            let mut g = new_game(seed);
            let mut agent = RandomAgent::with_play_weight(seed, 20.0);
            let mut n = 0;
            while !g.is_terminal() && n < 3000 {
                let a = agent.act(&g);
                // Nudge so antes (and shops) are reached.
                if matches!(a, Action::Play(_)) && seed % 2 == 0 {
                    let t = g.state().target;
                    g.state_mut().round_score = t - 1.0;
                }
                g.step(a).expect("legal");
                g.state().check_invariants().expect("invariants");
                n += 1;
            }
            g.into_state()
        };
        assert_eq!(run(seed), run(seed), "seed {seed}");
    }
}
