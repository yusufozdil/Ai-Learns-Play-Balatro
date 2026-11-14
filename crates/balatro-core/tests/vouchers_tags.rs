//! Phase 5.0 voucher and tag contract + prototypes: ids, requirements, redeeming (passive
//! modifiers, one-shot effects, price/sell re-scaling), boss rerolls, tag gain/consumption
//! timing (immediate, next shop, after the boss), Double Tag, tag rolls.
//! Guide: docs/design/shop_packs_vouchers_tags.md.

use balatro_core::actions::{self, Action, ActionError};
use balatro_core::blinds::BlindKind;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{
    TagId, VoucherId as V, try_tag_def, try_voucher_def, voucher_def_count,
};
use balatro_core::economy::Discount;
use balatro_core::enums::{HandType, Phase};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::shop::{self, ShopCard, ShopCardKind};
use balatro_core::state::{GameConfig, GameState, fold_shop_mods};
use balatro_core::voucher::{self, BossReroll, ShopMods, VoucherCtx};

// ----- helpers ---------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn win(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b11111)).expect("winning play");
}

/// Beat the Small blind: Shop phase.
fn in_shop(seed: u64) -> Game {
    let mut g = new_game(seed);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

/// Beat Small and Big, end up at the Boss blind select.
fn at_boss_select(seed: u64) -> Game {
    let mut g = new_game(seed);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win(&mut g);
        g.step(Action::EndShop).expect("end shop");
    }
    assert_eq!(
        (g.state().phase, g.state().blind),
        (Phase::BlindSelect, BlindKind::Boss)
    );
    g
}

fn redeem(g: &mut Game, id: V) {
    voucher::redeem(g.state_mut(), id);
}

// ----- ids ---------------------------------------------------------------------------------------

#[test]
fn there_are_32_vouchers_in_16_base_upgrade_pairs() {
    assert_eq!(V::COUNT, 32);
    let mut bases = 0;
    for id in V::ALL {
        match id.requires() {
            None => {
                bases += 1;
                assert!(!id.is_upgrade());
            }
            Some(base) => {
                assert!(id.is_upgrade());
                assert_eq!(base.index() + 1, id.index());
                assert!(base.requires().is_none());
            }
        }
    }
    assert_eq!(bases, 16);
    let pairs = [
        (V::OverstockPlus, V::Overstock),
        (V::Liquidation, V::ClearanceSale),
        (V::GlowUp, V::Hone),
        (V::RerollGlut, V::RerollSurplus),
        (V::OmenGlobe, V::CrystalBall),
        (V::Observatory, V::Telescope),
        (V::NachoTong, V::Grabber),
        (V::Recyclomancy, V::Wasteful),
        (V::TarotTycoon, V::TarotMerchant),
        (V::PlanetTycoon, V::PlanetMerchant),
        (V::MoneyTree, V::SeedMoney),
        (V::Antimatter, V::Blank),
        (V::Illusion, V::MagicTrick),
        (V::Petroglyph, V::Hieroglyph),
        (V::Retcon, V::DirectorsCut),
        (V::Palette, V::PaintBrush),
    ];
    for (up, base) in pairs {
        assert_eq!(up.requires(), Some(base), "{up:?}");
    }
}

#[test]
fn there_are_24_tags_and_double_is_index_17() {
    assert_eq!(TagId::COUNT, 24);
    assert_eq!(TagId::Double.index(), 17);
    assert_eq!(TagId::Economy.index(), 23);
}

#[test]
fn voucher_defs_are_unique_and_every_defined_id_is_registered() {
    for id in V::ALL {
        assert!(voucher_def_count(id) <= 1, "{id:?}");
        if let Some(d) = try_voucher_def(id) {
            assert_eq!(d.id, id);
            assert_eq!(d.cost, 10, "every voucher costs $10");
            assert!(!d.name.is_empty());
        }
    }
    for id in [
        V::Overstock,
        V::ClearanceSale,
        V::Grabber,
        V::SeedMoney,
        V::CrystalBall,
        V::Telescope,
        V::DirectorsCut,
        V::Retcon,
    ] {
        assert!(try_voucher_def(id).is_some(), "{id:?} is a prototype");
    }
    for id in TagId::ALL {
        if let Some(d) = try_tag_def(id) {
            assert_eq!(d.id, id);
            assert!(d.min_ante == 1 || d.min_ante == 2, "{id:?}");
        }
    }
}

// ----- passive modifiers and one-shot effects ---------------------------------------------------------

#[test]
fn shop_mods_merge_sums_slots_and_takes_the_best_of_the_rest() {
    let mut a = ShopMods {
        card_slots: 1,
        reroll_base: Some(3),
        tarot_weight: Some(9.6),
        edition_mult: 2,
        discount: Discount::Clearance,
        boss_reroll: BossReroll::OncePerAnte,
        ..ShopMods::NONE
    };
    let b = ShopMods {
        card_slots: 1,
        reroll_base: Some(1),
        tarot_weight: Some(32.0),
        edition_mult: 4,
        discount: Discount::Liquidation,
        boss_reroll: BossReroll::Unlimited,
        telescope: true,
        ..ShopMods::NONE
    };
    a.merge(&b);
    assert_eq!(a.card_slots, 2);
    assert_eq!(a.reroll_base, Some(1));
    assert_eq!(a.tarot_weight, Some(32.0));
    assert_eq!(a.edition_mult, 4);
    assert_eq!(a.discount, Discount::Liquidation);
    assert_eq!(a.boss_reroll, BossReroll::Unlimited);
    assert!(a.telescope);
    assert_eq!(ShopMods::default(), ShopMods::NONE);
}

#[test]
fn redeeming_records_the_voucher_and_refolds_the_cache() {
    let mut g = in_shop(1);
    redeem(&mut g, V::Overstock);
    redeem(&mut g, V::Telescope);
    assert_eq!(g.state().vouchers, vec![V::Overstock, V::Telescope]);
    assert_eq!(g.state().shop_mods.card_slots, 1);
    assert!(g.state().shop_mods.telescope);
    assert_eq!(g.state().shop_mods, fold_shop_mods(&g.state().vouchers));
    g.state().check_invariants().expect("invariants");
    // A cache that disagrees with the owned vouchers is an invariant violation.
    let mut bad = g.state().clone();
    bad.shop_mods.card_slots = 5;
    assert!(bad.check_invariants().is_err());
}

#[test]
fn overstock_adds_a_card_slot_to_every_later_shop() {
    let mut g = in_shop(2);
    g.state_mut().money = 30;
    g.state_mut().shop.vouchers = vec![Some(shop::ShopVoucher {
        id: V::Overstock,
        from_ante: false,
    })];
    g.step(Action::BuyVoucher(0)).expect("buy");
    assert_eq!(g.state().shop.cards.iter().flatten().count(), 3);
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert_eq!(g.state().shop.cards.iter().flatten().count(), 3);
    assert_eq!(g.state().shop.base_slots, 3);
}

#[test]
fn clearance_sale_discounts_cards_packs_and_vouchers_and_rescales_sell_values() {
    let mut g = in_shop(3);
    // A Rare joker of base $8: sell value 4 before, floor(6/2) = 3 after (price 8 -> 6).
    let mut j = JokerInstance::new(0, JokerId::Blueprint);
    j.sell_value = 4;
    let base = j.def().expect("def").base_cost;
    g.add_joker(j).expect("room");
    let old_price = i32::try_from(base).expect("small");
    redeem(&mut g, V::ClearanceSale);
    assert_eq!(g.state().shop_mods.discount, Discount::Clearance);
    let expect = balatro_core::economy::sell_value(
        balatro_core::economy::shop_price(old_price, 0, Discount::Clearance),
        0,
    );
    assert_eq!(g.state().jokers[0].sell_value, expect);
    assert!(expect < 4);
    // Prices: Tarot $3 -> floor(7 * 75 / 200) = 2; Mega pack $8 -> 6; voucher $10 -> 8 (7.5 -> 7?).
    let tarot = ShopCard {
        kind: ShopCardKind::Consumable(balatro_core::content::ConsumableId::TheFool),
        free: false,
    };
    assert_eq!(shop::card_price(g.state(), &tarot), 2);
    assert_eq!(shop::voucher_price(g.state(), V::Overstock), 7);
    let pack = shop::ShopPack {
        spec: balatro_core::packs::PackSpec {
            kind: balatro_core::hooks::PackKind::Arcana,
            size: balatro_core::packs::PackSize::Mega,
        },
        free: false,
    };
    assert_eq!(shop::pack_price_of(g.state(), &pack), 6);
}

#[test]
fn grabber_gives_a_hand_crystal_ball_a_slot_seed_money_a_higher_interest_cap() {
    let mut g = in_shop(4);
    redeem(&mut g, V::Grabber);
    redeem(&mut g, V::CrystalBall);
    redeem(&mut g, V::SeedMoney);
    let s = g.state();
    assert_eq!(s.base_hands, 5);
    assert_eq!(s.consumable_slots, 3);
    assert_eq!(s.interest_cap, 10);
    s.check_invariants().expect("invariants");
    // The extra hand shows up when the next round starts.
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    assert_eq!(g.state().hands_left, 5);
    // Interest: $50 held -> $10 with Seed Money.
    let mut g = in_shop(4);
    redeem(&mut g, V::SeedMoney);
    g.state_mut().money = 55;
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    g.state_mut().money = 55;
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let info = g.step(Action::Play(1)).expect("win");
    let co = info.round.expect("round").cash_out.expect("cash out");
    assert_eq!(co.interest, 10);
}

#[test]
fn voucher_ctx_changes_are_clamped() {
    let mut s = GameState::new(5, GameConfig::default());
    let ante0 = s.ante;
    let target0 = s.target;
    {
        let mut c = VoucherCtx::new(&mut s);
        c.add_hands(-10);
        c.add_discards(-10);
        c.add_hand_size(-100);
        c.add_joker_slots(1);
        c.raise_interest_cap(3); // never lowers
        c.add_ante(-1);
    }
    assert_eq!(s.base_hands, 1);
    assert_eq!(s.base_discards, 0);
    assert_eq!(s.base_hand_size, 1);
    assert_eq!(s.joker_slots, 6);
    assert_eq!(s.interest_cap, 5);
    assert_eq!(s.ante, ante0 - 1);
    assert_eq!(s.target, 100.0 * 1.0, "ante 0 uses the 100 base chips");
    assert!(s.target < target0);
}

#[test]
fn telescope_makes_the_first_celestial_item_the_most_played_hands_planet() {
    let mut g = in_shop(6);
    redeem(&mut g, V::Telescope);
    g.state_mut().hand_plays_run[HandType::Straight.index()] = 7;
    g.state_mut().money = 50;
    g.state_mut().shop.packs[1] = Some(shop::ShopPack {
        spec: balatro_core::packs::PackSpec {
            kind: balatro_core::hooks::PackKind::Celestial,
            size: balatro_core::packs::PackSize::Normal,
        },
        free: false,
    });
    g.step(Action::OpenPack(1)).expect("open");
    let first = g.state().pack.as_ref().expect("pack").items[0];
    assert_eq!(
        first,
        Some(balatro_core::packs::PackItem::Consumable(
            balatro_core::content::ConsumableId::Saturn
        ))
    );
}

// ----- boss rerolls -----------------------------------------------------------------------------------

#[test]
fn directors_cut_rerolls_the_boss_once_per_ante_for_ten_dollars() {
    let mut g = at_boss_select(7);
    assert!(!g.legal_actions().contains(&Action::RerollBoss));
    assert_eq!(
        g.step(Action::RerollBoss),
        Err(ActionError::BossRerollUnavailable)
    );
    redeem(&mut g, V::DirectorsCut);
    g.state_mut().money = 9;
    assert!(matches!(
        g.step(Action::RerollBoss),
        Err(ActionError::CannotAfford { cost: 10, money: 9 })
    ));
    g.state_mut().money = 25;
    let before = g.state().boss.expect("boss");
    let seen_before = g.state().boss_seen;
    assert!(g.legal_actions().contains(&Action::RerollBoss));
    let info = g.step(Action::RerollBoss).expect("reroll");
    assert_eq!(info.spent, Some(10));
    assert_eq!(g.state().money, 15);
    let after = g.state().boss.expect("boss");
    assert_ne!(before, after);
    // The old boss counts as unseen again, the new one as seen.
    assert_eq!(
        g.state().boss_seen[before.index()],
        seen_before[before.index()] - 1
    );
    assert_eq!(
        g.state().boss_seen[after.index()],
        seen_before[after.index()] + 1
    );
    // Once per ante.
    assert_eq!(
        g.step(Action::RerollBoss),
        Err(ActionError::BossRerollUnavailable)
    );
    // The requirement follows the new boss.
    let expected = balatro_core::blinds::requirement(
        g.state().ante,
        BlindKind::Boss,
        g.state().boss_mult(),
        g.state().config.stake.scaling(),
    );
    assert_eq!(g.state().target, expected);
    g.state().check_invariants().expect("invariants");
}

#[test]
fn retcon_rerolls_without_limit() {
    let mut g = at_boss_select(8);
    redeem(&mut g, V::DirectorsCut);
    redeem(&mut g, V::Retcon);
    g.state_mut().money = 100;
    for n in 1..=4 {
        g.step(Action::RerollBoss).expect("reroll");
        assert_eq!(g.state().money, 100 - 10 * n);
    }
    // Rerolling is a BlindSelect action only.
    g.step(Action::SelectBlind).expect("select");
    assert!(matches!(
        g.step(Action::RerollBoss),
        Err(ActionError::WrongPhase { .. })
    ));
}

#[test]
fn the_per_ante_reroll_count_resets_with_the_ante() {
    let mut g = at_boss_select(9);
    redeem(&mut g, V::DirectorsCut);
    g.state_mut().money = 100;
    g.step(Action::RerollBoss).expect("reroll");
    assert_eq!(g.state().boss_rerolls_ante, 1);
    g.step(Action::SelectBlind).expect("select boss");
    win(&mut g);
    assert_eq!(g.state().ante, 2);
    assert_eq!(g.state().boss_rerolls_ante, 0);
}

// ----- tags: rolls ------------------------------------------------------------------------------------

#[test]
fn offered_tags_are_defined_and_respect_the_min_ante() {
    for seed in 0..200 {
        let s = GameState::new(seed, GameConfig::default());
        for (i, t) in s.blind_tags.iter().enumerate() {
            let d = try_tag_def(*t).expect("only defined tags are offered while any is defined");
            assert!(d.min_ante <= s.ante.max(1));
            assert_eq!(s.blind_tag_hands[i], None, "only Orbital carries a hand");
        }
    }
}

#[test]
fn secret_hands_are_the_three_top_hands() {
    let secret: Vec<HandType> = HandType::ALL
        .into_iter()
        .filter(|h| h.is_secret())
        .collect();
    assert_eq!(
        secret,
        vec![
            HandType::FiveOfAKind,
            HandType::FlushHouse,
            HandType::FlushFive
        ]
    );
}

// ----- tags: effects ----------------------------------------------------------------------------------

fn skip_with(g: &mut Game, tag: TagId) {
    g.state_mut().blind_tags[0] = tag;
    g.state_mut().blind_tags[1] = tag;
    g.step(Action::SkipBlind).expect("skip");
}

#[test]
fn economy_doubles_money_up_to_forty_and_is_used_up() {
    for (money, gain) in [(4, 4), (30, 30), (100, 40), (0, 0), (-5, 0)] {
        let mut g = new_game(10);
        g.state_mut().money = money;
        skip_with(&mut g, TagId::Economy);
        assert_eq!(g.state().money, money + gain, "money {money}");
        assert!(g.state().tags.is_empty());
    }
}

#[test]
fn investment_pays_after_the_boss_cash_out_and_only_then() {
    let mut g = at_boss_select(11);
    g.state_mut().tags.push(TagId::Investment);
    g.step(Action::SelectBlind).expect("boss");
    g.state_mut().money = 6;
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let info = g.step(Action::Play(0b11111)).expect("win");
    let co = info.round.expect("round").cash_out.expect("cash out");
    assert_eq!(g.state().money, 6 + co.total + 25);
    assert!(g.state().tags.is_empty());
    // Not paid for Small/Big wins: the tag stays until a boss falls.
    let mut g = new_game(11);
    g.state_mut().tags.push(TagId::Investment);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g);
    assert_eq!(g.state().tags, vec![TagId::Investment]);
}

#[test]
fn coupon_makes_the_next_shops_cards_and_packs_free_but_not_the_voucher() {
    let mut g = new_game(12);
    skip_with(&mut g, TagId::Coupon);
    assert_eq!(g.state().tags, vec![TagId::Coupon], "waits for a shop");
    g.step(Action::SkipBlind).expect("skip big");
    // Two Coupons held now (both skips gave one): the first shop uses both flags the same way.
    g.step(Action::SelectBlind).expect("boss");
    win(&mut g);
    let s = g.state();
    assert!(s.tags.is_empty(), "all coupons are used up by that shop");
    for c in s.shop.cards.iter().flatten() {
        assert!(c.free);
        assert_eq!(shop::card_price(s, c), 0);
    }
    for p in s.shop.packs.iter().flatten() {
        assert_eq!(shop::pack_price_of(s, p), 0);
    }
    let v = s.shop.vouchers[0].expect("voucher");
    assert_eq!(shop::voucher_price(s, v.id), 10);
    // Buying a free item costs nothing and sells for $1.
    let money = s.money;
    let slot = s
        .shop
        .cards
        .iter()
        .position(Option::is_some)
        .expect("a card");
    let n_before = g.state().jokers.len() + g.state().consumables.len();
    g.step(Action::Buy(slot as u8)).expect("buy free");
    assert_eq!(g.state().money, money);
    assert!(
        g.state().jokers.len() + g.state().consumables.len() > n_before
            || g.state().deck.len() > 52
    );
}

#[test]
fn double_copies_the_next_tag_and_is_used_up() {
    let mut g = new_game(13);
    g.state_mut().tags.push(TagId::Double);
    g.state_mut().money = 10;
    skip_with(&mut g, TagId::Economy);
    // Economy twice: 10 -> 20 -> 40.
    assert_eq!(g.state().money, 40);
    assert!(
        g.state().tags.is_empty(),
        "Double and both Economy tags are gone"
    );
    // A held tag copy: Investment x2.
    let mut g = new_game(13);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Investment);
    assert_eq!(g.state().tags, vec![TagId::Investment, TagId::Investment]);
}

#[test]
fn double_does_not_copy_double() {
    let mut g = new_game(14);
    g.state_mut().tags.push(TagId::Double);
    skip_with(&mut g, TagId::Double);
    assert_eq!(g.state().tags, vec![TagId::Double, TagId::Double]);
}

#[test]
fn diet_cola_makes_a_double_tag_that_copies_the_next_skip() {
    let mut g = in_shop(15);
    let mut j = JokerInstance::new(0, JokerId::DietCola);
    j.sell_value = 3;
    g.add_joker(j).expect("room");
    g.step(Action::SellJoker(0)).expect("sell");
    assert_eq!(g.state().tags, vec![TagId::Double]);
    g.step(Action::EndShop).expect("end");
    g.state_mut().money = 7;
    skip_with(&mut g, TagId::Economy);
    assert_eq!(g.state().money, 7 + 7 + 14);
}

#[test]
fn the_boss_tag_rerolls_the_boss_at_once() {
    let mut g = new_game(16);
    let before = g.state().boss.expect("boss");
    skip_with(&mut g, TagId::Boss);
    let after = g.state().boss.expect("boss");
    assert_ne!(before, after);
    assert!(g.state().tags.is_empty());
    // Skipping twice (Small then Big) rerolled twice; the seen counters stay consistent.
    assert_eq!(g.state().boss_seen.iter().sum::<u32>(), 1);
    g.state().check_invariants().expect("invariants");
}

#[test]
fn juggle_style_round_bonuses_last_one_round() {
    let mut g = new_game(17);
    g.state_mut().round_hand_size = 3;
    g.step(Action::SelectBlind).expect("select");
    assert_eq!(g.state().deck.hand.len(), 11);
    win(&mut g);
    assert_eq!(g.state().round_hand_size, 0);
}

#[test]
fn unused_discards_are_summed_over_won_rounds() {
    let mut g = new_game(18);
    g.step(Action::SelectBlind).expect("select");
    win(&mut g); // 4 discards unused (Red Deck)
    assert_eq!(g.state().stats.unused_discards, 4);
}

#[test]
fn actions_validate_purely_and_agree_with_legal_actions() {
    for seed in 0..30 {
        let g = in_shop(seed);
        let s = g.state();
        let legal = g.legal_actions();
        for i in 0..4u8 {
            for a in [
                Action::Buy(i),
                Action::BuyAndUse(i),
                Action::OpenPack(i),
                Action::BuyVoucher(i),
            ] {
                assert_eq!(actions::validate(s, a).is_ok(), legal.contains(&a), "{a:?}");
            }
        }
        assert_eq!(
            actions::validate(s, Action::Reroll).is_ok(),
            legal.contains(&Action::Reroll)
        );
    }
}
