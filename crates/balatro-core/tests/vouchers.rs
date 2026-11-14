//! Phase 5C vouchers: every voucher defined, its effect at `Game::step` level (or on a shop state
//! for pure roll statistics), upgrades gated by their base, cache == fold, determinism.
//! Guide: docs/design/shop_packs_vouchers_tags.md.

use balatro_core::actions::Action;
use balatro_core::consumable::{ConsumableInstance, ConsumableKind};
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{ConsumableId as C, VoucherId as V, try_voucher_def};
use balatro_core::economy::{self, Discount};
use balatro_core::enums::{Edition, HandType, Phase};
use balatro_core::game::Game;
use balatro_core::hooks::PackKind;
use balatro_core::joker::JokerInstance;
use balatro_core::packs::{PackItem, PackSize, PackSpec};
use balatro_core::shop::{self, ShopCard, ShopCardKind, ShopPack, ShopVoucher};
use balatro_core::state::{GameConfig, GameState, fold_shop_mods};
use balatro_core::voucher;

// ----- helpers ---------------------------------------------------------------------------------

fn in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    g
}

fn in_shop(seed: u64) -> Game {
    let mut g = in_play(seed);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

fn shop_state(seed: u64) -> GameState {
    let mut s = GameState::new(seed, GameConfig::default());
    s.phase = Phase::Shop;
    s.stats.shops_entered = 1;
    s
}

fn redeem(g: &mut Game, id: V) {
    voucher::redeem(g.state_mut(), id);
}

/// Own the base (if any) through `redeem`, then buy `id` from the shop slot with `Game::step`.
fn buy(g: &mut Game, id: V) {
    if let Some(base) = id.requires()
        && !g.state().vouchers.contains(&base)
    {
        redeem(g, base);
    }
    g.state_mut().money = 100;
    g.state_mut().ante_voucher = Some(id);
    g.state_mut().shop.vouchers = vec![Some(ShopVoucher {
        id,
        from_ante: true,
    })];
    g.step(Action::BuyVoucher(0)).expect("buy voucher");
    assert!(g.state().vouchers.contains(&id));
    g.state().check_invariants().expect("invariants");
}

fn redeem_state(s: &mut GameState, ids: &[V]) {
    for &id in ids {
        voucher::redeem(s, id);
    }
}

fn next_round(g: &mut Game) {
    g.step(Action::EndShop).expect("end shop");
    g.step(Action::SelectBlind).expect("select");
}

fn card_count(g: &Game) -> usize {
    g.state().shop.cards.iter().flatten().count()
}

// ----- registry -------------------------------------------------------------------------------

#[test]
fn all_32_vouchers_are_defined_and_cost_10() {
    for id in V::ALL {
        let d = try_voucher_def(id).unwrap_or_else(|| panic!("{id:?} undefined"));
        assert_eq!(d.id, id);
        assert_eq!(d.cost, 10);
    }
}

#[test]
fn every_voucher_can_be_bought_through_the_game_and_keeps_the_cache_equal_to_the_fold() {
    for id in V::ALL {
        let mut g = in_shop(3);
        buy(&mut g, id);
        let s = g.state();
        assert_eq!(s.shop_mods, fold_shop_mods(&s.vouchers), "{id:?}");
        assert!(s.shop.vouchers[0].is_none(), "{id:?}");
        assert_eq!(s.ante_voucher, None);
    }
}

#[test]
fn an_upgrade_is_offered_only_after_its_base() {
    for up in V::ALL.into_iter().filter(|v| v.is_upgrade()) {
        let base = up.requires().expect("base");
        let mut s = shop_state(5);
        for _ in 0..400 {
            let v = shop::pick_voucher(&mut s, &[]).expect("eligible");
            assert!(v.requires().is_none(), "{v:?} offered without its base");
        }
        let mut s = shop_state(5);
        redeem_state(&mut s, &[base]);
        let seen = (0..800).any(|_| shop::pick_voucher(&mut s, &[]) == Some(up));
        assert!(seen, "{up:?} never offered after {base:?}");
        redeem_state(&mut s, &[up]);
        assert!((0..300).all(|_| shop::pick_voucher(&mut s, &[]) != Some(up)));
    }
}

// ----- shop layout and prices --------------------------------------------------------------------

#[test]
fn overstock_plus_makes_four_card_slots() {
    let mut g = in_shop(7);
    assert_eq!(card_count(&g), 2);
    buy(&mut g, V::Overstock);
    assert_eq!(card_count(&g), 3);
    buy(&mut g, V::OverstockPlus);
    assert_eq!(card_count(&g), 4);
    assert_eq!(shop::card_slot_count(g.state()), 4);
    g.step(Action::Reroll).expect("reroll");
    assert_eq!(card_count(&g), 4);
}

#[test]
fn liquidation_halves_prices_and_reprices_owned_jokers() {
    let mut g = in_shop(8);
    let mut j = JokerInstance::new(0, JokerId::Blueprint);
    j.sell_value = economy::sell_value(economy::shop_price(10, 0, Discount::None), 0);
    let slot = g.add_joker(j).expect("room");
    let old = g.state().jokers[slot].sell_value;
    redeem(&mut g, V::ClearanceSale);
    redeem(&mut g, V::Liquidation);
    assert_eq!(g.state().shop_mods.discount, Discount::Liquidation);
    assert!(g.state().jokers[slot].sell_value < old);
    assert!(g.state().jokers[slot].sell_value >= 1);
    // $10 voucher costs $5; packs: $4 -> $2, $8 -> $4.
    assert_eq!(shop::voucher_price(g.state(), V::Hone), 5);
    let pack = |size| ShopPack {
        spec: PackSpec {
            kind: PackKind::Arcana,
            size,
        },
        free: false,
    };
    assert_eq!(shop::pack_price_of(g.state(), &pack(PackSize::Normal)), 2);
    assert_eq!(shop::pack_price_of(g.state(), &pack(PackSize::Mega)), 4);
    // A $10 card costs $5 (Planet/Tarot base 3 -> 1 or 2; check a Spectral-priced one by math).
    let card = ShopCard {
        kind: ShopCardKind::Consumable(C::Wraith),
        free: false,
    };
    let base = shop::card_base_cost(&card.kind);
    assert_eq!(
        shop::card_price(g.state(), &card),
        economy::shop_price(base, 0, Discount::Liquidation)
    );
    assert!(shop::card_price(g.state(), &card) * 2 <= base + 1);
    g.state().check_invariants().expect("invariants");
}

#[test]
fn reroll_surplus_and_glut_lower_the_reroll_price() {
    let mut g = in_shop(9);
    g.state_mut().money = 100;
    assert_eq!(shop::reroll_price(g.state()), 5);
    buy(&mut g, V::RerollSurplus);
    assert_eq!(shop::reroll_price(g.state()), 3);
    let info = g.step(Action::Reroll).expect("reroll");
    assert_eq!(info.spent, Some(3));
    assert_eq!(shop::reroll_price(g.state()), 4);
    buy(&mut g, V::RerollGlut);
    // One paid reroll already: base 1 + 1.
    assert_eq!(shop::reroll_price(g.state()), 2);
    g.state_mut().money = 100;
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    assert_eq!(shop::reroll_price(g.state()), 1, "new shop starts at $1");
}

// ----- rolls (statistical) -----------------------------------------------------------------------

fn edition_rate(vs: &[V], n: usize) -> (f64, f64, f64) {
    let mut s = shop_state(11);
    redeem_state(&mut s, vs);
    let (mut f, mut h, mut p) = (0usize, 0usize, 0usize);
    for _ in 0..n {
        let c = shop::random_playing_card(
            &mut s.rng,
            balatro_core::rng::RngKey::Shop,
            0.0,
            0.0,
            s.shop_mods.edition_mult,
        );
        match c.edition {
            Edition::Foil => f += 1,
            Edition::Holographic => h += 1,
            Edition::Polychrome => p += 1,
            _ => {}
        }
    }
    let n = n as f64;
    (f as f64 / n, h as f64 / n, p as f64 / n)
}

#[test]
fn hone_doubles_and_glow_up_quadruples_edition_odds() {
    let n = 100_000;
    let (f0, h0, p0) = edition_rate(&[], n);
    let (f1, h1, p1) = edition_rate(&[V::Hone], n);
    let (f2, h2, p2) = edition_rate(&[V::Hone, V::GlowUp], n);
    let close = |a: f64, b: f64| (a - b).abs() < 0.004;
    assert!(close(f0, 0.04) && close(h0, 0.028) && close(p0, 0.012));
    assert!(close(f1, 0.08) && close(h1, 0.056) && close(p1, 0.024));
    assert!(close(f2, 0.16) && close(h2, 0.112) && close(p2, 0.048));
}

#[test]
fn hone_and_glow_up_also_scale_shop_joker_editions() {
    let rate = |vs: &[V]| {
        let mut s = shop_state(12);
        redeem_state(&mut s, vs);
        let mut non_base = 0;
        for _ in 0..20_000 {
            if let ShopCardKind::Joker(j) = shop::roll_card(&mut s).kind
                && j.edition != Edition::Base
            {
                non_base += 1;
            }
        }
        f64::from(non_base)
    };
    let (a, b, c) = (rate(&[]), rate(&[V::Hone]), rate(&[V::Hone, V::GlowUp]));
    assert!(b > a * 1.5, "{a} {b}");
    assert!(c > b * 1.5, "{b} {c}");
}

fn type_share(vs: &[V], kind: ConsumableKind) -> f64 {
    let mut s = shop_state(13);
    redeem_state(&mut s, vs);
    let n = 20_000;
    let mut hits = 0;
    for _ in 0..n {
        if let ShopCardKind::Consumable(id) = shop::roll_card(&mut s).kind
            && id.kind() == kind
        {
            hits += 1;
        }
    }
    f64::from(hits) / f64::from(n)
}

#[test]
fn tarot_merchant_and_tycoon_raise_the_tarot_share() {
    // Weights: joker 20, tarot 4 / 9.6 / 32, planet 4.
    let near = |a: f64, b: f64| (a - b).abs() < 0.02;
    assert!(near(type_share(&[], ConsumableKind::Tarot), 4.0 / 28.0));
    assert!(near(
        type_share(&[V::TarotMerchant], ConsumableKind::Tarot),
        9.6 / 33.6
    ));
    assert!(near(
        type_share(&[V::TarotMerchant, V::TarotTycoon], ConsumableKind::Tarot),
        32.0 / 56.0
    ));
}

#[test]
fn planet_merchant_and_tycoon_raise_the_planet_share() {
    let near = |a: f64, b: f64| (a - b).abs() < 0.02;
    assert!(near(type_share(&[], ConsumableKind::Planet), 4.0 / 28.0));
    assert!(near(
        type_share(&[V::PlanetMerchant], ConsumableKind::Planet),
        9.6 / 33.6
    ));
    assert!(near(
        type_share(
            &[V::PlanetMerchant, V::PlanetTycoon],
            ConsumableKind::Planet
        ),
        32.0 / 56.0
    ));
}

#[test]
fn magic_trick_puts_playing_cards_in_the_shop_and_illusion_decorates_them() {
    use balatro_core::enums::{Enhancement, Seal};
    let playing = |vs: &[V]| {
        let mut s = shop_state(14);
        redeem_state(&mut s, vs);
        let (mut n, mut decorated) = (0u32, 0u32);
        for _ in 0..8000 {
            if let ShopCardKind::PlayingCard(c) = shop::roll_card(&mut s).kind {
                n += 1;
                if c.enhancement != Enhancement::None || c.seal != Seal::None {
                    decorated += 1;
                }
            }
        }
        (n, decorated)
    };
    assert_eq!(playing(&[]), (0, 0));
    let (n, d) = playing(&[V::MagicTrick]);
    assert!(n > 800, "weight 4 of 32: about 1000 of 8000, got {n}");
    assert_eq!(d, 0, "plain without Illusion");
    let (n, d) = playing(&[V::MagicTrick, V::Illusion]);
    assert!(n > 800);
    // 40% carry an enhancement; shop cards never get seals (mirrored game bug, wiki: Illusion).
    let share = f64::from(d) / f64::from(n);
    assert!((share - 0.40).abs() < 0.06, "{share}");
}

#[test]
fn a_magic_trick_card_can_be_bought_with_step() {
    let mut g = in_shop(15);
    redeem(&mut g, V::MagicTrick);
    g.state_mut().money = 20;
    let card = balatro_core::card::Card::new(
        0,
        balatro_core::enums::Rank::Ten,
        balatro_core::enums::Suit::Hearts,
    );
    g.state_mut().shop.cards[0] = Some(ShopCard {
        kind: ShopCardKind::PlayingCard(card),
        free: false,
    });
    let n = g.state().deck.len();
    g.step(Action::Buy(0)).expect("buy");
    assert_eq!(g.state().deck.len(), n + 1);
}

// ----- packs / scoring ---------------------------------------------------------------------------

fn pack_items(s: &mut GameState, spec: PackSpec) -> Vec<PackItem> {
    balatro_core::packs::generate_items(s, spec)
        .into_iter()
        .flatten()
        .collect()
}

#[test]
fn omen_globe_lets_spectrals_into_arcana_packs_about_20_percent_of_the_time() {
    let ratio = |vs: &[V]| {
        let mut s = shop_state(16);
        redeem_state(&mut s, vs);
        let spec = PackSpec {
            kind: PackKind::Arcana,
            size: PackSize::Mega,
        };
        let (mut spectral, mut total) = (0u32, 0u32);
        for _ in 0..2000 {
            for item in pack_items(&mut s, spec) {
                total += 1;
                if let PackItem::Consumable(id) = item
                    && id.kind() == ConsumableKind::Spectral
                    && id != C::TheSoul
                {
                    spectral += 1;
                }
            }
        }
        f64::from(spectral) / f64::from(total)
    };
    assert_eq!(ratio(&[]), 0.0);
    let r = ratio(&[V::CrystalBall, V::OmenGlobe]);
    assert!((r - 0.2).abs() < 0.02, "{r}");
}

#[test]
fn an_arcana_pack_opens_through_the_game_with_omen_globe_owned() {
    let mut g = in_shop(17);
    buy(&mut g, V::OmenGlobe);
    g.state_mut().shop.packs[0] = Some(ShopPack {
        spec: PackSpec {
            kind: PackKind::Arcana,
            size: PackSize::Normal,
        },
        free: true,
    });
    g.step(Action::OpenPack(0)).expect("open");
    assert_eq!(g.state().phase, Phase::PackOpen);
    assert_eq!(g.state().pack.as_ref().expect("pack").items.len(), 3);
}

#[test]
fn observatory_multiplies_the_matching_held_planet_by_1_5() {
    let play = |observatory: bool| {
        let mut g = in_play(18);
        if observatory {
            redeem(&mut g, V::Telescope);
            redeem(&mut g, V::Observatory);
        }
        g.add_consumable(ConsumableInstance::new(C::Pluto))
            .expect("room");
        g.step(Action::Play(1)).expect("play");
        (
            g.state().round_score,
            g.state().hand_plays_run[HandType::HighCard.index()],
        )
    };
    let (plain, n0) = play(false);
    let (obs, n1) = play(true);
    assert_eq!((n0, n1), (1, 1));
    assert!(plain > 0.0);
    assert!((obs / plain - 1.5).abs() < 1e-9, "{obs} vs {plain}");
}

#[test]
fn observatory_ignores_planets_of_other_hands() {
    let play = |observatory: bool| {
        let mut g = in_play(18);
        if observatory {
            redeem(&mut g, V::Telescope);
            redeem(&mut g, V::Observatory);
        }
        g.add_consumable(ConsumableInstance::new(C::Mercury))
            .expect("room");
        g.step(Action::Play(1)).expect("play");
        g.state().round_score
    };
    assert_eq!(play(false), play(true));
}

// ----- one-shot run changes ----------------------------------------------------------------------

#[test]
fn nacho_tong_gives_a_second_extra_hand() {
    let mut g = in_shop(19);
    buy(&mut g, V::Grabber);
    buy(&mut g, V::NachoTong);
    assert_eq!(g.state().base_hands, 6);
    next_round(&mut g);
    assert_eq!(g.state().hands_left, 6);
}

#[test]
fn wasteful_and_recyclomancy_add_discards() {
    let mut g = in_shop(20);
    let d0 = g.state().base_discards;
    buy(&mut g, V::Wasteful);
    assert_eq!(g.state().base_discards, d0 + 1);
    buy(&mut g, V::Recyclomancy);
    assert_eq!(g.state().base_discards, d0 + 2);
    next_round(&mut g);
    assert_eq!(g.state().discards_left, d0 + 2);
}

#[test]
fn paint_brush_and_palette_add_hand_size() {
    let mut g = in_shop(21);
    buy(&mut g, V::PaintBrush);
    buy(&mut g, V::Palette);
    assert_eq!(g.state().base_hand_size, 10);
    next_round(&mut g);
    assert_eq!(g.state().deck.hand.len(), 10);
}

#[test]
fn blank_does_nothing_and_antimatter_adds_a_joker_slot() {
    let mut g = in_shop(22);
    let before = g.state().joker_slots;
    let snap = |g: &Game| {
        let s = g.state();
        (
            s.base_hands,
            s.base_discards,
            s.base_hand_size,
            s.interest_cap,
            s.consumable_slots,
            s.ante,
        )
    };
    let s0 = snap(&g);
    buy(&mut g, V::Blank);
    assert_eq!(g.state().joker_slots, before);
    assert_eq!(snap(&g), s0);
    assert_eq!(g.state().shop_mods, balatro_core::voucher::ShopMods::NONE);
    buy(&mut g, V::Antimatter);
    assert_eq!(g.state().joker_slots, before + 1);
    assert_eq!(snap(&g), s0);
}

#[test]
fn money_tree_raises_the_interest_cap_to_20() {
    let mut g = in_shop(23);
    buy(&mut g, V::SeedMoney);
    assert_eq!(g.state().interest_cap, 10);
    buy(&mut g, V::MoneyTree);
    assert_eq!(g.state().interest_cap, 20);
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.state_mut().money = 200;
    let info = g.step(Action::Play(1)).expect("win");
    let co = info.round.expect("round").cash_out.expect("cash out");
    assert_eq!(co.interest, 20, "$200 held pays the capped $20");
}

#[test]
fn hieroglyph_and_petroglyph_lower_the_ante_and_trade_hands_and_discards() {
    let mut g = in_shop(24);
    g.state_mut().ante = 4;
    let (h0, d0) = (g.state().base_hands, g.state().base_discards);
    buy(&mut g, V::Hieroglyph);
    assert_eq!(g.state().ante, 3);
    let req = |g: &Game, ante| {
        let s = g.state();
        balatro_core::blinds::requirement(ante, s.blind, s.boss_mult(), s.config.stake.scaling())
    };
    assert_eq!(g.state().target, req(&g, 3), "blind target recomputed");
    assert_eq!(g.state().base_hands, h0 - 1);
    buy(&mut g, V::Petroglyph);
    assert_eq!(g.state().ante, 2);
    assert_eq!(g.state().target, req(&g, 2));
    assert_eq!(g.state().base_discards, d0 - 1);
    assert_eq!(g.state().base_hands, h0 - 1);
    next_round(&mut g);
    assert_eq!(
        (g.state().hands_left, g.state().discards_left),
        (h0 - 1, d0 - 1)
    );
}

// ----- determinism -------------------------------------------------------------------------------

#[test]
fn buying_vouchers_is_deterministic() {
    let run = || {
        let mut g = in_shop(25);
        for id in [
            V::Overstock,
            V::OverstockPlus,
            V::Hone,
            V::MagicTrick,
            V::Illusion,
        ] {
            buy(&mut g, id);
        }
        g.state_mut().money = 100;
        g.step(Action::Reroll).expect("reroll");
        format!("{:?}", g.state().shop)
    };
    assert_eq!(run(), run());
}
