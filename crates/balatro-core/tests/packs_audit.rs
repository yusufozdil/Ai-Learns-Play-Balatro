//! Phase 5B pack audit: the 15 packs against docs/mechanics/packs.md (table of size, picks,
//! price and shop weight; every pack kind draws from the right pool).

use balatro_core::consumable::ConsumableKind;
use balatro_core::content::ConsumableId as C;
use balatro_core::content::VoucherId;
use balatro_core::data::shop_weights::pack_weight;
use balatro_core::enums::{HandType, Rarity};
use balatro_core::game::Game;
use balatro_core::hooks::PackKind;
use balatro_core::packs::{self, PackItem, PackSize, PackSpec};
use balatro_core::shop::{self, ShopPack};
use balatro_core::state::{GameConfig, GameState};
use balatro_core::voucher;

use PackKind as K;
use PackSize as S;

/// Source: https://balatrowiki.org/w/Booster_Packs (verified 2026-11).
/// (kind, size, cards shown, picks, price, shop weight)
const TABLE: [(PackKind, PackSize, u8, u8, i32, f64); 15] = [
    (K::Arcana, S::Normal, 3, 1, 4, 4.0),
    (K::Arcana, S::Jumbo, 5, 1, 6, 2.0),
    (K::Arcana, S::Mega, 5, 2, 8, 0.5),
    (K::Celestial, S::Normal, 3, 1, 4, 4.0),
    (K::Celestial, S::Jumbo, 5, 1, 6, 2.0),
    (K::Celestial, S::Mega, 5, 2, 8, 0.5),
    (K::Standard, S::Normal, 3, 1, 4, 4.0),
    (K::Standard, S::Jumbo, 5, 1, 6, 2.0),
    (K::Standard, S::Mega, 5, 2, 8, 0.5),
    (K::Buffoon, S::Normal, 2, 1, 4, 1.2),
    (K::Buffoon, S::Jumbo, 4, 1, 6, 0.6),
    (K::Buffoon, S::Mega, 4, 2, 8, 0.15),
    (K::Spectral, S::Normal, 2, 1, 4, 0.6),
    (K::Spectral, S::Jumbo, 4, 1, 6, 0.3),
    (K::Spectral, S::Mega, 4, 2, 8, 0.07),
];

fn spec(kind: PackKind, size: PackSize) -> PackSpec {
    PackSpec { kind, size }
}

#[test]
fn the_fifteen_packs_match_the_wiki_table() {
    let g = Game::new(1, GameConfig::default());
    for (kind, size, cards, picks, price, weight) in TABLE {
        let sp = spec(kind, size);
        assert_eq!(sp.cards(), cards, "{kind:?} {size:?} cards");
        assert_eq!(sp.picks(), picks, "{kind:?} {size:?} picks");
        let p = ShopPack {
            spec: sp,
            free: false,
        };
        assert_eq!(
            shop::pack_price_of(g.state(), &p),
            price,
            "{kind:?} {size:?}"
        );
        let free = ShopPack {
            spec: sp,
            free: true,
        };
        assert_eq!(shop::pack_price_of(g.state(), &free), 0);
        assert!(
            (pack_weight(kind, size) - weight).abs() < 1e-12,
            "{kind:?} {size:?} weight"
        );
    }
    let total: f64 = TABLE.iter().map(|t| t.5).sum();
    assert!((total - 22.42).abs() < 1e-9, "total weight {total}");
}

#[test]
fn rolled_pack_specs_follow_the_weights() {
    use balatro_core::rng::GameRng;
    let mut rng = GameRng::new(5);
    let n = 200_000;
    let mut counts = [0u32; 15];
    for _ in 0..n {
        counts[shop::roll_pack_spec(&mut rng).index()] += 1;
    }
    for (kind, size, _, _, _, weight) in TABLE {
        let got = f64::from(counts[spec(kind, size).index()]) / f64::from(n);
        let want = weight / 22.42;
        assert!(
            (got - want).abs() < 0.004,
            "{kind:?} {size:?}: {got} vs {want}"
        );
    }
}

fn state(seed: u64) -> GameState {
    Game::new(seed, GameConfig::default()).state().clone()
}

#[test]
fn every_pack_has_the_right_number_of_items_of_the_right_pool() {
    for (kind, size, cards, ..) in TABLE {
        for seed in 0..60 {
            let mut s = state(seed);
            let items = packs::generate_items(&mut s, spec(kind, size));
            assert_eq!(items.len(), usize::from(cards), "{kind:?} {size:?}");
            for it in items.into_iter().map(|i| i.expect("item")) {
                match (kind, it) {
                    (K::Arcana, PackItem::Consumable(id)) => assert!(
                        id.kind() == ConsumableKind::Tarot || id == C::TheSoul,
                        "{id:?} in Arcana"
                    ),
                    (K::Celestial, PackItem::Consumable(id)) => assert!(
                        id.kind() == ConsumableKind::Planet || id == C::BlackHole,
                        "{id:?} in Celestial"
                    ),
                    (K::Spectral, PackItem::Consumable(id)) => {
                        assert_eq!(id.kind(), ConsumableKind::Spectral, "{id:?}")
                    }
                    (K::Standard, PackItem::Card(_)) => {}
                    (K::Buffoon, PackItem::Joker(j)) => {
                        let r = j.def().expect("def").rarity;
                        assert_ne!(r, Rarity::Legendary, "no legendary in a Buffoon pack");
                    }
                    (k, it) => panic!("{k:?} pack gave {it:?}"),
                }
            }
        }
    }
}

fn soul_black_hole_rate(kind: PackKind, id: C) -> f64 {
    let mut hits = 0u32;
    let mut total = 0u32;
    for seed in 0..3000 {
        let mut s = state(seed);
        for it in packs::generate_items(&mut s, spec(kind, S::Jumbo)) {
            total += 1;
            hits += u32::from(it == Some(PackItem::Consumable(id)));
        }
    }
    f64::from(hits) / f64::from(total)
}

#[test]
fn the_soul_and_black_hole_appear_at_about_point_three_percent() {
    // 15000 items per pack kind: expect ~45 hits; accept 0.15%..0.6%.
    for (kind, id) in [
        (K::Arcana, C::TheSoul),
        (K::Celestial, C::BlackHole),
        (K::Spectral, C::TheSoul),
        (K::Spectral, C::BlackHole),
    ] {
        let r = soul_black_hole_rate(kind, id);
        assert!((0.0015..0.006).contains(&r), "{kind:?} {id:?}: {r}");
    }
    // Never in the wrong pack.
    assert_eq!(soul_black_hole_rate(K::Celestial, C::TheSoul), 0.0);
    assert_eq!(soul_black_hole_rate(K::Arcana, C::BlackHole), 0.0);
}

#[test]
fn telescope_forces_the_most_played_hands_planet_into_item_zero() {
    for seed in 0..20 {
        let mut s = state(seed);
        voucher::redeem(&mut s, VoucherId::Telescope);
        s.hand_plays_run[HandType::Flush.index()] = 4;
        s.hand_plays_run[HandType::Pair.index()] = 2;
        let items = packs::generate_items(&mut s, spec(K::Celestial, S::Mega));
        assert_eq!(
            items[0],
            Some(PackItem::Consumable(C::planet_for(HandType::Flush)))
        );
        // Not for other kinds, not without plays.
        let mut s = state(seed);
        voucher::redeem(&mut s, VoucherId::Telescope);
        let items = packs::generate_items(&mut s, spec(K::Celestial, S::Mega));
        assert_eq!(items.len(), 5);
    }
}

#[test]
fn packs_have_no_duplicates_without_showman() {
    for seed in 0..100 {
        let mut s = state(seed);
        let items = packs::generate_items(&mut s, spec(K::Celestial, S::Mega));
        let mut ids: Vec<C> = items
            .iter()
            .filter_map(|i| match i {
                Some(PackItem::Consumable(id)) => Some(*id),
                _ => None,
            })
            .collect();
        let n = ids.len();
        ids.sort_by_key(|i| i.index());
        ids.dedup();
        assert_eq!(ids.len(), n, "seed {seed}");
    }
}
