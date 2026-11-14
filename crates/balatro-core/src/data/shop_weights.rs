//! Shop type weights, rarity odds, edition odds, booster-pack tables and card-generation odds.
//! Sources: docs/mechanics/shop.md §2-§5, packs.md §1-§4, notes.md §4.1d. Numbers marked
//! UNVERIFIED there are recalled from the original game and kept here in one place.

use crate::card::Card;
use crate::enums::{Edition, Enhancement, Rarity, Seal};
use crate::hooks::PackKind;
use crate::packs::PackSize;

// ----- shop card slots --------------------------------------------------------------------------

/// Default random card slots of a shop (Overstock +1, Overstock Plus +2).
pub const BASE_CARD_SLOTS: u32 = 2;
/// Booster-pack slots of every shop.
pub const BOOSTER_SLOTS: usize = 2;

/// Type weights of one shop card slot (shop.md §2): Joker 20, Tarot 4, Planet 4.
pub const WEIGHT_JOKER: f64 = 20.0;
/// Tarot weight without a voucher.
pub const WEIGHT_TAROT: f64 = 4.0;
/// Planet weight without a voucher.
pub const WEIGHT_PLANET: f64 = 4.0;
/// Playing-card weight once Magic Trick is owned (shop.md §2).
pub const WEIGHT_PLAYING_CARD: f64 = 4.0;
/// Spectral weight (Ghost Deck only; no implemented deck uses it yet).
pub const WEIGHT_SPECTRAL_GHOST_DECK: f64 = 2.0;
/// Tarot/Planet weight with the Merchant voucher (shop.md §2; notes.md §4.1d).
pub const WEIGHT_MERCHANT: f64 = 9.6;
/// Tarot/Planet weight with the Tycoon voucher.
pub const WEIGHT_TYCOON: f64 = 32.0;

// ----- jokers -----------------------------------------------------------------------------------

/// Rarity weights of a shop/pack/created joker (shop.md §3): Common 70, Uncommon 25, Rare 5.
/// Legendary never appears here (only The Soul asks for it explicitly).
pub const RARITY_WEIGHTS: [(Rarity, f64); 3] = [
    (Rarity::Common, 70.0),
    (Rarity::Uncommon, 25.0),
    (Rarity::Rare, 5.0),
];

/// Joker edition thresholds (shop.md §4, notes.md §4.1d): one uniform roll `r`; the band that
/// contains `1 - r` wins. Negative `0.3%` (never scaled), then Polychrome up to `0.6% * m`,
/// Holographic up to `2% * m`, Foil up to `4% * m`, where `m` = Hone 2 / Glow Up 4.
pub const JOKER_NEGATIVE_BAND: f64 = 0.003;
/// Cumulative Polychrome threshold at `m = 1`.
pub const JOKER_POLY_BAND: f64 = 0.006;
/// Cumulative Holographic threshold at `m = 1`.
pub const JOKER_HOLO_BAND: f64 = 0.02;
/// Cumulative Foil threshold at `m = 1`.
pub const JOKER_FOIL_BAND: f64 = 0.04;

/// Playing-card edition odds in shop/Standard packs (shop.md §4/§5, packs.md §4): Foil 4%,
/// Holographic 2.8%, Polychrome 1.2%, each scaled by `m`, never Negative.
pub const CARD_FOIL: f64 = 0.04;
/// Holographic share.
pub const CARD_HOLO: f64 = 0.028;
/// Polychrome share.
pub const CARD_POLY: f64 = 0.012;

// ----- playing cards (Standard pack, Illusion) ---------------------------------------------------

/// Enhancement chance of a Standard pack card (UNVERIFIED, packs.md §4).
pub const STANDARD_ENHANCEMENT_CHANCE: f64 = 0.4;
/// Seal chance of a Standard pack card (UNVERIFIED).
pub const STANDARD_SEAL_CHANCE: f64 = 0.2;
/// Illusion: enhancement chance of a shop playing card (shop.md §5).
pub const ILLUSION_ENHANCEMENT_CHANCE: f64 = 0.4;
/// Illusion: seal chance of a shop playing card (UNVERIFIED; the original bugs it, we model the
/// intended behaviour).
pub const ILLUSION_SEAL_CHANCE: f64 = 0.2;
/// Enhancements a random playing card can get: all 8 (UNVERIFIED that Stone is included).
pub const CARD_ENHANCEMENTS: [Enhancement; 8] = [
    Enhancement::Bonus,
    Enhancement::Mult,
    Enhancement::Wild,
    Enhancement::Glass,
    Enhancement::Steel,
    Enhancement::Stone,
    Enhancement::Gold,
    Enhancement::Lucky,
];
/// Seals a random playing card can get.
pub const CARD_SEALS: [Seal; 4] = [Seal::Gold, Seal::Red, Seal::Blue, Seal::Purple];

/// Per-slot chance of The Soul in Arcana/Spectral packs and Black Hole in Celestial/Spectral
/// packs (consumables.md §1: 0.3%). Not affected by Oops! All 6s.
pub const SOUL_BLACK_HOLE_CHANCE: f64 = 0.003;
/// Omen Globe: chance that an Arcana pack slot is a Spectral (UNVERIFIED, packs.md §4).
pub const OMEN_GLOBE_SPECTRAL_CHANCE: f64 = 0.2;

// ----- booster packs ----------------------------------------------------------------------------

/// Cards shown by a pack of `kind` and `size` (packs.md §1).
pub const fn pack_cards(kind: PackKind, size: PackSize) -> u8 {
    match (kind, size) {
        (PackKind::Buffoon | PackKind::Spectral, PackSize::Normal) => 2,
        (PackKind::Buffoon | PackKind::Spectral, _) => 4,
        (_, PackSize::Normal) => 3,
        _ => 5,
    }
}

/// Items that may be taken from a pack of `size`: Normal/Jumbo 1, Mega 2 (packs.md §1).
pub const fn pack_picks(size: PackSize) -> u8 {
    match size {
        PackSize::Mega => 2,
        _ => 1,
    }
}

/// Base price of a pack of `size` (shop.md §6): $4 / $6 / $8.
pub const fn pack_price(size: PackSize) -> i32 {
    use crate::data::prices::{PACK_JUMBO, PACK_MEGA, PACK_NORMAL};
    match size {
        PackSize::Normal => PACK_NORMAL,
        PackSize::Jumbo => PACK_JUMBO,
        PackSize::Mega => PACK_MEGA,
    }
}

/// Shop appearance weight of one pack (packs.md §2); total 22.42.
pub const fn pack_weight(kind: PackKind, size: PackSize) -> f64 {
    let (n, j, m) = match kind {
        PackKind::Standard | PackKind::Arcana | PackKind::Celestial => (4.0, 2.0, 0.5),
        PackKind::Buffoon => (1.2, 0.6, 0.15),
        PackKind::Spectral => (0.6, 0.3, 0.07),
    };
    match size {
        PackSize::Normal => n,
        PackSize::Jumbo => j,
        PackSize::Mega => m,
    }
}

/// Edition of a random playing card (shop, Standard pack), rolled with the scale `m` (Hone 2 / Glow Up 4) from one uniform `r` in `[0, 1)`.
pub fn card_edition(r: f64, m: u32) -> Edition {
    let m = f64::from(m.max(1));
    if r < CARD_FOIL * m {
        Edition::Foil
    } else if r < (CARD_FOIL + CARD_HOLO) * m {
        Edition::Holographic
    } else if r < (CARD_FOIL + CARD_HOLO + CARD_POLY) * m {
        Edition::Polychrome
    } else {
        Edition::Base
    }
}

/// Edition of a shop/pack/created joker from one uniform `r` in `[0, 1)` (see
/// [`JOKER_NEGATIVE_BAND`]).
pub fn joker_edition(r: f64, m: u32) -> Edition {
    let m = f64::from(m.max(1));
    let x = 1.0 - r;
    if x < JOKER_NEGATIVE_BAND {
        Edition::Negative
    } else if x < JOKER_POLY_BAND * m {
        Edition::Polychrome
    } else if x < JOKER_HOLO_BAND * m {
        Edition::Holographic
    } else if x < JOKER_FOIL_BAND * m {
        Edition::Foil
    } else {
        Edition::Base
    }
}

/// Price surcharge of an edition (shop.md §4: Foil +2, Holographic +3, Polychrome +5, Negative +5).
pub const fn edition_surcharge(e: Edition) -> i32 {
    use crate::data::prices::{EDITION_FOIL, EDITION_HOLO, EDITION_NEGATIVE, EDITION_POLY};
    match e {
        Edition::Base => 0,
        Edition::Foil => EDITION_FOIL,
        Edition::Holographic => EDITION_HOLO,
        Edition::Polychrome => EDITION_POLY,
        Edition::Negative => EDITION_NEGATIVE,
    }
}

/// Surcharge of a playing card's edition (same table, no Negative).
pub fn card_surcharge(c: &Card) -> i32 {
    edition_surcharge(c.edition)
}

// ----- tags / vouchers --------------------------------------------------------------------------

/// Economy tag: doubles the money, gaining at most this much (tags.md).
pub const TAG_ECONOMY_CAP: i32 = 40;
/// Investment tag payout.
pub const TAG_INVESTMENT: i32 = 25;
/// Speed tag: dollars per blind skipped this run.
pub const TAG_SPEED_PER_SKIP: i32 = 5;
/// Orbital tag: hand levels.
pub const TAG_ORBITAL_LEVELS: i32 = 3;
/// Juggle tag: hand size next round.
pub const TAG_JUGGLE_HAND_SIZE: i32 = 3;
/// Top-up tag: Common jokers created.
pub const TAG_TOP_UP_JOKERS: usize = 2;
/// Boss reroll price (Director's Cut / Retcon).
pub const BOSS_REROLL_COST: i32 = 10;
