//! Prices, interest caps and money constants.
//! Sources: docs/mechanics/shop.md §6-§8, economy.md, blinds.md §4/§8.

/// Base interest cap in dollars (economy.md "Interest cap changes").
pub const INTEREST_CAP_BASE: i32 = 5;
/// Interest cap with Seed Money.
pub const INTEREST_CAP_SEED_MONEY: i32 = 10;
/// Interest cap with Money Tree.
pub const INTEREST_CAP_MONEY_TREE: i32 = 20;
/// Dollars held per interest step.
pub const INTEREST_STEP: i32 = 5;
/// Extra interest per step granted by To the Moon.
pub const TO_THE_MOON_BONUS: i32 = 1;

/// Dollars per unused hand (blinds.md §4).
pub const HAND_REWARD: i32 = 1;
/// Dollars per unused discard in the base game (economy.md).
pub const DISCARD_REWARD: i32 = 0;
/// Gold card held in hand at round end (blinds.md §6.7a).
pub const GOLD_CARD_PAYOUT: i32 = 3;
/// Gold card with Red Seal held in hand at round end.
pub const GOLD_CARD_RED_SEAL_PAYOUT: i32 = 6;

/// Foil surcharge (shop.md §4/§6).
pub const EDITION_FOIL: i32 = 2;
/// Holographic surcharge.
pub const EDITION_HOLO: i32 = 3;
/// Polychrome surcharge.
pub const EDITION_POLY: i32 = 5;
/// Negative surcharge.
pub const EDITION_NEGATIVE: i32 = 5;

/// Normal pack price (shop.md §6).
pub const PACK_NORMAL: i32 = 4;
/// Jumbo pack price.
pub const PACK_JUMBO: i32 = 6;
/// Mega pack price.
pub const PACK_MEGA: i32 = 8;
/// Voucher price (shop.md §6).
pub const VOUCHER: i32 = 10;
/// Tarot card base price.
pub const TAROT: i32 = 3;
/// Planet card base price.
pub const PLANET: i32 = 3;
/// Spectral card base price.
pub const SPECTRAL: i32 = 4;
/// Playing card base price (Magic Trick).
pub const PLAYING_CARD: i32 = 1;

/// Clearance Sale discount percent (shop.md §6).
pub const CLEARANCE_PERCENT: i32 = 25;
/// Liquidation discount percent (shop.md §6).
pub const LIQUIDATION_PERCENT: i32 = 50;

/// Reroll starting cost (shop.md §7).
pub const REROLL_BASE: i32 = 5;
/// Reroll Surplus starting price.
pub const REROLL_SURPLUS_BASE: i32 = 3;
/// Reroll Glut starting price.
pub const REROLL_GLUT_BASE: i32 = 1;
/// Cost increase per reroll within a shop.
pub const REROLL_INCREMENT: i32 = 1;

/// Starting money, Red Deck.
pub const STARTING_MONEY: i32 = 4;
