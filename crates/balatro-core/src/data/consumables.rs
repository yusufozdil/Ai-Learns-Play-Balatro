//! Shared numbers of consumable effects (several cards use them).
//! Card-specific numbers live as named consts next to the card in `content/*.rs`
//! (joker counts convention, see hooks.md).
//! Sources: docs/mechanics/consumables.md §2-§3.

use crate::enums::{Edition, Enhancement};

/// Editions that Wheel of Fortune and Aura can give, with their weights (Foil 50 / Holographic
/// 35 / Polychrome 15). Never Negative.
pub const EDITION_ROLL: [(Edition, f64); 3] = [
    (Edition::Foil, 50.0),
    (Edition::Holographic, 35.0),
    (Edition::Polychrome, 15.0),
];

/// Enhancements a Familiar/Grim/Incantation card can get: every enhancement except None and
/// Stone, chosen uniformly.
pub const ADDED_CARD_ENHANCEMENTS: [Enhancement; 7] = [
    Enhancement::Bonus,
    Enhancement::Mult,
    Enhancement::Wild,
    Enhancement::Glass,
    Enhancement::Steel,
    Enhancement::Gold,
    Enhancement::Lucky,
];
