//! Shared numbers of consumable effects (several cards use them).
//! Card-specific numbers live as named consts next to the card in `content/*.rs`
//! (hooks.md "Joker sayıları" convention, notes.md §4.1a).
//! Sources: docs/mechanics/consumables.md §2-§3, notes.md §4.1b.

use crate::enums::{Edition, Enhancement};

/// Editions that Wheel of Fortune and Aura can give, with their weights (Foil 50 / Holographic
/// 35 / Polychrome 15; notes.md §4.1b "Aura"). Never Negative.
pub const EDITION_ROLL: [(Edition, f64); 3] = [
    (Edition::Foil, 50.0),
    (Edition::Holographic, 35.0),
    (Edition::Polychrome, 15.0),
];

/// Enhancements a Familiar/Grim/Incantation card can get: every enhancement except None and
/// Stone (notes.md §4.1b), chosen uniformly.
pub const ADDED_CARD_ENHANCEMENTS: [Enhancement; 7] = [
    Enhancement::Bonus,
    Enhancement::Mult,
    Enhancement::Wild,
    Enhancement::Glass,
    Enhancement::Steel,
    Enhancement::Gold,
    Enhancement::Lucky,
];
