//! Numeric effects of playing-card modifiers (enhancements, seals, editions) and of joker
//! editions, used by the scoring pipeline.
//! Sources: docs/mechanics/cards.md (balatrowiki.org/w/Card_Modifiers, /w/Glass_Card,
//! /w/Stone_Card, /w/Gold_Seal) and docs/mechanics/scoring.md.

/// Bonus card: +30 chips when scored.
pub const BONUS_CHIPS: f64 = 30.0;
/// Stone card: +50 chips when scored (and no rank chips).
pub const STONE_CHIPS: f64 = 50.0;
/// Mult card: +4 mult when scored.
pub const MULT_CARD_MULT: f64 = 4.0;
/// Glass card: x2 mult when scored.
pub const GLASS_XMULT: f64 = 2.0;
/// Glass card: 1 in 4 chance to break, rolled once per card after scoring.
pub const GLASS_BREAK_ODDS: (u32, u32) = (1, 4);
/// Steel card: x1.5 mult while held in hand during scoring.
pub const STEEL_XMULT: f64 = 1.5;
/// Lucky card: +20 mult on a successful mult roll.
pub const LUCKY_MULT: f64 = 20.0;
/// Lucky card: 1 in 5 chance for the mult bonus (rolled on every trigger).
pub const LUCKY_MULT_ODDS: (u32, u32) = (1, 5);
/// Lucky card: $20 on a successful money roll.
pub const LUCKY_MONEY: i32 = 20;
/// Lucky card: 1 in 15 chance for the money bonus (rolled on every trigger).
pub const LUCKY_MONEY_ODDS: (u32, u32) = (1, 15);
/// Gold card: $3 if held in hand at end of round (per trigger: Red Seal / Mime repeat it).
pub const GOLD_CARD_MONEY: i32 = 3;
/// Gold Seal: $3 each time the card triggers while scoring.
pub const GOLD_SEAL_MONEY: i32 = 3;
/// Foil edition (cards and jokers): +50 chips.
pub const FOIL_CHIPS: f64 = 50.0;
/// Holographic edition (cards and jokers): +10 mult.
pub const HOLO_MULT: f64 = 10.0;
/// Polychrome edition (cards and jokers): x1.5 mult.
pub const POLYCHROME_XMULT: f64 = 1.5;
/// Observatory voucher: x1.5 mult per held Planet card matching the played hand type.
pub const OBSERVATORY_XMULT: f64 = 1.5;
/// Safety cap on the number of triggers of one card in one hand (1 + retriggers). Balatro has
/// no cap; this only guarantees termination if a hook returns a nonsensical count.
pub const MAX_TRIGGERS_PER_CARD: u32 = 256;
