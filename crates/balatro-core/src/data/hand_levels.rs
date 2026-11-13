//! Base chips/mult per hand type and per-level increments.
//! Source: docs/mechanics/hands.md §1 (balatrowiki.org/w/Poker_Hands).

use crate::enums::HandType;

/// Level-1 values and per-level increments for one hand type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandBase {
    /// Chips at level 1.
    pub chips: f64,
    /// Mult at level 1.
    pub mult: f64,
    /// Chips added per level above 1.
    pub chips_per_level: f64,
    /// Mult added per level above 1.
    pub mult_per_level: f64,
}

const fn hb(chips: f64, mult: f64, chips_per_level: f64, mult_per_level: f64) -> HandBase {
    HandBase {
        chips,
        mult,
        chips_per_level,
        mult_per_level,
    }
}

/// Indexed by `HandType::index()`.
pub const HAND_BASE: [HandBase; HandType::COUNT] = [
    hb(5.0, 1.0, 10.0, 1.0),    // High Card
    hb(10.0, 2.0, 15.0, 1.0),   // Pair
    hb(20.0, 2.0, 20.0, 1.0),   // Two Pair
    hb(30.0, 3.0, 20.0, 2.0),   // Three of a Kind
    hb(30.0, 4.0, 30.0, 3.0),   // Straight
    hb(35.0, 4.0, 15.0, 2.0),   // Flush
    hb(40.0, 4.0, 25.0, 2.0),   // Full House
    hb(60.0, 7.0, 30.0, 3.0),   // Four of a Kind
    hb(100.0, 8.0, 40.0, 4.0),  // Straight Flush
    hb(120.0, 12.0, 35.0, 3.0), // Five of a Kind
    hb(140.0, 14.0, 40.0, 4.0), // Flush House
    hb(160.0, 16.0, 50.0, 3.0), // Flush Five
];

/// Chips and mult of `hand` at `level` (level >= 1; level 0 is treated as 1).
/// `value = base + (level - 1) * increment`.
pub fn chips_mult(hand: HandType, level: u32) -> (f64, f64) {
    let b = &HAND_BASE[hand.index()];
    let extra = level.saturating_sub(1) as f64;
    (
        b.chips + extra * b.chips_per_level,
        b.mult + extra * b.mult_per_level,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_one_and_growth() {
        assert_eq!(chips_mult(HandType::Flush, 1), (35.0, 4.0));
        assert_eq!(chips_mult(HandType::Flush, 3), (65.0, 8.0));
        assert_eq!(chips_mult(HandType::HighCard, 0), (5.0, 1.0));
        assert_eq!(chips_mult(HandType::FlushFive, 2), (210.0, 19.0));
    }
}
