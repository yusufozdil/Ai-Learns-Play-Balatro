//! Poker hand evaluation (Four Fingers / Shortcut / Smeared / Wild / Stone / debuff rules),
//! `contains` bitset and scoring-card mask; fast all-subsets evaluation.
//! Rules: docs/mechanics/hands.md (decisions in notes.md §4). Implementation: Phase 1B.
//!
//! INTERFACE (orchestrator-owned): the `HandEval` type and the `evaluate` signature are fixed;
//! the scoring pipeline (Phase 1C) depends on them. Function bodies belong to Phase 1B.

use crate::card::{Card, RuleFlags};
use crate::enums::HandType;

/// Result of evaluating the played cards (1..=5 cards, in played left-to-right order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandEval {
    /// The hand type (highest tier that matches).
    pub hand_type: HandType,
    /// Bit `i` set = played card `i` is in the scoring set (Stone always; Splash = all).
    pub scoring_mask: u8,
    /// Bit `HandType::index()` set = the played cards contain that pattern
    /// (e.g. a Full House contains Pair, Two Pair, Three of a Kind and Full House).
    /// The chosen `hand_type` is always contained.
    pub contains: u16,
}

impl HandEval {
    /// True if the played cards contain `ht` (for "if played hand contains a Pair" effects).
    #[inline]
    pub fn contains(&self, ht: HandType) -> bool {
        self.contains & (1 << ht.index()) != 0
    }

    /// True if played card `i` scores.
    #[inline]
    pub fn scores(&self, i: usize) -> bool {
        i < 8 && self.scoring_mask & (1 << i) != 0
    }
}

/// Evaluate played cards (1..=5, left-to-right order) under the active rule flags.
/// Never panics on any input of length 0..=5 (length 0 yields High Card with empty masks).
pub fn evaluate(cards: &[Card], flags: &RuleFlags) -> HandEval {
    let _ = (cards, flags);
    unimplemented!("Phase 1B")
}
