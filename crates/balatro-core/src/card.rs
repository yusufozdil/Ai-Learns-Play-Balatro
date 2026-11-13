//! `Card` (Copy): playing card with rank/suit/enhancement/edition/seal, chip value,
//! face/suit predicates that respect rule flags (Pareidolia, Smeared, Wild).

use crate::enums::{Edition, Enhancement, Rank, Seal, Suit};
use serde::{Deserialize, Serialize};

/// Joker-driven rule switches read by hand evaluation and scoring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleFlags {
    /// Flushes and straights need only 4 cards.
    pub four_fingers: bool,
    /// Straights may skip one rank.
    pub shortcut: bool,
    /// Hearts == Diamonds and Spades == Clubs.
    pub smeared: bool,
    /// Every card counts as a face card.
    pub pareidolia: bool,
    /// All played cards score.
    pub splash: bool,
}

/// A playing card. `uid` is unique within a run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Card {
    /// Unique id within a run.
    pub uid: u32,
    /// Printed rank.
    pub rank: Rank,
    /// Printed suit.
    pub suit: Suit,
    /// Enhancement.
    pub enhancement: Enhancement,
    /// Edition.
    pub edition: Edition,
    /// Seal.
    pub seal: Seal,
    /// Permanent extra chips (e.g. from Hiker).
    pub perma_bonus: i32,
    /// Debuffed by a boss blind.
    pub debuffed: bool,
    /// Face down (boss blind effect).
    pub face_down: bool,
}

impl Card {
    /// Plain card with no enhancement, edition or seal.
    pub fn new(uid: u32, rank: Rank, suit: Suit) -> Self {
        Card {
            uid,
            rank,
            suit,
            enhancement: Enhancement::None,
            edition: Edition::Base,
            seal: Seal::None,
            perma_bonus: 0,
            debuffed: false,
            face_down: false,
        }
    }

    /// True for Stone cards (no rank, no suit).
    pub fn is_stone(&self) -> bool {
        self.enhancement == Enhancement::Stone
    }

    /// Rank for hand evaluation; `None` for Stone cards.
    pub fn effective_rank(&self) -> Option<Rank> {
        if self.is_stone() {
            None
        } else {
            Some(self.rank)
        }
    }

    /// Chips from the rank alone: face value (A = 11, J/Q/K = 10), 0 for Stone.
    /// Stone's +50 chips is an enhancement effect applied by scoring, not included here;
    /// `perma_bonus` is also not included.
    pub fn base_chips(&self) -> i32 {
        if self.is_stone() {
            0
        } else {
            self.rank.chip_value()
        }
    }

    /// Face card: J/Q/K; with Pareidolia every non-Stone card is a face card. Stone never is.
    pub fn is_face(&self, flags: &RuleFlags) -> bool {
        if self.is_stone() {
            return false;
        }
        flags.pareidolia || self.rank.is_face_rank()
    }

    /// Suit membership: Stone -> never; Wild -> every suit (unless debuffed: a debuffed Wild
    /// card reverts to its printed suit, docs/mechanics/hands.md §4.2); Smeared makes
    /// Hearts==Diamonds and Spades==Clubs; otherwise exact match.
    pub fn has_suit(&self, suit: Suit, flags: &RuleFlags) -> bool {
        if self.is_stone() {
            return false;
        }
        if self.enhancement == Enhancement::Wild && !self.debuffed {
            return true;
        }
        if flags.smeared {
            self.suit.is_red() == suit.is_red()
        } else {
            self.suit == suit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_small() {
        assert!(std::mem::size_of::<Card>() <= 20);
    }
}
