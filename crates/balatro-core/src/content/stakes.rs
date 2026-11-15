//! Stakes (all 8, cumulative; docs/mechanics/stakes.md). Each stake applies the modifiers of
//! every stake below it, so the flags below are monotone in the stake's rank.

use crate::data::blind_chips::StakeScaling;
use crate::data::stakes::BLUE_STAKE_DISCARD_PENALTY;
use serde::{Deserialize, Serialize};

/// Stake (difficulty). Default White. Declaration order = difficulty order.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum StakeKind {
    /// Base difficulty.
    #[default]
    White,
    /// Small blind gives no reward money.
    Red,
    /// Required score scales faster (second chip table).
    Green,
    /// Shop and pack jokers can be Eternal.
    Black,
    /// -1 discard every round.
    Blue,
    /// Required score scales faster again (third chip table).
    Purple,
    /// Shop and pack jokers can be Perishable.
    Orange,
    /// Shop and pack jokers can be Rental.
    Gold,
}

impl StakeKind {
    /// All stakes, easiest first.
    pub const ALL: [StakeKind; 8] = [
        StakeKind::White,
        StakeKind::Red,
        StakeKind::Green,
        StakeKind::Black,
        StakeKind::Blue,
        StakeKind::Purple,
        StakeKind::Orange,
        StakeKind::Gold,
    ];

    /// Number of stakes.
    pub const COUNT: usize = Self::ALL.len();

    /// Index in [`StakeKind::ALL`] (0 = White).
    pub fn index(self) -> usize {
        self as usize
    }

    /// Lowercase name used by the Python constructors.
    pub fn name(self) -> &'static str {
        match self {
            StakeKind::White => "white",
            StakeKind::Red => "red",
            StakeKind::Green => "green",
            StakeKind::Black => "black",
            StakeKind::Blue => "blue",
            StakeKind::Purple => "purple",
            StakeKind::Orange => "orange",
            StakeKind::Gold => "gold",
        }
    }

    /// Parse a (case-insensitive) stake name.
    pub fn from_name(name: &str) -> Option<StakeKind> {
        let n = name.to_ascii_lowercase();
        Self::ALL.into_iter().find(|s| s.name() == n)
    }

    /// Blind chip table of this stake: White/Red the base table, Green/Black/Blue the second,
    /// Purple/Orange/Gold the third.
    pub fn scaling(self) -> StakeScaling {
        if self >= StakeKind::Purple {
            StakeScaling::Purple
        } else if self >= StakeKind::Green {
            StakeScaling::Green
        } else {
            StakeScaling::White
        }
    }

    /// Small blind pays no reward (Red Stake and above).
    pub fn no_small_reward(self) -> bool {
        self >= StakeKind::Red
    }

    /// Shop and pack jokers can be Eternal (Black Stake and above).
    pub fn eternal_jokers(self) -> bool {
        self >= StakeKind::Black
    }

    /// Discards removed from the base every round (Blue Stake and above).
    pub fn discard_penalty(self) -> u32 {
        if self >= StakeKind::Blue {
            BLUE_STAKE_DISCARD_PENALTY
        } else {
            0
        }
    }

    /// Shop and pack jokers can be Perishable (Orange Stake and above).
    pub fn perishable_jokers(self) -> bool {
        self >= StakeKind::Orange
    }

    /// Shop and pack jokers can be Rental (Gold Stake).
    pub fn rental_jokers(self) -> bool {
        self >= StakeKind::Gold
    }
}
