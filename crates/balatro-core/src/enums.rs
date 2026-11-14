//! Core enums: Suit, Rank, HandType, Enhancement, Edition, Seal, Phase, Rarity.
//!
//! Every enum has `ALL`, `COUNT` and a stable 0-based `index()` (used for one-hot encoding, so
//! the order of variants must never change).

use serde::{Deserialize, Serialize};

macro_rules! simple_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $var:ident),+ $(,)? }, default = $def:ident) => {
        simple_enum!(@define $(#[$m])* $name { $($(#[$vm])* $var),+ });
        impl Default for $name {
            fn default() -> Self { Self::$def }
        }
    };
    (@one $var:ident) => { 1 };
    (@define $(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $var:ident),+ }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[repr(u8)]
        pub enum $name { $($(#[$vm])* $var),+ }
        impl $name {
            /// All variants in declaration order.
            pub const ALL: [Self; 0 $(+ simple_enum!(@one $var))+] = [$(Self::$var),+];
            /// Number of variants.
            pub const COUNT: usize = Self::ALL.len();
            /// Stable 0-based index (position in `ALL`).
            pub fn index(self) -> usize {
                self as usize
            }
        }
    };
}

simple_enum!(
    /// Card suit.
    Suit {
        Spades,
        Hearts,
        Clubs,
        Diamonds
    },
    default = Spades
);

impl Suit {
    /// Hearts and Diamonds are red.
    pub fn is_red(self) -> bool {
        matches!(self, Suit::Hearts | Suit::Diamonds)
    }
}

/// Card rank; discriminant is the numeric value (Ace = 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum Rank {
    Two = 2,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace = 14,
}

impl Rank {
    /// All ranks, Two to Ace.
    pub const ALL: [Self; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    /// Number of ranks (13).
    pub const COUNT: usize = Self::ALL.len();
    /// 0 for Two ... 12 for Ace.
    pub fn index(self) -> usize {
        self as usize - 2
    }
    /// Chips a card of this rank contributes: 2-10 face value, J/Q/K 10, Ace 11.
    pub fn chip_value(self) -> i32 {
        match self {
            Rank::Jack | Rank::Queen | Rank::King => 10,
            Rank::Ace => 11,
            r => r as i32,
        }
    }
    /// Jack, Queen or King (ignores Pareidolia; see `Card::is_face`).
    pub fn is_face_rank(self) -> bool {
        matches!(self, Rank::Jack | Rank::Queen | Rank::King)
    }
    /// Numeric value 2..=14 to rank.
    pub fn from_value(v: u8) -> Option<Rank> {
        if (2..=14).contains(&v) {
            Some(Self::ALL[v as usize - 2])
        } else {
            None
        }
    }
}

simple_enum!(
    /// Poker hand types in ascending base strength.
    HandType {
        HighCard,
        Pair,
        TwoPair,
        ThreeOfAKind,
        Straight,
        Flush,
        FullHouse,
        FourOfAKind,
        StraightFlush,
        FiveOfAKind,
        FlushHouse,
        FlushFive
    },
    default = HighCard
);

impl HandType {
    /// Secret hands (Five of a Kind, Flush House, Flush Five): their Planet and Orbital tag only
    /// appear once the hand was played.
    pub fn is_secret(self) -> bool {
        matches!(
            self,
            HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive
        )
    }
}

simple_enum!(
    /// Card enhancement.
    Enhancement {
        None,
        Bonus,
        Mult,
        Wild,
        Glass,
        Steel,
        Stone,
        Gold,
        Lucky
    },
    default = None
);

simple_enum!(
    /// Card/joker edition.
    Edition {
        Base,
        Foil,
        Holographic,
        Polychrome,
        Negative
    },
    default = Base
);

simple_enum!(
    /// Card seal.
    Seal {
        None,
        Gold,
        Red,
        Blue,
        Purple
    },
    default = None
);

simple_enum!(
    /// Game phase.
    Phase {
        BlindSelect,
        Play,
        Shop,
        PackOpen,
        GameOver,
        Won
    },
    default = BlindSelect
);

simple_enum!(
    /// Joker rarity.
    Rarity {
        Common,
        Uncommon,
        Rare,
        Legendary
    },
    default = Common
);
