//! Starting decks (all 15; docs/mechanics/decks.md). A [`DeckDef`] is plain data describing what
//! a deck changes; the game layer reads the flags (`GameState::new`, cash-out, shop, scoring).

use crate::card::Card;
use crate::content::ids::{ConsumableId, VoucherId};
use crate::data::prices::{DISCARD_REWARD, HAND_REWARD};
use crate::data::stakes::{
    GREEN_DECK_DISCARD_REWARD, GREEN_DECK_HAND_REWARD, PAINTED_DECK_HAND_SIZE, PLASMA_BLIND_MULT,
    YELLOW_DECK_MONEY,
};
use crate::deck::{Deck, standard_deck};
use crate::enums::{Rank, Suit};
use crate::rng::{GameRng, RngKey};
use serde::{Deserialize, Serialize};

/// Starting deck. Default Red.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeckKind {
    /// +1 discard every round.
    #[default]
    Red,
    /// +1 hand every round.
    Blue,
    /// Start with +$10.
    Yellow,
    /// $2 per unused hand and $1 per unused discard at cash-out; no interest.
    Green,
    /// +1 joker slot, -1 hand every round.
    Black,
    /// Start with Crystal Ball and two copies of The Fool.
    Magic,
    /// Start with Telescope, -1 consumable slot.
    Nebula,
    /// Spectral cards can appear in the shop; start with Hex.
    Ghost,
    /// No face cards (40 cards).
    Abandoned,
    /// 26 Spades and 26 Hearts.
    Checkered,
    /// Start with Tarot Merchant, Planet Merchant and Overstock.
    Zodiac,
    /// +2 hand size, -1 joker slot.
    Painted,
    /// A Double Tag after every Boss Blind.
    Anaglyph,
    /// Chips and mult are balanced; blinds are twice as big.
    Plasma,
    /// Random ranks and suits.
    Erratic,
}

/// How the 52 starting cards are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeckComposition {
    /// Standard 52.
    Standard,
    /// Standard minus Jacks, Queens and Kings.
    NoFaces,
    /// Two of every rank in Spades and in Hearts.
    Checkered,
    /// 52 cards with independent random rank and suit.
    Erratic,
}

/// What a deck changes (the base is a plain Red-less deck: all zero / false / empty).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeckDef {
    /// Hands per round delta.
    pub hands: i32,
    /// Discards per round delta.
    pub discards: i32,
    /// Hand size delta.
    pub hand_size: i32,
    /// Joker slots delta.
    pub joker_slots: i32,
    /// Consumable slots delta.
    pub consumable_slots: i32,
    /// Starting money delta.
    pub money: i32,
    /// Vouchers owned from the start (redeemed in order).
    pub vouchers: &'static [VoucherId],
    /// Consumables held from the start.
    pub consumables: &'static [ConsumableId],
    /// Starting cards.
    pub composition: DeckComposition,
    /// No interest at cash-out.
    pub no_interest: bool,
    /// Dollars per unused hand at cash-out.
    pub hand_reward: i32,
    /// Dollars per unused discard at cash-out.
    pub discard_reward: i32,
    /// Spectral cards can appear in the shop's card slots.
    pub spectral_in_shop: bool,
    /// Chips and mult are balanced at scoring step 7.
    pub plasma: bool,
    /// Multiplier on every blind's chip requirement.
    pub blind_size_mult: f64,
    /// A Double Tag is granted when a Boss Blind is beaten.
    pub double_tag_after_boss: bool,
}

const BASE: DeckDef = DeckDef {
    hands: 0,
    discards: 0,
    hand_size: 0,
    joker_slots: 0,
    consumable_slots: 0,
    money: 0,
    vouchers: &[],
    consumables: &[],
    composition: DeckComposition::Standard,
    no_interest: false,
    hand_reward: HAND_REWARD,
    discard_reward: DISCARD_REWARD,
    spectral_in_shop: false,
    plasma: false,
    blind_size_mult: 1.0,
    double_tag_after_boss: false,
};

static RED: DeckDef = DeckDef {
    discards: 1,
    ..BASE
};
static BLUE: DeckDef = DeckDef { hands: 1, ..BASE };
static YELLOW: DeckDef = DeckDef {
    money: YELLOW_DECK_MONEY,
    ..BASE
};
static GREEN: DeckDef = DeckDef {
    no_interest: true,
    hand_reward: GREEN_DECK_HAND_REWARD,
    discard_reward: GREEN_DECK_DISCARD_REWARD,
    ..BASE
};
static BLACK: DeckDef = DeckDef {
    hands: -1,
    joker_slots: 1,
    ..BASE
};
static MAGIC: DeckDef = DeckDef {
    vouchers: &[VoucherId::CrystalBall],
    consumables: &[ConsumableId::TheFool, ConsumableId::TheFool],
    ..BASE
};
static NEBULA: DeckDef = DeckDef {
    consumable_slots: -1,
    vouchers: &[VoucherId::Telescope],
    ..BASE
};
static GHOST: DeckDef = DeckDef {
    spectral_in_shop: true,
    consumables: &[ConsumableId::Hex],
    ..BASE
};
static ABANDONED: DeckDef = DeckDef {
    composition: DeckComposition::NoFaces,
    ..BASE
};
static CHECKERED: DeckDef = DeckDef {
    composition: DeckComposition::Checkered,
    ..BASE
};
static ZODIAC: DeckDef = DeckDef {
    vouchers: &[
        VoucherId::TarotMerchant,
        VoucherId::PlanetMerchant,
        VoucherId::Overstock,
    ],
    ..BASE
};
static PAINTED: DeckDef = DeckDef {
    hand_size: PAINTED_DECK_HAND_SIZE,
    joker_slots: -1,
    ..BASE
};
static ANAGLYPH: DeckDef = DeckDef {
    double_tag_after_boss: true,
    ..BASE
};
static PLASMA: DeckDef = DeckDef {
    plasma: true,
    blind_size_mult: PLASMA_BLIND_MULT,
    ..BASE
};
static ERRATIC: DeckDef = DeckDef {
    composition: DeckComposition::Erratic,
    ..BASE
};

impl DeckKind {
    /// All decks, in the order of the wiki's deck table.
    pub const ALL: [DeckKind; 15] = [
        DeckKind::Red,
        DeckKind::Blue,
        DeckKind::Yellow,
        DeckKind::Green,
        DeckKind::Black,
        DeckKind::Magic,
        DeckKind::Nebula,
        DeckKind::Ghost,
        DeckKind::Abandoned,
        DeckKind::Checkered,
        DeckKind::Zodiac,
        DeckKind::Painted,
        DeckKind::Anaglyph,
        DeckKind::Plasma,
        DeckKind::Erratic,
    ];

    /// Number of decks.
    pub const COUNT: usize = Self::ALL.len();

    /// Index in [`DeckKind::ALL`] (0 = Red).
    pub fn index(self) -> usize {
        self as usize
    }

    /// Lowercase name used by the Python constructors.
    pub fn name(self) -> &'static str {
        match self {
            DeckKind::Red => "red",
            DeckKind::Blue => "blue",
            DeckKind::Yellow => "yellow",
            DeckKind::Green => "green",
            DeckKind::Black => "black",
            DeckKind::Magic => "magic",
            DeckKind::Nebula => "nebula",
            DeckKind::Ghost => "ghost",
            DeckKind::Abandoned => "abandoned",
            DeckKind::Checkered => "checkered",
            DeckKind::Zodiac => "zodiac",
            DeckKind::Painted => "painted",
            DeckKind::Anaglyph => "anaglyph",
            DeckKind::Plasma => "plasma",
            DeckKind::Erratic => "erratic",
        }
    }

    /// Parse a (case-insensitive) deck name.
    pub fn from_name(name: &str) -> Option<DeckKind> {
        let n = name.to_ascii_lowercase();
        Self::ALL.into_iter().find(|d| d.name() == n)
    }

    /// What this deck changes.
    pub fn def(self) -> &'static DeckDef {
        match self {
            DeckKind::Red => &RED,
            DeckKind::Blue => &BLUE,
            DeckKind::Yellow => &YELLOW,
            DeckKind::Green => &GREEN,
            DeckKind::Black => &BLACK,
            DeckKind::Magic => &MAGIC,
            DeckKind::Nebula => &NEBULA,
            DeckKind::Ghost => &GHOST,
            DeckKind::Abandoned => &ABANDONED,
            DeckKind::Checkered => &CHECKERED,
            DeckKind::Zodiac => &ZODIAC,
            DeckKind::Painted => &PAINTED,
            DeckKind::Anaglyph => &ANAGLYPH,
            DeckKind::Plasma => &PLASMA,
            DeckKind::Erratic => &ERRATIC,
        }
    }
}

/// Build the starting cards of a deck (unshuffled draw pile). Only [`DeckComposition::Erratic`]
/// consumes randomness (`RngKey::Deck`, 2 draws per card, rank then suit).
pub fn build_deck(composition: DeckComposition, rng: &mut GameRng) -> Deck {
    match composition {
        DeckComposition::Standard => Deck::standard(),
        DeckComposition::NoFaces => {
            let mut uid = 0;
            let cards: Vec<Card> = standard_deck(&mut uid)
                .into_iter()
                .filter(|c| !c.rank.is_face_rank())
                .collect();
            Deck::from_cards(cards, uid)
        }
        DeckComposition::Checkered => {
            let mut uid = 0;
            let mut cards = Vec::with_capacity(52);
            for suit in [Suit::Spades, Suit::Hearts] {
                for _ in 0..2 {
                    for rank in Rank::ALL {
                        cards.push(Card::new(uid, rank, suit));
                        uid += 1;
                    }
                }
            }
            Deck::from_cards(cards, uid)
        }
        DeckComposition::Erratic => {
            let mut cards = Vec::with_capacity(52);
            for uid in 0..52u32 {
                let rank = Rank::ALL[rng.range(RngKey::Deck, 0, Rank::COUNT)];
                let suit = Suit::ALL[rng.range(RngKey::Deck, 0, Suit::COUNT)];
                cards.push(Card::new(uid, rank, suit));
            }
            Deck::from_cards(cards, 52)
        }
    }
}
