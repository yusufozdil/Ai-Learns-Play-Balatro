//! Permanent deck + per-round piles (draw/hand/played/discard), drawing and reshuffling.
//! Cards are owned by exactly one pile; the full deck is the union of all piles.

use crate::card::Card;
use crate::enums::{Rank, Suit};
use crate::rng::{GameRng, RngKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Deck operation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DeckError {
    /// Selection mask had no bits set.
    #[error("empty selection mask")]
    EmptyMask,
    /// Mask referenced a hand position that does not exist.
    #[error("mask {mask:#b} selects positions beyond hand size {hand_len}")]
    MaskOutOfRange {
        /// Offending mask.
        mask: u16,
        /// Current hand size.
        hand_len: usize,
    },
}

/// Which pile to add a card to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pile {
    /// Draw pile.
    Draw,
    /// Hand.
    Hand,
    /// Played cards.
    Played,
    /// Discard pile.
    Discard,
}

/// Build a standard 52-card deck, allocating uids from `next_uid` (incremented per card).
/// Order: suits in `Suit::ALL` order, ranks Two..Ace.
pub fn standard_deck(next_uid: &mut u32) -> Vec<Card> {
    let mut v = Vec::with_capacity(52);
    for &suit in Suit::ALL.iter() {
        for &rank in Rank::ALL.iter() {
            v.push(Card::new(*next_uid, rank, suit));
            *next_uid += 1;
        }
    }
    v
}

/// All card piles of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deck {
    /// Draw pile; the top is the END of the vector.
    pub draw_pile: Vec<Card>,
    /// Cards in hand (position = action index).
    pub hand: Vec<Card>,
    /// Cards currently played/scoring.
    pub played: Vec<Card>,
    /// Discard pile.
    pub discard_pile: Vec<Card>,
    next_uid: u32,
}

fn sort_key(c: &Card) -> (bool, std::cmp::Reverse<u8>, usize, u32) {
    (
        c.is_stone(),
        std::cmp::Reverse(c.rank as u8),
        c.suit.index(),
        c.uid,
    )
}

impl Deck {
    /// Fresh standard deck, all cards in the draw pile (unshuffled).
    pub fn standard() -> Self {
        let mut next_uid = 0;
        let draw_pile = standard_deck(&mut next_uid);
        Deck {
            draw_pile,
            hand: Vec::new(),
            played: Vec::new(),
            discard_pile: Vec::new(),
            next_uid,
        }
    }

    /// Total number of cards across all piles.
    pub fn len(&self) -> usize {
        self.draw_pile.len() + self.hand.len() + self.played.len() + self.discard_pile.len()
    }

    /// True if there are no cards at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate all cards (draw, hand, played, discard order).
    pub fn iter_all(&self) -> impl Iterator<Item = &Card> {
        self.draw_pile
            .iter()
            .chain(&self.hand)
            .chain(&self.played)
            .chain(&self.discard_pile)
    }

    /// The uid the next allocation will return.
    pub fn next_uid(&self) -> u32 {
        self.next_uid
    }

    /// Allocate a fresh unique uid.
    pub fn alloc_uid(&mut self) -> u32 {
        let u = self.next_uid;
        self.next_uid += 1;
        u
    }

    /// Move all cards into the draw pile, clear `debuffed`/`face_down`, shuffle with `RngKey::Deck`.
    /// Piles are first merged in a fixed order (draw, hand, played, discard) so the result is
    /// deterministic for a given RNG state.
    pub fn reset_for_round(&mut self, rng: &mut GameRng) {
        let mut hand = std::mem::take(&mut self.hand);
        let mut played = std::mem::take(&mut self.played);
        let mut disc = std::mem::take(&mut self.discard_pile);
        self.draw_pile.append(&mut hand);
        self.draw_pile.append(&mut played);
        self.draw_pile.append(&mut disc);
        for c in &mut self.draw_pile {
            c.debuffed = false;
            c.face_down = false;
        }
        // Canonical order first so the shuffle does not depend on prior pile history.
        self.draw_pile.sort_unstable_by_key(|c| c.uid);
        rng.shuffle(RngKey::Deck, &mut self.draw_pile);
    }

    /// Draw up to `n` cards from the top (end) of the draw pile into the hand, then sort the
    /// hand. Returns the number actually drawn.
    pub fn draw(&mut self, n: usize) -> usize {
        let k = n.min(self.draw_pile.len());
        for _ in 0..k {
            if let Some(c) = self.draw_pile.pop() {
                self.hand.push(c);
            }
        }
        self.sort_hand();
        k
    }

    /// Draw until the hand holds `hand_size` cards (or the pile is empty). Returns cards drawn.
    pub fn draw_to_hand_size(&mut self, hand_size: usize) -> usize {
        self.draw(hand_size.saturating_sub(self.hand.len()))
    }

    /// Sort hand: rank descending, then suit (Spades, Hearts, Clubs, Diamonds), then uid;
    /// Stone cards last.
    pub fn sort_hand(&mut self) {
        self.hand.sort_unstable_by_key(sort_key);
    }

    /// Remove and return the hand cards selected by `mask` (bit i = position i), in left-to-right
    /// order. Errors on an empty mask or bits beyond the hand size; the hand is unchanged on error.
    pub fn take_from_hand(&mut self, mask: u16) -> Result<Vec<Card>, DeckError> {
        if mask == 0 {
            return Err(DeckError::EmptyMask);
        }
        let n = self.hand.len();
        if n < 16 && (mask >> n) != 0 {
            return Err(DeckError::MaskOutOfRange { mask, hand_len: n });
        }
        let mut taken = Vec::with_capacity(mask.count_ones() as usize);
        // Compact the kept cards in place (order preserved).
        let mut kept = 0;
        for i in 0..n {
            let c = self.hand[i];
            if i < 16 && (mask >> i) & 1 == 1 {
                taken.push(c);
            } else {
                self.hand[kept] = c;
                kept += 1;
            }
        }
        self.hand.truncate(kept);
        Ok(taken)
    }

    /// Move all played cards to the discard pile.
    pub fn move_played_to_discard(&mut self) {
        self.discard_pile.append(&mut self.played);
    }

    /// Discard the hand cards selected by `mask`; returns them. Same errors as `take_from_hand`.
    pub fn discard_from_hand(&mut self, mask: u16) -> Result<Vec<Card>, DeckError> {
        let taken = self.take_from_hand(mask)?;
        self.discard_pile.extend_from_slice(&taken);
        Ok(taken)
    }

    /// Permanently remove a card by uid from any pile (destroyed cards).
    pub fn remove_card(&mut self, uid: u32) -> Option<Card> {
        for pile in [
            &mut self.draw_pile,
            &mut self.hand,
            &mut self.played,
            &mut self.discard_pile,
        ] {
            if let Some(i) = pile.iter().position(|c| c.uid == uid) {
                return Some(pile.remove(i));
            }
        }
        None
    }

    /// Add a card to a pile (pushed at the end; for `Draw` that is the top of the pile).
    /// Adding to the hand does not re-sort; call `sort_hand` if needed.
    pub fn add_card(&mut self, card: Card, pile: Pile) {
        match pile {
            Pile::Draw => self.draw_pile.push(card),
            Pile::Hand => self.hand.push(card),
            Pile::Played => self.played.push(card),
            Pile::Discard => self.discard_pile.push(card),
        }
    }
}
