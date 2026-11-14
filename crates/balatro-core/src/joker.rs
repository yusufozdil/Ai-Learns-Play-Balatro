//! Jokers: the per-run [`JokerInstance`] (small, `Copy`) and the static per-kind [`JokerDef`]
//! (rarity, cost, compatibility flags, initial state, hook table).
//!
//! Contract guide for joker authors: `docs/design/hooks.md`. Hook signatures live in
//! [`crate::hooks`], the scoring pipeline in [`crate::scoring`], definitions in
//! [`crate::content::jokers`].

use crate::content::jokers::{JokerId, try_def};
use crate::economy;
use crate::enums::{Edition, HandType, Rank, Rarity, Suit};
use crate::hooks::Hooks;
use serde::{Deserialize, Serialize};

/// Sticker state of a joker (stakes). The game layer applies their effects (eternal: cannot be
/// sold or destroyed; perishable: debuffed when `perish_rounds_left` hits 0; rental: costs money
/// each round). Hooks never need to look at them except to skip eternal jokers when choosing a
/// joker to destroy (Ceremonial Dagger, Madness).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stickers {
    /// Cannot be sold or destroyed.
    pub eternal: bool,
    /// Has the perishable sticker.
    pub perishable: bool,
    /// Rounds left before a perishable joker becomes debuffed.
    pub perish_rounds_left: u8,
    /// Has the rental sticker.
    pub rental: bool,
}

/// One joker owned by the run (or shown in a shop/pack). `Copy`, 32 bytes.
///
/// Generic state fields (meaning is per joker; see the joker's doc comment and
/// `docs/design/hooks.md` "State conventions"):
/// - `value`: the scaling amount the joker currently gives (+mult, +chips or the xmult factor),
///   or a decaying amount (Ice Cream chips, Popcorn mult, Ramen xmult).
/// - `counter`: an integer count/countdown (Seltzer hands left, Yorick discards toward the next
///   step, Invisible rounds held, Loyalty offset).
/// - `aux`: a chosen target, packed in byte lanes via the `aux_*` helpers (suit, rank, hand type).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JokerInstance {
    /// Unique id within the run (assigned by the game layer). Side effects target jokers by uid.
    pub uid: u32,
    /// Which joker this is.
    pub id: JokerId,
    /// Edition (Foil/Holo/Polychrome are applied by the pipeline; Negative is a slot effect).
    pub edition: Edition,
    /// Stickers.
    pub stickers: Stickers,
    /// Debuffed jokers do nothing at all (no hooks, no edition). Baseball Card still counts them.
    pub debuffed: bool,
    /// Current sell value in dollars (Egg and Gift Card raise it).
    pub sell_value: i32,
    /// Generic scaling value (see type docs).
    pub value: f64,
    /// Generic integer counter (see type docs).
    pub counter: u32,
    /// Packed target lanes (see `aux_suit`, `aux_rank`, `aux_hand`, `aux_byte`).
    pub aux: u32,
}

const LANE_SUIT: u32 = 0;
const LANE_RANK: u32 = 8;
const LANE_HAND: u32 = 16;
const LANE_FREE: u32 = 24;

impl JokerInstance {
    /// A fresh instance: base edition, no stickers, state from the definition's `init_*`
    /// fields, sell value from [`JokerDef::default_sell_value`]. Unknown ids get an inert state.
    pub fn new(uid: u32, id: JokerId) -> Self {
        let (value, counter, sell_value) = match try_def(id) {
            Some(d) => (d.init_value, d.init_counter, d.default_sell_value()),
            None => (0.0, 0, 1),
        };
        JokerInstance {
            uid,
            id,
            edition: Edition::Base,
            stickers: Stickers::default(),
            debuffed: false,
            sell_value,
            value,
            counter,
            aux: 0,
        }
    }

    /// Builder: set the edition.
    pub fn with_edition(mut self, edition: Edition) -> Self {
        self.edition = edition;
        self
    }

    /// Builder: set the debuffed flag.
    pub fn with_debuff(mut self, debuffed: bool) -> Self {
        self.debuffed = debuffed;
        self
    }

    /// Builder: set `value`.
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = value;
        self
    }

    /// Builder: set `counter`.
    pub fn with_counter(mut self, counter: u32) -> Self {
        self.counter = counter;
        self
    }

    /// The static definition (`None` only if the registry has no def for this id).
    pub fn def(&self) -> Option<&'static JokerDef> {
        try_def(self.id)
    }

    /// Reset `value`/`counter` to the definition's initial values and clear `aux`.
    pub fn reset_state(&mut self) {
        if let Some(d) = try_def(self.id) {
            self.value = d.init_value;
            self.counter = d.init_counter;
        }
        self.aux = 0;
    }

    fn lane(&self, shift: u32) -> u8 {
        (self.aux >> shift) as u8
    }

    fn set_lane(&mut self, shift: u32, v: u8) {
        self.aux = (self.aux & !(0xFF << shift)) | ((v as u32) << shift);
    }

    /// Chosen suit (Ancient Joker, Castle, The Idol), byte lane 0.
    pub fn aux_suit(&self) -> Option<Suit> {
        let v = self.lane(LANE_SUIT) as usize;
        if v == 0 {
            None
        } else {
            Suit::ALL.get(v - 1).copied()
        }
    }

    /// Set (or clear) the chosen suit.
    pub fn set_aux_suit(&mut self, suit: Option<Suit>) {
        self.set_lane(LANE_SUIT, suit.map_or(0, |s| s.index() as u8 + 1));
    }

    /// Chosen rank (Mail-In Rebate, The Idol), byte lane 1.
    pub fn aux_rank(&self) -> Option<Rank> {
        Rank::from_value(self.lane(LANE_RANK))
    }

    /// Set (or clear) the chosen rank.
    pub fn set_aux_rank(&mut self, rank: Option<Rank>) {
        self.set_lane(LANE_RANK, rank.map_or(0, |r| r as u8));
    }

    /// Chosen hand type (To Do List), byte lane 2.
    pub fn aux_hand(&self) -> Option<HandType> {
        let v = self.lane(LANE_HAND) as usize;
        if v == 0 {
            None
        } else {
            HandType::ALL.get(v - 1).copied()
        }
    }

    /// Set (or clear) the chosen hand type.
    pub fn set_aux_hand(&mut self, hand: Option<HandType>) {
        self.set_lane(LANE_HAND, hand.map_or(0, |h| h.index() as u8 + 1));
    }

    /// Free byte lane 3 for any other small per-joker value.
    pub fn aux_byte(&self) -> u8 {
        self.lane(LANE_FREE)
    }

    /// Set the free byte lane 3.
    pub fn set_aux_byte(&mut self, v: u8) {
        self.set_lane(LANE_FREE, v);
    }
}

/// Copy-target resolver of a copier joker: given the copier's slot and the number of jokers,
/// return the slot whose ability it copies (or `None`).
pub type CopyFn = fn(slot: usize, len: usize) -> Option<usize>;

/// Blueprint: the joker immediately to the right.
pub fn copy_right(slot: usize, len: usize) -> Option<usize> {
    let t = slot + 1;
    (t < len).then_some(t)
}

/// Brainstorm: the leftmost joker.
pub fn copy_leftmost(_slot: usize, len: usize) -> Option<usize> {
    (len > 0).then_some(0)
}

/// Static definition of one joker kind. Build with [`JokerDef::new`] and struct-update syntax:
///
/// ```ignore
/// JokerDef {
///     hooks: Hooks { joker_main: Some(joker_main), ..Hooks::NONE },
///     ..JokerDef::new(JokerId::Joker, "Joker", Rarity::Common, 2)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JokerDef {
    /// Id (exactly one def per id across all `DEFS` tables).
    pub id: JokerId,
    /// Display name (English, for logs/debug).
    pub name: &'static str,
    /// Rarity (Baseball Card reads it).
    pub rarity: Rarity,
    /// Base shop cost in dollars.
    pub base_cost: u32,
    /// Blueprint/Brainstorm can copy this joker (balatrowiki.org/w/Blueprint lists the
    /// incompatible ones). A copier aimed at an incompatible joker gives nothing.
    pub blueprint_compatible: bool,
    /// Can carry the Eternal sticker (false for self-destroying jokers).
    pub eternal_compatible: bool,
    /// Can carry the Perishable sticker.
    pub perishable_compatible: bool,
    /// Initial `JokerInstance::value` (e.g. Ice Cream 100, Hologram 1.0).
    pub init_value: f64,
    /// Initial `JokerInstance::counter` (e.g. Seltzer 10).
    pub init_counter: u32,
    /// `Some` for copiers (Blueprint, Brainstorm): which slot this joker copies.
    pub copies: Option<CopyFn>,
    /// Hook table.
    pub hooks: Hooks,
}

impl JokerDef {
    /// Def with defaults: compatible with Blueprint/Eternal/Perishable, zero initial state,
    /// not a copier, no hooks.
    pub const fn new(id: JokerId, name: &'static str, rarity: Rarity, base_cost: u32) -> Self {
        JokerDef {
            id,
            name,
            rarity,
            base_cost,
            blueprint_compatible: true,
            eternal_compatible: true,
            perishable_compatible: true,
            init_value: 0.0,
            init_counter: 0,
            copies: None,
            hooks: Hooks::NONE,
        }
    }

    /// Sell value of a base-edition copy bought at full price:
    /// [`economy::sell_value`]`(base_cost, 0)` = `max(1, floor(base_cost / 2))`.
    /// TODO(Phase 5): the shop stores `economy::sell_value(price_paid, 0)` at purchase instead
    /// (edition surcharge and discounts included, notes.md §4.1d).
    pub fn default_sell_value(&self) -> i32 {
        economy::sell_value(i32::try_from(self.base_cost).unwrap_or(i32::MAX), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_is_small() {
        assert!(std::mem::size_of::<JokerInstance>() <= 32);
    }

    #[test]
    fn aux_lanes_are_independent() {
        let mut j = JokerInstance::new(1, JokerId::ALL[0]);
        assert_eq!(j.aux_suit(), None);
        assert_eq!(j.aux_rank(), None);
        assert_eq!(j.aux_hand(), None);
        j.set_aux_suit(Some(Suit::Clubs));
        j.set_aux_rank(Some(Rank::Ace));
        j.set_aux_hand(Some(HandType::FlushFive));
        j.set_aux_byte(7);
        assert_eq!(j.aux_suit(), Some(Suit::Clubs));
        assert_eq!(j.aux_rank(), Some(Rank::Ace));
        assert_eq!(j.aux_hand(), Some(HandType::FlushFive));
        assert_eq!(j.aux_byte(), 7);
        j.set_aux_rank(None);
        assert_eq!(j.aux_rank(), None);
        assert_eq!(j.aux_suit(), Some(Suit::Clubs));
        j.set_aux_suit(Some(Suit::Spades));
        assert_eq!(j.aux_suit(), Some(Suit::Spades));
        assert_eq!(j.aux_hand(), Some(HandType::FlushFive));
    }

    #[test]
    fn copy_targets() {
        assert_eq!(copy_right(0, 2), Some(1));
        assert_eq!(copy_right(1, 2), None);
        assert_eq!(copy_leftmost(3, 4), Some(0));
        assert_eq!(copy_leftmost(0, 0), None);
    }
}
