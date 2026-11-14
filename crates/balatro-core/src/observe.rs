//! Read-only accessors for the observation encoder (Phase 8A): what a *player* can see.
//!
//! Every field of [`GameState`] is public, so the encoder can read everything the player sees
//! directly (money, ante, blind, target, scores, hands/discards left, hand levels and
//! [`crate::data::hand_levels::chips_mult`], joker instances with `value`/`counter`/`aux`/
//! `edition`/`stickers`/`sell_value`, consumables, the market with its prices through
//! [`crate::shop::card_price`] / [`GameState::reroll_price`], pack items, vouchers, tags, the boss
//! id with [`GameState::boss_in_effect`] / `boss_disabled`). This module adds what must NOT be
//! read straight from the state because the player cannot see it, plus a few derived readers.
//!
//! # Information-leak rules for the observation
//!
//! * **Face-down cards** (The House/Wheel/Fish/Mark): read hand cards through
//!   [`Card::visible_card`] (`None` = hidden: encode a "face-down" token). Never read their
//!   rank/suit/enhancement/edition/seal, nor their `uid` (it identifies the card across
//!   rounds). Known residual leak: the hand is sorted by rank, so the *position* of a
//!   face-down card tells its rank bracket (it was sorted with the hidden identity when drawn).
//!   The encoder may drop the hand ordering for face-down cards (they are interchangeable
//!   tokens) but cannot be given a fix here without changing hand order, i.e. game behaviour.
//! * **Draw pile**: only its size and [`GameState::draw_pile_histogram`] (a histogram, like
//!   the in-game deck viewer). Never its order, never `draw_pile[i]`.
//! * **Amber Acorn**: while [`GameState::jokers_hidden`], jokers are face down and shuffled:
//!   read them through [`GameState::visible_joker`] (`None` = hidden); the count
//!   (`jokers.len()`) stays visible. The recurrent policy may remember what it owned.
//! * **Card and joker `uid`s** identify objects across time (a hidden card's `uid` equals its
//!   `uid` seen earlier): do not feed them to the model. Use slots/positions as ids.
//! * **RNG state, `state.shop` future rerolls, pack contents before opening, tags' hidden
//!   hands**: not observable; the market cards, vouchers and pack *kinds* in the shop are.
//! * The fair preview ([`crate::game::Game::preview_play_fair`]) is the only scoring preview
//!   that is safe to use as an observation feature.

use crate::card::Card;
use crate::data::hand_levels::chips_mult;
use crate::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use crate::joker::JokerInstance;
use crate::shop;
use crate::state::GameState;

impl Card {
    /// The card as the player sees it: `None` while it is face down, otherwise the card with
    /// its `uid` cleared (a `uid` would identify the card across rounds, see the module docs).
    /// The `debuffed` flag of a visible card is visible (the game draws it).
    pub fn visible_card(&self) -> Option<Card> {
        if self.face_down {
            None
        } else {
            let mut c = *self;
            c.uid = 0;
            Some(c)
        }
    }
}

/// Counts of a set of cards by rank x suit (Stone cards are counted apart), enhancement,
/// edition and seal. What a deck viewer shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardHistogram {
    /// Cards counted.
    pub total: u32,
    /// Non-Stone cards by `[rank.index()][suit.index()]` (printed rank/suit).
    pub rank_suit: [[u16; Suit::COUNT]; Rank::COUNT],
    /// Stone cards.
    pub stone: u16,
    /// Cards per `Enhancement::index()` (Stone included).
    pub by_enhancement: [u16; Enhancement::COUNT],
    /// Cards per `Edition::index()`.
    pub by_edition: [u16; Edition::COUNT],
    /// Cards per `Seal::index()`.
    pub by_seal: [u16; Seal::COUNT],
}

impl Default for CardHistogram {
    fn default() -> Self {
        CardHistogram {
            total: 0,
            rank_suit: [[0; Suit::COUNT]; Rank::COUNT],
            stone: 0,
            by_enhancement: [0; Enhancement::COUNT],
            by_edition: [0; Edition::COUNT],
            by_seal: [0; Seal::COUNT],
        }
    }
}

impl CardHistogram {
    /// Count `cards`. Allocation-free.
    pub fn from_cards<'c>(cards: impl IntoIterator<Item = &'c Card>) -> Self {
        let mut h = CardHistogram::default();
        for c in cards {
            h.total += 1;
            h.by_enhancement[c.enhancement.index()] += 1;
            h.by_edition[c.edition.index()] += 1;
            h.by_seal[c.seal.index()] += 1;
            if c.is_stone() {
                h.stone += 1;
            } else {
                h.rank_suit[c.rank.index()][c.suit.index()] += 1;
            }
        }
        h
    }

    /// Non-Stone cards of `rank` and `suit`.
    pub fn count(&self, rank: Rank, suit: Suit) -> u16 {
        self.rank_suit[rank.index()][suit.index()]
    }
}

impl GameState {
    /// Amber Acorn is in effect: the jokers are face down (their count is still visible).
    pub fn jokers_hidden(&self) -> bool {
        self.boss_in_effect() && self.boss_state.jokers_hidden
    }

    /// Joker `slot` as the player sees it: `None` for an empty slot or while the jokers are
    /// face down (Amber Acorn).
    pub fn visible_joker(&self, slot: usize) -> Option<&JokerInstance> {
        if self.jokers_hidden() {
            None
        } else {
            self.jokers.get(slot)
        }
    }

    /// The hand as the player sees it, left to right: `None` for a face-down card, see
    /// [`Card::visible_card`].
    pub fn visible_hand(&self) -> impl Iterator<Item = Option<Card>> + '_ {
        self.deck.hand.iter().map(Card::visible_card)
    }

    /// Histogram of the cards still in the draw pile (the deck viewer's view; no order).
    pub fn draw_pile_histogram(&self) -> CardHistogram {
        CardHistogram::from_cards(&self.deck.draw_pile)
    }

    /// Histogram of the discard pile (face-up for the player).
    pub fn discard_histogram(&self) -> CardHistogram {
        CardHistogram::from_cards(&self.deck.discard_pile)
    }

    /// Histogram of the whole deck (every pile).
    pub fn deck_histogram(&self) -> CardHistogram {
        CardHistogram::from_cards(self.deck.iter_all())
    }

    /// Price of the next market reroll (`0` while a free reroll is left).
    pub fn reroll_price(&self) -> i32 {
        shop::reroll_price(self)
    }

    /// Price of the market card in `slot` (`None` for an empty slot).
    pub fn shop_card_price(&self, slot: usize) -> Option<i32> {
        let c = self.shop.cards.get(slot)?.as_ref()?;
        Some(shop::card_price(self, c))
    }

    /// Price of the market pack in `slot` (`None` for an empty slot).
    pub fn shop_pack_price(&self, slot: usize) -> Option<i32> {
        let p = self.shop.packs.get(slot)?.as_ref()?;
        Some(shop::pack_price_of(self, p))
    }

    /// Price of the market voucher in `slot` (`None` for an empty slot).
    pub fn shop_voucher_price(&self, slot: usize) -> Option<i32> {
        let v = self.shop.vouchers.get(slot)?.as_ref()?;
        Some(shop::voucher_price(self, v.id))
    }

    /// Base `(chips, mult)` of a hand type at its current level.
    pub fn hand_chips_mult(&self, hand: HandType) -> (f64, f64) {
        chips_mult(hand, self.hand_levels[hand.index()])
    }
}
