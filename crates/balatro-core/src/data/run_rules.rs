//! Run-start values (Red Deck, White Stake) and per-round limits of the game loop.
//! Sources: docs/mechanics/blinds.md §1, §5-§7.
//! Starting money lives in [`crate::data::prices::STARTING_MONEY`].

/// Ante a run starts at (blinds.md §1).
pub const STARTING_ANTE: i32 = 1;
/// Hands per round before deck/voucher/joker changes.
pub const BASE_HANDS: u32 = 4;
/// Discards per round before deck/voucher/joker changes.
pub const BASE_DISCARDS: u32 = 3;
/// Red Deck: +1 discard every round (blinds.md §7).
pub const RED_DECK_EXTRA_DISCARDS: u32 = 1;
/// Hand size.
pub const BASE_HAND_SIZE: u32 = 8;
/// Joker slots.
pub const BASE_JOKER_SLOTS: u32 = 5;
/// Consumable slots.
pub const BASE_CONSUMABLE_SLOTS: u32 = 2;
/// Cards in the standard starting deck.
pub const STANDARD_DECK_SIZE: u32 = 52;

/// The run is won by defeating the boss of this ante (blinds.md §1).
pub const WIN_ANTE: i32 = 8;
/// Showdown (finisher) bosses appear when `ante % SHOWDOWN_ANTE_INTERVAL == 0` (blinds.md §1).
pub const SHOWDOWN_ANTE_INTERVAL: i32 = 8;

/// At most this many cards may be played or discarded at once (blinds.md §6.3).
pub const MAX_SELECTED_CARDS: u32 = 5;
/// The effective hand size (cards dealt up to) is clamped to this. Since Phase 8A this is a game
/// rule limit only: it no longer comes from the width of card-selection masks (see
/// [`MAX_SELECTABLE_HAND`]). Kept at 16 so every run behaves as before the mask widening.
pub const MAX_HAND_SIZE: u32 = 16;
/// Hand positions a card-selection mask can address (`u32` bitmasks, bit i = hand card i). The
/// hand may hold more cards than the hand size (Certificate, DNA, Familiar/Grim/Incantation add
/// cards to it), so this is larger than [`MAX_HAND_SIZE`]; cards past position 31 (a hand over
/// 32 cards, practically unreachable) cannot be selected.
pub const MAX_SELECTABLE_HAND: usize = 32;
/// Hand size never drops below this (assumption: a round always draws at least one card).
pub const MIN_HAND_SIZE: u32 = 1;
/// Hands per round never drop below this (assumption: Troubadour & co. cannot zero them).
pub const MIN_HANDS_PER_ROUND: u32 = 1;

/// Safety cap on side-effect cascades (an effect fires an event that queues effects, ...).
/// Base-game chains are at most 2-3 deep; leftovers beyond the cap are dropped and counted.
pub const MAX_EFFECT_CASCADE: u32 = 8;
