//! Numbers of the stakes and decks (docs/mechanics/stakes.md, docs/mechanics/decks.md).

/// Blue Stake and above: discards removed from every round (stakes.md §1).
pub const BLUE_STAKE_DISCARD_PENALTY: u32 = 1;

/// Chance that a shop / pack joker carries the Eternal sticker (Black Stake and above, stakes.md §2).
pub const ETERNAL_CHANCE: f64 = 0.3;
/// Chance that a shop / pack joker carries the Perishable sticker (Orange Stake and above). One
/// roll decides Eternal and Perishable, so they never combine (stakes.md §2).
pub const PERISHABLE_CHANCE: f64 = 0.3;
/// Chance that a shop / pack joker carries the Rental sticker (Gold Stake, independent roll).
pub const RENTAL_CHANCE: f64 = 0.3;
/// Rounds a Perishable joker works before it is debuffed for good.
pub const PERISHABLE_ROUNDS: u8 = 5;
/// Rental jokers cost this much to buy (stakes.md §2).
pub const RENTAL_BUY_PRICE: i32 = 1;
/// Rent charged per Rental joker at the end of every won round, before the payout.
pub const RENTAL_RENT: i32 = 3;

/// Yellow Deck: extra starting money.
pub const YELLOW_DECK_MONEY: i32 = 10;
/// Green Deck: dollars per unused hand at cash-out (replaces the base $1).
pub const GREEN_DECK_HAND_REWARD: i32 = 2;
/// Green Deck: dollars per unused discard at cash-out.
pub const GREEN_DECK_DISCARD_REWARD: i32 = 1;
/// Plasma Deck: multiplier on every blind's chip requirement.
pub const PLASMA_BLIND_MULT: f64 = 2.0;
/// Painted Deck: extra hand size.
pub const PAINTED_DECK_HAND_SIZE: i32 = 2;
