//! `JokerId`: one variant per joker kind.
//!
//! PROVISIONAL: only the Phase 1C prototype jokers. The orchestrator regenerates this file from
//! `docs/mechanics/jokers.csv` (Phase 3.0). Adding variants to the `joker_ids!` list is the only
//! change needed; the registry, `ALL` and `COUNT` follow automatically. Variant order defines
//! `index()` (used for observation encoding), so append only, never reorder.

use serde::{Deserialize, Serialize};

macro_rules! joker_ids {
    (@one $v:ident) => { 1 };
    ($($(#[$m:meta])* $v:ident),+ $(,)?) => {
        /// Joker kind.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[repr(u16)]
        pub enum JokerId {
            $($(#[$m])* $v),+
        }

        impl JokerId {
            /// All ids in declaration order.
            pub const ALL: [JokerId; 0 $(+ joker_ids!(@one $v))+] = [$(JokerId::$v),+];
            /// Number of ids.
            pub const COUNT: usize = Self::ALL.len();
            /// Stable 0-based index (position in `ALL`).
            pub const fn index(self) -> usize {
                self as usize
            }
        }
    };
}

joker_ids! {
    /// +4 mult.
    Joker,
    /// +3 mult per scored Diamond.
    GreedyJoker,
    /// +8 mult if the hand contains a Pair.
    JollyJoker,
    /// x3 mult; 1 in 1000 to be destroyed at end of round.
    Cavendish,
    /// Retrigger played 2, 3, 4, 5.
    Hack,
    /// Retrigger held-in-hand effects.
    Mime,
    /// x1.5 mult per held King.
    Baron,
    /// +2x the lowest held rank as mult.
    RaisedFist,
    /// +1 mult per consecutive hand without a scoring face card.
    RideTheBus,
    /// Scored cards permanently gain +5 chips.
    Hiker,
    /// 1 in 4 to level up the played hand.
    SpaceJoker,
    /// +100 chips, -5 per hand played.
    IceCream,
    /// +mult = times this hand type was played this run.
    Supernova,
    /// +2 chips per $1.
    Bull,
    /// Flushes and straights with 4 cards.
    FourFingers,
    /// Doubles all listed probabilities.
    OopsAll6s,
    /// Copies the joker to the right.
    Blueprint,
    /// Copies the leftmost joker.
    Brainstorm,
    /// Uncommon jokers each give x1.5 mult.
    BaseballCard,
    /// +$4 at end of round.
    GoldenJoker,
    /// Levels up the first discarded hand each round.
    BurntJoker,
    /// +$5 if 3+ face cards are discarded at once.
    FacelessJoker,
}
