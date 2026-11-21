//! Ante -> base chip requirement per stake, and blind multipliers.
//! Source: docs/mechanics/blinds.md §2 (balatrowiki.org/w/Blinds_and_Antes), §3, §4.

use serde::{Deserialize, Serialize};

/// Stake-dependent blind chip scaling. Only the base-chip table differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum StakeScaling {
    /// White Stake (Red Stake only changes rewards): the default table.
    #[default]
    White,
    /// Green Stake and above (blinds.md §2 "Green+").
    Green,
    /// Purple Stake and above (blinds.md §2 "Purple+").
    Purple,
}

/// Base chips for ante <= 0 (Hieroglyph / Petroglyph), all stakes (blinds.md §2).
pub const BASE_CHIPS_ANTE_LE_0: f64 = 100.0;

/// White Stake base chips for antes 1..=8 (blinds.md §2).
pub const WHITE_BASE: [f64; 8] = [
    300.0, 800.0, 2_000.0, 5_000.0, 11_000.0, 20_000.0, 35_000.0, 50_000.0,
];
/// Green+ base chips for antes 1..=8 (blinds.md §2).
pub const GREEN_BASE: [f64; 8] = [
    300.0, 900.0, 2_600.0, 8_000.0, 20_000.0, 36_000.0, 60_000.0, 100_000.0,
];
/// Purple+ base chips for antes 1..=8 (blinds.md §2).
pub const PURPLE_BASE: [f64; 8] = [
    300.0, 1_000.0, 3_200.0, 9_000.0, 25_000.0, 60_000.0, 110_000.0, 200_000.0,
];

/// Endless formula constants (blinds.md §2 "Endless"): `A8 * (A + (B*c)^(1 + C*c))^c`, c = ante-8.
pub const ENDLESS_A8: f64 = 50_000.0;
/// See [`ENDLESS_A8`].
pub const ENDLESS_A: f64 = 1.6;
/// See [`ENDLESS_A8`].
pub const ENDLESS_B: f64 = 0.75;
/// See [`ENDLESS_A8`].
pub const ENDLESS_C: f64 = 0.2;
/// Significant digits kept in endless base (blinds.md §2/§3: truncate).
pub const ENDLESS_SIG_DIGITS: u32 = 2;

/// Small blind multiplier (blinds.md §3).
pub const MULT_SMALL: f64 = 1.0;
/// Big blind multiplier (blinds.md §3).
pub const MULT_BIG: f64 = 1.5;
/// Default boss multiplier (blinds.md §3).
pub const MULT_BOSS_DEFAULT: f64 = 2.0;
/// The Wall boss multiplier (blinds.md §3). Chicot reverts it to the default.
pub const MULT_WALL: f64 = 4.0;
/// Violet Vessel boss multiplier (blinds.md §3). Chicot reverts it to the default.
pub const MULT_VIOLET_VESSEL: f64 = 6.0;
/// The Needle boss multiplier (blinds.md §3); unchanged by Chicot.
pub const MULT_NEEDLE: f64 = 1.0;

/// Small blind reward in dollars (blinds.md §4).
pub const REWARD_SMALL: i32 = 3;
/// Big blind reward in dollars (blinds.md §4).
pub const REWARD_BIG: i32 = 4;
/// Boss blind reward in dollars (blinds.md §4).
pub const REWARD_BOSS: i32 = 5;
/// Showdown boss reward in dollars (blinds.md §4).
pub const REWARD_SHOWDOWN: i32 = 8;
