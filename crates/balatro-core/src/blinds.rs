//! Pure blind math: base chips per ante, requirement per blind kind, blind rewards.
//! Source: docs/mechanics/blinds.md §2-§4; decisions in notes.md §4.1c.

use crate::data::blind_chips::{
    BASE_CHIPS_ANTE_LE_0, ENDLESS_A, ENDLESS_A8, ENDLESS_B, ENDLESS_C, ENDLESS_SIG_DIGITS,
    GREEN_BASE, MULT_BIG, MULT_SMALL, PURPLE_BASE, REWARD_BIG, REWARD_BOSS, REWARD_SHOWDOWN,
    REWARD_SMALL, WHITE_BASE,
};

pub use crate::data::blind_chips::{
    MULT_BOSS_DEFAULT, MULT_NEEDLE, MULT_VIOLET_VESSEL, MULT_WALL, StakeScaling,
};

/// Which blind of the ante.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlindKind {
    /// Small blind.
    Small,
    /// Big blind.
    Big,
    /// Boss blind.
    Boss,
}

/// Truncate a positive finite value to `digits` significant digits (round toward zero).
/// Non-finite or non-positive inputs are returned unchanged.
fn truncate_sig(x: f64, digits: u32) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    let d = digits as i32;
    let mut shift = x.log10().floor() as i32 - (d - 1);
    let mut m = (x / 10f64.powi(shift)).floor();
    // Guard against log10 being off by one at decade boundaries.
    if m >= 10f64.powi(d) {
        shift += 1;
        m = (x / 10f64.powi(shift)).floor();
    } else if m < 10f64.powi(d - 1) {
        shift -= 1;
        m = (x / 10f64.powi(shift)).floor();
    }
    // Parsing a decimal literal is correctly rounded, so the result equals the literal `m e shift`.
    format!("{}e{}", m as u64, shift).parse().unwrap_or(x)
}

/// Base chips of an ante (before the blind multiplier). Ante <= 0 gives 100; antes above 8 use the
/// endless formula truncated to 2 significant digits (`f64::INFINITY` from ante 39).
/// Green/Purple only change antes 1..=8; endless uses the White formula for all stakes (the docs
/// give no other formula).
pub fn base_chips(ante: i32, scaling: StakeScaling) -> f64 {
    if ante <= 0 {
        return BASE_CHIPS_ANTE_LE_0;
    }
    if ante <= 8 {
        let table = match scaling {
            StakeScaling::White => &WHITE_BASE,
            StakeScaling::Green => &GREEN_BASE,
            StakeScaling::Purple => &PURPLE_BASE,
        };
        return table[(ante - 1) as usize];
    }
    let c = f64::from(ante - 8);
    let raw = ENDLESS_A8 * (ENDLESS_A + (ENDLESS_B * c).powf(1.0 + ENDLESS_C * c)).powf(c);
    if raw.is_finite() {
        truncate_sig(raw, ENDLESS_SIG_DIGITS)
    } else {
        f64::INFINITY
    }
}

/// Chip requirement of a blind: truncated base times the multiplier (notes.md §4.1c).
/// `boss_mult` is only used for `BlindKind::Boss` (default [`MULT_BOSS_DEFAULT`]; the caller passes
/// [`MULT_WALL`], [`MULT_VIOLET_VESSEL`] or [`MULT_NEEDLE`] for those bosses, and the default when
/// Chicot disables Wall/Vessel).
pub fn requirement(ante: i32, kind: BlindKind, boss_mult: f64, scaling: StakeScaling) -> f64 {
    let mult = match kind {
        BlindKind::Small => MULT_SMALL,
        BlindKind::Big => MULT_BIG,
        BlindKind::Boss => boss_mult,
    };
    base_chips(ante, scaling) * mult
}

/// Money reward for beating a blind (blinds.md §4). `stake_no_small_reward`: Red Stake and above.
pub fn blind_reward(kind: BlindKind, is_showdown: bool, stake_no_small_reward: bool) -> i32 {
    match kind {
        BlindKind::Small if stake_no_small_reward => 0,
        BlindKind::Small => REWARD_SMALL,
        BlindKind::Big => REWARD_BIG,
        BlindKind::Boss if is_showdown => REWARD_SHOWDOWN,
        BlindKind::Boss => REWARD_BOSS,
    }
}
