//! Deterministic RNG with independent keyed streams (PCG64 MCG). Stream seeds are derived with
//! FNV-1a from (run seed, key) so that consuming one subsystem never shifts another's sequence.
//! All probability rolls go through [`GameRng::chance`] (Oops! All 6s hook point).

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};

/// FNV-1a 64-bit hash (stable across platforms; never use `DefaultHasher`).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Independent random streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum RngKey {
    /// Deck shuffles.
    Deck,
    /// Shop generation.
    Shop,
    /// Booster packs.
    Packs,
    /// Boss blind selection.
    Bosses,
    /// Tags.
    Tags,
    /// Joker effects/generation.
    Jokers,
    /// Card effects (Glass, Lucky, ...).
    Cards,
    /// Consumable generation/effects.
    Consumables,
    /// Everything else.
    Misc,
}

impl RngKey {
    /// All keys.
    pub const ALL: [RngKey; 9] = [
        RngKey::Deck,
        RngKey::Shop,
        RngKey::Packs,
        RngKey::Bosses,
        RngKey::Tags,
        RngKey::Jokers,
        RngKey::Cards,
        RngKey::Consumables,
        RngKey::Misc,
    ];
    /// Number of keys.
    pub const COUNT: usize = Self::ALL.len();
    /// Stable name used in seed derivation (do not change).
    pub fn name(self) -> &'static str {
        match self {
            RngKey::Deck => "deck",
            RngKey::Shop => "shop",
            RngKey::Packs => "packs",
            RngKey::Bosses => "bosses",
            RngKey::Tags => "tags",
            RngKey::Jokers => "jokers",
            RngKey::Cards => "cards",
            RngKey::Consumables => "consumables",
            RngKey::Misc => "misc",
        }
    }
}

/// Game RNG: one independent PCG stream per [`RngKey`]. Fully serializable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameRng {
    seed: u64,
    streams: [Pcg64Mcg; RngKey::COUNT],
}

impl GameRng {
    /// Create from a numeric seed.
    pub fn new(seed: u64) -> Self {
        let streams = std::array::from_fn(|i| {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&seed.to_le_bytes());
            bytes.extend_from_slice(RngKey::ALL[i].name().as_bytes());
            Pcg64Mcg::seed_from_u64(fnv1a64(&bytes))
        });
        GameRng { seed, streams }
    }

    /// Create from a string seed (FNV-1a of the bytes).
    pub fn from_seed_str(s: &str) -> Self {
        Self::new(fnv1a64(s.as_bytes()))
    }

    /// The run seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Next raw 64 bits from a stream.
    pub fn next_u64(&mut self, key: RngKey) -> u64 {
        self.streams[key as usize].next_u64()
    }

    /// Uniform float in [0, 1) with 53 bits of precision.
    pub fn float(&mut self, key: RngKey) -> f64 {
        (self.next_u64(key) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[lo, hi_exclusive)` (unbiased, rejection sampling).
    /// Returns `lo` if the range is empty (`hi_exclusive <= lo`).
    pub fn range(&mut self, key: RngKey, lo: usize, hi_exclusive: usize) -> usize {
        if hi_exclusive <= lo {
            return lo;
        }
        let span = (hi_exclusive - lo) as u64;
        // Reject values in the incomplete final bucket.
        let zone = u64::MAX - (u64::MAX % span + 1) % span;
        loop {
            let x = self.next_u64(key);
            if x <= zone {
                return lo + (x % span) as usize;
            }
        }
    }

    /// Fisher-Yates shuffle: for i from len-1 down to 1, swap `i` with a uniform `j` in `0..=i`.
    pub fn shuffle<T>(&mut self, key: RngKey, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.range(key, 0, i + 1);
            items.swap(i, j);
        }
    }

    /// Uniform index in `0..len`, or `None` if `len == 0`.
    pub fn choose_index(&mut self, key: RngKey, len: usize) -> Option<usize> {
        if len == 0 {
            None
        } else {
            Some(self.range(key, 0, len))
        }
    }

    /// Weighted index; non-positive (and NaN) weights are ignored. `None` if no positive weight.
    pub fn weighted_index(&mut self, key: RngKey, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
        if total.is_nan() || total <= 0.0 || !total.is_finite() {
            return None;
        }
        let target = self.float(key) * total;
        let mut acc = 0.0;
        let mut last = None;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                acc += w;
                last = Some(i);
                if target < acc {
                    return Some(i);
                }
            }
        }
        last // floating-point rounding fallback
    }

    /// Probability roll: true with probability `min(1, numerator * numerator_mult / denominator)`.
    /// `numerator_mult` models "Oops! All 6s" (2 per copy doubling). Always consumes one float
    /// so the stream advances identically regardless of outcome. Zero denominator -> false.
    pub fn chance(
        &mut self,
        key: RngKey,
        numerator: u32,
        denominator: u32,
        numerator_mult: u32,
    ) -> bool {
        let f = self.float(key);
        if denominator == 0 {
            return false;
        }
        let p = (numerator as f64 * numerator_mult as f64 / denominator as f64).min(1.0);
        f < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_bounds() {
        let mut r = GameRng::new(1);
        for _ in 0..1000 {
            let v = r.range(RngKey::Misc, 5, 8);
            assert!((5..8).contains(&v));
        }
    }
}
