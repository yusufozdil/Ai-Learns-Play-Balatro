//! Deterministic RNG with keyed sub-streams (PCG64). Stream seeds are derived with FNV-1a
//! from (run seed, key) so that changing one subsystem never shifts another subsystem's sequence.
//! All probability rolls go through `chance(num, den, prob_mult)` (Oops! All 6s hook point).
//! Phase 1A.
