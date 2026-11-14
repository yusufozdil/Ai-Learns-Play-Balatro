//! Jokers. `ids.rs` (orchestrator-owned, generated from docs/mechanics/jokers.csv) defines
//! [`JokerId`]; each `waveN.rs` provides `pub static DEFS: &[JokerDef]` for its jokers and is
//! listed in [`SOURCES`]. The registry is an id-indexed table built at compile time: lookups are
//! one array index, no hashing.
//!
//! Joker authors: follow `docs/design/hooks.md`.

pub mod ids;
pub mod prototypes;
pub mod wave1a;
pub mod wave1b;
pub mod wave2a;
pub mod wave2b;
pub mod wave3;
pub mod wave4a;
pub mod wave4b;
pub mod wave5;

pub use ids::JokerId;

use crate::joker::JokerDef;

/// Every per-file definition table. Phase 3: append `waveN::DEFS` here (one line per file).
pub static SOURCES: &[&[JokerDef]] = &[
    prototypes::DEFS,
    wave1a::DEFS,
    wave1b::DEFS,
    wave2a::DEFS,
    wave2b::DEFS,
    wave3::DEFS,
    wave4a::DEFS,
    wave4b::DEFS,
    wave5::DEFS,
];

/// `REGISTRY[id.index()]` = the def of `id`, or `None` if no file defines it yet. If an id is
/// defined twice the last one wins here; `tests/joker_prototypes.rs::registry_*` rejects that.
static REGISTRY: [Option<&'static JokerDef>; JokerId::COUNT] = build_registry();

const fn build_registry() -> [Option<&'static JokerDef>; JokerId::COUNT] {
    let mut table: [Option<&'static JokerDef>; JokerId::COUNT] = [None; JokerId::COUNT];
    let mut s = 0;
    while s < SOURCES.len() {
        let defs: &'static [JokerDef] = SOURCES[s];
        let mut i = 0;
        while i < defs.len() {
            let d = &defs[i];
            table[d.id.index()] = Some(d);
            i += 1;
        }
        s += 1;
    }
    table
}

/// Definition of `id`, or `None` if no `DEFS` table defines it (treated as an inert joker).
#[inline]
pub fn try_def(id: JokerId) -> Option<&'static JokerDef> {
    REGISTRY[id.index()]
}

/// Iterate every definition of every source table (duplicates included), in source order.
pub fn all_defs() -> impl Iterator<Item = &'static JokerDef> {
    SOURCES.iter().flat_map(|s| s.iter())
}

/// Number of definitions of `id` across all source tables (must be exactly 1).
pub fn def_count(id: JokerId) -> usize {
    all_defs().filter(|d| d.id == id).count()
}
