//! Content registry: collects every `DEFS` table (jokers, consumables, bosses, vouchers, tags)
//! into id-indexed tables built at compile time (one array index per lookup, no hashing).
//! Shared file: orchestrator-owned.
//!
//! Consumables: `planets.rs`, `tarots.rs`, `spectrals.rs` each provide
//! `pub static DEFS: &[ConsumableDef]`; bosses: `bosses.rs` provides `pub static DEFS: &[BossDef]`.
//! Content authors: follow `docs/design/consumables_bosses.md`.

pub mod bosses;
pub mod decks;
pub mod ids;
pub mod jokers;
pub mod planets;
pub mod spectrals;
pub mod stakes;
pub mod tags;
pub mod tarots;
pub mod vouchers;

pub use ids::{BossId, ConsumableId, TagId, VoucherId};

use crate::boss::BossDef;
use crate::consumable::ConsumableDef;
use crate::tag::TagDef;
use crate::voucher::VoucherDef;

/// Every consumable definition table.
pub static CONSUMABLE_SOURCES: &[&[ConsumableDef]] =
    &[planets::DEFS, tarots::DEFS, spectrals::DEFS];

/// Every boss definition table.
pub static BOSS_SOURCES: &[&[BossDef]] = &[bosses::DEFS];

/// Every voucher definition table.
pub static VOUCHER_SOURCES: &[&[VoucherDef]] = &[vouchers::DEFS];

/// Every tag definition table.
pub static TAG_SOURCES: &[&[TagDef]] = &[tags::DEFS];

static VOUCHER_REGISTRY: [Option<&'static VoucherDef>; VoucherId::COUNT] = build_voucher_registry();

static TAG_REGISTRY: [Option<&'static TagDef>; TagId::COUNT] = build_tag_registry();

static CONSUMABLE_REGISTRY: [Option<&'static ConsumableDef>; ConsumableId::COUNT] =
    build_consumable_registry();

static BOSS_REGISTRY: [Option<&'static BossDef>; BossId::COUNT] = build_boss_registry();

const fn build_consumable_registry() -> [Option<&'static ConsumableDef>; ConsumableId::COUNT] {
    let mut table: [Option<&'static ConsumableDef>; ConsumableId::COUNT] =
        [None; ConsumableId::COUNT];
    let mut s = 0;
    while s < CONSUMABLE_SOURCES.len() {
        let defs: &'static [ConsumableDef] = CONSUMABLE_SOURCES[s];
        let mut i = 0;
        while i < defs.len() {
            table[defs[i].id.index()] = Some(&defs[i]);
            i += 1;
        }
        s += 1;
    }
    table
}

const fn build_voucher_registry() -> [Option<&'static VoucherDef>; VoucherId::COUNT] {
    let mut table: [Option<&'static VoucherDef>; VoucherId::COUNT] = [None; VoucherId::COUNT];
    let mut s = 0;
    while s < VOUCHER_SOURCES.len() {
        let defs: &'static [VoucherDef] = VOUCHER_SOURCES[s];
        let mut i = 0;
        while i < defs.len() {
            table[defs[i].id.index()] = Some(&defs[i]);
            i += 1;
        }
        s += 1;
    }
    table
}

const fn build_tag_registry() -> [Option<&'static TagDef>; TagId::COUNT] {
    let mut table: [Option<&'static TagDef>; TagId::COUNT] = [None; TagId::COUNT];
    let mut s = 0;
    while s < TAG_SOURCES.len() {
        let defs: &'static [TagDef] = TAG_SOURCES[s];
        let mut i = 0;
        while i < defs.len() {
            table[defs[i].id.index()] = Some(&defs[i]);
            i += 1;
        }
        s += 1;
    }
    table
}

const fn build_boss_registry() -> [Option<&'static BossDef>; BossId::COUNT] {
    let mut table: [Option<&'static BossDef>; BossId::COUNT] = [None; BossId::COUNT];
    let mut s = 0;
    while s < BOSS_SOURCES.len() {
        let defs: &'static [BossDef] = BOSS_SOURCES[s];
        let mut i = 0;
        while i < defs.len() {
            table[defs[i].id.index()] = Some(&defs[i]);
            i += 1;
        }
        s += 1;
    }
    table
}

/// Definition of a consumable, `None` while no content file defines it (such a card can be
/// held and sold but not used).
#[inline]
pub fn try_consumable_def(id: ConsumableId) -> Option<&'static ConsumableDef> {
    CONSUMABLE_REGISTRY[id.index()]
}

/// Definition of a boss, `None` while no content file defines it (undefined bosses are never
/// chosen).
#[inline]
pub fn try_boss_def(id: BossId) -> Option<&'static BossDef> {
    BOSS_REGISTRY[id.index()]
}

/// Number of consumable definitions of `id` across all tables (must be 0 or 1; 1 once Phase 4
/// is complete).
pub fn consumable_def_count(id: ConsumableId) -> usize {
    CONSUMABLE_SOURCES
        .iter()
        .flat_map(|s| s.iter())
        .filter(|d| d.id == id)
        .count()
}

/// Number of boss definitions of `id` (must be 0 or 1; 1 once Phase 4 is complete).
pub fn boss_def_count(id: BossId) -> usize {
    BOSS_SOURCES
        .iter()
        .flat_map(|s| s.iter())
        .filter(|d| d.id == id)
        .count()
}

/// Definition of a voucher, `None` while no content file defines it (undefined vouchers are
/// never offered).
#[inline]
pub fn try_voucher_def(id: VoucherId) -> Option<&'static VoucherDef> {
    VOUCHER_REGISTRY[id.index()]
}

/// Definition of a tag, `None` while no content file defines it (an undefined tag can be
/// offered only when no tag is defined; it has no effect).
#[inline]
pub fn try_tag_def(id: TagId) -> Option<&'static TagDef> {
    TAG_REGISTRY[id.index()]
}

/// Number of voucher definitions of `id` (must be 0 or 1; 1 once Phase 5B/5C is complete).
pub fn voucher_def_count(id: VoucherId) -> usize {
    VOUCHER_SOURCES
        .iter()
        .flat_map(|s| s.iter())
        .filter(|d| d.id == id)
        .count()
}

/// Number of tag definitions of `id` (must be 0 or 1; 1 once Phase 5B/5C is complete).
pub fn tag_def_count(id: TagId) -> usize {
    TAG_SOURCES
        .iter()
        .flat_map(|s| s.iter())
        .filter(|d| d.id == id)
        .count()
}
