//! Content registry: collects every `DEFS` table (jokers, consumables, bosses, vouchers, tags)
//! and validates that each id has exactly one definition. Shared file: orchestrator-owned.

pub mod bosses;
pub mod decks;
pub mod jokers;
pub mod planets;
pub mod spectrals;
pub mod stakes;
pub mod tags;
pub mod tarots;
pub mod vouchers;
