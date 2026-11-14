//! `balatro-core`: headless, deterministic rules engine for Balatro-style runs.
//!
//! Pure Rust, no Python dependency. See `prd.md` §3 for the architecture and `notes.md` §1 for
//! non-negotiable rules (determinism, no copied game assets/code, errors instead of panics).

pub mod actions;
pub mod agents;
pub mod blinds;
pub mod boss;
pub mod card;
pub mod consumable;
pub mod content;
pub mod data;
pub mod deck;
pub mod economy;
pub mod enums;
pub mod game;
mod game_shop;
pub mod hand_eval;
pub mod hooks;
pub mod joker;
pub mod packs;
pub mod rng;
pub mod scoring;
pub mod serialize;
pub mod shop;
pub mod state;
pub mod tag;
pub mod voucher;

/// Crate version, exposed to Python as a smoke test.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
