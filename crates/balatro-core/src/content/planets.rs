//! Planet cards (12). Phase 4A owns this file (tests: `tests/consumables_planets_tarots.rs`).
//!
//! All 12 Planets share one effect: level the hand of `ConsumableId::planet_hand()` by 1
//! (consumables.md §1; chips/mult per level live in `data::hand_levels`). Phase 4.0 defined
//! them all because the effect is generic; 4A adds the per-planet tests.
//! Guide: `docs/design/consumables_bosses.md`.

use crate::consumable::{ConsumableDef, UseCtx};
use crate::content::ConsumableId as C;

pub static DEFS: &[ConsumableDef] = &[
    ConsumableDef::new(C::Pluto, "Pluto", planet),
    ConsumableDef::new(C::Mercury, "Mercury", planet),
    ConsumableDef::new(C::Uranus, "Uranus", planet),
    ConsumableDef::new(C::Venus, "Venus", planet),
    ConsumableDef::new(C::Saturn, "Saturn", planet),
    ConsumableDef::new(C::Jupiter, "Jupiter", planet),
    ConsumableDef::new(C::Earth, "Earth", planet),
    ConsumableDef::new(C::Mars, "Mars", planet),
    ConsumableDef::new(C::Neptune, "Neptune", planet),
    ConsumableDef::new(C::PlanetX, "Planet X", planet),
    ConsumableDef::new(C::Ceres, "Ceres", planet),
    ConsumableDef::new(C::Eris, "Eris", planet),
];

// --- Planets: https://balatrowiki.org/w/Planet_Cards ------------------------------------------
const PLANET_LEVELS: i32 = 1;

fn planet(ctx: &mut UseCtx<'_>) {
    if let Some(hand) = ctx.used().planet_hand() {
        ctx.level_up(hand, PLANET_LEVELS);
    }
}
