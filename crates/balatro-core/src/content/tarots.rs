//! Tarot cards (22). Phase 4A owns this file (tests: `tests/consumables_planets_tarots.rs`).
//!
//! Phase 4.0 prototypes: The Fool, The Magician, The Hermit, Death. 4A adds the other 18 to
//! `DEFS` below. Guide: `docs/design/consumables_bosses.md`.

use crate::consumable::{ConsumableDef, UseCheck, UseCtx};
use crate::content::ConsumableId as C;
use crate::enums::Enhancement;

pub static DEFS: &[ConsumableDef] = &[
    ConsumableDef {
        can_use: Some(fool_can_use),
        ..ConsumableDef::new(C::TheFool, "The Fool", fool)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 2,
        ..ConsumableDef::new(C::TheMagician, "The Magician", magician)
    },
    ConsumableDef::new(C::TheHermit, "The Hermit", hermit),
    ConsumableDef {
        min_targets: 2,
        max_targets: 2,
        ..ConsumableDef::new(C::Death, "Death", death)
    },
];

// --- The Fool: https://balatrowiki.org/w/The_Fool --------------------------------------------
// Copies the last Tarot/Planet used this run (never itself; Spectrals never count; the game
// layer keeps `GameState::last_tarot_planet`). Needs a free slot after it leaves its own.

fn fool_can_use(c: &UseCheck<'_>) -> bool {
    c.state().last_tarot_planet.is_some() && c.free_consumable_slots() > 0
}

fn fool(ctx: &mut UseCtx<'_>) {
    if let Some(id) = ctx.state().last_tarot_planet {
        ctx.add_consumable(id);
    }
}

// --- The Magician: https://balatrowiki.org/w/The_Magician ------------------------------------

fn magician(ctx: &mut UseCtx<'_>) {
    for k in 0..ctx.target_count() {
        if let Some(pos) = ctx.target_pos(k) {
            ctx.set_enhancement(pos, Enhancement::Lucky);
        }
    }
}

// --- The Hermit: https://balatrowiki.org/w/The_Hermit ----------------------------------------
// Doubles money, at most +$20; $0 when money <= 0 (notes.md §4.1b).
const HERMIT_CAP: i32 = 20;

fn hermit(ctx: &mut UseCtx<'_>) {
    let gain = ctx.money().clamp(0, HERMIT_CAP);
    ctx.add_money(gain);
}

// --- Death: https://balatrowiki.org/w/Death ---------------------------------------------------
// The LEFT selected card becomes a copy of the RIGHT one (hand order; notes.md §4.1b/§4.2).

fn death(ctx: &mut UseCtx<'_>) {
    if let (Some(left), Some(right)) = (ctx.target_pos(0), ctx.target_pos(1)) {
        ctx.copy_card(left, right);
    }
}
