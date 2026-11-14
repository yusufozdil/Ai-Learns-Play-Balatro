//! Spectral cards (18). Phase 4B owns this file (tests: `tests/consumables_spectrals.rs`).
//!
//! Phase 4.0 prototypes: Black Hole, Ankh. 4B adds the other 16 to `DEFS` below.
//! Guide: `docs/design/consumables_bosses.md`.

use crate::consumable::{ConsumableDef, UseCheck, UseCtx};
use crate::content::ConsumableId as C;
use crate::enums::HandType;

pub static DEFS: &[ConsumableDef] = &[
    ConsumableDef::new(C::BlackHole, "Black Hole", black_hole),
    ConsumableDef {
        can_use: Some(ankh_can_use),
        ..ConsumableDef::new(C::Ankh, "Ankh", ankh)
    },
];

// --- Black Hole: https://balatrowiki.org/w/Black_Hole -----------------------------------------
const BLACK_HOLE_LEVELS: i32 = 1;

fn black_hole(ctx: &mut UseCtx<'_>) {
    for ht in HandType::ALL {
        ctx.level_up(ht, BLACK_HOLE_LEVELS);
    }
}

// --- Ankh: https://balatrowiki.org/w/Ankh -----------------------------------------------------
// Copy a random joker (any, Eternal included; the copy loses Negative), destroy every other
// non-Eternal joker. Needs at least one joker and a free joker slot (consumables.md §3: the
// slot check happens before the destruction). Destructions are queued BEFORE the copy so the
// copy's room check sees the freed slots.

fn ankh_can_use(c: &UseCheck<'_>) -> bool {
    !c.jokers().is_empty() && c.free_joker_slots() > 0
}

fn ankh(ctx: &mut UseCtx<'_>) {
    let n = ctx.jokers().len();
    if n == 0 {
        return;
    }
    let keep = ctx.rng_range(0, n);
    for slot in 0..n {
        if slot != keep {
            ctx.destroy_joker(slot); // refuses Eternal jokers
        }
    }
    ctx.copy_joker(keep, true);
}
