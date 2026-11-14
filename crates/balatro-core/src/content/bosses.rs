//! Boss blinds (23 regular + 5 Showdown). Phase 4C owns this file (tests: `tests/bosses.rs`).
//!
//! Phase 4.0 prototypes: The Hook, The Wall, The Psychic, The Plant, The Needle, The Flint,
//! The Mark, Verdant Leaf. 4C adds the other 20 to `DEFS` below.
//! Guide: `docs/design/consumables_bosses.md`; rules: docs/mechanics/bosses.md, notes.md §4.1c.

use crate::blinds::{MULT_NEEDLE, MULT_WALL};
use crate::boss::{BossCtx, BossDef, BossHooks, DrawCtx, PlayCtx};
use crate::card::{Card, RuleFlags};
use crate::content::BossId as B;

pub static DEFS: &[BossDef] = &[
    BossDef {
        hooks: BossHooks {
            before_play: Some(hook_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheHook, "The Hook", 1)
    },
    BossDef {
        mult: MULT_WALL,
        ..BossDef::new(B::TheWall, "The Wall", 2)
    },
    BossDef {
        hooks: BossHooks {
            before_play: Some(psychic_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::ThePsychic, "The Psychic", 1)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(plant_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::ThePlant, "The Plant", 4)
    },
    BossDef {
        mult: MULT_NEEDLE,
        keep_mult_when_disabled: true,
        hooks: BossHooks {
            on_blind_start: Some(needle_blind_start),
            on_disable: Some(needle_disable),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheNeedle, "The Needle", 2)
    },
    BossDef {
        halve_base: true,
        ..BossDef::new(B::TheFlint, "The Flint", 2)
    },
    BossDef {
        hooks: BossHooks {
            face_down: Some(mark_face_down),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheMark, "The Mark", 2)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(verdant_debuffs),
            on_joker_sold: Some(verdant_joker_sold),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::VerdantLeaf, "Verdant Leaf", 8)
    },
];

// --- The Hook: https://balatrowiki.org/w/The_Hook ---------------------------------------------
// Before the hand scores, 2 random held cards are discarded (fewer if fewer are held). They fire
// a forced discard event (Green Joker/Ramen react; Burnt/Trading Card do not, notes.md §4.2).
const HOOK_DISCARDS: usize = 2;

fn hook_before_play(ctx: &mut PlayCtx<'_>) {
    if ctx.discard_random_held(HOOK_DISCARDS) > 0 {
        ctx.trigger();
    }
}

// --- The Psychic: https://balatrowiki.org/w/The_Psychic ---------------------------------------
const PSYCHIC_CARDS: usize = 5;

fn psychic_before_play(ctx: &mut PlayCtx<'_>) {
    if ctx.played().len() != PSYCHIC_CARDS {
        ctx.invalidate();
    }
}

// --- The Plant: https://balatrowiki.org/w/The_Plant -------------------------------------------
// Face cards debuffed (Pareidolia makes every card a face card).

fn plant_debuffs(card: &Card, rules: &RuleFlags) -> bool {
    card.is_face(rules)
}

// --- The Needle: https://balatrowiki.org/w/The_Needle -----------------------------------------
// One hand only; requirement x1 even when disabled. `removed` = hands taken away (given back if
// the boss is disabled mid-round).
const NEEDLE_HANDS: u32 = 1;

fn needle_blind_start(ctx: &mut BossCtx<'_>) {
    let hands = ctx.state().hands_left;
    ctx.boss_mut().removed = hands.saturating_sub(NEEDLE_HANDS);
    ctx.set_hands_left(hands.min(NEEDLE_HANDS));
}

fn needle_disable(ctx: &mut BossCtx<'_>) {
    let n = ctx.boss().removed;
    ctx.add_hands(n);
}

// --- The Mark: https://balatrowiki.org/w/The_Mark ---------------------------------------------

fn mark_face_down(ctx: &mut DrawCtx<'_>, card: &Card) -> bool {
    ctx.is_face(card)
}

// --- Verdant Leaf: https://balatrowiki.org/w/Verdant_Leaf -------------------------------------
// Every card debuffed until a joker is sold (any joker, any time during the round).

fn verdant_debuffs(_: &Card, _: &RuleFlags) -> bool {
    true
}

fn verdant_joker_sold(ctx: &mut BossCtx<'_>) {
    ctx.disable();
}
