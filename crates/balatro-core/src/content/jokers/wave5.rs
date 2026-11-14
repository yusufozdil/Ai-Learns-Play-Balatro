//! Phase 3 joker package `wave5`: Luchador, MrBones, SmearedJoker, Showman, InvisibleJoker, Chicot.
//!
//! These jokers act on the game layer rather than on the score: they emit `DisableBoss`,
//! `PreventDeath` and `CopyJoker` side effects or set passive flags that later systems read
//! (`RuleFlags::smeared` for hand evaluation and suit tests, `Modifiers::allow_duplicates` for
//! the Phase 5 shop/pack generators).
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave5.rs`.

use crate::blinds::BlindKind;
use crate::content::jokers::JokerId;
use crate::enums::Rarity;
use crate::hooks::{EventCtx, Hooks, Modifiers, RoundEndInfo, SideEffect};
use crate::joker::{JokerDef, JokerInstance};

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        // Copyable (wiki: selling a Blueprint/Brainstorm that copies Luchador also disables the
        // boss; `fire_sell_self` resolves copiers). Sold to work, so never Eternal.
        eternal_compatible: false,
        hooks: Hooks {
            on_sell_self: Some(luchador_sell_self),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Luchador, "Luchador", Rarity::Uncommon, 5)
    },
    JokerDef {
        blueprint_compatible: false,
        eternal_compatible: false,
        hooks: Hooks {
            on_round_end: Some(mr_bones_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MrBones, "Mr. Bones", Rarity::Uncommon, 5)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(smeared_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SmearedJoker, "Smeared Joker", Rarity::Uncommon, 7)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(showman_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Showman, "Showman", Rarity::Uncommon, 5)
    },
    JokerDef {
        blueprint_compatible: false,
        eternal_compatible: false,
        hooks: Hooks {
            on_acquire: Some(invisible_acquire),
            on_round_end: Some(invisible_round_end),
            on_sell_self: Some(invisible_sell_self),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::InvisibleJoker, "Invisible Joker", Rarity::Rare, 8)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_acquire: Some(chicot_acquire),
            on_blind_select: Some(chicot_blind_select),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Chicot, "Chicot", Rarity::Legendary, 20)
    },
];

// --- Luchador: https://balatrowiki.org/w/Luchador -------------------------------------------
// "Sell this card to disable the current Boss Blind." Only during a Boss Blind whose effect is
// still active (a boss can be disabled once per round). Debuffed Luchador sells for its price but
// does nothing (the dispatcher skips debuffed jokers).
//
// `run.in_round` tells "inside the Boss Blind round" from "in the shop before it" (during the
// shop `run.blind` is already the next blind), so a shop sale never disables the next boss.

fn luchador_sell_self(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let run = ctx.run();
    if run.in_round && run.blind == BlindKind::Boss && run.boss.active {
        ctx.push_effect(SideEffect::DisableBoss);
    }
}

// --- Mr. Bones: https://balatrowiki.org/w/Mr._Bones -----------------------------------------
// When a round is lost with at least 25% of the required chips scored, prevent the game over
// and destroy itself. The game layer then treats the blind as won without the blind reward
// (interest is still paid) and ignores the save when the run is lost by running out of cards.
// If several Mr. Bones are owned, only the leftmost one is spent (the save is already queued
// when the others run).
const MR_BONES_SAVE_FRACTION: f64 = 0.25;

fn mr_bones_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, info: &RoundEndInfo) {
    if info.won || info.round_score < info.target * MR_BONES_SAVE_FRACTION {
        return;
    }
    let already_saved = ctx
        .effects()
        .iter()
        .any(|e| matches!(e, SideEffect::PreventDeath));
    if !already_saved {
        ctx.push_effect(SideEffect::PreventDeath);
        ctx.destroy_self();
    }
}

// --- Smeared Joker: https://balatrowiki.org/w/Smeared_Joker ---------------------------------
// Hearts == Diamonds and Spades == Clubs for hand evaluation and every suit test
// (`Card::has_suit`: Wild still matches every suit, a debuffed Wild reverts to its printed suit
// which is then smeared, Stone has no suit).

fn smeared_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.rules.smeared = true;
}

// --- Showman: https://balatrowiki.org/w/Showman ---------------------------------------------
// Joker, Tarot, Planet and Spectral cards may appear multiple times (shop, packs, and card
// generators such as Riff-raff / The Emperor). Read by Phase 5 via `Modifiers::allow_duplicates`.

fn showman_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.allow_duplicates = true;
}

// --- Invisible Joker: https://balatrowiki.org/w/Invisible_Joker -----------------------------
// counter = rounds held (0..=2, counted at every round end). After 2 rounds, selling it
// duplicates a random OTHER joker; the copy loses Negative (the game layer copies state,
// edition and stickers and fires `on_acquire` on the copy). No other joker: nothing happens.
// A copy of an Invisible Joker restarts at 0/2 (`on_acquire` resets the counter).
const INVISIBLE_ROUNDS: u32 = 2;

fn invisible_acquire(j: &mut JokerInstance, _: &mut EventCtx<'_>) {
    j.counter = 0;
}

fn invisible_round_end(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: &RoundEndInfo) {
    j.counter = (j.counter + 1).min(INVISIBLE_ROUNDS);
}

fn invisible_sell_self(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if j.counter < INVISIBLE_ROUNDS {
        return;
    }
    let me = ctx.slot();
    let eligible = |i: usize, ctx: &EventCtx<'_>| i != me && !ctx.is_doomed(i);
    let n = (0..ctx.jokers().len())
        .filter(|&i| eligible(i, ctx))
        .count();
    if n == 0 {
        return;
    }
    let pick = ctx.rng_range(0, n);
    let target = (0..ctx.jokers().len())
        .filter(|&i| eligible(i, ctx))
        .nth(pick)
        .and_then(|i| ctx.jokers().get(i))
        .map(|t| t.uid);
    if let Some(uid) = target {
        ctx.push_effect(SideEffect::CopyJoker {
            uid,
            strip_negative: true,
        });
    }
}

// --- Chicot: https://balatrowiki.org/w/Chicot -----------------------------------------------
// "Disables effect of every Boss Blind" (activation: on blind select). Emits `DisableBoss` when
// a Boss Blind is selected, and when Chicot is acquired while an active boss is current (e.g.
// The Soul used during a boss round). Not copyable. Phase 4 applies the consequences (no boss
// set-up / revert it, Wall and Violet Vessel back to the x2 requirement, The Needle keeps x1).

fn chicot_blind_select(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if ctx.run().blind == BlindKind::Boss {
        ctx.push_effect(SideEffect::DisableBoss);
    }
}

fn chicot_acquire(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let run = ctx.run();
    if run.in_round && run.blind == BlindKind::Boss && run.boss.active {
        ctx.push_effect(SideEffect::DisableBoss);
    }
}
