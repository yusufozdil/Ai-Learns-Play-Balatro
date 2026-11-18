//! Phase 3 joker package `wave2a`: CeremonialDagger, LoyaltyCard, GrosMichel, Runner, Constellation, GreenJoker, CardSharp, RedCard, Madness, SquareJoker, Vampire, Hologram, Obelisk, FortuneTeller, LuckyCat.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave2a.rs`.
//!
//! All fifteen are stateful (scaling or counting). Conventions used here:
//! - a Blueprint/Brainstorm copy runs the hook on a scratch copy of the target's state, so
//!   scaling never double-counts; effects that are NOT joker state (destroying jokers, stripping
//!   card enhancements) check `ctx.copied()` and do nothing in a copy;
//! - xmult jokers store the factor in `value` (start 1.0) and skip the multiplication at 1.0;
//! - float accumulation (`+= 0.1`) is deliberately the same plain f64 addition the original
//!   game performs, so scores match bit for bit.
//!
//! Wiki "Eternal"/"Perishable" pages: only Gros Michel cannot be Eternal; Ceremonial Dagger,
//! Runner, Constellation, Green Joker, Red Card, Madness, Square Joker, Vampire, Hologram,
//! Obelisk and Lucky Cat cannot be Perishable.

use crate::blinds::BlindKind;
use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{Enhancement, HandType, Rarity};
use crate::hooks::{
    ConsumableKind, ConsumableView, DiscardInfo, EventCtx, Hooks, PackKind, RoundEndInfo, RunFlag,
    SideEffect,
};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            on_blind_select: Some(dagger_blind_select),
            joker_main: Some(dagger_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::CeremonialDagger,
            "Ceremonial Dagger",
            Rarity::Uncommon,
            6,
        )
    },
    JokerDef {
        hooks: Hooks {
            on_acquire: Some(loyalty_acquire),
            joker_main: Some(loyalty_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::LoyaltyCard, "Loyalty Card", Rarity::Uncommon, 5)
    },
    JokerDef {
        eternal_compatible: false,
        hooks: Hooks {
            joker_main: Some(gros_michel_main),
            on_round_end: Some(gros_michel_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GrosMichel, "Gros Michel", Rarity::Common, 5)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            before_scoring: Some(runner_before),
            joker_main: Some(runner_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Runner, "Runner", Rarity::Common, 5)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            on_consumable_used: Some(constellation_used),
            joker_main: Some(constellation_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Constellation, "Constellation", Rarity::Uncommon, 6)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            before_scoring: Some(green_before),
            on_discard: Some(green_discard),
            joker_main: Some(green_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GreenJoker, "Green Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(card_sharp_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::CardSharp, "Card Sharp", Rarity::Uncommon, 6)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            on_pack_skip: Some(red_card_skip),
            joker_main: Some(red_card_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::RedCard, "Red Card", Rarity::Common, 5)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            on_blind_select: Some(madness_blind_select),
            joker_main: Some(madness_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Madness, "Madness", Rarity::Uncommon, 7)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            before_scoring: Some(square_before),
            joker_main: Some(square_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SquareJoker, "Square Joker", Rarity::Common, 4)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            before_scoring: Some(vampire_before),
            joker_main: Some(vampire_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Vampire, "Vampire", Rarity::Uncommon, 7)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            on_cards_added: Some(hologram_added),
            joker_main: Some(hologram_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Hologram, "Hologram", Rarity::Uncommon, 7)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            before_scoring: Some(obelisk_before),
            joker_main: Some(obelisk_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Obelisk, "Obelisk", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(fortune_teller_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::FortuneTeller, "Fortune Teller", Rarity::Common, 6)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            on_card_scored: Some(lucky_cat_scored),
            joker_main: Some(lucky_cat_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::LuckyCat, "Lucky Cat", Rarity::Uncommon, 6)
    },
];

// --- Ceremonial Dagger: https://balatrowiki.org/w/Ceremonial_Dagger --------------------------
// value = current +mult (starts 0). When ANY blind is selected (boss included): destroy the joker
// immediately to the right and permanently add 2x its sell value. Nothing happens when this
// is the rightmost joker, or when the right neighbour is Eternal (no destruction, no mult), or
// is already being destroyed in the same event. A Blueprint copy neither destroys nor scales.
const DAGGER_SELL_FACTOR: f64 = 2.0;

fn dagger_blind_select(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if ctx.copied() {
        return;
    }
    let target = ctx.slot() + 1;
    let Some(victim) = ctx.jokers().get(target) else {
        return;
    };
    let gain = DAGGER_SELL_FACTOR * f64::from(victim.sell_value.max(0));
    if !ctx.is_doomed(target) && ctx.destroy_joker(target) {
        j.value += gain;
    }
}

fn dagger_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Loyalty Card: https://balatrowiki.org/w/Loyalty_Card -----------------------------------
// counter = `hands_played_run` when the card was acquired (offset); aux_byte = 1 once recorded
// (a duplicate made by Ankh/Invisible keeps the original's offset: it fires `on_acquire` again
// but is already marked). Every played hand counts, also hands that score nothing. The
// hand number `n = hands_played_run - offset` (current hand included) gives X4 when
// `n % 6 == 0`: hands 1-5 nothing, the 6th hand X4, then it repeats.
pub(crate) const LOYALTY_PERIOD: u32 = 6;
pub(crate) const LOYALTY_XMULT: f64 = 4.0;
const LOYALTY_RECORDED: u8 = 1;

fn loyalty_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if j.aux_byte() != LOYALTY_RECORDED {
        j.counter = ctx.run().hands_played_run;
        j.set_aux_byte(LOYALTY_RECORDED);
    }
}

/// Loyalty Card fires on the hand that brings `hands_played_run` (current hand included) to
/// `offset + 6k`, k >= 1. Shared with the observation's contribution block.
pub(crate) fn loyalty_fires(hands_played_run: u32, offset: u32) -> bool {
    let n = hands_played_run.saturating_sub(offset);
    n > 0 && n.is_multiple_of(LOYALTY_PERIOD)
}

fn loyalty_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if loyalty_fires(ctx.run().hands_played_run, j.counter) {
        ctx.x_mult(LOYALTY_XMULT);
    }
}

// --- Gros Michel: https://balatrowiki.org/w/Gros_Michel ------------------------------------
// No state. +15 mult; at the end of every round 1 in 6 (Oops! All 6s doubles it) it destroys
// itself and the run flag `GrosMichelExtinct` is set (Cavendish becomes available). End-of-round
// effects are never copied, so Blueprint cannot trigger the roll. Cannot be Eternal.
pub(crate) const GROS_MICHEL_MULT: f64 = 15.0;
const GROS_MICHEL_EXTINCT_ODDS: (u32, u32) = (1, 6);

fn gros_michel_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    ctx.add_mult(GROS_MICHEL_MULT);
}

fn gros_michel_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    let (n, d) = GROS_MICHEL_EXTINCT_ODDS;
    if ctx.chance(n, d) {
        ctx.destroy_self();
        ctx.push_effect(SideEffect::SetRunFlag(RunFlag::GrosMichelExtinct));
    }
}

// --- Runner: https://balatrowiki.org/w/Runner -----------------------------------------------
// value = current +chips (starts 0). +15 whenever the played hand CONTAINS a Straight (Straight
// Flush and Royal Flush included); the triggering hand already benefits.
const RUNNER_GAIN: f64 = 15.0;

fn runner_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Straight) {
        j.value += RUNNER_GAIN;
    }
}

fn runner_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_chips(j.value);
    }
}

// --- Constellation: https://balatrowiki.org/w/Constellation ---------------------------------
// value = current xmult (starts 1.0). +0.1 for every Planet card used AFTER acquisition (not
// retroactive). Spectral cards such as Black Hole are not Planets. `fire_consumable_used` is
// called by the game layer after the card's own effect.
const CONSTELLATION_GAIN: f64 = 0.1;

fn constellation_used(j: &mut JokerInstance, _: &mut EventCtx<'_>, used: &ConsumableView) {
    if used.kind() == ConsumableKind::Planet {
        j.value += CONSTELLATION_GAIN;
    }
}

fn constellation_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Green Joker: https://balatrowiki.org/w/Green_Joker ------------------------------------
// value = current +mult (starts 0, never below 0). +1 for every hand played (the played hand
// already benefits), -1 for every discard ACTION (not per card).
pub(crate) const GREEN_HAND_GAIN: f64 = 1.0;
const GREEN_DISCARD_LOSS: f64 = 1.0;

fn green_before(j: &mut JokerInstance, _: &mut ScoreCtx<'_>) {
    j.value += GREEN_HAND_GAIN;
}

fn green_discard(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: &DiscardInfo<'_>) {
    j.value = (j.value - GREEN_DISCARD_LOSS).max(0.0);
}

fn green_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Card Sharp: https://balatrowiki.org/w/Card_Sharp -------------------------------------
// No state: X3 when the played hand type was already played earlier this round.
// `hand_plays_round` includes the current hand, so "already played" is `> 1`. Only the exact
// hand type counts (a Straight Flush after a Flush does not).
pub(crate) const CARD_SHARP_XMULT: f64 = 3.0;

fn card_sharp_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let plays = ctx.run().hand_plays_round[ctx.hand_type().index()];
    if plays > 1 {
        ctx.x_mult(CARD_SHARP_XMULT);
    }
}

// --- Red Card: https://balatrowiki.org/w/Red_Card ------------------------------------------
// value = current +mult (starts 0). +3 each time a Booster Pack is skipped (once per pack, even
// Mega); not retroactive. `fire_pack_skip` is called once per skip by the pack system.
const RED_CARD_GAIN: f64 = 3.0;

fn red_card_skip(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: PackKind) {
    j.value += RED_CARD_GAIN;
}

fn red_card_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Madness: https://balatrowiki.org/w/Madness --------------------------------------------
// value = current xmult (starts 1.0). On Small/Big Blind select (NOT Boss): +0.5, and destroy a
// random other joker that is not Eternal and not already doomed (other Madnesses are fair game,
// itself never). The gain does not depend on a destruction. With no candidate nothing is
// rolled. A Blueprint copy does nothing (no gain, no destruction, no roll).
const MADNESS_GAIN: f64 = 0.5;

fn madness_blind_select(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if ctx.copied() || ctx.run().blind == BlindKind::Boss {
        return;
    }
    j.value += MADNESS_GAIN;
    let me = ctx.slot();
    let eligible = |ctx: &EventCtx<'_>, slot: usize| {
        slot != me && !ctx.jokers()[slot].stickers.eternal && !ctx.is_doomed(slot)
    };
    let n = (0..ctx.jokers().len())
        .filter(|&slot| eligible(ctx, slot))
        .count();
    if n > 0 {
        let k = ctx.rng_range(0, n);
        let pick = (0..ctx.jokers().len())
            .filter(|&slot| eligible(ctx, slot))
            .nth(k);
        if let Some(slot) = pick {
            ctx.destroy_joker(slot);
        }
    }
}

fn madness_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Square Joker: https://balatrowiki.org/w/Square_Joker ----------------------------------
// value = current +chips (starts 0). +4 when the hand has exactly 4 PLAYED cards (they need not
// score, they may be debuffed); the current hand benefits.
const SQUARE_CARDS: usize = 4;
const SQUARE_GAIN: f64 = 4.0;

fn square_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.played().len() == SQUARE_CARDS {
        j.value += SQUARE_GAIN;
    }
}

fn square_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_chips(j.value);
    }
}

// --- Vampire: https://balatrowiki.org/w/Vampire --------------------------------------------
// value = current xmult (starts 1.0). Before scoring, every scoring, non-debuffed ENHANCED card
// (Stone included) loses its enhancement permanently (before its effect applies) and Vampire
// gains 0.1 per such card (one addition of 0.1 x count). Editions and seals stay. Jokers run
// `before_scoring` left to right, so a Midas Mask to the LEFT feeds Vampire, one to the right
// re-golds what Vampire stripped. Only the owner strips: a Blueprint copy neither strips nor
// scales (a second Vampire to the right of the first finds nothing left).
const VAMPIRE_GAIN: f64 = 0.1;

fn vampire_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.copied() {
        return;
    }
    let mut eaten = 0u32;
    for i in ctx.scoring_indices() {
        let strip = ctx
            .played()
            .get(i)
            .is_some_and(|c| !c.debuffed && c.enhancement != Enhancement::None);
        if strip {
            ctx.set_enhancement(i, Enhancement::None);
            eaten += 1;
        }
    }
    if eaten > 0 {
        j.value += VAMPIRE_GAIN * f64::from(eaten);
    }
}

fn vampire_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Hologram: https://balatrowiki.org/w/Hologram ------------------------------------------
// value = current xmult (starts 1.0). +0.25 for each playing card added to the deck after
// acquisition (per card, not per event; keeping the card is not required). Fired by the game
// layer with the added cards for DNA, Marble Joker, Certificate, packs, Spectrals.
const HOLOGRAM_GAIN: f64 = 0.25;

fn hologram_added(j: &mut JokerInstance, _: &mut EventCtx<'_>, cards: &[Card]) {
    j.value += HOLOGRAM_GAIN * cards.len() as f64;
}

fn hologram_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Obelisk: https://balatrowiki.org/w/Obelisk --------------------------------------------
// value = current xmult (starts 1.0). Before scoring (so the played hand benefits, and a
// boss-rejected hand neither gains nor resets): if the played hand type is STRICTLY the most
// played this run (play counts already include this hand, so a tie is NOT the most played),
// reset to 1.0; otherwise +0.2.
const OBELISK_GAIN: f64 = 0.2;
const OBELISK_RESET: f64 = 1.0;

fn obelisk_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let played = ctx.hand_type().index();
    let plays = ctx.run().hand_plays_run;
    let strictly_most = plays
        .iter()
        .enumerate()
        .all(|(i, &p)| i == played || p < plays[played]);
    if strictly_most {
        j.value = OBELISK_RESET;
    } else {
        j.value += OBELISK_GAIN;
    }
}

fn obelisk_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Fortune Teller: https://balatrowiki.org/w/Fortune_Teller ------------------------------
// No own state: +1 mult per Tarot card used this run, read from `run().stats.tarots_used`
// (retroactive: Tarots used before buying it count, per the wiki). The consumable system
// (Phase 4) must update `RunStats::tarots_used` when a Tarot is used.
pub(crate) const FORTUNE_TELLER_MULT_PER_TAROT: f64 = 1.0;

fn fortune_teller_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let used = ctx.run().stats.tarots_used;
    if used > 0 {
        ctx.add_mult(FORTUNE_TELLER_MULT_PER_TAROT * f64::from(used));
    }
}

// --- Lucky Cat: https://balatrowiki.org/w/Lucky_Cat ----------------------------------------
// value = current xmult (starts 1.0). +0.25 each time a Lucky card successfully triggers (its
// mult OR money roll hits; both on one trigger count once). Counted per trigger, so a
// retriggered Lucky card can count twice. Scales while the cards score, so the current hand
// benefits.
const LUCKY_CAT_GAIN: f64 = 0.25;

fn lucky_cat_scored(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, _: &Card) {
    if ctx.lucky_triggered() {
        j.value += LUCKY_CAT_GAIN;
    }
}

fn lucky_cat_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}
