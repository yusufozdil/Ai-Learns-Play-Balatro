//! Phase 3 joker package `wave4b`: TurtleBean, Hallucination, Juggler, Drunkard, DietCola, Seltzer, SockAndBuskin, Troubadour, Certificate, HangingChad, MerryAndy, Cartomancer, Perkeo.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave4b.rs`.

use crate::card::Card;
use crate::content::ids::TagId;
use crate::content::jokers::JokerId;
use crate::deck::Pile;
use crate::enums::{Rank, Rarity, Seal, Suit};
use crate::hooks::{
    ConsumableKind, EventCtx, Hooks, Modifiers, PackKind, RoundEndInfo, SideEffect,
};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        blueprint_compatible: false,
        eternal_compatible: false,
        init_value: TURTLE_BEAN_START,
        hooks: Hooks {
            modifiers: Some(turtle_bean_modifiers),
            on_round_end: Some(turtle_bean_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TurtleBean, "Turtle Bean", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_pack_open: Some(hallucination_pack_open),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Hallucination, "Hallucination", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(juggler_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Juggler, "Juggler", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(drunkard_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Drunkard, "Drunkard", Rarity::Common, 4)
    },
    JokerDef {
        eternal_compatible: false,
        hooks: Hooks {
            on_sell_self: Some(diet_cola_sell),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::DietCola, "Diet Cola", Rarity::Uncommon, 6)
    },
    JokerDef {
        eternal_compatible: false,
        init_counter: SELTZER_HANDS,
        hooks: Hooks {
            card_retriggers: Some(seltzer_retriggers),
            after_scoring: Some(seltzer_after),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Seltzer, "Seltzer", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            card_retriggers: Some(sock_and_buskin_retriggers),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::SockAndBuskin,
            "Sock and Buskin",
            Rarity::Uncommon,
            6,
        )
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(troubadour_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Troubadour, "Troubadour", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_first_hand_drawn: Some(certificate_first_hand),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Certificate, "Certificate", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            card_retriggers: Some(hanging_chad_retriggers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::HangingChad, "Hanging Chad", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(merry_andy_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MerryAndy, "Merry Andy", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_blind_select: Some(cartomancer_blind_select),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Cartomancer, "Cartomancer", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_shop_exit: Some(perkeo_shop_exit),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Perkeo, "Perkeo", Rarity::Legendary, 20)
    },
];

// --- Turtle Bean: https://balatrowiki.org/w/Turtle_Bean --------------------------------------
// value = current hand size bonus. Starts at +5, -1 at the end of every round (not
// Blueprint-copyable; the end-of-round hook is never copied anyway), destroyed when it reaches
// 0 ("Eaten!", lasts 5 rounds). Not Eternal-compatible.
const TURTLE_BEAN_START: f64 = 5.0;
const TURTLE_BEAN_DECAY: f64 = 1.0;

fn turtle_bean_modifiers(j: &JokerInstance, m: &mut Modifiers) {
    // `value` only ever holds small whole numbers; truncation is exact.
    m.hand_size += j.value.max(0.0) as i32;
}

fn turtle_bean_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    j.value -= TURTLE_BEAN_DECAY;
    if j.value <= 0.0 {
        j.value = 0.0;
        ctx.destroy_self();
    }
}

// --- Hallucination: https://balatrowiki.org/w/Hallucination ----------------------------------
// 1 in 2 chance to create a Tarot whenever ANY booster pack is opened (needs room; the game
// layer drops the creation when there is none). One roll per opened pack, per copy.
const HALLUCINATION_ODDS: (u32, u32) = (1, 2);

fn hallucination_pack_open(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: PackKind) {
    let (n, d) = HALLUCINATION_ODDS;
    if ctx.chance(n, d) {
        ctx.push_effect(SideEffect::CreateConsumable {
            kind: ConsumableKind::Tarot,
            negative: false,
        });
    }
}

// --- Juggler: https://balatrowiki.org/w/Juggler ----------------------------------------------
const JUGGLER_HAND_SIZE: i32 = 1;

fn juggler_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.hand_size += JUGGLER_HAND_SIZE;
}

// --- Drunkard: https://balatrowiki.org/w/Drunkard --------------------------------------------
const DRUNKARD_DISCARDS: i32 = 1;

fn drunkard_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.discards_per_round += DRUNKARD_DISCARDS;
}

// --- Diet Cola: https://balatrowiki.org/w/Diet_Cola ------------------------------------------
// Selling it creates a free Double Tag. A sold Blueprint/Brainstorm that copies it creates one
// too (`fire_sell_self` resolves copiers). Not Eternal-compatible.
// The Double Tag (typed id).
const DOUBLE_TAG_ID: TagId = TagId::Double;

fn diet_cola_sell(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    ctx.push_effect(SideEffect::CreateTag { id: DOUBLE_TAG_ID });
}

// --- Seltzer: https://balatrowiki.org/w/Seltzer ---------------------------------------------
// counter = hands left. Every scoring card triggers one extra time; after the hand it counts
// down, and the 10th hand is still retriggered before the joker is destroyed ("Drank!"). The
// countdown also runs after a hand the boss rejected. Not Eternal-compatible.
const SELTZER_HANDS: u32 = 10;
const SELTZER_RETRIGGERS: u32 = 1;

fn seltzer_retriggers(_: &JokerInstance, _: &ScoreCtx<'_>, _: &Card) -> u32 {
    SELTZER_RETRIGGERS
}

fn seltzer_after(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.copied() {
        return;
    }
    j.counter = j.counter.saturating_sub(1);
    if j.counter == 0 {
        ctx.destroy_self();
    }
}

// --- Sock and Buskin: https://balatrowiki.org/w/Sock_and_Buskin ------------------------------
// Played scoring face cards trigger one extra time (face = ctx.is_face: Pareidolia makes every
// card a face card, a debuffed card is never retriggered). Held faces are unaffected.
const SOCK_AND_BUSKIN_RETRIGGERS: u32 = 1;

fn sock_and_buskin_retriggers(_: &JokerInstance, ctx: &ScoreCtx<'_>, card: &Card) -> u32 {
    if ctx.is_face(card) {
        SOCK_AND_BUSKIN_RETRIGGERS
    } else {
        0
    }
}

// --- Troubadour: https://balatrowiki.org/w/Troubadour ----------------------------------------
const TROUBADOUR_HAND_SIZE: i32 = 2;
const TROUBADOUR_HANDS: i32 = -1;

fn troubadour_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.hand_size += TROUBADOUR_HAND_SIZE;
    m.hands_per_round += TROUBADOUR_HANDS;
}

// --- Certificate: https://balatrowiki.org/w/Certificate --------------------------------------
// When the round's first hand is drawn: a random playing card (random rank and suit, no
// enhancement) with a random seal (Gold / Red / Blue / Purple, equal odds) goes straight into
// the hand, beyond the hand size, and stays in the deck. Draws: rank, suit, seal, each one
// `rng_range` on RngKey::Jokers.
const CERTIFICATE_SEALS: [Seal; 4] = [Seal::Gold, Seal::Red, Seal::Blue, Seal::Purple];

fn certificate_first_hand(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let rank = Rank::ALL[ctx.rng_range(0, Rank::COUNT)];
    let suit = Suit::ALL[ctx.rng_range(0, Suit::COUNT)];
    let seal = CERTIFICATE_SEALS[ctx.rng_range(0, CERTIFICATE_SEALS.len())];
    let mut card = Card::new(0, rank, suit);
    card.seal = seal;
    ctx.push_effect(SideEffect::AddCard {
        card,
        to: Pile::Hand,
    });
}

// --- Hanging Chad: https://balatrowiki.org/w/Hanging_Chad ------------------------------------
// The first scoring card (position 0 among scoring cards) triggers 2 extra times. A Red Seal on
// it adds its own retrigger on top (4 triggers in total); the Red Seal itself is not retriggered.
const HANGING_CHAD_RETRIGGERS: u32 = 2;

fn hanging_chad_retriggers(_: &JokerInstance, ctx: &ScoreCtx<'_>, _: &Card) -> u32 {
    if ctx.scoring_pos() == 0 {
        HANGING_CHAD_RETRIGGERS
    } else {
        0
    }
}

// --- Merry Andy: https://balatrowiki.org/w/Merry_Andy ----------------------------------------
const MERRY_ANDY_DISCARDS: i32 = 3;
const MERRY_ANDY_HAND_SIZE: i32 = -1;

fn merry_andy_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.discards_per_round += MERRY_ANDY_DISCARDS;
    m.hand_size += MERRY_ANDY_HAND_SIZE;
}

// --- Cartomancer: https://balatrowiki.org/w/Cartomancer --------------------------------------
// Creates a random Tarot whenever a blind is selected (any blind; needs room, the game layer
// drops it otherwise). Skipping a blind does not trigger it (it is "On Blind Select").
fn cartomancer_blind_select(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    ctx.push_effect(SideEffect::CreateConsumable {
        kind: ConsumableKind::Tarot,
        negative: false,
    });
}

// --- Perkeo: https://balatrowiki.org/w/Perkeo ------------------------------------------------
// When leaving the shop: a Negative copy of 1 random held consumable (one `rng_range` draw over
// all held consumables, Negative ones included). Nothing happens with no consumables, and no
// draw is made then. Negative copies need no slot.
fn perkeo_shop_exit(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let n = ctx.run().consumables.len();
    if n == 0 {
        return;
    }
    let index = ctx.rng_range(0, n);
    ctx.push_effect(SideEffect::CopyConsumable {
        index,
        negative: true,
    });
}
