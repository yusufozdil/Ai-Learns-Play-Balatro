//! Prototype jokers that exercise every part of the Phase 1C hook contract. Each block cites
//! its wiki page; numbers are named constants next to the joker (convention for all waves).
//! Tests: `tests/joker_prototypes.rs`.

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{HandType, Rank, Rarity, Suit};
use crate::hooks::{DiscardInfo, EventCtx, Hooks, Modifiers, RoundEndInfo};
use crate::joker::{JokerDef, JokerInstance, copy_leftmost, copy_right};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        hooks: Hooks {
            joker_main: Some(joker_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Joker, "Joker", Rarity::Common, 2)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(greedy_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GreedyJoker, "Greedy Joker", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(jolly_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::JollyJoker, "Jolly Joker", Rarity::Common, 3)
    },
    JokerDef {
        eternal_compatible: false,
        hooks: Hooks {
            joker_main: Some(cavendish_main),
            on_round_end: Some(cavendish_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Cavendish, "Cavendish", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            card_retriggers: Some(hack_retriggers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Hack, "Hack", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            held_retriggers: Some(mime_held_retriggers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Mime, "Mime", Rarity::Uncommon, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_held: Some(baron_held),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Baron, "Baron", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            on_held: Some(raised_fist_held),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::RaisedFist, "Raised Fist", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            before_scoring: Some(ride_the_bus_before),
            joker_main: Some(ride_the_bus_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::RideTheBus, "Ride the Bus", Rarity::Common, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(hiker_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Hiker, "Hiker", Rarity::Uncommon, 5)
    },
    JokerDef {
        hooks: Hooks {
            before_scoring: Some(space_before),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SpaceJoker, "Space Joker", Rarity::Uncommon, 5)
    },
    JokerDef {
        eternal_compatible: false,
        init_value: ICE_CREAM_START,
        hooks: Hooks {
            joker_main: Some(ice_cream_main),
            after_scoring: Some(ice_cream_after),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::IceCream, "Ice Cream", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(supernova_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Supernova, "Supernova", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(bull_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Bull, "Bull", Rarity::Uncommon, 6)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(four_fingers_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::FourFingers, "Four Fingers", Rarity::Uncommon, 7)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(oops_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::OopsAll6s, "Oops! All 6s", Rarity::Uncommon, 4)
    },
    JokerDef {
        copies: Some(copy_right),
        ..JokerDef::new(JokerId::Blueprint, "Blueprint", Rarity::Rare, 10)
    },
    JokerDef {
        copies: Some(copy_leftmost),
        ..JokerDef::new(JokerId::Brainstorm, "Brainstorm", Rarity::Rare, 10)
    },
    JokerDef {
        hooks: Hooks {
            on_other_joker: Some(baseball_other),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::BaseballCard, "Baseball Card", Rarity::Rare, 8)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(golden_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GoldenJoker, "Golden Joker", Rarity::Common, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_discard: Some(burnt_discard),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::BurntJoker, "Burnt Joker", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            on_discard: Some(faceless_discard),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::FacelessJoker, "Faceless Joker", Rarity::Common, 4)
    },
];

// --- Joker: https://balatrowiki.org/w/Joker -------------------------------------------------
const JOKER_MULT: f64 = 4.0;

fn joker_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    ctx.add_mult(JOKER_MULT);
}

// --- Greedy Joker: https://balatrowiki.org/w/Greedy_Joker -----------------------------------
const GREEDY_MULT: f64 = 3.0;

fn greedy_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Diamonds) {
        ctx.add_mult(GREEDY_MULT);
    }
}

// --- Jolly Joker: https://balatrowiki.org/w/Jolly_Joker -------------------------------------
const JOLLY_MULT: f64 = 8.0;

fn jolly_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Pair) {
        ctx.add_mult(JOLLY_MULT);
    }
}

// --- Cavendish: https://balatrowiki.org/w/Cavendish -----------------------------------------
const CAVENDISH_XMULT: f64 = 3.0;
const CAVENDISH_DESTROY_ODDS: (u32, u32) = (1, 1000);

fn cavendish_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    ctx.x_mult(CAVENDISH_XMULT);
}

fn cavendish_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    let (n, d) = CAVENDISH_DESTROY_ODDS;
    if ctx.chance(n, d) {
        ctx.destroy_self();
    }
}

// --- Hack: https://balatrowiki.org/w/Hack ---------------------------------------------------
fn hack_retriggers(_: &JokerInstance, _: &ScoreCtx<'_>, card: &Card) -> u32 {
    let low = matches!(card.rank, Rank::Two | Rank::Three | Rank::Four | Rank::Five);
    u32::from(low && !card.is_stone())
}

// --- Mime: https://balatrowiki.org/w/Mime ---------------------------------------------------
fn mime_held_retriggers(_: &JokerInstance, _: &ScoreCtx<'_>, _: &Card) -> u32 {
    1
}

// --- Baron: https://balatrowiki.org/w/Baron -------------------------------------------------
const BARON_XMULT: f64 = 1.5;

fn baron_held(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if card.rank == Rank::King && !card.is_stone() {
        ctx.x_mult(BARON_XMULT);
    }
}

// --- Raised Fist: https://balatrowiki.org/w/Raised_Fist -------------------------------------
// Lowest-ranked held card (Stone excluded; ties: rightmost) adds 2x its chip value (face 10,
// Ace 11) as mult. A debuffed lowest card still counts as lowest but gives +0 (the pipeline
// skips debuffed held cards).
const RAISED_FIST_FACTOR: f64 = 2.0;

fn raised_fist_held(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    let held = ctx.held();
    let mut lowest: Option<usize> = None;
    for (i, c) in held.iter().enumerate() {
        if c.is_stone() {
            continue;
        }
        if lowest.is_none_or(|b| c.rank <= held[b].rank) {
            lowest = Some(i);
        }
    }
    if lowest == Some(ctx.card_index()) {
        ctx.add_mult(RAISED_FIST_FACTOR * f64::from(card.rank.chip_value()));
    }
}

// --- Ride the Bus: https://balatrowiki.org/w/Ride_the_Bus -----------------------------------
// value = current +mult. Before scoring: reset to 0 if a (non-debuffed) face card scores, else
// +1; the current hand benefits.
const RIDE_THE_BUS_GAIN: f64 = 1.0;

fn ride_the_bus_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let face_scored = ctx.scoring_indices().any(|i| ctx.is_face(&ctx.played()[i]));
    if face_scored {
        j.value = 0.0;
    } else {
        j.value += RIDE_THE_BUS_GAIN;
    }
}

fn ride_the_bus_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Hiker: https://balatrowiki.org/w/Hiker -------------------------------------------------
const HIKER_CHIPS: i32 = 5;

fn hiker_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, _: &Card) {
    let i = ctx.card_index();
    ctx.add_perma_bonus(i, HIKER_CHIPS);
}

// --- Space Joker: https://balatrowiki.org/w/Space_Joker -------------------------------------
const SPACE_ODDS: (u32, u32) = (1, 4);

fn space_before(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let (n, d) = SPACE_ODDS;
    if ctx.chance(n, d) {
        let ht = ctx.hand_type();
        ctx.level_up_hand(ht, 1);
    }
}

// --- Ice Cream: https://balatrowiki.org/w/Ice_Cream -----------------------------------------
// value = chips it currently gives. -5 after every hand played (also boss-invalid hands);
// melts (self-destructs) when it reaches 0.
const ICE_CREAM_START: f64 = 100.0;
const ICE_CREAM_DECAY: f64 = 5.0;

fn ice_cream_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_chips(j.value);
    }
}

fn ice_cream_after(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.copied() {
        return;
    }
    j.value -= ICE_CREAM_DECAY;
    if j.value <= 0.0 {
        j.value = 0.0;
        ctx.destroy_self();
    }
}

// --- Supernova: https://balatrowiki.org/w/Supernova -----------------------------------------
fn supernova_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let plays = ctx.run().hand_plays_run[ctx.hand_type().index()];
    if plays > 0 {
        ctx.add_mult(f64::from(plays));
    }
}

// --- Bull: https://balatrowiki.org/w/Bull ---------------------------------------------------
const BULL_CHIPS_PER_DOLLAR: f64 = 2.0;

fn bull_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let money = ctx.money();
    if money > 0 {
        ctx.add_chips(BULL_CHIPS_PER_DOLLAR * f64::from(money));
    }
}

// --- Four Fingers: https://balatrowiki.org/w/Four_Fingers -----------------------------------
fn four_fingers_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.rules.four_fingers = true;
}

// --- Oops! All 6s: https://balatrowiki.org/w/Oops!_All_6s -----------------------------------
const OOPS_PROB_FACTOR: u32 = 2;

fn oops_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.prob_mult = m.prob_mult.saturating_mul(OOPS_PROB_FACTOR);
}

// --- Blueprint: https://balatrowiki.org/w/Blueprint (copies: copy_right) ---------------------
// --- Brainstorm: https://balatrowiki.org/w/Brainstorm (copies: copy_leftmost) ----------------

// --- Baseball Card: https://balatrowiki.org/w/Baseball_Card ---------------------------------
const BASEBALL_XMULT: f64 = 1.5;

fn baseball_other(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, other: usize) {
    if ctx
        .joker_def(other)
        .is_some_and(|d| d.rarity == Rarity::Uncommon)
    {
        ctx.x_mult(BASEBALL_XMULT);
    }
}

// --- Golden Joker: https://balatrowiki.org/w/Golden_Joker -----------------------------------
const GOLDEN_MONEY: i32 = 4;

fn golden_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    ctx.add_money(GOLDEN_MONEY);
}

// --- Burnt Joker: https://balatrowiki.org/w/Burnt_Joker -------------------------------------
fn burnt_discard(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    if ctx.run().discards_used_round == 1 && !d.cards.is_empty() {
        ctx.level_up_hand(d.hand_type, 1);
    }
}

// --- Faceless Joker: https://balatrowiki.org/w/Faceless_Joker -------------------------------
const FACELESS_MIN_FACES: usize = 3;
const FACELESS_MONEY: i32 = 5;

fn faceless_discard(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    let faces = d.cards.iter().filter(|c| ctx.is_face(c)).count();
    if faces >= FACELESS_MIN_FACES {
        ctx.add_money(FACELESS_MONEY);
    }
}
