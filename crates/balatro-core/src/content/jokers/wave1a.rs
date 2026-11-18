//! Phase 3 joker package `wave1a`: LustyJoker, WrathfulJoker, GluttonousJoker, ZanyJoker, MadJoker, CrazyJoker, DrollJoker, SlyJoker, WilyJoker, CleverJoker, DeviousJoker, CraftyJoker, HalfJoker, JokerStencil, Banner, MysticSummit, Misprint, Fibonacci, SteelJoker, ScaryFace, AbstractJoker, EvenSteven, OddTodd, Scholar.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave1a.rs`.

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{Enhancement, HandType, Rank, Rarity, Suit};
use crate::hooks::Hooks;
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(lusty_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::LustyJoker, "Lusty Joker", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(wrathful_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::WrathfulJoker, "Wrathful Joker", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(gluttonous_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::GluttonousJoker,
            "Gluttonous Joker",
            Rarity::Common,
            5,
        )
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(zany_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ZanyJoker, "Zany Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(mad_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MadJoker, "Mad Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(crazy_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::CrazyJoker, "Crazy Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(droll_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::DrollJoker, "Droll Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(sly_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SlyJoker, "Sly Joker", Rarity::Common, 3)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(wily_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::WilyJoker, "Wily Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(clever_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::CleverJoker, "Clever Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(devious_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::DeviousJoker, "Devious Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(crafty_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::CraftyJoker, "Crafty Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(half_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::HalfJoker, "Half Joker", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(stencil_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::JokerStencil, "Joker Stencil", Rarity::Uncommon, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(banner_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Banner, "Banner", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(summit_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MysticSummit, "Mystic Summit", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(misprint_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Misprint, "Misprint", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(fibonacci_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Fibonacci, "Fibonacci", Rarity::Uncommon, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(steel_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SteelJoker, "Steel Joker", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(scary_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ScaryFace, "Scary Face", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(abstract_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::AbstractJoker, "Abstract Joker", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(even_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::EvenSteven, "Even Steven", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(odd_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::OddTodd, "Odd Todd", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(scholar_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Scholar, "Scholar", Rarity::Common, 4)
    },
];

// --- Lusty Joker: https://balatrowiki.org/w/Lusty_Joker ---
// +3 Mult per scored Hearts card (Wild / Smeared aware via `has_suit`).
const LUSTY_MULT: f64 = 3.0;

fn lusty_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Hearts) {
        ctx.add_mult(LUSTY_MULT);
    }
}

// --- Wrathful Joker: https://balatrowiki.org/w/Wrathful_Joker ---
// +3 Mult per scored Spades card (Wild / Smeared aware via `has_suit`).
const WRATHFUL_MULT: f64 = 3.0;

fn wrathful_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Spades) {
        ctx.add_mult(WRATHFUL_MULT);
    }
}

// --- Gluttonous Joker: https://balatrowiki.org/w/Gluttonous_Joker ---
// +3 Mult per scored Clubs card (Wild / Smeared aware via `has_suit`).
const GLUTTONOUS_MULT: f64 = 3.0;

fn gluttonous_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Clubs) {
        ctx.add_mult(GLUTTONOUS_MULT);
    }
}

// --- Zany Joker: https://balatrowiki.org/w/Zany_Joker ---
const ZANY_MULT: f64 = 12.0;

fn zany_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::ThreeOfAKind) {
        ctx.add_mult(ZANY_MULT);
    }
}

// --- Mad Joker: https://balatrowiki.org/w/Mad_Joker ---
const MAD_MULT: f64 = 10.0;

fn mad_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::TwoPair) {
        ctx.add_mult(MAD_MULT);
    }
}

// --- Crazy Joker: https://balatrowiki.org/w/Crazy_Joker ---
const CRAZY_MULT: f64 = 12.0;

fn crazy_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Straight) {
        ctx.add_mult(CRAZY_MULT);
    }
}

// --- Droll Joker: https://balatrowiki.org/w/Droll_Joker ---
const DROLL_MULT: f64 = 10.0;

fn droll_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Flush) {
        ctx.add_mult(DROLL_MULT);
    }
}

// --- Sly Joker: https://balatrowiki.org/w/Sly_Joker ---
const SLY_CHIPS: f64 = 50.0;

fn sly_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Pair) {
        ctx.add_chips(SLY_CHIPS);
    }
}

// --- Wily Joker: https://balatrowiki.org/w/Wily_Joker ---
const WILY_CHIPS: f64 = 100.0;

fn wily_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::ThreeOfAKind) {
        ctx.add_chips(WILY_CHIPS);
    }
}

// --- Clever Joker: https://balatrowiki.org/w/Clever_Joker ---
const CLEVER_CHIPS: f64 = 80.0;

fn clever_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::TwoPair) {
        ctx.add_chips(CLEVER_CHIPS);
    }
}

// --- Devious Joker: https://balatrowiki.org/w/Devious_Joker ---
const DEVIOUS_CHIPS: f64 = 100.0;

fn devious_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Straight) {
        ctx.add_chips(DEVIOUS_CHIPS);
    }
}

// --- Crafty Joker: https://balatrowiki.org/w/Crafty_Joker ---
const CRAFTY_CHIPS: f64 = 80.0;

fn crafty_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Flush) {
        ctx.add_chips(CRAFTY_CHIPS);
    }
}

// --- Half Joker: https://balatrowiki.org/w/Half_Joker ---
const HALF_MULT: f64 = 20.0;
const HALF_MAX_CARDS: usize = 3;

fn half_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.played().len() <= HALF_MAX_CARDS {
        ctx.add_mult(HALF_MULT);
    }
}

// --- Joker Stencil: https://balatrowiki.org/w/Joker_Stencil ---
// X1 per empty joker slot; every Joker Stencil owned counts as one extra empty slot (its own).
// `joker_slots` already includes Negative bonuses and Negative jokers are in `jokers()`, so
// they cancel out. A Blueprint copy reads the same board (Blueprint is not a Stencil).
pub(crate) const STENCIL_XMULT_PER_SLOT: f64 = 1.0;

/// Empty joker slots as Joker Stencil counts them (each Stencil counts as empty). Shared with
/// the observation's contribution block (`crate::contribution`).
pub(crate) fn stencil_empty_slots(joker_slots: u32, jokers: &[JokerInstance]) -> i64 {
    let slots = i64::from(joker_slots);
    let owned = jokers.len() as i64;
    let stencils = jokers
        .iter()
        .filter(|jk| jk.id == JokerId::JokerStencil)
        .count() as i64;
    (slots - owned + stencils).max(0)
}

fn stencil_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let empty = stencil_empty_slots(ctx.run().joker_slots, ctx.jokers());
    if empty > 0 {
        ctx.x_mult(STENCIL_XMULT_PER_SLOT * empty as f64);
    }
}

// --- Banner: https://balatrowiki.org/w/Banner ---
pub(crate) const BANNER_CHIPS_PER_DISCARD: f64 = 30.0;

fn banner_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let left = ctx.run().discards_left;
    if left > 0 {
        ctx.add_chips(BANNER_CHIPS_PER_DISCARD * f64::from(left));
    }
}

// --- Mystic Summit: https://balatrowiki.org/w/Mystic_Summit ---
pub(crate) const SUMMIT_MULT: f64 = 15.0;

fn summit_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.run().discards_left == 0 {
        ctx.add_mult(SUMMIT_MULT);
    }
}

// --- Misprint: https://balatrowiki.org/w/Misprint ---
// Uniform integer +0..=+23 Mult (wiki: "+0 to +23"); `rng_range` upper bound is exclusive.
pub(crate) const MISPRINT_MIN: usize = 0;
pub(crate) const MISPRINT_MAX_EXCLUSIVE: usize = 24;

fn misprint_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let m = ctx.rng_range(MISPRINT_MIN, MISPRINT_MAX_EXCLUSIVE);
    if m > 0 {
        ctx.add_mult(m as f64);
    }
}

// --- Fibonacci: https://balatrowiki.org/w/Fibonacci ---
const FIBONACCI_MULT: f64 = 8.0;

fn fibonacci_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    let hit = matches!(
        card.rank,
        Rank::Ace | Rank::Two | Rank::Three | Rank::Five | Rank::Eight
    );
    if hit && !card.is_stone() {
        ctx.add_mult(FIBONACCI_MULT);
    }
}

// --- Steel Joker: https://balatrowiki.org/w/Steel_Joker ---
// X(1 + 0.2 n), n = Steel cards in the full deck.
pub(crate) const STEEL_XMULT_PER_CARD: f64 = 0.2;

fn steel_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let n = ctx.deck().by_enhancement[Enhancement::Steel.index()];
    if n > 0 {
        ctx.x_mult(1.0 + STEEL_XMULT_PER_CARD * f64::from(n));
    }
}

// --- Scary Face: https://balatrowiki.org/w/Scary_Face ---
const SCARY_CHIPS: f64 = 30.0;

fn scary_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.is_face(card) {
        ctx.add_chips(SCARY_CHIPS);
    }
}

// --- Abstract Joker: https://balatrowiki.org/w/Abstract_Joker ---
pub(crate) const ABSTRACT_MULT_PER_JOKER: f64 = 3.0;

fn abstract_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let n = ctx.jokers().len();
    if n > 0 {
        ctx.add_mult(ABSTRACT_MULT_PER_JOKER * n as f64);
    }
}

// --- Even Steven: https://balatrowiki.org/w/Even_Steven ---
const EVEN_MULT: f64 = 4.0;

fn even_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    let hit = matches!(
        card.rank,
        Rank::Two | Rank::Four | Rank::Six | Rank::Eight | Rank::Ten
    );
    if hit && !card.is_stone() {
        ctx.add_mult(EVEN_MULT);
    }
}

// --- Odd Todd: https://balatrowiki.org/w/Odd_Todd ---
const ODD_CHIPS: f64 = 31.0;

fn odd_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    let hit = matches!(
        card.rank,
        Rank::Ace | Rank::Three | Rank::Five | Rank::Seven | Rank::Nine
    );
    if hit && !card.is_stone() {
        ctx.add_chips(ODD_CHIPS);
    }
}

// --- Scholar: https://balatrowiki.org/w/Scholar ---
const SCHOLAR_CHIPS: f64 = 20.0;
const SCHOLAR_MULT: f64 = 4.0;

fn scholar_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if card.rank == Rank::Ace && !card.is_stone() {
        ctx.add_chips(SCHOLAR_CHIPS);
        ctx.add_mult(SCHOLAR_MULT);
    }
}
