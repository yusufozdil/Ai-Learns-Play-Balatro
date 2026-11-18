//! Phase 3 joker package `wave2b`: scaling / counting jokers that keep state and react to
//! events: FlashCard, Popcorn, SpareTrousers, AncientJoker, Ramen, Castle, Campfire, Throwback,
//! GlassJoker, WeeJoker, TheIdol, HitTheRoad, Canio, Yorick.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave2b.rs`.
//!
//! Conventions used here (hooks.md §7): `value` = what the joker currently gives (+mult,
//! +chips, or the xmult factor, starting at 1.0), `counter` = integer countdowns, `aux_*` =
//! chosen targets. Copies (Blueprint/Brainstorm) run on a scratch copy of the target's state,
//! so no scaling hook needs a `copied()` check. End-of-round and acquire hooks are never
//! copied.

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{Enhancement, HandType, Rank, Rarity, Suit};
use crate::hooks::{DiscardInfo, EventCtx, Hooks, RoundEndInfo, SoldItem};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            joker_main: Some(flash_card_main),
            on_reroll: Some(flash_card_reroll),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::FlashCard, "Flash Card", Rarity::Uncommon, 5)
    },
    JokerDef {
        eternal_compatible: false,
        init_value: POPCORN_START,
        hooks: Hooks {
            joker_main: Some(popcorn_main),
            on_round_end: Some(popcorn_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Popcorn, "Popcorn", Rarity::Common, 5)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            before_scoring: Some(spare_trousers_before),
            joker_main: Some(spare_trousers_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::SpareTrousers,
            "Spare Trousers",
            Rarity::Uncommon,
            6,
        )
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(ancient_scored),
            on_acquire: Some(ancient_acquire),
            on_round_end: Some(ancient_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::AncientJoker, "Ancient Joker", Rarity::Rare, 8)
    },
    JokerDef {
        eternal_compatible: false,
        init_value: RAMEN_START,
        hooks: Hooks {
            joker_main: Some(ramen_main),
            on_discard: Some(ramen_discard),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Ramen, "Ramen", Rarity::Uncommon, 6)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            joker_main: Some(castle_main),
            on_acquire: Some(castle_acquire),
            on_discard: Some(castle_discard),
            on_round_end: Some(castle_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Castle, "Castle", Rarity::Uncommon, 6)
    },
    JokerDef {
        init_value: 1.0,
        hooks: Hooks {
            joker_main: Some(campfire_main),
            on_sell_other: Some(campfire_sell_other),
            on_round_end: Some(campfire_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Campfire, "Campfire", Rarity::Rare, 9)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(throwback_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Throwback, "Throwback", Rarity::Uncommon, 6)
    },
    JokerDef {
        perishable_compatible: false,
        init_value: 1.0,
        hooks: Hooks {
            joker_main: Some(glass_joker_main),
            on_cards_destroyed: Some(glass_joker_destroyed),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GlassJoker, "Glass Joker", Rarity::Uncommon, 6)
    },
    JokerDef {
        perishable_compatible: false,
        hooks: Hooks {
            on_card_scored: Some(wee_scored),
            joker_main: Some(wee_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::WeeJoker, "Wee Joker", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(idol_scored),
            on_acquire: Some(idol_acquire),
            on_round_end: Some(idol_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheIdol, "The Idol", Rarity::Uncommon, 6)
    },
    JokerDef {
        init_value: 1.0,
        hooks: Hooks {
            joker_main: Some(hit_the_road_main),
            on_discard: Some(hit_the_road_discard),
            on_round_end: Some(hit_the_road_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::HitTheRoad, "Hit the Road", Rarity::Rare, 8)
    },
    JokerDef {
        init_value: 1.0,
        hooks: Hooks {
            joker_main: Some(canio_main),
            on_cards_destroyed: Some(canio_destroyed),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Canio, "Canio", Rarity::Legendary, 20)
    },
    JokerDef {
        init_value: 1.0,
        init_counter: YORICK_DISCARDS,
        hooks: Hooks {
            joker_main: Some(yorick_main),
            on_discard: Some(yorick_discard),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Yorick, "Yorick", Rarity::Legendary, 20)
    },
];

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Pick a uniformly random non-Stone card of the full deck (all piles) and return its printed
/// rank and suit. `None` if the deck has no such card. One `RngKey::Jokers` draw when there is
/// at least one candidate (none otherwise), so the number of rolls is deterministic.
fn random_deck_card(ctx: &mut EventCtx<'_>) -> Option<(Rank, Suit)> {
    let n = ctx.deck().iter_all().filter(|c| !c.is_stone()).count();
    if n == 0 {
        return None;
    }
    let k = ctx.rng_range(0, n);
    ctx.deck()
        .iter_all()
        .filter(|c| !c.is_stone())
        .nth(k)
        .map(|c| (c.rank, c.suit))
}

// --- Flash Card: https://balatrowiki.org/w/Flash_Card ---------------------------------------
// value = current +mult (starts +0). +2 per shop reroll (free rerolls count: balatrowiki
// Chaos_the_Clown). Not retroactive (only rerolls after acquisition).
const FLASH_CARD_GAIN: f64 = 2.0;

fn flash_card_reroll(j: &mut JokerInstance, _: &mut EventCtx<'_>) {
    j.value += FLASH_CARD_GAIN;
}

fn flash_card_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Popcorn: https://balatrowiki.org/w/Popcorn --------------------------------------------
// value = current +mult. -4 at the end of every round; eaten (destroyed) when it reaches 0, so
// it lives exactly five round ends (20 -> 16 -> 12 -> 8 -> 4 -> 0).
const POPCORN_START: f64 = 20.0;
const POPCORN_DECAY: f64 = 4.0;

fn popcorn_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

fn popcorn_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    j.value -= POPCORN_DECAY;
    if j.value <= 0.0 {
        j.value = 0.0;
        ctx.destroy_self();
    }
}

// --- Spare Trousers: https://balatrowiki.org/w/Spare_Trousers -------------------------------
// value = current +mult (starts +0). +2 whenever the played hand CONTAINS Two Pair (Two Pair,
// Full House, Flush House; also a Flush that holds two pairs). Scales in `before_scoring`, so
// the hand that triggers the gain already benefits (game order: scale on "before", pay on
// "joker_main"). Debuffed cards still count for `contains`.
const SPARE_TROUSERS_GAIN: f64 = 2.0;

fn spare_trousers_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::TwoPair) {
        j.value += SPARE_TROUSERS_GAIN;
    }
}

fn spare_trousers_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_mult(j.value);
    }
}

// --- Ancient Joker: https://balatrowiki.org/w/Ancient_Joker --------------------------------
// aux_suit = target suit. Each scoring card of that suit gives x1.5 (per trigger, Wild always
// matches). First pick on acquire: uniform over the 4 suits. At every round end: a new suit,
// uniform over the 3 OTHER suits (the suit never repeats, update 1.0.1f). Acquire keeps an
// already chosen suit (a duplicated joker keeps its target).
const ANCIENT_XMULT: f64 = 1.5;

fn ancient_pick(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let cur = j.aux_suit();
    let n = Suit::ALL.iter().filter(|s| Some(**s) != cur).count();
    let k = ctx.rng_range(0, n);
    let pick = Suit::ALL.iter().copied().filter(|s| Some(*s) != cur).nth(k);
    j.set_aux_suit(pick);
}

fn ancient_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if j.aux_suit().is_none() {
        ancient_pick(j, ctx);
    }
}

fn ancient_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    ancient_pick(j, ctx);
}

fn ancient_scored(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if let Some(suit) = j.aux_suit()
        && ctx.has_suit(card, suit)
    {
        ctx.x_mult(ANCIENT_XMULT);
    }
}

// --- Ramen: https://balatrowiki.org/w/Ramen -------------------------------------------------
// value = current xmult (starts 2.0). -0.01 per discarded card (forced discards count, unscored
// played cards do not). When a discard would bring it to x1.0 or below it is eaten instead.
// Sell value never changes.
//
// Bit-exact with the real game, NO rounding: card.lua:3109-3128 (`context.discard`, per card)
// tests `x_mult - extra <= 1` and otherwise SMODS.scale_card subtracts `extra` in plain double
// arithmetic (`initial + -1 * scalar`, smods/src/utils.lua:2840 `additive_scaling`). Repeated
// `2.0 - 0.01` therefore drifts (99 cards leave 1.0099999999999991, not 1.01) and the
// destroy test uses that drifted value; the 100th card still destroys it.
const RAMEN_START: f64 = 2.0;
const RAMEN_LOSS: f64 = 0.01;
pub(crate) const RAMEN_FLOOR: f64 = 1.0;

fn ramen_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > RAMEN_FLOOR {
        ctx.x_mult(j.value);
    }
}

fn ramen_discard(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    for _ in d.cards {
        if j.value - RAMEN_LOSS <= RAMEN_FLOOR {
            ctx.destroy_self();
            return;
        }
        j.value -= RAMEN_LOSS;
    }
}

// --- Castle: https://balatrowiki.org/w/Castle -----------------------------------------------
// value = current +chips (starts +0), aux_suit = target suit. +3 per discarded (non-debuffed)
// card of the target suit (Wild matches). The target is the suit of a uniformly random non-Stone
// deck card (= weighted by the deck's suit composition), picked on acquire and at every round
// end; Spades when the deck has no suited card.
const CASTLE_GAIN: f64 = 3.0;
const CASTLE_FALLBACK_SUIT: Suit = Suit::Spades;

fn castle_pick(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let suit = random_deck_card(ctx).map_or(CASTLE_FALLBACK_SUIT, |(_, s)| s);
    j.set_aux_suit(Some(suit));
}

fn castle_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if j.aux_suit().is_none() {
        castle_pick(j, ctx);
    }
}

fn castle_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    castle_pick(j, ctx);
}

fn castle_discard(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    let Some(suit) = j.aux_suit() else { return };
    let n = d
        .cards
        .iter()
        .filter(|c| !c.debuffed && ctx.has_suit(c, suit))
        .count();
    j.value += CASTLE_GAIN * n as f64;
}

fn castle_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_chips(j.value);
    }
}

// --- Campfire: https://balatrowiki.org/w/Campfire -------------------------------------------
// value = current xmult (starts 1.0). +0.25 for every card sold (another joker or a
// consumable; never itself). Resets to 1.0 when a Boss Blind is defeated (a debuffed Campfire
// does not reset: debuffed jokers get no hooks).
const CAMPFIRE_GAIN: f64 = 0.25;
const CAMPFIRE_RESET: f64 = 1.0;

fn campfire_sell_other(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: &SoldItem) {
    j.value += CAMPFIRE_GAIN;
}

fn campfire_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, info: &RoundEndInfo) {
    if info.boss_defeated(ctx.run()) {
        j.value = CAMPFIRE_RESET;
    }
}

fn campfire_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Throwback: https://balatrowiki.org/w/Throwback -----------------------------------------
// No own state: x(1 + 0.25 * blinds skipped this run). Skips from before the joker was
// acquired count (wiki), which is exactly `run.stats.blinds_skipped`.
pub(crate) const THROWBACK_GAIN: f64 = 0.25;

fn throwback_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let skipped = ctx.run().stats.blinds_skipped;
    if skipped > 0 {
        ctx.x_mult(1.0 + THROWBACK_GAIN * f64::from(skipped));
    }
}

// --- Glass Joker: https://balatrowiki.org/w/Glass_Joker -------------------------------------
// value = current xmult (starts 1.0). +0.75 per destroyed Glass playing card (broken by its own
// roll, Hanged Man, Sixth Sense, ...). The destroyed-card snapshots carry the enhancement the
// card had before removal. (The wiki lists Immolate/Familiar/Grim/Incantation as not counting
// because of a game bug; Phase 4 may mirror that if it matters.)
const GLASS_JOKER_GAIN: f64 = 0.75;

fn glass_joker_destroyed(j: &mut JokerInstance, _: &mut EventCtx<'_>, cards: &[Card]) {
    let n = cards
        .iter()
        .filter(|c| c.enhancement == Enhancement::Glass)
        .count();
    j.value += GLASS_JOKER_GAIN * n as f64;
}

fn glass_joker_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Wee Joker: https://balatrowiki.org/w/Wee_Joker -----------------------------------------
// value = current +chips (starts +0). +8 per scoring 2 (every trigger, retriggers count). Scales
// in `on_card_scored`, so the hand that scores the 2 already benefits when the joker pays in
// `joker_main`. Stone cards have no rank; debuffed cards never reach the hook.
const WEE_GAIN: f64 = 8.0;

fn wee_scored(j: &mut JokerInstance, _: &mut ScoreCtx<'_>, card: &Card) {
    if card.rank == Rank::Two && !card.is_stone() {
        j.value += WEE_GAIN;
    }
}

fn wee_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 {
        ctx.add_chips(j.value);
    }
}

// --- The Idol: https://balatrowiki.org/w/The_Idol ------------------------------------------
// aux_rank + aux_suit = target card. Each scoring card with that rank and suit gives x2 (per
// trigger; Wild matches the suit). The target is the printed rank/suit of a uniformly random
// non-Stone card of the full deck, picked on acquire and at every round end; Ace of Spades if
// the deck has none. Acquire keeps an already chosen target.
const IDOL_XMULT: f64 = 2.0;
const IDOL_FALLBACK: (Rank, Suit) = (Rank::Ace, Suit::Spades);

fn idol_pick(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let (rank, suit) = random_deck_card(ctx).unwrap_or(IDOL_FALLBACK);
    j.set_aux_rank(Some(rank));
    j.set_aux_suit(Some(suit));
}

fn idol_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    if j.aux_rank().is_none() || j.aux_suit().is_none() {
        idol_pick(j, ctx);
    }
}

fn idol_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    idol_pick(j, ctx);
}

fn idol_scored(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if let (Some(rank), Some(suit)) = (j.aux_rank(), j.aux_suit())
        && !card.is_stone()
        && card.rank == rank
        && ctx.has_suit(card, suit)
    {
        ctx.x_mult(IDOL_XMULT);
    }
}

// --- Hit the Road: https://balatrowiki.org/w/Hit_the_Road -----------------------------------
// value = current xmult (starts 1.0). +0.5 per Jack discarded this round (each Jack of a
// multi-card discard counts; wiki is silent, the card text says "for every Jack"). Debuffed
// Jacks do not count (wiki: The Plant "renders Hit the Road useless"; balatrowiki.org/w/Debuffed:
// discard jokers do not activate from debuffed cards). Resets to 1.0 at the end of every round.
const HIT_THE_ROAD_GAIN: f64 = 0.5;
const HIT_THE_ROAD_RESET: f64 = 1.0;

fn hit_the_road_discard(j: &mut JokerInstance, _: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    let jacks = d
        .cards
        .iter()
        .filter(|c| c.rank == Rank::Jack && !c.is_stone() && !c.debuffed)
        .count();
    j.value += HIT_THE_ROAD_GAIN * jacks as f64;
}

fn hit_the_road_round_end(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: &RoundEndInfo) {
    j.value = HIT_THE_ROAD_RESET;
}

fn hit_the_road_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Canio: https://balatrowiki.org/w/Canio -------------------------------------------------
// value = current xmult (starts 1.0). +1.0 per destroyed face card (joker-rule face: J/Q/K, any
// card with Pareidolia, never a debuffed card).
const CANIO_GAIN: f64 = 1.0;

fn canio_destroyed(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, cards: &[Card]) {
    let n = cards.iter().filter(|c| ctx.is_face(c)).count();
    j.value += CANIO_GAIN * n as f64;
}

fn canio_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}

// --- Yorick: https://balatrowiki.org/w/Yorick -----------------------------------------------
// value = current xmult (starts 1.0), counter = cards left until the next step (starts 23).
// Every discarded card decrements it; at 0 it gains +1.0 and the counter resets to 23. A
// multi-card discard carries over naturally (card by card).
const YORICK_DISCARDS: u32 = 23;
const YORICK_GAIN: f64 = 1.0;

fn yorick_discard(j: &mut JokerInstance, _: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    for _ in d.cards {
        if j.counter == 0 {
            j.counter = YORICK_DISCARDS;
        }
        j.counter -= 1;
        if j.counter == 0 {
            j.counter = YORICK_DISCARDS;
            j.value += YORICK_GAIN;
        }
    }
}

fn yorick_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 1.0 {
        ctx.x_mult(j.value);
    }
}
