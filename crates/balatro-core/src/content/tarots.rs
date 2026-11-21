//! Tarot cards (22): The Fool, The Magician, The Hermit and Death (Phase 4.0 prototypes) plus
//! the other 18 (Phase 4A). Tests: `tests/consumables_planets_tarots.rs`.
//! Guide: `docs/design/consumables_bosses.md`.

use crate::consumable::{ConsumableDef, ConsumableKind, UseCheck, UseCtx};
use crate::content::ConsumableId as C;
use crate::data::consumables::EDITION_ROLL;
use crate::enums::{Edition, Enhancement, Rank, Suit};

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
    ConsumableDef {
        can_use: Some(free_slot_can_use),
        ..ConsumableDef::new(C::TheHighPriestess, "The High Priestess", high_priestess)
    },
    targeted(C::TheEmpress, "The Empress", 2, empress),
    ConsumableDef {
        can_use: Some(free_slot_can_use),
        ..ConsumableDef::new(C::TheEmperor, "The Emperor", emperor)
    },
    targeted(C::TheHierophant, "The Hierophant", 2, hierophant),
    targeted(C::TheLovers, "The Lovers", 1, lovers),
    targeted(C::TheChariot, "The Chariot", 1, chariot),
    targeted(C::Justice, "Justice", 1, justice),
    ConsumableDef::new(C::TheHermit, "The Hermit", hermit),
    ConsumableDef {
        can_use: Some(wheel_can_use),
        ..ConsumableDef::new(C::TheWheelOfFortune, "The Wheel of Fortune", wheel)
    },
    targeted(C::Strength, "Strength", 2, strength),
    targeted(C::TheHangedMan, "The Hanged Man", 2, hanged_man),
    ConsumableDef {
        min_targets: 2,
        max_targets: 2,
        ..ConsumableDef::new(C::Death, "Death", death)
    },
    ConsumableDef::new(C::Temperance, "Temperance", temperance),
    targeted(C::TheDevil, "The Devil", 1, devil),
    targeted(C::TheTower, "The Tower", 1, tower),
    targeted(C::TheStar, "The Star", 3, star),
    targeted(C::TheMoon, "The Moon", 3, moon),
    targeted(C::TheSun, "The Sun", 3, sun),
    ConsumableDef {
        can_use: Some(judgement_can_use),
        ..ConsumableDef::new(C::Judgement, "Judgement", judgement)
    },
    targeted(C::TheWorld, "The World", 3, world),
];

/// A card taking `1..=max` selected hand cards.
const fn targeted(id: C, name: &'static str, max: u8, f: fn(&mut UseCtx<'_>)) -> ConsumableDef {
    ConsumableDef {
        min_targets: 1,
        max_targets: max,
        ..ConsumableDef::new(id, name, f)
    }
}

/// Set the enhancement of every selected card.
fn enhance_targets(ctx: &mut UseCtx<'_>, e: Enhancement) {
    for k in 0..ctx.target_count() {
        if let Some(pos) = ctx.target_pos(k) {
            ctx.set_enhancement(pos, e);
        }
    }
}

/// Set the suit of every selected card.
fn suit_targets(ctx: &mut UseCtx<'_>, s: Suit) {
    for k in 0..ctx.target_count() {
        if let Some(pos) = ctx.target_pos(k) {
            ctx.set_suit(pos, s);
        }
    }
}

/// Cards that create consumables need a free slot (after the used card left its own).
fn free_slot_can_use(c: &UseCheck<'_>) -> bool {
    c.free_consumable_slots() > 0
}

// --- The Fool: https://balatrowiki.org/w/The_Fool --------------------------------------------
// Copies the last Tarot/Planet used this run (Spectrals never count; the game layer keeps
// `GameState::last_tarot_planet`). Using The Fool itself records `c_fool` as the last card, as in
// the base game (`set_consumeable_usage` sets `last_tarot_planet` for every Tarot), and
// `Card:can_use_consumeable` refuses a Fool while that is the record: a second Fool is dead until
// another Tarot/Planet is used. Needs a free slot after it leaves its own.

fn fool_can_use(c: &UseCheck<'_>) -> bool {
    matches!(c.state().last_tarot_planet, Some(id) if id != C::TheFool)
        && c.free_consumable_slots() > 0
}

fn fool(ctx: &mut UseCtx<'_>) {
    if let Some(id) = ctx.state().last_tarot_planet {
        ctx.add_consumable(id);
    }
}

// --- The Magician: https://balatrowiki.org/w/The_Magician ------------------------------------

fn magician(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Lucky);
}

// --- The High Priestess: https://balatrowiki.org/w/The_High_Priestess ------------------------
// Up to 2 random Planets, limited by the free slots.
const PRIESTESS_CREATES: u32 = 2;

fn high_priestess(ctx: &mut UseCtx<'_>) {
    for _ in 0..PRIESTESS_CREATES.min(ctx.free_consumable_slots()) {
        ctx.create_consumable(ConsumableKind::Planet);
    }
}

// --- The Empress / Hierophant / Lovers / Chariot / Justice / Devil / Tower -------------------
// Enhancements: https://balatrowiki.org/w/The_Empress (and the pages of the other six).

fn empress(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Mult);
}

fn hierophant(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Bonus);
}

fn lovers(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Wild);
}

fn chariot(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Steel);
}

fn justice(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Glass);
}

fn devil(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Gold);
}

fn tower(ctx: &mut UseCtx<'_>) {
    enhance_targets(ctx, Enhancement::Stone);
}

// --- The Emperor: https://balatrowiki.org/w/The_Emperor -------------------------------------
// Up to 2 random Tarots, limited by the free slots.
const EMPEROR_CREATES: u32 = 2;

fn emperor(ctx: &mut UseCtx<'_>) {
    for _ in 0..EMPEROR_CREATES.min(ctx.free_consumable_slots()) {
        ctx.create_consumable(ConsumableKind::Tarot);
    }
}

// --- The Hermit: https://balatrowiki.org/w/The_Hermit ----------------------------------------
// Doubles money, at most +$20; $0 when money <= 0.
const HERMIT_CAP: i32 = 20;

fn hermit(ctx: &mut UseCtx<'_>) {
    let gain = ctx.money().clamp(0, HERMIT_CAP);
    ctx.add_money(gain);
}

// --- The Wheel of Fortune: https://balatrowiki.org/w/The_Wheel_of_Fortune --------------------
// 1 in 4 (Oops! All 6s applies): a random joker without an edition gets Foil / Holographic /
// Polychrome (50/35/15). The card is used up on "Nope!". Never Negative.
const WHEEL_NUM: u32 = 1;
const WHEEL_DEN: u32 = 4;

fn wheel_can_use(c: &UseCheck<'_>) -> bool {
    c.jokers().iter().any(|j| j.edition == Edition::Base)
}

fn wheel(ctx: &mut UseCtx<'_>) {
    if !ctx.chance(WHEEL_NUM, WHEEL_DEN) {
        return;
    }
    let eligible: Vec<usize> = ctx
        .jokers()
        .iter()
        .enumerate()
        .filter(|(_, j)| j.edition == Edition::Base)
        .map(|(i, _)| i)
        .collect();
    if eligible.is_empty() {
        return;
    }
    let slot = eligible[ctx.rng_range(0, eligible.len())];
    let weights: Vec<f64> = EDITION_ROLL.iter().map(|&(_, w)| w).collect();
    if let Some(k) = ctx.weighted_index(&weights) {
        ctx.set_joker_edition(slot, EDITION_ROLL[k].0);
    }
}

// --- Strength: https://balatrowiki.org/w/Strength --------------------------------------------
// Rank +1; King -> Ace, Ace -> 2. Stone cards have no rank and are skipped.

fn next_rank(r: Rank) -> Rank {
    match r {
        Rank::Ace => Rank::Two,
        r => Rank::from_value(r as u8 + 1).unwrap_or(Rank::Two),
    }
}

fn strength(ctx: &mut UseCtx<'_>) {
    for k in 0..ctx.target_count() {
        let Some(pos) = ctx.target_pos(k) else {
            continue;
        };
        let Some(c) = ctx.hand().get(pos) else {
            continue;
        };
        if c.enhancement == Enhancement::Stone {
            continue;
        }
        let next = next_rank(c.rank);
        ctx.set_rank(pos, next);
    }
}

// --- The Hanged Man: https://balatrowiki.org/w/The_Hanged_Man -------------------------------
// Destroys the selected cards (follow-up events go through the SideEffect path).

fn hanged_man(ctx: &mut UseCtx<'_>) {
    for k in 0..ctx.target_count() {
        if let Some(pos) = ctx.target_pos(k) {
            ctx.destroy_card(pos);
        }
    }
}

// --- Death: https://balatrowiki.org/w/Death ---------------------------------------------------
// The LEFT selected card becomes a copy of the RIGHT one (hand order).

fn death(ctx: &mut UseCtx<'_>) {
    if let (Some(left), Some(right)) = (ctx.target_pos(0), ctx.target_pos(1)) {
        ctx.copy_card(left, right);
    }
}

// --- Temperance: https://balatrowiki.org/w/Temperance ----------------------------------------
// Total sell value of all jokers, at most $50.
const TEMPERANCE_CAP: i32 = 50;

fn temperance(ctx: &mut UseCtx<'_>) {
    let total = ctx
        .jokers()
        .iter()
        .fold(0i32, |a, j| a.saturating_add(j.sell_value.max(0)));
    ctx.add_money(total.min(TEMPERANCE_CAP));
}

// --- The Star / Moon / Sun / World: https://balatrowiki.org/w/The_Star (and the others) -------

fn star(ctx: &mut UseCtx<'_>) {
    suit_targets(ctx, Suit::Diamonds);
}

fn moon(ctx: &mut UseCtx<'_>) {
    suit_targets(ctx, Suit::Clubs);
}

fn sun(ctx: &mut UseCtx<'_>) {
    suit_targets(ctx, Suit::Hearts);
}

fn world(ctx: &mut UseCtx<'_>) {
    suit_targets(ctx, Suit::Spades);
}

// --- Judgement: https://balatrowiki.org/w/Judgement -------------------------------------------
// Creates a random joker (shop odds); needs a free joker slot (the game layer drops it
// otherwise).

fn judgement_can_use(c: &UseCheck<'_>) -> bool {
    c.free_joker_slots() > 0
}

fn judgement(ctx: &mut UseCtx<'_>) {
    ctx.create_joker(None);
}
