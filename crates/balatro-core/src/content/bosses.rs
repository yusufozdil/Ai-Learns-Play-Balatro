//! Boss blinds (23 regular + 5 Showdown). Phase 4C owns this file (tests: `tests/bosses.rs`).
//!
//! Phase 4.0 prototypes: The Hook, The Wall, The Psychic, The Plant, The Needle, The Flint,
//! The Mark, Verdant Leaf. Phase 4C added the other 20 (all 28 are defined).
//! Guide: `docs/design/consumables_bosses.md`; rules: docs/mechanics/bosses.md

use crate::blinds::{MULT_NEEDLE, MULT_VIOLET_VESSEL, MULT_WALL};
use crate::boss::{BossCtx, BossDef, BossHooks, DrawCtx, DrawKind, PlayCtx};
use crate::card::{Card, RuleFlags};
use crate::content::BossId as B;
use crate::enums::{HandType, Suit};

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
    BossDef {
        hooks: BossHooks {
            on_ante_start: Some(ox_ante_start),
            before_play: Some(ox_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheOx, "The Ox", 6)
    },
    BossDef {
        hooks: BossHooks {
            face_down: Some(house_face_down),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheHouse, "The House", 2)
    },
    BossDef {
        hooks: BossHooks {
            face_down: Some(wheel_face_down),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheWheel, "The Wheel", 2)
    },
    BossDef {
        hooks: BossHooks {
            before_play: Some(arm_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheArm, "The Arm", 2)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(club_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheClub, "The Club", 1)
    },
    BossDef {
        hooks: BossHooks {
            face_down: Some(fish_face_down),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheFish, "The Fish", 2)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(goad_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheGoad, "The Goad", 1)
    },
    BossDef {
        hooks: BossHooks {
            on_blind_start: Some(water_blind_start),
            on_disable: Some(water_disable),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheWater, "The Water", 2)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(window_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheWindow, "The Window", 1)
    },
    BossDef {
        hand_size: MANACLE_HAND_SIZE,
        ..BossDef::new(B::TheManacle, "The Manacle", 1)
    },
    BossDef {
        hooks: BossHooks {
            before_play: Some(eye_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheEye, "The Eye", 3)
    },
    BossDef {
        hooks: BossHooks {
            on_blind_start: Some(mouth_blind_start),
            before_play: Some(mouth_before_play),
            after_play: Some(mouth_after_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheMouth, "The Mouth", 2)
    },
    BossDef {
        hooks: BossHooks {
            draw_count: Some(serpent_draw_count),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheSerpent, "The Serpent", 5)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(pillar_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::ThePillar, "The Pillar", 1)
    },
    BossDef {
        hooks: BossHooks {
            debuffs_card: Some(head_debuffs),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheHead, "The Head", 1)
    },
    BossDef {
        hooks: BossHooks {
            before_play: Some(tooth_before_play),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::TheTooth, "The Tooth", 3)
    },
    BossDef {
        hooks: BossHooks {
            on_blind_start: Some(amber_blind_start),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::AmberAcorn, "Amber Acorn", 8)
    },
    BossDef {
        mult: MULT_VIOLET_VESSEL,
        ..BossDef::new(B::VioletVessel, "Violet Vessel", 8)
    },
    BossDef {
        hooks: BossHooks {
            on_blind_start: Some(crimson_pick),
            after_play: Some(crimson_pick),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::CrimsonHeart, "Crimson Heart", 8)
    },
    BossDef {
        hooks: BossHooks {
            after_draw: Some(cerulean_after_draw),
            ..BossHooks::NONE
        },
        ..BossDef::new(B::CeruleanBell, "Cerulean Bell", 8)
    },
];

// --- The Hook: https://balatrowiki.org/w/The_Hook ---------------------------------------------
// Before the hand scores, 2 random held cards are discarded (fewer if fewer are held). They fire
// a forced discard event (Green Joker/Ramen react; Burnt/Trading Card do not).
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

// --- The Ox: https://balatrowiki.org/w/The_Ox -------------------------------------------------
// The most played hand of the run is fixed when the ante starts (`boss().hand`); playing it sets
// money to $0. Ties go to the highest tier; with nothing played yet (only
// possible in tests, the Ox appears from ante 6) the fallback is High Card. The boss triggers
// (Matador) whenever the hand matches, even at $0.

fn ox_ante_start(ctx: &mut BossCtx<'_>) {
    let plays = ctx.state().hand_plays_run;
    let mut best = HandType::HighCard;
    let mut best_n = 0;
    for ht in HandType::ALL.into_iter().rev() {
        if plays[ht.index()] > best_n {
            best = ht;
            best_n = plays[ht.index()];
        }
    }
    ctx.boss_mut().hand = Some(best);
}

fn ox_before_play(ctx: &mut PlayCtx<'_>) {
    if ctx.boss().hand == Some(ctx.hand_type()) {
        ctx.set_money(0);
        ctx.trigger();
    }
}

// --- The House: https://balatrowiki.org/w/The_House -------------------------------------------
// The opening hand is drawn face down (refills are face up).

fn house_face_down(ctx: &mut DrawCtx<'_>, _: &Card) -> bool {
    ctx.kind() == DrawKind::RoundStart
}

// --- The Wheel: https://balatrowiki.org/w/The_Wheel ------------------------------------------
// Each drawn card is face down with probability 1/7, rolled at draw time (Oops! All 6s applies).
const WHEEL_NUM: u32 = 1;
const WHEEL_DEN: u32 = 7;

fn wheel_face_down(ctx: &mut DrawCtx<'_>, _: &Card) -> bool {
    ctx.chance(WHEEL_NUM, WHEEL_DEN)
}

// --- The Arm: https://balatrowiki.org/w/The_Arm -----------------------------------------------
// The played hand loses a level (never below 1) before it scores, so it scores at the lowered
// level. Triggers only if the level actually dropped.
const ARM_LEVELS: i32 = -1;

fn arm_before_play(ctx: &mut PlayCtx<'_>) {
    if ctx.add_levels(ctx.hand_type(), ARM_LEVELS) != 0 {
        ctx.trigger();
    }
}

// --- Suit bosses: https://balatrowiki.org/w/The_Club (Goad, Window, Head) --------------------
// `has_suit` makes Wild cards match every suit and respects Smeared Joker.

fn club_debuffs(card: &Card, rules: &RuleFlags) -> bool {
    card.has_suit(Suit::Clubs, rules)
}

fn goad_debuffs(card: &Card, rules: &RuleFlags) -> bool {
    card.has_suit(Suit::Spades, rules)
}

fn window_debuffs(card: &Card, rules: &RuleFlags) -> bool {
    card.has_suit(Suit::Diamonds, rules)
}

fn head_debuffs(card: &Card, rules: &RuleFlags) -> bool {
    card.has_suit(Suit::Hearts, rules)
}

// --- The Fish: https://balatrowiki.org/w/The_Fish --------------------------------------------
// Cards drawn after a played hand are face down (not the opening hand, not after discards).

fn fish_face_down(ctx: &mut DrawCtx<'_>, _: &Card) -> bool {
    ctx.kind() == DrawKind::AfterPlay
}

// --- The Water: https://balatrowiki.org/w/The_Water ------------------------------------------
// Start with 0 discards; `removed` remembers them for `on_disable`.

fn water_blind_start(ctx: &mut BossCtx<'_>) {
    let n = ctx.state().discards_left;
    ctx.boss_mut().removed = n;
    ctx.set_discards_left(0);
}

fn water_disable(ctx: &mut BossCtx<'_>) {
    let n = ctx.boss().removed;
    ctx.add_discards(n);
}

// --- The Manacle: https://balatrowiki.org/w/The_Manacle --------------------------------------
// Hand size -1 while in effect (static field `hand_size`).
const MANACLE_HAND_SIZE: i32 = -1;

// --- The Eye: https://balatrowiki.org/w/The_Eye -----------------------------------------------
// A hand type already played this round scores 0. `plays_round` includes the current hand, so
// anything above 1 is a repeat (rejected repeats count as played, as in the base game).

fn eye_before_play(ctx: &mut PlayCtx<'_>) {
    if ctx.plays_round(ctx.hand_type()) > 1 {
        ctx.invalidate();
    }
}

// --- The Mouth: https://balatrowiki.org/w/The_Mouth ------------------------------------------
// Only the first hand type of the round may be played. The first type is stored in
// `boss().hand` by `after_play` (the per-round counters also count rejected hands, so they
// cannot tell which type came first) and cleared at blind start.

fn mouth_blind_start(ctx: &mut BossCtx<'_>) {
    ctx.boss_mut().hand = None;
}

fn mouth_before_play(ctx: &mut PlayCtx<'_>) {
    if let Some(first) = ctx.boss().hand
        && first != ctx.hand_type()
    {
        ctx.invalidate();
    }
}

fn mouth_after_play(ctx: &mut BossCtx<'_>) {
    if ctx.boss().hand.is_none() {
        let last = ctx.state().last_hand;
        ctx.boss_mut().hand = last;
    }
}

// --- The Serpent: https://balatrowiki.org/w/The_Serpent --------------------------------------
// After a play or discard exactly 3 cards are drawn (the opening draw is normal).
const SERPENT_DRAW: usize = 3;

fn serpent_draw_count(kind: DrawKind, default: usize) -> usize {
    match kind {
        DrawKind::AfterPlay | DrawKind::AfterDiscard => SERPENT_DRAW,
        DrawKind::RoundStart | DrawKind::Refill => default,
    }
}

// --- The Pillar: https://balatrowiki.org/w/The_Pillar ----------------------------------------
// Cards played earlier this ante (`Card::played_ante`, cleared at ante start) are debuffed.

fn pillar_debuffs(card: &Card, _: &RuleFlags) -> bool {
    card.played_ante
}

// --- The Tooth: https://balatrowiki.org/w/The_Tooth ------------------------------------------
// $1 per played card, taken before scoring; money may go negative.
const TOOTH_PER_CARD: i32 = 1;

fn tooth_before_play(ctx: &mut PlayCtx<'_>) {
    let n = i32::try_from(ctx.played().len()).unwrap_or(i32::MAX);
    if n > 0 {
        ctx.add_money(-n.saturating_mul(TOOTH_PER_CARD));
        ctx.trigger();
    }
}

// --- Amber Acorn: https://balatrowiki.org/w/Amber_Acorn --------------------------------------
// Jokers are shuffled and hidden (`jokers_hidden`, read by the observation in Phase 8).

fn amber_blind_start(ctx: &mut BossCtx<'_>) {
    ctx.shuffle_jokers();
    ctx.boss_mut().jokers_hidden = true;
}

// --- Violet Vessel: https://balatrowiki.org/w/Violet_Vessel ----------------------------------
// Requirement x6 (static field `mult`).

// --- Crimson Heart: https://balatrowiki.org/w/Crimson_Heart -----------------------------------
// One random joker is debuffed for the next hand: picked at blind start and after every played
// hand; never the previous pick unless it is the only candidate; jokers already debuffed by
// something else are not eligible. Restored on round end / disable by the game layer.

fn crimson_pick(ctx: &mut BossCtx<'_>) {
    let prev = ctx.boss().joker_uid;
    let eligible: Vec<u32> = ctx
        .jokers()
        .iter()
        .filter(|j| !j.debuffed || Some(j.uid) == prev)
        .map(|j| j.uid)
        .collect();
    let fresh: Vec<u32> = eligible
        .iter()
        .copied()
        .filter(|&u| Some(u) != prev)
        .collect();
    let pool = if fresh.is_empty() { &eligible } else { &fresh };
    if pool.is_empty() {
        ctx.set_debuffed_joker(None);
        return;
    }
    let pick = pool[ctx.rng_range(0, pool.len())];
    ctx.set_debuffed_joker(Some(pick));
}

// --- Cerulean Bell: https://balatrowiki.org/w/Cerulean_Bell ----------------------------------
// After every draw the forced card is kept if it is still in hand, else a random hand card is
// chosen (none if the hand is empty).

fn cerulean_after_draw(ctx: &mut BossCtx<'_>) {
    if let Some(uid) = ctx.boss().card_uid
        && ctx.hand().iter().any(|c| c.uid == uid)
    {
        return;
    }
    let n = ctx.hand().len();
    let uid = if n == 0 {
        None
    } else {
        let i = ctx.rng_range(0, n);
        Some(ctx.hand()[i].uid)
    };
    ctx.boss_mut().card_uid = uid;
}
