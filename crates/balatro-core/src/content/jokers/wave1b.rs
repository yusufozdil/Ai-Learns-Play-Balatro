//! Phase 3 joker package `wave1b`: Blackboard, BlueJoker, Photograph, Erosion, StoneJoker, WalkieTalkie, SmileyFace, Acrobat, Swashbuckler, Bloodstone, Arrowhead, OnyxAgate, FlowerPot, SeeingDouble, TheDuo, TheTrio, TheFamily, TheOrder, TheTribe, Stuntman, ShootTheMoon, DriversLicense, Bootstraps, Triboulet.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave1b.rs`.

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{Enhancement, HandType, Rank, Rarity, Suit};
use crate::hooks::{Hooks, Modifiers};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        hooks: Hooks {
            joker_main: Some(blackboard_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Blackboard, "Blackboard", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(blue_joker_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::BlueJoker, "Blue Joker", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(photograph_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Photograph, "Photograph", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(erosion_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Erosion, "Erosion", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(stone_joker_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::StoneJoker, "Stone Joker", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(walkie_talkie_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::WalkieTalkie, "Walkie Talkie", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(smiley_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SmileyFace, "Smiley Face", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(acrobat_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Acrobat, "Acrobat", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(swashbuckler_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Swashbuckler, "Swashbuckler", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(bloodstone_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Bloodstone, "Bloodstone", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(arrowhead_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Arrowhead, "Arrowhead", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(onyx_agate_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::OnyxAgate, "Onyx Agate", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(flower_pot_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::FlowerPot, "Flower Pot", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(seeing_double_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SeeingDouble, "Seeing Double", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(the_duo_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheDuo, "The Duo", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(the_trio_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheTrio, "The Trio", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(the_family_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheFamily, "The Family", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(the_order_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheOrder, "The Order", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(the_tribe_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TheTribe, "The Tribe", Rarity::Rare, 8)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(stuntman_main),
            modifiers: Some(stuntman_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Stuntman, "Stuntman", Rarity::Rare, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_held: Some(shoot_the_moon_held),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ShootTheMoon, "Shoot the Moon", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(drivers_license_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::DriversLicense, "Driver's License", Rarity::Rare, 7)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(bootstraps_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Bootstraps, "Bootstraps", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(triboulet_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Triboulet, "Triboulet", Rarity::Legendary, 20)
    },
];

// --- Blackboard: https://balatrowiki.org/w/Blackboard ---------------------------------------
// X3 if every held card is a Spade or Club (Wild counts; Stone has no suit and blocks; an empty
// hand counts as true). A debuffed Wild reverts to its printed suit (`Card::has_suit`).
const BLACKBOARD_XMULT: f64 = 3.0;

fn blackboard_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let ok = ctx
        .held()
        .iter()
        .all(|c| ctx.has_suit(c, Suit::Spades) || ctx.has_suit(c, Suit::Clubs));
    if ok {
        ctx.x_mult(BLACKBOARD_XMULT);
    }
}

// --- Blue Joker: https://balatrowiki.org/w/Blue_Joker ---------------------------------------
const BLUE_JOKER_CHIPS_PER_CARD: f64 = 2.0;

fn blue_joker_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let n = ctx.deck().draw_pile;
    if n > 0 {
        ctx.add_chips(BLUE_JOKER_CHIPS_PER_CARD * f64::from(n));
    }
}

// --- Photograph: https://balatrowiki.org/w/Photograph ---------------------------------------
// The first scoring face card (debuffed cards are never face; Pareidolia makes all cards face)
// gives X2; it repeats if that card is retriggered.
const PHOTOGRAPH_XMULT: f64 = 2.0;

fn photograph_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, _: &Card) {
    let first = ctx
        .scoring_indices()
        .find(|&i| ctx.is_face(&ctx.played()[i]));
    if first == Some(ctx.card_index()) {
        ctx.x_mult(PHOTOGRAPH_XMULT);
    }
}

// --- Erosion: https://balatrowiki.org/w/Erosion ---------------------------------------------
const EROSION_MULT_PER_CARD: f64 = 4.0;

fn erosion_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let d = ctx.deck();
    let missing = d.starting_size.saturating_sub(d.total);
    if missing > 0 {
        ctx.add_mult(EROSION_MULT_PER_CARD * f64::from(missing));
    }
}

// --- Stone Joker: https://balatrowiki.org/w/Stone_Joker -------------------------------------
const STONE_JOKER_CHIPS_PER_CARD: f64 = 25.0;

fn stone_joker_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let n = ctx.deck().enhancement(Enhancement::Stone);
    if n > 0 {
        ctx.add_chips(STONE_JOKER_CHIPS_PER_CARD * f64::from(n));
    }
}

// --- Walkie Talkie: https://balatrowiki.org/w/Walkie_Talkie --------------------------------
const WALKIE_CHIPS: f64 = 10.0;
const WALKIE_MULT: f64 = 4.0;

fn walkie_talkie_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if !card.is_stone() && matches!(card.rank, Rank::Ten | Rank::Four) {
        ctx.add_chips(WALKIE_CHIPS);
        ctx.add_mult(WALKIE_MULT);
    }
}

// --- Smiley Face: https://balatrowiki.org/w/Smiley_Face ------------------------------------
const SMILEY_MULT: f64 = 5.0;

fn smiley_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.is_face(card) {
        ctx.add_mult(SMILEY_MULT);
    }
}

// --- Acrobat: https://balatrowiki.org/w/Acrobat --------------------------------------------
const ACROBAT_XMULT: f64 = 3.0;

fn acrobat_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.run().is_final_hand() {
        ctx.x_mult(ACROBAT_XMULT);
    }
}

// --- Swashbuckler: https://balatrowiki.org/w/Swashbuckler ----------------------------------
// +mult = sum of the sell values of all OTHER jokers (debuffed ones included).
fn swashbuckler_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let me = ctx.slot();
    let total: i32 = ctx
        .jokers()
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != me)
        .map(|(_, j)| j.sell_value)
        .sum();
    if total > 0 {
        ctx.add_mult(f64::from(total));
    }
}

// --- Bloodstone: https://balatrowiki.org/w/Bloodstone --------------------------------------
const BLOODSTONE_XMULT: f64 = 1.5;
const BLOODSTONE_ODDS: (u32, u32) = (1, 2);

fn bloodstone_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Hearts) {
        let (n, d) = BLOODSTONE_ODDS;
        if ctx.chance(n, d) {
            ctx.x_mult(BLOODSTONE_XMULT);
        }
    }
}

// --- Arrowhead: https://balatrowiki.org/w/Arrowhead ----------------------------------------
const ARROWHEAD_CHIPS: f64 = 50.0;

fn arrowhead_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Spades) {
        ctx.add_chips(ARROWHEAD_CHIPS);
    }
}

// --- Onyx Agate: https://balatrowiki.org/w/Onyx_Agate --------------------------------------
const ONYX_AGATE_MULT: f64 = 7.0;

fn onyx_agate_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Clubs) {
        ctx.add_mult(ONYX_AGATE_MULT);
    }
}

// --- Flower Pot: https://balatrowiki.org/w/Flower_Pot --------------------------------------
// X3 if the scoring cards can be assigned to the four suits one-to-one (a Wild card fills only
// one suit; needs >= 4 scoring cards). Wiki (1.0.1f): "Debuffed cards still count for their
// suit"; "Debuffed Wild cards don't count at all, though this seems to be a bug" (mirrored:
// it changes the outcome). Smeared is respected through `has_suit` (backtracking matching).
const FLOWER_POT_XMULT: f64 = 3.0;
const ALL_SUITS: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades];

/// Suit test for Flower Pot: debuffed cards keep their (printed) suit, a debuffed Wild card has
/// none.
fn flower_pot_has_suit(ctx: &ScoreCtx<'_>, card: &Card, suit: Suit) -> bool {
    if card.debuffed && card.enhancement == Enhancement::Wild {
        return false;
    }
    ctx.has_suit(card, suit)
}

fn assign_suits(ctx: &ScoreCtx<'_>, cards: &[usize], suit_idx: usize, used: u8) -> bool {
    if suit_idx == ALL_SUITS.len() {
        return true;
    }
    for (k, &i) in cards.iter().enumerate() {
        if used & (1 << k) == 0
            && flower_pot_has_suit(ctx, &ctx.played()[i], ALL_SUITS[suit_idx])
            && assign_suits(ctx, cards, suit_idx + 1, used | (1 << k))
        {
            return true;
        }
    }
    false
}

/// Up to 8 scoring indices (the scoring mask is 8 bits wide).
const MAX_SCORING: usize = 8;

/// Scoring card indices in a fixed buffer (no allocation while scoring); `live_only` drops
/// debuffed cards.
fn scoring_buf(ctx: &ScoreCtx<'_>, live_only: bool) -> ([usize; MAX_SCORING], usize) {
    let mut buf = [0usize; MAX_SCORING];
    let mut n = 0;
    for i in ctx.scoring_indices() {
        if n < MAX_SCORING && !(live_only && ctx.played()[i].debuffed) {
            buf[n] = i;
            n += 1;
        }
    }
    (buf, n)
}

fn flower_pot_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let (buf, n) = scoring_buf(ctx, false);
    let cards = &buf[..n];
    if cards.len() >= ALL_SUITS.len() && assign_suits(ctx, cards, 0, 0) {
        ctx.x_mult(FLOWER_POT_XMULT);
    }
}

// --- Seeing Double: https://balatrowiki.org/w/Seeing_Double --------------------------------
// X2 if two DIFFERENT scoring cards exist: one Club and one non-Club (Wild counts as either,
// not both; Stone never qualifies; debuffed cards do not count).
// Smeared Joker bug (wiki: "playing any hand (even High Card) containing a Club or Spade will
// trigger Seeing Double. This interaction is a bug. This interaction doesn't apply if the card is
// Wild."): mirrored, since it decides outcomes. With Smeared a live non-Wild black card is a
// Club and a non-Club at once, so it triggers by itself.
const SEEING_DOUBLE_XMULT: f64 = 2.0;

fn seeing_double_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let (buf, n) = scoring_buf(ctx, true);
    let cards = &buf[..n];
    if ctx.run().rules.smeared
        && cards.iter().any(|&i| {
            let c = &ctx.played()[i];
            c.enhancement != Enhancement::Wild && ctx.has_suit(c, Suit::Clubs)
        })
    {
        ctx.x_mult(SEEING_DOUBLE_XMULT);
        return;
    }
    let is_club = |i: usize| ctx.has_suit(&ctx.played()[i], Suit::Clubs);
    let is_other = |i: usize| {
        let c = &ctx.played()[i];
        [Suit::Hearts, Suit::Diamonds, Suit::Spades]
            .iter()
            .any(|&s| ctx.has_suit(c, s))
    };
    let ok = cards
        .iter()
        .any(|&a| is_club(a) && cards.iter().any(|&b| b != a && is_other(b)));
    if ok {
        ctx.x_mult(SEEING_DOUBLE_XMULT);
    }
}

// --- The Duo / Trio / Family / Order / Tribe ------------------------------------------------
// https://balatrowiki.org/w/The_Duo  (also The_Trio, The_Family, The_Order, The_Tribe)
const THE_DUO_XMULT: f64 = 2.0;
const THE_TRIO_XMULT: f64 = 3.0;
const THE_FAMILY_XMULT: f64 = 4.0;
const THE_ORDER_XMULT: f64 = 3.0;
const THE_TRIBE_XMULT: f64 = 2.0;

fn the_duo_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Pair) {
        ctx.x_mult(THE_DUO_XMULT);
    }
}
fn the_trio_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::ThreeOfAKind) {
        ctx.x_mult(THE_TRIO_XMULT);
    }
}
fn the_family_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::FourOfAKind) {
        ctx.x_mult(THE_FAMILY_XMULT);
    }
}
fn the_order_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Straight) {
        ctx.x_mult(THE_ORDER_XMULT);
    }
}
fn the_tribe_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::Flush) {
        ctx.x_mult(THE_TRIBE_XMULT);
    }
}

// --- Stuntman: https://balatrowiki.org/w/Stuntman ------------------------------------------
// +250 chips in joker_main (copyable); -2 hand size is passive (modifiers, never copied).
const STUNTMAN_CHIPS: f64 = 250.0;
const STUNTMAN_HAND_SIZE: i32 = -2;

fn stuntman_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    ctx.add_chips(STUNTMAN_CHIPS);
}

fn stuntman_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.hand_size += STUNTMAN_HAND_SIZE;
}

// --- Shoot the Moon: https://balatrowiki.org/w/Shoot_the_Moon -------------------------------
const SHOOT_THE_MOON_MULT: f64 = 13.0;

fn shoot_the_moon_held(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if card.rank == Rank::Queen && !card.is_stone() {
        ctx.add_mult(SHOOT_THE_MOON_MULT);
    }
}

// --- Driver's License: https://balatrowiki.org/w/Driver%27s_License -------------------------
const DRIVERS_LICENSE_MIN_ENHANCED: u32 = 16;
const DRIVERS_LICENSE_XMULT: f64 = 3.0;

fn drivers_license_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.deck().enhanced() >= DRIVERS_LICENSE_MIN_ENHANCED {
        ctx.x_mult(DRIVERS_LICENSE_XMULT);
    }
}

// --- Bootstraps: https://balatrowiki.org/w/Bootstraps ---------------------------------------
// +2 Mult per full $5 held; debt (negative money) gives nothing.
const BOOTSTRAPS_MULT: f64 = 2.0;
const BOOTSTRAPS_STEP: i32 = 5;

fn bootstraps_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    let steps = ctx.money().max(0) / BOOTSTRAPS_STEP;
    if steps > 0 {
        ctx.add_mult(BOOTSTRAPS_MULT * f64::from(steps));
    }
}

// --- Triboulet: https://balatrowiki.org/w/Triboulet ----------------------------------------
const TRIBOULET_XMULT: f64 = 2.0;

fn triboulet_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if !card.is_stone() && matches!(card.rank, Rank::King | Rank::Queen) {
        ctx.x_mult(TRIBOULET_XMULT);
    }
}
