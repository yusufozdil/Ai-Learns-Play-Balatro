//! Phase 3 joker package `wave4a`: MarbleJoker, EightBall, Dusk, Pareidolia, Burglar, DNA, Splash, SixthSense, Superposition, Seance, RiffRaff, Shortcut, Vagabond, MidasMask.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave4a.rs`.
//!
//! Rule changers (Pareidolia, Splash, Shortcut) only set `Modifiers::rules` flags; hand
//! evaluation, `ctx.is_face` and the scoring set honour them. Creators (8 Ball, Superposition,
//! Seance, Vagabond, Sixth Sense, Riff-Raff, Marble, DNA) queue `SideEffect`s; the game layer
//! checks slot room when applying them (consumable and joker slots alike).

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::deck::Pile;
use crate::enums::{Enhancement, HandType, Rank, Rarity, Suit};
use crate::hooks::{ConsumableKind, EventCtx, Hooks, Modifiers, SideEffect};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        hooks: Hooks {
            on_blind_select: Some(marble_blind_select),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MarbleJoker, "Marble Joker", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(eight_ball_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::EightBall, "8 Ball", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            card_retriggers: Some(dusk_retriggers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Dusk, "Dusk", Rarity::Uncommon, 5)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(pareidolia_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Pareidolia, "Pareidolia", Rarity::Uncommon, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_blind_select: Some(burglar_blind_select),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Burglar, "Burglar", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            before_scoring: Some(dna_before),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::DNA, "DNA", Rarity::Rare, 8)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(splash_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Splash, "Splash", Rarity::Common, 3)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            after_scoring: Some(sixth_sense_after),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::SixthSense, "Sixth Sense", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(superposition_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Superposition, "Superposition", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(seance_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Seance, "Séance", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_blind_select: Some(riff_raff_blind_select),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::RiffRaff, "Riff-Raff", Rarity::Common, 6)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(shortcut_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Shortcut, "Shortcut", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(vagabond_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Vagabond, "Vagabond", Rarity::Rare, 8)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            before_scoring: Some(midas_mask_before),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MidasMask, "Midas Mask", Rarity::Uncommon, 7)
    },
];

// --- Marble Joker: https://balatrowiki.org/w/Marble_Joker -----------------------------------
// When a blind is selected (any kind), one Stone card is added to the deck. The card's printed
// rank/suit is random (it only shows if the Stone enhancement is later removed); it is drawn
// on `RngKey::Jokers`. A Blueprint copy adds another one. The game layer allocates the uid and
// puts the card in the draw pile.
const MARBLE_CARDS: usize = 1;

fn marble_blind_select(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    for _ in 0..MARBLE_CARDS {
        let rank = Rank::ALL[ctx.rng_range(0, Rank::COUNT)];
        let suit = Suit::ALL[ctx.rng_range(0, Suit::COUNT)];
        let mut card = Card::new(0, rank, suit);
        card.enhancement = Enhancement::Stone;
        ctx.push_effect(SideEffect::AddCard {
            card,
            to: Pile::Draw,
        });
    }
}

// --- 8 Ball: https://balatrowiki.org/w/8_Ball -----------------------------------------------
// Each played scoring 8 (not Stone, not debuffed: the pipeline skips debuffed cards) has a
// 1 in 4 chance to create a Tarot card; rolled on every trigger (retriggers roll again).
// Consumable-slot room is checked by the game layer when the effect is applied.
const EIGHT_BALL_RANK: Rank = Rank::Eight;
const EIGHT_BALL_ODDS: (u32, u32) = (1, 4);

fn eight_ball_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if card.is_stone() || card.rank != EIGHT_BALL_RANK {
        return;
    }
    let (n, d) = EIGHT_BALL_ODDS;
    if ctx.chance(n, d) {
        ctx.push_effect(SideEffect::CreateConsumable {
            kind: ConsumableKind::Tarot,
            negative: false,
        });
    }
}

// --- Dusk: https://balatrowiki.org/w/Dusk ---------------------------------------------------
// Retriggers every played scoring card once on the final hand of the round (hands_left == 0
// while scoring). Blueprint copies add one more retrigger each.
const DUSK_RETRIGGERS: u32 = 1;

fn dusk_retriggers(_: &JokerInstance, ctx: &ScoreCtx<'_>, _: &Card) -> u32 {
    if ctx.run().is_final_hand() {
        DUSK_RETRIGGERS
    } else {
        0
    }
}

// --- Pareidolia: https://balatrowiki.org/w/Pareidolia ---------------------------------------
// Passive: every card counts as a face card (Stone included). Not copied by Blueprint.
fn pareidolia_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.rules.pareidolia = true;
}

// --- Burglar: https://balatrowiki.org/w/Burglar ---------------------------------------------
// When a blind is selected: +3 hands this round and all discards are lost (set to 0).
pub(crate) const BURGLAR_HANDS: i32 = 3;

fn burglar_blind_select(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    ctx.push_effect(SideEffect::AddHands(BURGLAR_HANDS));
    ctx.push_effect(SideEffect::ClearDiscards);
}

// --- DNA: https://balatrowiki.org/w/DNA -----------------------------------------------------
// If the first hand of the round holds exactly one card, a permanent copy of it is added to the
// deck and drawn into the hand. The copy is taken before scoring (it does not keep changes made
// while scoring, e.g. Hiker chips) and is a fresh card: debuff and face-down are not copied.
// Blueprint copies add one more copy each.
const DNA_FIRST_HAND: u32 = 1;
const DNA_CARDS: usize = 1;

fn dna_before(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.run().hands_played_round != DNA_FIRST_HAND || ctx.played().len() != DNA_CARDS {
        return;
    }
    let mut copy = ctx.played()[0];
    copy.debuffed = false;
    copy.face_down = false;
    ctx.push_effect(SideEffect::AddCard {
        card: copy,
        to: Pile::Hand,
    });
}

// --- Splash: https://balatrowiki.org/w/Splash -----------------------------------------------
// Passive: every played card scores. Not copied by Blueprint.
fn splash_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.rules.splash = true;
}

// --- Sixth Sense: https://balatrowiki.org/w/Sixth_Sense -------------------------------------
// If the first hand of the round is a single 6, destroy it after scoring and create a Spectral
// card (room checked by the game layer). A debuffed 6 still counts. Stone cards have no rank.
// Not copied by Blueprint. Runs in `after_scoring`, which also runs after a boss-invalid hand;
// the wiki says a hand the boss rejects (The Psychic...) does not activate it, so skip those.
const SIXTH_SENSE_FIRST_HAND: u32 = 1;
const SIXTH_SENSE_RANK: Rank = Rank::Six;
const SIXTH_SENSE_CARDS: usize = 1;

fn sixth_sense_after(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.hand_invalid()
        || ctx.run().hands_played_round != SIXTH_SENSE_FIRST_HAND
        || ctx.played().len() != SIXTH_SENSE_CARDS
    {
        return;
    }
    let card = ctx.played()[0];
    if card.is_stone() || card.rank != SIXTH_SENSE_RANK {
        return;
    }
    ctx.push_effect(SideEffect::DestroyCard { uid: card.uid });
    ctx.push_effect(SideEffect::CreateConsumable {
        kind: ConsumableKind::Spectral,
        negative: false,
    });
}

// --- Superposition: https://balatrowiki.org/w/Superposition ---------------------------------
// Creates a Tarot card if the played hand contains a Straight and an Ace. The Ace must be a
// scoring card (with Four Fingers the fifth, non-scoring card does not count). A debuffed
// scoring Ace still counts (the game checks the rank only).
const SUPERPOSITION_RANK: Rank = Rank::Ace;

fn superposition_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if !ctx.contains(HandType::Straight) {
        return;
    }
    let ace = ctx.scoring_indices().any(|i| {
        let c = &ctx.played()[i];
        !c.is_stone() && c.rank == SUPERPOSITION_RANK
    });
    if ace {
        ctx.push_effect(SideEffect::CreateConsumable {
            kind: ConsumableKind::Tarot,
            negative: false,
        });
    }
}

// --- Séance: https://balatrowiki.org/w/S%C3%A9ance ------------------------------------------
// Creates a random Spectral card if the played hand is a Straight Flush (Royal Flush included).
// `contains` is used: only a Straight Flush hand can contain a Straight Flush.
fn seance_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.contains(HandType::StraightFlush) {
        ctx.push_effect(SideEffect::CreateConsumable {
            kind: ConsumableKind::Spectral,
            negative: false,
        });
    }
}

// --- Riff-Raff: https://balatrowiki.org/w/Riff-Raff -----------------------------------------
// When a blind is selected (any kind), create 2 Common Jokers as joker-slot room allows (the
// game layer drops what does not fit).
const RIFF_RAFF_JOKERS: usize = 2;

fn riff_raff_blind_select(_: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    for _ in 0..RIFF_RAFF_JOKERS {
        ctx.push_effect(SideEffect::CreateJoker {
            rarity: Some(Rarity::Common),
        });
    }
}

// --- Shortcut: https://balatrowiki.org/w/Shortcut -------------------------------------------
// Passive: Straights may skip one rank (10 8 7 6 5). Not copied by Blueprint.
fn shortcut_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.rules.shortcut = true;
}

// --- Vagabond: https://balatrowiki.org/w/Vagabond -------------------------------------------
// Creates a Tarot card if the hand is played with $4 or less. `ctx.money()` includes money
// gained earlier in this hand (Gold Seal cards, jokers to the left).
const VAGABOND_MAX_MONEY: i32 = 4;

fn vagabond_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.money() <= VAGABOND_MAX_MONEY {
        ctx.push_effect(SideEffect::CreateConsumable {
            kind: ConsumableKind::Tarot,
            negative: false,
        });
    }
}

// --- Midas Mask: https://balatrowiki.org/w/Midas_Mask ---------------------------------------
// Before scoring, every played scoring face card (`ctx.is_face`: Pareidolia aware, never a
// debuffed card) becomes a Gold card, overwriting any enhancement; edition and seal stay. The
// change is permanent and applies left to right, so later jokers of this step (Vampire) see it.
// Not copied by Blueprint.
fn midas_mask_before(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    for i in ctx.scoring_indices() {
        let face = ctx.played().get(i).is_some_and(|c| ctx.is_face(c));
        if face {
            ctx.set_enhancement(i, Enhancement::Gold);
        }
    }
}
