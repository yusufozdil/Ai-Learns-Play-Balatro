//! Spectral cards (18). Phase 4B owns this file (tests: `tests/consumables_spectrals.rs`).
//!
//! Phase 4.0 prototypes: Black Hole, Ankh. 4B adds the other 16.
//! Guide: `docs/design/consumables_bosses.md`; rules `docs/mechanics/consumables.md` §3,
//! decisions `notes.md` §4.1b.

use crate::card::Card;
use crate::consumable::{ConsumableDef, UseCheck, UseCtx};
use crate::content::ConsumableId as C;
use crate::data::consumables::{ADDED_CARD_ENHANCEMENTS, EDITION_ROLL};
use crate::deck::Pile;
use crate::enums::{Edition, HandType, Rank, Rarity, Seal, Suit};

pub static DEFS: &[ConsumableDef] = &[
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Familiar, "Familiar", familiar)
    },
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Grim, "Grim", grim)
    },
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Incantation, "Incantation", incantation)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        ..ConsumableDef::new(C::Talisman, "Talisman", talisman)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        can_use: Some(aura_can_use),
        ..ConsumableDef::new(C::Aura, "Aura", aura)
    },
    ConsumableDef {
        can_use: Some(free_joker_slot),
        ..ConsumableDef::new(C::Wraith, "Wraith", wraith)
    },
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Sigil, "Sigil", sigil)
    },
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Ouija, "Ouija", ouija)
    },
    ConsumableDef {
        can_use: Some(has_editionless_joker),
        ..ConsumableDef::new(C::Ectoplasm, "Ectoplasm", ectoplasm)
    },
    ConsumableDef {
        needs_hand: true,
        can_use: Some(hand_of_two),
        ..ConsumableDef::new(C::Immolate, "Immolate", immolate)
    },
    ConsumableDef {
        can_use: Some(ankh_can_use),
        ..ConsumableDef::new(C::Ankh, "Ankh", ankh)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        ..ConsumableDef::new(C::DejaVu, "Deja Vu", deja_vu)
    },
    ConsumableDef {
        can_use: Some(has_editionless_joker),
        ..ConsumableDef::new(C::Hex, "Hex", hex)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        ..ConsumableDef::new(C::Trance, "Trance", trance)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        ..ConsumableDef::new(C::Medium, "Medium", medium)
    },
    ConsumableDef {
        min_targets: 1,
        max_targets: 1,
        ..ConsumableDef::new(C::Cryptid, "Cryptid", cryptid)
    },
    ConsumableDef {
        can_use: Some(free_joker_slot),
        ..ConsumableDef::new(C::TheSoul, "The Soul", the_soul)
    },
    ConsumableDef::new(C::BlackHole, "Black Hole", black_hole),
];

// --- shared helpers ------------------------------------------------------------------------

/// Fewest hand cards for the cards that destroy or rewrite the whole hand (Familiar, Grim,
/// Incantation, Sigil, Ouija, Immolate: the wiki says they cannot be used with one card).
const MIN_HAND_FOR_HAND_CARDS: usize = 2;

fn hand_of_two(c: &UseCheck<'_>) -> bool {
    c.hand().len() >= MIN_HAND_FOR_HAND_CARDS
}

fn free_joker_slot(c: &UseCheck<'_>) -> bool {
    c.free_joker_slots() > 0
}

fn has_editionless_joker(c: &UseCheck<'_>) -> bool {
    c.jokers().iter().any(|j| j.edition == Edition::Base)
}

/// Destroy `n` distinct random hand cards (partial Fisher-Yates over the positions; `n` is
/// clamped to the hand size). One `rng_range` roll per destroyed card.
fn destroy_random_cards(ctx: &mut UseCtx<'_>, n: usize) {
    let mut pos: Vec<usize> = (0..ctx.hand().len()).collect();
    for k in 0..n.min(pos.len()) {
        let j = ctx.rng_range(k, pos.len());
        pos.swap(k, j);
        ctx.destroy_card(pos[k]);
    }
}

/// A random added card: a rank from `ranks`, a random suit and a random enhancement from
/// `ADDED_CARD_ENHANCEMENTS` (never Stone). Roll order: rank, suit, enhancement.
fn random_added_card(ctx: &mut UseCtx<'_>, ranks: &[Rank]) -> Card {
    let rank = ranks[ctx.rng_range(0, ranks.len())];
    let suit = Suit::ALL[ctx.rng_range(0, Suit::COUNT)];
    let enh = ADDED_CARD_ENHANCEMENTS[ctx.rng_range(0, ADDED_CARD_ENHANCEMENTS.len())];
    let mut card = Card::new(0, rank, suit);
    card.enhancement = enh;
    card
}

/// Destroy one random hand card, then add `count` random enhanced cards of `ranks` to the hand.
fn destroy_one_add_cards(ctx: &mut UseCtx<'_>, count: usize, ranks: &[Rank]) {
    destroy_random_cards(ctx, 1);
    for _ in 0..count {
        let card = random_added_card(ctx, ranks);
        ctx.add_card(card, Pile::Hand);
    }
}

/// Random joker slot without an edition (`None` when there is none). One roll when non-empty.
fn random_editionless_joker(ctx: &mut UseCtx<'_>) -> Option<usize> {
    let n = ctx
        .jokers()
        .iter()
        .filter(|j| j.edition == Edition::Base)
        .count();
    if n == 0 {
        return None;
    }
    let k = ctx.rng_range(0, n);
    ctx.jokers()
        .iter()
        .enumerate()
        .filter(|(_, j)| j.edition == Edition::Base)
        .nth(k)
        .map(|(slot, _)| slot)
}

// --- Familiar: https://balatrowiki.org/w/Familiar ---------------------------------------------
const FAMILIAR_CARDS: usize = 3;
const FACE_RANKS: [Rank; 3] = [Rank::Jack, Rank::Queen, Rank::King];

fn familiar(ctx: &mut UseCtx<'_>) {
    destroy_one_add_cards(ctx, FAMILIAR_CARDS, &FACE_RANKS);
}

// --- Grim: https://balatrowiki.org/w/Grim ------------------------------------------------------
const GRIM_CARDS: usize = 2;

fn grim(ctx: &mut UseCtx<'_>) {
    destroy_one_add_cards(ctx, GRIM_CARDS, &[Rank::Ace]);
}

// --- Incantation: https://balatrowiki.org/w/Incantation -----------------------------------------
const INCANTATION_CARDS: usize = 4;
const NUMBER_RANKS: [Rank; 9] = [
    Rank::Two,
    Rank::Three,
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Eight,
    Rank::Nine,
    Rank::Ten,
];

fn incantation(ctx: &mut UseCtx<'_>) {
    destroy_one_add_cards(ctx, INCANTATION_CARDS, &NUMBER_RANKS);
}

// --- Talisman / Deja Vu / Trance / Medium (seals; replace an existing seal) -------------------

fn set_target_seal(ctx: &mut UseCtx<'_>, seal: Seal) {
    if let Some(pos) = ctx.target_pos(0) {
        ctx.set_seal(pos, seal);
    }
}

// https://balatrowiki.org/w/Talisman
fn talisman(ctx: &mut UseCtx<'_>) {
    set_target_seal(ctx, Seal::Gold);
}

// https://balatrowiki.org/w/Deja_Vu
fn deja_vu(ctx: &mut UseCtx<'_>) {
    set_target_seal(ctx, Seal::Red);
}

// https://balatrowiki.org/w/Trance
fn trance(ctx: &mut UseCtx<'_>) {
    set_target_seal(ctx, Seal::Blue);
}

// https://balatrowiki.org/w/Medium
fn medium(ctx: &mut UseCtx<'_>) {
    set_target_seal(ctx, Seal::Purple);
}

// --- Aura: https://balatrowiki.org/w/Aura ------------------------------------------------------
// Foil 50 / Holographic 35 / Polychrome 15 (`EDITION_ROLL`, notes.md §4.1b). The target must have
// no edition (the base game only offers Aura on edition-less cards).

fn aura_can_use(c: &UseCheck<'_>) -> bool {
    c.targets()
        .get(0)
        .is_some_and(|t| t.edition == Edition::Base)
}

fn aura(ctx: &mut UseCtx<'_>) {
    let Some(pos) = ctx.target_pos(0) else {
        return;
    };
    let weights = EDITION_ROLL.map(|(_, w)| w);
    if let Some(i) = ctx.weighted_index(&weights) {
        ctx.set_edition(pos, EDITION_ROLL[i].0);
    }
}

// --- Wraith: https://balatrowiki.org/w/Wraith ---------------------------------------------------
// Random Rare joker and money set to $0 (also when negative,
// notes.md §4.1b). Needs a free joker slot.
const WRAITH_MONEY: i32 = 0;

fn wraith(ctx: &mut UseCtx<'_>) {
    ctx.create_joker(Some(Rarity::Rare));
    ctx.set_money(WRAITH_MONEY);
}

// --- The Soul: https://balatrowiki.org/w/The_Soul -----------------------------------------------
// Random Legendary joker. Needs a free joker slot.

fn the_soul(ctx: &mut UseCtx<'_>) {
    ctx.create_joker(Some(Rarity::Legendary));
}

// --- Sigil: https://balatrowiki.org/w/Sigil ------------------------------------------------------
// One random suit (the current one allowed) for every non-Stone hand card.

fn sigil(ctx: &mut UseCtx<'_>) {
    let suit = Suit::ALL[ctx.rng_range(0, Suit::COUNT)];
    for pos in 0..ctx.hand().len() {
        if !ctx.hand()[pos].is_stone() {
            ctx.set_suit(pos, suit);
        }
    }
}

// --- Ouija: https://balatrowiki.org/w/Ouija -------------------------------------------------------
// One random rank for every non-Stone hand card; flat -1 hand size per use.
const OUIJA_HAND_SIZE: i32 = -1;

fn ouija(ctx: &mut UseCtx<'_>) {
    let rank = Rank::ALL[ctx.rng_range(0, Rank::ALL.len())];
    for pos in 0..ctx.hand().len() {
        if !ctx.hand()[pos].is_stone() {
            ctx.set_rank(pos, rank);
        }
    }
    ctx.add_hand_size(OUIJA_HAND_SIZE);
}

// --- Ectoplasm: https://balatrowiki.org/w/Ectoplasm ------------------------------------------------
// Negative on a random edition-less joker; hand size -(1 + earlier uses this run). Wiki: "X = 1 +
// each time Ectoplasm has been used in the current run" (1+2+3 = -6 after three uses; 0.9.3L).

fn ectoplasm(ctx: &mut UseCtx<'_>) {
    if let Some(slot) = random_editionless_joker(ctx) {
        ctx.set_joker_edition(slot, Edition::Negative);
    }
    let earlier = i32::try_from(ctx.uses_of(C::Ectoplasm)).unwrap_or(i32::MAX - 1);
    ctx.add_hand_size(-(earlier + 1));
}

// --- Immolate: https://balatrowiki.org/w/Immolate ---------------------------------------------------
// `min(5, hand)` distinct random cards, +$20 (the wiki is silent below 5 cards: notes.md §4.1b).
const IMMOLATE_CARDS: usize = 5;
const IMMOLATE_MONEY: i32 = 20;

fn immolate(ctx: &mut UseCtx<'_>) {
    destroy_random_cards(ctx, IMMOLATE_CARDS);
    ctx.add_money(IMMOLATE_MONEY);
}

// --- Hex: https://balatrowiki.org/w/Hex ----------------------------------------------------------------
// Polychrome on a random edition-less joker, every other non-Eternal joker destroyed.

fn hex(ctx: &mut UseCtx<'_>) {
    let Some(keep) = random_editionless_joker(ctx) else {
        return;
    };
    for slot in 0..ctx.jokers().len() {
        if slot != keep {
            ctx.destroy_joker(slot); // refuses Eternal jokers
        }
    }
    ctx.set_joker_edition(keep, Edition::Polychrome);
}

// --- Cryptid: https://balatrowiki.org/w/Cryptid ------------------------------------------------------------
// Two exact copies (rank, suit, enhancement, edition, seal, bonus chips) added to the hand.
const CRYPTID_COPIES: usize = 2;

fn cryptid(ctx: &mut UseCtx<'_>) {
    let Some(&src) = ctx.target_pos(0).and_then(|p| ctx.hand().get(p)) else {
        return;
    };
    for _ in 0..CRYPTID_COPIES {
        let mut copy = Card::new(0, src.rank, src.suit);
        copy.enhancement = src.enhancement;
        copy.edition = src.edition;
        copy.seal = src.seal;
        copy.perma_bonus = src.perma_bonus;
        ctx.add_card(copy, Pile::Hand);
    }
}

// --- Black Hole: https://balatrowiki.org/w/Black_Hole -----------------------------------------
const BLACK_HOLE_LEVELS: i32 = 1;

fn black_hole(ctx: &mut UseCtx<'_>) {
    for ht in HandType::ALL {
        ctx.level_up(ht, BLACK_HOLE_LEVELS);
    }
}

// --- Ankh: https://balatrowiki.org/w/Ankh -----------------------------------------------------
// Copy a random joker (any, Eternal included; the copy loses Negative), destroy every other
// non-Eternal joker. Needs at least one joker and a free joker slot (consumables.md §3: the
// slot check happens before the destruction). Destructions are queued BEFORE the copy so the
// copy's room check sees the freed slots.

fn ankh_can_use(c: &UseCheck<'_>) -> bool {
    !c.jokers().is_empty() && c.free_joker_slots() > 0
}

fn ankh(ctx: &mut UseCtx<'_>) {
    let n = ctx.jokers().len();
    if n == 0 {
        return;
    }
    let keep = ctx.rng_range(0, n);
    for slot in 0..n {
        if slot != keep {
            ctx.destroy_joker(slot); // refuses Eternal jokers
        }
    }
    ctx.copy_joker(keep, true);
}
