//! Consumable expert: which held card to use now and on which hand cards, and how good a
//! Tarot/Spectral pack item is. Rule based (outcomes of random effects are never previewed).
//!
//! Planets are always used at once. Tarots/Spectrals follow a per-card rule: the enhancing cards
//! act on the cards the agent is about to play (or hold), suit cards only convert toward the
//! deck's dominant suit, the risky Spectrals are left alone.

use crate::actions::Action;
use crate::card::Card;
use crate::consumable::ConsumableKind;
use crate::content::ids::ConsumableId as C;
use crate::enums::{Edition, Enhancement, Seal, Suit};
use crate::game::Game;
use crate::state::GameState;

/// What the planner needs to know besides the game.
#[derive(Debug, Clone, Copy)]
pub struct Ctx {
    /// Suit the run builds around.
    pub dominant: Suit,
    /// The best play right now (hand-position mask), if a hand is dealt.
    pub best_mask: Option<u16>,
}

fn is_plain(c: &Card) -> bool {
    c.enhancement == Enhancement::None && c.edition == Edition::Base && c.seal == Seal::None
}

/// Hand positions the agent is likely to play: the scoring cards of the best play, or (no best
/// play known) the five highest-ranked cards.
fn likely_played(s: &GameState, ctx: &Ctx) -> Vec<usize> {
    let hand = &s.deck.hand;
    match ctx.best_mask {
        Some(m) => {
            let sel: Vec<Card> = hand
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < 16 && (m >> i) & 1 == 1)
                .map(|(_, c)| *c)
                .collect();
            let ev = crate::hand_eval::evaluate(&sel, &s.mods.rules);
            let mut out = Vec::new();
            let mut j = 0;
            for i in 0..hand.len().min(16) {
                if (m >> i) & 1 == 1 {
                    if ev.scores(j) {
                        out.push(i);
                    }
                    j += 1;
                }
            }
            out
        }
        None => (0..hand.len().min(5)).collect(),
    }
}

fn mask_of(idx: &[usize]) -> u16 {
    idx.iter().fold(0u16, |m, &i| m | (1 << i))
}

/// Up to `n` plain, undebuffed cards among `from`, in the order given.
fn pick_plain(s: &GameState, from: &[usize], n: usize) -> Vec<usize> {
    from.iter()
        .copied()
        .filter(|&i| {
            s.deck
                .hand
                .get(i)
                .is_some_and(|c| is_plain(c) && !c.debuffed && !c.is_stone())
        })
        .take(n)
        .collect()
}

fn others(s: &GameState, played: &[usize]) -> Vec<usize> {
    (0..s.deck.hand.len().min(16))
        .filter(|i| !played.contains(i))
        .collect()
}

fn joker_sell_total(s: &GameState) -> i32 {
    s.jokers.iter().map(|j| j.sell_value).sum()
}

fn tarot_suit(id: C) -> Option<Suit> {
    match id {
        C::TheStar => Some(Suit::Diamonds),
        C::TheMoon => Some(Suit::Clubs),
        C::TheSun => Some(Suit::Hearts),
        C::TheWorld => Some(Suit::Spades),
        _ => None,
    }
}

/// `Some((targets, value))` if `id` is worth using now on those hand cards (`0` for untargeted
/// cards); the caller still validates the action. `value` ranks pack items against each other.
pub fn plan(id: C, game: &Game, ctx: &Ctx) -> Option<(u16, f64)> {
    let s = game.state();
    let hand_len = s.deck.hand.len();
    let has_hand = hand_len >= 2;
    let consumable_room = (s.consumables.len() as u32) < s.consumable_capacity();
    if id.kind() == ConsumableKind::Planet {
        return Some((0, 1.0));
    }
    let played = likely_played(s, ctx);
    let held = others(s, &played);
    let on_played = |n: usize, v: f64| {
        let t = pick_plain(s, &played, n);
        (!t.is_empty()).then(|| (mask_of(&t), v))
    };
    let on_held = |n: usize, v: f64| {
        let t = pick_plain(s, &held, n);
        (!t.is_empty()).then(|| (mask_of(&t), v))
    };
    match id {
        C::BlackHole | C::TheSoul => Some((0, 3.0)),
        C::TheFool => (consumable_room && s.last_tarot_planet.is_some()).then_some((0, 0.3)),
        C::TheHighPriestess => consumable_room.then_some((0, 0.5)),
        C::TheEmperor => consumable_room.then_some((0, 0.25)),
        C::Judgement => s.has_joker_room().then_some((0, 0.7)),
        C::TheWheelOfFortune => s
            .jokers
            .iter()
            .any(|j| j.edition == Edition::Base)
            .then_some((0, 0.3)),
        C::TheHermit => (s.money >= 10).then(|| (0, 0.2 + f64::from(s.money.min(20)) / 40.0)),
        C::Temperance => {
            let t = joker_sell_total(s);
            (t >= 8).then(|| (0, 0.1 + f64::from(t.min(50)) / 50.0))
        }
        C::TheEmpress if has_hand => on_played(2, 0.45),
        C::TheHierophant if has_hand => on_played(2, 0.4),
        C::TheMagician if has_hand => on_played(2, 0.3),
        C::TheLovers if has_hand => on_played(1, 0.2),
        C::Justice if has_hand => on_played(1, 0.25),
        C::TheChariot if has_hand => on_held(1, 0.3),
        C::TheDevil if has_hand => on_held(1, 0.3),
        C::TheHangedMan if has_hand && s.deck.len() > 40 => {
            // Destroy the two lowest plain junk cards that do not play now.
            let mut junk: Vec<usize> = held
                .iter()
                .copied()
                .filter(|&i| {
                    let c = &s.deck.hand[i];
                    is_plain(c)
                        && !c.is_stone()
                        && (c.rank as u8) <= 6
                        && !c.has_suit(ctx.dominant, &s.mods.rules)
                })
                .collect();
            junk.reverse(); // the hand is sorted high to low: lowest last
            junk.truncate(2);
            (!junk.is_empty()).then(|| (mask_of(&junk), 0.2))
        }
        C::TheStar | C::TheMoon | C::TheSun | C::TheWorld if has_hand => {
            let suit = tarot_suit(id)?;
            if suit != ctx.dominant {
                return None;
            }
            let t: Vec<usize> = (0..hand_len.min(16))
                .filter(|&i| {
                    let c = &s.deck.hand[i];
                    !c.is_stone()
                        && c.enhancement != Enhancement::Wild
                        && !c.has_suit(suit, &s.mods.rules)
                })
                .take(3)
                .collect();
            (!t.is_empty()).then(|| (mask_of(&t), 0.35))
        }
        // Spectrals.
        C::Talisman if has_hand => on_played(1, 0.45),
        C::DejaVu if has_hand => on_played(1, 0.4),
        C::Aura if has_hand => on_played(1, 0.5),
        C::Cryptid if has_hand => on_played(1, 0.35),
        C::Trance if has_hand => on_held(1, 0.25),
        C::Familiar if hand_len >= 5 && s.deck.len() <= 60 => Some((0, 0.45)),
        C::Grim if hand_len >= 5 && s.deck.len() <= 60 => Some((0, 0.4)),
        C::Incantation if hand_len >= 5 && s.deck.len() <= 60 => Some((0, 0.3)),
        C::Wraith => (s.has_joker_room() && s.money <= 12).then_some((0, 0.9)),
        C::Immolate => (hand_len >= 5 && s.money < 12 && s.deck.len() >= 46).then_some((0, 0.4)),
        // Left alone: Strength, Death, The Tower, Medium, Sigil, Ouija, Ectoplasm, Ankh, Hex.
        _ => None,
    }
}

/// The best valid `UseConsumable` over the held cards (highest plan value), if any.
pub fn best_held_action(game: &Game, ctx: &Ctx) -> Option<Action> {
    let s = game.state();
    let mut best: Option<(f64, Action)> = None;
    for (slot, c) in s.consumables.iter().enumerate() {
        let Ok(slot) = u8::try_from(slot) else { break };
        let Some((targets, v)) = plan(c.id, game, ctx) else {
            continue;
        };
        let a = Action::UseConsumable { slot, targets };
        if game.validate(a).is_ok() && best.is_none_or(|(bv, _)| v > bv) {
            best = Some((v, a));
        }
    }
    best.map(|(_, a)| a)
}
