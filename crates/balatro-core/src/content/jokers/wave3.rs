//! Phase 3 joker package `wave3`: CreditCard, ChaosTheClown, DelayedGratification, BusinessCard, Egg, ToDoList, Cloud9, Rocket, GiftCard, ReservedParking, MailInRebate, ToTheMoon, TradingCard, GoldenTicket, RoughGem, Matador, Satellite, Astronomer.
//!
//! Follow `docs/design/hooks.md`. Each block cites its wiki page; numbers are named constants
//! next to the joker. Tests: `tests/jokers_wave3.rs`.

use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{Enhancement, HandType, Rank, Rarity, Suit};
use crate::hooks::{DiscardInfo, EventCtx, Hooks, Modifiers, RoundEndInfo, SellTarget, SideEffect};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

/// Definitions in this file.
pub static DEFS: &[JokerDef] = &[
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(credit_card_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::CreditCard, "Credit Card", Rarity::Common, 1)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(chaos_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ChaosTheClown, "Chaos the Clown", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(delayed_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::DelayedGratification,
            "Delayed Gratification",
            Rarity::Common,
            4,
        )
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(business_card_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::BusinessCard, "Business Card", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(egg_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Egg, "Egg", Rarity::Common, 4)
    },
    JokerDef {
        hooks: Hooks {
            before_scoring: Some(to_do_before),
            on_acquire: Some(to_do_acquire),
            on_round_end: Some(to_do_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ToDoList, "To Do List", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(cloud9_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Cloud9, "Cloud 9", Rarity::Uncommon, 7)
    },
    JokerDef {
        blueprint_compatible: false,
        init_value: ROCKET_BASE_PAYOUT,
        hooks: Hooks {
            on_round_end: Some(rocket_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Rocket, "Rocket", Rarity::Uncommon, 6)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(gift_card_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GiftCard, "Gift Card", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_held: Some(reserved_parking_held),
            ..Hooks::NONE
        },
        ..JokerDef::new(
            JokerId::ReservedParking,
            "Reserved Parking",
            Rarity::Common,
            6,
        )
    },
    JokerDef {
        hooks: Hooks {
            on_acquire: Some(mail_in_acquire),
            on_discard: Some(mail_in_discard),
            on_round_end: Some(mail_in_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MailInRebate, "Mail-In Rebate", Rarity::Common, 4)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(to_the_moon_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::ToTheMoon, "To the Moon", Rarity::Uncommon, 5)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_discard: Some(trading_card_discard),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::TradingCard, "Trading Card", Rarity::Uncommon, 6)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(golden_ticket_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::GoldenTicket, "Golden Ticket", Rarity::Common, 5)
    },
    JokerDef {
        hooks: Hooks {
            on_card_scored: Some(rough_gem_scored),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::RoughGem, "Rough Gem", Rarity::Uncommon, 7)
    },
    JokerDef {
        hooks: Hooks {
            joker_main: Some(matador_main),
            on_invalid_hand: Some(matador_invalid),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Matador, "Matador", Rarity::Uncommon, 7)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            on_round_end: Some(satellite_round_end),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Satellite, "Satellite", Rarity::Uncommon, 6)
    },
    JokerDef {
        blueprint_compatible: false,
        hooks: Hooks {
            modifiers: Some(astronomer_modifiers),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::Astronomer, "Astronomer", Rarity::Uncommon, 8)
    },
];

// --- Credit Card: https://balatrowiki.org/w/Credit_Card -------------------------------------
const CREDIT_CARD_DEBT: i32 = 20;

fn credit_card_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.debt_limit = m.debt_limit.saturating_add(CREDIT_CARD_DEBT);
}

// --- Chaos the Clown: https://balatrowiki.org/w/Chaos_the_Clown -----------------------------
const CHAOS_FREE_REROLLS: u32 = 1;

fn chaos_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.free_rerolls = m.free_rerolls.saturating_add(CHAOS_FREE_REROLLS);
}

// --- Delayed Gratification: https://balatrowiki.org/w/Delayed_Gratification -----------------
const DELAYED_PER_DISCARD: i32 = 2;

fn delayed_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    let run = ctx.run();
    if run.discards_used_round == 0 && run.discards_left > 0 {
        let left = i32::try_from(run.discards_left).unwrap_or(i32::MAX);
        ctx.add_money(DELAYED_PER_DISCARD.saturating_mul(left));
    }
}

// --- Business Card: https://balatrowiki.org/w/Business_Card ---------------------------------
const BUSINESS_CARD_ODDS: (u32, u32) = (1, 2);
const BUSINESS_CARD_MONEY: i32 = 2;

fn business_card_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    // One roll per trigger of each scoring face card (retriggers roll again).
    if ctx.is_face(card) && ctx.chance(BUSINESS_CARD_ODDS.0, BUSINESS_CARD_ODDS.1) {
        ctx.add_money(BUSINESS_CARD_MONEY);
    }
}

// --- Egg: https://balatrowiki.org/w/Egg -----------------------------------------------------
const EGG_SELL_GAIN: i32 = 3;

fn egg_round_end(j: &mut JokerInstance, _: &mut EventCtx<'_>, _: &RoundEndInfo) {
    j.sell_value = j.sell_value.saturating_add(EGG_SELL_GAIN);
}

// --- To Do List: https://balatrowiki.org/w/To_Do_List ---------------------------------------
// aux_hand = target hand type. Re-picked on acquire and at every round end (different from the
// current target). Secret hands are only eligible once played this run (wiki: any visible hand).
const TO_DO_MONEY: i32 = 4;

fn is_secret_hand(h: HandType) -> bool {
    matches!(
        h,
        HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive
    )
}

fn pick_to_do_target(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let current = j.aux_hand();
    let plays = ctx.run().hand_plays_run;
    let eligible =
        |h: &HandType| Some(*h) != current && (!is_secret_hand(*h) || plays[h.index()] > 0);
    let n = HandType::ALL.iter().filter(|h| eligible(h)).count();
    if n == 0 {
        return;
    }
    let k = ctx.rng_range(0, n);
    let pick = HandType::ALL.iter().filter(|h| eligible(h)).nth(k).copied();
    j.set_aux_hand(pick);
}

fn to_do_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    pick_to_do_target(j, ctx);
}

fn to_do_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    pick_to_do_target(j, ctx);
}

fn to_do_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.aux_hand() == Some(ctx.hand_type()) {
        ctx.add_money(TO_DO_MONEY);
    }
}

// --- Cloud 9: https://balatrowiki.org/w/Cloud_9 ---------------------------------------------
const CLOUD9_PER_NINE: i32 = 1;

fn cloud9_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    // Full deck (all piles); Stone cards have no rank and are not counted.
    let nines = i32::try_from(ctx.run().deck.rank(Rank::Nine)).unwrap_or(i32::MAX);
    ctx.add_money(CLOUD9_PER_NINE.saturating_mul(nines));
}

// --- Rocket: https://balatrowiki.org/w/Rocket -----------------------------------------------
// value = current payout in dollars. The +$2 for a defeated Boss Blind is applied BEFORE that
// round's payout (jokers.csv).
const ROCKET_BASE_PAYOUT: f64 = 1.0;
const ROCKET_BOSS_GAIN: f64 = 2.0;

fn rocket_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, info: &RoundEndInfo) {
    if info.boss_defeated(ctx.run()) {
        j.value += ROCKET_BOSS_GAIN;
    }
    ctx.add_money(j.value as i32);
}

// --- Gift Card: https://balatrowiki.org/w/Gift_Card -----------------------------------------
const GIFT_CARD_SELL_GAIN: i32 = 1;

fn gift_card_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    ctx.push_effect(SideEffect::AddSellValue {
        target: SellTarget::AllJokers,
        amount: GIFT_CARD_SELL_GAIN,
    });
    // Only when something is held: the game layer cannot apply this effect to consumables yet
    // (Phase 4) and drops it, so an empty queue entry would only inflate `effects_dropped`.
    if !ctx.run().consumables.is_empty() {
        ctx.push_effect(SideEffect::AddSellValue {
            target: SellTarget::AllConsumables,
            amount: GIFT_CARD_SELL_GAIN,
        });
    }
}

// --- Reserved Parking: https://balatrowiki.org/w/Reserved_Parking ---------------------------
const RESERVED_PARKING_ODDS: (u32, u32) = (1, 2);
const RESERVED_PARKING_MONEY: i32 = 1;

fn reserved_parking_held(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.is_face(card) && ctx.chance(RESERVED_PARKING_ODDS.0, RESERVED_PARKING_ODDS.1) {
        ctx.add_money(RESERVED_PARKING_MONEY);
    }
}

// --- Mail-In Rebate: https://balatrowiki.org/w/Mail-In_Rebate ------------------------------
// aux_rank = target rank: a random non-Stone deck card's rank (weighted by frequency), Ace if
// the deck has none. Re-picked on acquire and at every round end. Debuffed discarded cards pay
// nothing (wiki: "debuffed cards do not have a rank"; balatrowiki.org/w/Debuffed).
const MAIL_IN_MONEY: i32 = 5;
const MAIL_IN_DEFAULT_RANK: Rank = Rank::Ace;

fn pick_mail_in_rank(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    let n = ctx.deck().iter_all().filter(|c| !c.is_stone()).count();
    let rank = if n == 0 {
        MAIL_IN_DEFAULT_RANK
    } else {
        let k = ctx.rng_range(0, n);
        ctx.deck()
            .iter_all()
            .filter(|c| !c.is_stone())
            .nth(k)
            .map_or(MAIL_IN_DEFAULT_RANK, |c| c.rank)
    };
    j.set_aux_rank(Some(rank));
}

fn mail_in_acquire(j: &mut JokerInstance, ctx: &mut EventCtx<'_>) {
    pick_mail_in_rank(j, ctx);
}

fn mail_in_round_end(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    pick_mail_in_rank(j, ctx);
}

fn mail_in_discard(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    let Some(target) = j.aux_rank() else { return };
    let hits = d
        .cards
        .iter()
        .filter(|c| !c.is_stone() && !c.debuffed && c.rank == target)
        .count();
    let hits = i32::try_from(hits).unwrap_or(i32::MAX);
    if hits > 0 {
        ctx.add_money(MAIL_IN_MONEY.saturating_mul(hits));
    }
}

// --- To the Moon: https://balatrowiki.org/w/To_the_Moon -------------------------------------
const TO_THE_MOON_BONUS: i32 = 1;

fn to_the_moon_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.interest_per_5 = m.interest_per_5.saturating_add(TO_THE_MOON_BONUS);
}

// --- Trading Card: https://balatrowiki.org/w/Trading_Card -----------------------------------
const TRADING_CARD_MONEY: i32 = 3;

fn trading_card_discard(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    if ctx.run().discards_used_round == 1
        && let [card] = d.cards
    {
        ctx.push_effect(SideEffect::DestroyCard { uid: card.uid });
        ctx.add_money(TRADING_CARD_MONEY);
    }
}

// --- Golden Ticket: https://balatrowiki.org/w/Golden_Ticket --------------------------------
const GOLDEN_TICKET_MONEY: i32 = 4;

fn golden_ticket_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if card.enhancement == Enhancement::Gold {
        ctx.add_money(GOLDEN_TICKET_MONEY);
    }
}

// --- Rough Gem: https://balatrowiki.org/w/Rough_Gem -----------------------------------------
const ROUGH_GEM_MONEY: i32 = 1;

fn rough_gem_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Diamonds) {
        ctx.add_money(ROUGH_GEM_MONEY);
    }
}

// --- Matador: https://balatrowiki.org/w/Matador ---------------------------------------------
// Pays when the played hand triggered the Boss Blind ability (`run.boss.triggered`, set by the
// game layer in Phase 4). Valid hands pay in `joker_main`, rejected hands in `on_invalid_hand`.
const MATADOR_MONEY: i32 = 8;

fn matador_main(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.run().boss.triggered {
        ctx.add_money(MATADOR_MONEY);
    }
}

fn matador_invalid(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.run().boss.triggered {
        ctx.add_money(MATADOR_MONEY);
    }
}

// --- Satellite: https://balatrowiki.org/w/Satellite -----------------------------------------
const SATELLITE_PER_PLANET: i32 = 1;

fn satellite_round_end(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, _: &RoundEndInfo) {
    let n = i32::try_from(ctx.run().stats.unique_planets_used).unwrap_or(i32::MAX);
    ctx.add_money(SATELLITE_PER_PLANET.saturating_mul(n));
}

// --- Astronomer: https://balatrowiki.org/w/Astronomer --------------------------------------
fn astronomer_modifiers(_: &JokerInstance, m: &mut Modifiers) {
    m.free_planets = true;
}
