//! Phase 3 `wave3` economy jokers (content/jokers/wave3.rs): trigger / non-trigger / state /
//! Blueprint for each, plus Game-level checks of the cash-out path (joker payouts are not
//! interest base, notes.md §4.1c). Every expected number is derived in a comment.

use balatro_core::actions::Action;
use balatro_core::card::Card;
use balatro_core::content::BossId;
use balatro_core::content::ConsumableId;
use balatro_core::content::jokers::{JokerId, try_def};
use balatro_core::deck::Deck;
use balatro_core::enums::{Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{
    BlindKind, BossView, ConsumableView, DiscardInfo, EventInput, RoundEndInfo, RunView,
    SellTarget, SideEffect, collect_modifiers, fire_acquire, fire_discard, fire_round_end,
    resolve_source,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer};
use balatro_core::state::GameConfig;

use JokerId as J;
use Rank::*;
use Suit::*;

// ----- helpers -----

fn c(uid: u32, rank: Rank, suit: Suit) -> Card {
    Card::new(uid, rank, suit)
}

fn enh(mut card: Card, e: Enhancement) -> Card {
    card.enhancement = e;
    card
}

fn seal(mut card: Card, s: Seal) -> Card {
    card.seal = s;
    card
}

fn debuff(mut card: Card) -> Card {
    card.debuffed = true;
    card
}

fn ev(hand_type: HandType, scoring_mask: u8) -> HandEval {
    HandEval {
        hand_type,
        scoring_mask,
        contains: 1u16 << hand_type.index(),
    }
}

fn jk(uid: u32, id: JokerId) -> JokerInstance {
    JokerInstance::new(uid, id)
}

/// Jokers with uids 100, 101, ... in the given order.
fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| jk(100 + i as u32, id))
        .collect()
}

fn score(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    js: &mut [JokerInstance],
    run: &RunView<'_>,
) -> ScoreResult {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(11);
    Scorer::new().score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    })
}

/// `prob_mult` 1000: every `chance` succeeds; 0: none does.
fn pm(prob_mult: u32) -> RunView<'static> {
    RunView {
        prob_mult,
        ..RunView::default()
    }
}

fn pair_of(a: Card, b: Card) -> ([Card; 2], HandEval) {
    ([a, b], ev(HandType::Pair, 0b11))
}

fn money_of(played: &[Card], held: &[Card], js: &mut [JokerInstance], run: &RunView<'_>) -> i32 {
    score(played, held, ev(HandType::Pair, 0b11), js, run).money_delta
}

/// Round-end event; returns (money delta, queued effects).
fn end_round(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    deck: &Deck,
    won: bool,
) -> (i32, Vec<SideEffect>) {
    let mut rng = GameRng::new(2);
    let mut fx = Vec::new();
    let r = fire_round_end(
        EventInput {
            jokers: js,
            run,
            deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        &RoundEndInfo {
            won,
            saved: false,
            round_score: 400.0,
            target: 300.0,
        },
    );
    (r.money_delta, fx)
}

fn round_end(js: &mut [JokerInstance], run: &RunView<'_>) -> (i32, Vec<SideEffect>) {
    end_round(js, run, &Deck::standard(), true)
}

fn discard_event(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    cards: &[Card],
) -> (i32, Vec<SideEffect>) {
    let deck = Deck::standard();
    let mut rng = GameRng::new(2);
    let mut fx = Vec::new();
    let r = fire_discard(
        EventInput {
            jokers: js,
            run,
            deck: &deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        &DiscardInfo {
            cards,
            hand_type: HandType::HighCard,
            forced: false,
        },
    );
    (r.money_delta, fx)
}

fn acquire(js: &mut [JokerInstance], slot: usize, deck: &Deck, seed: u64) {
    let run = RunView::default();
    let mut rng = GameRng::new(seed);
    let mut fx = Vec::new();
    fire_acquire(
        EventInput {
            jokers: js,
            run: &run,
            deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        slot,
    );
}

fn deck_of(cards: Vec<Card>) -> Deck {
    let mut d = Deck::standard();
    d.draw_pile = cards;
    d
}

// ----- registry / flags -----

#[test]
fn all_wave3_jokers_are_defined_with_expected_blueprint_flags() {
    let not_copyable = [
        J::CreditCard,
        J::ChaosTheClown,
        J::DelayedGratification,
        J::Egg,
        J::Cloud9,
        J::Rocket,
        J::GiftCard,
        J::ToTheMoon,
        J::TradingCard,
        J::Satellite,
        J::Astronomer,
    ];
    let copyable = [
        J::BusinessCard,
        J::ToDoList,
        J::ReservedParking,
        J::MailInRebate,
        J::GoldenTicket,
        J::RoughGem,
        J::Matador,
    ];
    for id in not_copyable {
        let d = try_def(id).expect("defined");
        assert!(!d.blueprint_compatible, "{id:?} must be bp-incompatible");
        // A Blueprint in front of it gives nothing.
        assert_eq!(resolve_source(&jokers(&[J::Blueprint, id]), 0), None);
    }
    for id in copyable {
        let d = try_def(id).expect("defined");
        assert!(d.blueprint_compatible, "{id:?} must be copyable");
        assert_eq!(resolve_source(&jokers(&[J::Blueprint, id]), 0), Some(1));
    }
}

// ----- Credit Card / Chaos / To the Moon / Astronomer (modifiers) -----

#[test]
fn credit_card_adds_twenty_debt_limit_and_stacks() {
    assert_eq!(collect_modifiers(&jokers(&[])).debt_limit, 0);
    assert_eq!(collect_modifiers(&jokers(&[J::CreditCard])).debt_limit, 20);
    // Two copies: 40 (no cap in the modifier; each joker adds its own $20).
    let m = collect_modifiers(&jokers(&[J::CreditCard, J::CreditCard]));
    assert_eq!(m.debt_limit, 40);
    // Debuffed: nothing. Blueprint does not copy passives.
    let js = vec![jk(1, J::CreditCard).with_debuff(true)];
    assert_eq!(collect_modifiers(&js).debt_limit, 0);
    assert_eq!(
        collect_modifiers(&jokers(&[J::Blueprint, J::CreditCard])).debt_limit,
        20
    );
}

#[test]
fn chaos_the_clown_gives_one_free_reroll() {
    assert_eq!(
        collect_modifiers(&jokers(&[J::ChaosTheClown])).free_rerolls,
        1
    );
    assert_eq!(
        collect_modifiers(&jokers(&[J::ChaosTheClown, J::ChaosTheClown])).free_rerolls,
        2
    );
    assert_eq!(collect_modifiers(&jokers(&[J::Joker])).free_rerolls, 0);
    let js = vec![jk(1, J::ChaosTheClown).with_debuff(true)];
    assert_eq!(collect_modifiers(&js).free_rerolls, 0);
}

#[test]
fn to_the_moon_adds_one_interest_per_five() {
    assert_eq!(
        collect_modifiers(&jokers(&[J::ToTheMoon])).interest_per_5,
        1
    );
    assert_eq!(collect_modifiers(&jokers(&[])).interest_per_5, 0);
    let js = vec![jk(1, J::ToTheMoon).with_debuff(true)];
    assert_eq!(collect_modifiers(&js).interest_per_5, 0);
    // Blueprint copies no passives.
    assert_eq!(
        collect_modifiers(&jokers(&[J::Blueprint, J::ToTheMoon])).interest_per_5,
        1
    );
}

#[test]
fn astronomer_makes_planets_free() {
    assert!(collect_modifiers(&jokers(&[J::Astronomer])).free_planets);
    assert!(!collect_modifiers(&jokers(&[J::Joker])).free_planets);
    let js = vec![jk(1, J::Astronomer).with_debuff(true)];
    assert!(!collect_modifiers(&js).free_planets);
}

// ----- Delayed Gratification -----

#[test]
fn delayed_gratification_pays_two_per_unused_discard_if_none_used() {
    let mut js = jokers(&[J::DelayedGratification]);
    let run = |used, left| RunView {
        discards_used_round: used,
        discards_left: left,
        ..RunView::default()
    };
    // 3 discards left, none used: 3 x $2 = $6.
    assert_eq!(round_end(&mut js, &run(0, 3)).0, 6);
    // Red Deck: 4 discards -> $8.
    assert_eq!(round_end(&mut js, &run(0, 4)).0, 8);
    // One discard used (even with 3 left): nothing.
    assert_eq!(round_end(&mut js, &run(1, 3)).0, 0);
    // No discards at all (0 left, 0 used): nothing.
    assert_eq!(round_end(&mut js, &run(0, 0)).0, 0);
    // Debuffed: nothing.
    let mut d = vec![jk(1, J::DelayedGratification).with_debuff(true)];
    assert_eq!(round_end(&mut d, &run(0, 3)).0, 0);
    // Blueprint does not copy it (end-of-round hooks are never copied): still just $6.
    let mut bp = jokers(&[J::Blueprint, J::DelayedGratification]);
    assert_eq!(round_end(&mut bp, &run(0, 3)).0, 6);
}

// ----- Business Card -----

#[test]
fn business_card_pays_two_per_face_with_half_chance() {
    let (p, _) = pair_of(c(1, King, Spades), c(2, Queen, Hearts));
    // Always succeeds: two scoring faces x $2 = $4.
    let mut js = jokers(&[J::BusinessCard]);
    assert_eq!(money_of(&p, &[], &mut js, &pm(1000)), 4);
    // Never succeeds: $0.
    assert_eq!(money_of(&p, &[], &mut js, &pm(0)), 0);
    // Oops! All 6s doubles 1-in-2 to certainty: prob_mult 2 -> $4.
    assert_eq!(money_of(&p, &[], &mut js, &pm(2)), 4);
    // Non-face cards: $0 even when always succeeding.
    let (p7, _) = pair_of(c(1, Seven, Spades), c(2, Seven, Hearts));
    assert_eq!(money_of(&p7, &[], &mut js, &pm(1000)), 0);
    // Debuffed face is not scored: only the Queen pays $2.
    let (pd, _) = pair_of(debuff(c(1, King, Spades)), c(2, Queen, Hearts));
    assert_eq!(money_of(&pd, &[], &mut js, &pm(1000)), 2);
    // Red Seal Queen triggers twice: 2 + 2 + King 2 = $6.
    let (pr, _) = pair_of(c(1, King, Spades), seal(c(2, Queen, Hearts), Seal::Red));
    assert_eq!(money_of(&pr, &[], &mut js, &pm(1000)), 6);
    // Pareidolia makes 7s faces: $4.
    let mut run = pm(1000);
    run.rules.pareidolia = true;
    assert_eq!(money_of(&p7, &[], &mut js, &run), 4);
}

#[test]
fn business_card_blueprint_doubles_and_rolls_are_deterministic() {
    let (p, _) = pair_of(c(1, King, Spades), c(2, Queen, Hearts));
    // Blueprint + Business Card: two jokers pay -> 2 faces x ($2 + $2) = $8.
    let mut js = jokers(&[J::Blueprint, J::BusinessCard]);
    assert_eq!(money_of(&p, &[], &mut js, &pm(1000)), 8);
    // Same seed, same result at normal odds.
    let a = money_of(&p, &[], &mut jokers(&[J::BusinessCard]), &pm(1));
    let b = money_of(&p, &[], &mut jokers(&[J::BusinessCard]), &pm(1));
    assert_eq!(a, b);
    assert!(a == 0 || a == 2 || a == 4);
}

// ----- Egg -----

#[test]
fn egg_gains_three_sell_value_each_round() {
    let mut js = jokers(&[J::Egg]);
    // Base cost $4 -> sell value floor(4/2) = $2.
    assert_eq!(js[0].sell_value, 2);
    let run = RunView::default();
    let (money, fx) = round_end(&mut js, &run);
    assert_eq!(money, 0);
    assert!(fx.is_empty());
    assert_eq!(js[0].sell_value, 5); // 2 + 3
    round_end(&mut js, &run);
    assert_eq!(js[0].sell_value, 8); // 5 + 3
    // Debuffed: unchanged.
    let mut d = vec![jk(1, J::Egg).with_debuff(true)];
    round_end(&mut d, &run);
    assert_eq!(d[0].sell_value, 2);
    // Also on a lost round (the hook is unconditional; the run is over anyway).
    let mut l = jokers(&[J::Egg]);
    end_round(&mut l, &run, &Deck::standard(), false);
    assert_eq!(l[0].sell_value, 5);
}

// --- To Do List ---

#[test]
fn to_do_list_picks_target_on_acquire_and_changes_each_round() {
    let deck = Deck::standard();
    for seed in 0..50 {
        let mut js = jokers(&[J::ToDoList]);
        acquire(&mut js, 0, &deck, seed);
        let first = js[0].aux_hand().expect("target chosen on acquire");
        // Secret hands are not eligible before they are played.
        assert!(
            !matches!(
                first,
                HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive
            ),
            "seed {seed}: {first:?}"
        );
        // Re-picked at round end, never the same hand twice in a row.
        round_end(&mut js, &RunView::default());
        let second = js[0].aux_hand().expect("target");
        assert_ne!(first, second, "seed {seed}");
    }
}

#[test]
fn to_do_list_secret_hand_becomes_eligible_after_it_was_played() {
    let deck = Deck::standard();
    let mut run = RunView::default();
    run.hand_plays_run[HandType::FlushFive.index()] = 1;
    let mut seen = false;
    for seed in 0..300 {
        let mut js = jokers(&[J::ToDoList]);
        acquire(&mut js, 0, &deck, seed);
        round_end(&mut js, &run);
        seen |= js[0].aux_hand() == Some(HandType::FlushFive);
    }
    assert!(seen, "Flush Five never picked in 300 seeds");
    // Determinism: same seed, same pick.
    let pick = |seed| {
        let mut js = jokers(&[J::ToDoList]);
        acquire(&mut js, 0, &deck, seed);
        js[0].aux_hand()
    };
    assert_eq!(pick(7), pick(7));
}

#[test]
fn to_do_list_pays_four_on_matching_hand_only() {
    let (p, e) = pair_of(c(1, Seven, Spades), c(2, Seven, Hearts));
    let mut js = jokers(&[J::ToDoList]);
    js[0].set_aux_hand(Some(HandType::Pair));
    assert_eq!(
        score(&p, &[], e, &mut js, &RunView::default()).money_delta,
        4
    );
    // Target is a different hand: nothing.
    js[0].set_aux_hand(Some(HandType::Flush));
    assert_eq!(
        score(&p, &[], e, &mut js, &RunView::default()).money_delta,
        0
    );
    // No target chosen yet: nothing.
    js[0].set_aux_hand(None);
    assert_eq!(
        score(&p, &[], e, &mut js, &RunView::default()).money_delta,
        0
    );
    // Blueprint copies it with the target's state: $4 + $4.
    let mut bp = jokers(&[J::Blueprint, J::ToDoList]);
    bp[1].set_aux_hand(Some(HandType::Pair));
    assert_eq!(
        score(&p, &[], e, &mut bp, &RunView::default()).money_delta,
        8
    );
    // Debuffed: nothing.
    let mut d = vec![jk(1, J::ToDoList).with_debuff(true)];
    d[0].set_aux_hand(Some(HandType::Pair));
    assert_eq!(
        score(&p, &[], e, &mut d, &RunView::default()).money_delta,
        0
    );
}

// ----- Cloud 9 -----

#[test]
fn cloud9_pays_one_per_nine_in_full_deck() {
    let mut js = jokers(&[J::Cloud9]);
    let mut run = RunView::default();
    assert_eq!(round_end(&mut js, &run).0, 0);
    run.deck.by_rank[Nine.index()] = 4; // standard deck: four 9s -> $4
    assert_eq!(round_end(&mut js, &run).0, 4);
    run.deck.by_rank[Nine.index()] = 7; // 7 nines -> $7
    assert_eq!(round_end(&mut js, &run).0, 7);
    // Blueprint does not copy it.
    let mut bp = jokers(&[J::Blueprint, J::Cloud9]);
    assert_eq!(round_end(&mut bp, &run).0, 7);
    // Debuffed: nothing.
    let mut d = vec![jk(1, J::Cloud9).with_debuff(true)];
    assert_eq!(round_end(&mut d, &run).0, 0);
}

// ----- Rocket -----

#[test]
fn rocket_pays_and_grows_on_boss_defeat() {
    let mut js = jokers(&[J::Rocket]);
    assert_eq!(js[0].value, 1.0);
    let normal = RunView {
        blind: BlindKind::Big,
        ..RunView::default()
    };
    let boss = RunView {
        blind: BlindKind::Boss,
        ..RunView::default()
    };
    // Non-boss round: $1, unchanged.
    assert_eq!(round_end(&mut js, &normal).0, 1);
    assert_eq!(js[0].value, 1.0);
    // Boss defeated: 1 + 2 = 3 applied before the payout -> $3.
    assert_eq!(round_end(&mut js, &boss).0, 3);
    assert_eq!(js[0].value, 3.0);
    // Next normal round pays the new $3; next boss $5.
    assert_eq!(round_end(&mut js, &normal).0, 3);
    assert_eq!(round_end(&mut js, &boss).0, 5);
    assert_eq!(js[0].value, 5.0);
    // Boss round lost: no increase (payout unchanged).
    let mut l = jokers(&[J::Rocket]);
    let (_, _) = end_round(&mut l, &boss, &Deck::standard(), false);
    assert_eq!(l[0].value, 1.0);
    // Blueprint does not copy it.
    let mut bp = jokers(&[J::Blueprint, J::Rocket]);
    assert_eq!(round_end(&mut bp, &normal).0, 1);
}

// ----- Gift Card -----

#[test]
fn gift_card_adds_sell_value_to_jokers_and_consumables() {
    let mut js = jokers(&[J::GiftCard]);
    let (money, fx) = round_end(&mut js, &RunView::default());
    assert_eq!(money, 0);
    // Both effects are always queued (Phase 4 applies the consumable one; nothing held is a no-op).
    assert_eq!(
        fx,
        vec![
            SideEffect::AddSellValue {
                target: SellTarget::AllJokers,
                amount: 1
            },
            SideEffect::AddSellValue {
                target: SellTarget::AllConsumables,
                amount: 1
            },
        ]
    );
    let held = [ConsumableView::new(ConsumableId::planet_for(
        HandType::Pair,
    ))];
    let run = RunView {
        consumables: &held,
        ..RunView::default()
    };
    let (_, fx) = round_end(&mut js, &run);
    assert_eq!(
        fx,
        vec![
            SideEffect::AddSellValue {
                target: SellTarget::AllJokers,
                amount: 1
            },
            SideEffect::AddSellValue {
                target: SellTarget::AllConsumables,
                amount: 1
            },
        ]
    );
    // Debuffed: nothing queued.
    let mut d = vec![jk(1, J::GiftCard).with_debuff(true)];
    assert!(round_end(&mut d, &run).1.is_empty());
}

// ----- Reserved Parking -----

#[test]
fn reserved_parking_pays_one_per_held_face_with_half_chance() {
    let held = [
        c(10, King, Spades),
        c(11, Queen, Hearts),
        c(12, Five, Clubs),
    ];
    let (p, _) = pair_of(c(1, Seven, Spades), c(2, Seven, Hearts));
    let mut js = jokers(&[J::ReservedParking]);
    // Always: K and Q pay $1 each = $2; the 5 does not.
    assert_eq!(money_of(&p, &held, &mut js, &pm(1000)), 2);
    // Never: $0.
    assert_eq!(money_of(&p, &held, &mut js, &pm(0)), 0);
    // Oops! All 6s: certainty -> $2.
    assert_eq!(money_of(&p, &held, &mut js, &pm(2)), 2);
    // Debuffed held king is skipped: $1.
    let held_d = [debuff(c(10, King, Spades)), c(11, Queen, Hearts)];
    assert_eq!(money_of(&p, &held_d, &mut js, &pm(1000)), 1);
    // Pareidolia: all three held cards are faces -> $3.
    let mut run = pm(1000);
    run.rules.pareidolia = true;
    assert_eq!(money_of(&p, &held, &mut js, &run), 3);
    // Blueprint doubles: 2 faces x ($1 + $1) = $4.
    let mut bp = jokers(&[J::Blueprint, J::ReservedParking]);
    assert_eq!(money_of(&p, &held, &mut bp, &pm(1000)), 4);
    // Mime retriggers held abilities: each face twice -> $4.
    let mut mime = jokers(&[J::Mime, J::ReservedParking]);
    assert_eq!(money_of(&p, &held, &mut mime, &pm(1000)), 4);
}

// ----- Mail-In Rebate -----

#[test]
fn mail_in_rebate_target_comes_from_deck_cards() {
    // Deck of only 5s: the target must be Five.
    let deck = deck_of(vec![c(1, Five, Spades), c(2, Five, Hearts)]);
    for seed in 0..10 {
        let mut js = jokers(&[J::MailInRebate]);
        acquire(&mut js, 0, &deck, seed);
        assert_eq!(js[0].aux_rank(), Some(Five));
    }
    // Empty deck: defaults to Ace.
    let mut js = jokers(&[J::MailInRebate]);
    acquire(&mut js, 0, &deck_of(vec![]), 3);
    assert_eq!(js[0].aux_rank(), Some(Ace));
    // Only Stone cards: also Ace.
    let stones = deck_of(vec![enh(c(1, Five, Spades), Enhancement::Stone)]);
    let mut js = jokers(&[J::MailInRebate]);
    acquire(&mut js, 0, &stones, 3);
    assert_eq!(js[0].aux_rank(), Some(Ace));
    // Re-picked at round end from the current deck.
    let nines = deck_of(vec![c(1, Nine, Spades)]);
    let mut js = jokers(&[J::MailInRebate]);
    js[0].set_aux_rank(Some(Two));
    end_round(&mut js, &RunView::default(), &nines, true);
    assert_eq!(js[0].aux_rank(), Some(Nine));
    // Weighted by frequency: with 9 fives and 1 two, Five is picked far more often.
    let mut cards: Vec<Card> = (0..9).map(|i| c(i, Five, Spades)).collect();
    cards.push(c(20, Two, Hearts));
    let weighted = deck_of(cards);
    let fives = (0..200)
        .filter(|&seed| {
            let mut js = jokers(&[J::MailInRebate]);
            acquire(&mut js, 0, &weighted, seed);
            js[0].aux_rank() == Some(Five)
        })
        .count();
    assert!(fives > 150, "Five picked {fives}/200");
}

#[test]
fn mail_in_rebate_pays_five_per_discarded_target_rank() {
    let mut js = jokers(&[J::MailInRebate]);
    js[0].set_aux_rank(Some(Five));
    let run = RunView::default();
    let cards = [c(1, Five, Spades), c(2, Five, Hearts), c(3, Seven, Clubs)];
    // Two 5s: 2 x $5 = $10 (any discard, not only the first).
    let busy = RunView {
        discards_used_round: 3,
        ..RunView::default()
    };
    assert_eq!(discard_event(&mut js, &busy, &cards).0, 10);
    // No 5s: $0.
    assert_eq!(discard_event(&mut js, &run, &[c(3, Seven, Clubs)]).0, 0);
    // A Stone card is not a 5.
    let stone = [enh(c(4, Five, Spades), Enhancement::Stone)];
    assert_eq!(discard_event(&mut js, &run, &stone).0, 0);
    // Blueprint copy reads the target's rank and pays again: $10 + $10.
    let mut bp = jokers(&[J::Blueprint, J::MailInRebate]);
    bp[1].set_aux_rank(Some(Five));
    assert_eq!(discard_event(&mut bp, &run, &cards).0, 20);
    // Debuffed joker: nothing. No target: nothing.
    let mut d = vec![jk(1, J::MailInRebate).with_debuff(true)];
    d[0].set_aux_rank(Some(Five));
    assert_eq!(discard_event(&mut d, &run, &cards).0, 0);
    let mut n = jokers(&[J::MailInRebate]);
    assert_eq!(discard_event(&mut n, &run, &cards).0, 0);
}

// ----- Trading Card -----

#[test]
fn trading_card_destroys_single_first_discard_and_pays_three() {
    let mut js = jokers(&[J::TradingCard]);
    let first = RunView {
        discards_used_round: 1,
        ..RunView::default()
    };
    let (money, fx) = discard_event(&mut js, &first, &[c(42, Seven, Spades)]);
    assert_eq!(money, 3);
    assert_eq!(fx, vec![SideEffect::DestroyCard { uid: 42 }]);
    // Two cards: nothing.
    let two = [c(1, Seven, Spades), c(2, Eight, Spades)];
    let (money, fx) = discard_event(&mut js, &first, &two);
    assert_eq!((money, fx.len()), (0, 0));
    // Not the first discard of the round: nothing.
    let second = RunView {
        discards_used_round: 2,
        ..RunView::default()
    };
    let (money, fx) = discard_event(&mut js, &second, &[c(42, Seven, Spades)]);
    assert_eq!((money, fx.len()), (0, 0));
    // Blueprint does not copy it.
    let mut bp = jokers(&[J::Blueprint, J::TradingCard]);
    assert_eq!(discard_event(&mut bp, &first, &[c(42, Seven, Spades)]).0, 3);
}

// ----- Golden Ticket -----

#[test]
fn golden_ticket_pays_four_per_scored_gold_card() {
    let mut js = jokers(&[J::GoldenTicket]);
    let run = RunView::default();
    // One Gold + one plain: $4.
    let (p, _) = pair_of(
        enh(c(1, Seven, Spades), Enhancement::Gold),
        c(2, Seven, Hearts),
    );
    assert_eq!(money_of(&p, &[], &mut js, &run), 4);
    // Two Gold: $8.
    let (p2, _) = pair_of(
        enh(c(1, Seven, Spades), Enhancement::Gold),
        enh(c(2, Seven, Hearts), Enhancement::Gold),
    );
    assert_eq!(money_of(&p2, &[], &mut js, &run), 8);
    // No Gold: $0. A Gold card held in hand does not count.
    let (p3, _) = pair_of(c(1, Seven, Spades), c(2, Seven, Hearts));
    assert_eq!(money_of(&p3, &[], &mut js, &run), 0);
    let held = [enh(c(9, King, Clubs), Enhancement::Gold)];
    assert_eq!(money_of(&p3, &held, &mut js, &run), 0);
    // Debuffed Gold: skipped.
    let (pd, _) = pair_of(
        debuff(enh(c(1, Seven, Spades), Enhancement::Gold)),
        c(2, Seven, Hearts),
    );
    assert_eq!(money_of(&pd, &[], &mut js, &run), 0);
    // Red Seal on the Gold card: two triggers -> $8.
    let (pr, _) = pair_of(
        seal(enh(c(1, Seven, Spades), Enhancement::Gold), Seal::Red),
        c(2, Seven, Hearts),
    );
    assert_eq!(money_of(&pr, &[], &mut js, &run), 8);
    // Blueprint: 2 x $4 = $8.
    let mut bp = jokers(&[J::Blueprint, J::GoldenTicket]);
    assert_eq!(money_of(&p, &[], &mut bp, &run), 8);
}

// ----- Rough Gem -----

#[test]
fn rough_gem_pays_one_per_scored_diamond() {
    let mut js = jokers(&[J::RoughGem]);
    let run = RunView::default();
    // 7♦ 7♦: $2. 7♦ 7♠: $1. none: $0.
    let (pa, _) = pair_of(c(1, Seven, Diamonds), c(2, Seven, Diamonds));
    assert_eq!(money_of(&pa, &[], &mut js, &run), 2);
    let (pb, _) = pair_of(c(1, Seven, Diamonds), c(2, Seven, Spades));
    assert_eq!(money_of(&pb, &[], &mut js, &run), 1);
    let (pc, _) = pair_of(c(1, Seven, Hearts), c(2, Seven, Spades));
    assert_eq!(money_of(&pc, &[], &mut js, &run), 0);
    // Wild 7♠ counts as a diamond: $1 (the 7♥ does not).
    let (pw, _) = pair_of(
        enh(c(1, Seven, Spades), Enhancement::Wild),
        c(2, Seven, Hearts),
    );
    assert_eq!(money_of(&pw, &[], &mut js, &run), 1);
    // Smeared: 7♥ counts as a diamond -> $1.
    let mut smeared = RunView::default();
    smeared.rules.smeared = true;
    assert_eq!(money_of(&pc, &[], &mut js, &smeared), 1);
    // Debuffed diamond: skipped.
    let (pd, _) = pair_of(debuff(c(1, Seven, Diamonds)), c(2, Seven, Spades));
    assert_eq!(money_of(&pd, &[], &mut js, &run), 0);
    // Red Seal diamond: two triggers -> $2.
    let (pr, _) = pair_of(seal(c(1, Seven, Diamonds), Seal::Red), c(2, Seven, Spades));
    assert_eq!(money_of(&pr, &[], &mut js, &run), 2);
    // Blueprint: 2 diamonds x ($1 + $1) = $4.
    let mut bp = jokers(&[J::Blueprint, J::RoughGem]);
    assert_eq!(money_of(&pa, &[], &mut bp, &run), 4);
}

// ----- Matador -----

fn boss_run(triggered: bool, invalid: bool) -> RunView<'static> {
    RunView {
        blind: BlindKind::Boss,
        boss: BossView {
            id: Some(BossId::ThePsychic),
            active: true,
            halve_base: false,
            triggered,
            hand_invalid: invalid,
        },
        ..RunView::default()
    }
}

#[test]
fn matador_pays_eight_when_the_boss_is_triggered() {
    let (p, e) = pair_of(c(1, Seven, Spades), c(2, Seven, Hearts));
    let mut js = jokers(&[J::Matador]);
    // Triggered by a valid hand (e.g. a scoring debuffed card): $8.
    let r = score(&p, &[], e, &mut js, &boss_run(true, false));
    assert_eq!(r.money_delta, 8);
    assert!(!r.invalid);
    // Not triggered: $0. No boss at all: $0.
    assert_eq!(
        score(&p, &[], e, &mut js, &boss_run(false, false)).money_delta,
        0
    );
    assert_eq!(
        score(&p, &[], e, &mut js, &RunView::default()).money_delta,
        0
    );
    // Rejected hand (The Eye/Mouth/Psychic): still $8, paid once (on_invalid_hand only).
    let r = score(&p, &[], e, &mut js, &boss_run(true, true));
    assert!(r.invalid);
    assert_eq!(r.money_delta, 8);
    // Blueprint: two payers -> $16 (valid), $16 (invalid).
    let mut bp = jokers(&[J::Blueprint, J::Matador]);
    assert_eq!(
        score(&p, &[], e, &mut bp, &boss_run(true, false)).money_delta,
        16
    );
    assert_eq!(
        score(&p, &[], e, &mut bp, &boss_run(true, true)).money_delta,
        16
    );
    // Debuffed: nothing.
    let mut d = vec![jk(1, J::Matador).with_debuff(true)];
    assert_eq!(
        score(&p, &[], e, &mut d, &boss_run(true, false)).money_delta,
        0
    );
}

// ----- Satellite -----

#[test]
fn satellite_pays_one_per_unique_planet_used() {
    let mut js = jokers(&[J::Satellite]);
    let mut run = RunView::default();
    assert_eq!(round_end(&mut js, &run).0, 0);
    run.stats.unique_planets_used = 3;
    run.stats.planets_used = 9; // repeated uses do not count: only unique ones pay
    assert_eq!(round_end(&mut js, &run).0, 3);
    run.stats.unique_planets_used = 12;
    assert_eq!(round_end(&mut js, &run).0, 12);
    let mut bp = jokers(&[J::Blueprint, J::Satellite]);
    assert_eq!(round_end(&mut bp, &run).0, 12);
    let mut d = vec![jk(1, J::Satellite).with_debuff(true)];
    assert_eq!(round_end(&mut d, &run).0, 0);
}

// ----- Game level: real step flow, cash-out, money -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn select(g: &mut Game) {
    g.step(Action::SelectBlind).expect("select blind");
}

/// Beat the current blind with one hand (hands_left becomes 3 -> $3 hands money).
fn beat(g: &mut Game) -> StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("joker room")
}

#[test]
fn game_cloud9_payout_is_in_cash_out_but_not_interest_base() {
    let mut g = new_game(1);
    add(&mut g, J::Cloud9);
    g.state_mut().money = 25;
    select(&mut g);
    let r = beat(&mut g).round.expect("round");
    let co = r.cash_out.expect("cash out");
    // Interest on $25 = 5 (cap 5); the Cloud 9 payout ($4: four 9s in the 52-card deck) is not
    // part of the base. Blind $3 + hands $3 (3 left) + jokers $4.
    assert_eq!((co.interest_base, co.interest), (25, 5));
    assert_eq!((co.blind_reward, co.hands, co.jokers), (3, 3, 4));
    assert_eq!(co.total, 5 + 3 + 3 + 4);
    // 25 + 15 = 40.
    assert_eq!(g.state().money, 40);
}

#[test]
fn game_delayed_gratification_only_without_discards() {
    // No discards used: 4 left x $2 = $8. 4 + 0 interest + 3 + 3 + 8 = 18.
    let mut g = new_game(2);
    add(&mut g, J::DelayedGratification);
    select(&mut g);
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    assert_eq!(co.jokers, 8);
    assert_eq!(g.state().money, 18);

    // One discard used first: nothing. 4 + 3 + 3 = 10.
    let mut g = new_game(2);
    add(&mut g, J::DelayedGratification);
    select(&mut g);
    g.step(Action::Discard(0b1)).expect("discard");
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    assert_eq!(co.jokers, 0);
    assert_eq!(g.state().money, 10);
}

#[test]
fn game_to_the_moon_doubles_interest() {
    let mut g = new_game(3);
    add(&mut g, J::ToTheMoon);
    g.state_mut().money = 25;
    select(&mut g);
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    // Interest: min(25/5, 5) = 5, x (1 + 1) = 10. 25 + 10 + 3 + 3 = 41.
    assert_eq!(co.interest, 10);
    assert_eq!(g.state().money, 41);
    // Money over the cap earns nothing extra: $100 -> min(20, 5) x 2 = 10.
    let mut g = new_game(3);
    add(&mut g, J::ToTheMoon);
    g.state_mut().money = 100;
    select(&mut g);
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    assert_eq!(co.interest, 10);
}

#[test]
fn game_rocket_grows_after_each_boss() {
    let mut g = new_game(4);
    add(&mut g, J::Rocket);
    // Small: $1. 4 + 0 + 3 + 3 + 1 = 11.
    select(&mut g);
    beat(&mut g);
    assert_eq!(g.state().money, 11);
    g.step(Action::EndShop).expect("end shop");
    // Big: interest floor(11/5) = 2, blind 4, hands 3, rocket 1. 11 + 10 = 21.
    select(&mut g);
    beat(&mut g);
    assert_eq!(g.state().money, 21);
    g.step(Action::EndShop).expect("end shop");
    // Boss: interest 4, blind 5, hands 3, rocket 1 + 2 = 3 (increase before payout).
    // 21 + 4 + 5 + 3 + 3 = 36.
    select(&mut g);
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    assert_eq!(
        (co.interest, co.blind_reward, co.hands, co.jokers),
        (4, 5, 3, 3)
    );
    assert_eq!(g.state().money, 36);
    assert_eq!(g.state().jokers[0].value, 3.0);
    g.step(Action::EndShop).expect("end shop");
    // Ante 2 Small: interest min(7, cap 5) = 5, blind 3, hands 3, rocket 3. 36 + 14 = 50.
    select(&mut g);
    beat(&mut g);
    assert_eq!(g.state().money, 50);
}

#[test]
fn game_egg_and_gift_card_raise_sell_values() {
    let mut g = new_game(5);
    add(&mut g, J::Egg); // cost 4 -> sell $2
    add(&mut g, J::GiftCard); // cost 6 -> sell $3
    add(&mut g, J::Joker); // cost 2 -> sell $1
    let sells = |g: &Game| -> Vec<i32> { g.state().jokers.iter().map(|j| j.sell_value).collect() };
    assert_eq!(sells(&g), vec![2, 3, 1]);
    select(&mut g);
    beat(&mut g);
    // Egg: 2 + 3 (own) + 1 (Gift Card) = 6; Gift Card 3 + 1 = 4; Joker 1 + 1 = 2.
    assert_eq!(sells(&g), vec![6, 4, 2]);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    beat(&mut g);
    // Second round: Egg 6 + 3 + 1 = 10; Gift Card 5; Joker 3.
    assert_eq!(sells(&g), vec![10, 5, 3]);
    // Nothing was dropped for lack of consumables (no consumable effect queued).
    assert_eq!(g.state().stats.effects_dropped, 0);
}

#[test]
fn game_trading_card_destroys_first_single_discard() {
    let mut g = new_game(6);
    add(&mut g, J::TradingCard);
    select(&mut g);
    let uid = g.state().deck.hand[0].uid;
    let info = g.step(Action::Discard(0b1)).expect("discard");
    // +$3, the card leaves the deck entirely.
    assert_eq!((info.money_before, info.money_after), (4, 7));
    assert_eq!(info.cards_destroyed, 1);
    assert_eq!(g.state().deck.len(), 51);
    assert!(g.state().deck.iter_all().all(|c| c.uid != uid));
    // Second discard of a single card: nothing.
    let info = g.step(Action::Discard(0b1)).expect("discard");
    assert_eq!((info.money_before, info.money_after), (7, 7));
    assert_eq!(g.state().deck.len(), 51);
}

#[test]
fn game_mail_in_rebate_pays_for_discarded_target_rank() {
    let mut g = new_game(7);
    select(&mut g);
    add(&mut g, J::MailInRebate);
    let target = g.state().jokers[0].aux_rank().expect("target on acquire");
    let hits = g
        .state()
        .deck
        .hand
        .iter()
        .take(5)
        .filter(|c| c.rank == target)
        .count() as i32;
    // Discard the first 5 cards: $5 per card of the target rank.
    let info = g.step(Action::Discard(0b1_1111)).expect("discard");
    assert_eq!(info.money_after - info.money_before, 5 * hits);
}

#[test]
fn game_mail_in_rebate_retargets_at_round_end() {
    let mut g = new_game(8);
    add(&mut g, J::MailInRebate);
    assert!(g.state().jokers[0].aux_rank().is_some());
    select(&mut g);
    beat(&mut g);
    // Still a valid rank, deterministic with the seed.
    let a = g.state().jokers[0].aux_rank();
    assert!(a.is_some());
    let mut g2 = new_game(8);
    add(&mut g2, J::MailInRebate);
    select(&mut g2);
    beat(&mut g2);
    assert_eq!(g2.state().jokers[0].aux_rank(), a);
}

#[test]
fn game_credit_card_lowers_the_money_floor() {
    let mut g = new_game(9);
    assert_eq!(g.state().money_floor(), 0);
    assert!(!g.state().can_afford(5)); // $4 - 5 < 0
    add(&mut g, J::CreditCard);
    let s = g.state();
    assert_eq!(s.money_floor(), -20);
    assert!(s.can_afford(24)); // 4 - 24 = -20
    assert!(!s.can_afford(25)); // -21
}

#[test]
fn game_modifiers_reach_the_state_cache() {
    let mut g = new_game(10);
    add(&mut g, J::ChaosTheClown);
    add(&mut g, J::Astronomer);
    add(&mut g, J::ToTheMoon);
    let m = g.state().mods;
    assert_eq!(m.free_rerolls, 1);
    assert!(m.free_planets);
    assert_eq!(m.interest_per_5, 1);
}

#[test]
fn game_payouts_stack_across_jokers() {
    // Cloud 9 ($4) + Satellite (0 planets) + Delayed Gratification ($8): jokers item = $12.
    let mut g = new_game(11);
    add(&mut g, J::Cloud9);
    add(&mut g, J::Satellite);
    add(&mut g, J::DelayedGratification);
    select(&mut g);
    let co = beat(&mut g).round.expect("round").cash_out.expect("co");
    assert_eq!(co.jokers, 12);
    // 4 + 0 interest + 3 + 3 + 12 = 22.
    assert_eq!(g.state().money, 22);
}

// Phase 4.R decision: Trading Card's wiki page ("first discard of round has only 1 card")
// mentions no debuff exception, so a debuffed single card still triggers it.
#[test]
fn trading_card_still_fires_for_a_debuffed_single_card() {
    let mut js = jokers(&[J::TradingCard]);
    let first = RunView {
        discards_used_round: 1,
        ..RunView::default()
    };
    let mut k = c(7, Seven, Spades);
    k.debuffed = true;
    let (money, fx) = discard_event(&mut js, &first, &[k]);
    assert_eq!(money, 3);
    assert_eq!(fx, vec![SideEffect::DestroyCard { uid: 7 }]);
}
