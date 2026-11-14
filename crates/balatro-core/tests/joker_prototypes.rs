//! Prototype jokers (content/jokers/prototypes.rs): trigger / non-trigger / scaling for each,
//! ordering and copier combos, events, the registry check and a property test.
//! Every expected value is hand-computed in a comment.

use balatro_core::card::Card;
use balatro_core::content::jokers::{JokerId, def_count, try_def};
use balatro_core::deck::Deck;
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{
    BlindKind, DiscardInfo, EventInput, RoundEndInfo, RunView, SideEffect, collect_modifiers,
    fire_discard, fire_round_end, resolve_source,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer, TraceEvent, TraceOp, TraceSource};
use proptest::prelude::*;

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

fn ev(hand_type: HandType, scoring_mask: u8, contains: &[HandType]) -> HandEval {
    let mut bits = 1u16 << hand_type.index();
    for h in contains {
        bits |= 1 << h.index();
    }
    HandEval {
        hand_type,
        scoring_mask,
        contains: bits,
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

/// 7♠ 7♥ Pair (contains Pair): base 10 x 2 + 7 + 7 -> 24 x 2 = 48 with no jokers.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

/// 2♠ 2♥ Pair: 10 + 2 + 2 = 14 x 2 = 28 with no jokers.
fn pair2() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Two, Spades), c(2, Two, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
    trace: Vec<TraceEvent>,
    played: Vec<Card>,
}

fn score_with(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    js: &mut [JokerInstance],
    run: &RunView<'_>,
) -> Out {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(11);
    let mut scorer = Scorer::with_trace(true);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    });
    Out {
        res,
        effects: scorer.effects().to_vec(),
        trace: scorer.trace().to_vec(),
        played,
    }
}

fn score(played: &[Card], held: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> Out {
    score_with(played, held, eval, js, &RunView::default())
}

fn run_pm(prob_mult: u32) -> RunView<'static> {
    RunView {
        prob_mult,
        ..RunView::default()
    }
}

// ----- registry -----

#[test]
fn registry_no_id_defined_twice() {
    for id in JokerId::ALL {
        assert!(def_count(id) <= 1, "{id:?} has more than one JokerDef");
        if let Some(d) = try_def(id) {
            assert_eq!(d.id, id, "registry slot of {id:?} holds the wrong def");
        }
        assert_eq!(id.index(), id as usize);
    }
}

/// Phase 3 acceptance: every id has exactly one def. Un-ignore when all waves are merged.
#[test]
#[ignore = "enable at the end of Phase 3"]
fn registry_every_id_has_exactly_one_def() {
    let missing: Vec<JokerId> = JokerId::ALL
        .into_iter()
        .filter(|&id| def_count(id) != 1)
        .collect();
    assert!(missing.is_empty(), "jokers without exactly one def: {missing:?}");
}

// ----- simple jokers -----

#[test]
fn joker_adds_four_mult() {
    // 24 x (2 + 4) = 144.
    let (p, e) = pair7();
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Joker])).res.score, 144.0);
}

#[test]
fn greedy_joker_per_scored_diamond() {
    // 7♦ 7♠: one diamond -> 24 x 5 = 120.
    let e = ev(HandType::Pair, 0b11, &[]);
    let p = [c(1, Seven, Diamonds), c(2, Seven, Spades)];
    assert_eq!(
        score(&p, &[], e, &mut jokers(&[J::GreedyJoker])).res.score,
        120.0
    );
    // No diamond -> 48.
    let (p2, _) = pair7();
    assert_eq!(
        score(&p2, &[], e, &mut jokers(&[J::GreedyJoker])).res.score,
        48.0
    );
    // Red Seal diamond triggers twice: chips 10 + 7 + 7 + 7 = 31, mult 2 + 3 + 3 = 8 -> 248.
    let p3 = [seal(c(1, Seven, Diamonds), Seal::Red), c(2, Seven, Spades)];
    assert_eq!(
        score(&p3, &[], e, &mut jokers(&[J::GreedyJoker])).res.score,
        248.0
    );
    // Wild 7♠ counts as a diamond -> 120.
    let p4 = [
        enh(c(1, Seven, Spades), Enhancement::Wild),
        c(2, Seven, Hearts),
    ];
    assert_eq!(
        score(&p4, &[], e, &mut jokers(&[J::GreedyJoker])).res.score,
        120.0
    );
    // Debuffed diamond: skipped -> chips 17 x 2 = 34.
    let p5 = [debuff(c(1, Seven, Diamonds)), c(2, Seven, Spades)];
    assert_eq!(
        score(&p5, &[], e, &mut jokers(&[J::GreedyJoker])).res.score,
        34.0
    );
    // Smeared: 7♥ counts as a diamond -> 120.
    let mut run = RunView::default();
    run.rules.smeared = true;
    let o = score_with(&p2, &[], e, &mut jokers(&[J::GreedyJoker]), &run);
    assert_eq!(o.res.score, 120.0);
}

#[test]
fn jolly_joker_uses_contains() {
    // Pair -> 24 x 10 = 240.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, &[], e, &mut jokers(&[J::JollyJoker])).res.score,
        240.0
    );
    // Full House contains Pair: 40 + 7+7+7+2+2 = 65; mult 4 + 8 = 12 -> 780.
    let fh = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Seven, Diamonds),
        c(4, Two, Clubs),
        c(5, Two, Diamonds),
    ];
    let e = ev(
        HandType::FullHouse,
        0x1F,
        &[HandType::Pair, HandType::TwoPair, HandType::ThreeOfAKind],
    );
    assert_eq!(
        score(&fh, &[], e, &mut jokers(&[J::JollyJoker])).res.score,
        780.0
    );
    // High Card: no Pair -> 16.
    let e = ev(HandType::HighCard, 1, &[]);
    let p = [c(1, Ace, Spades)];
    assert_eq!(
        score(&p, &[], e, &mut jokers(&[J::JollyJoker])).res.score,
        16.0
    );
}

#[test]
fn cavendish_x3_and_round_end_destroy_roll() {
    // 24 x (2 x 3) = 144.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, &[], e, &mut jokers(&[J::Cavendish])).res.score,
        144.0
    );
    // End of round: forced roll destroys it, prob 0 keeps it.
    for (pm, destroyed) in [(1000, true), (0, false)] {
        let mut js = jokers(&[J::Cavendish]);
        let run = run_pm(pm);
        let deck = Deck::standard();
        let mut rng = GameRng::new(5);
        let mut fx = Vec::new();
        let info = RoundEndInfo {
            won: true,
            round_score: 400.0,
            target: 300.0,
        };
        fire_round_end(
            EventInput {
                jokers: &mut js,
                run: &run,
                deck: &deck,
                rng: &mut rng,
                effects: &mut fx,
            },
            &info,
        );
        assert_eq!(
            fx.contains(&SideEffect::DestroyJoker { uid: 100 }),
            destroyed
        );
    }
}

#[test]
fn hack_retriggers_low_ranks() {
    // Pair of 2s, both twice: 10 + 2 x 4 = 18 x 2 = 36.
    let (p, e) = pair2();
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Hack])).res.score, 36.0);
    // Sevens untouched: 48.
    let (p, e) = pair7();
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Hack])).res.score, 48.0);
    // Red Seal + Hack on 2♠: 3 triggers. chips 10 + 2*3 + 2*2 = 20 -> 40.
    let (mut p, e) = pair2();
    p[0] = seal(p[0], Seal::Red);
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Hack])).res.score, 40.0);
    // Stone card (printed 2) is not retriggered: High Card 5 + 11 + 50 = 66.
    let p = [c(1, Ace, Spades), enh(c(2, Two, Clubs), Enhancement::Stone)];
    let e = ev(HandType::HighCard, 0b11, &[]);
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Hack])).res.score, 66.0);
    // Gold Seal pays on every trigger: Gold Seal 2♠ + Hack = 2 triggers = $6 (score 36).
    let (mut p, e) = pair2();
    p[0] = seal(p[0], Seal::Gold);
    let o = score(&p, &[], e, &mut jokers(&[J::Hack]));
    assert_eq!((o.res.score, o.res.money_delta), (36.0, 6));
    // Debuffed 2♠ is skipped entirely (no chips, no retrigger); 2♥ scores twice:
    // chips 10 + 0 + 2 x 2 = 14 -> 28.
    let (mut p, e) = pair2();
    p[0] = debuff(p[0]);
    assert_eq!(score(&p, &[], e, &mut jokers(&[J::Hack])).res.score, 28.0);
}

#[test]
fn mime_retriggers_held_steel() {
    // Held Steel x1.5 twice: 2 x 2.25 = 4.5 -> 108.
    let (p, e) = pair7();
    let held = [enh(c(9, Three, Clubs), Enhancement::Steel)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::Mime])).res.score,
        108.0
    );
    // Nothing held with an effect -> 48.
    let held = [c(9, Three, Clubs)];
    assert_eq!(score(&p, &held, e, &mut jokers(&[J::Mime])).res.score, 48.0);
}

#[test]
fn baron_per_held_king() {
    // Two held Kings: 2 x 1.5 x 1.5 = 4.5 -> 108.
    let (p, e) = pair7();
    let held = [c(9, King, Spades), c(10, King, Hearts), c(11, Three, Clubs)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::Baron])).res.score,
        108.0
    );
    // One King debuffed: 2 x 1.5 = 3 -> 72.
    let held = [debuff(c(9, King, Spades)), c(10, King, Hearts)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::Baron])).res.score,
        72.0
    );
    // No King -> 48.
    let held = [c(9, Queen, Spades)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::Baron])).res.score,
        48.0
    );
}

#[test]
fn raised_fist_lowest_rightmost() {
    let (p, e) = pair7();
    // Held Q 4 9 4: lowest 4, rightmost (index 3) -> +8 -> 24 x 10 = 240.
    let held = [
        c(9, Queen, Spades),
        c(10, Four, Hearts),
        c(11, Nine, Clubs),
        c(12, Four, Diamonds),
    ];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::RaisedFist])).res.score,
        240.0
    );
    // Red Seal on the rightmost 4: triggers twice -> +16 -> 24 x 18 = 432.
    let mut h2 = held;
    h2[3] = seal(h2[3], Seal::Red);
    assert_eq!(
        score(&p, &h2, e, &mut jokers(&[J::RaisedFist])).res.score,
        432.0
    );
    // Red Seal on the other 4 does not matter -> 240.
    let mut h3 = held;
    h3[1] = seal(h3[1], Seal::Red);
    assert_eq!(
        score(&p, &h3, e, &mut jokers(&[J::RaisedFist])).res.score,
        240.0
    );
    // Lowest card debuffed: +0 -> 48.
    let mut h4 = held;
    h4[3] = debuff(h4[3]);
    assert_eq!(
        score(&p, &h4, e, &mut jokers(&[J::RaisedFist])).res.score,
        48.0
    );
    // Faces: lowest is J -> +20 -> 24 x 22 = 528. Ace alone -> +22 -> 24 x 24 = 576.
    let held = [c(9, King, Spades), c(10, Jack, Hearts), c(11, Queen, Clubs)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::RaisedFist])).res.score,
        528.0
    );
    let held = [c(9, Ace, Spades)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::RaisedFist])).res.score,
        576.0
    );
    // Stone cards never count -> 48.
    let held = [enh(c(9, Two, Spades), Enhancement::Stone)];
    assert_eq!(
        score(&p, &held, e, &mut jokers(&[J::RaisedFist])).res.score,
        48.0
    );
}

// ----- scaling / stateful -----

#[test]
fn ride_the_bus_scales_and_resets() {
    let mut js = jokers(&[J::RideTheBus]);
    let (p, e) = pair7();
    // Hand 1: no face -> value 1 -> 24 x 3 = 72. Hand 2 -> value 2 -> 96.
    assert_eq!(score(&p, &[], e, &mut js).res.score, 72.0);
    assert_eq!(score(&p, &[], e, &mut js).res.score, 96.0);
    assert_eq!(js[0].value, 2.0);
    // Non-scoring face kicker does not reset: value 3 -> 24 x 5 = 120.
    let kick = [c(1, Seven, Spades), c(2, Seven, Hearts), c(3, King, Clubs)];
    let ek = ev(HandType::Pair, 0b011, &[]);
    assert_eq!(score(&kick, &[], ek, &mut js).res.score, 120.0);
    // Debuffed scoring faces do not reset: value 4. chips 10 x (2 + 4) = 60.
    let deb = [debuff(c(1, King, Spades)), debuff(c(2, King, Hearts))];
    assert_eq!(score(&deb, &[], e, &mut js).res.score, 60.0);
    assert_eq!(js[0].value, 4.0);
    // A scoring face resets before the joker phase: K K -> 30 x 2 = 60, value 0.
    let kk = [c(1, King, Spades), c(2, King, Hearts)];
    assert_eq!(score(&kk, &[], e, &mut js).res.score, 60.0);
    assert_eq!(js[0].value, 0.0);
    // Pareidolia: every card is a face -> always resets.
    js[0].value = 5.0;
    let mut run = RunView::default();
    run.rules.pareidolia = true;
    assert_eq!(score_with(&p, &[], e, &mut js, &run).res.score, 48.0);
    assert_eq!(js[0].value, 0.0);
}

#[test]
fn hiker_perma_bonus_seen_only_by_later_triggers() {
    // First trigger does not see its own +5: 48. Both cards end with perma 5.
    let (p, e) = pair7();
    let o = score(&p, &[], e, &mut jokers(&[J::Hiker]));
    assert_eq!(o.res.score, 48.0);
    assert_eq!((o.played[0].perma_bonus, o.played[1].perma_bonus), (5, 5));
    // Red Seal: 2nd trigger sees +5: chips 10 + 7 + (7 + 5) + 7 = 36 -> 72; perma 10 and 5.
    let mut p2 = p.clone();
    p2[0] = seal(p2[0], Seal::Red);
    let o = score(&p2, &[], e, &mut jokers(&[J::Hiker]));
    assert_eq!(o.res.score, 72.0);
    assert_eq!((o.played[0].perma_bonus, o.played[1].perma_bonus), (10, 5));
    // Blueprint copies Hiker (card mutation, not joker state): +10 per card.
    let o = score(&p, &[], e, &mut jokers(&[J::Blueprint, J::Hiker]));
    assert_eq!((o.played[0].perma_bonus, o.played[1].perma_bonus), (10, 10));
    // Debuffed card gets nothing.
    let mut p3 = p.clone();
    p3[0] = debuff(p3[0]);
    let o = score(&p3, &[], e, &mut jokers(&[J::Hiker]));
    assert_eq!((o.played[0].perma_bonus, o.played[1].perma_bonus), (0, 5));
}

#[test]
fn space_joker_levels_current_hand() {
    // Forced: Pair L1 -> L2 for this hand: 25 + 14 = 39 x 3 = 117; level-up queued.
    let (p, e) = pair7();
    let o = score_with(&p, &[], e, &mut jokers(&[J::SpaceJoker]), &run_pm(1000));
    assert_eq!((o.res.hand_level, o.res.score), (2, 117.0));
    assert_eq!(
        o.effects,
        vec![SideEffect::LevelUpHand {
            hand: HandType::Pair,
            levels: 1
        }]
    );
    // Never: 48, nothing queued.
    let o = score_with(&p, &[], e, &mut jokers(&[J::SpaceJoker]), &run_pm(0));
    assert_eq!((o.res.hand_level, o.res.score), (1, 48.0));
    assert!(o.effects.is_empty());
}

#[test]
fn ice_cream_decays_and_melts() {
    // +100 chips: 124 x 2 = 248; then 95 left.
    let (p, e) = pair7();
    let mut js = jokers(&[J::IceCream]);
    assert_eq!(score(&p, &[], e, &mut js).res.score, 248.0);
    assert_eq!(js[0].value, 95.0);
    // At 5: gives +5 (29 x 2 = 58), then melts.
    js[0].value = 5.0;
    let o = score(&p, &[], e, &mut js);
    assert_eq!(o.res.score, 58.0);
    assert_eq!(o.effects, vec![SideEffect::DestroyJoker { uid: 100 }]);
    // Blueprint copy: +100 twice (224 x 2 = 448) but decays once, never destroys the target.
    let mut js = jokers(&[J::Blueprint, J::IceCream]);
    let o = score(&p, &[], e, &mut js);
    assert_eq!(o.res.score, 448.0);
    assert_eq!((js[0].value, js[1].value), (0.0, 95.0));
    assert!(o.effects.is_empty());
    // A boss-invalid hand still decays it (balatrowiki.org/w/Ice_Cream).
    let mut js = jokers(&[J::IceCream]);
    let mut run = RunView::default();
    run.boss.hand_invalid = true;
    let o = score_with(&p, &[], e, &mut js, &run);
    assert_eq!((o.res.score, js[0].value), (0.0, 95.0));
}

#[test]
fn supernova_counts_this_run() {
    // Pair played 3 times this run (incl. this one): 24 x 5 = 120. None: 48.
    let (p, e) = pair7();
    let mut run = RunView::default();
    run.hand_plays_run[HandType::Pair.index()] = 3;
    let o = score_with(&p, &[], e, &mut jokers(&[J::Supernova]), &run);
    assert_eq!(o.res.score, 120.0);
    let o = score(&p, &[], e, &mut jokers(&[J::Supernova]));
    assert_eq!(o.res.score, 48.0);
}

#[test]
fn bull_sees_money_earned_this_hand() {
    // $10: +20 chips -> 44 x 2 = 88.
    let (p, e) = pair7();
    let run = RunView {
        money: 10,
        ..RunView::default()
    };
    let o = score_with(&p, &[], e, &mut jokers(&[J::Bull]), &run);
    assert_eq!(o.res.score, 88.0);
    // Gold Seal pays $3 during the card phase -> Bull sees $13: +26 -> 50 x 2 = 100.
    let mut p2 = p.clone();
    p2[0] = seal(p2[0], Seal::Gold);
    let o = score_with(&p2, &[], e, &mut jokers(&[J::Bull]), &run);
    assert_eq!((o.res.score, o.res.money_delta), (100.0, 3));
    // In debt: nothing.
    let run = RunView {
        money: -5,
        ..RunView::default()
    };
    let o = score_with(&p, &[], e, &mut jokers(&[J::Bull]), &run);
    assert_eq!(o.res.score, 48.0);
}

// ----- rule providers -----

#[test]
fn four_fingers_and_oops_modifiers() {
    let m = collect_modifiers(&jokers(&[J::Joker, J::FourFingers]));
    assert!(m.rules.four_fingers);
    assert!(!m.rules.shortcut);
    assert_eq!(m.prob_mult, 1);
    let js = vec![jk(1, J::FourFingers).with_debuff(true)];
    assert!(!collect_modifiers(&js).rules.four_fingers);
    // Two Oops: x2 x2 = 4. A debuffed one adds nothing. Blueprint does not copy passives.
    let m = collect_modifiers(&jokers(&[J::OopsAll6s, J::OopsAll6s]));
    assert_eq!(m.prob_mult, 4);
    let js = vec![jk(1, J::Blueprint), jk(2, J::OopsAll6s).with_debuff(true)];
    assert_eq!(collect_modifiers(&js).prob_mult, 1);
    let m = collect_modifiers(&jokers(&[J::Blueprint, J::OopsAll6s]));
    assert_eq!(m.prob_mult, 2);
}

#[test]
fn oops_prob_mult_reaches_rolls() {
    // Space Joker 1 in 4 with two Oops (prob_mult 4): 4/4 = certain level-up -> 117.
    let js = jokers(&[J::OopsAll6s, J::OopsAll6s, J::SpaceJoker]);
    let run = run_pm(collect_modifiers(&js).prob_mult);
    let (p, e) = pair7();
    let mut js = js;
    for _ in 0..20 {
        assert_eq!(score_with(&p, &[], e, &mut js, &run).res.score, 117.0);
    }
}

// ----- ordering and copiers -----

#[test]
fn plus_mult_before_x_mult_matters() {
    // [Joker, Cavendish]: (2 + 4) x 3 = 18 -> 432. [Cavendish, Joker]: 2 x 3 + 4 = 10 -> 240.
    let (p, e) = pair7();
    let o = score(&p, &[], e, &mut jokers(&[J::Joker, J::Cavendish]));
    assert_eq!(o.res.score, 432.0);
    let o = score(&p, &[], e, &mut jokers(&[J::Cavendish, J::Joker]));
    assert_eq!(o.res.score, 240.0);
}

#[test]
fn blueprint_copies_x_mult() {
    // [Blueprint, Cavendish]: 2 x 3 x 3 = 18 -> 432.
    let (p, e) = pair7();
    let o = score(&p, &[], e, &mut jokers(&[J::Blueprint, J::Cavendish]));
    assert_eq!(o.res.score, 432.0);
    // The copy is traced at Blueprint's slot with Cavendish as source.
    assert!(
        o.trace
            .iter()
            .any(|t| t.source == TraceSource::Joker { slot: 0, src: 1 }
                && t.op == TraceOp::XMult(3.0))
    );
}

#[test]
fn blueprint_copies_hack_retriggers() {
    // [Blueprint, Hack], 2♠ 2♥: each card 3 triggers. chips 10 + 2 x 6 = 22 -> 44.
    let (p, e) = pair2();
    let o = score(&p, &[], e, &mut jokers(&[J::Blueprint, J::Hack]));
    assert_eq!(o.res.score, 44.0);
}

#[test]
fn brainstorm_copies_leftmost() {
    // [Joker, Brainstorm]: 2 + 4 + 4 = 10 -> 240.
    let (p, e) = pair7();
    let o = score(&p, &[], e, &mut jokers(&[J::Joker, J::Brainstorm]));
    assert_eq!(o.res.score, 240.0);
}

#[test]
fn copier_cycles_and_dead_ends_are_inert() {
    let (p, e) = pair7();
    // Blueprint -> Brainstorm -> Blueprint: cycle, both inert -> 48.
    let js = jokers(&[J::Blueprint, J::Brainstorm]);
    assert_eq!(resolve_source(&js, 0), None);
    assert_eq!(resolve_source(&js, 1), None);
    assert_eq!(score(&p, &[], e, &mut js.clone()).res.score, 48.0);
    // Brainstorm alone copies itself; Blueprint alone has no right neighbour.
    assert_eq!(resolve_source(&jokers(&[J::Brainstorm]), 0), None);
    assert_eq!(resolve_source(&jokers(&[J::Blueprint]), 0), None);
    // Chain Blueprint -> Blueprint -> Joker: +4 three times -> 14 -> 336.
    let js = jokers(&[J::Blueprint, J::Blueprint, J::Joker]);
    assert_eq!(resolve_source(&js, 0), Some(2));
    assert_eq!(score(&p, &[], e, &mut js.clone()).res.score, 336.0);
    // Incompatible target (Four Fingers) and debuffed target give nothing.
    let js = jokers(&[J::Blueprint, J::FourFingers]);
    assert_eq!(score(&p, &[], e, &mut js.clone()).res.score, 48.0);
    let js = vec![jk(1, J::Blueprint), jk(2, J::Joker).with_debuff(true)];
    assert_eq!(score(&p, &[], e, &mut js.clone()).res.score, 48.0);
    // Debuffed middle copier breaks the chain: [Blueprint, Blueprint(debuffed), Joker] -> only
    // the Joker itself: 2 + 4 = 6 -> 144.
    let js = vec![
        jk(1, J::Blueprint),
        jk(2, J::Blueprint).with_debuff(true),
        jk(3, J::Joker),
    ];
    assert_eq!(score(&p, &[], e, &mut js.clone()).res.score, 144.0);
}

#[test]
fn copier_uses_its_own_edition() {
    // [Blueprint(Foil), Joker(Polychrome)]:
    // slot 0: Foil +50 chips (74), copy +4 (6); slot 1: +4 (10), Poly x1.5 (15) -> 74 x 15 = 1110.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::Joker]);
    js[0].edition = Edition::Foil;
    js[1].edition = Edition::Polychrome;
    assert_eq!(score(&p, &[], e, &mut js).res.score, 1110.0);
}

#[test]
fn copies_never_mutate_target_state() {
    // [Blueprint, Ride the Bus]: RtB scales to 1 in before_scoring; the copy reads it later.
    // mult 2 + 1 (copy) + 1 = 4 -> 96. RtB value 1, Blueprint value untouched.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::RideTheBus]);
    assert_eq!(score(&p, &[], e, &mut js).res.score, 96.0);
    assert_eq!((js[0].value, js[1].value), (0.0, 1.0));
}

#[test]
fn baseball_card_ordering_with_polychrome() {
    // [Bull(Holo), Hack(Polychrome), Baseball, Joker], $0, 7♠ 7♥ (chips 24, mult 2):
    // slot 0 Bull (uncommon): Holo +10 -> 12; Bull +0; Baseball x1.5 -> 18
    // slot 1 Hack (uncommon): no main; Baseball x1.5 -> 27; Polychrome x1.5 -> 40.5
    // slot 2 Baseball (rare): nothing
    // slot 3 Joker (common): +4 -> 44.5        => 24 x 44.5 = 1068
    let (p, e) = pair7();
    let mut js = jokers(&[J::Bull, J::Hack, J::BaseballCard, J::Joker]);
    js[0].edition = Edition::Holographic;
    js[1].edition = Edition::Polychrome;
    let o = score(&p, &[], e, &mut js);
    assert_eq!((o.res.mult, o.res.score), (44.5, 1068.0));
    // Baseball's x1.5 for Hack comes before Hack's Polychrome.
    let bb = o
        .trace
        .iter()
        .position(|t| t.source == TraceSource::Joker { slot: 2, src: 2 } && t.mult == 27.0)
        .expect("baseball on hack");
    let poly = o
        .trace
        .iter()
        .position(|t| t.source == TraceSource::JokerEdition { slot: 1 } && t.mult == 40.5)
        .expect("hack polychrome");
    assert!(bb < poly);
}

#[test]
fn blueprint_copies_baseball() {
    // [Hack, Blueprint, Baseball]: at Hack's turn both the copy and Baseball apply:
    // 2 x 1.5 x 1.5 = 4.5 -> 108.
    let (p, e) = pair7();
    let o = score(
        &p,
        &[],
        e,
        &mut jokers(&[J::Hack, J::Blueprint, J::BaseballCard]),
    );
    assert_eq!(o.res.score, 108.0);
}

#[test]
fn debuffed_joker_inert_but_counted_by_baseball() {
    // [Hack(debuffed, Foil), Baseball], 2♠ 2♥: no retrigger, no Foil; Baseball still x1.5.
    // chips 14, mult 2 x 1.5 = 3 -> 42.
    let (p, e) = pair2();
    let mut js = jokers(&[J::Hack, J::BaseballCard]);
    js[0].debuffed = true;
    js[0].edition = Edition::Foil;
    assert_eq!(score(&p, &[], e, &mut js).res.score, 42.0);
    // Debuffed Baseball does nothing: 28.
    let mut js = jokers(&[J::Hack, J::BaseballCard]);
    js[0].debuffed = true;
    js[1].debuffed = true;
    assert_eq!(score(&p, &[], e, &mut js).res.score, 28.0);
}

#[test]
fn mime_baron_steel_red_seal_held_king() {
    // Held Steel King with Red Seal; Mime + Baron. Triggers = 1 + 1 (Red Seal) + 1 (Mime) = 3.
    // Each trigger: Steel x1.5 then Baron x1.5 -> mult 2 x 2.25^3 = 22.78125
    // -> 24 x 22.78125 = 546.75 -> 546.
    let (p, e) = pair7();
    let held = [seal(enh(c(9, King, Hearts), Enhancement::Steel), Seal::Red)];
    let o = score(&p, &held, e, &mut jokers(&[J::Mime, J::Baron]));
    assert_eq!((o.res.mult, o.res.score), (22.78125, 546.0));
    // Blueprint copying Mime adds another trigger: 2 x 2.25^4 = 51.2578125 -> 1230.1875 -> 1230.
    let o = score(
        &p,
        &held,
        e,
        &mut jokers(&[J::Blueprint, J::Mime, J::Baron]),
    );
    assert_eq!(o.res.score, 1230.0);
}

#[test]
fn mime_retriggers_gold_at_round_end() {
    // Held Gold card + Mime: $3 x 2 = $6; with Red Seal too: $9 (balatrowiki.org/w/Mime).
    let held = [enh(c(1, Three, Clubs), Enhancement::Gold)];
    let mut js = jokers(&[J::Mime]);
    let mut scorer = Scorer::new();
    let mut rng = GameRng::new(1);
    let run = RunView::default();
    assert_eq!(
        scorer.round_end_held(&held, &mut js, &run, &mut rng, HandType::Pair),
        6
    );
    let held = [seal(enh(c(1, Three, Clubs), Enhancement::Gold), Seal::Red)];
    assert_eq!(
        scorer.round_end_held(&held, &mut js, &run, &mut rng, HandType::Pair),
        9
    );
}

// ----- events -----

fn round_end(js: &mut [JokerInstance], run: &RunView<'_>) -> (i32, Vec<SideEffect>) {
    let deck = Deck::standard();
    let mut rng = GameRng::new(2);
    let mut fx = Vec::new();
    let info = RoundEndInfo {
        won: true,
        round_score: 400.0,
        target: 300.0,
    };
    let r = fire_round_end(
        EventInput {
            jokers: js,
            run,
            deck: &deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        &info,
    );
    (r.money_delta, fx)
}

#[test]
fn golden_joker_pays_at_round_end() {
    let run = RunView::default();
    assert_eq!(round_end(&mut jokers(&[J::GoldenJoker]), &run).0, 4);
    // Not copied by Blueprint.
    assert_eq!(
        round_end(&mut jokers(&[J::Blueprint, J::GoldenJoker]), &run).0,
        4
    );
    // Debuffed: nothing.
    let mut js = vec![jk(1, J::GoldenJoker).with_debuff(true)];
    assert_eq!(round_end(&mut js, &run).0, 0);
    // Does nothing during scoring.
    let (p, e) = pair7();
    let o = score(&p, &[], e, &mut jokers(&[J::GoldenJoker]));
    assert_eq!((o.res.score, o.res.money_delta), (48.0, 0));
}

fn discard(
    js: &mut [JokerInstance],
    cards: &[Card],
    hand_type: HandType,
    run: &RunView<'_>,
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
        &DiscardInfo { cards, hand_type },
    );
    (r.money_delta, fx)
}

#[test]
fn burnt_joker_levels_first_discard_only() {
    let cards = [c(1, Two, Hearts), c(2, Five, Hearts), c(3, Nine, Hearts)];
    let first = RunView {
        discards_used_round: 1,
        blind: BlindKind::Big,
        ..RunView::default()
    };
    let up = SideEffect::LevelUpHand {
        hand: HandType::Flush,
        levels: 1,
    };
    let (_, fx) = discard(
        &mut jokers(&[J::BurntJoker]),
        &cards,
        HandType::Flush,
        &first,
    );
    assert_eq!(fx, vec![up]);
    // Second discard of the round: nothing.
    let second = RunView {
        discards_used_round: 2,
        ..RunView::default()
    };
    let (_, fx) = discard(
        &mut jokers(&[J::BurntJoker]),
        &cards,
        HandType::Flush,
        &second,
    );
    assert!(fx.is_empty());
    // Blueprint copies it: two level-ups.
    let mut js = jokers(&[J::Blueprint, J::BurntJoker]);
    let (_, fx) = discard(&mut js, &cards, HandType::Flush, &first);
    assert_eq!(fx, vec![up, up]);
}

#[test]
fn faceless_joker_needs_three_faces() {
    let run = RunView::default();
    let faces = [c(1, King, Hearts), c(2, Queen, Spades), c(3, Jack, Clubs)];
    let ht = HandType::HighCard;
    assert_eq!(
        discard(&mut jokers(&[J::FacelessJoker]), &faces, ht, &run).0,
        5
    );
    let two = [c(1, King, Hearts), c(2, Queen, Spades), c(3, Three, Clubs)];
    assert_eq!(
        discard(&mut jokers(&[J::FacelessJoker]), &two, ht, &run).0,
        0
    );
    // Debuffed faces are not faces.
    let deb = [
        c(1, King, Hearts),
        c(2, Queen, Spades),
        debuff(c(3, Jack, Clubs)),
    ];
    assert_eq!(
        discard(&mut jokers(&[J::FacelessJoker]), &deb, ht, &run).0,
        0
    );
    // Pareidolia: any three cards.
    let mut par = RunView::default();
    par.rules.pareidolia = true;
    let low = [c(1, Two, Hearts), c(2, Three, Spades), c(3, Four, Clubs)];
    assert_eq!(
        discard(&mut jokers(&[J::FacelessJoker]), &low, ht, &par).0,
        5
    );
}

// ----- property test -----

fn arb_card(uid: u32) -> impl Strategy<Value = Card> {
    (
        0usize..13,
        0usize..4,
        0usize..Enhancement::COUNT,
        0usize..Edition::COUNT,
        0usize..Seal::COUNT,
        prop::bool::weighted(0.1),
        0i32..30,
    )
        .prop_map(move |(r, s, e, ed, se, deb, perma)| {
            let mut card = Card::new(uid, Rank::ALL[r], Suit::ALL[s]);
            card.enhancement = Enhancement::ALL[e];
            card.edition = Edition::ALL[ed];
            card.seal = Seal::ALL[se];
            card.debuffed = deb;
            card.perma_bonus = perma;
            card
        })
}

fn arb_joker() -> impl Strategy<Value = (usize, usize, bool, f64)> {
    (
        0usize..JokerId::COUNT,
        0usize..Edition::COUNT,
        prop::bool::weighted(0.15),
        0.0f64..120.0,
    )
}

type Snapshot = (ScoreResult, Vec<SideEffect>, Vec<JokerInstance>, Vec<Card>);

#[allow(clippy::too_many_arguments)]
fn run_once(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    js: &[JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> Snapshot {
    let mut played = played.to_vec();
    let mut js = js.to_vec();
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::new();
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: &mut js,
        run,
        rng: &mut rng,
    });
    (res, scorer.effects().to_vec(), js, played)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_hands_never_panic_and_are_deterministic(
        played in prop::collection::vec(arb_card(0), 0..=5),
        held in prop::collection::vec(arb_card(50), 0..=8),
        joker_specs in prop::collection::vec(arb_joker(), 0..=5),
        ht in 0usize..HandType::COUNT,
        mask in any::<u8>(),
        contains in any::<u16>(),
        money in -20i32..200,
        pm in prop::sample::select(vec![0u32, 1, 2, 4, 1000]),
        levels in prop::array::uniform12(1u32..6),
        plays in prop::array::uniform12(0u32..12),
        flags in any::<u8>(),
        seed in any::<u64>(),
    ) {
        let played: Vec<Card> = played
            .into_iter()
            .enumerate()
            .map(|(i, mut c)| { c.uid = i as u32 + 1; c })
            .collect();
        let held: Vec<Card> = held
            .into_iter()
            .enumerate()
            .map(|(i, mut c)| { c.uid = i as u32 + 20; c })
            .collect();
        let js: Vec<JokerInstance> = joker_specs
            .iter()
            .enumerate()
            .map(|(i, &(id, ed, deb, v))| {
                let mut j = JokerInstance::new(100 + i as u32, JokerId::ALL[id]);
                j.edition = Edition::ALL[ed];
                j.debuffed = deb;
                if j.value != 0.0 || v > 100.0 {
                    j.value = v;
                }
                j
            })
            .collect();
        let hand_type = HandType::ALL[ht];
        let eval = HandEval {
            hand_type,
            scoring_mask: mask,
            contains: contains | (1 << hand_type.index()),
        };
        let mut run = RunView {
            money,
            prob_mult: pm,
            hand_levels: levels,
            hand_plays_run: plays,
            hand_plays_round: plays,
            hands_left: u32::from(flags & 3),
            observatory: flags & 4 != 0,
            plasma_deck: flags & 8 != 0,
            ..RunView::default()
        };
        run.boss.halve_base = flags & 16 != 0;
        run.boss.hand_invalid = flags & 32 != 0;
        run.rules.pareidolia = flags & 64 != 0;
        run.rules.smeared = flags & 128 != 0;

        let a = run_once(&played, &held, eval, &js, &run, seed);
        let b = run_once(&played, &held, eval, &js, &run, seed);
        prop_assert!(a.0.score >= 0.0 && a.0.score.is_finite(), "score {:?}", a.0);
        prop_assert!(a.0.chips.is_finite() && a.0.mult.is_finite());
        prop_assert_eq!(a.0.score, a.0.score.floor());
        prop_assert_eq!(&a, &b);
        prop_assert_eq!(a.3.len(), played.len());
    }
}
