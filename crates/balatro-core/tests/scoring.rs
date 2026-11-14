//! Scoring pipeline without jokers: hand-computed scenarios for every card modifier, retriggers,
//! debuffs, held Steel, boss hook points, late modifiers and the trace. `HandEval` values are
//! built by hand (hand_eval::evaluate is implemented elsewhere).

use balatro_core::card::Card;
use balatro_core::content::ConsumableId;
use balatro_core::deck::Pile;
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{ConsumableView, RunView, SideEffect};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{
    ScoreInput, ScoreResult, Scorer, TraceOp, TraceSource, halve_round_up, round_score,
};

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

fn ed(mut card: Card, e: Edition) -> Card {
    card.edition = e;
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

/// Pair of sevens (7♠ 7♥), both scoring: base 10 x 2, card chips 7 + 7.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
    trace: Vec<balatro_core::scoring::TraceEvent>,
    played: Vec<Card>,
}

fn run_full(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    jokers: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> Out {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::with_trace(true);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers,
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

fn score(played: &[Card], held: &[Card], eval: HandEval, run: &RunView<'_>) -> Out {
    run_full(played, held, eval, &mut [], run, 7)
}

fn plain() -> RunView<'static> {
    RunView::default()
}

/// prob_mult huge: every probability roll succeeds. prob_mult 0: every roll fails.
fn always() -> RunView<'static> {
    RunView {
        prob_mult: 1000,
        ..RunView::default()
    }
}

fn never() -> RunView<'static> {
    RunView {
        prob_mult: 0,
        ..RunView::default()
    }
}

fn glass_rolls(o: &Out) -> usize {
    o.trace
        .iter()
        .filter(|t| matches!(t.op, TraceOp::GlassRoll { .. }))
        .count()
}

// ----- rounding helpers -----

#[test]
fn round_score_floors_once() {
    assert_eq!(round_score(7.0, 1.5), 10.0); // 10.5 -> 10
    assert_eq!(round_score(0.0, 5.0), 0.0);
    assert_eq!(round_score(f64::NAN, 2.0), 0.0);
    assert_eq!(round_score(-3.0, 2.0), 0.0);
    assert_eq!(halve_round_up(5.0), 3.0);
    assert_eq!(halve_round_up(1.0), 1.0);
    assert_eq!(halve_round_up(10.0), 5.0);
}

// ----- base hands -----

#[test]
fn high_card_ace() {
    // High Card L1 = 5 x 1; Ace +11 -> 16 x 1 = 16.
    let o = score(
        &[c(1, Ace, Spades)],
        &[],
        ev(HandType::HighCard, 0b1, &[]),
        &plain(),
    );
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (16.0, 1.0, 16.0));
    assert_eq!(o.res.money_delta, 0);
}

#[test]
fn pair_with_non_scoring_kicker() {
    // Pair L1 = 10 x 2; K + K = 20 -> 30 x 2 = 60. The 3 kicker does not score.
    let played = [c(1, King, Spades), c(2, King, Hearts), c(3, Three, Clubs)];
    let o = score(&played, &[], ev(HandType::Pair, 0b011, &[]), &plain());
    assert_eq!(o.res.score, 60.0);
}

#[test]
fn kicker_modifiers_do_not_apply() {
    // Kicker with Foil/Mult/Gold Seal does not score: 24 x 2 = 48, $0.
    let (mut played, eval) = pair7();
    played.push(seal(
        ed(enh(c(3, King, Clubs), Enhancement::Mult), Edition::Foil),
        Seal::Gold,
    ));
    let o = score(&played, &[], eval, &plain());
    assert_eq!(o.res.score, 48.0);
    assert_eq!(o.res.money_delta, 0);
}

#[test]
fn flush_five_cards() {
    // Flush L1 = 35 x 4; 2+5+7+9+10 = 33 -> 68 x 4 = 272.
    let played = [
        c(1, Two, Hearts),
        c(2, Five, Hearts),
        c(3, Seven, Hearts),
        c(4, Nine, Hearts),
        c(5, King, Hearts),
    ];
    let o = score(&played, &[], ev(HandType::Flush, 0x1F, &[]), &plain());
    assert_eq!(o.res.score, 272.0);
}

#[test]
fn level_three_pair_changes_base() {
    // Pair L3 = (10 + 2*15) x (2 + 2*1) = 40 x 4; 5 + 5 -> 50 x 4 = 200.
    let mut run = plain();
    run.hand_levels[HandType::Pair.index()] = 3;
    let played = [c(1, Five, Spades), c(2, Five, Hearts)];
    let o = score(&played, &[], ev(HandType::Pair, 0b11, &[]), &run);
    assert_eq!(o.res.hand_level, 3);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (50.0, 4.0, 200.0));
}

// ----- enhancements -----

#[test]
fn bonus_card() {
    // 10 + (7 + 30) + 7 = 54 x 2 = 108.
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Bonus);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 108.0);
}

#[test]
fn mult_card() {
    // chips 24, mult 2 + 4 = 6 -> 144.
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Mult);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 144.0);
}

#[test]
fn glass_card_x2_and_survives() {
    // chips 24, mult 2 x 2 = 4 -> 96. One break roll; prob 0 -> no break.
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Glass);
    let o = score(&played, &[], eval, &never());
    assert_eq!(o.res.score, 96.0);
    assert_eq!(glass_rolls(&o), 1);
    assert!(o.effects.is_empty());
}

#[test]
fn glass_card_breaks_once_even_when_retriggered() {
    // Red Seal Glass: chips 10 + 7 + 7 + 7 = 31, mult 2 x 2 x 2 = 8 -> 248.
    // Two triggers but exactly one break roll; forced break -> one DestroyCard.
    let (mut played, eval) = pair7();
    played[0] = seal(enh(played[0], Enhancement::Glass), Seal::Red);
    let o = score(&played, &[], eval, &always());
    assert_eq!(o.res.score, 248.0);
    assert_eq!(glass_rolls(&o), 1);
    assert_eq!(o.effects, vec![SideEffect::DestroyCard { uid: 1 }]);
}

#[test]
fn debuffed_glass_neither_scores_nor_breaks() {
    // Debuffed Glass adds nothing and is not rolled: chips 17 x 2 = 34.
    let (mut played, eval) = pair7();
    played[0] = debuff(enh(played[0], Enhancement::Glass));
    let o = score(&played, &[], eval, &always());
    assert_eq!(o.res.score, 34.0);
    assert_eq!(glass_rolls(&o), 0);
    assert!(o.effects.is_empty());
}

#[test]
fn stone_card_always_scores_without_rank_chips() {
    // High Card 5 x 1; Ace 11; Stone +50 (no rank chips) -> 66 x 1 = 66.
    let played = [
        c(1, Ace, Spades),
        enh(c(2, Three, Clubs), Enhancement::Stone),
    ];
    let o = score(&played, &[], ev(HandType::HighCard, 0b11, &[]), &plain());
    assert_eq!(o.res.score, 66.0);
}

#[test]
fn lucky_card_forced_hits() {
    // prob forced: +20 mult and $20. chips 24, mult 2 + 20 = 22 -> 528.
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Lucky);
    let o = score(&played, &[], eval, &always());
    assert_eq!(o.res.score, 528.0);
    assert_eq!(o.res.money_delta, 20);
}

#[test]
fn lucky_card_forced_misses() {
    // prob 0: nothing. 24 x 2 = 48, $0, one LuckyRoll(false, false).
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Lucky);
    let o = score(&played, &[], eval, &never());
    assert_eq!(o.res.score, 48.0);
    assert_eq!(o.res.money_delta, 0);
    let rolls: Vec<_> = o
        .trace
        .iter()
        .filter(|t| matches!(t.op, TraceOp::LuckyRoll { .. }))
        .collect();
    assert_eq!(rolls.len(), 1);
    assert_eq!(
        rolls[0].op,
        TraceOp::LuckyRoll {
            mult_hit: false,
            money_hit: false
        }
    );
}

#[test]
fn lucky_rerolls_on_every_trigger() {
    // Red Seal Lucky, forced: two triggers, two independent rolls.
    // chips 10 + 7 + 7 + 7 = 31; mult 2 + 20 + 20 = 42 -> 1302; $40.
    let (mut played, eval) = pair7();
    played[0] = seal(enh(played[0], Enhancement::Lucky), Seal::Red);
    let o = score(&played, &[], eval, &always());
    assert_eq!(o.res.score, 1302.0);
    assert_eq!(o.res.money_delta, 40);
    let rolls = o
        .trace
        .iter()
        .filter(|t| matches!(t.op, TraceOp::LuckyRoll { .. }))
        .count();
    assert_eq!(rolls, 2);
}

#[test]
fn lucky_seeded_rolls_match_trace_and_odds() {
    // With prob_mult 1, every seed's outcome must agree with its trace, and the hit rates over
    // 4000 fixed seeds must be near 1/5 (mult) and 1/15 (money). Deterministic (fixed seeds).
    let (mut played, eval) = pair7();
    played[0] = enh(played[0], Enhancement::Lucky);
    let (mut mult_hits, mut money_hits) = (0, 0);
    for seed in 0..4000u64 {
        let o = run_full(&played, &[], eval, &mut [], &plain(), seed);
        let roll = o
            .trace
            .iter()
            .find_map(|t| match t.op {
                TraceOp::LuckyRoll {
                    mult_hit,
                    money_hit,
                } => Some((mult_hit, money_hit)),
                _ => None,
            })
            .expect("lucky card rolled");
        let expect_mult = if roll.0 { 22.0 } else { 2.0 };
        let expect_money = if roll.1 { 20 } else { 0 };
        assert_eq!(o.res.mult, expect_mult, "seed {seed}");
        assert_eq!(o.res.money_delta, expect_money, "seed {seed}");
        mult_hits += usize::from(roll.0);
        money_hits += usize::from(roll.1);
    }
    assert!((650..=950).contains(&mult_hits), "mult hits {mult_hits}");
    assert!((180..=360).contains(&money_hits), "money hits {money_hits}");
}

#[test]
fn gold_and_wild_and_steel_do_nothing_when_played() {
    // Gold/Wild/Steel have no "when scored" effect: 24 x 2 = 48, $0.
    for e in [Enhancement::Gold, Enhancement::Wild, Enhancement::Steel] {
        let (mut played, eval) = pair7();
        played[0] = enh(played[0], e);
        let o = score(&played, &[], eval, &plain());
        assert_eq!((o.res.score, o.res.money_delta), (48.0, 0), "{e:?}");
    }
}

// ----- editions -----

#[test]
fn foil_card() {
    // chips 24 + 50 = 74 x 2 = 148.
    let (mut played, eval) = pair7();
    played[0] = ed(played[0], Edition::Foil);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 148.0);
}

#[test]
fn holographic_card() {
    // chips 24, mult 2 + 10 = 12 -> 288.
    let (mut played, eval) = pair7();
    played[0] = ed(played[0], Edition::Holographic);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 288.0);
}

#[test]
fn polychrome_card() {
    // chips 24, mult 2 x 1.5 = 3 -> 72.
    let (mut played, eval) = pair7();
    played[0] = ed(played[0], Edition::Polychrome);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 72.0);
}

#[test]
fn polychrome_fraction_is_floored_once() {
    // High Card: chips 5 + 2 = 7, mult 1 x 1.5 = 1.5 -> 10.5 -> floor 10.
    let played = [ed(c(1, Two, Spades), Edition::Polychrome)];
    let o = score(&played, &[], ev(HandType::HighCard, 0b1, &[]), &plain());
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (7.0, 1.5, 10.0));
}

#[test]
fn edition_after_enhancement_order() {
    // Mult card + Polychrome: (2 + 4) x 1.5 = 9 (not 2 x 1.5 + 4 = 7). 24 x 9 = 216.
    let (mut played, eval) = pair7();
    played[0] = ed(enh(played[0], Enhancement::Mult), Edition::Polychrome);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 216.0);
}

// ----- seals -----

#[test]
fn gold_seal_pays_per_card() {
    // Two Gold Seal cards: $3 each; score unchanged 48.
    let (mut played, eval) = pair7();
    played[0] = seal(played[0], Seal::Gold);
    played[1] = seal(played[1], Seal::Gold);
    let o = score(&played, &[], eval, &plain());
    assert_eq!((o.res.score, o.res.money_delta), (48.0, 6));
}

#[test]
fn red_seal_retriggers_once() {
    // 10 + 7 + 7 (retrigger) + 7 = 31 x 2 = 62.
    let (mut played, eval) = pair7();
    played[0] = seal(played[0], Seal::Red);
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 62.0);
}

#[test]
fn red_seal_repeats_enhancement_and_edition() {
    // Red Seal Mult card with Foil: chips 10 + (7 + 50) * 2 + 7 = 131;
    // mult 2 + 4 + 4 = 10 -> 1310.
    let (mut played, eval) = pair7();
    played[0] = ed(
        seal(enh(played[0], Enhancement::Mult), Seal::Red),
        Edition::Foil,
    );
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 1310.0);
}

#[test]
fn blue_and_purple_seals_do_nothing_when_scored() {
    for s in [Seal::Blue, Seal::Purple] {
        let (mut played, eval) = pair7();
        played[0] = seal(played[0], s);
        let o = score(&played, &[], eval, &plain());
        assert_eq!((o.res.score, o.res.money_delta), (48.0, 0));
        assert!(o.effects.is_empty());
    }
}

// ----- debuffed played cards -----

#[test]
fn debuffed_scoring_card_contributes_nothing() {
    // Debuffed 7♠ with Foil, Mult, Gold Seal, Red Seal: skipped entirely.
    // chips 10 + 7 = 17 x 2 = 34, $0.
    let (mut played, eval) = pair7();
    played[0] = debuff(seal(
        ed(enh(played[0], Enhancement::Mult), Edition::Foil),
        Seal::Gold,
    ));
    let o = score(&played, &[], eval, &plain());
    assert_eq!((o.res.score, o.res.money_delta), (34.0, 0));
    assert!(
        o.trace
            .iter()
            .any(|t| t.op == TraceOp::Debuffed { card: 0 })
    );
}

#[test]
fn perma_bonus_adds_to_card_chips() {
    // 10 + (7 + 15) + 7 = 39 x 2 = 78.
    let (mut played, eval) = pair7();
    played[0].perma_bonus = 15;
    assert_eq!(score(&played, &[], eval, &plain()).res.score, 78.0);
}

// ----- held cards -----

#[test]
fn steel_held() {
    // 24 x (2 x 1.5 = 3) = 72.
    let (played, eval) = pair7();
    let held = [enh(c(9, Three, Clubs), Enhancement::Steel)];
    assert_eq!(score(&played, &held, eval, &plain()).res.score, 72.0);
}

#[test]
fn steel_held_with_red_seal() {
    // Red Seal retriggers the held Steel: 2 x 1.5 x 1.5 = 4.5 -> 24 x 4.5 = 108.
    let (played, eval) = pair7();
    let held = [seal(enh(c(9, Three, Clubs), Enhancement::Steel), Seal::Red)];
    let o = score(&played, &held, eval, &plain());
    assert_eq!((o.res.mult, o.res.score), (4.5, 108.0));
}

#[test]
fn debuffed_steel_held_does_nothing() {
    let (played, eval) = pair7();
    let held = [debuff(enh(c(9, Three, Clubs), Enhancement::Steel))];
    assert_eq!(score(&played, &held, eval, &plain()).res.score, 48.0);
}

#[test]
fn held_editions_and_gold_do_nothing_during_scoring() {
    // Held Foil/Holo/Poly/Gold/Gold Seal: no effect while scoring. 48.
    let (played, eval) = pair7();
    let held = [
        ed(c(9, Three, Clubs), Edition::Foil),
        ed(c(10, Four, Clubs), Edition::Polychrome),
        seal(enh(c(11, Five, Clubs), Enhancement::Gold), Seal::Gold),
    ];
    let o = score(&played, &held, eval, &plain());
    assert_eq!((o.res.score, o.res.money_delta), (48.0, 0));
}

// ----- boss hook points -----

#[test]
fn flint_halves_pair_base_rounding_up() {
    // The Flint: Pair 10 x 2 -> 5 x 1; + 7 + 7 = 19 x 1 = 19.
    let (played, eval) = pair7();
    let mut run = plain();
    run.boss.halve_base = true;
    assert_eq!(score(&played, &[], eval, &run).res.score, 19.0);
}

#[test]
fn flint_rounds_odd_values_up() {
    // High Card 5 x 1 -> ceil(2.5)=3 x ceil(0.5)=1; Ace +11 -> 14.
    let mut run = plain();
    run.boss.halve_base = true;
    let o = score(
        &[c(1, Ace, Spades)],
        &[],
        ev(HandType::HighCard, 1, &[]),
        &run,
    );
    assert_eq!(o.res.score, 14.0);
    // Pair L2 = 25 x 3 -> 13 x 2; + 14 = 27 x 2 = 54.
    run.hand_levels[HandType::Pair.index()] = 2;
    let (played, eval) = pair7();
    assert_eq!(score(&played, &[], eval, &run).res.score, 54.0);
}

#[test]
fn invalid_hand_scores_zero_and_fires_nothing() {
    // Boss rejects the hand: score 0, no Gold Seal money, no Glass roll, no effects.
    let (mut played, eval) = pair7();
    played[0] = seal(enh(played[0], Enhancement::Glass), Seal::Gold);
    let mut run = always();
    run.boss.hand_invalid = true;
    let o = score(&played, &[], eval, &run);
    assert!(o.res.invalid);
    assert_eq!((o.res.score, o.res.chips, o.res.mult), (0.0, 0.0, 0.0));
    assert_eq!(o.res.money_delta, 0);
    assert!(o.effects.is_empty());
    assert_eq!(glass_rolls(&o), 0);
    assert_eq!(o.played, played);
}

// ----- late modifiers -----

#[test]
fn observatory_multiplies_per_matching_planet() {
    // Two Pair-Planets held + Observatory: 2 x 1.5 x 1.5 = 4.5 -> 24 x 4.5 = 108.
    // Without the voucher: 48.
    let planet = |h| ConsumableView::new(ConsumableId::planet_for(h));
    let cons = [
        planet(HandType::Pair),
        planet(HandType::Flush),
        planet(HandType::Pair),
    ];
    let (played, eval) = pair7();
    let mut run = RunView {
        consumables: &cons,
        ..plain()
    };
    assert_eq!(score(&played, &[], eval, &run).res.score, 48.0);
    run.observatory = true;
    assert_eq!(score(&played, &[], eval, &run).res.score, 108.0);
}

#[test]
fn plasma_deck_balances() {
    // chips 24, mult 2 -> both (24 + 2) / 2 = 13 -> 169.
    let (played, eval) = pair7();
    let run = RunView {
        plasma_deck: true,
        ..plain()
    };
    let o = score(&played, &[], eval, &run);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (13.0, 13.0, 169.0));
}

// ----- whole-hand checks -----

#[test]
fn straight_with_mixed_modifiers() {
    // Straight L1 30 x 4. Cards 5♠(Bonus) 6♥(Foil) 7♦(Mult, Gold Seal) 8♣(Red Seal) 9♠(Holo):
    // chips: 30 + (5+30) + (6+50) + 7 + 8 + 8 + 9 = 153
    // mult: 4 + 4 (Mult) + 10 (Holo) = 18 -> 153 x 18 = 2754; $3.
    let played = [
        enh(c(1, Five, Spades), Enhancement::Bonus),
        ed(c(2, Six, Hearts), Edition::Foil),
        seal(enh(c(3, Seven, Diamonds), Enhancement::Mult), Seal::Gold),
        seal(c(4, Eight, Clubs), Seal::Red),
        ed(c(5, Nine, Spades), Edition::Holographic),
    ];
    let o = score(&played, &[], ev(HandType::Straight, 0x1F, &[]), &plain());
    assert_eq!((o.res.chips, o.res.mult), (153.0, 18.0));
    assert_eq!((o.res.score, o.res.money_delta), (2754.0, 3));
}

#[test]
fn trace_ends_at_result_and_off_means_empty() {
    let (mut played, eval) = pair7();
    played[0] = ed(seal(played[0], Seal::Red), Edition::Polychrome);
    let held = [enh(c(9, Three, Clubs), Enhancement::Steel)];
    let o = score(&played, &held, eval, &plain());
    let last = o.trace.last().expect("trace recorded");
    assert_eq!((last.chips, last.mult), (o.res.chips, o.res.mult));
    assert_eq!(o.trace[0].source, TraceSource::Base);

    let mut p = played.clone();
    let mut rng = GameRng::new(1);
    let mut scorer = Scorer::new();
    let run = plain();
    let r = scorer.score_hand(ScoreInput {
        played: &mut p,
        held: &held,
        eval,
        jokers: &mut [],
        run: &run,
        rng: &mut rng,
    });
    assert_eq!(r, o.res);
    assert!(scorer.trace().is_empty());
}

#[test]
fn empty_and_oversized_inputs_do_not_panic() {
    // Nothing played: High Card base 5 x 1 = 5.
    let o = score(&[], &[], ev(HandType::HighCard, 0, &[]), &plain());
    assert_eq!(o.res.score, 5.0);
    // Mask bits beyond the played cards are ignored.
    let o = score(
        &[c(1, Ace, Spades)],
        &[],
        ev(HandType::HighCard, 0xFF, &[]),
        &plain(),
    );
    assert_eq!(o.res.score, 16.0);
}

// ----- end-of-round held effects -----

#[test]
fn round_end_held_gold_and_blue_seal() {
    // Gold $3; Gold + Red Seal $6; debuffed Gold $0; Blue Seal -> Planet of the last hand.
    let held = [
        enh(c(1, Three, Clubs), Enhancement::Gold),
        seal(enh(c(2, Five, Diamonds), Enhancement::Gold), Seal::Red),
        seal(c(3, Nine, Spades), Seal::Blue),
        debuff(enh(c(4, Four, Hearts), Enhancement::Gold)),
    ];
    let mut scorer = Scorer::new();
    let mut rng = GameRng::new(3);
    let money = scorer.round_end_held(&held, &mut [], &plain(), &mut rng, HandType::Flush);
    assert_eq!(money, 9);
    assert_eq!(
        scorer.effects(),
        &[SideEffect::CreatePlanetFor {
            hand: HandType::Flush
        }]
    );
}

#[test]
fn side_effect_add_card_shape() {
    // AddCard carries a template card and a destination pile (DNA / Marble / Certificate).
    let e = SideEffect::AddCard {
        card: c(0, Ace, Spades),
        to: Pile::Hand,
    };
    assert!(matches!(e, SideEffect::AddCard { to: Pile::Hand, .. }));
}

#[test]
fn same_seed_same_result() {
    let (mut played, eval) = pair7();
    played[0] = seal(enh(played[0], Enhancement::Lucky), Seal::Red);
    played[1] = enh(played[1], Enhancement::Glass);
    let a = run_full(&played, &[], eval, &mut [], &plain(), 99);
    let b = run_full(&played, &[], eval, &mut [], &plain(), 99);
    assert_eq!(a.res, b.res);
    assert_eq!(a.effects, b.effects);
    assert_eq!(a.trace, b.trace);
}
