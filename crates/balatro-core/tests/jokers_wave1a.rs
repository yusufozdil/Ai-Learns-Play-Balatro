//! Phase 3 package wave1a (content/jokers/wave1a.rs): trigger / non-trigger / edge cases and
//! Blueprint copies for each of the 24 jokers. Every expected value is hand-computed.
//! Base hands (level 1): High Card 5 x 1, Pair 10 x 2, Two Pair 20 x 2, Three of a Kind 30 x 3,
//! Straight 30 x 4, Flush 35 x 4. Rank chips: 2..9 = face value, 10/J/Q/K = 10, A = 11.

use balatro_core::card::Card;
use balatro_core::content::jokers::{JokerId, try_def};
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Rarity, Seal, Suit};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{DeckCounts, RunView};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer};

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

fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| JokerInstance::new(100 + i as u32, id))
        .collect()
}

fn score_seed(
    played: &[Card],
    eval: HandEval,
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> ScoreResult {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::with_trace(false);
    scorer.score_hand(ScoreInput {
        played: &mut played,
        held: &[],
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    })
}

fn score_run(played: &[Card], eval: HandEval, js: &mut [JokerInstance], run: &RunView<'_>) -> f64 {
    score_seed(played, eval, js, run, 11).score
}

fn score(played: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> f64 {
    score_run(played, eval, js, &RunView::default())
}

/// 7♠ 7♥ Pair: 10 + 14 = 24 x 2 = 48 bare.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

fn pair_of(rank: Rank) -> (Vec<Card>, HandEval) {
    (
        vec![c(1, rank, Spades), c(2, rank, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

/// 7♠ 7♥ 7♦: 30 + 21 = 51 x 3.
fn trips7() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Seven, Spades),
            c(2, Seven, Hearts),
            c(3, Seven, Diamonds),
        ],
        ev(HandType::ThreeOfAKind, 0b111, &[HandType::Pair]),
    )
}

/// 7♠ 7♥ 3♦ 3♣: 20 + 20 = 40 x 2.
fn two_pair() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Seven, Spades),
            c(2, Seven, Hearts),
            c(3, Three, Diamonds),
            c(4, Three, Clubs),
        ],
        ev(HandType::TwoPair, 0b1111, &[HandType::Pair]),
    )
}

/// 5♠ 6♥ 7♦ 8♣ 9♠: 30 + 35 = 65 x 4.
fn straight() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Five, Spades),
            c(2, Six, Hearts),
            c(3, Seven, Diamonds),
            c(4, Eight, Clubs),
            c(5, Nine, Spades),
        ],
        ev(HandType::Straight, 0b11111, &[]),
    )
}

/// 2♠ 4♠ 6♠ 8♠ 10♠: 35 + 30 = 65 x 4.
fn flush() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Two, Spades),
            c(2, Four, Spades),
            c(3, Six, Spades),
            c(4, Eight, Spades),
            c(5, Ten, Spades),
        ],
        ev(HandType::Flush, 0b11111, &[]),
    )
}

/// K♠ alone: 5 + 10 = 15 x 1.
fn high_card_k() -> (Vec<Card>, HandEval) {
    (vec![c(1, King, Spades)], ev(HandType::HighCard, 0b1, &[]))
}

fn run_with(discards_left: u32, joker_slots: u32) -> RunView<'static> {
    RunView {
        discards_left,
        joker_slots,
        ..RunView::default()
    }
}

// ----- registry / definitions -----

#[test]
fn all_24_defined_with_wiki_stats() {
    let table = [
        (J::LustyJoker, Rarity::Common, 5),
        (J::WrathfulJoker, Rarity::Common, 5),
        (J::GluttonousJoker, Rarity::Common, 5),
        (J::ZanyJoker, Rarity::Common, 4),
        (J::MadJoker, Rarity::Common, 4),
        (J::CrazyJoker, Rarity::Common, 4),
        (J::DrollJoker, Rarity::Common, 4),
        (J::SlyJoker, Rarity::Common, 3),
        (J::WilyJoker, Rarity::Common, 4),
        (J::CleverJoker, Rarity::Common, 4),
        (J::DeviousJoker, Rarity::Common, 4),
        (J::CraftyJoker, Rarity::Common, 4),
        (J::HalfJoker, Rarity::Common, 5),
        (J::JokerStencil, Rarity::Uncommon, 8),
        (J::Banner, Rarity::Common, 5),
        (J::MysticSummit, Rarity::Common, 5),
        (J::Misprint, Rarity::Common, 4),
        (J::Fibonacci, Rarity::Uncommon, 8),
        (J::SteelJoker, Rarity::Uncommon, 7),
        (J::ScaryFace, Rarity::Common, 4),
        (J::AbstractJoker, Rarity::Common, 4),
        (J::EvenSteven, Rarity::Common, 4),
        (J::OddTodd, Rarity::Common, 4),
        (J::Scholar, Rarity::Common, 4),
    ];
    for (id, rarity, cost) in table {
        let d = try_def(id).unwrap_or_else(|| panic!("{id:?} not defined"));
        assert_eq!(d.id, id);
        assert_eq!(d.rarity, rarity, "{id:?} rarity");
        assert_eq!(d.base_cost, cost, "{id:?} cost");
        assert!(d.blueprint_compatible && d.eternal_compatible && d.perishable_compatible);
    }
}

// ----- Lusty / Wrathful / Gluttonous -----

#[test]
fn lusty_triggers_on_hearts_only() {
    // 7♠ 7♥: one Heart scores: 24 x (2 + 3) = 120.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut jokers(&[J::LustyJoker])), 120.0);
    // 7♠ 7♣: no Heart: 48.
    let p2 = [c(1, Seven, Spades), c(2, Seven, Clubs)];
    assert_eq!(score(&p2, e, &mut jokers(&[J::LustyJoker])), 48.0);
}

#[test]
fn wrathful_and_gluttonous_suits() {
    // 7♠ 7♥: Spade scores once: 24 x 5 = 120; Gluttonous (Clubs) gives nothing: 48.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut jokers(&[J::WrathfulJoker])), 120.0);
    assert_eq!(score(&p, e, &mut jokers(&[J::GluttonousJoker])), 48.0);
    // 7♣ 7♣: two Clubs: 24 x (2 + 3 + 3) = 192.
    let pc = [c(1, Seven, Clubs), c(2, Seven, Clubs)];
    assert_eq!(score(&pc, e, &mut jokers(&[J::GluttonousJoker])), 192.0);
}

#[test]
fn suit_jokers_wild_and_debuffed_and_retrigger() {
    let e = ev(HandType::Pair, 0b11, &[]);
    // Wild 7♠ counts as Heart: both cards score Lusty: 24 x (2 + 3 + 3) = 192.
    let wild = [
        enh(c(1, Seven, Spades), Enhancement::Wild),
        c(2, Seven, Hearts),
    ];
    assert_eq!(score(&wild, e, &mut jokers(&[J::LustyJoker])), 192.0);
    // Debuffed 7♥ is skipped entirely: chips 10 + 7 = 17 x 2 = 34.
    let deb = [c(1, Seven, Spades), debuff(c(2, Seven, Hearts))];
    assert_eq!(score(&deb, e, &mut jokers(&[J::LustyJoker])), 34.0);
    // Red Seal on 7♥ retriggers: chips 10 + 7 + 14 = 31; mult 2 + 3 + 3 = 8 -> 248.
    let mut red = c(2, Seven, Hearts);
    red.seal = Seal::Red;
    let rp = [c(1, Seven, Spades), red];
    assert_eq!(score(&rp, e, &mut jokers(&[J::LustyJoker])), 248.0);
}

#[test]
fn suit_jokers_blueprint() {
    // [Blueprint, Lusty], 7♠ 7♥: +3 twice: 24 x (2 + 6) = 192.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::LustyJoker])),
        192.0
    );
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::WrathfulJoker])),
        192.0
    );
}

// ----- hand-type jokers -----

#[test]
fn zany_mad_crazy_droll_mult() {
    let (t, te) = trips7();
    // Zany: 51 x (3 + 12) = 765.
    assert_eq!(score(&t, te, &mut jokers(&[J::ZanyJoker])), 765.0);
    let (m, me) = two_pair();
    // Mad: 40 x (2 + 10) = 480.
    assert_eq!(score(&m, me, &mut jokers(&[J::MadJoker])), 480.0);
    let (s, se) = straight();
    // Crazy: 65 x (4 + 12) = 1040.
    assert_eq!(score(&s, se, &mut jokers(&[J::CrazyJoker])), 1040.0);
    let (f, fe) = flush();
    // Droll: 65 x (4 + 10) = 910.
    assert_eq!(score(&f, fe, &mut jokers(&[J::DrollJoker])), 910.0);
}

#[test]
fn sly_wily_clever_devious_crafty_chips() {
    let (p, pe) = pair7();
    // Sly: (24 + 50) x 2 = 148.
    assert_eq!(score(&p, pe, &mut jokers(&[J::SlyJoker])), 148.0);
    let (t, te) = trips7();
    // Wily: (51 + 100) x 3 = 453.
    assert_eq!(score(&t, te, &mut jokers(&[J::WilyJoker])), 453.0);
    let (m, me) = two_pair();
    // Clever: (40 + 80) x 2 = 240.
    assert_eq!(score(&m, me, &mut jokers(&[J::CleverJoker])), 240.0);
    let (s, se) = straight();
    // Devious: (65 + 100) x 4 = 660.
    assert_eq!(score(&s, se, &mut jokers(&[J::DeviousJoker])), 660.0);
    let (f, fe) = flush();
    // Crafty: (65 + 80) x 4 = 580.
    assert_eq!(score(&f, fe, &mut jokers(&[J::CraftyJoker])), 580.0);
}

#[test]
fn hand_type_jokers_do_not_trigger_elsewhere() {
    // High Card K: 15 x 1 = 15 for every one of them.
    let (h, he) = high_card_k();
    for id in [
        J::ZanyJoker,
        J::MadJoker,
        J::CrazyJoker,
        J::DrollJoker,
        J::SlyJoker,
        J::WilyJoker,
        J::CleverJoker,
        J::DeviousJoker,
        J::CraftyJoker,
    ] {
        assert_eq!(score(&h, he, &mut jokers(&[id])), 15.0, "{id:?}");
    }
    // Two Pair contains a Pair but not Three of a Kind: Zany / Wily silent (40 x 2 = 80).
    let (m, me) = two_pair();
    assert_eq!(score(&m, me, &mut jokers(&[J::ZanyJoker])), 80.0);
    assert_eq!(score(&m, me, &mut jokers(&[J::WilyJoker])), 80.0);
    // Sly sees the Pair inside Two Pair: (40 + 50) x 2 = 180.
    assert_eq!(score(&m, me, &mut jokers(&[J::SlyJoker])), 180.0);
    // Pair is not Two Pair: Mad / Clever silent (24 x 2 = 48).
    let (p, pe) = pair7();
    assert_eq!(score(&p, pe, &mut jokers(&[J::MadJoker])), 48.0);
    assert_eq!(score(&p, pe, &mut jokers(&[J::CleverJoker])), 48.0);
}

#[test]
fn full_house_contains_trips_and_pair() {
    // 7♠ 7♥ 7♦ 3♣ 3♦ Full House (40 x 4): chips 40 + 21 + 6 = 67.
    let p = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Seven, Diamonds),
        c(4, Three, Clubs),
        c(5, Three, Diamonds),
    ];
    let e = ev(
        HandType::FullHouse,
        0b11111,
        &[HandType::ThreeOfAKind, HandType::Pair, HandType::TwoPair],
    );
    // Zany: 67 x (4 + 12) = 1072.
    assert_eq!(score(&p, e, &mut jokers(&[J::ZanyJoker])), 1072.0);
    // Wily: (67 + 100) x 4 = 668.
    assert_eq!(score(&p, e, &mut jokers(&[J::WilyJoker])), 668.0);
}

#[test]
fn hand_type_blueprint() {
    // [Blueprint, Zany] trips: 51 x (3 + 24) = 1377.
    let (t, te) = trips7();
    assert_eq!(
        score(&t, te, &mut jokers(&[J::Blueprint, J::ZanyJoker])),
        1377.0
    );
    // [Blueprint, Sly] pair: (24 + 100) x 2 = 248.
    let (p, pe) = pair7();
    assert_eq!(
        score(&p, pe, &mut jokers(&[J::Blueprint, J::SlyJoker])),
        248.0
    );
}

// ----- Half Joker -----

#[test]
fn half_joker_card_count_edges() {
    // 3 cards (trips 7): 51 x (3 + 20) = 1173.
    let (t, te) = trips7();
    assert_eq!(score(&t, te, &mut jokers(&[J::HalfJoker])), 1173.0);
    // 1 card (K high card): 15 x (1 + 20) = 315.
    let (h, he) = high_card_k();
    assert_eq!(score(&h, he, &mut jokers(&[J::HalfJoker])), 315.0);
    // 4 cards played (3 score + a non-scoring 2♣): 4 > 3, nothing: 51 x 3 = 153.
    let mut four = t.clone();
    four.push(c(4, Two, Clubs));
    let e4 = ev(HandType::ThreeOfAKind, 0b0111, &[HandType::Pair]);
    assert_eq!(score(&four, e4, &mut jokers(&[J::HalfJoker])), 153.0);
    // 5 cards (straight): 65 x 4 = 260.
    let (s, se) = straight();
    assert_eq!(score(&s, se, &mut jokers(&[J::HalfJoker])), 260.0);
    // Blueprint copy: 1 card: 15 x (1 + 40) = 615.
    assert_eq!(
        score(&h, he, &mut jokers(&[J::Blueprint, J::HalfJoker])),
        615.0
    );
}

// ----- Joker Stencil -----

#[test]
fn stencil_counts_empty_slots_and_itself() {
    let (p, e) = pair7();
    let run = RunView::default(); // 5 slots
    // Alone: 4 empty + itself = X5: 24 x (2 x 5) = 240.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::JokerStencil]), &run),
        240.0
    );
    // [Stencil, Joker]: 3 empty + itself = X4; mult 2 x 4 = 8, +4 = 12 -> 288.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::JokerStencil, J::Joker]), &run),
        288.0
    );
    // [Joker, Stencil]: (2 + 4) x 4 = 24 -> 576.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::Joker, J::JokerStencil]), &run),
        576.0
    );
    // Full board [Stencil, Joker x4]: 0 empty + itself = X1 (no-op); 2 x 1 + 16 = 18 -> 432.
    assert_eq!(
        score_run(
            &p,
            e,
            &mut jokers(&[J::JokerStencil, J::Joker, J::Joker, J::Joker, J::Joker]),
            &run
        ),
        432.0
    );
}

#[test]
fn stencil_two_stencils_and_negative_slots() {
    let (p, e) = pair7();
    // Two Stencils: 5 - 2 + 2 = 5 each: 2 x 5 x 5 = 50 -> 1200.
    assert_eq!(
        score_run(
            &p,
            e,
            &mut jokers(&[J::JokerStencil, J::JokerStencil]),
            &RunView::default()
        ),
        1200.0
    );
    // A Negative joker raises capacity to 6 and occupies one: [Stencil, Negative Joker]
    // 6 - 2 + 1 = 5 -> X5: 2 x 5 = 10, +4 = 14 -> 336.
    let mut js = jokers(&[J::JokerStencil, J::Joker]);
    js[1] = js[1].with_edition(Edition::Negative);
    assert_eq!(score_run(&p, e, &mut js, &run_with(0, 6)), 336.0);
}

#[test]
fn stencil_blueprint_copy() {
    // [Blueprint, Stencil]: 5 - 2 + 1 = 4 empty-equivalent; copy X4 then Stencil X4:
    // 2 x 4 x 4 = 32 -> 768.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::JokerStencil])),
        768.0
    );
}

// ----- Banner / Mystic Summit -----

#[test]
fn banner_scales_with_discards() {
    let (p, e) = pair7();
    // 3 discards: (24 + 90) x 2 = 228.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::Banner]), &run_with(3, 5)),
        228.0
    );
    // 0 discards: 48.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::Banner]), &run_with(0, 5)),
        48.0
    );
    // Blueprint, 2 discards: (24 + 60 + 60) x 2 = 288.
    assert_eq!(
        score_run(
            &p,
            e,
            &mut jokers(&[J::Blueprint, J::Banner]),
            &run_with(2, 5)
        ),
        288.0
    );
}

#[test]
fn mystic_summit_only_at_zero_discards() {
    let (p, e) = pair7();
    // 0 discards: 24 x (2 + 15) = 408.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::MysticSummit]), &run_with(0, 5)),
        408.0
    );
    // 1 discard: 48.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::MysticSummit]), &run_with(1, 5)),
        48.0
    );
    // Blueprint at 0 discards: 24 x (2 + 30) = 768.
    assert_eq!(
        score_run(
            &p,
            e,
            &mut jokers(&[J::Blueprint, J::MysticSummit]),
            &run_with(0, 5)
        ),
        768.0
    );
}

// ----- Misprint -----

fn misprint_mult(seed: u64) -> f64 {
    // Pair 7s: 24 chips x (2 + m): recover m from the final mult.
    let (p, e) = pair7();
    let r = score_seed(
        &p,
        e,
        &mut jokers(&[J::Misprint]),
        &RunView::default(),
        seed,
    );
    r.mult - 2.0
}

#[test]
fn misprint_range_zero_to_twentythree() {
    let mut seen = [false; 24];
    for seed in 0..600 {
        let m = misprint_mult(seed);
        assert!(
            m.fract() == 0.0 && (0.0..=23.0).contains(&m),
            "seed {seed}: {m}"
        );
        seen[m as usize] = true;
    }
    assert!(seen[0] && seen[23], "extremes 0 and 23 must be reachable");
}

#[test]
fn misprint_is_deterministic_per_seed() {
    for seed in [1u64, 2, 3, 99] {
        assert_eq!(misprint_mult(seed), misprint_mult(seed));
    }
    // Different seeds are not all identical.
    let first = misprint_mult(0);
    assert!((1..50).any(|s| misprint_mult(s) != first));
}

#[test]
fn misprint_blueprint_rolls_independently() {
    // [Blueprint, Misprint] rolls twice: extra mult in 0..=46 and reproducible per seed.
    let (p, e) = pair7();
    let run = RunView::default();
    for seed in 0..100 {
        let a = score_seed(&p, e, &mut jokers(&[J::Blueprint, J::Misprint]), &run, seed);
        let b = score_seed(&p, e, &mut jokers(&[J::Blueprint, J::Misprint]), &run, seed);
        assert_eq!(a.score, b.score);
        assert!((0.0..=46.0).contains(&(a.mult - 2.0)));
    }
}

// ----- Fibonacci -----

#[test]
fn fibonacci_ranks() {
    // A 2 3 4 5 straight (A low), all score: chips 30 + 11 + 2 + 3 + 4 + 5 = 55, mult 4.
    // Fibonacci ranks: A, 2, 3, 5 (4 is not) -> +32: 55 x 36 = 1980.
    let p = [
        c(1, Ace, Spades),
        c(2, Two, Hearts),
        c(3, Three, Diamonds),
        c(4, Four, Clubs),
        c(5, Five, Spades),
    ];
    let e = ev(HandType::Straight, 0b11111, &[]);
    assert_eq!(score(&p, e, &mut jokers(&[J::Fibonacci])), 1980.0);
    // 8s: pair 8: 10 + 16 = 26 x (2 + 16) = 468.
    let (p8, e8) = pair_of(Eight);
    assert_eq!(score(&p8, e8, &mut jokers(&[J::Fibonacci])), 468.0);
}

#[test]
fn fibonacci_non_trigger_stone_and_blueprint() {
    // 7s: no trigger: 48.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut jokers(&[J::Fibonacci])), 48.0);
    // Stone card with printed Eight: High Card 5 + 50 = 55 x 1, no +8.
    let stone = [enh(c(1, Eight, Spades), Enhancement::Stone)];
    let he = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score(&stone, he, &mut jokers(&[J::Fibonacci])), 55.0);
    // Blueprint, pair of 2s: 14 chips x (2 + 4 triggers x 8) = 14 x 34 = 476.
    let (p2, e2) = pair_of(Two);
    assert_eq!(
        score(&p2, e2, &mut jokers(&[J::Blueprint, J::Fibonacci])),
        476.0
    );
}

// ----- Steel Joker -----

fn steel_run(n: u32) -> RunView<'static> {
    let mut by_enhancement = [0u32; Enhancement::COUNT];
    by_enhancement[Enhancement::Steel.index()] = n;
    RunView {
        deck: DeckCounts {
            by_enhancement,
            ..DeckCounts::default()
        },
        ..RunView::default()
    }
}

#[test]
fn steel_joker_scales_with_deck_steel_cards() {
    let (p, e) = pair7();
    // 0 Steel: no-op: 48.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::SteelJoker]), &steel_run(0)),
        48.0
    );
    // 2 Steel: X1.4: 24 x 2.8 = 67.2 -> 67.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::SteelJoker]), &steel_run(2)),
        67.0
    );
    // 5 Steel: X2: 24 x 4 = 96.
    assert_eq!(
        score_run(&p, e, &mut jokers(&[J::SteelJoker]), &steel_run(5)),
        96.0
    );
}

#[test]
fn steel_joker_blueprint_and_deck_counts_only_steel() {
    let (p, e) = pair7();
    // [Blueprint, Steel], 5 Steel: X2 twice: 2 x 2 x 2 = 8 -> 192.
    assert_eq!(
        score_run(
            &p,
            e,
            &mut jokers(&[J::Blueprint, J::SteelJoker]),
            &steel_run(5)
        ),
        192.0
    );
    // Other enhancements in the deck do not count.
    let mut by_enhancement = [0u32; Enhancement::COUNT];
    by_enhancement[Enhancement::Gold.index()] = 9;
    let run = RunView {
        deck: DeckCounts {
            by_enhancement,
            ..DeckCounts::default()
        },
        ..RunView::default()
    };
    assert_eq!(score_run(&p, e, &mut jokers(&[J::SteelJoker]), &run), 48.0);
}

// ----- Scary Face -----

#[test]
fn scary_face_faces_only() {
    // K♠ K♥: chips 10 + 10 + 10 + 30 + 30 = 90 x 2 = 180.
    let (k, ke) = pair_of(King);
    assert_eq!(score(&k, ke, &mut jokers(&[J::ScaryFace])), 180.0);
    // Jack and Queen count as faces: J♠ Q♥ with a Pair eval: 10 + 10 + 10 + 60 = 90 x 2 = 180.
    let jq = [c(1, Jack, Spades), c(2, Queen, Hearts)];
    assert_eq!(score(&jq, ke, &mut jokers(&[J::ScaryFace])), 180.0);
    // 7s: not faces: 48. Ace is not a face: 10 + 22 = 32 x 2 = 64.
    let (p, pe) = pair7();
    assert_eq!(score(&p, pe, &mut jokers(&[J::ScaryFace])), 48.0);
    let (a, ae) = pair_of(Ace);
    assert_eq!(score(&a, ae, &mut jokers(&[J::ScaryFace])), 64.0);
}

#[test]
fn scary_face_debuffed_and_blueprint() {
    // Debuffed K♥ is skipped: 10 + 10 + 30 = 50 x 2 = 100.
    let kd = [c(1, King, Spades), debuff(c(2, King, Hearts))];
    let e = ev(HandType::Pair, 0b11, &[]);
    assert_eq!(score(&kd, e, &mut jokers(&[J::ScaryFace])), 100.0);
    // [Blueprint, Scary] on K K: 10 + 20 + 120 = 150 x 2 = 300.
    let (k, ke) = pair_of(King);
    assert_eq!(
        score(&k, ke, &mut jokers(&[J::Blueprint, J::ScaryFace])),
        300.0
    );
}

// ----- Abstract Joker -----

#[test]
fn abstract_joker_counts_all_jokers() {
    let (p, e) = pair7();
    // Alone: +3: 24 x 5 = 120.
    assert_eq!(score(&p, e, &mut jokers(&[J::AbstractJoker])), 120.0);
    // [Abstract, Joker, Joker]: +9, +4, +4 -> 24 x 19 = 456.
    assert_eq!(
        score(&p, e, &mut jokers(&[J::AbstractJoker, J::Joker, J::Joker])),
        456.0
    );
    // [Blueprint, Abstract]: 2 jokers: +6 twice -> 24 x 14 = 336.
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::AbstractJoker])),
        336.0
    );
}

// ----- Even Steven / Odd Todd / Scholar -----

#[test]
fn even_steven_ranks() {
    // 2s: 14 x (2 + 8) = 140.
    let (p2, e2) = pair_of(Two);
    assert_eq!(score(&p2, e2, &mut jokers(&[J::EvenSteven])), 140.0);
    // 10s count as even: 30 x 10 = 300.
    let (p10, e10) = pair_of(Ten);
    assert_eq!(score(&p10, e10, &mut jokers(&[J::EvenSteven])), 300.0);
    // 7s, Aces, Kings are not even: 48 / 64 / 60.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, e7, &mut jokers(&[J::EvenSteven])), 48.0);
    let (a, ae) = pair_of(Ace);
    assert_eq!(score(&a, ae, &mut jokers(&[J::EvenSteven])), 64.0);
    let (k, ke) = pair_of(King);
    assert_eq!(score(&k, ke, &mut jokers(&[J::EvenSteven])), 60.0);
}

#[test]
fn even_steven_stone_and_blueprint() {
    // Stone (printed 2): 5 + 50 = 55 x 1, no bonus.
    let stone = [enh(c(1, Two, Spades), Enhancement::Stone)];
    let he = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score(&stone, he, &mut jokers(&[J::EvenSteven])), 55.0);
    // Blueprint on 2s: 14 x (2 + 4 triggers x 4) = 14 x 18 = 252.
    let (p2, e2) = pair_of(Two);
    assert_eq!(
        score(&p2, e2, &mut jokers(&[J::Blueprint, J::EvenSteven])),
        252.0
    );
}

#[test]
fn odd_todd_ranks() {
    // 7s: (10 + 14 + 62) x 2 = 172.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, e7, &mut jokers(&[J::OddTodd])), 172.0);
    // Aces are odd: (10 + 22 + 62) x 2 = 188.
    let (a, ae) = pair_of(Ace);
    assert_eq!(score(&a, ae, &mut jokers(&[J::OddTodd])), 188.0);
    // 8s and Kings are not: 26 x 2 = 52 / 30 x 2 = 60.
    let (p8, e8) = pair_of(Eight);
    assert_eq!(score(&p8, e8, &mut jokers(&[J::OddTodd])), 52.0);
    let (k, ke) = pair_of(King);
    assert_eq!(score(&k, ke, &mut jokers(&[J::OddTodd])), 60.0);
    // 9s odd: pair of 9s (10 + 18 + 62) x 2 = 180.
    let (p9, e9) = pair_of(Nine);
    assert_eq!(score(&p9, e9, &mut jokers(&[J::OddTodd])), 180.0);
}

#[test]
fn odd_todd_stone_and_blueprint() {
    let stone = [enh(c(1, Seven, Spades), Enhancement::Stone)];
    let he = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score(&stone, he, &mut jokers(&[J::OddTodd])), 55.0);
    // Blueprint on 7s: (10 + 14 + 4 x 31) x 2 = 148 x 2 = 296.
    let (p7, e7) = pair7();
    assert_eq!(
        score(&p7, e7, &mut jokers(&[J::Blueprint, J::OddTodd])),
        296.0
    );
}

#[test]
fn scholar_aces_only() {
    // A♠ A♥: chips 10 + 22 + 40 = 72, mult 2 + 8 = 10 -> 720.
    let (a, ae) = pair_of(Ace);
    assert_eq!(score(&a, ae, &mut jokers(&[J::Scholar])), 720.0);
    // 7s: 48. Kings: 60.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, e7, &mut jokers(&[J::Scholar])), 48.0);
    let (k, ke) = pair_of(King);
    assert_eq!(score(&k, ke, &mut jokers(&[J::Scholar])), 60.0);
    // One Ace and one 7 in a Pair-eval hand: 10 + 11 + 7 + 20 = 48 x 6 = 288.
    let mix = [c(1, Ace, Spades), c(2, Seven, Hearts)];
    assert_eq!(score(&mix, ae, &mut jokers(&[J::Scholar])), 288.0);
}

#[test]
fn scholar_debuffed_stone_and_blueprint() {
    // Debuffed Ace is skipped (no Scholar), the other Ace scores: chips 10 + 11 + 20 = 41, mult 2 + 4 = 6 -> 246.
    let ad = [c(1, Ace, Spades), debuff(c(2, Ace, Hearts))];
    let e = ev(HandType::Pair, 0b11, &[]);
    assert_eq!(score(&ad, e, &mut jokers(&[J::Scholar])), 246.0);
    // Stone Ace: 5 + 50 = 55, no bonus.
    let stone = [enh(c(1, Ace, Spades), Enhancement::Stone)];
    let he = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score(&stone, he, &mut jokers(&[J::Scholar])), 55.0);
    // [Blueprint, Scholar] on A A: chips 10 + 22 + 80 = 112, mult 2 + 16 = 18 -> 2016.
    let (a, ae) = pair_of(Ace);
    assert_eq!(
        score(&a, ae, &mut jokers(&[J::Blueprint, J::Scholar])),
        2016.0
    );
}
