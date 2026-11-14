//! Wave 1b jokers (content/jokers/wave1b.rs): trigger / non-trigger / edge cases / Blueprint.
//! Every expected value is hand-computed in a comment. Base hands at level 1:
//! High Card 5x1, Pair 10x2, Two Pair 20x2, Three of a Kind 30x3, Full House 40x4,
//! Straight 30x4, Flush 35x4, Four of a Kind 60x7, Straight Flush 100x8; card chips:
//! 2-10 face value, J/Q/K 10, A 11, Stone 50 (rank 0).

use balatro_core::card::Card;
use balatro_core::content::jokers::{JokerId, def_count, try_def};
use balatro_core::enums::{Enhancement, HandType, Rank, Rarity, Seal, Suit};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{DeckCounts, RunView, collect_modifiers};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, Scorer};

use JokerId as J;
use Rank::*;
use Suit::*;

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
fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| JokerInstance::new(100 + i as u32, id))
        .collect()
}
fn run_hand(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> f64 {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::new();
    scorer
        .score_hand(ScoreInput {
            played: &mut played,
            held,
            eval,
            jokers: js,
            run,
            rng: &mut rng,
        })
        .score
}
fn score(played: &[Card], held: &[Card], eval: HandEval, ids: &[JokerId]) -> f64 {
    run_hand(
        played,
        held,
        eval,
        &mut jokers(ids),
        &RunView::default(),
        11,
    )
}
fn score_run(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    ids: &[JokerId],
    run: &RunView<'_>,
) -> f64 {
    run_hand(played, held, eval, &mut jokers(ids), run, 11)
}

/// 7♠ 7♥ Pair: 10 + 7 + 7 = 24 chips x 2 = 48 with no jokers.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}
fn pair_of(r: Rank) -> (Vec<Card>, HandEval) {
    (
        vec![c(1, r, Spades), c(2, r, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}
fn deck_with(f: impl FnOnce(&mut DeckCounts)) -> RunView<'static> {
    let mut run = RunView::default();
    f(&mut run.deck);
    run
}
fn deck_enhanced(n: u32) -> RunView<'static> {
    deck_with(|d| {
        d.total = 52;
        d.by_enhancement[Enhancement::None.index()] = 52 - n;
        d.by_enhancement[Enhancement::Bonus.index()] = n;
    })
}

// ----- registry -----

#[test]
fn all_24_defined_once_with_expected_stats() {
    let table: [(JokerId, Rarity, u32); 24] = [
        (J::Blackboard, Rarity::Uncommon, 6),
        (J::BlueJoker, Rarity::Common, 5),
        (J::Photograph, Rarity::Common, 5),
        (J::Erosion, Rarity::Uncommon, 6),
        (J::StoneJoker, Rarity::Uncommon, 6),
        (J::WalkieTalkie, Rarity::Common, 4),
        (J::SmileyFace, Rarity::Common, 4),
        (J::Acrobat, Rarity::Uncommon, 6),
        (J::Swashbuckler, Rarity::Common, 4),
        (J::Bloodstone, Rarity::Uncommon, 7),
        (J::Arrowhead, Rarity::Uncommon, 7),
        (J::OnyxAgate, Rarity::Uncommon, 7),
        (J::FlowerPot, Rarity::Uncommon, 6),
        (J::SeeingDouble, Rarity::Uncommon, 6),
        (J::TheDuo, Rarity::Rare, 8),
        (J::TheTrio, Rarity::Rare, 8),
        (J::TheFamily, Rarity::Rare, 8),
        (J::TheOrder, Rarity::Rare, 8),
        (J::TheTribe, Rarity::Rare, 8),
        (J::Stuntman, Rarity::Rare, 7),
        (J::ShootTheMoon, Rarity::Common, 5),
        (J::DriversLicense, Rarity::Rare, 7),
        (J::Bootstraps, Rarity::Uncommon, 7),
        (J::Triboulet, Rarity::Legendary, 20),
    ];
    for (id, rarity, cost) in table {
        assert_eq!(def_count(id), 1, "{id:?}");
        let d = try_def(id).expect("defined");
        assert_eq!((d.rarity, d.base_cost), (rarity, cost), "{id:?}");
        assert!(d.blueprint_compatible && d.eternal_compatible && d.perishable_compatible);
    }
}

// ----- Blackboard -----

#[test]
fn blackboard() {
    let (p, e) = pair7();
    let b = [J::Blackboard];
    // All held Spades/Clubs: 24 x (2 x 3) = 144.
    assert_eq!(
        score(&p, &[c(3, King, Spades), c(4, Three, Clubs)], e, &b),
        144.0
    );
    // One Heart held: 48.
    assert_eq!(
        score(&p, &[c(3, King, Spades), c(4, Three, Hearts)], e, &b),
        48.0
    );
    // Empty held hand counts: 144.
    assert_eq!(score(&p, &[], e, &b), 144.0);
    // Wild Heart counts as Spade/Club: 144.
    assert_eq!(
        score(&p, &[enh(c(3, Two, Hearts), Enhancement::Wild)], e, &b),
        144.0
    );
    // Stone blocks: 48.
    assert_eq!(
        score(&p, &[enh(c(3, Two, Spades), Enhancement::Stone)], e, &b),
        48.0
    );
    // Blueprint: 24 x (2 x 3 x 3) = 432.
    assert_eq!(
        score(&p, &[c(3, King, Spades)], e, &[J::Blueprint, J::Blackboard]),
        432.0
    );
}

// ----- Blue Joker -----

#[test]
fn blue_joker() {
    let (p, e) = pair7();
    // 30 cards in the draw pile: +60 chips: (24 + 60) x 2 = 168.
    let run = deck_with(|d| d.draw_pile = 30);
    assert_eq!(score_run(&p, &[], e, &[J::BlueJoker], &run), 168.0);
    // Empty draw pile: 48.
    assert_eq!(score(&p, &[], e, &[J::BlueJoker]), 48.0);
    // Blueprint: +120 chips: (24 + 120) x 2 = 288.
    assert_eq!(
        score_run(&p, &[], e, &[J::Blueprint, J::BlueJoker], &run),
        288.0
    );
}

// ----- Photograph -----

#[test]
fn photograph() {
    let ph = [J::Photograph];
    // K♠ K♥ Pair: chips 10 + 10 + 10 = 30; only the first face x2: 30 x 4 = 120.
    let (p, e) = pair_of(King);
    assert_eq!(score(&p, &[], e, &ph), 120.0);
    // No face: 48.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, &[], e7, &ph), 48.0);
    // Red Seal on the first K: it triggers twice, x2 each: chips 40, mult 2 x2 x2 = 8 -> 320.
    let pr = [seal(c(1, King, Spades), Seal::Red), c(2, King, Hearts)];
    assert_eq!(score(&pr, &[], e, &ph), 320.0);
    // First K debuffed: the second K is the first scoring face: chips 10 + 10 = 20, mult 4 -> 80.
    let pd = [debuff(c(1, King, Spades)), c(2, King, Hearts)];
    assert_eq!(score(&pd, &[], e, &ph), 80.0);
    // Pareidolia: every card is a face; 7♠ is first: 24 x 4 = 96.
    let mut run = RunView::default();
    run.rules.pareidolia = true;
    assert_eq!(score_run(&p7, &[], e7, &ph, &run), 96.0);
    // Pareidolia + Stone first: chips 5 + 50 = 55, x2 -> 110.
    let ps = [enh(c(1, Two, Spades), Enhancement::Stone)];
    let es = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score_run(&ps, &[], es, &ph, &run), 110.0);
    // Non-scoring face (kicker) does not count: only 7♥ scores: 5 + 7 = 12 x 1.
    let pk = [c(1, King, Spades), c(2, Seven, Hearts)];
    let ek = ev(HandType::HighCard, 0b10, &[]);
    assert_eq!(score(&pk, &[], ek, &ph), 12.0);
    // Blueprint: first K gets x2 twice: 30 x (2 x 4) = 240.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::Photograph]), 240.0);
}

// ----- Erosion -----

#[test]
fn erosion() {
    let (p, e) = pair7();
    let ero = [J::Erosion];
    // 50 cards of 52: 2 below -> +8: 24 x 10 = 240.
    let run = deck_with(|d| d.total = 50);
    assert_eq!(score_run(&p, &[], e, &ero, &run), 240.0);
    // At or above the starting size: 48.
    let run52 = deck_with(|d| d.total = 52);
    assert_eq!(score_run(&p, &[], e, &ero, &run52), 48.0);
    let run60 = deck_with(|d| d.total = 60);
    assert_eq!(score_run(&p, &[], e, &ero, &run60), 48.0);
    // Abandoned Deck (start 40), 36 cards: 4 below -> +16: 24 x 18 = 432.
    let ab = deck_with(|d| {
        d.starting_size = 40;
        d.total = 36;
    });
    assert_eq!(score_run(&p, &[], e, &ero, &ab), 432.0);
    // Blueprint: +8 twice: 24 x 18 = 432.
    assert_eq!(
        score_run(&p, &[], e, &[J::Blueprint, J::Erosion], &run),
        432.0
    );
}

// ----- Stone Joker -----

#[test]
fn stone_joker() {
    let (p, e) = pair7();
    // 3 Stone cards: +75 chips: 99 x 2 = 198.
    let run = deck_with(|d| d.by_enhancement[Enhancement::Stone.index()] = 3);
    assert_eq!(score_run(&p, &[], e, &[J::StoneJoker], &run), 198.0);
    assert_eq!(score(&p, &[], e, &[J::StoneJoker]), 48.0);
    // Blueprint: +150: 174 x 2 = 348.
    assert_eq!(
        score_run(&p, &[], e, &[J::Blueprint, J::StoneJoker], &run),
        348.0
    );
}

// ----- Walkie Talkie -----

#[test]
fn walkie_talkie() {
    let w = [J::WalkieTalkie];
    // 10♠ 10♥ Pair: chips 10 + 10 + 10 + 2 x 10 = 50, mult 2 + 2 x 4 = 10 -> 500.
    let (p, e) = pair_of(Ten);
    assert_eq!(score(&p, &[], e, &w), 500.0);
    // 4♠ scoring: chips 5 + 4 + 10 = 19, mult 1 + 4 = 5 -> 95.
    let p4 = [c(1, Four, Spades), c(2, Seven, Hearts)];
    let e4 = ev(HandType::HighCard, 0b01, &[]);
    assert_eq!(score(&p4, &[], e4, &w), 95.0);
    // 7s: 48.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, &[], e7, &w), 48.0);
    // Stone with printed rank Ten: only Stone's 50: 5 + 50 = 55 x 1.
    let ps = [enh(c(1, Ten, Spades), Enhancement::Stone)];
    let es = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(score(&ps, &[], es, &w), 55.0);
    // Non-scoring 4 (kicker) does not trigger: only 7♥ scores: 5 + 7 = 12.
    let ek = ev(HandType::HighCard, 0b10, &[]);
    assert_eq!(score(&p4, &[], ek, &w), 12.0);
    // Blueprint: 4♠: mult 1 + 8 = 9, chips 5 + 4 + 20 = 29 -> 261.
    assert_eq!(score(&p4, &[], e4, &[J::Blueprint, J::WalkieTalkie]), 261.0);
}

// ----- Smiley Face -----

#[test]
fn smiley_face() {
    let s = [J::SmileyFace];
    // K♠ K♥ Pair: chips 30, mult 2 + 10 = 12 -> 360.
    let (p, e) = pair_of(King);
    assert_eq!(score(&p, &[], e, &s), 360.0);
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, &[], e7, &s), 48.0);
    // First K debuffed: chips 20, mult 2 + 5 = 7 -> 140.
    let pd = [debuff(c(1, King, Spades)), c(2, King, Hearts)];
    assert_eq!(score(&pd, &[], e, &s), 140.0);
    // Pareidolia on 7s: 24 x 12 = 288.
    let mut run = RunView::default();
    run.rules.pareidolia = true;
    assert_eq!(score_run(&p7, &[], e7, &s, &run), 288.0);
    // Blueprint: mult 2 + 20 = 22, chips 30 -> 660.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::SmileyFace]), 660.0);
}

// ----- Acrobat -----

#[test]
fn acrobat() {
    let (p, e) = pair7();
    let mut run = RunView {
        hands_left: 0,
        ..RunView::default()
    };
    // hands_left excludes the current hand: 0 = final hand: 24 x (2 x 3) = 144.
    assert_eq!(score_run(&p, &[], e, &[J::Acrobat], &run), 144.0);
    run.hands_left = 1;
    assert_eq!(score_run(&p, &[], e, &[J::Acrobat], &run), 48.0);
    // Blueprint on the final hand: 24 x 18 = 432.
    run.hands_left = 0;
    assert_eq!(
        score_run(&p, &[], e, &[J::Blueprint, J::Acrobat], &run),
        432.0
    );
}

// ----- Swashbuckler -----

#[test]
fn swashbuckler() {
    let (p, e) = pair7();
    // Alone: no others -> 48.
    assert_eq!(score(&p, &[], e, &[J::Swashbuckler]), 48.0);
    // Default sell values: Swashbuckler 2 (self, excluded), Joker 1, Greedy 2 (cost 5 -> 2).
    // Swash +3, Joker +4: mult 2 + 3 + 4 = 9 -> 24 x 9 = 216.
    assert_eq!(
        score(&p, &[], e, &[J::Swashbuckler, J::Joker, J::GreedyJoker]),
        216.0
    );
    // Explicit sell values: Joker 10, Swash 99 (own, ignored), Greedy 7.
    // Swash +17, Joker +4: mult 2 + 4 + 17 = 23 -> 24 x 23 = 552.
    let mut js = jokers(&[J::Joker, J::Swashbuckler, J::GreedyJoker]);
    js[0].sell_value = 10;
    js[1].sell_value = 99;
    js[2].sell_value = 7;
    assert_eq!(run_hand(&p, &[], e, &mut js, &RunView::default(), 1), 552.0);
    // Blueprint copy runs as the copied slot (1): each execution sums the others of slot 1 =
    // Blueprint's sell value 5 (cost 10): mult 2 + 5 + 5 = 12 -> 288.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::Swashbuckler]), 288.0);
}

// ----- Bloodstone -----

#[test]
fn bloodstone() {
    let b = [J::Bloodstone];
    let always = RunView {
        prob_mult: 1000,
        ..RunView::default()
    };
    let never = RunView {
        prob_mult: 0,
        ..RunView::default()
    };
    // 7♥ 7♥ both trigger: mult 2 x 1.5 x 1.5 = 4.5 -> 24 x 4.5 = 108.
    let hh = [c(1, Seven, Hearts), c(2, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b11, &[]);
    assert_eq!(score_run(&hh, &[], e, &b, &always), 108.0);
    assert_eq!(score_run(&hh, &[], e, &b, &never), 48.0);
    // One Heart (7♠ 7♥): 24 x 3 = 72.
    let (p, e7) = pair7();
    assert_eq!(score_run(&p, &[], e7, &b, &always), 72.0);
    // No heart (7♠ 7♣): 48 even with certainty.
    let nh = [c(1, Seven, Spades), c(2, Seven, Clubs)];
    assert_eq!(score_run(&nh, &[], e7, &b, &always), 48.0);
    // Wild counts as a Heart: 7♠ Wild + 7♣: 72.
    let w = [
        enh(c(1, Seven, Spades), Enhancement::Wild),
        c(2, Seven, Clubs),
    ];
    assert_eq!(score_run(&w, &[], e7, &b, &always), 72.0);
    // Debuffed Heart is skipped: 17 x 2 = 34.
    let d = [debuff(c(1, Seven, Hearts)), c(2, Seven, Spades)];
    assert_eq!(score_run(&d, &[], e7, &b, &always), 34.0);
    // Oops! All 6s: prob_mult 2 makes 1 in 2 certain, for every seed.
    let oops = RunView {
        prob_mult: 2,
        ..RunView::default()
    };
    for seed in 0..40 {
        let s = run_hand(&hh, &[], e, &mut jokers(&b), &oops, seed);
        assert_eq!(s, 108.0, "seed {seed}");
    }
    // Plain odds: deterministic per seed, both outcomes (72 / 48) appear over seeds.
    let plain = RunView::default();
    let mut seen = [false; 2];
    for seed in 0..64 {
        let s1 = run_hand(&p, &[], e7, &mut jokers(&b), &plain, seed);
        let s2 = run_hand(&p, &[], e7, &mut jokers(&b), &plain, seed);
        assert_eq!(s1, s2);
        assert!(s1 == 72.0 || s1 == 48.0);
        seen[usize::from(s1 == 72.0)] = true;
    }
    assert!(seen[0] && seen[1]);
    // Blueprint with certainty: both copies trigger on each Heart: 2 x 1.5^4 = 10.125 -> 243.
    assert_eq!(
        score_run(&hh, &[], e, &[J::Blueprint, J::Bloodstone], &always),
        243.0
    );
}

// ----- Arrowhead / Onyx Agate -----

#[test]
fn arrowhead() {
    let a = [J::Arrowhead];
    let e = ev(HandType::Pair, 0b11, &[]);
    // 7♠ 7♠: +100 chips: 124 x 2 = 248.
    let ss = [c(1, Seven, Spades), c(2, Seven, Spades)];
    assert_eq!(score(&ss, &[], e, &a), 248.0);
    // 7♠ 7♥: 74 x 2 = 148.
    let (p, _) = pair7();
    assert_eq!(score(&p, &[], e, &a), 148.0);
    // No spade: 48.
    let hh = [c(1, Seven, Hearts), c(2, Seven, Hearts)];
    assert_eq!(score(&hh, &[], e, &a), 48.0);
    // Wild counts: 7♥ Wild + 7♥: 74 x 2 = 148.
    let w = [
        enh(c(1, Seven, Hearts), Enhancement::Wild),
        c(2, Seven, Hearts),
    ];
    assert_eq!(score(&w, &[], e, &a), 148.0);
    // Debuffed spade skipped: 17 x 2 = 34.
    let d = [debuff(c(1, Seven, Spades)), c(2, Seven, Hearts)];
    assert_eq!(score(&d, &[], e, &a), 34.0);
    // Blueprint: +100 for the one spade: 124 x 2 = 248.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::Arrowhead]), 248.0);
}

#[test]
fn onyx_agate() {
    let o = [J::OnyxAgate];
    let e = ev(HandType::Pair, 0b11, &[]);
    // 7♣ 7♥: mult 2 + 7 = 9 -> 24 x 9 = 216.
    let ch = [c(1, Seven, Clubs), c(2, Seven, Hearts)];
    assert_eq!(score(&ch, &[], e, &o), 216.0);
    // 7♣ 7♣: 2 + 14 = 16 -> 384.
    let cc = [c(1, Seven, Clubs), c(2, Seven, Clubs)];
    assert_eq!(score(&cc, &[], e, &o), 384.0);
    let (p, _) = pair7();
    assert_eq!(score(&p, &[], e, &o), 48.0);
    // Smeared: 7♠ counts as a Club: 2 + 7 = 9 -> 216.
    let mut run = RunView::default();
    run.rules.smeared = true;
    assert_eq!(score_run(&p, &[], e, &o, &run), 216.0);
    // Blueprint: one Club, +7 twice: 2 + 14 = 16 -> 384.
    assert_eq!(score(&ch, &[], e, &[J::Blueprint, J::OnyxAgate]), 384.0);
}

// ----- Flower Pot -----

fn four_cards(cards: [Card; 4]) -> (Vec<Card>, HandEval) {
    (cards.to_vec(), ev(HandType::TwoPair, 0b1111, &[]))
}

#[test]
fn flower_pot() {
    // Two Pair 2♠ 2♥ 3♦ 3♣ (all four suits): base 20 x 2, chips 20 + 10 = 30,
    // mult 2 x 3 = 6 -> 180.
    let fp = [J::FlowerPot];
    let (p, e) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Hearts),
        c(3, Three, Diamonds),
        c(4, Three, Clubs),
    ]);
    assert_eq!(score(&p, &[], e, &fp), 180.0);
    // Missing a suit (♠ ♥ ♦ ♦): 30 x 2 = 60.
    let (pm, em) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Hearts),
        c(3, Three, Diamonds),
        c(4, Three, Diamonds),
    ]);
    assert_eq!(score(&pm, &[], em, &fp), 60.0);
    // One Wild fills the missing suit: 180.
    let (pw, ew) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Hearts),
        c(3, Three, Diamonds),
        enh(c(4, Three, Spades), Enhancement::Wild),
    ]);
    assert_eq!(score(&pw, &[], ew, &fp), 180.0);
    // Four Wilds: 180. Two Wilds + two distinct suits: 180.
    let wild = |u, r| enh(c(u, r, Spades), Enhancement::Wild);
    let (p4, e4) = four_cards([wild(1, Two), wild(2, Two), wild(3, Three), wild(4, Three)]);
    assert_eq!(score(&p4, &[], e4, &fp), 180.0);
    let (p22, e22) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Hearts),
        wild(3, Three),
        wild(4, Three),
    ]);
    assert_eq!(score(&p22, &[], e22, &fp), 180.0);
    // Two Wilds + two of the same suit (♠ ♠ W W): needs 3 more suits: 60.
    let (pf, ef) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Spades),
        wild(3, Three),
        wild(4, Three),
    ]);
    assert_eq!(score(&pf, &[], ef, &fp), 60.0);
    // Only 3 scoring cards (mask 0b0111, Three of a Kind 30 x 3): no trigger.
    // Chips 30 + 2 + 2 + 3 = 37 -> 37 x 3 = 111.
    let three = ev(HandType::ThreeOfAKind, 0b0111, &[]);
    assert_eq!(score(&pw, &[], three, &fp), 111.0);
    // A debuffed card does not count: 3 usable cards, and it gives no chips:
    // chips 20 + 2 + 3 + 3 = 28, mult 2 -> 56.
    let (pd, ed) = four_cards([
        debuff(c(1, Two, Spades)),
        c(2, Two, Hearts),
        c(3, Three, Diamonds),
        c(4, Three, Clubs),
    ]);
    assert_eq!(score(&pd, &[], ed, &fp), 56.0);
    // Smeared: ♠ ♠ ♥ ♥ = two of each colour -> triggers: 30 x 6 = 180. Without Smeared: 60.
    let (ps, es) = four_cards([
        c(1, Two, Spades),
        c(2, Two, Spades),
        c(3, Three, Hearts),
        c(4, Three, Hearts),
    ]);
    let mut run = RunView::default();
    run.rules.smeared = true;
    assert_eq!(score_run(&ps, &[], es, &fp, &run), 180.0);
    assert_eq!(score(&ps, &[], es, &fp), 60.0);
    // Blueprint: 30 x (2 x 3 x 3) = 540.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::FlowerPot]), 540.0);
}

// ----- Seeing Double -----

#[test]
fn seeing_double() {
    let sd = [J::SeeingDouble];
    let e = ev(HandType::Pair, 0b11, &[]);
    // 7♣ 7♥: 24 x (2 x 2) = 96.
    let ch = [c(1, Seven, Clubs), c(2, Seven, Hearts)];
    assert_eq!(score(&ch, &[], e, &sd), 96.0);
    // Two Clubs only: 48. No Club: 48.
    let cc = [c(1, Seven, Clubs), c(2, Seven, Clubs)];
    assert_eq!(score(&cc, &[], e, &sd), 48.0);
    let (p, _) = pair7();
    assert_eq!(score(&p, &[], e, &sd), 48.0);
    // Wild + 7♠: Wild is the Club, 7♠ the other: 96.
    let w = [
        enh(c(1, Seven, Hearts), Enhancement::Wild),
        c(2, Seven, Spades),
    ];
    assert_eq!(score(&w, &[], e, &sd), 96.0);
    // Two Wilds: 96.
    let ww = [
        enh(c(1, Seven, Hearts), Enhancement::Wild),
        enh(c(2, Seven, Hearts), Enhancement::Wild),
    ];
    assert_eq!(score(&ww, &[], e, &sd), 96.0);
    // A single Wild alone (High Card 5 + 7 = 12 x 1) cannot be both: no trigger.
    let one = [enh(c(1, Seven, Hearts), Enhancement::Wild)];
    assert_eq!(
        score(&one, &[], ev(HandType::HighCard, 0b1, &[]), &sd),
        12.0
    );
    // Wild + Stone: Stone has no suit -> no trigger. High Card: 5 + 7 + 50 = 62 x 1.
    let ws = [
        enh(c(1, Seven, Hearts), Enhancement::Wild),
        enh(c(2, Two, Clubs), Enhancement::Stone),
    ];
    assert_eq!(
        score(&ws, &[], ev(HandType::HighCard, 0b11, &[]), &sd),
        62.0
    );
    // Debuffed Club does not count: chips 10 + 7 = 17 x 2 = 34.
    let dc = [debuff(c(1, Seven, Clubs)), c(2, Seven, Hearts)];
    assert_eq!(score(&dc, &[], e, &sd), 34.0);
    // The Club must be a scoring card: kicker 7♣ ignored (High Card 7♥ only): 12.
    assert_eq!(
        score(&ch, &[], ev(HandType::HighCard, 0b10, &[]), &sd),
        12.0
    );
    // Blueprint: 24 x (2 x 2 x 2) = 192.
    assert_eq!(score(&ch, &[], e, &[J::Blueprint, J::SeeingDouble]), 192.0);
}

// ----- The Duo / Trio / Family / Order / Tribe -----

#[test]
fn duo_trio_family_order_tribe() {
    // Pair 7♠ 7♥: Duo 24 x 4 = 96; the others do nothing.
    let (p, e) = pair7();
    assert_eq!(score(&p, &[], e, &[J::TheDuo]), 96.0);
    for id in [J::TheTrio, J::TheFamily, J::TheOrder, J::TheTribe] {
        assert_eq!(score(&p, &[], e, &[id]), 48.0, "{id:?}");
    }
    // High Card 7♥: no Duo -> 5 + 7 = 12.
    let hc = [c(1, Seven, Hearts)];
    assert_eq!(
        score(&hc, &[], ev(HandType::HighCard, 0b1, &[]), &[J::TheDuo]),
        12.0
    );
    // Full House 7♠ 7♥ 7♦ 4♣ 4♠: chips 40 + 21 + 8 = 69; mult 4; contains Three + Pair:
    // Duo x2, Trio x3 -> 4 x 6 = 24 -> 1656.
    let fh = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Seven, Diamonds),
        c(4, Four, Clubs),
        c(5, Four, Spades),
    ];
    let efh = ev(
        HandType::FullHouse,
        0b11111,
        &[HandType::ThreeOfAKind, HandType::Pair],
    );
    assert_eq!(score(&fh, &[], efh, &[J::TheDuo, J::TheTrio]), 1656.0);
    // Four of a Kind 7777: chips 60 + 28 = 88; mult 7 x 4 (Family) x 3 (Trio) x 2 (Duo) = 168
    // -> 14784.
    let fk = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Seven, Diamonds),
        c(4, Seven, Clubs),
    ];
    let efk = ev(
        HandType::FourOfAKind,
        0b1111,
        &[HandType::ThreeOfAKind, HandType::Pair],
    );
    assert_eq!(
        score(&fk, &[], efk, &[J::TheFamily, J::TheTrio, J::TheDuo]),
        14784.0
    );
    // Straight 3-4-5-6-7: chips 30 + 25 = 55, mult 4 x 3 = 12 -> 660.
    let st = [
        c(1, Three, Spades),
        c(2, Four, Hearts),
        c(3, Five, Spades),
        c(4, Six, Spades),
        c(5, Seven, Spades),
    ];
    let est = ev(HandType::Straight, 0b11111, &[]);
    assert_eq!(score(&st, &[], est, &[J::TheOrder]), 660.0);
    // Tribe on a Straight that is not a Flush: 55 x 4 = 220.
    assert_eq!(score(&st, &[], est, &[J::TheTribe]), 220.0);
    // Straight Flush 3-7 spades: chips 100 + 25 = 125, mult 8 x 3 (Order) x 2 (Tribe) = 48
    // -> 6000.
    let sf: Vec<Card> = [Three, Four, Five, Six, Seven]
        .iter()
        .enumerate()
        .map(|(i, &r)| c(i as u32 + 1, r, Spades))
        .collect();
    let esf = ev(
        HandType::StraightFlush,
        0b11111,
        &[HandType::Straight, HandType::Flush],
    );
    assert_eq!(score(&sf, &[], esf, &[J::TheOrder, J::TheTribe]), 6000.0);
    // Flush 2,4,6,8,10 hearts: chips 35 + 30 = 65, mult 4 x 2 = 8 -> 520.
    let fl: Vec<Card> = [Two, Four, Six, Eight, Ten]
        .iter()
        .enumerate()
        .map(|(i, &r)| c(i as u32 + 1, r, Hearts))
        .collect();
    let efl = ev(HandType::Flush, 0b11111, &[]);
    assert_eq!(score(&fl, &[], efl, &[J::TheTribe]), 520.0);
    // Blueprint copy of Duo: 24 x (2 x 2 x 2) = 192.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::TheDuo]), 192.0);
    // Blueprint copy of Tribe on a Flush: 65 x (4 x 2 x 2) = 1040.
    assert_eq!(score(&fl, &[], efl, &[J::Blueprint, J::TheTribe]), 1040.0);
}

// ----- Stuntman -----

#[test]
fn stuntman() {
    let (p, e) = pair7();
    // +250 chips: (24 + 250) x 2 = 548.
    assert_eq!(score(&p, &[], e, &[J::Stuntman]), 548.0);
    // Blueprint copies the chips: (24 + 500) x 2 = 1048.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::Stuntman]), 1048.0);
    // Hand size -2 per Stuntman; Blueprint does not copy passives.
    assert_eq!(collect_modifiers(&jokers(&[J::Stuntman])).hand_size, -2);
    assert_eq!(
        collect_modifiers(&jokers(&[J::Stuntman, J::Stuntman])).hand_size,
        -4
    );
    assert_eq!(
        collect_modifiers(&jokers(&[J::Blueprint, J::Stuntman])).hand_size,
        -2
    );
    // Debuffed: no hand size change and no chips (48).
    let dbg = vec![JokerInstance::new(1, J::Stuntman).with_debuff(true)];
    assert_eq!(collect_modifiers(&dbg).hand_size, 0);
    let mut js = jokers(&[J::Stuntman]);
    js[0].debuffed = true;
    assert_eq!(run_hand(&p, &[], e, &mut js, &RunView::default(), 1), 48.0);
}

// ----- Shoot the Moon -----

#[test]
fn shoot_the_moon() {
    let (p, e) = pair7();
    let s = [J::ShootTheMoon];
    // Two held Queens: +26 -> 24 x 28 = 672.
    let held = [c(3, Queen, Spades), c(4, Queen, Hearts), c(5, King, Clubs)];
    assert_eq!(score(&p, &held, e, &s), 672.0);
    // No Queen: 48.
    assert_eq!(score(&p, &[c(3, King, Clubs)], e, &s), 48.0);
    // Debuffed Queen: skipped -> 48.
    assert_eq!(score(&p, &[debuff(c(3, Queen, Spades))], e, &s), 48.0);
    // Stone with printed rank Queen: no effect -> 48.
    assert_eq!(
        score(&p, &[enh(c(3, Queen, Spades), Enhancement::Stone)], e, &s),
        48.0
    );
    // Queen with Red Seal triggers twice: +26 -> 672.
    assert_eq!(
        score(&p, &[seal(c(3, Queen, Spades), Seal::Red)], e, &s),
        672.0
    );
    // Blueprint with one Queen: +13 twice = +26 -> 672.
    assert_eq!(
        score(
            &p,
            &[c(3, Queen, Spades)],
            e,
            &[J::Blueprint, J::ShootTheMoon]
        ),
        672.0
    );
}

// ----- Driver's License -----

#[test]
fn drivers_license() {
    let (p, e) = pair7();
    let d = [J::DriversLicense];
    // 16 enhanced: 24 x (2 x 3) = 144; 20: 144; 15: 48; 0: 48.
    assert_eq!(score_run(&p, &[], e, &d, &deck_enhanced(16)), 144.0);
    assert_eq!(score_run(&p, &[], e, &d, &deck_enhanced(20)), 144.0);
    assert_eq!(score_run(&p, &[], e, &d, &deck_enhanced(15)), 48.0);
    assert_eq!(score_run(&p, &[], e, &d, &deck_enhanced(0)), 48.0);
    // Blueprint: 24 x 18 = 432.
    assert_eq!(
        score_run(
            &p,
            &[],
            e,
            &[J::Blueprint, J::DriversLicense],
            &deck_enhanced(16)
        ),
        432.0
    );
}

// ----- Bootstraps -----

#[test]
fn bootstraps() {
    let (p, e) = pair7();
    let b = [J::Bootstraps];
    let money = |m: i32| RunView {
        money: m,
        ..RunView::default()
    };
    // $12 -> 2 steps -> +4: 24 x 6 = 144.
    assert_eq!(score_run(&p, &[], e, &b, &money(12)), 144.0);
    // $4 -> 0; $5 -> +2 (24 x 4 = 96); $25 -> +10 (24 x 12 = 288).
    assert_eq!(score_run(&p, &[], e, &b, &money(4)), 48.0);
    assert_eq!(score_run(&p, &[], e, &b, &money(5)), 96.0);
    assert_eq!(score_run(&p, &[], e, &b, &money(25)), 288.0);
    // Debt or zero: nothing.
    assert_eq!(score_run(&p, &[], e, &b, &money(-3)), 48.0);
    assert_eq!(score_run(&p, &[], e, &b, &money(0)), 48.0);
    // Money gained earlier in the hand counts: $3 + Gold Seal $3 = $6 -> +2: 24 x 4 = 96.
    let pg = [seal(c(1, Seven, Spades), Seal::Gold), c(2, Seven, Hearts)];
    assert_eq!(score_run(&pg, &[], e, &b, &money(3)), 96.0);
    // Blueprint: +4 twice at $12: 2 + 8 = 10 -> 240.
    assert_eq!(
        score_run(&p, &[], e, &[J::Blueprint, J::Bootstraps], &money(12)),
        240.0
    );
}

// ----- Triboulet -----

#[test]
fn triboulet() {
    let t = [J::Triboulet];
    // K♠ K♥: chips 30, mult 2 x 2 x 2 = 8 -> 240.
    let (p, e) = pair_of(King);
    assert_eq!(score(&p, &[], e, &t), 240.0);
    // K♠ Q♥ (High Card, both scoring by mask): chips 5 + 20 = 25, mult 1 x 4 = 4 -> 100.
    let kq = [c(1, King, Spades), c(2, Queen, Hearts)];
    assert_eq!(
        score(&kq, &[], ev(HandType::HighCard, 0b11, &[]), &t),
        100.0
    );
    // Jacks do not count: J♠ J♥ chips 30 x 2 = 60.
    let (pj, ej) = pair_of(Jack);
    assert_eq!(score(&pj, &[], ej, &t), 60.0);
    // Stone with printed rank King: 5 + 50 = 55 x 1.
    let ps = [enh(c(1, King, Spades), Enhancement::Stone)];
    assert_eq!(score(&ps, &[], ev(HandType::HighCard, 0b1, &[]), &t), 55.0);
    // Red Seal K: 3 triggers in total (K1 twice, K2 once): chips 10 + 10 + 10 + 10 = 40,
    // mult 2 x 2^3 = 16 -> 640.
    let pr = [seal(c(1, King, Spades), Seal::Red), c(2, King, Hearts)];
    assert_eq!(score(&pr, &[], e, &t), 640.0);
    // Blueprint: each K gives x2 x2: mult 2 x 4 x 4 = 32, chips 30 -> 960.
    assert_eq!(score(&p, &[], e, &[J::Blueprint, J::Triboulet]), 960.0);
    // Pareidolia does not matter (rank-based): 7s unchanged.
    let (p7, e7) = pair7();
    let mut run = RunView::default();
    run.rules.pareidolia = true;
    assert_eq!(score_run(&p7, &[], e7, &t, &run), 48.0);
}
