//! Integration: real `hand_eval::evaluate` output fed into the scoring pipeline (1B + 1C were
//! developed in parallel against a fixed interface; this checks they agree end to end).

use balatro_core::card::{Card, RuleFlags};
use balatro_core::enums::{Enhancement, HandType, Rank, Suit};
use balatro_core::hand_eval::{evaluate, evaluate_all_subsets};
use balatro_core::hooks::RunView;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, Scorer};

use Rank::*;
use Suit::*;

fn score(played: &[Card], held: &[Card], jokers: &mut [JokerInstance]) -> (HandType, f64) {
    let run = RunView::default();
    let eval = evaluate(played, &run.rules);
    let mut played = played.to_vec();
    let mut rng = GameRng::new(1);
    let mut scorer = Scorer::with_trace(false);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers,
        run: &run,
        rng: &mut rng,
    });
    (res.hand_type, res.score)
}

#[test]
fn pair_with_kickers_scores_only_the_pair() {
    // Pair of 7s + K + 2 kickers: (10 + 7 + 7) x 2 = 48.
    let played = [
        Card::new(1, King, Spades),
        Card::new(2, Seven, Hearts),
        Card::new(3, Seven, Clubs),
        Card::new(4, Two, Diamonds),
    ];
    assert_eq!(score(&played, &[], &mut []), (HandType::Pair, 48.0));
}

#[test]
fn flush_scores_all_five() {
    // Flush: (35 + 2+5+9+10+11) x 4 = 72 x 4 = 288.
    let played = [
        Card::new(1, Ace, Hearts),
        Card::new(2, Jack, Hearts),
        Card::new(3, Nine, Hearts),
        Card::new(4, Five, Hearts),
        Card::new(5, Two, Hearts),
    ];
    assert_eq!(score(&played, &[], &mut []), (HandType::Flush, 288.0));
}

#[test]
fn stone_kicker_scores_with_high_card() {
    // High Card A + Stone: (5 + 11 + 50) x 1 = 66.
    let mut stone = Card::new(2, Three, Clubs);
    stone.enhancement = Enhancement::Stone;
    let played = [Card::new(1, Ace, Spades), stone];
    assert_eq!(score(&played, &[], &mut []), (HandType::HighCard, 66.0));
}

#[test]
fn best_subset_of_eight_matches_direct_scoring() {
    // Every subset from evaluate_all_subsets scores identically to scoring it directly.
    let hand = [
        Card::new(1, Ace, Spades),
        Card::new(2, Ace, Hearts),
        Card::new(3, King, Spades),
        Card::new(4, Queen, Spades),
        Card::new(5, Jack, Spades),
        Card::new(6, Ten, Spades),
        Card::new(7, Four, Diamonds),
        Card::new(8, Two, Clubs),
    ];
    let mut subsets = Vec::new();
    evaluate_all_subsets(&hand, &RuleFlags::default(), &mut subsets);
    assert_eq!(subsets.len(), 218);
    let mut best = (0u16, 0.0f64);
    for &(mask, eval) in &subsets {
        let played: Vec<Card> = (0..hand.len())
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| hand[i])
            .collect();
        assert_eq!(evaluate(&played, &RuleFlags::default()), eval);
        let (_, s) = score(&played, &[], &mut []);
        if s > best.1 {
            best = (mask, s);
        }
    }
    // Royal flush A-K-Q-J-10 of spades: (100 + 11+10+10+10+10) x 8 = 1208.
    assert_eq!(best.1, 1208.0);
    assert_eq!(best.0, 0b0011_1101);
}
