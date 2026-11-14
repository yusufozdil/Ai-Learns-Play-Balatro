//! Tests for poker hand evaluation (`balatro_core::hand_eval`).

use balatro_core::card::{Card, RuleFlags};
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::hand_eval::{HandEval, evaluate, evaluate_all_subsets};
use proptest::prelude::*;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn rank_of(s: &str) -> Rank {
    match s {
        "2" => Rank::Two,
        "3" => Rank::Three,
        "4" => Rank::Four,
        "5" => Rank::Five,
        "6" => Rank::Six,
        "7" => Rank::Seven,
        "8" => Rank::Eight,
        "9" => Rank::Nine,
        "10" | "T" => Rank::Ten,
        "J" => Rank::Jack,
        "Q" => Rank::Queen,
        "K" => Rank::King,
        "A" => Rank::Ace,
        other => panic!("bad rank {other}"),
    }
}

fn suit_of(c: char) -> Suit {
    match c {
        'S' => Suit::Spades,
        'H' => Suit::Hearts,
        'C' => Suit::Clubs,
        'D' => Suit::Diamonds,
        other => panic!("bad suit {other}"),
    }
}

/// Parse "10H AS 3Dw KCs 2Sd": rank, uppercase suit letter, then optional lowercase modifiers:
/// `w` Wild, `s` Stone, `d` debuffed.
fn hand(s: &str) -> Vec<Card> {
    s.split_whitespace()
        .enumerate()
        .map(|(i, tok)| {
            let split = tok.find(['S', 'H', 'C', 'D']).expect("suit letter");
            let mut card = Card::new(
                i as u32,
                rank_of(&tok[..split]),
                suit_of(tok.as_bytes()[split] as char),
            );
            for m in tok[split + 1..].chars() {
                match m {
                    'w' => card.enhancement = Enhancement::Wild,
                    's' => card.enhancement = Enhancement::Stone,
                    'd' => card.debuffed = true,
                    other => panic!("bad modifier {other}"),
                }
            }
            card
        })
        .collect()
}

fn ev(s: &str, flags: &RuleFlags) -> HandEval {
    evaluate(&hand(s), flags)
}

const DEF: RuleFlags = RuleFlags {
    four_fingers: false,
    shortcut: false,
    smeared: false,
    pareidolia: false,
    splash: false,
};
const FF: RuleFlags = RuleFlags {
    four_fingers: true,
    ..DEF
};
const SC: RuleFlags = RuleFlags {
    shortcut: true,
    ..DEF
};
const SM: RuleFlags = RuleFlags {
    smeared: true,
    ..DEF
};
const SPL: RuleFlags = RuleFlags {
    splash: true,
    ..DEF
};

#[track_caller]
fn check(s: &str, flags: &RuleFlags, ht: HandType, mask: u8) {
    let e = ev(s, flags);
    assert_eq!(e.hand_type, ht, "hand_type of `{s}` ({e:?})");
    assert_eq!(
        e.scoring_mask, mask,
        "scoring_mask of `{s}`: got {:#010b}, want {mask:#010b} ({e:?})",
        e.scoring_mask
    );
    assert!(e.contains(ht), "`{s}` must contain its own type");
}

#[track_caller]
fn not_type(s: &str, flags: &RuleFlags, ht: HandType) {
    let e = ev(s, flags);
    assert_ne!(e.hand_type, ht, "`{s}` must not be {ht:?} ({e:?})");
}

#[track_caller]
fn has(s: &str, flags: &RuleFlags, ht: HandType) {
    assert!(ev(s, flags).contains(ht), "`{s}` must contain {ht:?}");
}

#[track_caller]
fn lacks(s: &str, flags: &RuleFlags, ht: HandType) {
    assert!(!ev(s, flags).contains(ht), "`{s}` must not contain {ht:?}");
}

// ---------------------------------------------------------------------------------------------
// Exhaustive census
// ---------------------------------------------------------------------------------------------

#[test]
fn census_all_five_card_hands() {
    let mut deck = Vec::with_capacity(52);
    for suit in Suit::ALL {
        for rank in Rank::ALL {
            deck.push(Card::new(deck.len() as u32, rank, suit));
        }
    }
    let mut counts = [0u64; HandType::COUNT];
    let mut total = 0u64;
    for a in 0..52 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                for d in c + 1..52 {
                    for e in d + 1..52 {
                        let cards = [deck[a], deck[b], deck[c], deck[d], deck[e]];
                        counts[evaluate(&cards, &DEF).hand_type.index()] += 1;
                        total += 1;
                    }
                }
            }
        }
    }
    assert_eq!(total, 2_598_960);
    let want = [
        (HandType::HighCard, 1_302_540),
        (HandType::Pair, 1_098_240),
        (HandType::TwoPair, 123_552),
        (HandType::ThreeOfAKind, 54_912),
        (HandType::Straight, 10_200),
        (HandType::Flush, 5_108),
        (HandType::FullHouse, 3_744),
        (HandType::FourOfAKind, 624),
        (HandType::StraightFlush, 40),
        (HandType::FiveOfAKind, 0),
        (HandType::FlushHouse, 0),
        (HandType::FlushFive, 0),
    ];
    for (ht, n) in want {
        assert_eq!(counts[ht.index()], n, "{ht:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Per hand type: positives and negatives
// ---------------------------------------------------------------------------------------------

#[test]
fn empty_and_single_cards() {
    let e = evaluate(&[], &DEF);
    assert_eq!(e.hand_type, HandType::HighCard);
    assert_eq!(e.scoring_mask, 0);
    assert_eq!(e.contains, 0);
    let e = evaluate(&[], &SPL);
    assert_eq!((e.scoring_mask, e.contains), (0, 0));
    check("5S", &DEF, HandType::HighCard, 0b1);
}

#[test]
fn high_card() {
    check("AS 7H 4D", &DEF, HandType::HighCard, 0b001);
    check("2S 9H KD 5C 7S", &DEF, HandType::HighCard, 0b00100);
    check("2S 9H 5D 7C JS", &DEF, HandType::HighCard, 0b10000);
    // Ace-low possibility does not change which card is highest.
    check("AS 2H 3D 4C 6S", &DEF, HandType::HighCard, 0b00001);
    not_type("2S 2H", &DEF, HandType::HighCard);
    not_type("2S 3H 4D 5C 6S", &DEF, HandType::HighCard);
}

#[test]
fn pair() {
    check("9S 9H 2D", &DEF, HandType::Pair, 0b011);
    check("2S 9H 9D KC 3S", &DEF, HandType::Pair, 0b00110);
    check("AS 5H 7D 5C 2S", &DEF, HandType::Pair, 0b01010);
    not_type("9S 9H 9D", &DEF, HandType::Pair);
    not_type("9S 9H 2D 2C", &DEF, HandType::Pair);
}

#[test]
fn two_pair() {
    check("KS KH 2D 2C", &DEF, HandType::TwoPair, 0b1111);
    check("3S 9H 3D 9C 5S", &DEF, HandType::TwoPair, 0b01111);
    check("AS 5H 7D 5C AH", &DEF, HandType::TwoPair, 0b11011);
    not_type("KS KH KD 2C 2S", &DEF, HandType::TwoPair);
    not_type("KS KH KD KC", &DEF, HandType::TwoPair);
    not_type("KS KH 2D 3C", &DEF, HandType::TwoPair);
}

#[test]
fn three_of_a_kind() {
    check("7S 7H 7D", &DEF, HandType::ThreeOfAKind, 0b111);
    check("2S 7H 7D 7C 9S", &DEF, HandType::ThreeOfAKind, 0b01110);
    check("AS 7H 3D 7C 7S", &DEF, HandType::ThreeOfAKind, 0b11010);
    not_type("7S 7H 7D 7C", &DEF, HandType::ThreeOfAKind);
    not_type("7S 7H 7D 2C 2S", &DEF, HandType::ThreeOfAKind);
}

#[test]
fn straight() {
    check("2S 3H 4D 5C 6S", &DEF, HandType::Straight, 0b11111);
    check("AS KH QD JC 10S", &DEF, HandType::Straight, 0b11111);
    check("5S 9H 7D 6C 8S", &DEF, HandType::Straight, 0b11111);
    check("AS 2H 3D 4C 5S", &DEF, HandType::Straight, 0b11111);
    not_type("2S 3H 4D 5C 7S", &DEF, HandType::Straight);
    not_type("2S 3H 4D 5C", &DEF, HandType::Straight);
    not_type("2S 3S 4S 5S 6S", &DEF, HandType::Straight);
    not_type("2S 3H 4D 5C 5S", &DEF, HandType::Straight);
}

#[test]
fn ace_low_and_no_wrap() {
    check("AS 2H 3D 4C 5S", &DEF, HandType::Straight, 0b11111);
    check("3S 2H AD 5C 4S", &DEF, HandType::Straight, 0b11111);
    check("KS AH 2D 3C 4S", &DEF, HandType::HighCard, 0b00010);
    check("QS KH AD 2C 3S", &DEF, HandType::HighCard, 0b00100);
    check("QS KH AD 2C 3S", &SC, HandType::HighCard, 0b00100);
    // 10-J-Q-K-A and A-2-3-4-5 are the only ace straights.
    check("10S JH QD KC AS", &DEF, HandType::Straight, 0b11111);
}

#[test]
fn flush() {
    check("2S 5S 9S JS KS", &DEF, HandType::Flush, 0b11111);
    check("AH 3H 5H 7H 9H", &DEF, HandType::Flush, 0b11111);
    check("2D 2D 9D JD KD", &DEF, HandType::Flush, 0b11111);
    not_type("2S 5S 9S JS KH", &DEF, HandType::Flush);
    not_type("2S 5S 9S JS", &DEF, HandType::Flush);
    not_type("2S 3S 4S 5S 6S", &DEF, HandType::Flush);
}

#[test]
fn full_house() {
    check("KS KH KD 2C 2S", &DEF, HandType::FullHouse, 0b11111);
    check("2C 2S KS KH KD", &DEF, HandType::FullHouse, 0b11111);
    check("3S 5H 5D 3C 5S", &DEF, HandType::FullHouse, 0b11111);
    not_type("KS KH KD KC 2S", &DEF, HandType::FullHouse);
    not_type("KS KH KD 2C 3S", &DEF, HandType::FullHouse);
    not_type("KS KH KD 2C", &DEF, HandType::FullHouse);
}

#[test]
fn four_of_a_kind() {
    check("9S 9H 9D 9C 2S", &DEF, HandType::FourOfAKind, 0b01111);
    check("2S 9S 9H 9D 9C", &DEF, HandType::FourOfAKind, 0b11110);
    check("9S 9H 9D 9C", &DEF, HandType::FourOfAKind, 0b1111);
    // Beats Flush on tier even when all one suit.
    check("9S 9S 9S 9S 2S", &DEF, HandType::FourOfAKind, 0b01111);
    not_type("9S 9H 9D 2C 2S", &DEF, HandType::FourOfAKind);
    not_type("9S 9H 9D 2C", &DEF, HandType::FourOfAKind);
}

#[test]
fn straight_flush() {
    check("2S 3S 4S 5S 6S", &DEF, HandType::StraightFlush, 0b11111);
    check("9H 10H JH QH KH", &DEF, HandType::StraightFlush, 0b11111);
    check("AS 2S 3S 4S 5S", &DEF, HandType::StraightFlush, 0b11111);
    check("10D JD QD KD AD", &DEF, HandType::StraightFlush, 0b11111);
    not_type("2S 3S 4S 5S 7S", &DEF, HandType::StraightFlush);
    not_type("2S 3S 4S 5S 6H", &DEF, HandType::StraightFlush);
    not_type("2S 3S 4S 5S", &DEF, HandType::StraightFlush);
}

#[test]
fn five_of_a_kind() {
    check("AS AH AD AC AS", &DEF, HandType::FiveOfAKind, 0b11111);
    check("7S 7H 7D 7C 7H", &DEF, HandType::FiveOfAKind, 0b11111);
    check("2S 2D 2C 2H 2D", &DEF, HandType::FiveOfAKind, 0b11111);
    not_type("5S 5H 5D 5C 2S", &DEF, HandType::FiveOfAKind);
    not_type("5S 5S 5S 5S 5S", &DEF, HandType::FiveOfAKind);
    not_type("5S 5H 5D 5C", &DEF, HandType::FiveOfAKind);
}

#[test]
fn flush_house() {
    check("KH KH KH 2H 2H", &DEF, HandType::FlushHouse, 0b11111);
    check("3S 3S 5S 5S 5S", &DEF, HandType::FlushHouse, 0b11111);
    check("AD 9D AD 9D 9D", &DEF, HandType::FlushHouse, 0b11111);
    not_type("KH KH KH 2H 2S", &DEF, HandType::FlushHouse);
    not_type("KH KH KH KH 2H", &DEF, HandType::FlushHouse);
}

#[test]
fn flush_five() {
    check("AS AS AS AS AS", &DEF, HandType::FlushFive, 0b11111);
    check("7H 7H 7H 7H 7H", &DEF, HandType::FlushFive, 0b11111);
    // Wiki example: Wild + Wild Ace of Diamonds + three Aces of Spades.
    check("AHw ADw AS AS AS", &DEF, HandType::FlushFive, 0b11111);
    check("AHw AS AS AS AS", &DEF, HandType::FlushFive, 0b11111);
    not_type("AS AS AS AS AH", &DEF, HandType::FlushFive);
    not_type("AS AS AS AS KS", &DEF, HandType::FlushFive);
}

// ---------------------------------------------------------------------------------------------
// Four Fingers
// ---------------------------------------------------------------------------------------------

#[test]
fn four_fingers_flush() {
    check("2S 5S 9S KS 3H", &FF, HandType::Flush, 0b01111);
    check("2S 3H 5S 9S KS", &FF, HandType::Flush, 0b11101);
    check("2S 5S 9S KS", &FF, HandType::Flush, 0b1111);
    check("3H 2S 5S 9S KS", &FF, HandType::Flush, 0b11110);
    // Without Four Fingers none of these is a flush.
    not_type("2S 5S 9S KS 3H", &DEF, HandType::Flush);
    not_type("2S 5S 9S KS", &DEF, HandType::Flush);
    // Three suited cards are never enough.
    not_type("2S 5S 9S KH 3H", &FF, HandType::Flush);
}

#[test]
fn four_fingers_straight() {
    check("2S 3H 4D 5C 5S", &FF, HandType::Straight, 0b11111);
    check("2S 3H 4D 5C KS", &FF, HandType::Straight, 0b01111);
    check("KS 2S 3H 4D 5C", &FF, HandType::Straight, 0b11110);
    check("10S JH QD KC", &FF, HandType::Straight, 0b1111);
    check("9S 9H 10D JC QS", &FF, HandType::Straight, 0b11111);
    check("10S 9H 9D 8C 7S", &FF, HandType::Straight, 0b11111);
    check("AS 2H 3D 4C 9S", &FF, HandType::Straight, 0b01111);
    check("AS KH QD JC 2S", &FF, HandType::Straight, 0b01111);
    // 2-3-4-6 has a hole, and a 3-card run is not enough.
    not_type("2S 3H 4D 6C KS", &FF, HandType::Straight);
    not_type("2S 3H 4D", &FF, HandType::Straight);
    // Without Four Fingers, 2-3-4-5-5 is just a pair.
    check("2S 3H 4D 5C 5S", &DEF, HandType::Pair, 0b11000);
}

#[test]
fn four_fingers_straight_flush_wiki_example() {
    // 2-3-4-5 straight; 2S 3S 5S 10S flush; union covers all five.
    check("2S 3S 4D 5S 10S", &FF, HandType::StraightFlush, 0b11111);
    let e = ev("2S 3S 4D 5S 10S", &FF);
    assert!(e.contains(HandType::Straight) && e.contains(HandType::Flush));
    // Same cards without Four Fingers: nothing.
    check("2S 3S 4D 5S 10S", &DEF, HandType::HighCard, 0b10000);
    // 4-card straight flush.
    check("2S 3S 4S 5S", &FF, HandType::StraightFlush, 0b1111);
}

#[test]
fn four_fingers_does_not_shrink_full_house_or_five() {
    check("KS KH KD 2C", &FF, HandType::ThreeOfAKind, 0b0111);
    check("KS KH KD 2C 3S", &FF, HandType::ThreeOfAKind, 0b00111);
    check("AS AS AS AS", &FF, HandType::FourOfAKind, 0b1111);
    lacks("KS KH KD 2C", &FF, HandType::FullHouse);
    // Flush House / Flush Five need only 4 suit-matching cards.
    check("KH KH KH 2H 2S", &FF, HandType::FlushHouse, 0b11111);
    check("KH KH KH 2H 2S", &DEF, HandType::FullHouse, 0b11111);
    check("AS AS AS AS AH", &FF, HandType::FlushFive, 0b11111);
    check("AS AS AS AS AH", &DEF, HandType::FiveOfAKind, 0b11111);
}

// ---------------------------------------------------------------------------------------------
// Shortcut
// ---------------------------------------------------------------------------------------------

#[test]
fn shortcut_straights() {
    check("10S 8H 6D 5C 3S", &SC, HandType::Straight, 0b11111);
    check("10S 8H 6D 5C 3S", &DEF, HandType::HighCard, 0b00001);
    // Gap of two missing ranks (5 -> 2) is invalid.
    not_type("10S 8H 6D 5C 2S", &SC, HandType::Straight);
    check("QS 9H 7D 5C 3S", &SC, HandType::HighCard, 0b00001);
    // Ace low: 1,3,5,7,9 (all differences 2). Ace high: A K J 9 7.
    check("AS 3H 5D 7C 9S", &SC, HandType::Straight, 0b11111);
    check("AS KH JD 9C 7S", &SC, HandType::Straight, 0b11111);
    // Never wraps: Q K A 2 4 is not a straight.
    not_type("QS KH AD 2C 4S", &SC, HandType::Straight);
    // Duplicates still break a 5-card shortcut straight.
    check("4S 4H 6D 8C 10S", &SC, HandType::Pair, 0b00011);
    // With Four Fingers: 4-run 2-4-6-8, King excluded.
    let both = RuleFlags {
        four_fingers: true,
        shortcut: true,
        ..DEF
    };
    check("2S 4H 6D 8C KS", &both, HandType::Straight, 0b01111);
    check("2S 4H 6D 8C", &both, HandType::Straight, 0b1111);
    not_type("2S 4H 6D 9C", &both, HandType::Straight);
}

// ---------------------------------------------------------------------------------------------
// Suits: Wild, Smeared, debuffed Wild
// ---------------------------------------------------------------------------------------------

#[test]
fn wild_cards() {
    check("2S 5S 9S JS KHw", &DEF, HandType::Flush, 0b11111);
    check("2S 5S 9S JS KH", &DEF, HandType::HighCard, 0b10000);
    check("2Hw 5Cw 9Dw JS KS", &DEF, HandType::Flush, 0b11111);
    // Wild + four Aces of Spades = Flush Five (highest tier possible).
    check("ADw AS AS AS AS", &DEF, HandType::FlushFive, 0b11111);
    // Wild helps a straight flush.
    check("2S 3S 4S 5S 6Hw", &DEF, HandType::StraightFlush, 0b11111);
}

#[test]
fn smeared_joker() {
    check("2S 5C 9S JC KS", &SM, HandType::Flush, 0b11111);
    check("2H 5D 9H JD KH", &SM, HandType::Flush, 0b11111);
    not_type("2S 5C 9S JC KH", &SM, HandType::Flush);
    not_type("2S 5C 9S JC KS", &DEF, HandType::Flush);
    // Smeared + Four Fingers: 4 black cards + 1 red.
    let both = RuleFlags {
        four_fingers: true,
        smeared: true,
        ..DEF
    };
    check("2S 5C 9S JC KH", &both, HandType::Flush, 0b01111);
    check("2H 5D 9D JC KH", &both, HandType::Flush, 0b10111);
    check("2H 5S 9D KH 3D", &both, HandType::Flush, 0b11101);
}

#[test]
fn debuffed_wild_uses_printed_suit() {
    // Debuffed Wild Hearts is no longer a Spade.
    check("2S 5S 9S JS KHwd", &DEF, HandType::HighCard, 0b10000);
    // A debuffed Wild Spade still is a Spade.
    check("2S 5S 9S JS KSwd", &DEF, HandType::Flush, 0b11111);
    // Smeared: printed Diamonds counts as red.
    check("2H 5H 9H JH KDwd", &SM, HandType::Flush, 0b11111);
    check("2H 5H 9H JH KDwd", &DEF, HandType::HighCard, 0b10000);
    // Non-debuffed Wild still matches everything.
    check("2S 5S 9S JS KHw", &DEF, HandType::Flush, 0b11111);
}

#[test]
fn debuffed_cards_still_form_the_hand() {
    check("9Sd 9Hd", &DEF, HandType::Pair, 0b11);
    check("2Sd 5Sd 9Sd JSd KSd", &DEF, HandType::Flush, 0b11111);
    check("2Sd 3Hd 4Dd 5Cd 6Sd", &DEF, HandType::Straight, 0b11111);
    check("9Sd 9H 2D", &DEF, HandType::Pair, 0b011);
    check("KSd", &DEF, HandType::HighCard, 0b1);
}

// ---------------------------------------------------------------------------------------------
// Stone cards, Splash
// ---------------------------------------------------------------------------------------------

#[test]
fn stone_cards() {
    // All-Stone play: High Card, every Stone scores, nothing else contained.
    let e = ev("KSs 2Hs", &DEF);
    assert_eq!(e.hand_type, HandType::HighCard);
    assert_eq!(e.scoring_mask, 0b11);
    assert_eq!(e.contains, 1 << HandType::HighCard.index());
    check("9Ss 9Hs 9Ds 9Cs 9Ss", &DEF, HandType::HighCard, 0b11111);
    // Stone kicker scores with a pair.
    check("9S 9H 2Ds", &DEF, HandType::Pair, 0b111);
    check("2Ds 9S 9H", &DEF, HandType::Pair, 0b111);
    // Stone kicker with High Card.
    check("AS 5H 3Ds", &DEF, HandType::HighCard, 0b101);
    // A Stone never pairs up, whatever its printed rank.
    check("9S 9Hs", &DEF, HandType::HighCard, 0b11);
    lacks("9S 9Hs", &DEF, HandType::Pair);
    // Stone does not complete a flush or a straight without Four Fingers.
    check("2S 5S 9S JS KSs", &DEF, HandType::HighCard, 0b11000);
    check("2S 3H 4D 5C 6Ss", &DEF, HandType::HighCard, 0b11000);
    // With Four Fingers a Stone can fill the 5th slot and scores.
    check("2S 5S 9S JS KSs", &FF, HandType::Flush, 0b11111);
    check("2S 3H 4D 5C 9Ds", &FF, HandType::Straight, 0b11111);
    // Stones on top of a Full House can't exist (5 cards), but a Stone breaks it.
    not_type("KS KH KD 2C 2Ss", &DEF, HandType::FullHouse);
    // Stone keeps scoring when debuffed.
    check("9S 9H 2Dsd", &DEF, HandType::Pair, 0b111);
}

#[test]
fn splash_scores_everything_but_keeps_type() {
    check("9S 9H 2D 5C 7S", &SPL, HandType::Pair, 0b11111);
    check("AS 5H 3D", &SPL, HandType::HighCard, 0b111);
    let e = ev("9S 9H 2D 5C 7S", &SPL);
    assert!(e.contains(HandType::Pair) && !e.contains(HandType::TwoPair));
    check("2S 3H 4D 5C 6S", &SPL, HandType::Straight, 0b11111);
}

// ---------------------------------------------------------------------------------------------
// Contains
// ---------------------------------------------------------------------------------------------

#[test]
fn contains_relations() {
    for ht in [
        HandType::Pair,
        HandType::TwoPair,
        HandType::ThreeOfAKind,
        HandType::FullHouse,
    ] {
        has("KS KH KD 2C 2S", &DEF, ht);
    }
    lacks("KS KH KD 2C 2S", &DEF, HandType::FourOfAKind);
    has("KS KH KD KC 2S", &DEF, HandType::Pair);
    has("KS KH KD KC 2S", &DEF, HandType::ThreeOfAKind);
    has("KS KH KD KC 2S", &DEF, HandType::FourOfAKind);
    lacks("KS KH KD KC 2S", &DEF, HandType::TwoPair);
    lacks("KS KH KD KC 2S", &DEF, HandType::FullHouse);
    has("2S 2S 9S JS KS", &DEF, HandType::Pair);
    has("2S 2S 9S JS KS", &DEF, HandType::Flush);
    lacks("2S 2S 9S JS KS", &DEF, HandType::Straight);
    has("2S 3S 4S 5S 6S", &DEF, HandType::Straight);
    has("2S 3S 4S 5S 6S", &DEF, HandType::Flush);
    lacks("2S 3S 4S 5S 6S", &DEF, HandType::Pair);
    has("KH KH KH 2H 2H", &DEF, HandType::FullHouse);
    has("KH KH KH 2H 2H", &DEF, HandType::Flush);
    has("AS AS AS AS AS", &DEF, HandType::FiveOfAKind);
    has("AS AS AS AS AS", &DEF, HandType::FourOfAKind);
    lacks("AS AS AS AS AS", &DEF, HandType::TwoPair);
    has("AS AS AS AS AS", &DEF, HandType::FlushFive);
    has("5S 8D", &DEF, HandType::HighCard);
    lacks("5S 8D", &DEF, HandType::Pair);
    // Two Pair is contained even when no 4-run exists.
    has("2S 2H 3D 3C 4S", &FF, HandType::TwoPair);
    // Splash does not change contains.
    lacks("9S 9H 2D 5C 7S", &SPL, HandType::TwoPair);
}

// ---------------------------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------------------------

#[test]
fn hand_eval_helpers() {
    let e = ev("9S 9H 2D", &DEF);
    assert!(e.scores(0) && e.scores(1) && !e.scores(2) && !e.scores(8));
    assert!(e.contains(HandType::Pair));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic]
fn more_than_five_cards_trips_debug_assert() {
    let h = hand("2S 3S 4S 5S 6S 7S");
    let _ = evaluate(&h, &DEF);
}

// ---------------------------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------------------------

fn arb_card() -> impl Strategy<Value = Card> {
    (
        0usize..13,
        0usize..4,
        0usize..Enhancement::COUNT,
        0usize..Edition::COUNT,
        0usize..Seal::COUNT,
        any::<bool>(),
    )
        .prop_map(|(r, s, en, ed, se, deb)| {
            let mut c = Card::new(0, Rank::ALL[r], Suit::ALL[s]);
            c.enhancement = Enhancement::ALL[en];
            c.edition = Edition::ALL[ed];
            c.seal = Seal::ALL[se];
            c.debuffed = deb;
            c
        })
}

fn arb_flags() -> impl Strategy<Value = RuleFlags> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(four_fingers, shortcut, smeared, pareidolia, splash)| RuleFlags {
                four_fingers,
                shortcut,
                smeared,
                pareidolia,
                splash,
            },
        )
}

/// Independent brute-force reference for `hand_type` and `contains` (no bit tricks).
fn reference(cards: &[Card], f: &RuleFlags) -> (HandType, u16) {
    let live: Vec<&Card> = cards.iter().filter(|c| !c.is_stone()).collect();
    let k = if f.four_fingers { 4 } else { 5 };
    let mut rank_counts = std::collections::BTreeMap::new();
    for c in &live {
        *rank_counts.entry(c.rank as u8).or_insert(0usize) += 1;
    }
    let groups: Vec<usize> = rank_counts.values().copied().collect();
    let ge = |n: usize| groups.iter().filter(|&&g| g >= n).count();
    let pair = ge(2) >= 1;
    let two_pair = ge(2) >= 2;
    let three = ge(3) >= 1;
    let four = ge(4) >= 1;
    let five = ge(5) >= 1;
    let full_house = ge(3) >= 1 && ge(2) >= 2;

    let mut flush = false;
    let mut straight = false;
    let n = live.len();
    for sub in 0u32..(1 << n) {
        if sub.count_ones() as usize != k {
            continue;
        }
        let pick: Vec<&Card> = (0..n)
            .filter(|i| sub >> i & 1 == 1)
            .map(|i| live[i])
            .collect();
        if Suit::ALL
            .iter()
            .any(|&s| pick.iter().all(|c| c.has_suit(s, f)))
        {
            flush = true;
        }
        // Straight: try every combination of Ace-low / Ace-high.
        let vals: Vec<i32> = pick.iter().map(|c| c.rank as i32).collect();
        for low in [false, true] {
            let mut v: Vec<i32> = vals
                .iter()
                .map(|&x| if low && x == 14 { 1 } else { x })
                .collect();
            v.sort();
            let max_gap = if f.shortcut { 2 } else { 1 };
            if v.windows(2).all(|w| w[1] > w[0] && w[1] - w[0] <= max_gap) {
                straight = true;
            }
        }
    }

    let ht = if five && flush {
        HandType::FlushFive
    } else if full_house && flush {
        HandType::FlushHouse
    } else if five {
        HandType::FiveOfAKind
    } else if straight && flush {
        HandType::StraightFlush
    } else if four {
        HandType::FourOfAKind
    } else if full_house {
        HandType::FullHouse
    } else if flush {
        HandType::Flush
    } else if straight {
        HandType::Straight
    } else if three {
        HandType::ThreeOfAKind
    } else if two_pair {
        HandType::TwoPair
    } else if pair {
        HandType::Pair
    } else {
        HandType::HighCard
    };
    let mut contains = 0u16;
    let mut set = |b: bool, t: HandType| {
        if b {
            contains |= 1 << t.index();
        }
    };
    set(!cards.is_empty(), HandType::HighCard);
    set(pair, HandType::Pair);
    set(two_pair, HandType::TwoPair);
    set(three, HandType::ThreeOfAKind);
    set(four, HandType::FourOfAKind);
    set(five, HandType::FiveOfAKind);
    set(full_house, HandType::FullHouse);
    set(straight, HandType::Straight);
    set(flush, HandType::Flush);
    set(straight && flush, HandType::StraightFlush);
    set(full_house && flush, HandType::FlushHouse);
    set(five && flush, HandType::FlushFive);
    (ht, contains)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    #[test]
    fn evaluate_invariants(cards in prop::collection::vec(arb_card(), 0..=5), flags in arb_flags()) {
        let a = evaluate(&cards, &flags);
        let b = evaluate(&cards, &flags);
        prop_assert_eq!(a, b);
        let n = cards.len();
        let all = if n == 0 { 0u8 } else { ((1u16 << n) - 1) as u8 };
        prop_assert_eq!(a.scoring_mask & !all, 0, "mask only has played positions");
        if n > 0 {
            prop_assert!(a.contains(a.hand_type));
            prop_assert!(a.contains(HandType::HighCard));
        } else {
            prop_assert_eq!(a.contains, 0);
        }
        for (i, c) in cards.iter().enumerate() {
            if c.is_stone() {
                prop_assert!(a.scores(i), "stone {} must score", i);
            }
        }
        if flags.splash {
            prop_assert_eq!(a.scoring_mask, all);
        }
        // Every non-Stone play scores at least one card.
        if cards.iter().any(|c| !c.is_stone()) {
            prop_assert!(a.scoring_mask != 0);
        }
        // Splash never changes type or contains.
        let ns = RuleFlags { splash: false, ..flags };
        let c = evaluate(&cards, &ns);
        prop_assert_eq!((a.hand_type, a.contains), (c.hand_type, c.contains));
    }

    #[test]
    fn matches_brute_force_reference(cards in prop::collection::vec(arb_card(), 0..=5), flags in arb_flags()) {
        let got = evaluate(&cards, &flags);
        let (ht, contains) = reference(&cards, &flags);
        prop_assert_eq!(got.hand_type, ht);
        prop_assert_eq!(got.contains, contains);
    }

    #[test]
    fn all_subsets_agree_with_evaluate(cards in prop::collection::vec(arb_card(), 8..=8), flags in arb_flags()) {
        let mut out = Vec::new();
        evaluate_all_subsets(&cards, &flags, &mut out);
        prop_assert_eq!(out.len(), 218);
        let mut last = 0u32;
        for (mask, e) in &out {
            prop_assert!(*mask > last);
            last = *mask;
            let sel: Vec<Card> = (0..8).filter(|i| mask >> i & 1 == 1).map(|i| cards[i]).collect();
            prop_assert!((1..=5).contains(&sel.len()));
            prop_assert_eq!(*e, evaluate(&sel, &flags));
        }
    }

    #[test]
    fn all_subsets_any_size(cards in prop::collection::vec(arb_card(), 0..=10), flags in arb_flags()) {
        let mut out = vec![(0u32, evaluate(&[], &flags))];
        evaluate_all_subsets(&cards, &flags, &mut out);
        let n = cards.len();
        let expect: usize = (1..=5usize.min(n)).map(|k| binom(n, k)).sum();
        prop_assert_eq!(out.len(), expect);
        for (mask, e) in &out {
            let sel: Vec<Card> = (0..n).filter(|i| mask >> i & 1 == 1).map(|i| cards[i]).collect();
            prop_assert_eq!(*e, evaluate(&sel, &flags));
        }
    }
}

fn binom(n: usize, k: usize) -> usize {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

#[test]
fn all_subsets_eight_cards_is_218() {
    let h = hand("2S 3H 4D 5C 6S 9H KD AC");
    let mut out = Vec::new();
    evaluate_all_subsets(&h, &DEF, &mut out);
    assert_eq!(out.len(), 218);
    assert_eq!(out[0].0, 1);
    // Full set of the first five cards is a straight.
    let m = 0b11111u32;
    let found = out.iter().find(|(k, _)| *k == m).map(|(_, e)| *e);
    assert_eq!(found.map(|e| e.hand_type), Some(HandType::Straight));
    // Buffer is reused without growing on a second call.
    let cap = out.capacity();
    evaluate_all_subsets(&h, &DEF, &mut out);
    assert_eq!((out.len(), out.capacity()), (218, cap));
    // Empty hand yields nothing.
    evaluate_all_subsets(&[], &DEF, &mut out);
    assert!(out.is_empty());
}
