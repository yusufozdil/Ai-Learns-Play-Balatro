//! Poker hand evaluation (Four Fingers / Shortcut / Smeared / Wild / Stone / debuff rules),
//! `contains` bitset and scoring-card mask; fast all-subsets evaluation.
//! Rules: docs/mechanics/hands.md (decisions in notes.md §4). Implementation: Phase 1B.
//!
//! INTERFACE (orchestrator-owned): the `HandEval` type and the `evaluate` signature are fixed;
//! the scoring pipeline (Phase 1C) depends on them.

use crate::card::{Card, RuleFlags};
use crate::enums::{HandType, Suit};

/// Result of evaluating the played cards (0..=5 cards, in played left-to-right order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandEval {
    /// The hand type (highest tier that matches).
    pub hand_type: HandType,
    /// Bit `i` set = played card `i` is in the scoring set (Stone always; Splash = all).
    pub scoring_mask: u8,
    /// Bit `HandType::index()` set = the played cards contain that pattern
    /// (e.g. a Full House contains Pair, Two Pair, Three of a Kind and Full House).
    /// The chosen `hand_type` is always contained.
    pub contains: u16,
}

impl HandEval {
    /// True if the played cards contain `ht` (for "if played hand contains a Pair" effects).
    #[inline]
    pub fn contains(&self, ht: HandType) -> bool {
        self.contains & (1 << ht.index()) != 0
    }

    /// True if played card `i` scores.
    #[inline]
    pub fn scores(&self, i: usize) -> bool {
        i < 8 && self.scoring_mask & (1 << i) != 0
    }
}

/// Maximum number of played cards `evaluate` looks at (extra cards are ignored in release builds).
pub const MAX_PLAYED: usize = 5;

/// Maximum hand size accepted by [`evaluate_all_subsets`] (extra cards are ignored in release).
pub const MAX_HAND: usize = 16;

/// Per-card data the evaluator needs, precomputed once so subset scans do not redo it.
///
/// `rank == 0` marks a Stone card (no rank, no suit: `suits == 0`). Otherwise `rank` is the
/// numeric rank (2..=14, Ace high) and bit `Suit::index()` of `suits` is set iff
/// [`Card::has_suit`] says the card belongs to that suit under the active flags (Wild, Smeared,
/// debuffed-Wild all live inside `has_suit`).
#[derive(Clone, Copy)]
struct Info {
    rank: u8,
    suits: u8,
}

impl Info {
    const NONE: Info = Info { rank: 0, suits: 0 };

    #[inline]
    fn new(card: &Card, flags: &RuleFlags) -> Info {
        match card.effective_rank() {
            None => Info::NONE,
            Some(rank) => {
                let mut suits = 0u8;
                for s in Suit::ALL {
                    suits |= (card.has_suit(s, flags) as u8) << s.index();
                }
                Info {
                    rank: rank as u8,
                    suits,
                }
            }
        }
    }
}

/// `SUIT_LANES[mask]` spreads the 4 suit bits into 4 byte lanes so per-suit card counts can be
/// accumulated with plain integer additions.
const SUIT_LANES: [u32; 16] = {
    let mut t = [0u32; 16];
    let mut m = 0;
    while m < 16 {
        let mut s = 0;
        while s < 4 {
            if (m >> s) & 1 != 0 {
                t[m] |= 1 << (8 * s);
            }
            s += 1;
        }
        m += 1;
    }
    t
};

/// Bit of a `HandType` in the `contains` bitset.
#[inline]
const fn bit(ht: HandType) -> u16 {
    1 << (ht as u16)
}

/// Pick the better of two card masks: more cards first, then the one holding the leftmost card
/// on which they differ. Used to choose deterministically between several qualifying subsets.
#[inline]
fn better(a: u8, b: u8) -> u8 {
    let (pa, pb) = (a.count_ones(), b.count_ones());
    if pa != pb {
        return if pa > pb { a } else { b };
    }
    let d = a ^ b;
    if d == 0 || a & d.wrapping_neg() != 0 {
        a
    } else {
        b
    }
}

/// Find the straight run among distinct ranks `rm` (bit `r` = rank value `r`, Ace as bit 14).
/// Returns the run as a bit set over rank values 1..=14 (Ace-low is value 1), or 0 if there is no
/// run of at least `k` ranks. Ace counts as 1 or 14, never both, and there is no wrap-around.
#[inline]
fn straight_run(rm: u16, k: u32, shortcut: bool) -> u16 {
    let rm = if rm & (1 << 14) != 0 { rm | 2 } else { rm };
    if !shortcut {
        let m5 = rm & (rm >> 1) & (rm >> 2) & (rm >> 3) & (rm >> 4);
        if m5 != 0 {
            return 0b11111 << m5.trailing_zeros();
        }
        if k == 4 {
            let m4 = rm & (rm >> 1) & (rm >> 2) & (rm >> 3);
            if m4 != 0 {
                return 0b1111 << m4.trailing_zeros();
            }
        }
        0
    } else {
        // Chains of ranks whose adjacent differences are 1 or 2. With at most 5 cards the
        // Ace-low and Ace-high copies can never end up in the same chain.
        let (mut seg, mut len, mut prev) = (0u16, 0u32, 0u32);
        let (mut best, mut best_len) = (0u16, 0u32);
        let mut bits = rm;
        while bits != 0 {
            let v = bits.trailing_zeros();
            bits &= bits - 1;
            if seg != 0 && v - prev <= 2 {
                seg |= 1 << v;
                len += 1;
            } else {
                seg = 1 << v;
                len = 1;
            }
            prev = v;
            if len > best_len {
                best_len = len;
                best = seg;
            }
        }
        if best_len >= k { best } else { 0 }
    }
}

/// Mask of non-Stone cards whose rank occurs at least `min` times.
#[inline]
fn group_mask(info: &[Info; MAX_PLAYED], n: usize, counts: &[u8; 15], min: u8) -> u8 {
    let mut mask = 0u8;
    for (i, inf) in info.iter().enumerate().take(n) {
        mask |= ((inf.rank != 0 && counts[inf.rank as usize] >= min) as u8) << i;
    }
    mask
}

/// Evaluate `n` (0..=5) precomputed cards.
#[inline]
fn eval_infos(info: &[Info; MAX_PLAYED], n: usize, flags: &RuleFlags) -> HandEval {
    debug_assert!(n <= MAX_PLAYED);
    if n == 0 {
        return HandEval {
            hand_type: HandType::HighCard,
            scoring_mask: 0,
            contains: 0,
        };
    }
    let all: u8 = (1u8 << n) - 1;

    // Pass 1: stones, rank histogram, per-suit counts, highest card.
    let mut counts = [0u8; 15];
    let (mut stones, mut nn) = (0u8, 0u32);
    let (mut rm, mut lanes) = (0u16, 0u32);
    let (mut hi_rank, mut hi_idx) = (0u8, 0usize);
    for (i, inf) in info.iter().enumerate().take(n) {
        if inf.rank == 0 {
            stones |= 1 << i;
            continue;
        }
        nn += 1;
        counts[inf.rank as usize] += 1;
        rm |= 1 << inf.rank;
        lanes += SUIT_LANES[(inf.suits & 15) as usize];
        if inf.rank > hi_rank {
            hi_rank = inf.rank;
            hi_idx = i;
        }
    }

    // Rank-multiplicity summary over the distinct ranks.
    let distinct = rm.count_ones();
    let (mut m, mut ge2, mut ge3) = (0u8, 0u32, 0u32);
    let mut bits = rm;
    while bits != 0 {
        let c = counts[bits.trailing_zeros() as usize];
        bits &= bits - 1;
        m = m.max(c);
        ge2 += (c >= 2) as u32;
        ge3 += (c >= 3) as u32;
    }
    let five = m >= 5;
    let four = m >= 4;
    let three = m >= 3;
    let full_house = ge3 >= 1 && ge2 >= 2;

    let k: u32 = if flags.four_fingers { 4 } else { 5 };

    // Flush: some suit shared by at least k non-Stone cards.
    let mut flush_mask = 0u8;
    if nn >= k {
        for s in 0..4 {
            if (lanes >> (8 * s)) & 0xFF >= k {
                let mut mask = 0u8;
                for (i, inf) in info.iter().enumerate().take(n) {
                    mask |= ((inf.suits >> s) & 1) << i;
                }
                flush_mask = better(flush_mask, mask);
            }
        }
    }
    let flush = flush_mask != 0;

    // Straight: a run of k distinct (non-Stone) ranks; every card whose rank is in the run scores.
    let mut straight_mask = 0u8;
    if distinct >= k {
        let run = straight_run(rm, k, flags.shortcut);
        if run != 0 {
            for (i, inf) in info.iter().enumerate().take(n) {
                let r = inf.rank as u32;
                let hit = r != 0 && ((run >> r) & 1 != 0 || (r == 14 && run & 2 != 0));
                straight_mask |= (hit as u8) << i;
            }
        }
    }
    let straight = straight_mask != 0;

    let hand_type;
    let mut mask;
    if five && flush {
        hand_type = HandType::FlushFive;
        mask = all;
    } else if full_house && flush {
        hand_type = HandType::FlushHouse;
        mask = all;
    } else if five {
        hand_type = HandType::FiveOfAKind;
        mask = all;
    } else if straight && flush {
        hand_type = HandType::StraightFlush;
        mask = straight_mask | flush_mask;
    } else if four {
        hand_type = HandType::FourOfAKind;
        mask = group_mask(info, n, &counts, 4);
    } else if full_house {
        hand_type = HandType::FullHouse;
        mask = all;
    } else if flush {
        hand_type = HandType::Flush;
        mask = flush_mask;
    } else if straight {
        hand_type = HandType::Straight;
        mask = straight_mask;
    } else if three {
        hand_type = HandType::ThreeOfAKind;
        mask = group_mask(info, n, &counts, 3);
    } else if ge2 >= 2 {
        hand_type = HandType::TwoPair;
        mask = group_mask(info, n, &counts, 2);
    } else if ge2 == 1 {
        hand_type = HandType::Pair;
        mask = group_mask(info, n, &counts, 2);
    } else {
        hand_type = HandType::HighCard;
        mask = if nn > 0 { 1 << hi_idx } else { 0 };
    }
    mask |= stones;
    if flags.splash {
        mask = all;
    }

    let mut contains = bit(hand_type) | bit(HandType::HighCard);
    if m >= 2 {
        contains |= bit(HandType::Pair);
    }
    if ge2 >= 2 {
        contains |= bit(HandType::TwoPair);
    }
    if three {
        contains |= bit(HandType::ThreeOfAKind);
    }
    if four {
        contains |= bit(HandType::FourOfAKind);
    }
    if five {
        contains |= bit(HandType::FiveOfAKind);
    }
    if full_house {
        contains |= bit(HandType::FullHouse);
    }
    if straight {
        contains |= bit(HandType::Straight);
    }
    if flush {
        contains |= bit(HandType::Flush);
    }
    if straight && flush {
        contains |= bit(HandType::StraightFlush);
    }
    if full_house && flush {
        contains |= bit(HandType::FlushHouse);
    }
    if five && flush {
        contains |= bit(HandType::FlushFive);
    }

    HandEval {
        hand_type,
        scoring_mask: mask,
        contains,
    }
}

/// Evaluate played cards (0..=5, left-to-right order) under the active rule flags.
///
/// Never panics on any input of length 0..=5 (length 0 yields High Card with empty masks).
/// More than 5 cards is a caller bug: it trips a `debug_assert!` in debug builds, and in release
/// builds only the first 5 cards are evaluated. Does not allocate.
///
/// Rules (decisions in `notes.md` §4.1, sources in `docs/mechanics/hands.md`):
/// - Stone cards are ignored for typing and `contains` but always score.
/// - Suit logic goes exclusively through [`Card::has_suit`] (Wild, Smeared, debuffed Wild).
/// - Priority: Flush Five, Flush House, Five of a Kind, Straight Flush, Four of a Kind,
///   Full House, Flush, Straight, Three of a Kind, Two Pair, Pair, High Card.
/// - With Four Fingers, if several 4-card flushes qualify (Wild/Smeared), the one with the most
///   matching cards wins, ties going to the one holding the leftmost differing card.
pub fn evaluate(cards: &[Card], flags: &RuleFlags) -> HandEval {
    debug_assert!(
        cards.len() <= MAX_PLAYED,
        "evaluate: at most {MAX_PLAYED} cards"
    );
    let n = cards.len().min(MAX_PLAYED);
    let mut info = [Info::NONE; MAX_PLAYED];
    for (slot, card) in info.iter_mut().zip(&cards[..n]) {
        *slot = Info::new(card, flags);
    }
    eval_infos(&info, n, flags)
}

/// Evaluate every non-empty selection of at most 5 cards from `hand` (at most 16 cards).
///
/// Clears `out` and pushes `(selection_mask, eval)` for every mask with 1..=5 bits set, in
/// increasing mask order. Bit `j` of the mask selects `hand[j]`; the played order of a selection
/// is hand order, so `eval` equals `evaluate` on the selected cards in that order (and
/// `scoring_mask` indexes the selection, not the hand). Per-card data is computed once and no
/// per-subset allocation happens (`out` only grows if its capacity is short). A hand longer than
/// 16 cards trips a `debug_assert!`; in release only the first 16 are used.
pub fn evaluate_all_subsets(hand: &[Card], flags: &RuleFlags, out: &mut Vec<(u16, HandEval)>) {
    debug_assert!(
        hand.len() <= MAX_HAND,
        "evaluate_all_subsets: at most {MAX_HAND} cards"
    );
    out.clear();
    let len = hand.len().min(MAX_HAND);
    let mut infos = [Info::NONE; MAX_HAND];
    for (slot, card) in infos.iter_mut().zip(&hand[..len]) {
        *slot = Info::new(card, flags);
    }
    // Number of selections of 1..=5 cards out of `len`.
    let mut total = 0usize;
    let mut binom = 1usize; // C(len, 0)
    for j in 1..=MAX_PLAYED.min(len) {
        binom = binom * (len - j + 1) / j;
        total += binom;
    }
    out.reserve(total);

    for mask in 1u32..(1u32 << len) {
        if mask.count_ones() as usize > MAX_PLAYED {
            continue;
        }
        let mut sel = [Info::NONE; MAX_PLAYED];
        let mut n = 0;
        let mut bits = mask;
        while bits != 0 {
            sel[n] = infos[bits.trailing_zeros() as usize];
            n += 1;
            bits &= bits - 1;
        }
        out.push((mask as u16, eval_infos(&sel, n, flags)));
    }
}
