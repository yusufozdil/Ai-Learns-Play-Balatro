//! Criterion benchmarks for `hand_eval`: single 5-card evaluation and the 8-card all-subsets scan.

use balatro_core::card::{Card, RuleFlags};
use balatro_core::enums::{Rank, Suit};
use balatro_core::hand_eval::{HandEval, evaluate, evaluate_all_subsets};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Tiny deterministic xorshift generator (benchmark inputs only; not game RNG).
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// `n` distinct random cards from a standard 52-card deck.
fn random_cards(rng: &mut XorShift, n: usize) -> Vec<Card> {
    let mut deck: Vec<u8> = (0..52).collect();
    for i in 0..n {
        let j = i + (rng.next() % (52 - i as u64)) as usize;
        deck.swap(i, j);
    }
    deck[..n]
        .iter()
        .map(|&d| {
            Card::new(
                d as u32,
                Rank::ALL[(d % 13) as usize],
                Suit::ALL[(d / 13) as usize],
            )
        })
        .collect()
}

fn bench_evaluate(c: &mut Criterion) {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let hands: Vec<[Card; 5]> = (0..1024)
        .map(|_| {
            let v = random_cards(&mut rng, 5);
            [v[0], v[1], v[2], v[3], v[4]]
        })
        .collect();
    let flags = RuleFlags::default();
    let mut group = c.benchmark_group("evaluate");
    group.throughput(Throughput::Elements(hands.len() as u64));
    group.bench_function("5_card_default_flags_x1024", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for h in &hands {
                let e = evaluate(black_box(h), &flags);
                acc = acc.wrapping_add(e.contains as u32 + e.scoring_mask as u32);
            }
            acc
        })
    });
    let all_flags = RuleFlags {
        four_fingers: true,
        shortcut: true,
        smeared: true,
        ..RuleFlags::default()
    };
    group.bench_function("5_card_all_modifiers_x1024", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for h in &hands {
                let e = evaluate(black_box(h), &all_flags);
                acc = acc.wrapping_add(e.contains as u32 + e.scoring_mask as u32);
            }
            acc
        })
    });
    group.finish();
}

fn bench_all_subsets(c: &mut Criterion) {
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
    let hands: Vec<Vec<Card>> = (0..256).map(|_| random_cards(&mut rng, 8)).collect();
    let flags = RuleFlags::default();
    let mut out: Vec<(u16, HandEval)> = Vec::with_capacity(256);
    let mut group = c.benchmark_group("evaluate_all_subsets");
    group.throughput(Throughput::Elements(218));
    let mut i = 0usize;
    group.bench_function("8_card_hand_218_subsets", |b| {
        b.iter(|| {
            i = (i + 1) & 255;
            evaluate_all_subsets(black_box(&hands[i]), &flags, &mut out);
            out.len()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_evaluate, bench_all_subsets);
criterion_main!(benches);
