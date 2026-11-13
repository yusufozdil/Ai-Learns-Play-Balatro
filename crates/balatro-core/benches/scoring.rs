//! Criterion benchmarks for `scoring::Scorer::score_hand` (Phase 1C).
//! Target: >= 1M calls/s with 5 simple jokers on a 5-card hand (release, one core).

use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::RunView;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, Scorer};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn flush_hand() -> (Vec<Card>, Vec<Card>, HandEval) {
    let mut played = vec![
        Card::new(1, Rank::Two, Suit::Diamonds),
        Card::new(2, Rank::Five, Suit::Diamonds),
        Card::new(3, Rank::Seven, Suit::Diamonds),
        Card::new(4, Rank::Nine, Suit::Diamonds),
        Card::new(5, Rank::King, Suit::Diamonds),
    ];
    played[1].enhancement = Enhancement::Mult;
    played[2].edition = Edition::Foil;
    played[3].seal = Seal::Red;
    let mut held = vec![
        Card::new(6, Rank::King, Suit::Spades),
        Card::new(7, Rank::Four, Suit::Hearts),
        Card::new(8, Rank::Queen, Suit::Clubs),
    ];
    held[0].enhancement = Enhancement::Steel;
    let eval = HandEval {
        hand_type: HandType::Flush,
        scoring_mask: 0x1F,
        contains: 1 << HandType::Flush.index(),
    };
    (played, held, eval)
}

fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| JokerInstance::new(i as u32, id))
        .collect()
}

fn bench_case(c: &mut Criterion, name: &str, ids: &[JokerId], trace: bool) {
    let (played, held, eval) = flush_hand();
    let mut buf = played.clone();
    let mut js = jokers(ids);
    let run = RunView {
        money: 12,
        ..RunView::default()
    };
    let mut rng = GameRng::new(42);
    let mut scorer = Scorer::with_trace(trace);
    let mut group = c.benchmark_group("score_hand");
    group.throughput(Throughput::Elements(1));
    group.bench_function(name, |b| {
        b.iter(|| {
            buf.copy_from_slice(&played);
            let r = scorer.score_hand(ScoreInput {
                played: black_box(&mut buf),
                held: black_box(&held),
                eval: black_box(eval),
                jokers: &mut js,
                run: black_box(&run),
                rng: &mut rng,
            });
            black_box(r.score)
        })
    });
    group.finish();
}

fn benches(c: &mut Criterion) {
    use JokerId::*;
    bench_case(c, "no_jokers", &[], false);
    bench_case(
        c,
        "5_simple_jokers",
        &[Joker, GreedyJoker, JollyJoker, Cavendish, Hack],
        false,
    );
    bench_case(
        c,
        "5_simple_jokers_traced",
        &[Joker, GreedyJoker, JollyJoker, Cavendish, Hack],
        true,
    );
    bench_case(
        c,
        "5_copier_held_jokers",
        &[Blueprint, Mime, Baron, Brainstorm, BaseballCard],
        false,
    );
}

criterion_group!(scoring, benches);
criterion_main!(scoring);
