//! Criterion benchmarks for the game loop (Phase 2C, extended in Phase 7A): `legal_actions_into`
//! in every phase, `validate`, `GameState::clone`, `check_invariants`, and full games with a
//! random agent (steps/s), jokerless and with 5 random jokers + a market-weighted agent.
//!
//! "Steps/s" counts one agent decision per step: `legal_actions_into`, a weighted/uniform pick and
//! `Game::step`, plus the amortised cost of `Game::new` (the RL reset). `check_invariants` only
//! runs inside `Game::step` in debug builds (`cfg(debug_assertions)`), never in release.

use balatro_core::actions::Action;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[path = "support/scenario.rs"]
mod scenario;
use scenario::*;

fn bench_legal_actions(c: &mut Criterion) {
    let ids = joker_ids();
    let mut buf = Vec::with_capacity(512);
    let mut group = c.benchmark_group("legal_actions_into");
    let cases: [(&str, Game); 6] = [
        ("play_8_cards", reach(Phase::Play, &JOKERLESS, &ids, 0)),
        (
            "play_8_cards_5_jokers",
            reach(Phase::Play, &JOKERS_MARKET, &ids, 0),
        ),
        (
            "blind_select",
            reach(Phase::BlindSelect, &JOKERS_MARKET, &ids, 0),
        ),
        ("shop", reach(Phase::Shop, &JOKERS_MARKET, &ids, 0)),
        ("shop_late", reach(Phase::Shop, &JOKERS_MARKET, &ids, 6)),
        ("pack_open", reach(Phase::PackOpen, &JOKERS_MARKET, &ids, 0)),
    ];
    for (name, g) in &cases {
        group.bench_function(*name, |b| {
            b.iter(|| {
                g.legal_actions_into(black_box(&mut buf));
                black_box(buf.len())
            })
        });
    }
    group.finish();
    let g = reach(Phase::Play, &JOKERLESS, &ids, 0);
    c.bench_function("validate/play", |b| {
        b.iter(|| g.validate(black_box(Action::Play(0b10110))))
    });
}

fn bench_state(c: &mut Criterion) {
    let ids = joker_ids();
    let cases: [(&str, Game); 4] = [
        ("play_jokerless", reach(Phase::Play, &JOKERLESS, &ids, 0)),
        ("play_5_jokers", reach(Phase::Play, &JOKERS_MARKET, &ids, 0)),
        ("shop_5_jokers", reach(Phase::Shop, &JOKERS_MARKET, &ids, 2)),
        ("pack_open", reach(Phase::PackOpen, &JOKERS_MARKET, &ids, 0)),
    ];
    let mut group = c.benchmark_group("game_state_clone");
    for (name, g) in &cases {
        group.bench_function(*name, |b| b.iter(|| black_box(g.state().clone())));
    }
    group.finish();
    let mut group = c.benchmark_group("game_clone");
    for (name, g) in &cases {
        group.bench_function(*name, |b| b.iter(|| black_box(g.clone())));
    }
    group.finish();
    let mut group = c.benchmark_group("check_invariants");
    for (name, g) in &cases {
        group.bench_function(*name, |b| {
            b.iter(|| black_box(g.state().check_invariants()))
        });
    }
    group.finish();
    c.bench_function("game_new", |b| {
        let mut seed = 0;
        b.iter(|| {
            seed += 1;
            black_box(Game::new(black_box(seed), GameConfig::default()))
        })
    });
}

fn bench_games(c: &mut Criterion) {
    let ids = joker_ids();
    let seeds: Vec<u64> = (0..64).collect();
    let mut group = c.benchmark_group("steps_per_sec");
    for (name, sc) in [
        ("a_random_jokerless", JOKERLESS),
        ("b_weighted_5_jokers_market_shortcut", JOKERS_MARKET),
        ("c_weighted_5_jokers_market", JOKERS_MARKET_NO_SHORTCUT),
    ] {
        let mut buf = Vec::with_capacity(512);
        let mut w = Vec::with_capacity(512);
        let total: u64 = seeds
            .iter()
            .map(|&s| play_game(s, &sc, &ids, &mut buf, &mut w))
            .sum();
        group.throughput(Throughput::Elements(total));
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut n = 0;
                for &s in &seeds {
                    n += play_game(black_box(s), &sc, &ids, &mut buf, &mut w);
                }
                n
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_legal_actions, bench_state, bench_games);
criterion_main!(benches);
