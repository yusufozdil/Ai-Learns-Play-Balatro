//! Criterion benchmarks for the game loop (Phase 2C): `legal_actions_into` in a Play phase with
//! an 8-card hand, `validate`, and full random games (steps/s).

use balatro_core::actions::Action;
use balatro_core::game::Game;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn play_phase_game() -> Game {
    let mut g = Game::new(1, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    g
}

fn bench_legal_actions(c: &mut Criterion) {
    let g = play_phase_game();
    let mut buf = Vec::with_capacity(512);
    c.bench_function("legal_actions_into/play_8_cards", |b| {
        b.iter(|| {
            g.legal_actions_into(black_box(&mut buf));
            black_box(buf.len())
        })
    });
    c.bench_function("validate/play", |b| {
        b.iter(|| g.validate(black_box(Action::Play(0b10110))))
    });
}

/// Play one full random game; returns the number of steps taken.
fn random_game(seed: u64, buf: &mut Vec<Action>) -> u64 {
    let mut g = Game::new(seed, GameConfig::default());
    let mut rng = GameRng::new(seed ^ 0x5EED);
    let mut steps = 0;
    while !g.is_terminal() && steps < 2000 {
        g.legal_actions_into(buf);
        let Some(i) = rng.choose_index(RngKey::Misc, buf.len()) else {
            break;
        };
        if g.step(buf[i]).is_err() {
            break;
        }
        steps += 1;
    }
    steps
}

fn bench_random_game(c: &mut Criterion) {
    // Fix the step count per iteration so criterion can report steps/s.
    let mut probe = Vec::new();
    let seeds: Vec<u64> = (0..16).collect();
    let total: u64 = seeds.iter().map(|&s| random_game(s, &mut probe)).sum();
    let mut group = c.benchmark_group("random_game");
    group.throughput(Throughput::Elements(total));
    group.bench_function("16_games_steps_per_sec", |b| {
        let mut buf = Vec::with_capacity(512);
        b.iter(|| {
            let mut n = 0;
            for &s in &seeds {
                n += random_game(black_box(s), &mut buf);
            }
            n
        })
    });
    group.finish();
}

criterion_group!(benches, bench_legal_actions, bench_random_game);
criterion_main!(benches);
