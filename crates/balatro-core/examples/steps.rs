//! Single-core steps/s of the benchmark scenarios without criterion (for profilers):
//! `cargo run --release -p balatro-core --example steps -- [seconds] [scenario prefix]`.
//! Profile with `CARGO_PROFILE_RELEASE_DEBUG=true` and any sampling profiler (`perf`, `samply`).

#[path = "../benches/support/scenario.rs"]
mod scenario;

use scenario::*;
use std::time::Instant;

fn main() {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let only = std::env::args().nth(2);
    let ids = joker_ids();
    let mut buf = Vec::with_capacity(512);
    let mut w = Vec::with_capacity(512);
    for (name, sc) in [
        ("a_random_jokerless", JOKERLESS),
        ("b_weighted_5_jokers_market_shortcut", JOKERS_MARKET),
        ("c_weighted_5_jokers_market", JOKERS_MARKET_NO_SHORTCUT),
    ] {
        if only.as_deref().is_some_and(|o| !name.starts_with(o)) {
            continue;
        }
        let t = Instant::now();
        let (mut steps, mut games) = (0u64, 0u64);
        while t.elapsed().as_secs_f64() < secs {
            steps += play_game(games, &sc, &ids, &mut buf, &mut w);
            games += 1;
        }
        let e = t.elapsed().as_secs_f64();
        println!(
            "{name}: {:.0} steps/s ({games} games, {:.1} steps/game)",
            steps as f64 / e,
            steps as f64 / games as f64
        );
    }
}
