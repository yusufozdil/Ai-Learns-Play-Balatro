//! Poor man's profiler (no `perf` needed): times `legal_actions_into`, the agent pick and
//! `Game::step` per action kind in the benchmark scenarios.
//! `cargo run --release -p balatro-core --example breakdown -- [games] [a|b|c]`.

#[path = "../benches/support/scenario.rs"]
mod scenario;

use balatro_core::actions::Action;
use balatro_core::rng::GameRng;
use scenario::*;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn kind(a: &Action) -> &'static str {
    match a {
        Action::SelectBlind => "SelectBlind",
        Action::SkipBlind => "SkipBlind",
        Action::Play(_) => "Play",
        Action::Discard(_) => "Discard",
        Action::SellJoker(_) => "SellJoker",
        Action::UseConsumable { .. } => "UseConsumable",
        Action::SellConsumable(_) => "SellConsumable",
        Action::EndShop => "EndShop",
        Action::MoveJoker { .. } => "MoveJoker",
        Action::Buy(_) => "Buy",
        Action::BuyAndUse(_) => "BuyAndUse",
        Action::OpenPack(_) => "OpenPack",
        Action::BuyVoucher(_) => "BuyVoucher",
        Action::Reroll => "Reroll",
        Action::PickPackItem { .. } => "PickPackItem",
        Action::SkipPack => "SkipPack",
        Action::RerollBoss => "RerollBoss",
    }
}

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let which = std::env::args().nth(2).unwrap_or_else(|| "b".into());
    let sc = match which.as_str() {
        "a" => JOKERLESS,
        "c" => JOKERS_MARKET_NO_SHORTCUT,
        _ => JOKERS_MARKET,
    };
    let ids = joker_ids();
    let mut buf = Vec::with_capacity(512);
    let mut w = Vec::with_capacity(512);
    let mut by_kind: BTreeMap<&str, (u64, Duration)> = BTreeMap::new();
    let (mut t_new, mut t_legal, mut t_pick) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut steps = 0u64;
    let wall = Instant::now();
    for seed in 0..games {
        let t = Instant::now();
        let mut g = new_game(seed, &sc, &ids);
        t_new += t.elapsed();
        let mut rng = GameRng::new(seed ^ 0x5EED);
        while !g.is_terminal() {
            let t = Instant::now();
            g.legal_actions_into(&mut buf);
            t_legal += t.elapsed();
            if buf.is_empty() {
                break;
            }
            let t = Instant::now();
            let a = pick_from(&mut g, &mut rng, &sc, &buf, &mut w);
            t_pick += t.elapsed();
            let t = Instant::now();
            let r = g.step(a);
            let d = t.elapsed();
            let e = by_kind.entry(kind(&a)).or_default();
            e.0 += 1;
            e.1 += d;
            if r.is_err() {
                break;
            }
            steps += 1;
        }
    }
    let total = wall.elapsed();
    println!(
        "scenario {which}: {games} games, {steps} steps, {:.0} ns/step wall ({:.0} steps/s)",
        total.as_nanos() as f64 / steps as f64,
        steps as f64 / total.as_secs_f64()
    );
    let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();
    println!(
        "Game::new {:.1}%  legal_actions {:.1}% ({:.0} ns/step)  pick {:.1}%",
        pct(t_new),
        pct(t_legal),
        t_legal.as_nanos() as f64 / steps as f64,
        pct(t_pick)
    );
    let mut rows: Vec<_> = by_kind.into_iter().collect();
    rows.sort_by_key(|(_, (_, d))| std::cmp::Reverse(*d));
    for (k, (n, d)) in rows {
        println!(
            "step {k:<15} n={n:>9} {:>6.1}%  {:>8.0} ns/call",
            pct(d),
            d.as_nanos() as f64 / n as f64
        );
    }
}
