//! Baseline evaluation: `eval_agents --agent random|greedy|heuristic --games N --seed S
//! [--threads T]`.
//!
//! Plays N seeded games and prints the mean ante reached, the ante histogram, the win rate, mean
//! steps and games per second. Games are independent and the aggregate does not depend on the
//! thread count.

use balatro_core::agents::{Agent, GreedyAgent, HeuristicAgent, HeuristicConfig, RandomAgent};
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_STEPS: usize = 20_000;

/// Result of one game.
#[derive(Clone, Copy)]
struct Outcome {
    ante: usize,
    won: bool,
    steps: usize,
}

fn play_one(agent: &str, cfg: &HeuristicConfig, gseed: u64) -> Result<Outcome, String> {
    let mut game = Game::new(gseed, GameConfig::default());
    let mut a: Box<dyn Agent> = match agent {
        "random" => Box::new(RandomAgent::new(gseed)),
        "heuristic" => Box::new(HeuristicAgent::with_config(gseed, cfg.clone())),
        _ => Box::new(GreedyAgent::new()),
    };
    let mut steps = 0usize;
    while !game.is_terminal() {
        if steps >= MAX_STEPS {
            return Err(format!("seed {gseed}: no termination"));
        }
        let act = a.act(&game);
        if let Err(e) = game.step(act) {
            return Err(format!("seed {gseed}: agent chose illegal {act:?}: {e}"));
        }
        steps += 1;
    }
    let st = game.state();
    Ok(Outcome {
        ante: st.ante.clamp(1, 9) as usize,
        won: st.phase == Phase::Won,
        steps,
    })
}

fn main() -> ExitCode {
    let mut agent = String::from("greedy");
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut threads = 1usize;
    let mut cfg = HeuristicConfig::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let v = it.next();
        match (a.as_str(), v) {
            ("--agent", Some(v)) => agent = v,
            ("--games", Some(v)) => games = v.parse().unwrap_or(games),
            ("--seed", Some(v)) => seed = v.parse().unwrap_or(seed),
            ("--threads", Some(v)) => threads = v.parse().unwrap_or(threads).max(1),
            ("--set", Some(v)) => {
                // Heuristic config knob: `--set name=value` (repeatable).
                let ok = v.split_once('=').is_some_and(|(k, val)| cfg.set(k, val));
                if !ok {
                    eprintln!("bad --set {v}");
                    return ExitCode::from(2);
                }
            }
            _ => {
                eprintln!(
                    "usage: eval_agents --agent random|greedy|heuristic --games N --seed S \
                     [--threads T] [--set knob=value]"
                );
                return ExitCode::from(2);
            }
        }
    }
    if !matches!(agent.as_str(), "random" | "greedy" | "heuristic") {
        eprintln!("unknown agent {agent}");
        return ExitCode::from(2);
    }

    let next = AtomicU64::new(0);
    let results: Mutex<Vec<(u64, Outcome)>> = Mutex::new(Vec::new());
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= games {
                        break;
                    }
                    match play_one(&agent, &cfg, seed.wrapping_add(i)) {
                        Ok(o) => {
                            if let Ok(mut r) = results.lock() {
                                r.push((i, o));
                            }
                        }
                        Err(e) => {
                            if let Ok(mut f) = failure.lock() {
                                *f = Some(e);
                            }
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Ok(f) = failure.lock()
        && let Some(e) = f.as_ref()
    {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let secs = t0.elapsed().as_secs_f64();
    let res = results.into_inner().unwrap_or_default();

    let mut hist = [0u64; 10];
    let mut ante_sum = 0u64;
    let mut wins = 0u64;
    let mut steps_sum = 0u64;
    let mut ante1_boss_beaten = 0u64;
    for (_, o) in &res {
        hist[o.ante] += 1;
        ante_sum += o.ante as u64;
        steps_sum += o.steps as u64;
        wins += u64::from(o.won);
        // Ante 1 boss beaten <=> the run got past ante 1.
        ante1_boss_beaten += u64::from(o.ante >= 2);
    }
    let n = games.max(1) as f64;
    println!("agent={agent} games={games} seed={seed} threads={threads}");
    println!("mean ante reached: {:.3}", ante_sum as f64 / n);
    println!("ante histogram (final ante):");
    for (ante, c) in hist.iter().enumerate().skip(1) {
        println!("  ante {ante}: {c:>8} ({:.2}%)", 100.0 * *c as f64 / n);
    }
    println!(
        "ante-1 boss beaten: {:.2}%",
        100.0 * ante1_boss_beaten as f64 / n
    );
    println!("win rate: {:.3}% ({wins} wins)", 100.0 * wins as f64 / n);
    println!("mean steps: {:.1}", steps_sum as f64 / n);
    println!("games/sec: {:.1} ({secs:.2}s)", n / secs);
    ExitCode::SUCCESS
}
