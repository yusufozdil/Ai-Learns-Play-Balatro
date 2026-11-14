//! Baseline evaluation: `eval_agents --agent random|greedy --games N --seed S`.
//!
//! Plays N seeded games (no jokers, Shop is a stub) and prints the mean ante reached, the ante
//! histogram, the win rate, mean steps and games per second.

use balatro_core::agents::{Agent, GreedyAgent, RandomAgent};
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use std::process::ExitCode;
use std::time::Instant;

const MAX_STEPS: usize = 20_000;

fn main() -> ExitCode {
    let mut agent = String::from("greedy");
    let mut games = 1000u64;
    let mut seed = 0u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let v = it.next();
        match (a.as_str(), v) {
            ("--agent", Some(v)) => agent = v,
            ("--games", Some(v)) => games = v.parse().unwrap_or(games),
            ("--seed", Some(v)) => seed = v.parse().unwrap_or(seed),
            _ => {
                eprintln!("usage: eval_agents --agent random|greedy --games N --seed S");
                return ExitCode::from(2);
            }
        }
    }
    if agent != "random" && agent != "greedy" {
        eprintln!("unknown agent {agent}");
        return ExitCode::from(2);
    }

    let mut hist = [0u64; 10];
    let mut ante_sum = 0u64;
    let mut wins = 0u64;
    let mut steps_sum = 0u64;
    let mut ante1_boss_beaten = 0u64;
    let t0 = Instant::now();
    for g_i in 0..games {
        let gseed = seed.wrapping_add(g_i);
        let mut game = Game::new(gseed, GameConfig::default());
        let mut a: Box<dyn Agent> = if agent == "random" {
            Box::new(RandomAgent::new(gseed))
        } else {
            Box::new(GreedyAgent::new())
        };
        let mut steps = 0usize;
        while !game.is_terminal() {
            if steps >= MAX_STEPS {
                eprintln!("seed {gseed}: no termination");
                return ExitCode::FAILURE;
            }
            let act = a.act(&game);
            if let Err(e) = game.step(act) {
                eprintln!("seed {gseed}: agent chose illegal {act:?}: {e}");
                return ExitCode::FAILURE;
            }
            steps += 1;
        }
        let st = game.state();
        let ante = st.ante.clamp(1, 9) as usize;
        hist[ante] += 1;
        ante_sum += ante as u64;
        steps_sum += steps as u64;
        if st.phase == Phase::Won {
            wins += 1;
        }
        // Ante 1 boss beaten <=> the run got past ante 1.
        if st.ante >= 2 {
            ante1_boss_beaten += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let n = games.max(1) as f64;
    println!("agent={agent} games={games} seed={seed}");
    println!("mean ante reached: {:.3}", ante_sum as f64 / n);
    println!("ante histogram (final ante):");
    for (ante, c) in hist.iter().enumerate().skip(1) {
        println!("  ante {ante}: {c:>8} ({:.2}%)", 100.0 * *c as f64 / n);
    }
    println!(
        "ante-1 boss beaten: {:.2}%",
        100.0 * ante1_boss_beaten as f64 / n
    );
    println!("win rate: {:.3}%", 100.0 * wins as f64 / n);
    println!("mean steps: {:.1}", steps_sum as f64 / n);
    println!("games/sec: {:.1} ({secs:.2}s)", n / secs);
    ExitCode::SUCCESS
}
