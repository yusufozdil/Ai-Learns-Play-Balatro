//! Power-meter probe: `power_probe [--games N] [--k K] [--seed S]`. Plays heuristic-agent games and
//! prints, per ante, the mean build power at the Small-blind BlindSelect (v1 / v2) and the cost
//! per `power` call.

use balatro_core::agents::power::{PowerCfg, power};
use balatro_core::agents::{Agent, HeuristicAgent, HeuristicConfig};
use balatro_core::blinds::BlindKind;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use std::time::Instant;

fn main() {
    let (mut games, mut k, mut seed) = (50u64, 8usize, 0u64);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let v = it.next().unwrap_or_default();
        match a.as_str() {
            "--games" => games = v.parse().unwrap_or(games),
            "--k" => k = v.parse().unwrap_or(k),
            "--seed" => seed = v.parse().unwrap_or(seed),
            _ => {}
        }
    }
    let mut sums = [[0.0f64; 3]; 9]; // [ante][v1, v2, n]
    let (mut calls, mut dur) = ([0usize; 2], [0.0f64; 2]);
    for gi in 0..games {
        let mut g = Game::new(seed + gi, GameConfig::default());
        let mut agent = HeuristicAgent::with_config(seed + gi, HeuristicConfig::default());
        let mut steps = 0;
        while !g.is_terminal() && steps < 20_000 {
            let s = g.state();
            if s.phase == Phase::BlindSelect && s.blind == BlindKind::Small {
                let ante = s.ante.clamp(1, 8) as usize;
                for (i, boss_aware) in [false, true].into_iter().enumerate() {
                    let cfg = PowerCfg {
                        boss_aware,
                        ..PowerCfg::default()
                    };
                    let t = Instant::now();
                    let r = power(s, k, &cfg);
                    dur[i] += t.elapsed().as_secs_f64();
                    calls[i] += 1;
                    sums[ante][i] += r.phi;
                }
                sums[ante][2] += 1.0;
            }
            let a = agent.act(&g);
            g.step(a).unwrap();
            steps += 1;
        }
    }
    println!(
        "k={k}: us/call v1 {:.1} v2 {:.1}",
        dur[0] / calls[0] as f64 * 1e6,
        dur[1] / calls[1] as f64 * 1e6
    );
    for (a, s) in sums.iter().enumerate().skip(1) {
        if s[2] > 0.0 {
            println!(
                "ante {a}: n={} phi_v1 {:.3} phi_v2 {:.3}",
                s[2],
                s[0] / s[2],
                s[1] / s[2]
            );
        }
    }
}
