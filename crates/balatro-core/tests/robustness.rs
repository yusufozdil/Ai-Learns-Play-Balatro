//! Robustness (Phase 2D): mass random games, determinism, clone-continuation and invalid-action
//! fuzzing. Phase 3.R adds games that start with random boards of defined jokers. The 100k runs
//! are `#[ignore]`: `cargo test --release -p balatro-core --test robustness -- --ignored`.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, RandomAgent};
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::enums::{Edition, Enhancement, Phase, Seal};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::{GameConfig, GameState};

/// Far above any real run.
const MAX_STEPS: usize = 5000;

fn new_game(seed: u64, with_jokers: bool) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    if with_jokers {
        let mut p = GameRng::new(seed ^ 0x70C3);
        for _ in 0..p.range(RngKey::Misc, 1, 6) {
            let id = JokerId::ALL[p.range(RngKey::Misc, 0, JokerId::COUNT)];
            let _ = g.add_joker(JokerInstance::new(0, id));
        }
    }
    g
}

/// Plays `games` seeded games from `first_seed`; returns total steps.
fn mass_run(first_seed: u64, games: u64) -> u64 {
    let mut total = 0u64;
    for seed in first_seed..first_seed + games {
        let mut g = new_game(seed, seed % 2 == 1);
        // Mix: mostly uniform random, some Play-biased (reaches deeper antes).
        let mut agent = match seed % 4 {
            0 | 1 => RandomAgent::new(seed),
            2 => RandomAgent::with_play_weight(seed, 6.0),
            _ => RandomAgent::with_play_weight(seed, 30.0),
        };
        let mut steps = 0usize;
        while !g.is_terminal() {
            assert!(steps < MAX_STEPS, "seed {seed}: no termination");
            let a = agent.act(&g);
            // Every 8th game: nudge the round score so blinds are beaten and later antes,
            // shops and the win path are exercised too (random play almost never gets there).
            if seed % 8 == 7 && matches!(a, Action::Play(_)) {
                let t = g.state().target;
                g.state_mut().round_score = t - 1.0;
            }
            if let Err(e) = g.step(a) {
                panic!("seed {seed}: legal-looking {a:?} rejected: {e}");
            }
            g.state()
                .check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
            steps += 1;
        }
        assert!(g.outcome().is_some(), "seed {seed}");
        total += steps as u64;
    }
    total
}

#[test]
fn ten_thousand_random_games() {
    let steps = mass_run(0, 10_000);
    assert!(steps > 40_000);
}

#[test]
#[ignore = "100k games; run with `cargo test --release -- --ignored`"]
fn hundred_thousand_random_games() {
    let steps = mass_run(1_000_000, 100_000);
    eprintln!("100k games: {steps} steps");
    assert!(steps > 400_000);
}

// ----- Phase 3.R: random boards of DEFINED jokers ------------------------------------------

/// Every joker id with a definition, in index order.
fn defined_ids() -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs().map(|d| d.id).collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

/// A run that starts with 0-5 random defined jokers (random editions, some Eternal), and for
/// half the seeds a deck where about a quarter of the cards carry a random enhancement, seal
/// and edition (Glass breaks, Lucky, Steel, Gold, Red/Gold/Blue/Purple seals, Stone cards).
fn new_joker_board_game(seed: u64, ids: &[JokerId]) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    let mut p = GameRng::new(seed ^ 0x3A11_70C3);
    let n = p.range(RngKey::Misc, 0, 6);
    for _ in 0..n {
        let mut j = JokerInstance::new(0, ids[p.range(RngKey::Misc, 0, ids.len())]);
        j.edition = match p.range(RngKey::Misc, 0, 10) {
            0 => Edition::Foil,
            1 => Edition::Holographic,
            2 => Edition::Polychrome,
            3 => Edition::Negative,
            _ => Edition::Base,
        };
        let eternal_ok = j.def().is_some_and(|d| d.eternal_compatible);
        j.stickers.eternal = eternal_ok && p.range(RngKey::Misc, 0, 10) == 0;
        let _ = g.add_joker(j);
    }
    if seed.is_multiple_of(2) {
        for c in &mut g.state_mut().deck.draw_pile {
            if p.range(RngKey::Misc, 0, 4) != 0 {
                continue;
            }
            c.enhancement = Enhancement::ALL[p.range(RngKey::Misc, 0, Enhancement::COUNT)];
            c.seal = Seal::ALL[p.range(RngKey::Misc, 0, Seal::COUNT)];
            c.edition = [
                Edition::Base,
                Edition::Foil,
                Edition::Holographic,
                Edition::Polychrome,
            ][p.range(RngKey::Misc, 0, 4)];
        }
    }
    g.state().check_invariants().expect("start state");
    g
}

/// Plays `games` random-joker games; returns (total steps, games that reached a shop).
fn mass_run_jokers(first_seed: u64, games: u64) -> (u64, u64) {
    let ids = defined_ids();
    let (mut total, mut shops) = (0u64, 0u64);
    for seed in first_seed..first_seed + games {
        let mut g = new_joker_board_game(seed, &ids);
        let mut agent = match seed % 3 {
            0 => RandomAgent::new(seed),
            1 => RandomAgent::with_play_weight(seed, 6.0),
            _ => RandomAgent::with_play_weight(seed, 30.0),
        };
        // A third of the games take the score shortcut so round ends, cash-outs, shops,
        // end-of-round joker hooks and later antes are reached.
        let shortcut = seed % 3 == 2;
        let mut steps = 0usize;
        let mut saw_shop = false;
        while !g.is_terminal() {
            assert!(steps < MAX_STEPS, "seed {seed}: no termination");
            let a = agent.act(&g);
            if shortcut && matches!(a, Action::Play(_)) {
                let t = g.state().target;
                g.state_mut().round_score = t - 1.0;
            }
            if let Err(e) = g.step(a) {
                panic!("seed {seed}: legal-looking {a:?} rejected: {e}");
            }
            let s = g.state();
            s.check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
            assert!(
                s.jokers.iter().all(|j| j.value.is_finite()),
                "seed {seed}: non-finite joker value"
            );
            saw_shop |= s.phase == Phase::Shop;
            steps += 1;
        }
        assert!(g.outcome().is_some(), "seed {seed}");
        total += steps as u64;
        shops += u64::from(saw_shop);
    }
    (total, shops)
}

#[test]
fn ten_thousand_random_joker_board_games() {
    let (steps, shops) = mass_run_jokers(0, 10_000);
    assert!(steps > 40_000, "{steps}");
    assert!(shops > 3_000, "{shops} games reached a shop");
}

#[test]
#[ignore = "100k games; run with `cargo test --release -- --ignored`"]
fn hundred_thousand_random_joker_board_games() {
    let (steps, shops) = mass_run_jokers(2_000_000, 100_000);
    eprintln!("100k random-joker games: {steps} steps, {shops} reached a shop");
    assert!(steps > 400_000);
}

// ----- Phase 5.0: market, packs, vouchers and tags under random play -----------------------------------

/// Action-kind counters of the shop-heavy runs.
#[derive(Default, Debug)]
struct ShopTally {
    steps: u64,
    buy: u64,
    buy_use: u64,
    open_pack: u64,
    pick: u64,
    skip_pack: u64,
    voucher: u64,
    reroll: u64,
    reroll_boss: u64,
    move_joker: u64,
    skip_blind: u64,
}

/// Every game takes the score shortcut on Play (so shops come round after round) and a random
/// agent chooses among ALL legal actions. `effects_unimplemented` must stay 0.
fn shop_heavy_run(first_seed: u64, games: u64) -> ShopTally {
    let mut t = ShopTally::default();
    for seed in first_seed..first_seed + games {
        let mut g = Game::new(seed, GameConfig::default());
        // Start rich enough to buy things; every 4th game gets the boss-reroll vouchers.
        g.state_mut().money = 40;
        if seed % 4 == 0 {
            balatro_core::voucher::redeem(g.state_mut(), balatro_core::content::VoucherId::Retcon);
        }
        let mut agent = RandomAgent::with_play_weight(seed, 3.0);
        let mut steps = 0usize;
        while !g.is_terminal() {
            assert!(steps < MAX_STEPS * 4, "seed {seed}: no termination");
            let a = agent.act(&g);
            if matches!(a, Action::Play(_)) {
                let tg = g.state().target;
                g.state_mut().round_score = tg - 1.0;
            }
            if let Err(e) = g.step(a) {
                panic!("seed {seed}: legal-looking {a:?} rejected: {e}");
            }
            g.state()
                .check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
            assert_eq!(g.state().stats.effects_unimplemented, 0);
            match a {
                Action::Buy(_) => t.buy += 1,
                Action::BuyAndUse(_) => t.buy_use += 1,
                Action::OpenPack(_) => t.open_pack += 1,
                Action::PickPackItem { .. } => t.pick += 1,
                Action::SkipPack => t.skip_pack += 1,
                Action::BuyVoucher(_) => t.voucher += 1,
                Action::Reroll => t.reroll += 1,
                Action::RerollBoss => t.reroll_boss += 1,
                Action::MoveJoker { .. } => t.move_joker += 1,
                Action::SkipBlind => t.skip_blind += 1,
                _ => {}
            }
            steps += 1;
        }
        t.steps += steps as u64;
    }
    t
}

#[test]
fn ten_thousand_shop_heavy_games() {
    let t = shop_heavy_run(5_000_000, 2_000);
    for (name, n) in [
        ("buy", t.buy),
        ("buy_use", t.buy_use),
        ("open_pack", t.open_pack),
        ("pick", t.pick),
        ("skip_pack", t.skip_pack),
        ("voucher", t.voucher),
        ("reroll", t.reroll),
        ("move_joker", t.move_joker),
        ("skip_blind", t.skip_blind),
    ] {
        assert!(n > 0, "{name} was never used: {t:?}");
    }
}

#[test]
#[ignore = "100k shop-heavy games; run with `cargo test --release -- --ignored`"]
fn hundred_thousand_shop_heavy_games() {
    let t = shop_heavy_run(6_000_000, 100_000);
    eprintln!("100k shop-heavy games: {t:?}");
    assert!(t.pick > 100_000 && t.buy > 100_000 && t.reroll_boss > 0);
}

fn run_recorded(seed: u64, agent_seed: u64, jokers: bool) -> (Vec<Action>, GameState) {
    let mut g = new_game(seed, jokers);
    let mut agent = RandomAgent::with_play_weight(agent_seed, 4.0);
    let mut log = Vec::new();
    while !g.is_terminal() && log.len() < MAX_STEPS {
        let a = agent.act(&g);
        g.step(a).unwrap();
        log.push(a);
    }
    (log, g.into_state())
}

#[test]
fn same_seeds_same_trajectory_and_state() {
    for seed in 0..100u64 {
        let jokers = seed % 2 == 0;
        let (l1, s1) = run_recorded(seed, seed + 7, jokers);
        let (l2, s2) = run_recorded(seed, seed + 7, jokers);
        assert_eq!(l1, l2, "seed {seed}");
        assert_eq!(s1, s2, "seed {seed}");
    }
}

#[test]
fn greedy_is_deterministic() {
    for seed in 0..30u64 {
        let run = || {
            let mut g = new_game(seed, seed % 2 == 1);
            let mut a = GreedyAgent::new();
            let mut log = Vec::new();
            while !g.is_terminal() && log.len() < MAX_STEPS {
                let act = a.act(&g);
                g.step(act).unwrap();
                log.push(act);
            }
            (log, g.into_state())
        };
        assert_eq!(run(), run(), "seed {seed}");
    }
}

#[test]
fn clone_mid_game_continues_identically() {
    for seed in 0..200u64 {
        let mut g = new_game(seed, seed % 2 == 0);
        let mut agent = RandomAgent::with_play_weight(seed, 4.0);
        let cut = (seed % 17) as usize + 1;
        for _ in 0..cut {
            if g.is_terminal() {
                break;
            }
            let a = agent.act(&g);
            g.step(a).unwrap();
        }
        let mut c1 = g.clone();
        let mut c2 = g.clone();
        assert_eq!(c1.state(), c2.state());
        let mut agent2 = agent.clone();
        let mut steps = 0;
        while !c1.is_terminal() && steps < MAX_STEPS {
            let a = agent.act(&c1);
            let b = agent2.act(&c2);
            assert_eq!(a, b);
            c1.step(a).unwrap();
            c2.step(b).unwrap();
            assert_eq!(c1.state(), c2.state(), "seed {seed} step {steps}");
            steps += 1;
        }
    }
}

#[test]
fn invalid_actions_leave_state_untouched() {
    let mut rejected = 0u64;
    for seed in 0..300u64 {
        let mut g = new_game(seed, seed % 2 == 0);
        let mut fuzz = GameRng::new(seed ^ 0xF00D);
        let mut good = RandomAgent::with_play_weight(seed, 4.0);
        let mut steps = 0;
        while !g.is_terminal() && steps < MAX_STEPS {
            // A few arbitrary (mostly invalid) actions before each real one.
            for _ in 0..3 {
                let a = random_action(&mut fuzz);
                let before = g.state().clone();
                if g.step(a).is_err() {
                    rejected += 1;
                    assert_eq!(
                        *g.state(),
                        before,
                        "seed {seed}: {a:?} changed state on Err"
                    );
                } else {
                    g.state()
                        .check_invariants()
                        .unwrap_or_else(|e| panic!("seed {seed} after {a:?}: {e}"));
                }
                if g.is_terminal() {
                    break;
                }
            }
            if g.is_terminal() {
                break;
            }
            let a = good.act(&g);
            g.step(a).unwrap();
            steps += 1;
        }
        // Terminal game rejects everything without change.
        let before = g.state().clone();
        assert!(g.step(Action::EndShop).is_err());
        assert_eq!(*g.state(), before);
    }
    assert!(rejected > 1000);
}

fn random_action(r: &mut GameRng) -> Action {
    let m = |r: &mut GameRng| r.range(RngKey::Misc, 0, 1 << 16) as u16;
    let b = |r: &mut GameRng| r.range(RngKey::Misc, 0, 8) as u8;
    match r.range(RngKey::Misc, 0, 16) {
        0 => Action::SelectBlind,
        1 => Action::SkipBlind,
        2 | 3 => Action::Play(m(r) & 0xFF),
        4 => Action::Play(m(r)),
        5 | 6 => Action::Discard(m(r) & 0xFF),
        7 => Action::Discard(m(r)),
        8 => Action::SellJoker(b(r)),
        9 => Action::EndShop,
        10 => Action::UseConsumable {
            slot: b(r),
            targets: m(r),
        },
        11 => Action::SellConsumable(b(r)),
        12 => Action::Buy(b(r)),
        13 => Action::Reroll,
        14 => Action::MoveJoker {
            from: b(r),
            to: b(r),
        },
        _ => Action::PickPackItem {
            index: b(r),
            targets: m(r),
        },
    }
}
