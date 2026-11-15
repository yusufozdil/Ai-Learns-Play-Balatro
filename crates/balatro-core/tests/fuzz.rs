//! Phase 6B: large-scale fuzzing. Random/weighted/Greedy/Heuristic agents on randomized starting
//! states (jokers with editions and stickers, enhanced decks, consumables, advanced antes,
//! showdown bosses, endless mode, score shortcuts), with invariants checked after EVERY step,
//! legal-action validation, invalid-action fuzzing, replay determinism and clone-continuation.
//!
//! Small version (default package): ~100k steps. Full version (10M steps, all cores):
//! `cargo test --release -p balatro-core --test fuzz -- --ignored --nocapture fuzz_10m`
//! (`FUZZ_STEPS=<n>` overrides the step target, `FUZZ_THREADS=<n>` the thread count.)
//!
//! A failure prints `FUZZ FAIL seed=<s> step=<i> kind=<agent>: <message>`; reproduce with
//! `FUZZ_REPRO=<s> cargo test --release -p balatro-core --test fuzz -- --ignored --nocapture repro`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, HeuristicAgent, RandomAgent};
use balatro_core::blinds;
use balatro_core::consumable::ConsumableInstance;
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::content::{BossId, ConsumableId};
use balatro_core::enums::{Edition, Enhancement, Phase, Seal};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::serialize::Replay;
use balatro_core::state::{DeckKind, GameConfig, GameState, Outcome, StakeKind};

/// Steps after which a game is cut off (an error for plain games that should terminate).
const STEP_CAP: u64 = 24_000;

/// Weighted-by-kind random agent: favours market, packs, consumables, MoveJoker, RerollBoss.
struct WeightedAgent {
    rng: GameRng,
    legal: Vec<Action>,
    w: Vec<f64>,
}

impl WeightedAgent {
    fn new(seed: u64) -> Self {
        WeightedAgent {
            rng: GameRng::new(seed ^ 0x57E1_6417),
            legal: Vec::new(),
            w: Vec::new(),
        }
    }
}

impl Agent for WeightedAgent {
    fn act(&mut self, game: &Game) -> Action {
        game.legal_actions_into(&mut self.legal);
        if self.legal.is_empty() {
            return Action::EndShop;
        }
        self.w.clear();
        self.w.extend(self.legal.iter().map(|a| match a {
            Action::Play(_) => 2.0,
            Action::Discard(_) => 1.0,
            Action::EndShop => 0.2,
            Action::SkipPack => 0.4,
            Action::SelectBlind => 1.0,
            Action::SkipBlind => 1.5,
            Action::SellJoker(_) | Action::SellConsumable(_) => 1.0,
            _ => 5.0,
        }));
        let i = self.rng.weighted_index(RngKey::Misc, &self.w).unwrap_or(0);
        self.legal[i]
    }
}

enum AnyAgent {
    Random(RandomAgent),
    Weighted(WeightedAgent),
    Greedy(GreedyAgent),
    Heuristic(Box<HeuristicAgent>),
}

impl AnyAgent {
    fn act(&mut self, g: &Game) -> Action {
        match self {
            AnyAgent::Random(a) => a.act(g),
            AnyAgent::Weighted(a) => a.act(g),
            AnyAgent::Greedy(a) => a.act(g),
            AnyAgent::Heuristic(a) => a.act(g),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            AnyAgent::Random(_) => "random",
            AnyAgent::Weighted(_) => "weighted",
            AnyAgent::Greedy(_) => "greedy",
            AnyAgent::Heuristic(_) => "heuristic",
        }
    }
}

fn defined_ids() -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs().map(|d| d.id).collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

#[derive(Default, Clone, Copy, Debug)]
struct Tally {
    steps: u64,
    games: u64,
    random: u64,
    weighted: u64,
    greedy: u64,
    heuristic: u64,
    invalid_tried: u64,
    invalid_rejected: u64,
    legal_checked: u64,
    replays: u64,
    clone_checks: u64,
    endless: u64,
    advanced_ante: u64,
    max_ante: i32,
    wins: u64,
    truncated: u64,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.steps += o.steps;
        self.games += o.games;
        self.random += o.random;
        self.weighted += o.weighted;
        self.greedy += o.greedy;
        self.heuristic += o.heuristic;
        self.invalid_tried += o.invalid_tried;
        self.invalid_rejected += o.invalid_rejected;
        self.legal_checked += o.legal_checked;
        self.replays += o.replays;
        self.clone_checks += o.clone_checks;
        self.endless += o.endless;
        self.advanced_ante += o.advanced_ante;
        self.max_ante = self.max_ante.max(o.max_ante);
        self.wins += o.wins;
        self.truncated += o.truncated;
    }
}

struct Fail {
    seed: u64,
    step: u64,
    kind: &'static str,
    msg: String,
}

impl Fail {
    fn line(&self) -> String {
        let step = if self.step == u64::MAX {
            "?".to_string()
        } else {
            self.step.to_string()
        };
        format!(
            "FUZZ FAIL seed={} step={step} kind={}: {}",
            self.seed, self.kind, self.msg
        )
    }
}

fn random_action(r: &mut GameRng) -> Action {
    let m = |r: &mut GameRng| r.range(RngKey::Misc, 0, 1 << 16) as u32;
    let b = |r: &mut GameRng| r.range(RngKey::Misc, 0, 10) as u8;
    match r.range(RngKey::Misc, 0, 22) {
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
        15 => Action::PickPackItem {
            index: b(r),
            targets: m(r),
        },
        16 => Action::BuyAndUse(b(r)),
        17 => Action::OpenPack(b(r)),
        18 => Action::BuyVoucher(b(r)),
        19 => Action::SkipPack,
        20 => Action::RerollBoss,
        _ => Action::UseConsumable {
            slot: b(r),
            targets: m(r) & 0x1F,
        },
    }
}

/// Everything about one fuzz game, derived from the seed.
struct Setup {
    game: Game,
    agent: AnyAgent,
    /// Probability (percent) that a Play action first sets the round score to target-1.
    shortcut_pct: usize,
    /// Untouched fresh run: replayable from (seed, config, actions).
    pure: bool,
    endless: bool,
    advanced: bool,
}

fn setup(seed: u64, ids: &[JokerId]) -> Setup {
    let mut p = GameRng::new(seed ^ 0xF022_5EED);
    let pct = |r: &mut GameRng, n: usize| r.range(RngKey::Misc, 0, n);
    let kind = pct(&mut p, 100);
    let pure = pct(&mut p, 10) == 0;
    let endless = !pure && pct(&mut p, 4) == 0;
    // Deck and stake come from their own stream (the other draws keep their sequence). Half of
    // the games stay Red / White; the rest sample every deck and stake.
    let mut dr = GameRng::new(seed ^ 0xDECC_57A4);
    let (deck, stake) = if dr.range(RngKey::Misc, 0, 2) == 0 {
        (DeckKind::default(), StakeKind::default())
    } else {
        (
            DeckKind::ALL[dr.range(RngKey::Misc, 0, DeckKind::COUNT)],
            StakeKind::ALL[dr.range(RngKey::Misc, 0, StakeKind::COUNT)],
        )
    };
    let config = GameConfig {
        deck,
        stake,
        endless,
    };
    let mut g = Game::new(seed, config);
    let mut advanced = false;
    let mut shortcut_pct = 0;
    if !pure {
        shortcut_pct = [0, 0, 20, 60, 100][pct(&mut p, 5)];
        for _ in 0..pct(&mut p, 7) {
            let mut j = JokerInstance::new(0, ids[pct(&mut p, ids.len())]);
            j.edition = match pct(&mut p, 10) {
                0 => Edition::Foil,
                1 => Edition::Holographic,
                2 => Edition::Polychrome,
                3 => Edition::Negative,
                _ => Edition::Base,
            };
            if let Some(d) = j.def() {
                if d.eternal_compatible && pct(&mut p, 8) == 0 {
                    j.stickers.eternal = true;
                } else if pct(&mut p, 8) == 0 {
                    j.stickers.perishable = true;
                    j.stickers.perish_rounds_left = 5;
                }
                if pct(&mut p, 8) == 0 {
                    j.stickers.rental = true;
                }
            }
            let _ = g.add_joker(j);
        }
        if pct(&mut p, 2) == 0 {
            let density = 2 + pct(&mut p, 5);
            for c in &mut g.state_mut().deck.draw_pile {
                if pct(&mut p, density) != 0 {
                    continue;
                }
                c.enhancement = Enhancement::ALL[pct(&mut p, Enhancement::COUNT)];
                c.seal = Seal::ALL[pct(&mut p, Seal::COUNT)];
                c.edition = [
                    Edition::Base,
                    Edition::Foil,
                    Edition::Holographic,
                    Edition::Polychrome,
                ][pct(&mut p, 4)];
            }
        }
        for _ in 0..pct(&mut p, 4) {
            let mut c =
                ConsumableInstance::new(ConsumableId::ALL[pct(&mut p, ConsumableId::COUNT)]);
            c.negative = pct(&mut p, 6) == 0;
            let _ = g.add_consumable(c);
        }
        if pct(&mut p, 3) == 0 {
            g.state_mut().money = pct(&mut p, 300) as i32;
        }
        // Advanced ante / random boss (the run still sits at BlindSelect).
        if pct(&mut p, 4) == 0 {
            let s = g.state_mut();
            s.ante = [3, 5, 7, 8, 8, 9, 12, 16, 17, 25][pct(&mut p, 10)];
            if pct(&mut p, 2) == 0 {
                s.boss = Some(BossId::ALL[pct(&mut p, BossId::COUNT)]);
            }
            s.target =
                blinds::requirement(s.ante, s.blind, s.boss_mult(), s.config.stake.scaling())
                    * s.config.deck.def().blind_size_mult;
            advanced = true;
        }
    }
    g.state().check_invariants().expect("start state");
    let agent = match kind {
        0..=34 => AnyAgent::Random(RandomAgent::with_play_weight(
            seed,
            [1.0, 1.0, 4.0, 15.0][pct(&mut p, 4)],
        )),
        35..=64 => AnyAgent::Weighted(WeightedAgent::new(seed)),
        65..=97 => AnyAgent::Greedy(GreedyAgent::new()),
        _ => AnyAgent::Heuristic(Box::new(HeuristicAgent::new(seed))),
    };
    Setup {
        game: g,
        agent,
        shortcut_pct,
        pure,
        endless,
        advanced,
    }
}

fn check(s: &GameState) -> Result<(), String> {
    s.check_invariants()?;
    if s.round_score.is_nan() || s.stats.best_hand.is_nan() {
        return Err(format!(
            "NaN score {} / {}",
            s.round_score, s.stats.best_hand
        ));
    }
    if s.target.is_nan() || s.target < 0.0 {
        return Err(format!("bad target {}", s.target));
    }
    if s.money < -1_000_000 {
        return Err(format!("money out of range: {}", s.money));
    }
    if s.jokers.iter().any(|j| !j.value.is_finite()) {
        return Err("non-finite joker value".into());
    }
    if s.stats.effects_unimplemented != 0 {
        return Err("effects_unimplemented != 0".into());
    }
    Ok(())
}

/// Plays one fuzz game. `Err` carries the first failure (step index and message).
fn play_game(seed: u64, ids: &[JokerId], t: &mut Tally) -> Result<(), Fail> {
    let Setup {
        mut game,
        mut agent,
        shortcut_pct,
        pure,
        endless,
        advanced,
    } = setup(seed, ids);
    let kind = agent.name();
    let mut cnt = 0u64;
    macro_rules! fail {
        ($($arg:tt)*) => {
            return Err(Fail { seed, step: cnt, kind, msg: format!($($arg)*) })
        };
    }
    let mut r = GameRng::new(seed ^ 0xBADC_0FFE);
    let truncatable = endless || shortcut_pct > 0 || advanced;
    let mut replay = Replay::record(seed, game.state().config);
    let clone_at = if r.range(RngKey::Misc, 0, 40) == 0 {
        r.range(RngKey::Misc, 1, 60) as u64
    } else {
        u64::MAX
    };
    let mut clone_left = 0u32;
    let mut twin: Option<Game> = None;
    while !game.is_terminal() {
        if cnt >= STEP_CAP {
            if truncatable {
                t.truncated += 1;
                break;
            }
            fail!("no termination within {STEP_CAP} steps");
        }
        // Invalid-action fuzz (full state compare on 1 in 16 tries).
        if r.range(RngKey::Misc, 0, 6) == 0 {
            let a = random_action(&mut r);
            t.invalid_tried += 1;
            let snap = (r.range(RngKey::Misc, 0, 16) == 0).then(|| game.state().clone());
            if game.validate(a).is_err() {
                if game.step(a).is_ok() {
                    fail!("{a:?} rejected by validate but accepted by step");
                }
                t.invalid_rejected += 1;
                if let Some(b) = snap
                    && *game.state() != b
                {
                    fail!("{a:?} changed state on Err");
                }
            } else {
                // Legal after all: take it (recorded; the twin is only started on the main path).
                if twin.is_some() {
                    continue;
                }
                if let Err(e) = game.step(a) {
                    fail!("{a:?} validated but step failed: {e}");
                }
                replay.push(a);
                cnt += 1;
                t.steps += 1;
                if let Err(e) = check(game.state()) {
                    fail!("after {a:?}: {e}");
                }
                if game.is_terminal() {
                    break;
                }
            }
        }
        // Sampled: every legal action validates.
        if cnt.is_multiple_of(24) {
            for a in game.legal_actions() {
                if let Err(e) = game.validate(a) {
                    fail!("legal action {a:?} fails validate: {e}");
                }
            }
            t.legal_checked += 1;
        }
        let a = agent.act(&game);
        if cnt == clone_at && twin.is_none() {
            twin = Some(game.clone());
            clone_left = 150;
            t.clone_checks += 1;
        }
        if shortcut_pct > 0
            && matches!(a, Action::Play(_))
            && r.range(RngKey::Misc, 0, 100) < shortcut_pct
        {
            let tg = game.state().target;
            if tg.is_finite() {
                game.state_mut().round_score = tg - 1.0;
                if let Some(c) = twin.as_mut() {
                    c.state_mut().round_score = tg - 1.0;
                }
            }
        }
        if let Err(e) = game.step(a) {
            fail!("legal-looking {a:?} rejected: {e}");
        }
        replay.push(a);
        if let Some(c) = twin.as_mut() {
            if let Err(e) = c.step(a) {
                fail!("clone rejected {a:?}: {e}");
            }
            if c.state() != game.state() {
                fail!("clone diverged after {a:?}");
            }
            clone_left -= 1;
            if clone_left == 0 || game.is_terminal() {
                twin = None;
            }
        }
        cnt += 1;
        t.steps += 1;
        if let Err(e) = check(game.state()) {
            fail!("after {a:?}: {e}");
        }
        if matches!(game.state().phase, Phase::GameOver | Phase::Won) != game.is_terminal() {
            fail!("phase/terminal mismatch");
        }
    }
    if game.is_terminal() && game.outcome().is_none() {
        fail!("terminal without outcome");
    }
    t.games += 1;
    t.max_ante = t.max_ante.max(game.state().ante);
    t.wins += u64::from(matches!(game.outcome(), Some(Outcome::Win)));
    t.endless += u64::from(endless);
    t.advanced_ante += u64::from(advanced);
    match kind {
        "random" => t.random += 1,
        "weighted" => t.weighted += 1,
        "greedy" => t.greedy += 1,
        _ => t.heuristic += 1,
    }
    // Replay determinism for untouched runs (only accepted actions were recorded).
    if pure {
        match replay.play() {
            Ok(g2) => {
                if g2.state() != game.state() {
                    fail!("replay produced a different final state");
                }
                match Replay::from_json(&replay.to_json()) {
                    Ok(r2) if r2 == replay => {}
                    _ => fail!("replay JSON round trip differs"),
                }
            }
            Err(e) => fail!("replay failed: {e}"),
        }
        t.replays += 1;
    }
    Ok(())
}

/// Runs one seed with panic capture; a panic becomes a failure carrying the seed.
fn run_seed(seed: u64, ids: &[JokerId], t: &mut Tally) -> Result<(), Fail> {
    let mut local = *t;
    let res = catch_unwind(AssertUnwindSafe(|| {
        let r = play_game(seed, ids, &mut local);
        (r, local)
    }));
    match res {
        Ok((r, l)) => {
            *t = l;
            r
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic".into());
            Err(Fail {
                seed,
                step: u64::MAX,
                kind: "panic",
                msg,
            })
        }
    }
}

/// Fuzz until `target_steps` steps (games indexed from `first_seed`), on `threads` threads.
fn fuzz(first_seed: u64, target_steps: u64, threads: usize) -> (Tally, Vec<Fail>) {
    let next = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let total = Mutex::new(Tally::default());
    let fails = Mutex::new(Vec::<Fail>::new());
    let ids = defined_ids();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut t = Tally::default();
                while done.load(Ordering::Relaxed) < target_steps {
                    let seed = first_seed + next.fetch_add(1, Ordering::Relaxed);
                    let before = t.steps;
                    if let Err(f) = run_seed(seed, &ids, &mut t) {
                        eprintln!("{}", f.line());
                        fails.lock().unwrap().push(f);
                    }
                    done.fetch_add(t.steps - before, Ordering::Relaxed);
                    if fails.lock().unwrap().len() > 20 {
                        break;
                    }
                }
                total.lock().unwrap().add(&t);
            });
        }
    });
    (total.into_inner().unwrap(), fails.into_inner().unwrap())
}

fn report(t: &Tally, fails: &[Fail]) {
    let f: Vec<String> = fails.iter().map(Fail::line).collect();
    assert!(
        f.is_empty(),
        "fuzz failures ({}):\n{}",
        f.len(),
        f.join("\n")
    );
    assert!(t.invalid_rejected > 0 && t.replays > 0 && t.legal_checked > 0);
}

#[test]
fn fuzz_small_100k_steps() {
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get().min(4));
    let (t, fails) = fuzz(10_000_000, 100_000, threads);
    eprintln!("small fuzz: {t:?}");
    report(&t, &fails);
    assert!(t.steps >= 100_000);
}

#[test]
#[ignore = "10M steps, all cores; run in release with --ignored --nocapture"]
fn fuzz_10m_steps() {
    let steps = std::env::var("FUZZ_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000_000u64);
    let threads = std::env::var("FUZZ_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let start = Instant::now();
    let (t, fails) = fuzz(100_000_000, steps, threads);
    eprintln!(
        "fuzz: {} steps, {} games, {threads} threads, {:.1}s\n{t:?}",
        t.steps,
        t.games,
        start.elapsed().as_secs_f64()
    );
    report(&t, &fails);
    assert!(t.steps >= steps);
}

#[test]
#[ignore = "reproduce one failing seed: FUZZ_REPRO=<seed>"]
fn repro() {
    let seed: u64 = std::env::var("FUZZ_REPRO")
        .ok()
        .and_then(|v| v.parse().ok())
        .expect("set FUZZ_REPRO=<seed>");
    let ids = defined_ids();
    let mut t = Tally::default();
    if let Err(f) = run_seed(seed, &ids, &mut t) {
        panic!("{}", f.line());
    }
    eprintln!("seed {seed} clean: {t:?}");
}
