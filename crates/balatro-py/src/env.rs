//! One environment: a run plus everything the RL interface keeps next to it (episode counters,
//! the cached masks, the sampler RNG, the teacher agent), the reward and the structured-action
//! step. Pure Rust (no pyo3): the pool (`vec_env.rs`) and the debug class (`single.rs`) share it.
//!
//! Rules: `docs/design/rl_interface.md` sections 4 (actions), 7 (reward), 8 (done, auto-reset).
//! An action that is malformed or rejected by `validate` is a no-op flagged `invalid`; the only
//! `Err` values are internal inconsistencies.

use crate::encode::{MaskView, SelRow, write_masks, write_obs};
use crate::layout::{Row, sel_row};
use balatro_core::action_mask::{ActionKind, CardSelect};
use balatro_core::actions::Action;
use balatro_core::agents::{Agent, HeuristicAgent};
use balatro_core::blinds::BlindKind;
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::enums::Phase;
use balatro_core::game::{Game, StepInfo};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::RngKey;
use balatro_core::state::{DeckKind, GameConfig, GameState, Outcome, StakeKind};
use std::sync::Arc;

/// Not done.
pub const DONE_NO: u8 = 0;
/// The run was lost.
pub const DONE_LOSS: u8 = 1;
/// The run was won (ante-8 boss beaten).
pub const DONE_WIN: u8 = 2;
/// The episode hit `max_episode_steps` and was cut off (neither win nor loss).
pub const DONE_TRUNCATED: u8 = 3;

/// Default cap on steps per episode; longer episodes are truncated (done code 3).
pub const DEFAULT_MAX_EPISODE_STEPS: u32 = 10_000;

/// Tunable reward terms (section 7 of the design document).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardConfig {
    /// A blind is beaten.
    pub blind: f32,
    /// The boss blind of an ante is beaten (on top of `blind`).
    pub ante: f32,
    /// The run is won.
    pub win: f32,
    /// The run is lost.
    pub loss: f32,
    /// Weight of the in-round progress shaping `d min(score / target, 1)` (0 = off).
    pub shaping: f32,
    /// Reward of an invalid action.
    pub invalid: f32,
    /// Reward (cost when negative) of every valid `MoveJoker` action. Reordering is the only
    /// action that changes nothing but the joker order, so a policy can swap two jokers back and
    /// forth forever; a small negative value breaks that loop (0 = off).
    pub reorder: f32,
    /// Weight of `ln(min(score / target, margin_cap))` added when a blind is beaten (0 = off).
    pub margin: f32,
    /// Cap of the score / target ratio inside the margin term.
    pub margin_cap: f32,
    /// Reward (cost when negative) of every played hand a boss rejected (score 0): The Psychic,
    /// The Eye, The Mouth (0 = off).
    pub zero_play: f32,
    /// Endless runs only (`GameConfig.endless`): the episode ends (done, not truncated) after the
    /// boss of this ante is beaten. Lives here so the env constructors keep their signatures.
    pub endless_max_ante: i32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            blind: 1.0,
            ante: 1.0,
            win: 10.0,
            loss: 0.0,
            shaping: 0.0,
            invalid: -0.05,
            reorder: 0.0,
            margin: 0.0,
            margin_cap: 4.0,
            zero_play: 0.0,
            endless_max_ante: 12,
        }
    }
}

impl RewardConfig {
    /// Set a term by name; `false` for an unknown name.
    pub fn set(&mut self, key: &str, v: f32) -> bool {
        match key {
            "blind" => self.blind = v,
            "ante" => self.ante = v,
            "win" => self.win = v,
            "loss" => self.loss = v,
            "shaping" => self.shaping = v,
            "invalid" => self.invalid = v,
            "reorder" => self.reorder = v,
            "margin" => self.margin = v,
            "margin_cap" => self.margin_cap = v,
            "zero_play" => self.zero_play = v,
            "endless_max_ante" => self.endless_max_ante = v as i32,
            _ => return false,
        }
        true
    }
}

/// Most distinct `(deck, stake)` pairs of a [`ConfigPool`].
pub const MAX_CONFIGS: usize = DeckKind::COUNT * StakeKind::COUNT;

/// A set of `(deck, stake)` pairs; every episode of every env draws one uniformly (a pure
/// function of `(base_seed, env, episode)`). Empty = the fixed `Shared::config` for every episode.
#[derive(Debug, Clone, Copy)]
pub struct ConfigPool {
    len: usize,
    items: [(DeckKind, StakeKind); MAX_CONFIGS],
    /// Cumulative sampling weights (`set_weights`); `weighted == false`: uniform (`hash % len`,
    /// the original, bit-identical rule).
    weighted: bool,
    cum: [f64; MAX_CONFIGS],
}

impl ConfigPool {
    /// No sampling.
    pub const EMPTY: ConfigPool = ConfigPool {
        len: 0,
        items: [(DeckKind::Red, StakeKind::White); MAX_CONFIGS],
        weighted: false,
        cum: [0.0; MAX_CONFIGS],
    };

    /// Cartesian product of `decks` and `stakes` (duplicates dropped, order kept).
    pub fn product(decks: &[DeckKind], stakes: &[StakeKind]) -> ConfigPool {
        let mut p = ConfigPool::EMPTY;
        for &d in decks {
            for &s in stakes {
                if !p.items[..p.len].contains(&(d, s)) {
                    p.items[p.len] = (d, s);
                    p.len += 1;
                }
            }
        }
        p
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The pairs in sampling order.
    pub fn items(&self) -> &[(DeckKind, StakeKind)] {
        &self.items[..self.len]
    }

    /// Sampling weights, one per pair in `items()` order (>= 0, finite, positive sum). They apply
    /// to the episodes that start after the call (an episode's config is still a pure function of
    /// `(base_seed, env, episode)` and the weights at that time). `None` restores the uniform rule.
    pub fn set_weights(&mut self, w: Option<&[f64]>) -> Result<(), String> {
        let Some(w) = w else {
            self.weighted = false;
            return Ok(());
        };
        if w.len() != self.len {
            return Err(format!("expected {} weights, got {}", self.len, w.len()));
        }
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
            return Err("weights must be finite, >= 0, with a positive sum".into());
        }
        let mut acc = 0.0;
        for (c, x) in self.cum.iter_mut().zip(w) {
            acc += x;
            *c = acc;
        }
        self.weighted = true;
        Ok(())
    }

    /// The pair for the hash `h` (uniform, or by the cumulative weights).
    fn pick(&self, h: u64) -> (DeckKind, StakeKind) {
        if !self.weighted {
            return self.items[(h % self.len() as u64) as usize];
        }
        let total = self.cum[self.len - 1];
        let t = (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * total;
        let i = self.cum[..self.len]
            .iter()
            .position(|&c| t < c)
            .unwrap_or(self.len - 1);
        self.items[i]
    }

    /// No pairs (the fixed config is used).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Settings shared by all envs of a pool.
#[derive(Debug, Clone)]
pub struct Shared {
    /// Run options (the fixed config; `endless` also applies to sampled configs).
    pub config: GameConfig,
    /// Per-episode `(deck, stake)` sampling (empty = off, `config` always).
    pub configs: ConfigPool,
    /// Steps per episode before truncation.
    pub max_steps: u32,
    /// Base of the per-env, per-episode seeds.
    pub base_seed: u64,
    /// Reward terms.
    pub reward: RewardConfig,
    /// Non-natural episode starts (scenarios, boss-hall states); off by default.
    pub reset: ResetCfg,
}

/// Where an episode started (`Env::origin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// A fresh run from a seed (also after `import_state`: that is flagged `imported`).
    Natural,
    /// A randomized plausible mid-run state (`balatro_core::scenario`).
    Scenario,
    /// A boss-hall pool state with freshly re-sampled hidden information.
    Hall,
}

/// Non-natural reset sources. All off (`0.0`) = the original auto-reset, bit for bit.
#[derive(Debug, Clone, Default)]
pub struct ResetCfg {
    /// Probability that an auto-reset starts from a scenario (a pure function of
    /// `(base_seed, env, episode)`).
    pub scenario_prob: f64,
    /// Probability that an auto-reset starts from a pool state (only with a non-empty pool).
    pub hall_prob: f64,
    /// Parsed pool states (validated once in `EnvPool::set_reset_pool`).
    pub hall: Arc<Vec<GameState>>,
    /// Indices into `hall` per stake (`StakeKind::index`): an auto-reset only draws a pool state
    /// of the stake the normal (deck, stake) sampling picked, so `set_config_weights` keeps
    /// governing the stake mix; with no state of that stake the reset is a natural one.
    pub hall_by_stake: Arc<Vec<Vec<u32>>>,
}

impl Shared {
    /// Run options of episode `episode` of env `env`.
    pub fn config_for(&self, env: u64, episode: u64) -> GameConfig {
        if self.configs.is_empty() {
            return self.config;
        }
        let h = splitmix64(derive_seed(self.base_seed, env, episode) ^ CONFIG_SALT);
        let (deck, stake) = self.configs.pick(h);
        GameConfig {
            deck,
            stake,
            endless: self.config.endless,
        }
    }
}

/// Salt that separates the config choice from the game seeds.
const CONFIG_SALT: u64 = 0xC0F1_6000_0000_0001;
/// Salt of the reset-source choice (scenario / hall) and of the hall pool index.
const RESET_SALT: u64 = 0x5E5E_7000_0000_0001;
/// Salt of the fresh seed a hall state is re-sampled with.
const HALL_SALT: u64 = 0x4A11_0000_0000_0001;

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of env `env` in episode `episode`. Pure function of its arguments.
pub fn derive_seed(base: u64, env: u64, episode: u64) -> u64 {
    let s = splitmix64(base);
    let s = splitmix64(s ^ env.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    splitmix64(s ^ episode.wrapping_mul(0xA076_1D64_78BD_642F))
}

/// Salt that separates the teacher agent's seed stream from the game seeds.
const TEACHER_SALT: u64 = 0x7EAC_4E52_0000_0001;

/// Uniform in `[0, n)` from a SplitMix64 stream (`n >= 1`).
fn below(rng: &mut u64, n: u32) -> u32 {
    *rng = splitmix64(*rng);
    ((u128::from(*rng) * u128::from(n)) >> 64) as u32
}

/// The `k`-th (0-based) set bit of `bits`.
fn nth_bit(mut bits: u32, k: u32) -> u32 {
    for _ in 0..k {
        bits &= bits - 1;
    }
    bits.trailing_zeros()
}

/// Summary of one finished episode (for logging).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Finished {
    /// Env index.
    pub env: u32,
    /// Ante reached.
    pub ante: i32,
    /// Done code (`DONE_LOSS`, `DONE_WIN`, `DONE_TRUNCATED`).
    pub code: u8,
    /// Steps taken (invalid actions included).
    pub steps: u32,
    /// Sum of the rewards.
    pub ret: f32,
    /// Blinds beaten.
    pub rounds_won: u32,
    /// Invalid actions taken.
    pub invalid: u32,
    /// Deck of the run (`DeckKind::index`).
    pub deck: DeckKind,
    /// Stake of the run.
    pub stake: StakeKind,
    /// The episode started from an imported state (`Env::import_game`), not from a seed: it
    /// began mid-run and its steps / return count from the import only.
    pub imported: bool,
    /// The episode started from a randomized scenario (`ResetCfg::scenario_prob`).
    pub scenario: bool,
    /// The episode started from a boss-hall pool state (`ResetCfg::hall_prob`).
    pub hall: bool,
    /// The start joker the episode began with (`Env::start_joker`; `None` for imported episodes).
    pub start_joker: Option<JokerId>,
}

/// Progress `min(score / target, 1)` of the round in play (0 outside `Play`).
fn progress(s: &GameState) -> f32 {
    if s.phase == Phase::Play && s.target.is_finite() && s.target > 0.0 {
        (s.round_score / s.target).clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// Parse `[kind, arg0, arg1, cards]` into an action; `None` if malformed.
pub fn parse_action(p: [i64; 4]) -> Option<Action> {
    let kind = ActionKind::from_index(usize::try_from(p[0]).ok()?)?;
    let operand = |v: i64| u8::try_from(v).ok();
    let (a0, a1) = (
        if kind.has_arg0() { operand(p[1])? } else { 0 },
        if kind.has_arg1() { operand(p[2])? } else { 0 },
    );
    let cards = if kind.may_take_cards() {
        u32::try_from(p[3]).ok()?
    } else {
        0
    };
    Some(Action::from_parts(kind, a0, a1, cards))
}

/// The structured form of an action as an `int64` row.
pub fn action_row(a: &Action) -> [i64; 4] {
    let (k, a0, a1, m) = a.parts();
    [k.index() as i64, i64::from(a0), i64::from(a1), i64::from(m)]
}

/// One run with its interface state.
pub struct Env {
    /// Env index (seeds).
    pub idx: u32,
    /// The game.
    pub game: Game,
    /// Masks of the current state (recomputed after every step / reset).
    pub view: MaskView,
    episode: u64,
    steps: u32,
    ep_return: f32,
    ep_invalid: u32,
    /// The current episode started from an imported state.
    imported: bool,
    /// How the current episode started.
    origin: Origin,
    /// Joker added at every (hard or auto) reset; `None` = none (bit-identical default).
    pub start_joker: Option<JokerId>,
    /// The start joker the CURRENT episode actually began with (`start_joker` may have been
    /// changed since).
    ep_start_joker: Option<JokerId>,
    /// SplitMix64 state of `random_actions`.
    act_rng: u64,
    /// The teacher (heuristic) agent, created on first use, dropped on every reset.
    teacher: Option<Box<HeuristicAgent>>,
    /// Finished episodes not yet drained.
    pub finished: Vec<Finished>,
}

/// What one step produced.
#[derive(Debug, Clone, Copy)]
pub struct StepOut {
    /// Reward of the transition.
    pub reward: f32,
    /// Done code.
    pub done: u8,
    /// The action was invalid (no-op).
    pub invalid: bool,
}

impl Env {
    /// New env at episode 0.
    pub fn new(idx: u32, sh: &Shared) -> Env {
        let game = Game::new(
            derive_seed(sh.base_seed, u64::from(idx), 0),
            sh.config_for(u64::from(idx), 0),
        );
        let view = MaskView::compute(game.state());
        Env {
            idx,
            game,
            view,
            episode: 0,
            steps: 0,
            ep_return: 0.0,
            ep_invalid: 0,
            imported: false,
            origin: Origin::Natural,
            start_joker: None,
            ep_start_joker: None,
            act_rng: splitmix64(derive_seed(sh.base_seed, u64::from(idx), u64::MAX)),
            teacher: None,
            finished: Vec::new(),
        }
    }

    /// Recompute the cached masks (after the state was changed from outside).
    pub fn refresh(&mut self) {
        self.view = MaskView::compute(self.game.state());
    }

    /// Back to episode 0 with fresh seeds (the `reset()` entry point).
    pub fn hard_reset(&mut self, sh: &Shared) {
        let idx = u64::from(self.idx);
        self.game
            .reset_with_seed(derive_seed(sh.base_seed, idx, 0), sh.config_for(idx, 0));
        self.episode = 0;
        self.steps = 0;
        self.ep_return = 0.0;
        self.ep_invalid = 0;
        self.imported = false;
        self.origin = Origin::Natural;
        self.act_rng = splitmix64(derive_seed(sh.base_seed, idx, u64::MAX));
        self.teacher = None;
        self.finished.clear();
        self.apply_start_joker();
        self.refresh();
    }

    /// An independent copy of this env (state, counters, sampler stream, pending finished
    /// records). The teacher agent is not copied: the clone creates a new one on first use.
    pub fn fork(&self) -> Env {
        Env {
            idx: self.idx,
            game: self.game.clone(),
            view: self.view,
            episode: self.episode,
            steps: self.steps,
            ep_return: self.ep_return,
            ep_invalid: self.ep_invalid,
            imported: self.imported,
            origin: self.origin,
            start_joker: self.start_joker,
            ep_start_joker: self.ep_start_joker,
            act_rng: self.act_rng,
            teacher: None,
            finished: self.finished.clone(),
        }
    }

    /// Give the fresh run its start joker (no-op without one).
    fn apply_start_joker(&mut self) {
        self.ep_start_joker = self.start_joker;
        if let Some(id) = self.start_joker {
            // A fresh run has an empty joker row, so there is always room.
            let _ = self.game.add_joker(JokerInstance::new(0, id));
        }
    }

    /// Replace the run mid-episode by `game` (branching; see `docs/design/rl_interface.md`).
    /// The imported run starts a fresh episode as far as the interface is concerned: step counter,
    /// return and invalid counter restart at 0, the teacher agent is dropped, the episode is
    /// flagged `imported` in its `Finished` record. The episode counter is NOT advanced: the next
    /// auto-reset after the imported episode is the same one the replaced episode would have
    /// reached (the abandoned episode produces no `Finished` record), so the seeds of the other
    /// episodes of this env do not depend on whether an import happened. `act_rng` keeps running.
    pub fn import_game(&mut self, game: Game, sh: &Shared) -> Result<(), String> {
        if game.is_terminal() {
            return Err("cannot import a finished run".into());
        }
        if game.state().config.endless != sh.config.endless {
            return Err(format!(
                "state has endless={} but this env was built with endless={} (done codes and \
                 the ante cap depend on it)",
                game.state().config.endless,
                sh.config.endless
            ));
        }
        let view = MaskView::compute(game.state());
        if view.kinds == 0 {
            return Err("imported state has no addressable action".into());
        }
        // Everything is validated: from here on the import cannot fail.
        self.game = game;
        self.view = view;
        self.ep_start_joker = None;
        self.steps = 0;
        self.ep_return = 0.0;
        self.ep_invalid = 0;
        self.imported = true;
        self.origin = Origin::Natural;
        self.teacher = None;
        Ok(())
    }

    /// Start the next episode (auto-reset) with a derived seed; with the reset sources of
    /// `sh.reset` enabled, possibly from a scenario or a boss-hall state (a natural reset: the
    /// previous episode already ended with `done`).
    fn next_episode(&mut self, sh: &Shared) {
        self.episode += 1;
        self.steps = 0;
        self.ep_return = 0.0;
        self.ep_invalid = 0;
        self.imported = false;
        self.origin = Origin::Natural;
        self.teacher = None;
        let idx = u64::from(self.idx);
        let seed = derive_seed(sh.base_seed, idx, self.episode);
        let config = sh.config_for(idx, self.episode);
        let rc = &sh.reset;
        if rc.scenario_prob > 0.0 || (rc.hall_prob > 0.0 && !rc.hall.is_empty()) {
            let h = splitmix64(seed ^ RESET_SALT);
            let u = (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            if u < rc.scenario_prob {
                self.game = balatro_core::scenario::random_scenario(seed, config);
                self.origin = Origin::Scenario;
                self.ep_start_joker = None;
                self.refresh();
                return;
            }
            let same_stake = rc
                .hall_by_stake
                .get(config.stake.index())
                .filter(|v| !v.is_empty());
            if rc.hall_prob > 0.0
                && u >= rc.scenario_prob
                && u < rc.scenario_prob + rc.hall_prob
                && let Some(ids) = same_stake
            {
                let pick = ids[(splitmix64(h) % ids.len() as u64) as usize] as usize;
                let mut g = Game::from_state(rc.hall[pick].clone());
                g.reseed_hidden(splitmix64(seed ^ HALL_SALT), None);
                self.game = g;
                self.origin = Origin::Hall;
                self.ep_start_joker = None;
                self.refresh();
                return;
            }
        }
        self.game.reset_with_seed(seed, config);
        self.apply_start_joker();
        self.refresh();
    }

    /// A natural episode sitting at the Blind select of an ante-`min_ante`-or-later boss: a state
    /// worth keeping in a boss hall.
    pub fn hall_candidate(&self, min_ante: i32) -> bool {
        let s = self.game.state();
        self.origin == Origin::Natural
            && !self.imported
            && s.phase == Phase::BlindSelect
            && s.blind == BlindKind::Boss
            && s.ante >= min_ante
    }

    /// Where the current episode started.
    pub fn origin(&self) -> Origin {
        self.origin
    }

    fn reward_of(sh: &Shared, info: &StepInfo, game: &Game, before: f32, won_before: bool) -> f32 {
        let r = &sh.reward;
        let mut x = 0.0;
        let mut after = progress(game.state());
        if r.zero_play != 0.0 && info.hand.is_some_and(|h| h.invalid) {
            x += r.zero_play;
        }
        if let Some(rr) = &info.round {
            if rr.won {
                x += r.blind;
                if r.margin != 0.0 && rr.target.is_finite() && rr.target > 0.0 {
                    let ratio = (rr.round_score / rr.target).max(1.0);
                    x += r.margin * (ratio.min(f64::from(r.margin_cap))).ln() as f32;
                }
                if rr.blind == BlindKind::Boss {
                    x += r.ante;
                }
            }
            after = if rr.target.is_finite() && rr.target > 0.0 {
                (rr.round_score / rr.target).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
        }
        // `won` flips exactly once per run (at the ante-8 boss; in a non-endless run that is also
        // the `Outcome::Win` transition), so an endless run that continues pays `win` once.
        if !won_before && game.state().won {
            x += r.win;
        }
        if game.outcome() == Some(Outcome::Loss) {
            x += r.loss;
        }
        if r.shaping != 0.0 {
            x += r.shaping * (after - before);
        }
        x
    }

    /// Apply the structured action `parts`; auto-reset when the episode ends if `auto_reset`.
    pub fn step_parts(
        &mut self,
        parts: [i64; 4],
        sh: &Shared,
        auto_reset: bool,
    ) -> Result<StepOut, String> {
        let action = parse_action(parts).filter(|a| self.game.validate(*a).is_ok());
        let (reward, invalid, cap_hit) = match action {
            None => (sh.reward.invalid, true, false),
            Some(a) => {
                let before = progress(self.game.state());
                let won_before = self.game.state().won;
                let info = self.game.step(a).map_err(|e| {
                    format!("env {}: validated action {a:?} rejected: {e}", self.idx)
                })?;
                let mut x = Self::reward_of(sh, &info, &self.game, before, won_before);
                if matches!(a, Action::MoveJoker { .. }) {
                    x += sh.reward.reorder;
                }
                let cap = info.round.is_some_and(|r| {
                    r.won && r.blind == BlindKind::Boss && r.ante >= sh.reward.endless_max_ante
                });
                (x, false, cap)
            }
        };
        self.steps += 1;
        self.ep_return += reward;
        if invalid {
            self.ep_invalid += 1;
        }
        let mut done = match self.game.outcome() {
            Some(Outcome::Win) => DONE_WIN,
            Some(Outcome::Loss) => DONE_LOSS,
            None if self.game.is_terminal() => DONE_LOSS,
            None => DONE_NO,
        };
        if sh.config.endless {
            // Endless: a run that passed ante 8 reports as won even if it dies later; the episode
            // also ends (done, terminal value 0) once the boss of `endless_max_ante` is beaten.
            if done == DONE_NO && cap_hit {
                done = DONE_LOSS;
            }
            if done != DONE_NO && self.game.state().won {
                done = DONE_WIN;
            }
        }
        if done == DONE_NO && self.steps >= sh.max_steps {
            done = DONE_TRUNCATED;
        }
        if done != DONE_NO {
            let s = self.game.state();
            self.finished.push(Finished {
                env: self.idx,
                ante: s.ante,
                code: done,
                steps: self.steps,
                ret: self.ep_return,
                rounds_won: s.stats.rounds_won,
                invalid: self.ep_invalid,
                deck: s.config.deck,
                stake: s.config.stake,
                imported: self.imported,
                scenario: self.origin == Origin::Scenario,
                hall: self.origin == Origin::Hall,
                start_joker: self.ep_start_joker,
            });
            if auto_reset {
                self.next_episode(sh);
            } else {
                self.refresh();
            }
        } else {
            self.refresh();
            if self.view.kinds == 0 {
                return Err(format!(
                    "env {}: non-terminal state without any addressable action",
                    self.idx
                ));
            }
        }
        Ok(StepOut {
            reward,
            done,
            invalid,
        })
    }

    /// Write observation and masks of the current state.
    pub fn write(&self, row: &mut Row<'_>) {
        write_obs(self.game.state(), row);
        write_masks(&self.view, row);
    }

    /// A uniformly random mask-conforming action (section 10): kind, `arg0`, `arg1` uniform among
    /// the legal ones, cards by the sequential rule choosing uniformly among `{stop} + next`.
    pub fn random_parts(&mut self) -> [i64; 4] {
        let v = self.view;
        if v.kinds == 0 {
            return [0; 4];
        }
        let rng = &mut self.act_rng;
        let kind = nth_bit(v.kinds, below(rng, v.kinds.count_ones())) as usize;
        let a0bits = v.arg0[kind];
        let arg0 = nth_bit(a0bits, below(rng, a0bits.count_ones())) as usize;
        let mut arg1 = 0;
        if kind == ActionKind::MoveJoker.index() {
            let bits = v.arg1 & !(1u32 << arg0);
            if bits != 0 {
                arg1 = nth_bit(bits, below(rng, bits.count_ones())) as usize;
            }
        }
        let mut cards = 0u32;
        if let Some(r) = sel_row(kind, arg0) {
            let row: SelRow = v.sel[r];
            if row.max > 0 {
                cards = random_selection(&row.select(), rng);
            }
        }
        [kind as i64, arg0 as i64, arg1 as i64, i64::from(cards)]
    }

    /// The action the fair-information heuristic agent plays here (`[0; 4]` in a terminal state).
    pub fn teacher_parts(&mut self, sh: &Shared) -> [i64; 4] {
        if self.game.is_terminal() {
            return [0; 4];
        }
        let (base, idx, ep) = (sh.base_seed, u64::from(self.idx), self.episode);
        let agent = self.teacher.get_or_insert_with(|| {
            Box::new(HeuristicAgent::new(derive_seed(
                base,
                idx,
                ep ^ TEACHER_SALT,
            )))
        });
        action_row(&agent.act(&self.game))
    }
}

/// Sequential uniform selection under `sel`: at every step uniform among `{stop if allowed} ∪
/// next candidates` (the rule of design section 6, through the core's helpers).
pub fn random_selection(sel: &CardSelect, rng: &mut u64) -> u32 {
    let mut partial = 0u32;
    loop {
        let next = sel.next_candidates(partial);
        let can_stop = sel.can_finish(partial);
        let n = next.count_ones() + u32::from(can_stop);
        if n == 0 {
            return partial; // cannot happen (the rule never dead-ends)
        }
        let r = below(rng, n);
        if can_stop && r == 0 {
            return partial;
        }
        partial |= 1 << nth_bit(next, r - u32::from(can_stop));
    }
}

/// RNG streams by name (`"deck"`, `"shop"`, `"packs"`, `"bosses"`, `"tags"`, `"jokers"`,
/// `"cards"`, `"consumables"`, `"misc"`); `None` = all (see `Game::reseed_hidden`).
pub fn parse_streams(names: Option<Vec<String>>) -> Result<Option<Vec<RngKey>>, String> {
    let Some(names) = names else {
        return Ok(None);
    };
    names
        .iter()
        .map(|n| {
            RngKey::ALL
                .iter()
                .copied()
                .find(|k| k.name() == n.to_ascii_lowercase())
                .ok_or_else(|| {
                    let all: Vec<&str> = RngKey::ALL.iter().map(|k| k.name()).collect();
                    format!("unknown rng stream {n:?} (valid: {})", all.join(", "))
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Decode an exported state (`Env::export_json` / `state_json`) into a game, checking its
/// invariants; `reseed = Some(seed)` re-samples the hidden information (see
/// `Game::reseed_hidden`; `streams` picks the streams, `None` = all).
pub fn game_from_json(
    json: &str,
    reseed: Option<u64>,
    streams: Option<&[RngKey]>,
) -> Result<Game, String> {
    let state = balatro_core::serialize::state_from_json(json).map_err(|e| e.to_string())?;
    let mut g = Game::from_state(state);
    if let Some(seed) = reseed {
        g.reseed_hidden(seed, streams);
    }
    Ok(g)
}

/// A joker by name: the display name (`"Greedy Joker"`), the id (`"GreedyJoker"`) or snake case
/// (`"greedy_joker"`), compared ignoring case and every non-alphanumeric character.
pub fn parse_joker(name: &str) -> Result<JokerId, String> {
    let norm = |s: &str| {
        s.chars()
            .filter(char::is_ascii_alphanumeric)
            .map(|c| c.to_ascii_lowercase())
            .collect::<String>()
    };
    let want = norm(name);
    if !want.is_empty() {
        for d in all_defs() {
            if norm(d.name) == want || norm(&format!("{:?}", d.id)) == want {
                return Ok(d.id);
            }
        }
    }
    Err(format!(
        "unknown joker {name:?} (display name, e.g. \"Greedy Joker\")"
    ))
}

/// Parse the deck/stake names accepted by the Python constructors.
pub fn parse_config(deck: &str, stake: &str, endless: bool) -> Result<GameConfig, String> {
    let deck = DeckKind::from_name(deck).ok_or_else(|| {
        let names: Vec<&str> = DeckKind::ALL.iter().map(|d| d.name()).collect();
        format!("unknown deck {deck:?} (valid: {})", names.join(", "))
    })?;
    let stake = StakeKind::from_name(stake).ok_or_else(|| {
        let names: Vec<&str> = StakeKind::ALL.iter().map(|d| d.name()).collect();
        format!("unknown stake {stake:?} (valid: {})", names.join(", "))
    })?;
    Ok(GameConfig {
        deck,
        stake,
        endless,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Owned;

    fn shared() -> Shared {
        Shared {
            config: GameConfig::default(),
            configs: ConfigPool::EMPTY,
            max_steps: 100_000,
            base_seed: 5,
            reward: RewardConfig::default(),
            reset: ResetCfg::default(),
        }
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(parse_action([17, 0, 0, 0]).is_none());
        assert!(parse_action([-1, 0, 0, 0]).is_none());
        assert!(parse_action([4, 256, 0, 0]).is_none());
        assert!(parse_action([2, 0, 0, -1]).is_none());
        assert!(parse_action([2, 0, 0, 1 << 32]).is_none());
        // Unused operands are ignored.
        assert_eq!(parse_action([0, 999, -5, 77]), Some(Action::SelectBlind));
        assert_eq!(parse_action([2, 99, 99, 0b11]), Some(Action::Play(0b11)));
    }

    #[test]
    fn random_policy_is_never_invalid_and_runs_to_the_end() {
        let sh = shared();
        let mut e = Env::new(0, &sh);
        let mut finished = 0;
        for _ in 0..40_000 {
            let p = e.random_parts();
            let out = e.step_parts(p, &sh, true).unwrap();
            assert!(!out.invalid, "mask-conforming action rejected: {p:?}");
            finished += usize::from(out.done != DONE_NO);
        }
        assert!(finished > 0);
        assert_eq!(e.finished.iter().map(|f| f.invalid).sum::<u32>(), 0);
    }

    #[test]
    fn reorder_term_charges_only_valid_move_joker() {
        let base = shared();
        let mut sh = base.clone();
        sh.reward.reorder = -0.5;
        let (mut a, mut b) = (Env::new(0, &base), Env::new(0, &sh));
        let mut moves = 0;
        for _ in 0..200_000 {
            let p = a.random_parts();
            let (ra, rb) = (
                a.step_parts(p, &base, true).unwrap(),
                b.step_parts(p, &sh, true).unwrap(),
            );
            assert!(!ra.invalid && ra.done == rb.done);
            let is_move = p[0] == ActionKind::MoveJoker.index() as i64;
            moves += i32::from(is_move);
            assert_eq!(rb.reward - ra.reward, if is_move { -0.5 } else { 0.0 });
        }
        assert!(moves > 0, "no MoveJoker was ever played");
    }

    #[test]
    fn invalid_action_is_a_noop() {
        let sh = shared();
        let mut e = Env::new(0, &sh);
        let before = e.game.state().clone();
        // Fresh run is in BlindSelect: Play is illegal.
        let out = e.step_parts([2, 0, 0, 1], &sh, true).unwrap();
        assert!(out.invalid && out.done == DONE_NO);
        assert_eq!(out.reward, sh.reward.invalid);
        assert_eq!(*e.game.state(), before);
        let out = e.step_parts([99, 0, 0, 0], &sh, true).unwrap();
        assert!(out.invalid);
        assert_eq!(*e.game.state(), before);
    }

    #[test]
    fn obs_and_masks_write_without_panicking_on_random_play() {
        let sh = shared();
        let mut e = Env::new(1, &sh);
        let mut o = Owned::new(1);
        let b = o.bufs();
        for _ in 0..5_000 {
            let p = e.random_parts();
            e.step_parts(p, &sh, true).unwrap();
            // SAFETY: single-threaded, one live row.
            let mut row = unsafe { b.row(0) };
            e.write(&mut row);
        }
    }

    // ----- Phase 10F reward options ---------------------------------------------------------

    fn blind_report(blind: BlindKind, round_score: f64, target: f64, saved: bool) -> StepInfo {
        use balatro_core::game::RoundReport;
        StepInfo {
            action: Action::EndShop,
            phase_before: Phase::Play,
            phase_after: Phase::Shop,
            money_before: 0,
            money_after: 0,
            hand: None,
            round: Some(RoundReport {
                won: true,
                saved,
                out_of_cards: false,
                ante: 1,
                blind,
                round_score,
                target,
                held_money: 0,
                cash_out: None,
            }),
            tag: None,
            sold_for: None,
            spent: None,
            used: None,
            cards_destroyed: 0,
            cards_added: 0,
        }
    }

    fn reward_for(sh: &Shared, info: &StepInfo) -> f32 {
        let g = Game::new(1, GameConfig::default());
        Env::reward_of(sh, info, &g, 0.0, false)
    }

    #[test]
    fn margin_term_value_and_cap() {
        let mut sh = shared();
        let base = reward_for(&sh, &blind_report(BlindKind::Small, 450.0, 300.0, false));
        assert_eq!(base, 1.0);
        sh.reward.margin = 0.5;
        let r = reward_for(&sh, &blind_report(BlindKind::Small, 450.0, 300.0, false));
        assert!((r - (1.0 + 0.5 * 1.5f32.ln())).abs() < 1e-6, "{r}");
        // Exactly at target: no bonus. Capped at margin_cap (default 4).
        let r = reward_for(&sh, &blind_report(BlindKind::Small, 300.0, 300.0, false));
        assert_eq!(r, 1.0);
        let r = reward_for(&sh, &blind_report(BlindKind::Small, 1e9, 300.0, false));
        assert!((r - (1.0 + 0.5 * 4.0f32.ln())).abs() < 1e-6, "{r}");
        sh.reward.set("margin_cap", 2.0);
        let r = reward_for(&sh, &blind_report(BlindKind::Boss, 1e9, 300.0, false));
        assert!((r - (2.0 + 0.5 * 2.0f32.ln())).abs() < 1e-6, "{r}");
        // A blind saved below its target never pays a negative margin.
        let r = reward_for(&sh, &blind_report(BlindKind::Small, 100.0, 300.0, true));
        assert_eq!(r, 1.0);
    }

    /// Beat the blind in play by seeding the round score at the target, then playing 5 cards.
    fn beat_blind(e: &mut Env, sh: &Shared) -> StepOut {
        assert!(
            !e.step_parts(action_row(&Action::SelectBlind), sh, true)
                .unwrap()
                .invalid
        );
        let t = e.game.state().target;
        e.game.state_mut().round_score = t;
        e.refresh();
        let n = e.game.state().deck.hand.len().min(5);
        let play = action_row(&Action::Play((1u32 << n) - 1));
        let out = e.step_parts(play, sh, true).unwrap();
        assert!(!out.invalid);
        out
    }

    /// Beat blinds (ending shops) until the first `done`; the rewards of the boss steps by ante.
    fn run_to_done(sh: &Shared, max_blinds: usize) -> (Env, Vec<(i32, BlindKind, f32, u8)>) {
        let mut e = Env::new(0, sh);
        let mut log = Vec::new();
        for _ in 0..max_blinds {
            let (ante, blind) = (e.game.state().ante, e.game.state().blind);
            let out = beat_blind(&mut e, sh);
            log.push((ante, blind, out.reward, out.done));
            if out.done != DONE_NO {
                break;
            }
            if e.game.state().phase == Phase::Shop {
                let p = action_row(&Action::EndShop);
                assert!(!e.step_parts(p, sh, true).unwrap().invalid);
            }
        }
        (e, log)
    }

    fn boss_game(boss: balatro_core::content::BossId) -> (Env, Shared) {
        let sh = shared();
        let mut e = Env::new(0, &sh);
        for _ in 0..2 {
            beat_blind(&mut e, &sh);
            let p = action_row(&Action::EndShop);
            e.step_parts(p, &sh, true).unwrap();
        }
        e.game.state_mut().boss = Some(boss);
        assert!(
            !e.step_parts(action_row(&Action::SelectBlind), &sh, true)
                .unwrap()
                .invalid
        );
        assert!(e.game.state().boss_in_effect());
        (e, sh)
    }

    fn plain_hand(e: &mut Env, pair: bool) {
        use balatro_core::enums::{Edition, Enhancement, Rank, Seal, Suit};
        let ranks = [
            Rank::Two,
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
        ];
        for (c, r) in e.game.state_mut().deck.hand.iter_mut().zip(ranks) {
            c.rank = r;
            c.suit = Suit::Spades;
            c.enhancement = Enhancement::None;
            c.edition = Edition::Base;
            c.seal = Seal::None;
            c.perma_bonus = 0;
        }
        if pair {
            e.game.state_mut().deck.hand[1].rank = Rank::Two;
            e.game.state_mut().deck.hand[1].suit = Suit::Hearts;
        }
        e.refresh();
    }

    fn play_mask(e: &mut Env, sh: &Shared, mask: u32) -> f32 {
        let out = e
            .step_parts(action_row(&Action::Play(mask)), sh, true)
            .unwrap();
        assert!(!out.invalid && out.done == DONE_NO);
        out.reward
    }

    #[test]
    fn zero_play_under_the_eye() {
        use balatro_core::content::BossId;
        let (mut e, mut sh) = boss_game(BossId::TheEye);
        sh.reward.zero_play = -0.3;
        plain_hand(&mut e, false);
        assert_eq!(play_mask(&mut e, &sh, 0b1), 0.0, "first High Card is valid");
        plain_hand(&mut e, false);
        assert_eq!(
            play_mask(&mut e, &sh, 0b1),
            -0.3,
            "repeated High Card scores 0"
        );
    }

    #[test]
    fn zero_play_under_the_mouth_and_off_by_default() {
        use balatro_core::content::BossId;
        let (mut e, mut sh) = boss_game(BossId::TheMouth);
        sh.reward.zero_play = 0.7;
        plain_hand(&mut e, true);
        assert_eq!(play_mask(&mut e, &sh, 0b100), 0.0);
        plain_hand(&mut e, true);
        assert_eq!(
            play_mask(&mut e, &sh, 0b11),
            0.7,
            "Pair after High Card is rejected"
        );
        // Off by default: the same rejected hand pays nothing.
        let (mut e, sh) = boss_game(BossId::TheMouth);
        plain_hand(&mut e, true);
        play_mask(&mut e, &sh, 0b100);
        plain_hand(&mut e, true);
        assert_eq!(play_mask(&mut e, &sh, 0b11), 0.0);
    }

    #[test]
    fn non_endless_ante_8_boss_wins_and_ends() {
        let sh = shared();
        let (e, log) = run_to_done(&sh, 100);
        let (ante, blind, reward, done) = *log.last().unwrap();
        assert_eq!((ante, blind, done), (8, BlindKind::Boss, DONE_WIN));
        assert_eq!(reward, 1.0 + 1.0 + 10.0);
        assert_eq!(log.len(), 24);
        assert_eq!(e.finished.len(), 1);
    }

    #[test]
    fn endless_continues_pays_win_once_and_stops_at_max_ante() {
        let mut sh = shared();
        sh.config.endless = true;
        sh.reward.endless_max_ante = 10;
        let (e, log) = run_to_done(&sh, 100);
        let wins = log.iter().filter(|l| l.2 >= 12.0).count();
        assert_eq!(wins, 1, "win pays once");
        let ante8 = log
            .iter()
            .find(|l| l.0 == 8 && l.1 == BlindKind::Boss)
            .unwrap();
        assert_eq!(
            (ante8.2, ante8.3),
            (12.0, DONE_NO),
            "run continues after the win"
        );
        let (ante, blind, reward, done) = *log.last().unwrap();
        assert_eq!((ante, blind, done), (10, BlindKind::Boss, DONE_WIN));
        assert_eq!(reward, 2.0, "no second win payment");
        assert_eq!(log.len(), 30);
        let f = e.finished.last().unwrap();
        assert_eq!(f.code, DONE_WIN);
    }

    #[test]
    fn endless_death_after_ante_8_reports_won_and_before_reports_loss() {
        let mut sh = shared();
        sh.config.endless = true;
        // Die at once: not won.
        let mut e = Env::new(0, &sh);
        e.step_parts(action_row(&Action::SelectBlind), &sh, true)
            .unwrap();
        e.game.state_mut().hands_left = 1;
        let out = e
            .step_parts(action_row(&Action::Play(0b1)), &sh, true)
            .unwrap();
        assert_eq!(out.done, DONE_LOSS);
        // Pass ante 8, then die in ante 9.
        let mut e = Env::new(0, &sh);
        let mut last = beat_blind(&mut e, &sh);
        for _ in 0..24 {
            let p = action_row(&Action::EndShop);
            e.step_parts(p, &sh, true).unwrap();
            let (ante, blind) = (e.game.state().ante, e.game.state().blind);
            if ante == 9 && blind == BlindKind::Small {
                break;
            }
            last = beat_blind(&mut e, &sh);
        }
        assert_eq!(last.done, DONE_NO);
        assert!(e.game.state().won);
        e.step_parts(action_row(&Action::SelectBlind), &sh, true)
            .unwrap();
        e.game.state_mut().hands_left = 1;
        let out = e
            .step_parts(action_row(&Action::Play(0b1)), &sh, true)
            .unwrap();
        assert_eq!(
            out.done, DONE_WIN,
            "died in endless after passing ante 8 -> won"
        );
        assert_eq!(out.reward, 0.0);
        assert_eq!(e.finished.last().unwrap().ante, 9);
        // The auto-reset started a fresh, non-won run.
        assert!(!e.game.state().won);
    }
}
