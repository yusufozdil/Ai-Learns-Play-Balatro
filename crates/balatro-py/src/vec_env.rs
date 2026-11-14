//! `VecEnv`: N independent runs stepped in parallel (rayon) with the GIL released, writing into
//! caller-owned numpy buffers (no per-step allocation on the hot path).
//!
//! Layering:
//! * [`EnvPool`] is plain Rust (no pyo3): envs, seed derivation, auto-reset, parallel stepping.
//! * [`VecEnv`] is the thin pyo3 shell: extracts numpy slices, detaches from the GIL, maps
//!   errors to Python exceptions.
//!
//! Determinism: env `i` in episode `k` is seeded with `derive_seed(base_seed, i, k)`; every env is
//! stepped by exactly one thread and never reads another env, so results do not depend on the
//! thread count or on rayon's work distribution. Per-env sampling for `random_actions` uses a
//! per-env SplitMix64 stream for the same reason.
//!
//! Action interface (TEMPORARY, Phase 8 replaces it): an action is an index into the env's
//! current `legal_actions()` list. Observation, reward and `legal_count` come from
//! [`crate::interface`].

use crate::interface::{self, OBS_DIM};
use balatro_core::actions::Action;
use balatro_core::game::Game;
use balatro_core::state::{DeckKind, GameConfig, StakeKind};
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Default cap on steps per episode; longer episodes are truncated (done code 3).
pub const DEFAULT_MAX_EPISODE_STEPS: u32 = 10_000;

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

/// Summary of one finished episode (for logging).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Finished {
    /// Env index.
    pub env: u32,
    /// Ante reached.
    pub ante: i32,
    /// Done code (`DONE_LOSS`, `DONE_WIN`, `DONE_TRUNCATED`).
    pub code: u8,
    /// Steps taken.
    pub steps: u32,
    /// Sum of the (placeholder) rewards.
    pub ret: f32,
    /// Blinds beaten.
    pub rounds_won: u32,
}

/// Settings shared by all envs of a pool.
#[derive(Debug, Clone, Copy)]
struct Shared {
    config: GameConfig,
    max_steps: u32,
    base_seed: u64,
}

struct Env {
    idx: u32,
    game: Game,
    /// Legal actions of the current state (recomputed after every step / reset).
    legal: Vec<Action>,
    episode: u64,
    steps: u32,
    ep_return: f32,
    /// SplitMix64 state for `random_actions`.
    act_rng: u64,
    finished: Vec<Finished>,
}

impl Env {
    fn new(idx: u32, sh: &Shared) -> Env {
        let mut e = Env {
            idx,
            game: Game::new(derive_seed(sh.base_seed, u64::from(idx), 0), sh.config),
            legal: Vec::new(),
            episode: 0,
            steps: 0,
            ep_return: 0.0,
            act_rng: splitmix64(derive_seed(sh.base_seed, u64::from(idx), u64::MAX)),
            finished: Vec::new(),
        };
        e.game.legal_actions_into(&mut e.legal);
        e
    }

    /// Back to episode 0 with fresh seeds (the `reset()` entry point).
    fn hard_reset(&mut self, sh: &Shared) {
        let idx = self.idx;
        self.game = Game::new(derive_seed(sh.base_seed, u64::from(idx), 0), sh.config);
        self.episode = 0;
        self.steps = 0;
        self.ep_return = 0.0;
        self.act_rng = splitmix64(derive_seed(sh.base_seed, u64::from(idx), u64::MAX));
        self.finished.clear();
        self.game.legal_actions_into(&mut self.legal);
    }

    /// Start the next episode (auto-reset) with a derived seed.
    fn next_episode(&mut self, sh: &Shared) {
        self.episode += 1;
        self.steps = 0;
        self.ep_return = 0.0;
        self.game = Game::new(
            derive_seed(sh.base_seed, u64::from(self.idx), self.episode),
            sh.config,
        );
        self.game.legal_actions_into(&mut self.legal);
    }

    /// Apply legal-action index `a` (already range-checked), auto-reset when the episode ends.
    /// Returns `(reward, done_code)`; the observation is written by the caller afterwards.
    fn step(&mut self, a: usize, sh: &Shared) -> Result<(f32, u8), String> {
        let Some(&action) = self.legal.get(a) else {
            return Err(format!("env {}: action index {a} out of range", self.idx));
        };
        let info = self
            .game
            .step(action)
            .map_err(|e| format!("env {}: legal action {action:?} rejected: {e}", self.idx))?;
        let reward = interface::step_reward(&info, &self.game);
        self.steps += 1;
        self.ep_return += reward;
        let mut done = interface::done_code(&self.game);
        if done == interface::DONE_NO && self.steps >= sh.max_steps {
            done = interface::DONE_TRUNCATED;
        }
        if done != interface::DONE_NO {
            let s = self.game.state();
            self.finished.push(Finished {
                env: self.idx,
                ante: s.ante,
                code: done,
                steps: self.steps,
                ret: self.ep_return,
                rounds_won: s.stats.rounds_won,
            });
            self.next_episode(sh);
        } else {
            self.game.legal_actions_into(&mut self.legal);
            if self.legal.is_empty() {
                return Err(format!(
                    "env {}: non-terminal state without legal actions",
                    self.idx
                ));
            }
        }
        Ok((reward, done))
    }

    fn write_obs(&self, obs: &mut [f32], legal_count: &mut i32) {
        interface::write_obs(&self.game, self.legal.len(), obs);
        *legal_count = self.legal.len() as i32;
    }
}

/// Plain-Rust batch of envs (see module docs).
pub struct EnvPool {
    shared: Shared,
    envs: Vec<Env>,
    threads: usize,
    pool: Option<rayon::ThreadPool>,
}

impl EnvPool {
    /// `threads == 0` means one per available core.
    pub fn new(
        num_envs: usize,
        seed: u64,
        threads: usize,
        config: GameConfig,
        max_steps: u32,
    ) -> Result<Self, String> {
        let shared = Shared {
            config,
            max_steps: max_steps.max(1),
            base_seed: seed,
        };
        let envs = (0..num_envs).map(|i| Env::new(i as u32, &shared)).collect();
        let mut p = EnvPool {
            shared,
            envs,
            threads: 1,
            pool: None,
        };
        p.set_threads(threads)?;
        Ok(p)
    }

    /// Change the worker count (`0` = available cores). 1 runs inline on the calling thread.
    pub fn set_threads(&mut self, threads: usize) -> Result<(), String> {
        let t = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        };
        self.pool = if t > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .thread_name(|i| format!("balatro-env-{i}"))
                    .build()
                    .map_err(|e| format!("cannot build thread pool: {e}"))?,
            )
        } else {
            None
        };
        self.threads = t;
        Ok(())
    }

    /// Worker count in use.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Number of envs.
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn chunk(&self) -> usize {
        // ~4 chunks per worker: balances uneven env costs without drowning in task overhead.
        self.envs.len().div_ceil(self.threads * 4).max(1)
    }

    /// Reset every env to episode 0 (optionally with a new base seed) and write observations.
    pub fn reset(&mut self, seed: Option<u64>, obs: &mut [f32], legal_count: &mut [i32]) {
        if let Some(s) = seed {
            self.shared.base_seed = s;
        }
        let sh = self.shared;
        let chunk = self.chunk();
        let work = |envs: &mut [Env], obs: &mut [f32], lc: &mut [i32]| {
            for ((e, o), l) in envs
                .iter_mut()
                .zip(obs.as_chunks_mut::<OBS_DIM>().0.iter_mut())
                .zip(lc.iter_mut())
            {
                e.hard_reset(&sh);
                e.write_obs(o, l);
            }
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_chunks_mut(chunk)
                    .zip(obs.par_chunks_mut(chunk * OBS_DIM))
                    .zip(legal_count.par_chunks_mut(chunk))
                    .for_each(|((e, o), l)| work(e, o, l));
            }),
            None => work(&mut self.envs, obs, legal_count),
        }
    }

    /// `actions[i]` indexes env i's legal-action list. All indices are validated before any env
    /// is touched, so an error leaves the pool unchanged.
    pub fn step(
        &mut self,
        actions: &[i64],
        obs: &mut [f32],
        reward: &mut [f32],
        done: &mut [u8],
        legal_count: &mut [i32],
    ) -> Result<(), String> {
        for (e, &a) in self.envs.iter().zip(actions) {
            if a < 0 || a as usize >= e.legal.len() {
                return Err(format!(
                    "env {}: action index {a} out of range (legal_count = {})",
                    e.idx,
                    e.legal.len()
                ));
            }
        }
        let sh = self.shared;
        let chunk = self.chunk();
        let work = |envs: &mut [Env],
                    acts: &[i64],
                    obs: &mut [f32],
                    rew: &mut [f32],
                    dn: &mut [u8],
                    lc: &mut [i32]|
         -> Result<(), String> {
            for (i, e) in envs.iter_mut().enumerate() {
                let (r, d) = e.step(acts[i] as usize, &sh)?;
                rew[i] = r;
                dn[i] = d;
                e.write_obs(&mut obs[i * OBS_DIM..(i + 1) * OBS_DIM], &mut lc[i]);
            }
            Ok(())
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_chunks_mut(chunk)
                    .zip(actions.par_chunks(chunk))
                    .zip(obs.par_chunks_mut(chunk * OBS_DIM))
                    .zip(reward.par_chunks_mut(chunk))
                    .zip(done.par_chunks_mut(chunk))
                    .zip(legal_count.par_chunks_mut(chunk))
                    .map(|(((((e, a), o), r), d), l)| work(e, a, o, r, d, l))
                    .reduce(|| Ok(()), |x, y| x.and(y))
            }),
            None => work(&mut self.envs, actions, obs, reward, done, legal_count),
        }
    }

    /// Uniform random legal-action index for every env (per-env deterministic streams).
    pub fn random_actions(&mut self, out: &mut [i64]) {
        for (e, o) in self.envs.iter_mut().zip(out.iter_mut()) {
            e.act_rng = splitmix64(e.act_rng);
            let n = e.legal.len() as u128;
            *o = ((u128::from(e.act_rng) * n) >> 64) as i64;
        }
    }

    /// Take (and clear) the summaries of the episodes that finished since the last call, ordered
    /// by env index (then by completion order within an env).
    pub fn drain_finished(&mut self) -> Vec<Finished> {
        let mut v = Vec::new();
        for e in &mut self.envs {
            v.append(&mut e.finished);
        }
        v
    }
}

/// Parse the deck/stake names accepted by the Python constructors.
pub fn parse_config(deck: &str, stake: &str, endless: bool) -> Result<GameConfig, String> {
    let deck = match deck.to_ascii_lowercase().as_str() {
        "red" => DeckKind::Red,
        other => return Err(format!("unknown deck {other:?} (implemented: \"red\")")),
    };
    let stake = match stake.to_ascii_lowercase().as_str() {
        "white" => StakeKind::White,
        other => return Err(format!("unknown stake {other:?} (implemented: \"white\")")),
    };
    Ok(GameConfig {
        deck,
        stake,
        endless,
    })
}

fn val_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn slice_err(name: &str, e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!(
        "`{name}` must be a C-contiguous, writable numpy array: {e}"
    ))
}

/// Vectorised environment: `num_envs` runs stepped in parallel with the GIL released.
///
/// All arrays are caller-owned and filled in place. Dtypes: `obs` float32 `(n, obs_dim)`,
/// `reward` float32 `(n,)`, `done` uint8 `(n,)` (0 running, 1 lost, 2 won, 3 truncated),
/// `legal_count` int32 `(n,)`, `actions` int64 `(n,)` (index into the env's legal-action list).
/// After a step with `done[i] != 0` the env has already auto-reset: `obs[i]` and
/// `legal_count[i]` belong to the next episode, `reward[i]` to the finishing transition.
#[pyclass(module = "balatro_rl._engine")]
pub struct VecEnv {
    pool: EnvPool,
}

#[pymethods]
impl VecEnv {
    /// `num_threads = 0` uses all cores; 1 steps inline. `deck`/`stake` currently only accept
    /// "red"/"white". Episodes longer than `max_episode_steps` are truncated (done code 3).
    #[new]
    #[pyo3(signature = (num_envs, seed=0, num_threads=0, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
                        deck="red", stake="white", endless=false))]
    fn new(
        num_envs: usize,
        seed: u64,
        num_threads: usize,
        max_episode_steps: u32,
        deck: &str,
        stake: &str,
        endless: bool,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(val_err("num_envs must be >= 1".into()));
        }
        if u32::try_from(num_envs).is_err() {
            return Err(val_err("num_envs too large".into()));
        }
        let config = parse_config(deck, stake, endless).map_err(val_err)?;
        let pool = EnvPool::new(num_envs, seed, num_threads, config, max_episode_steps)
            .map_err(PyRuntimeError::new_err)?;
        Ok(VecEnv { pool })
    }

    /// Number of envs.
    #[getter]
    fn num_envs(&self) -> usize {
        self.pool.num_envs()
    }

    /// Observation length.
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_DIM
    }

    /// Worker threads in use.
    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.threads()
    }

    /// Change the worker count (0 = all cores). Does not affect results.
    fn set_num_threads(&mut self, num_threads: usize) -> PyResult<()> {
        self.pool
            .set_threads(num_threads)
            .map_err(PyRuntimeError::new_err)
    }

    /// Reset all envs to episode 0 (optionally with a new base `seed`) and fill `obs_out`
    /// and `legal_count_out`.
    #[pyo3(signature = (obs_out, legal_count_out, seed=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        mut obs_out: PyReadwriteArray2<'_, f32>,
        mut legal_count_out: PyReadwriteArray1<'_, i32>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let n = self.pool.num_envs();
        check_shape2(&obs_out, n, "obs_out")?;
        let obs = obs_out
            .as_slice_mut()
            .map_err(|e| slice_err("obs_out", e))?;
        let lc = legal_count_out
            .as_slice_mut()
            .map_err(|e| slice_err("legal_count_out", e))?;
        check_len(lc.len(), n, "legal_count_out")?;
        let pool = &mut self.pool;
        py.detach(|| pool.reset(seed, obs, lc));
        Ok(())
    }

    /// Step every env with `actions[i]` = index into its legal-action list. Raises `ValueError`
    /// (leaving all envs untouched) if any index is out of `[0, legal_count)`.
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<'_, i64>,
        mut obs_out: PyReadwriteArray2<'_, f32>,
        mut reward_out: PyReadwriteArray1<'_, f32>,
        mut done_out: PyReadwriteArray1<'_, u8>,
        mut legal_count_out: PyReadwriteArray1<'_, i32>,
    ) -> PyResult<()> {
        let n = self.pool.num_envs();
        check_shape2(&obs_out, n, "obs_out")?;
        let acts = actions.as_slice().map_err(|e| slice_err("actions", e))?;
        let obs = obs_out
            .as_slice_mut()
            .map_err(|e| slice_err("obs_out", e))?;
        let rew = reward_out
            .as_slice_mut()
            .map_err(|e| slice_err("reward_out", e))?;
        let dn = done_out
            .as_slice_mut()
            .map_err(|e| slice_err("done_out", e))?;
        let lc = legal_count_out
            .as_slice_mut()
            .map_err(|e| slice_err("legal_count_out", e))?;
        check_len(acts.len(), n, "actions")?;
        check_len(rew.len(), n, "reward_out")?;
        check_len(dn.len(), n, "done_out")?;
        check_len(lc.len(), n, "legal_count_out")?;
        let pool = &mut self.pool;
        py.detach(|| pool.step(acts, obs, rew, dn, lc))
            .map_err(val_err)
    }

    /// Fill `out` (int64, `(n,)`) with a uniform random legal-action index per env
    /// (deterministic per-env streams, reseeded by `reset`).
    fn random_actions(&mut self, mut out: PyReadwriteArray1<'_, i64>) -> PyResult<()> {
        let n = self.pool.num_envs();
        let o = out.as_slice_mut().map_err(|e| slice_err("out", e))?;
        check_len(o.len(), n, "out")?;
        self.pool.random_actions(o);
        Ok(())
    }

    /// Episodes finished since the last call, as `(env, ante, done_code, steps, return,
    /// blinds_beaten)` tuples (and cleared).
    fn drain_finished(&mut self) -> Vec<(u32, i32, u8, u32, f32, u32)> {
        self.pool
            .drain_finished()
            .into_iter()
            .map(|f| (f.env, f.ante, f.code, f.steps, f.ret, f.rounds_won))
            .collect()
    }
}

fn check_len(got: usize, want: usize, name: &str) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(val_err(format!(
            "`{name}` has length {got}, expected {want}"
        )))
    }
}

fn check_shape2(a: &PyReadwriteArray2<'_, f32>, n: usize, name: &str) -> PyResult<()> {
    let sh = a.shape();
    if sh == [n, OBS_DIM] {
        // `as_slice_mut` also accepts Fortran-ordered arrays, which would silently scramble rows.
        if !a.is_c_contiguous() {
            return Err(val_err(format!("`{name}` must be C-contiguous")));
        }
        Ok(())
    } else {
        Err(val_err(format!(
            "`{name}` has shape {sh:?}, expected ({n}, {OBS_DIM})"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(threads: usize, n: usize, steps: usize) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        let mut p = EnvPool::new(n, 7, threads, GameConfig::default(), 500).unwrap();
        let mut obs = vec![0.0; n * OBS_DIM];
        let mut lc = vec![0; n];
        p.reset(None, &mut obs, &mut lc);
        let (mut rew, mut dn) = (vec![0.0; n], vec![0u8; n]);
        let mut acts = vec![0i64; n];
        let (mut all_r, mut all_d) = (Vec::new(), Vec::new());
        for _ in 0..steps {
            p.random_actions(&mut acts);
            p.step(&acts, &mut obs, &mut rew, &mut dn, &mut lc).unwrap();
            all_r.extend_from_slice(&rew);
            all_d.extend_from_slice(&dn);
        }
        (obs, all_r, all_d)
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let a = run(1, 37, 400);
        let b = run(5, 37, 400);
        assert_eq!(a, b);
        assert!(a.2.iter().any(|&d| d != 0), "some episode should end");
    }

    #[test]
    fn seeds_are_distinct() {
        let s: std::collections::HashSet<u64> = (0..100)
            .flat_map(|e| (0..10).map(move |k| derive_seed(1, e, k)))
            .collect();
        assert_eq!(s.len(), 1000);
    }

    #[test]
    fn rejects_bad_index_without_side_effects() {
        let mut p = EnvPool::new(2, 1, 1, GameConfig::default(), 100).unwrap();
        let (mut obs, mut lc) = (vec![0.0; 2 * OBS_DIM], vec![0; 2]);
        p.reset(None, &mut obs, &mut lc);
        let (mut r, mut d) = (vec![0.0; 2], vec![0u8; 2]);
        let bad = [0, lc[1] as i64];
        assert!(p.step(&bad, &mut obs, &mut r, &mut d, &mut lc).is_err());
        assert!(p.step(&[-1, 0], &mut obs, &mut r, &mut d, &mut lc).is_err());
        assert!(p.step(&[0, 0], &mut obs, &mut r, &mut d, &mut lc).is_ok());
    }

    /// Cost split of one random-play env step (run with `--ignored --nocapture`, release).
    #[test]
    #[ignore]
    fn cost_breakdown() {
        use std::time::Instant;
        let sh = Shared {
            config: GameConfig::default(),
            max_steps: 10_000,
            base_seed: 3,
        };
        let mut e = Env::new(0, &sh);
        let mut obs = [0.0f32; OBS_DIM];
        let (mut t_step, mut t_obs, mut n) = (0u128, 0u128, 0u64);
        let mut legal_total = 0u64;
        let mut rng = 1u64;
        while n < 400_000 {
            rng = splitmix64(rng);
            let a = ((u128::from(rng) * e.legal.len() as u128) >> 64) as usize;
            let t = Instant::now();
            e.step(a, &sh).unwrap(); // includes legal_actions_into (+ auto-reset)
            t_step += t.elapsed().as_nanos();
            legal_total += e.legal.len() as u64;
            let t = Instant::now();
            let mut lc = 0;
            e.write_obs(&mut obs, &mut lc);
            t_obs += t.elapsed().as_nanos();
            n += 1;
        }
        // legal generation alone on the same kind of states
        let mut e2 = Env::new(0, &sh);
        let mut t_legal = 0u128;
        let mut buf = Vec::new();
        let mut m = 0u64;
        while m < 400_000 {
            rng = splitmix64(rng);
            let a = ((u128::from(rng) * e2.legal.len() as u128) >> 64) as usize;
            e2.step(a, &sh).unwrap();
            let t = Instant::now();
            e2.game.legal_actions_into(&mut buf);
            t_legal += t.elapsed().as_nanos();
            m += 1;
        }
        println!(
            "step incl. legal+reset {:.0} | legal_actions alone {:.0} | obs {:.0} | avg legal {:.0}",
            t_step as f64 / n as f64,
            t_legal as f64 / m as f64,
            t_obs as f64 / n as f64,
            legal_total as f64 / n as f64
        );
    }
}
