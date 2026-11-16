//! `VecEnv`: N independent runs stepped in parallel (rayon) with the GIL released, writing into
//! caller-owned numpy buffers (no per-step allocation on the hot path).
//!
//! Layering:
//! * [`EnvPool`] is plain Rust (no pyo3): envs, parallel stepping, helpers.
//! * [`VecEnv`] is the thin pyo3 shell: extracts the buffer dict, detaches from the GIL, maps
//!   errors to Python exceptions.
//!
//! Determinism: env `i` in episode `k` is seeded with `derive_seed(base_seed, i, k)`; every env is
//! stepped by exactly one thread and never reads another env, so results do not depend on the
//! thread count or on rayon's work distribution. The per-env samplers use per-env streams for the
//! same reason.
//!
//! Interface: `docs/design/rl_interface.md`. Buffers: [`crate::layout`].

use crate::env::{
    ConfigPool, DEFAULT_MAX_EPISODE_STEPS, Env, Finished, RewardConfig, Shared, game_from_json,
    parse_config, parse_joker, parse_streams,
};
use balatro_core::content::jokers::JokerId;
use balatro_core::rng::RngKey;
use balatro_core::serialize::state_to_json;
use crate::layout::{Bufs, PyBufs};
use balatro_core::state::GameConfig;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

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
        reward: RewardConfig,
        configs: ConfigPool,
    ) -> Result<Self, String> {
        let shared = Shared {
            config,
            configs,
            max_steps: max_steps.max(1),
            base_seed: seed,
            reward,
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

    /// Reset every env to episode 0 (optionally with a new base seed) and write observations and
    /// masks.
    pub fn reset(&mut self, seed: Option<u64>, bufs: &Bufs<'_>) {
        if let Some(s) = seed {
            self.shared.base_seed = s;
        }
        let sh = self.shared;
        let chunk = self.chunk();
        let work = |ci: usize, envs: &mut [Env]| {
            for (j, e) in envs.iter_mut().enumerate() {
                e.hard_reset(&sh);
                // SAFETY: env index `ci * chunk + j` belongs to this chunk only.
                let mut row = unsafe { bufs.row(ci * chunk + j) };
                e.write(&mut row);
                row.reward[0] = 0.0;
                row.done[0] = 0;
                row.invalid[0] = 0;
            }
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_chunks_mut(chunk)
                    .enumerate()
                    .for_each(|(ci, e)| work(ci, e));
            }),
            None => work(0, &mut self.envs),
        }
    }

    /// Step env `i` with `actions[4 * i..4 * i + 4]`; writes observation, masks, reward, done and
    /// invalid into `bufs`.
    pub fn step(&mut self, actions: &[i64], bufs: &Bufs<'_>) -> Result<(), String> {
        let sh = self.shared;
        let chunk = self.chunk();
        let work = |ci: usize, envs: &mut [Env], acts: &[i64]| -> Result<(), String> {
            for (j, e) in envs.iter_mut().enumerate() {
                let a = [
                    acts[4 * j],
                    acts[4 * j + 1],
                    acts[4 * j + 2],
                    acts[4 * j + 3],
                ];
                let out = e.step_parts(a, &sh, true)?;
                // SAFETY: env index `ci * chunk + j` belongs to this chunk only.
                let mut row = unsafe { bufs.row(ci * chunk + j) };
                row.reward[0] = out.reward;
                row.done[0] = out.done;
                row.invalid[0] = u8::from(out.invalid);
                e.write(&mut row);
            }
            Ok(())
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_chunks_mut(chunk)
                    .zip(actions.par_chunks(chunk * 4))
                    .enumerate()
                    .map(|(ci, (e, a))| work(ci, e, a))
                    .reduce(|| Ok(()), |x, y| x.and(y))
            }),
            None => work(0, &mut self.envs, actions),
        }
    }

    /// Uniformly random mask-conforming structured actions (`out.len() == 4 * num_envs`).
    pub fn random_actions(&mut self, out: &mut [i64]) {
        let chunk = self.chunk();
        let work = |envs: &mut [Env], out: &mut [i64]| {
            for (e, o) in envs.iter_mut().zip(out.as_chunks_mut::<4>().0.iter_mut()) {
                o.copy_from_slice(&e.random_parts());
            }
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_chunks_mut(chunk)
                    .zip(out.par_chunks_mut(chunk * 4))
                    .for_each(|(e, o)| work(e, o));
            }),
            None => work(&mut self.envs, out),
        }
    }

    /// The heuristic agent's action in the envs with `select[i]` (every env for `None`;
    /// `out.len() == 4 * num_envs`). The other rows of `out` are left untouched and their teacher
    /// agents are not consulted: a curriculum asks the teacher about a fraction of the envs only
    /// (a full call costs 4-8k actions/s).
    pub fn teacher_actions_where(&mut self, out: &mut [i64], select: Option<&[bool]>) {
        let sh = self.shared;
        let work = |e: &mut Env, o: &mut [i64], on: bool| {
            if on {
                o.copy_from_slice(&e.teacher_parts(&sh));
            }
        };
        let on = |i: usize| select.is_none_or(|s| s[i]);
        // A teacher action costs from microseconds (Play) to milliseconds (market valuation):
        // hand out one env at a time so that work stealing balances the load.
        match &self.pool {
            Some(pool) => pool.install(|| {
                self.envs
                    .par_iter_mut()
                    .zip(out.par_chunks_mut(4))
                    .enumerate()
                    .with_max_len(1)
                    .for_each(|(i, (e, o))| work(e, o, on(i)));
            }),
            None => {
                for (i, (e, o)) in self
                    .envs
                    .iter_mut()
                    .zip(out.as_chunks_mut::<4>().0.iter_mut())
                    .enumerate()
                {
                    work(e, o, on(i));
                }
            }
        }
    }

    /// Full state of env `i` as JSON (round-trips through [`EnvPool::import_state`]).
    pub fn export_state(&self, i: usize) -> Result<String, String> {
        let e = self.envs.get(i).ok_or_else(|| self.index_err(i))?;
        Ok(state_to_json(e.game.state(), false))
    }

    /// Replace env `i`'s run by the exported state `json` (see `Env::import_game` for the episode
    /// bookkeeping), optionally re-sampling the hidden information first (`reseed`, `streams`:
    /// see `Game::reseed_hidden`), and write the env's observation and masks into row `i` of
    /// `bufs` (reward, done and invalid of that row are zeroed). Other envs are untouched.
    pub fn import_state(
        &mut self,
        i: usize,
        json: &str,
        reseed: Option<u64>,
        streams: Option<&[RngKey]>,
        bufs: &Bufs<'_>,
    ) -> Result<(), String> {
        if i >= self.envs.len() {
            return Err(self.index_err(i));
        }
        let game = game_from_json(json, reseed, streams)?;
        let sh = self.shared;
        let e = &mut self.envs[i];
        e.import_game(game, &sh)?;
        // SAFETY: single-threaded, no other row reference is alive.
        let mut row = unsafe { bufs.row(i) };
        e.write(&mut row);
        row.reward[0] = 0.0;
        row.done[0] = 0;
        row.invalid[0] = 0;
        Ok(())
    }

    /// Per-env start joker (`None` = none), applied at the next reset or auto-reset of the env.
    pub fn set_start_jokers(&mut self, jokers: &[Option<JokerId>]) -> Result<(), String> {
        if jokers.len() != self.envs.len() {
            return Err(format!(
                "expected {} start jokers (one per env), got {}",
                self.envs.len(),
                jokers.len()
            ));
        }
        for (e, j) in self.envs.iter_mut().zip(jokers) {
            e.start_joker = *j;
        }
        Ok(())
    }

    fn index_err(&self, i: usize) -> String {
        format!("env index {i} out of range (num_envs = {})", self.envs.len())
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

/// The `(deck, stake)` sampling pool of the constructor (empty when neither list is given).
fn parse_config_pool(
    decks: Option<Vec<String>>,
    stakes: Option<Vec<String>>,
    deck: &str,
    stake: &str,
) -> PyResult<ConfigPool> {
    if decks.is_none() && stakes.is_none() {
        return Ok(ConfigPool::EMPTY);
    }
    let decks = decks.unwrap_or_else(|| vec![deck.to_string()]);
    let stakes = stakes.unwrap_or_else(|| vec![stake.to_string()]);
    if decks.is_empty() || stakes.is_empty() {
        return Err(val_err("`decks` and `stakes` must not be empty".into()));
    }
    let mut ds = Vec::new();
    let mut ss = Vec::new();
    for name in &decks {
        // Reuse the name validation (and its error message) of the fixed-config parser.
        ds.push(parse_config(name, "white", false).map_err(val_err)?.deck);
    }
    for name in &stakes {
        ss.push(parse_config("red", name, false).map_err(val_err)?.stake);
    }
    Ok(ConfigPool::product(&ds, &ss))
}

fn val_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn slice_err(name: &str, e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!(
        "`{name}` must be a C-contiguous, writable numpy array: {e}"
    ))
}

/// Parse the `reward` dict of the constructor.
pub fn parse_reward(d: Option<&Bound<'_, PyDict>>) -> PyResult<RewardConfig> {
    let mut r = RewardConfig::default();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            let val: f32 = v.extract()?;
            if !r.set(&key, val) {
                return Err(val_err(format!(
                    "unknown reward term {key:?} (blind, ante, win, loss, shaping, invalid, reorder, margin, margin_cap, zero_play, endless_max_ante)"
                )));
            }
        }
    }
    Ok(r)
}

fn check_actions(a: &PyReadonlyArray2<'_, i64>, n: usize) -> PyResult<()> {
    if a.shape() != [n, 4] {
        return Err(val_err(format!(
            "`actions` has shape {:?}, expected ({n}, 4)",
            a.shape()
        )));
    }
    if !a.is_c_contiguous() {
        return Err(val_err("`actions` must be C-contiguous".into()));
    }
    Ok(())
}

/// Vectorised environment: `num_envs` runs stepped in parallel with the GIL released.
///
/// All arrays are caller-owned and filled in place; their names, dtypes and shapes are
/// `layout()["buffers"]`. `actions` is int64 `(n, 4)` = `[kind, arg0, arg1, card_mask]`. After a
/// step with `done[i] != 0` the env has already auto-reset: the observation and masks belong to
/// the next episode, `reward[i]` to the finishing transition. See
/// `docs/design/rl_interface.md`.
#[pyclass(module = "balatro_rl._engine")]
pub struct VecEnv {
    pool: EnvPool,
}

#[pymethods]
impl VecEnv {
    /// `num_threads = 0` uses all cores; 1 steps inline. `deck` (15 decks) and `stake` (8 stakes) are
    /// lowercase names, default "red"/"white". Episodes longer than `max_episode_steps` are truncated (done code 3).
    /// `reward` overrides terms of the default reward (blind, ante, win, loss, shaping, invalid, reorder, margin,
    /// margin_cap, zero_play, endless_max_ante).
    /// `decks` / `stakes` (lists of names, either may be omitted: `deck` / `stake` is then the
    /// single entry) make every episode of every env draw its (deck, stake) uniformly from the
    /// product (a pure function of seed, env and episode index; independent of the thread count).
    #[new]
    #[pyo3(signature = (num_envs, seed=0, num_threads=0, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
                        deck="red", stake="white", endless=false, reward=None, decks=None,
                        stakes=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        seed: u64,
        num_threads: usize,
        max_episode_steps: u32,
        deck: &str,
        stake: &str,
        endless: bool,
        reward: Option<&Bound<'_, PyDict>>,
        decks: Option<Vec<String>>,
        stakes: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(val_err("num_envs must be >= 1".into()));
        }
        if u32::try_from(num_envs).is_err() {
            return Err(val_err("num_envs too large".into()));
        }
        let config = parse_config(deck, stake, endless).map_err(val_err)?;
        let reward = parse_reward(reward)?;
        let configs = parse_config_pool(decks, stakes, deck, stake)?;
        let pool = EnvPool::new(
            num_envs,
            seed,
            num_threads,
            config,
            max_episode_steps,
            reward,
            configs,
        )
        .map_err(PyRuntimeError::new_err)?;
        Ok(VecEnv { pool })
    }

    /// Number of envs.
    #[getter]
    fn num_envs(&self) -> usize {
        self.pool.num_envs()
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

    /// Reset all envs to episode 0 (optionally with a new base `seed`) and fill the observation
    /// and mask arrays of `buffers` (`reward`/`done`/`invalid` are zeroed).
    #[pyo3(signature = (buffers, seed=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        buffers: &Bound<'_, PyDict>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let mut pb = PyBufs::extract(buffers, self.pool.num_envs())?;
        let bufs = pb.bufs()?;
        let pool = &mut self.pool;
        py.detach(|| pool.reset(seed, &bufs));
        Ok(())
    }

    /// Step every env with `actions[i] = [kind, arg0, arg1, card_mask]`. Malformed or illegal
    /// actions are no-ops flagged in `invalid` (they never raise).
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray2<'_, i64>,
        buffers: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let n = self.pool.num_envs();
        check_actions(&actions, n)?;
        let acts = actions.as_slice().map_err(|e| slice_err("actions", e))?;
        let mut pb = PyBufs::extract(buffers, n)?;
        let bufs = pb.bufs()?;
        let pool = &mut self.pool;
        py.detach(|| pool.step(acts, &bufs))
            .map_err(|e| PyRuntimeError::new_err(format!("internal environment error: {e}")))
    }

    /// Fill `out` (int64 `(n, 4)`) with uniformly random mask-conforming actions (deterministic
    /// per-env streams, reseeded by `reset`).
    fn random_actions(
        &mut self,
        py: Python<'_>,
        mut out: PyReadwriteArray2<'_, i64>,
    ) -> PyResult<()> {
        let n = self.pool.num_envs();
        if out.shape() != [n, 4] || !out.is_c_contiguous() {
            return Err(val_err(format!(
                "`out` must be a C-contiguous int64 array of shape ({n}, 4)"
            )));
        }
        let o = out.as_slice_mut().map_err(|e| slice_err("out", e))?;
        let pool = &mut self.pool;
        py.detach(|| pool.random_actions(o));
        Ok(())
    }

    /// Fill `out` (int64 `(n, 4)`) with the action of the fair-information heuristic team agent
    /// for every env's current state (parallel, GIL released). With `select` (bool `(n,)`) only
    /// the envs where it is true are computed; the other rows of `out` are left untouched.
    #[pyo3(signature = (out, select=None))]
    fn teacher_actions(
        &mut self,
        py: Python<'_>,
        mut out: PyReadwriteArray2<'_, i64>,
        select: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<()> {
        let n = self.pool.num_envs();
        if out.shape() != [n, 4] || !out.is_c_contiguous() {
            return Err(val_err(format!(
                "`out` must be a C-contiguous int64 array of shape ({n}, 4)"
            )));
        }
        let o = out.as_slice_mut().map_err(|e| slice_err("out", e))?;
        let sel = match &select {
            Some(s) => {
                if s.shape() != [n] || !s.is_c_contiguous() {
                    return Err(val_err(format!(
                        "`select` must be a C-contiguous bool array of shape ({n},)"
                    )));
                }
                Some(s.as_slice().map_err(|e| slice_err("select", e))?)
            }
            None => None,
        };
        let pool = &mut self.pool;
        py.detach(|| pool.teacher_actions_where(o, sel));
        Ok(())
    }

    /// Full state of env `i` as a JSON string.
    fn export_state(&self, i: usize) -> PyResult<String> {
        self.pool.export_state(i).map_err(val_err)
    }

    /// Replace env `i`'s run by an exported state (JSON string) and write its observation and
    /// masks into row `i` of `buffers`. `reseed` (u64) re-samples the hidden information of the
    /// imported state first (`streams`: list of stream names, default all). The imported run is a
    /// fresh episode for the interface (counters restart, flagged `imported` in
    /// `drain_finished_ex`). Raises on malformed JSON, invariant violations or a finished run.
    #[pyo3(signature = (buffers, i, state, reseed=None, streams=None))]
    fn import_state(
        &mut self,
        buffers: &Bound<'_, PyDict>,
        i: usize,
        state: &str,
        reseed: Option<u64>,
        streams: Option<Vec<String>>,
    ) -> PyResult<()> {
        let streams = parse_streams(streams).map_err(val_err)?;
        let mut pb = PyBufs::extract(buffers, self.pool.num_envs())?;
        let bufs = pb.bufs()?;
        self.pool
            .import_state(i, state, reseed, streams.as_deref(), &bufs)
            .map_err(val_err)
    }

    /// Per-env start joker (list of `num_envs` names or `None`), added to the fresh run at the
    /// next `reset()` / auto-reset of that env. `None` entries: no start joker.
    fn set_start_jokers(&mut self, jokers: Vec<Option<String>>) -> PyResult<()> {
        let ids = jokers
            .iter()
            .map(|j| j.as_deref().map(parse_joker).transpose())
            .collect::<Result<Vec<_>, _>>()
            .map_err(val_err)?;
        self.pool.set_start_jokers(&ids).map_err(val_err)
    }

    /// Like `drain_finished` (and sharing its queue) with named fields: a list of dicts with
    /// `env, ante, code, steps, return, blinds_beaten, invalid, deck, stake, imported,
    /// start_joker`.
    fn drain_finished_ex<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.pool
            .drain_finished()
            .into_iter()
            .map(|f| {
                let d = PyDict::new(py);
                d.set_item("env", f.env)?;
                d.set_item("ante", f.ante)?;
                d.set_item("code", f.code)?;
                d.set_item("steps", f.steps)?;
                d.set_item("return", f.ret)?;
                d.set_item("blinds_beaten", f.rounds_won)?;
                d.set_item("invalid", f.invalid)?;
                d.set_item("deck", f.deck.name())?;
                d.set_item("stake", f.stake.name())?;
                d.set_item("imported", f.imported)?;
                d.set_item(
                    "start_joker",
                    f.start_joker.and_then(|id| {
                        balatro_core::content::jokers::all_defs()
                            .find(|d| d.id == id)
                            .map(|d| d.name)
                    }),
                )?;
                Ok(d)
            })
            .collect()
    }

    /// Episodes finished since the last call, as `(env, ante, done_code, steps, return,
    /// blinds_beaten, invalid_actions)` tuples (and cleared).
    #[allow(clippy::type_complexity)]
    fn drain_finished(&mut self) -> Vec<(u32, i32, u8, u32, f32, u32, u32)> {
        self.pool
            .drain_finished()
            .into_iter()
            .map(|f| {
                (
                    f.env,
                    f.ante,
                    f.code,
                    f.steps,
                    f.ret,
                    f.rounds_won,
                    f.invalid,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{derive_seed, parse_joker};
    use crate::layout::{GLOBAL_I, Owned};
    use balatro_core::content::{decks::DeckKind, stakes::StakeKind};

    fn run(threads: usize, n: usize, steps: usize) -> (Owned, Vec<f32>, Vec<u8>) {
        let mut p = EnvPool::new(
            n,
            7,
            threads,
            GameConfig::default(),
            500,
            RewardConfig::default(),
            ConfigPool::EMPTY,
        )
        .unwrap();
        let mut o = Owned::new(n);
        {
            let b = o.bufs();
            p.reset(None, &b);
        }
        let mut acts = vec![0i64; n * 4];
        let (mut all_r, mut all_d) = (Vec::new(), Vec::new());
        for _ in 0..steps {
            p.random_actions(&mut acts);
            let b = o.bufs();
            p.step(&acts, &b).unwrap();
            all_r.extend_from_slice(&o.reward);
            all_d.extend_from_slice(&o.done);
        }
        (o, all_r, all_d)
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let a = run(1, 37, 400);
        let b = run(5, 37, 400);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
        assert_eq!(a.0.global_f, b.0.global_f);
        assert_eq!(a.0.hand_f, b.0.hand_f);
        assert_eq!(a.0.offer_i, b.0.offer_i);
        assert_eq!(a.0.arg0_mask, b.0.arg0_mask);
        assert_eq!(a.0.sel_candidates, b.0.sel_candidates);
        assert!(a.2.iter().any(|&d| d != 0), "some episode should end");
        assert!(a.0.invalid.iter().all(|&i| i == 0));
    }

    fn pool(n: usize, seed: u64, threads: usize, max_steps: u32) -> EnvPool {
        EnvPool::new(
            n,
            seed,
            threads,
            GameConfig::default(),
            max_steps,
            RewardConfig::default(),
            ConfigPool::EMPTY,
        )
        .unwrap()
    }

    #[test]
    fn export_import_round_trip_continues_identically() {
        let n = 5;
        let mut src = pool(n, 4, 3, 3_000);
        let mut o = Owned::new(n);
        {
            let b = o.bufs();
            src.reset(None, &b);
        }
        let mut acts = vec![0i64; n * 4];
        for _ in 0..60 {
            src.teacher_actions_where(&mut acts, None);
            let b = o.bufs();
            src.step(&acts, &b).unwrap();
        }
        let state = src.export_state(2).unwrap();
        // Import into another pool (different thread count and seed).
        let mut dst = pool(3, 99, 1, 3_000);
        let mut o2 = Owned::new(3);
        {
            let b = o2.bufs();
            dst.reset(None, &b);
            dst.import_state(1, &state, None, None, &b).unwrap();
        }
        assert_eq!(
            o.hand_f[2 * (o.hand_f.len() / n)..3 * (o.hand_f.len() / n)],
            o2.hand_f[o2.hand_f.len() / 3..2 * (o2.hand_f.len() / 3)]
        );
        for _ in 0..200 {
            let mut a = vec![0i64; n * 4];
            src.teacher_actions_where(&mut a, Some(&[false, false, true, false, false]));
            let mut b3 = vec![0i64; 3 * 4];
            b3[4..8].copy_from_slice(&a[8..12]);
            {
                let b = o.bufs();
                src.step(&a, &b).unwrap();
            }
            let r_src = o.reward[2];
            let d_src = o.done[2];
            {
                let b = o2.bufs();
                dst.step(&b3, &b).unwrap();
            }
            assert_eq!(r_src, o2.reward[1]);
            assert_eq!(d_src, o2.done[1]);
            if d_src != 0 {
                break;
            }
        }
        // Flagged as imported, counters restart.
        let fin = dst.drain_finished();
        assert!(fin.iter().all(|f| f.env != 1 || f.imported));
        assert!(dst.import_state(7, &state, None, None, &o2.bufs()).is_err());
        assert!(dst.import_state(0, "{", None, None, &o2.bufs()).is_err());
    }

    #[test]
    fn start_joker_applies_at_reset_and_auto_reset() {
        let n = 2;
        let mut p = pool(n, 6, 1, 3);
        p.set_start_jokers(&[None, Some(parse_joker("greedy joker").unwrap())])
            .unwrap();
        assert!(p.set_start_jokers(&[None]).is_err());
        let mut o = Owned::new(n);
        {
            let b = o.bufs();
            p.reset(None, &b);
        }
        let per = o.joker_valid.len() / n;
        assert!(!o.joker_valid[0]);
        assert!(o.joker_valid[per]);
        let mut acts = vec![0i64; n * 4];
        let mut resets = 0;
        for _ in 0..12 {
            p.random_actions(&mut acts);
            let b = o.bufs();
            p.step(&acts, &b).unwrap();
            if o.done[1] != 0 {
                resets += 1;
                assert!(o.joker_valid[per]);
            }
        }
        assert!(resets >= 3);
    }

    #[test]
    fn seeds_are_distinct() {
        let s: std::collections::HashSet<u64> = (0..100)
            .flat_map(|e| (0..10).map(move |k| derive_seed(1, e, k)))
            .collect();
        assert_eq!(s.len(), 1000);
    }

    #[test]
    fn teacher_actions_are_valid_and_thread_independent() {
        let go = |threads: usize| {
            let n = 6;
            let mut p = EnvPool::new(
                n,
                3,
                threads,
                GameConfig::default(),
                2_000,
                RewardConfig::default(),
                ConfigPool::EMPTY,
            )
            .unwrap();
            let mut o = Owned::new(n);
            {
                let b = o.bufs();
                p.reset(None, &b);
            }
            let mut acts = vec![0i64; n * 4];
            let mut log = Vec::new();
            for _ in 0..120 {
                p.teacher_actions_where(&mut acts, None);
                log.extend_from_slice(&acts);
                let b = o.bufs();
                p.step(&acts, &b).unwrap();
                assert!(o.invalid.iter().all(|&i| i == 0), "teacher action rejected");
            }
            log
        };
        assert_eq!(go(1), go(3));
    }

    #[test]
    fn masked_teacher_actions_only_touch_selected_rows() {
        let n = 6;
        let mut p = EnvPool::new(
            n,
            3,
            2,
            GameConfig::default(),
            2_000,
            RewardConfig::default(),
            ConfigPool::EMPTY,
        )
        .unwrap();
        let mut o = Owned::new(n);
        {
            let b = o.bufs();
            p.reset(None, &b);
        }
        let mut full = vec![0i64; n * 4];
        p.teacher_actions_where(&mut full, None);
        let mut q = EnvPool::new(
            n,
            3,
            1,
            GameConfig::default(),
            2_000,
            RewardConfig::default(),
            ConfigPool::EMPTY,
        )
        .unwrap();
        let mut o2 = Owned::new(n);
        {
            let b = o2.bufs();
            q.reset(None, &b);
        }
        let sel = [true, false, true, false, false, true];
        let mut part = vec![-7i64; n * 4];
        q.teacher_actions_where(&mut part, Some(&sel));
        for i in 0..n {
            let row = &part[4 * i..4 * i + 4];
            if sel[i] {
                assert_eq!(row, &full[4 * i..4 * i + 4]);
            } else {
                assert_eq!(row, &[-7i64; 4]);
            }
        }
    }

    #[test]
    fn sampled_configs_are_thread_independent_and_cover_the_pool() {
        let pool = ConfigPool::product(
            &[DeckKind::Red, DeckKind::Plasma, DeckKind::Green],
            &[StakeKind::White, StakeKind::Gold],
        );
        assert_eq!(pool.len(), 6);
        let go = |threads: usize| {
            let n = 24;
            let mut p = EnvPool::new(
                n,
                9,
                threads,
                GameConfig::default(),
                300,
                RewardConfig::default(),
                pool,
            )
            .unwrap();
            let mut o = Owned::new(n);
            {
                let b = o.bufs();
                p.reset(None, &b);
            }
            let mut acts = vec![0i64; n * 4];
            let mut seen = std::collections::HashSet::new();
            let mut log = Vec::new();
            for _ in 0..200 {
                for i in 0..n {
                    seen.insert((
                        o.global_i[i * GLOBAL_I.len() + 8],
                        o.global_i[i * GLOBAL_I.len() + 9],
                    ));
                }
                log.extend_from_slice(&o.global_i);
                p.random_actions(&mut acts);
                let b = o.bufs();
                p.step(&acts, &b).unwrap();
            }
            (seen, log)
        };
        let (a, la) = go(1);
        let (b, lb) = go(4);
        assert_eq!(la, lb);
        assert_eq!(a, b);
        assert_eq!(a.len(), 6, "every (deck, stake) pair is sampled: {a:?}");
        assert!(
            a.iter()
                .all(|&(d, s)| [0, 3, 13].contains(&d) && [0, 7].contains(&s))
        );
    }
}
