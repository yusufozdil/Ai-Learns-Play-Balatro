//! `Game`: one run exposed to Python for debugging and for the interface tests. It speaks the
//! same structured actions, observation and masks as `VecEnv` (no auto-reset: a finished run
//! stays inspectable). Not performance-critical; use `VecEnv` for training.

use crate::env::{DEFAULT_MAX_EPISODE_STEPS, Env, RewardConfig, Shared, parse_config};
use crate::layout::{MASK_NAMES, OBS_NAMES, Owned, sel_row};
use crate::vec_env::parse_reward;
use balatro_core::action_mask::ActionKind;
use balatro_core::actions::Action;
use balatro_core::rng::GameRng;
use balatro_core::serialize::state_to_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// One run (no auto-reset).
#[pyclass(name = "Game", module = "balatro_rl._engine", unsendable)]
pub struct PyGame {
    env: Env,
    sh: Shared,
}

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed=0, deck="red", stake="white", endless=false, reward=None,
                        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS, env_index=0))]
    fn new(
        seed: u64,
        deck: &str,
        stake: &str,
        endless: bool,
        reward: Option<&Bound<'_, PyDict>>,
        max_episode_steps: u32,
        env_index: u32,
    ) -> PyResult<Self> {
        let config = parse_config(deck, stake, endless).map_err(PyValueError::new_err)?;
        let reward: RewardConfig = parse_reward(reward)?;
        let sh = Shared {
            config,
            max_steps: max_episode_steps.max(1),
            base_seed: seed,
            reward,
        };
        // Env `env_index` of a pool with base seed `seed`: the same first episode as
        // `VecEnv(.., seed)[env_index]` (0 by default).
        let env = Env::new(env_index, &sh);
        Ok(PyGame { env, sh })
    }

    /// Full state as JSON.
    fn state_json(&self) -> String {
        state_to_json(self.env.game.state(), false)
    }

    /// The observation as a dict of numpy arrays (per-env shapes, see `layout()`).
    fn observation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.snapshot(py, &OBS_NAMES)
    }

    /// The masks as a dict of numpy arrays (per-env shapes, see `layout()`).
    fn masks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.snapshot(py, &MASK_NAMES)
    }

    /// Apply the structured action `[kind, arg0, arg1, card_mask]`; returns
    /// `(reward, done_code, invalid)`. Illegal or malformed actions are no-ops.
    fn step_parts(
        &mut self,
        kind: i64,
        arg0: i64,
        arg1: i64,
        card_mask: i64,
    ) -> PyResult<(f32, u8, bool)> {
        if self.env.game.is_terminal() {
            return Err(PyValueError::new_err("the run is over"));
        }
        let out = self
            .env
            .step_parts([kind, arg0, arg1, card_mask], &self.sh, false)
            .map_err(PyValueError::new_err)?;
        Ok((out.reward, out.done, out.invalid))
    }

    /// Every legal action as `(kind, arg0, arg1, card_mask)` (enumeration: debug and tests only).
    fn legal_parts(&self) -> Vec<(u8, u8, u8, u32)> {
        let mut v = Vec::new();
        self.env.game.legal_actions_into(&mut v);
        v.iter()
            .map(|a: &Action| {
                let (k, a0, a1, m) = a.parts();
                (k.index() as u8, a0, a1, m)
            })
            .collect()
    }

    /// Legal actions as text (`Debug` format).
    fn legal_actions(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.env.game.legal_actions_into(&mut v);
        v.iter().map(|a| format!("{a:?}")).collect()
    }

    /// Number of legal actions (flat list, may be large).
    fn legal_count(&self) -> usize {
        let mut v = Vec::new();
        self.env.game.legal_count(&mut v)
    }

    /// Core `CardSelect::next_candidates` of `(kind, arg0)` for the partial selection `partial`
    /// (used to check the Python rule). Raises if the choice takes no cards.
    fn select_next_candidates(&self, kind: usize, arg0: u8, partial: u32) -> PyResult<u32> {
        Ok(self.select(kind, arg0)?.next_candidates(partial))
    }

    /// Core `CardSelect::can_finish`.
    fn select_can_finish(&self, kind: usize, arg0: u8, partial: u32) -> PyResult<bool> {
        Ok(self.select(kind, arg0)?.can_finish(partial))
    }

    /// The action the fair-information heuristic agent plays here.
    fn teacher_parts<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let a = self.env.teacher_parts(&self.sh);
        PyTuple::new(py, a)
    }

    /// A uniformly random mask-conforming action (the `VecEnv.random_actions` sampler).
    fn random_parts<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let a = self.env.random_parts();
        PyTuple::new(py, a)
    }

    /// The run is over.
    fn is_terminal(&self) -> bool {
        self.env.game.is_terminal()
    }

    /// `"win"`, `"loss"` or `None` while running.
    fn outcome(&self) -> Option<&'static str> {
        use balatro_core::state::Outcome;
        self.env.game.outcome().map(|o| match o {
            Outcome::Win => "win",
            Outcome::Loss => "loss",
        })
    }

    /// TEST HOOK: mark the hand cards in `mask` face down (their real identity stays in the
    /// state, as with The House/Wheel/Fish/Mark).
    fn debug_face_down(&mut self, mask: u32) {
        let s = self.env.game.state_mut();
        for (i, c) in s.deck.hand.iter_mut().enumerate() {
            if i < 32 && mask >> i & 1 == 1 {
                c.face_down = true;
            }
        }
    }

    /// TEST HOOK: make the run behave as under Cerulean Bell with the card at hand position
    /// `pos` force-selected (needs the Play phase and a card at `pos`). Returns whether it applied.
    fn debug_force_card(&mut self, pos: usize) -> bool {
        use balatro_core::blinds::BlindKind;
        use balatro_core::content::ids::BossId;
        use balatro_core::enums::Phase;
        let s = self.env.game.state_mut();
        let Some(uid) = s.deck.hand.get(pos).map(|c| c.uid) else {
            return false;
        };
        if s.phase != Phase::Play {
            return false;
        }
        s.blind = BlindKind::Boss;
        s.boss = Some(BossId::CeruleanBell);
        s.boss_disabled = false;
        s.boss_state.card_uid = Some(uid);
        self.env.refresh();
        true
    }

    /// TEST HOOK: change everything the player cannot see, leaving everything visible as is:
    /// permute the identities (uids included) among the face-down hand cards, shuffle the draw
    /// pile order and replace the RNG state.
    fn debug_scramble_hidden(&mut self, seed: u64) {
        let mut z = seed;
        let s = self.env.game.state_mut();
        let idx: Vec<usize> = s
            .deck
            .hand
            .iter()
            .enumerate()
            .filter(|(_, c)| c.face_down)
            .map(|(i, _)| i)
            .collect();
        let mut cards: Vec<_> = idx.iter().map(|&i| s.deck.hand[i]).collect();
        for i in (1..cards.len()).rev() {
            cards.swap(i, (splitmix(&mut z) % (i as u64 + 1)) as usize);
        }
        for (&i, c) in idx.iter().zip(cards) {
            s.deck.hand[i] = c;
        }
        let pile = &mut s.deck.draw_pile;
        for i in (1..pile.len()).rev() {
            pile.swap(i, (splitmix(&mut z) % (i as u64 + 1)) as usize);
        }
        s.rng = GameRng::new(splitmix(&mut z));
        self.env.refresh();
    }
}

impl PyGame {
    fn snapshot<'py>(&self, py: Python<'py>, names: &[&str]) -> PyResult<Bound<'py, PyDict>> {
        let mut o = Owned::new(1);
        {
            let b = o.bufs();
            // SAFETY: the only row of a private single-threaded buffer.
            let mut row = unsafe { b.row(0) };
            self.env.write(&mut row);
        }
        o.to_py(py, names, true)
    }

    fn select(&self, kind: usize, arg0: u8) -> PyResult<balatro_core::action_mask::CardSelect> {
        let k = ActionKind::from_index(kind)
            .ok_or_else(|| PyValueError::new_err(format!("kind {kind} out of range")))?;
        if sel_row(kind, usize::from(arg0)).is_none() {
            return Err(PyValueError::new_err("this (kind, arg0) takes no cards"));
        }
        self.env
            .game
            .card_select(k, arg0)
            .ok_or_else(|| PyValueError::new_err("this (kind, arg0) takes no cards here"))
    }
}
