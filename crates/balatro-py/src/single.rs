//! `Game`: one run exposed to Python for debugging (state JSON, legal actions as text, step by
//! legal-action index). Not performance-critical; use `VecEnv` for training.

use crate::interface::{self, OBS_DIM};
use crate::vec_env::parse_config;
use balatro_core::actions::Action;
use balatro_core::game::Game as CoreGame;
use balatro_core::serialize::state_to_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One run. Actions are indices into [`Game::legal_actions`] (same convention as `VecEnv`).
#[pyclass(name = "Game", module = "balatro_rl._engine")]
pub struct PyGame {
    game: CoreGame,
    legal: Vec<Action>,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed=0, deck="red", stake="white", endless=false))]
    fn new(seed: u64, deck: &str, stake: &str, endless: bool) -> PyResult<Self> {
        let config = parse_config(deck, stake, endless).map_err(PyValueError::new_err)?;
        let game = CoreGame::new(seed, config);
        let mut legal = Vec::new();
        game.legal_actions_into(&mut legal);
        Ok(PyGame { game, legal })
    }

    /// Full state as JSON.
    fn state_json(&self) -> String {
        state_to_json(self.game.state(), false)
    }

    /// Legal actions as text (`Debug` format), in index order.
    fn legal_actions(&self) -> Vec<String> {
        self.legal.iter().map(|a| format!("{a:?}")).collect()
    }

    /// Number of legal actions.
    fn legal_count(&self) -> usize {
        self.legal.len()
    }

    /// Apply legal action `index`; returns `(reward, done_code)` (done codes as in `VecEnv`;
    /// no auto-reset here).
    fn step(&mut self, index: usize) -> PyResult<(f32, u8)> {
        let Some(&action) = self.legal.get(index) else {
            return Err(PyValueError::new_err(format!(
                "action index {index} out of range (legal_count = {})",
                self.legal.len()
            )));
        };
        let info = self
            .game
            .step(action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reward = interface::step_reward(&info, &self.game);
        self.game.legal_actions_into(&mut self.legal);
        Ok((reward, interface::done_code(&self.game)))
    }

    /// The run is over.
    fn is_terminal(&self) -> bool {
        self.game.is_terminal()
    }

    /// `"win"`, `"loss"` or `None` while running.
    fn outcome(&self) -> Option<&'static str> {
        use balatro_core::state::Outcome;
        self.game.outcome().map(|o| match o {
            Outcome::Win => "win",
            Outcome::Loss => "loss",
        })
    }

    /// The placeholder observation vector (same as the `VecEnv` row).
    fn observation(&self) -> Vec<f32> {
        let mut v = vec![0.0; OBS_DIM];
        interface::write_obs(&self.game, self.legal.len(), &mut v);
        v
    }
}
