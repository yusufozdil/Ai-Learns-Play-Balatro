//! Reference agents for baselines and robustness tests.
//!
//! An agent owns its own seeded RNG (never the game's `state.rng`) and only reads the game.

pub mod greedy;
pub mod random;

use crate::actions::Action;
use crate::game::Game;

pub use greedy::GreedyAgent;
pub use random::RandomAgent;

/// A policy: picks one legal action for a non-terminal game.
pub trait Agent {
    /// Choose the next action. The game must not be terminal; the returned action is legal.
    fn act(&mut self, game: &Game) -> Action;
}
