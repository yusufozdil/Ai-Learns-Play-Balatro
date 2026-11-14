//! `GameState` <-> JSON (serde) for replays, debugging and differential testing, plus the
//! [`Replay`] format (seed + config + action list). Phase 2C.
//!
//! Replays rely on determinism: `Game::new(seed, config)` followed by the same actions always
//! reaches the same state. A replay therefore stays valid only while the rules are unchanged;
//! bump [`REPLAY_VERSION`] when a change alters behaviour for existing action lists.

use crate::actions::{Action, ActionError};
use crate::game::{Game, StepInfo};
use crate::state::{GameConfig, GameState};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current [`Replay`] format version.
pub const REPLAY_VERSION: u32 = 1;

/// JSON (de)serialization failure.
#[derive(Debug, Error)]
pub enum SerializeError {
    /// Malformed or non-matching JSON.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// The decoded state breaks a `GameState` invariant.
    #[error("decoded state violates an invariant: {0}")]
    Invariant(String),
}

/// Why a replay could not be played.
#[derive(Debug, Error)]
pub enum ReplayError {
    /// The replay was written by an incompatible format version.
    #[error("unsupported replay version {found} (this build reads version {expected})")]
    Version {
        /// Version found in the file.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },
    /// An action was rejected.
    #[error("action #{step} ({action:?}) was rejected: {source}")]
    InvalidAction {
        /// Zero-based index of the failing action.
        step: usize,
        /// The failing action.
        action: Action,
        /// Why it was rejected.
        source: ActionError,
    },
}

/// JSON of a state; `pretty` selects indentation.
pub fn state_to_json(state: &GameState, pretty: bool) -> String {
    // Serializing plain data with finite numbers cannot fail; never panic if it ever does.
    let r = if pretty {
        serde_json::to_string_pretty(state)
    } else {
        serde_json::to_string(state)
    };
    r.unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

/// Parse a state from JSON and check its invariants.
pub fn state_from_json(json: &str) -> Result<GameState, SerializeError> {
    let state: GameState = serde_json::from_str(json)?;
    state
        .check_invariants()
        .map_err(|e| SerializeError::Invariant(e.to_string()))?;
    Ok(state)
}

/// A recorded run: everything needed to reproduce it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    /// Format version ([`REPLAY_VERSION`]).
    pub version: u32,
    /// Run seed.
    pub seed: u64,
    /// Run options.
    pub config: GameConfig,
    /// Actions in order.
    pub actions: Vec<Action>,
}

impl Replay {
    /// Empty replay for a new run.
    pub fn record(seed: u64, config: GameConfig) -> Self {
        Replay {
            version: REPLAY_VERSION,
            seed,
            config,
            actions: Vec::new(),
        }
    }

    /// Append an action (call only for actions that `step` accepted).
    pub fn push(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Replay from the start; returns the final game, or the step that failed.
    pub fn play(&self) -> Result<Game, ReplayError> {
        if self.version != REPLAY_VERSION {
            return Err(ReplayError::Version {
                found: self.version,
                expected: REPLAY_VERSION,
            });
        }
        let mut game = Game::new(self.seed, self.config);
        for (step, &action) in self.actions.iter().enumerate() {
            game.step(action)
                .map_err(|source| ReplayError::InvalidAction {
                    step,
                    action,
                    source,
                })?;
        }
        Ok(game)
    }

    /// Pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Parse from JSON (the version is checked by [`Replay::play`]).
    pub fn from_json(json: &str) -> Result<Self, SerializeError> {
        Ok(serde_json::from_str(json)?)
    }
}

/// A [`Game`] that records every accepted action into a [`Replay`].
#[derive(Debug)]
pub struct RecordingGame {
    game: Game,
    replay: Replay,
}

impl RecordingGame {
    /// New recorded run.
    pub fn new(seed: u64, config: GameConfig) -> Self {
        RecordingGame {
            game: Game::new(seed, config),
            replay: Replay::record(seed, config),
        }
    }

    /// The game.
    pub fn game(&self) -> &Game {
        &self.game
    }

    /// The replay so far.
    pub fn replay(&self) -> &Replay {
        &self.replay
    }

    /// Step; the action is recorded only if accepted.
    pub fn step(&mut self, action: Action) -> Result<StepInfo, ActionError> {
        let info = self.game.step(action)?;
        self.replay.push(action);
        Ok(info)
    }
}
