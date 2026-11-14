//! Uniform (optionally Play-biased) random agent.

use super::Agent;
use crate::actions::Action;
use crate::game::Game;
use crate::rng::{GameRng, RngKey};

/// Picks among the legal actions at random, with its own seeded RNG.
#[derive(Debug, Clone)]
pub struct RandomAgent {
    rng: GameRng,
    play_weight: f64,
    legal: Vec<Action>,
    weights: Vec<f64>,
}

impl RandomAgent {
    /// Uniform over legal actions.
    pub fn new(seed: u64) -> Self {
        Self::with_play_weight(seed, 1.0)
    }

    /// `play_weight` multiplies the weight of every `Play` action (others weigh 1), e.g. 5.0
    /// plays five times as often as it would uniformly. Non-positive values are treated as 1.
    pub fn with_play_weight(seed: u64, play_weight: f64) -> Self {
        RandomAgent {
            rng: GameRng::new(seed ^ 0xA6E7_0001_5EED_0001),
            play_weight: if play_weight > 0.0 { play_weight } else { 1.0 },
            legal: Vec::new(),
            weights: Vec::new(),
        }
    }
}

impl Agent for RandomAgent {
    fn act(&mut self, game: &Game) -> Action {
        game.legal_actions_into(&mut self.legal);
        if self.legal.is_empty() {
            // Terminal game: nothing legal. Return a harmless action (step will reject it).
            return Action::EndShop;
        }
        let idx = if (self.play_weight - 1.0).abs() < f64::EPSILON {
            self.rng.range(RngKey::Misc, 0, self.legal.len())
        } else {
            self.weights.clear();
            let w = self.play_weight;
            self.weights.extend(
                self.legal
                    .iter()
                    .map(|a| if matches!(a, Action::Play(_)) { w } else { 1.0 }),
            );
            self.rng
                .weighted_index(RngKey::Misc, &self.weights)
                .unwrap_or(0)
        };
        self.legal[idx]
    }
}
