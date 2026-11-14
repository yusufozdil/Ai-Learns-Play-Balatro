//! Heuristic team agent (Phase 5D): one rule-based expert per phase plus a shared strategy state.
//!
//! * **Blind expert** (`market.rs::blind_action`): Select, or Skip a Small/Big blind for a
//!   valuable tag; uses planets/utility consumables and arranges jokers before the blind.
//! * **Hand expert** (`hand.rs`): previews every selection (`Game::preview_play`), wins the round
//!   with the cheapest scoring play, otherwise decides Play vs Discard with a Monte-Carlo estimate
//!   of the best hand after each of a few structured discards; uses consumables on the cards it
//!   is about to play; handles Verdant Leaf.
//! * **Market expert** (`market.rs::shop_action`): values jokers by simulation (the preview of
//!   sampled hands with and without the joker, plus a static prior, `jokers.rs`), planets by the
//!   same simulation (level-up), packs/vouchers by tables; sells the weakest joker to make room;
//!   arranges the joker order by simulation (xmult right, +chips/+mult left falls out of it);
//!   keeps an interest reserve in `Bank` mode.
//! * **Pack expert** (`market.rs::pack_action`): best joker / planet / tarot target / playing
//!   card of an open pack, or skip.
//!
//! The [`Strategy`] is the state the experts share: target hand weights, dominant suit and the
//! economy mode. The agent owns its RNG (never `state.rng`) and is deterministic given the game.
//!
//! Information note: like the Greedy agent, scoring previews run on a clone of the game's RNG, so
//! probabilistic card/joker effects (Lucky, Misprint, Bloodstone...) of the played hand resolve
//! exactly as the real play would, and face-down cards are read normally. A fair-information
//! teacher would need a preview that hides both.

mod consumables;
mod hand;
mod jokers;
mod market;
mod valuer;

use super::Agent;
use crate::actions::Action;
use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{HandType, Phase, Suit};
use crate::game::Game;
use crate::rng::GameRng;
use crate::scoring::Scorer;
use crate::state::GameState;
use valuer::Valuer;

/// Economy mode: build the joker row, or protect the interest reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EconMode {
    /// Early game or a weak build: money is for jokers and planets.
    Build,
    /// Strong enough: keep money at the interest cap and only buy clear upgrades.
    Bank,
}

/// Run-long shared state of the experts.
#[derive(Debug, Clone)]
pub struct Strategy {
    /// Weight of each hand type as a play target (sums to ~1): sampled best hands under the
    /// current jokers/levels blended with the hands actually played.
    pub target_hands: [f64; HandType::COUNT],
    /// Suit the run builds around (suit jokers, deck composition).
    pub dominant: Suit,
    /// Economy mode.
    pub econ: EconMode,
}

impl Strategy {
    fn new() -> Self {
        Strategy {
            target_hands: [0.0; HandType::COUNT],
            dominant: Suit::Spades,
            econ: EconMode::Build,
        }
    }

    /// The hand type with the largest target weight.
    pub fn top_target(&self) -> HandType {
        // First maximum in `HandType::ALL` order (weights are never negative).
        let mut best = HandType::ALL[0];
        let mut w = self.target_hands[best.index()];
        for h in HandType::ALL {
            if self.target_hands[h.index()] > w {
                w = self.target_hands[h.index()];
                best = h;
            }
        }
        best
    }

    fn refresh(&mut self, s: &GameState) {
        // Dominant suit: deck composition plus the suits the owned jokers pay for.
        let mut score = [0.0f64; 4];
        for c in s.deck.iter_all() {
            for suit in Suit::ALL {
                if c.has_suit(suit, &s.mods.rules) {
                    score[suit.index()] += 1.0;
                }
            }
        }
        for j in &s.jokers {
            let suit = match j.id {
                JokerId::GreedyJoker | JokerId::RoughGem => Some(Suit::Diamonds),
                JokerId::LustyJoker | JokerId::Bloodstone => Some(Suit::Hearts),
                JokerId::WrathfulJoker | JokerId::Arrowhead => Some(Suit::Spades),
                JokerId::GluttonousJoker | JokerId::OnyxAgate => Some(Suit::Clubs),
                _ => None,
            };
            if let Some(su) = suit {
                score[su.index()] += 6.0;
            }
        }
        let mut best = Suit::Spades;
        let mut bv = -1.0;
        for suit in Suit::ALL {
            if score[suit.index()] > bv {
                bv = score[suit.index()];
                best = suit;
            }
        }
        self.dominant = best;
        self.econ = if s.ante <= 1 || s.jokers.len() < 3 {
            EconMode::Build
        } else {
            EconMode::Bank
        };
    }
}

/// Tunable knobs. `Default` is the tuned setting; tests and sweeps may change fields.
#[derive(Debug, Clone)]
pub struct HeuristicConfig {
    /// Sampled hands per market evaluation.
    pub shop_samples: usize,
    /// Cards beyond the hand size in a market sample (the agent sees extra cards by discarding).
    pub shop_extra_cards: usize,
    /// Sampled hands for joker-order search.
    pub arrange_samples: usize,
    /// Sampled draws per discard candidate.
    pub mc_samples: usize,
    /// Doublings of score a dollar must buy (Build mode).
    pub lambda_build: f64,
    /// Same in Bank mode.
    pub lambda_bank: f64,
    /// Rounds of interest counted when a purchase drops money below an interest step.
    pub interest_rounds: f64,
    /// Minimum net value (value minus price penalty) of a purchase.
    pub buy_threshold: f64,
    /// Value (doublings) a tag needs for the agent to skip a blind for it.
    pub skip_threshold: f64,
    /// Money to keep after a reroll when a joker slot is free.
    pub reroll_floor_room: i32,
    /// Money to keep after a reroll when the joker row is full.
    pub reroll_floor_full: i32,
    /// Rerolls per shop.
    pub max_rerolls: u32,
    /// Round score estimate = best-hand estimate x hands x this factor.
    pub power_factor: f64,
    /// Lowest factor the price penalty shrinks to when behind.
    pub min_lambda_scale: f64,
    /// Weight of a planet's level-up value against joker values.
    pub planet_scale: f64,
    /// Antes ahead whose boss requirement the build is measured against.
    pub pressure_horizon: i32,
    /// Relative gain a discard needs over playing now (last hand).
    pub discard_last: f64,
    /// Same with more hands left: base.
    pub discard_base: f64,
    /// Same: extra per hand left per discard left.
    pub discard_slope: f64,
    /// Play without discarding when the best hand already scores `pace x need / hands left`.
    pub pace: f64,
    /// Money to keep after a $10 boss reroll.
    pub boss_reroll_reserve: i32,
    /// Smallest gain of a joker-order move.
    pub arrange_min_gain: f64,
    /// Smallest value of a Buffoon pack joker worth taking.
    pub pack_joker_min: f64,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        HeuristicConfig {
            shop_samples: 8,
            shop_extra_cards: 2,
            arrange_samples: 4,
            mc_samples: 6,
            lambda_build: 0.02,
            lambda_bank: 0.02,
            interest_rounds: 1.0,
            buy_threshold: 0.0,
            skip_threshold: 0.6,
            reroll_floor_room: 6,
            reroll_floor_full: 30,
            max_rerolls: 5,
            power_factor: 2.0,
            min_lambda_scale: 0.15,
            planet_scale: 1.5,
            pressure_horizon: 3,
            discard_last: 1.02,
            discard_base: 1.3,
            discard_slope: 0.3,
            pace: 1.0,
            boss_reroll_reserve: 0,
            arrange_min_gain: 0.02,
            pack_joker_min: 0.15,
        }
    }
}

impl HeuristicConfig {
    /// Set a knob by name (tuning sweeps from the command line); `false` for an unknown name or
    /// an unparsable value.
    pub fn set(&mut self, key: &str, val: &str) -> bool {
        macro_rules! f {
            ($($name:ident),*) => {
                match key {
                    $(stringify!($name) => match val.parse() {
                        Ok(v) => { self.$name = v; true }
                        Err(_) => false,
                    },)*
                    _ => false,
                }
            };
        }
        f!(
            shop_samples,
            shop_extra_cards,
            arrange_samples,
            mc_samples,
            lambda_build,
            lambda_bank,
            interest_rounds,
            buy_threshold,
            skip_threshold,
            reroll_floor_room,
            reroll_floor_full,
            max_rerolls,
            power_factor,
            min_lambda_scale,
            planet_scale,
            pressure_horizon,
            arrange_min_gain,
            pack_joker_min,
            discard_last,
            discard_base,
            discard_slope,
            pace,
            boss_reroll_reserve
        )
    }
}

/// The heuristic team agent.
#[derive(Debug)]
pub struct HeuristicAgent {
    cfg: HeuristicConfig,
    rng: GameRng,
    valuer: Valuer,
    scorer: Scorer,
    strat: Strategy,
    /// Boss requirement over the estimated round score (>1: behind).
    pressure: f64,
    /// Joker-order moves made in the current visit.
    arrange_moves: (u64, u32),
    /// Joker row (uids in order) the order search last settled on.
    arranged_sig: Option<u64>,
    /// Market samples and the visit they belong to.
    samples: Option<(u64, Vec<Vec<Card>>)>,
    /// Joker-order samples and the visit they belong to.
    arrange_samples: Option<(u64, Vec<Vec<Card>>)>,
    /// Loop guard: `(visit key, steps taken in this visit)`.
    guard: (u64, u32),
}

impl HeuristicAgent {
    /// Agent with the default config; `seed` seeds only the agent's own sampling RNG.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, HeuristicConfig::default())
    }

    /// Agent with an explicit config.
    pub fn with_config(seed: u64, cfg: HeuristicConfig) -> Self {
        HeuristicAgent {
            cfg,
            rng: GameRng::new(seed ^ 0x4845_5552_0000_5D5D),
            valuer: Valuer::new(),
            scorer: Scorer::new(),
            strat: Strategy::new(),
            pressure: 1.0,
            arrange_moves: (u64::MAX, 0),
            arranged_sig: None,
            samples: None,
            arrange_samples: None,
            guard: (u64::MAX, 0),
        }
    }

    /// The shared strategy state (read-only).
    pub fn strategy(&self) -> &Strategy {
        &self.strat
    }

    /// Boss requirement over the estimated round score at the last market evaluation.
    pub fn pressure(&self) -> f64 {
        self.pressure
    }

    /// Forget everything tied to one run (call when reusing the agent for a new game).
    pub fn reset(&mut self) {
        self.strat = Strategy::new();
        self.arranged_sig = None;
        self.samples = None;
        self.arrange_samples = None;
        self.arrange_moves = (u64::MAX, 0);
        self.guard = (u64::MAX, 0);
    }

    /// Key of the current "visit" (a round, a shop, a pack): counters reset when it changes.
    fn visit_key(s: &GameState) -> u64 {
        (u64::from(s.round) << 24)
            ^ (u64::from(s.stats.shops_entered) << 8)
            ^ (s.phase as u64)
            ^ (u64::from(s.stats.rounds_won) << 40)
    }

    /// Count a step in the current visit; returns the steps taken so far (before this one).
    fn tick(&mut self, s: &GameState) -> u32 {
        let k = Self::visit_key(s);
        if self.guard.0 != k {
            self.guard = (k, 0);
        }
        let n = self.guard.1;
        self.guard.1 += 1;
        n
    }

    fn fallback(game: &Game) -> Action {
        let s = game.state();
        let legal = game.legal_actions();
        let pick = |f: fn(&Action) -> bool| legal.iter().copied().find(f);
        match s.phase {
            Phase::BlindSelect => Action::SelectBlind,
            Phase::Shop => Action::EndShop,
            Phase::PackOpen => Action::SkipPack,
            _ => pick(|a| matches!(a, Action::Play(_)))
                .or_else(|| pick(|a| matches!(a, Action::Discard(_))))
                .or_else(|| legal.first().copied())
                .unwrap_or(Action::EndShop),
        }
    }
}

impl Agent for HeuristicAgent {
    fn act(&mut self, game: &Game) -> Action {
        let s = game.state();
        self.strat.refresh(s);
        self.valuer.full = false;
        let a = match s.phase {
            Phase::BlindSelect => self.blind_action(game),
            Phase::Play => self.play_action(game),
            Phase::Shop => self.shop_action(game),
            Phase::PackOpen => self.pack_action(game),
            Phase::GameOver | Phase::Won => Action::EndShop,
        };
        if game.validate(a).is_ok() {
            a
        } else {
            Self::fallback(game)
        }
    }
}
