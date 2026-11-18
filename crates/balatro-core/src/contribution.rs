//! Derived values for the RL observation (interface v3, `docs/design/rl_interface.md` §3.2 and
//! §3.4): the effective hands / discards of the next round and a per-joker "current
//! contribution" `(chips, mult, xmult, money)`.
//!
//! **Reference hand.** A contribution is what the joker would add to the *next scored hand*:
//!
//! * in `Phase::Play`: the live round (`hands_left`, `discards_left`, `discards_used_round`,
//!   `hand_plays_round`, the draw pile, the visible hand);
//! * in every other phase (Blind select, Shop, open pack): the first hand of the next round,
//!   before any hand or discard: hands / discards = [`next_round_budget`], no discard used, no
//!   hand played this round, draw pile = deck size - hand size.
//!
//! Money, joker list, sell values, the full-deck counts and the run counters are read as they
//! are now. The full-deck counts (Steel, Stone, Driver's License, Cloud 9, Erosion) are what the
//! scoring hooks read too; the real game shows them on the joker ("Currently X1.6").
//!
//! **What is in the block.**
//!
//! * Unconditional effects: their value (Joker +4 mult, Stuntman +250 chips, Misprint the
//!   expected +11.5 mult).
//! * Effects derived from the state: one arm per joker (Banner, Raised Fist, Abstract, Blue
//!   Joker, Swashbuckler, Delayed Gratification, Mystic Summit, Bull, Bootstraps, Joker Stencil,
//!   Fortune Teller, Throwback, Satellite, Loyalty Card, Erosion, Stone, Steel, Driver's License,
//!   Cloud 9, Acrobat, Supernova, Card Sharp, Baseball Card, Golden Joker, Rocket, To the Moon),
//!   evaluated on the reference hand with the same constants (and, where there is logic, the
//!   same helper functions) as the hooks.
//! * Scaling jokers: a generic arm reads `value` (+mult, +chips or the xmult factor, by joker).
//! * Conditional effects that depend on the *played* hand (suits, ranks, "contains a Pair",
//!   Half Joker's card count, held Kings / Queens, probabilities on scored cards) are 0: the
//!   item id carries them. Supernova and Card Sharp are evaluated for the most favourable hand
//!   type (the player chooses it): Supernova = most plays of any type + 1 (the hand itself),
//!   Card Sharp = X3 when some type was already played this round.
//! * Raised Fist: Play = 2 x chips of the lowest *visible* non-Stone card in hand (0 if that card
//!   is debuffed; assumes it stays held); otherwise the exact expectation of `2 x chips(lowest
//!   rank)` over a uniformly random `hand_size`-card hand from the full deck (hypergeometric,
//!   Stone cards take a slot but never count; the played cards are assumed not to include it).
//! * Burglar: 0 (its effect is in [`next_round_budget`]). Editions are not included (they are
//!   their own input). `money` is per round: round-end payouts (Golden, Cloud 9, Rocket,
//!   Satellite, Delayed Gratification) and To the Moon's extra interest at the current money.
//! * Copiers (Blueprint, Brainstorm) give their resolved source's value as a copy (round-end
//!   and passive effects are never copied: their money is 0). Debuffed jokers give nothing.

use std::cell::OnceCell;

use crate::blinds::BlindKind;
use crate::content::ids::BossId;
use crate::content::jokers::{
    JokerId, prototypes as p, try_def, wave1a as w1a, wave1b as w1b, wave2a as w2a, wave2b as w2b,
    wave3 as w3, wave4a as w4a,
};
use crate::economy;
use crate::enums::{Enhancement, Phase, Rank, Rarity};
use crate::hooks::{DeckCounts, resolve_source};
use crate::state::GameState;

/// What one joker adds to the reference hand. `xmult == 1.0` is no factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contribution {
    /// `+chips`.
    pub chips: f64,
    /// `+mult`.
    pub mult: f64,
    /// `xmult` factor.
    pub xmult: f64,
    /// Dollars per round.
    pub money: f64,
}

impl Contribution {
    /// No effect.
    pub const NONE: Contribution = Contribution {
        chips: 0.0,
        mult: 0.0,
        xmult: 1.0,
        money: 0.0,
    };

    fn chips(x: f64) -> Self {
        Contribution {
            chips: x,
            ..Self::NONE
        }
    }

    fn mult(x: f64) -> Self {
        Contribution {
            mult: x,
            ..Self::NONE
        }
    }

    fn xmult(x: f64) -> Self {
        Contribution {
            xmult: x,
            ..Self::NONE
        }
    }

    fn money(x: f64) -> Self {
        Contribution {
            money: x,
            ..Self::NONE
        }
    }
}

impl Default for Contribution {
    fn default() -> Self {
        Self::NONE
    }
}

/// Number of jokers (debuffed ones excluded) whose blind-select hooks resolve to Burglar
/// (Blueprint / Brainstorm copies count).
fn burglars(s: &GameState) -> u32 {
    (0..s.jokers.len())
        .filter(|&i| {
            resolve_source(&s.jokers, i).is_some_and(|k| s.jokers[k].id == JokerId::Burglar)
        })
        .count() as u32
}

/// `(hands, discards)` of the next round, as the round will start: base + joker / voucher /
/// deck / stake modifiers (`round_hands` / `round_discards`), then the boss set-up when the next
/// blind is The Needle (1 hand) or The Water (0 discards) and no active Chicot disables it, then
/// every Burglar (+3 hands, discards lost).
///
/// In `Phase::Play` it is what the *current* round started with: `hands_left +
/// hands_played_round`, `discards_left + discards_used_round` (the same quantity, observed).
/// Not modelled: the order-dependent Water + Burglar + Chicot combination (Chicot runs the
/// Water's give-back after Burglar cleared the discards).
pub fn next_round_budget(s: &GameState) -> (u32, u32) {
    if s.phase == Phase::Play {
        return (
            s.hands_left.saturating_add(s.hands_played_round),
            s.discards_left.saturating_add(s.discards_used_round),
        );
    }
    let mut hands = s.round_hands();
    let mut discards = s.round_discards();
    let chicot = s
        .jokers
        .iter()
        .any(|j| j.id == JokerId::Chicot && !j.debuffed);
    if s.blind == BlindKind::Boss && !s.boss_disabled && !chicot {
        match s.boss {
            Some(BossId::TheNeedle) => hands = hands.min(1),
            Some(BossId::TheWater) => discards = 0,
            _ => {}
        }
    }
    let n = burglars(s);
    if n > 0 {
        let add = u32::try_from(w4a::BURGLAR_HANDS)
            .unwrap_or(0)
            .saturating_mul(n);
        hands = hands.saturating_add(add);
        discards = 0;
    }
    (hands, discards)
}

/// The reference hand of a state (see the module docs), shared by every joker of one
/// observation. Full-deck counts are computed lazily (only if a joker reads them).
pub struct ContribCtx<'a> {
    s: &'a GameState,
    in_play: bool,
    /// Hands left before the reference hand is played (the reference hand included).
    hands: u32,
    /// Discards left at the reference hand.
    discards: u32,
    /// Discards used this round before the reference hand.
    discards_used: u32,
    deck: OnceCell<DeckCounts>,
}

impl<'a> ContribCtx<'a> {
    /// Reference hand of `s`.
    pub fn new(s: &'a GameState) -> Self {
        let in_play = s.phase == Phase::Play;
        let (hands, discards, discards_used) = if in_play {
            (s.hands_left, s.discards_left, s.discards_used_round)
        } else {
            let (h, d) = next_round_budget(s);
            (h, d, 0)
        };
        ContribCtx {
            s,
            in_play,
            hands,
            discards,
            discards_used,
            deck: OnceCell::new(),
        }
    }

    fn deck(&self) -> &DeckCounts {
        self.deck.get_or_init(|| self.s.deck_counts())
    }

    /// Cards left in the draw pile when the reference hand scores.
    fn draw_pile(&self) -> u32 {
        if self.in_play {
            self.s.deck.draw_pile.len() as u32
        } else {
            (self.s.deck.len() as u32).saturating_sub(self.s.hand_size())
        }
    }

    /// Expected Raised Fist mult (see the module docs).
    fn raised_fist(&self) -> f64 {
        let factor = p::RAISED_FIST_FACTOR;
        if self.in_play {
            let mut low: Option<crate::card::Card> = None;
            for c in self.s.deck.hand.iter().filter_map(|c| c.visible_card()) {
                if !c.is_stone() && low.is_none_or(|b| c.rank <= b.rank) {
                    low = Some(c);
                }
            }
            return match low {
                Some(c) if !c.debuffed => factor * f64::from(c.rank.chip_value()),
                _ => 0.0,
            };
        }
        let d = self.deck();
        let n = d.total as usize;
        let h = (self.s.hand_size() as usize).min(n);
        if h == 0 {
            return 0.0;
        }
        // P(no card below rank r among h drawn) = C(n - below, h) / C(n, h).
        let p_ge = |below: usize| -> f64 {
            if n - below < h {
                return 0.0;
            }
            (0..h)
                .map(|i| (n - below - i) as f64 / (n - i) as f64)
                .product()
        };
        let mut below = 0usize;
        let mut prev = 1.0; // P(min >= Two)
        let mut e = 0.0;
        for r in Rank::ALL {
            below += d.by_rank[r.index()] as usize;
            let next = p_ge(below.min(n)); // P(min > r) (or no non-Stone card at all)
            e += (prev - next) * f64::from(r.chip_value());
            prev = next;
        }
        factor * e
    }

    /// Contribution of the joker at `slot` (0 for an empty, hidden or debuffed slot).
    pub fn joker(&self, slot: usize) -> Contribution {
        let s = self.s;
        if s.visible_joker(slot).is_none() {
            return Contribution::NONE;
        }
        let Some(src) = resolve_source(&s.jokers, slot) else {
            return Contribution::NONE;
        };
        let copied = src != slot;
        let j = &s.jokers[src];
        let money = f64::from(s.money.max(0));
        let positive = |v: f64| v.max(0.0);
        let factor = |v: f64| v.max(1.0);
        use Contribution as C;
        use JokerId as J;
        match j.id {
            // --- unconditional ---------------------------------------------------------------
            J::Joker => C::mult(p::JOKER_MULT),
            J::GrosMichel => C::mult(w2a::GROS_MICHEL_MULT),
            J::Cavendish => C::xmult(p::CAVENDISH_XMULT),
            J::Stuntman => C::chips(w1b::STUNTMAN_CHIPS),
            J::Misprint => {
                C::mult((w1a::MISPRINT_MIN + w1a::MISPRINT_MAX_EXCLUSIVE - 1) as f64 / 2.0)
            }
            // --- derived from the state ------------------------------------------------------
            J::Banner => C::chips(w1a::BANNER_CHIPS_PER_DISCARD * f64::from(self.discards)),
            J::MysticSummit => {
                if self.discards == 0 {
                    C::mult(w1a::SUMMIT_MULT)
                } else {
                    C::NONE
                }
            }
            J::RaisedFist => C::mult(self.raised_fist()),
            J::AbstractJoker => C::mult(w1a::ABSTRACT_MULT_PER_JOKER * s.jokers.len() as f64),
            J::BlueJoker => C::chips(w1b::BLUE_JOKER_CHIPS_PER_CARD * f64::from(self.draw_pile())),
            J::Swashbuckler => {
                C::mult(positive(f64::from(w1b::swashbuckler_total(&s.jokers, src))))
            }
            J::Bull => C::chips(p::BULL_CHIPS_PER_DOLLAR * money),
            J::Bootstraps => {
                C::mult(w1b::BOOTSTRAPS_MULT * f64::from(s.money.max(0) / w1b::BOOTSTRAPS_STEP))
            }
            J::JokerStencil => {
                let empty = w1a::stencil_empty_slots(s.joker_capacity(), &s.jokers);
                C::xmult(factor(w1a::STENCIL_XMULT_PER_SLOT * empty as f64))
            }
            J::FortuneTeller => {
                C::mult(w2a::FORTUNE_TELLER_MULT_PER_TAROT * f64::from(s.stats.run.tarots_used))
            }
            J::Throwback => {
                C::xmult(1.0 + w2b::THROWBACK_GAIN * f64::from(s.stats.run.blinds_skipped))
            }
            J::LoyaltyCard => {
                if w2a::loyalty_fires(s.stats.hands_played.saturating_add(1), j.counter) {
                    C::xmult(w2a::LOYALTY_XMULT)
                } else {
                    C::NONE
                }
            }
            J::Erosion => {
                let d = self.deck();
                let missing = d.starting_size.saturating_sub(d.total);
                C::mult(w1b::EROSION_MULT_PER_CARD * f64::from(missing))
            }
            J::StoneJoker => C::chips(
                w1b::STONE_JOKER_CHIPS_PER_CARD
                    * f64::from(self.deck().enhancement(Enhancement::Stone)),
            ),
            J::SteelJoker => C::xmult(
                1.0 + w1a::STEEL_XMULT_PER_CARD
                    * f64::from(self.deck().enhancement(Enhancement::Steel)),
            ),
            J::DriversLicense => {
                if self.deck().enhanced() >= w1b::DRIVERS_LICENSE_MIN_ENHANCED {
                    C::xmult(w1b::DRIVERS_LICENSE_XMULT)
                } else {
                    C::NONE
                }
            }
            J::Acrobat => {
                if self.hands == 1 {
                    C::xmult(w1b::ACROBAT_XMULT)
                } else {
                    C::NONE
                }
            }
            J::Supernova => {
                let most = s.hand_plays_run.iter().copied().max().unwrap_or(0);
                C::mult(f64::from(most.saturating_add(1)))
            }
            J::CardSharp => {
                let repeat = self.in_play && s.hand_plays_round.iter().any(|&n| n > 0);
                if repeat {
                    C::xmult(w2a::CARD_SHARP_XMULT)
                } else {
                    C::NONE
                }
            }
            J::BaseballCard => {
                let n = s
                    .jokers
                    .iter()
                    .enumerate()
                    .filter(|&(k, o)| {
                        k != slot && try_def(o.id).is_some_and(|d| d.rarity == Rarity::Uncommon)
                    })
                    .count();
                C::xmult(p::BASEBALL_XMULT.powi(n as i32))
            }
            // --- money per round (round-end and passive effects are never copied) -------------
            J::DelayedGratification if !copied => {
                if self.discards_used == 0 && self.discards > 0 {
                    C::money(f64::from(w3::DELAYED_PER_DISCARD) * f64::from(self.discards))
                } else {
                    C::NONE
                }
            }
            J::GoldenJoker if !copied => C::money(f64::from(p::GOLDEN_MONEY)),
            J::Cloud9 if !copied => {
                C::money(f64::from(w3::CLOUD9_PER_NINE) * f64::from(self.deck().rank(Rank::Nine)))
            }
            J::Rocket if !copied => {
                let boss = if s.blind == BlindKind::Boss {
                    w3::ROCKET_BOSS_GAIN
                } else {
                    0.0
                };
                C::money(f64::from((j.value + boss) as i32))
            }
            J::Satellite if !copied => C::money(
                f64::from(w3::SATELLITE_PER_PLANET) * f64::from(s.stats.run.unique_planets_used),
            ),
            J::ToTheMoon if !copied => {
                if s.config.deck.def().no_interest {
                    C::NONE
                } else {
                    C::money(f64::from(economy::interest(s.money, s.interest_cap, 0)))
                }
            }
            // --- scaling jokers: `value` ------------------------------------------------------
            // Green Joker gains +1 before every hand scores (unconditional): the next hand pays
            // `value + 1`. The other before-scoring gains are conditional on the hand.
            J::GreenJoker => C::mult(positive(j.value) + w2a::GREEN_HAND_GAIN),
            J::RideTheBus
            | J::RedCard
            | J::FlashCard
            | J::SpareTrousers
            | J::CeremonialDagger
            | J::Popcorn => C::mult(positive(j.value)),
            J::Runner | J::SquareJoker | J::WeeJoker | J::Castle | J::IceCream => {
                C::chips(positive(j.value))
            }
            J::Constellation
            | J::Madness
            | J::Vampire
            | J::Hologram
            | J::Obelisk
            | J::LuckyCat
            | J::Campfire
            | J::GlassJoker
            | J::Canio
            | J::Yorick
            | J::HitTheRoad => C::xmult(factor(j.value)),
            J::Ramen => C::xmult(if j.value > w2b::RAMEN_FLOOR {
                j.value
            } else {
                1.0
            }),
            _ => C::NONE,
        }
    }
}

/// Contribution of the joker at `slot` of `s` (see the module docs). For many jokers of one
/// state build one [`ContribCtx`] instead.
pub fn joker_contribution(s: &GameState, slot: usize) -> Contribution {
    ContribCtx::new(s).joker(slot)
}
