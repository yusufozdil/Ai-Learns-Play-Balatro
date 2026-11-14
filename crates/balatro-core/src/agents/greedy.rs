//! Greedy agent: plays the highest-scoring hand, discards toward a flush/straight or keeps the
//! scoring cards of its best hand when the blind cannot be beaten this turn. In the market it buys
//! and opens packs with simple value rules (`shop_action` / `pack_action`); it always selects the
//! next blind.

use super::Agent;
use crate::actions::Action;
use crate::card::Card;
use crate::consumable::ConsumableKind;
use crate::enums::{HandType, Phase, Suit};
use crate::game::Game;
use crate::hand_eval;
use crate::packs::PackItem;
use crate::scoring::Scorer;
use crate::shop::{self, ShopCardKind};

/// Deterministic one-ply greedy agent.
///
/// Information: by default full (previews clone the real RNG and read face-down cards: the
/// behaviour of every baseline measured before Phase 8A, kept as the default so those numbers
/// stay comparable). [`GreedyAgent::with_fair_info`] makes it decide on [`Game::fair_view`]
/// (hidden cards sampled, independent RNG stream; the seeds come from a counter, so the agent
/// stays deterministic).
#[derive(Debug, Default)]
pub struct GreedyAgent {
    scorer: Scorer,
    fair_info: bool,
    fair_counter: u64,
}

impl GreedyAgent {
    /// New greedy agent (full information).
    pub fn new() -> Self {
        Self::default()
    }

    /// Greedy agent that decides on the fair view of the game (`true`) or on full information.
    pub fn with_fair_info(mut self, fair_info: bool) -> Self {
        self.fair_info = fair_info;
        self
    }

    /// Best `Play` mask by previewed score (lowest mask wins ties), its score and hand type.
    fn best_play(&mut self, game: &Game) -> Option<(u32, f64, HandType)> {
        let n = game.state().deck.hand.len().min(16);
        let mut best: Option<(u32, f64, HandType)> = None;
        for mask in 1u32..(1u32 << n) {
            if mask.count_ones() > 5 {
                continue;
            }
            if let Ok(r) = game.preview_play_with(&mut self.scorer, mask) {
                let score = if r.invalid { 0.0 } else { r.score };
                if best.is_none_or(|(_, s, _)| score > s) {
                    best = Some((mask, score, r.hand_type));
                }
            }
        }
        best
    }

    /// Discard mask chosen by the heuristic, or `None` to play instead.
    fn discard_plan(game: &Game, best_mask: u32, best_type: HandType) -> Option<u32> {
        let s = game.state();
        let hand = &s.deck.hand;
        let rules = &s.mods.rules;
        if best_type >= HandType::Flush {
            return None;
        }
        // Flush chase: most common suit.
        let mut flush_k = 0usize;
        let mut flush_mask = 0u32;
        for suit in Suit::ALL {
            let mut m = 0u32;
            let mut k = 0usize;
            for (i, c) in hand.iter().enumerate().take(16) {
                if c.has_suit(suit, rules) {
                    m |= 1 << i;
                    k += 1;
                }
            }
            if k > flush_k {
                flush_k = k;
                flush_mask = m;
            }
        }
        // Straight chase: best window of 5 consecutive ranks (Ace low or high).
        let (straight_k, straight_keep) = best_straight_window(hand);

        // Agents only look at the 16 leftmost cards (`hand_eval::MAX_HAND`); a longer hand (Certificate
        // & co.) keeps its extra cards out of the discard plan.
        let all = (1u32 << hand.len().min(16)) - 1;
        // Limit to 5 cards (lowest positions first); `None` if nothing to discard.
        let cap = |m: u32| -> Option<u32> {
            let mut out = 0u32;
            let mut cnt = 0;
            for i in 0..16 {
                if m & (1 << i) != 0 && cnt < 5 {
                    out |= 1 << i;
                    cnt += 1;
                }
            }
            (out != 0).then_some(out)
        };
        if flush_k >= 4 || (flush_k >= 3 && best_type <= HandType::HighCard) {
            return cap(all & !flush_mask);
        }
        if straight_k >= 4 && best_type < HandType::Straight {
            return cap(all & !straight_keep);
        }
        if best_type >= HandType::Straight {
            return None;
        }
        // Keep the scoring cards of the best hand, throw away the rest.
        let sel: Vec<Card> = hand
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < 16 && best_mask & (1 << *i) != 0)
            .map(|(_, c)| *c)
            .collect();
        let eval = hand_eval::evaluate(&sel, rules);
        let mut keep = 0u32;
        let mut j = 0;
        for i in 0..hand.len().min(16) {
            if best_mask & (1 << i) != 0 {
                if eval.scores(j) {
                    keep |= 1 << i;
                }
                j += 1;
            }
        }
        cap(all & !keep)
    }
}

/// `(cards present in the best 5-rank window, mask of one card per present rank in it)`.
fn best_straight_window(hand: &[Card]) -> (usize, u32) {
    // Index 1..=14, Ace counts at 1 and 14.
    let mut pos = [None::<usize>; 15];
    for (i, c) in hand.iter().enumerate().take(16) {
        if c.is_stone() {
            continue;
        }
        let v = c.rank as usize;
        if pos[v].is_none() {
            pos[v] = Some(i);
        }
        if v == 14 && pos[1].is_none() {
            pos[1] = Some(i);
        }
    }
    let mut best = (0usize, 0u32);
    for start in 1..=10usize {
        let mut k = 0;
        let mut mask = 0u32;
        for p in &pos[start..start + 5] {
            if let Some(i) = *p
                && mask & (1 << i) == 0
            {
                k += 1;
                mask |= 1 << i;
            }
        }
        if k > best.0 {
            best = (k, mask);
        }
    }
    best
}

impl GreedyAgent {
    /// Minimal shop policy (the full heuristic team is Phase 5D): buy the most expensive joker
    /// that fits and that the money covers, then Buy & Use affordable Planets, then leave.
    fn shop_action(game: &Game) -> Action {
        let s = game.state();
        let legal = game.legal_actions();
        let mut best_joker: Option<(i32, Action)> = None;
        let mut planet: Option<Action> = None;
        for a in &legal {
            match *a {
                Action::Buy(i) => {
                    if let Some(Some(c)) = s.shop.cards.get(usize::from(i))
                        && matches!(c.kind, ShopCardKind::Joker(_))
                    {
                        let price = shop::card_price(s, c);
                        if price <= s.money && best_joker.is_none_or(|(p, _)| price > p) {
                            best_joker = Some((price, *a));
                        }
                    }
                }
                Action::BuyAndUse(i) => {
                    if let Some(Some(c)) = s.shop.cards.get(usize::from(i))
                        && let ShopCardKind::Consumable(id) = c.kind
                        && id.kind() == ConsumableKind::Planet
                        && shop::card_price(s, c) <= s.money
                        && planet.is_none()
                    {
                        planet = Some(*a);
                    }
                }
                _ => {}
            }
        }
        best_joker
            .map(|(_, a)| a)
            .or(planet)
            .unwrap_or(Action::EndShop)
    }

    /// Take the first joker or Planet of an open pack, otherwise skip it.
    fn pack_action(game: &Game) -> Action {
        let s = game.state();
        let Some(pack) = &s.pack else {
            return Action::SkipPack;
        };
        for a in game.legal_actions() {
            if let Action::PickPackItem { index, .. } = a
                && let Some(Some(item)) = pack.items.get(usize::from(index))
            {
                let good = match item {
                    PackItem::Joker(_) => true,
                    PackItem::Consumable(id) => id.kind() == ConsumableKind::Planet,
                    PackItem::Card(_) => false,
                };
                if good {
                    return a;
                }
            }
        }
        Action::SkipPack
    }
}

impl Agent for GreedyAgent {
    fn act(&mut self, real: &Game) -> Action {
        if self.fair_info && !real.is_terminal() {
            self.fair_counter = self.fair_counter.wrapping_add(1);
            let fair = real.fair_view(self.fair_counter ^ 0x6772_6565_6479_FA17);
            let a = self.act_on(&fair);
            return if real.validate(a).is_ok() {
                a
            } else {
                real.legal_actions()
                    .into_iter()
                    .next()
                    .unwrap_or(Action::EndShop)
            };
        }
        self.act_on(real)
    }
}

impl GreedyAgent {
    fn act_on(&mut self, game: &Game) -> Action {
        let s = game.state();
        match s.phase {
            Phase::BlindSelect => Action::SelectBlind,
            Phase::Play => {
                let Some((mask, score, ht)) = self.best_play(game) else {
                    // No playable hand: take any legal action.
                    return game
                        .legal_actions()
                        .into_iter()
                        .next()
                        .unwrap_or(Action::EndShop);
                };
                let need = s.target - s.round_score;
                if score >= need || s.discards_left == 0 {
                    return Action::Play(mask);
                }
                // A boss may forbid the planned discard (Cerulean Bell's forced card).
                match Self::discard_plan(game, mask, ht) {
                    Some(d) if game.validate(Action::Discard(d)).is_ok() => Action::Discard(d),
                    _ => Action::Play(mask),
                }
            }
            Phase::Shop => Self::shop_action(game),
            Phase::PackOpen => Self::pack_action(game),
            _ => game
                .legal_actions()
                .into_iter()
                .next()
                .unwrap_or(Action::EndShop),
        }
    }
}
