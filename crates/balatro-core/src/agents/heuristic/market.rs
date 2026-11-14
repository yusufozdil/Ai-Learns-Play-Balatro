//! Blind, market and pack experts, and the joker-order search they share.
//!
//! Values are in doublings of the score ("0.5" = the build's hands score 1.41x as much). A
//! purchase is worth `value - price penalty`, the penalty being `lambda * (price + rounds * lost
//! interest dollars)`; `lambda` is higher in `Bank` mode. Joker/planet values come from
//! simulation (`valuer.rs`) plus a static prior (`jokers.rs`).

use super::consumables::{self, Ctx};
use super::jokers::{self, edition_prior, is_copier};
use super::valuer::{self, Valuer};
use super::{EconMode, HeuristicAgent};
use crate::actions::Action;
use crate::blinds::BlindKind;
use crate::card::Card;
use crate::consumable::ConsumableKind;
use crate::content::ids::{BossId, ConsumableId, VoucherId};
use crate::economy;
use crate::enums::{Edition, Enhancement, HandType, Seal};
use crate::game::Game;
use crate::hooks::PackKind;
use crate::joker::JokerInstance;
use crate::packs::{self, PackItem, PackSize, PackSpec};
use crate::rng::fnv1a64;
use crate::shop::{self, ShopCardKind};
use crate::state::GameState;

/// One market evaluation context: a neutral simulation game, shared sample hands and the
/// baseline value, plus cached level-up gains.
struct Eval {
    sim: Game,
    samples: Vec<Vec<Card>>,
    base: f64,
    level_cache: [Option<f64>; HandType::COUNT],
}

/// The scoring cards of a typical hand of type `ht` (empty for the secret hands).
fn canonical_hand(ht: HandType, flush_suit: crate::enums::Suit) -> Vec<Card> {
    use crate::enums::{Rank as R, Suit as S};
    let spec: Vec<(R, S)> = match ht {
        HandType::HighCard => vec![(R::King, S::Spades)],
        HandType::Pair => vec![(R::King, S::Spades), (R::King, S::Hearts)],
        HandType::TwoPair => vec![
            (R::King, S::Spades),
            (R::King, S::Hearts),
            (R::Nine, S::Diamonds),
            (R::Nine, S::Clubs),
        ],
        HandType::ThreeOfAKind => vec![
            (R::King, S::Spades),
            (R::King, S::Hearts),
            (R::King, S::Clubs),
        ],
        HandType::Straight => vec![
            (R::Ten, S::Spades),
            (R::Nine, S::Hearts),
            (R::Eight, S::Diamonds),
            (R::Seven, S::Clubs),
            (R::Six, S::Spades),
        ],
        HandType::Flush => [R::King, R::Jack, R::Eight, R::Six, R::Three]
            .iter()
            .map(|&r| (r, flush_suit))
            .collect(),
        HandType::FullHouse => vec![
            (R::King, S::Spades),
            (R::King, S::Hearts),
            (R::King, S::Clubs),
            (R::Nine, S::Diamonds),
            (R::Nine, S::Spades),
        ],
        HandType::FourOfAKind => vec![
            (R::King, S::Spades),
            (R::King, S::Hearts),
            (R::King, S::Clubs),
            (R::King, S::Diamonds),
        ],
        HandType::StraightFlush => [R::Nine, R::Eight, R::Seven, R::Six, R::Five]
            .iter()
            .map(|&r| (r, flush_suit))
            .collect(),
        _ => Vec::new(),
    };
    spec.into_iter()
        .enumerate()
        .map(|(i, (r, s))| Card::new(90_000 + i as u32, r, s))
        .collect()
}

fn voucher_value(id: VoucherId) -> f64 {
    use VoucherId::*;
    match id {
        Overstock => 0.15,
        OverstockPlus => 0.10,
        ClearanceSale => 0.15,
        Liquidation => 0.10,
        Hone => 0.20,
        GlowUp => 0.15,
        RerollSurplus => 0.15,
        RerollGlut => 0.10,
        CrystalBall => 0.15,
        OmenGlobe => 0.05,
        Telescope => 0.10,
        Observatory => 0.15,
        Grabber => 0.55,
        NachoTong => 0.45,
        Wasteful => 0.40,
        Recyclomancy => 0.30,
        TarotMerchant => 0.10,
        TarotTycoon => 0.05,
        PlanetMerchant => 0.25,
        PlanetTycoon => 0.15,
        SeedMoney => 0.30,
        MoneyTree => 0.25,
        Blank => 0.05,
        Antimatter => 0.75,
        MagicTrick => 0.0,
        Illusion => 0.0,
        Hieroglyph => 0.30,
        Petroglyph => 0.20,
        DirectorsCut => 0.30,
        Retcon => 0.20,
        PaintBrush => 0.50,
        Palette => 0.40,
    }
}

/// Static value of buying a Tarot to use later (needs a free consumable slot).
fn tarot_shop_value(id: ConsumableId, s: &GameState, dominant: crate::enums::Suit) -> f64 {
    use ConsumableId::*;
    match id {
        TheHighPriestess => 0.35,
        TheEmperor => 0.15,
        Judgement => 0.6,
        Temperance => {
            if s.jokers.iter().map(|j| j.sell_value).sum::<i32>() >= 10 {
                0.25
            } else {
                0.0
            }
        }
        TheHermit => {
            if s.money >= 14 {
                0.2
            } else {
                0.0
            }
        }
        TheWheelOfFortune => 0.15,
        TheStar => suit_tarot(dominant, crate::enums::Suit::Diamonds),
        TheMoon => suit_tarot(dominant, crate::enums::Suit::Clubs),
        TheSun => suit_tarot(dominant, crate::enums::Suit::Hearts),
        TheWorld => suit_tarot(dominant, crate::enums::Suit::Spades),
        TheEmpress | TheHierophant => 0.2,
        TheChariot | TheDevil | TheMagician | TheHangedMan => 0.1,
        _ => 0.0,
    }
}

fn suit_tarot(dominant: crate::enums::Suit, s: crate::enums::Suit) -> f64 {
    if dominant == s { 0.2 } else { 0.0 }
}

/// Value of a Standard-pack card for the deck (about 1.0 = worth taking).
fn standard_card_value(c: &Card, dominant: crate::enums::Suit) -> f64 {
    let mut v = 0.0;
    v += match c.enhancement {
        Enhancement::Mult => 1.0,
        Enhancement::Bonus => 0.8,
        Enhancement::Glass => 0.8,
        Enhancement::Lucky => 0.6,
        Enhancement::Steel => 0.6,
        Enhancement::Gold => 0.6,
        Enhancement::Wild => 0.5,
        Enhancement::Stone => -0.5,
        Enhancement::None => 0.0,
    };
    v += match c.seal {
        Seal::Red => 1.0,
        Seal::Gold => 0.8,
        Seal::Blue => 0.5,
        Seal::Purple => 0.4,
        Seal::None => 0.0,
    };
    v += match c.edition {
        Edition::Polychrome => 1.2,
        Edition::Holographic => 1.0,
        Edition::Foil => 0.8,
        _ => 0.0,
    };
    if c.suit == dominant {
        v += 0.3;
    }
    v += (f64::from(c.rank as u8) - 8.0).clamp(-4.0, 6.0) * 0.05;
    v
}

impl HeuristicAgent {
    // ----- shared evaluation ------------------------------------------------------------------

    /// Sample hands cached for one visit (shop/round); `extra` cards beyond the hand size.
    fn cached_samples(
        slot: &mut Option<(u64, Vec<Vec<Card>>)>,
        key: u64,
        game: &Game,
        rng: &mut crate::rng::GameRng,
        n: usize,
        size: usize,
    ) -> Vec<Vec<Card>> {
        if slot.as_ref().is_none_or(|(k, _)| *k != key) {
            *slot = Some((key, Valuer::sample_hands(game, rng, n, size)));
        }
        slot.as_ref().map(|(_, v)| v.clone()).unwrap_or_default()
    }

    fn market_eval(&mut self, game: &Game) -> Eval {
        let s = game.state();
        let key = (u64::from(s.stats.shops_entered) << 32) | u64::from(s.round);
        let size = s.hand_size() as usize + self.cfg.shop_extra_cards;
        let samples = Self::cached_samples(
            &mut self.samples,
            key,
            game,
            &mut self.rng,
            self.cfg.shop_samples,
            size,
        );
        let mut sim = Valuer::sim_game(game);
        let (base, hist) = self.valuer.base_and_histogram(&mut sim, &samples);
        // Target hand weights: what the build would play, blended with what it has played.
        let n: f64 = hist.iter().sum::<f64>().max(1.0);
        let played: f64 = s.hand_plays_run.iter().map(|&x| f64::from(x)).sum::<f64>();
        for h in HandType::ALL {
            let sample_share = hist[h.index()] / n;
            let played_share = if played > 0.0 {
                f64::from(s.hand_plays_run[h.index()]) / played
            } else {
                sample_share
            };
            self.strat.target_hands[h.index()] = 0.6 * sample_share + 0.4 * played_share;
        }
        // Jokers that build around specific hands pull the target toward them.
        for j in &s.jokers {
            let hs = jokers::target_hands(j.id);
            for h in hs {
                self.strat.target_hands[h.index()] += 0.1 / hs.len() as f64;
            }
        }
        // Pressure: the ante's boss requirement over what the build is expected to score in a
        // round (best-hand estimate x hands x a factor for discards/repeat plays).
        let power = base.exp2() * f64::from(s.round_hands().max(1)) * self.cfg.power_factor;
        let need = crate::blinds::requirement(
            s.ante + self.cfg.pressure_horizon,
            BlindKind::Boss,
            2.0,
            s.config.stake.scaling(),
        );
        self.pressure = need / power.max(1.0);
        Eval {
            sim,
            samples,
            base,
            level_cache: [None; HandType::COUNT],
        }
    }

    /// Value of one level of `ht`: how much a canonical hand of that type scores more under the
    /// current jokers, weighted by how much the run plays the type.
    fn level_gain(&mut self, ev: &mut Eval, ht: HandType) -> f64 {
        if let Some(g) = ev.level_cache[ht.index()] {
            return g;
        }
        let cards = canonical_hand(ht, self.strat.dominant);
        let gain = if cards.is_empty() {
            0.0
        } else {
            let mut before = Valuer::sim_game_from(&ev.sim);
            let mut after = valuer::with_levels(&ev.sim, Some(ht), 1);
            let b = self.valuer.score_cards(&mut before, &cards).max(1.0);
            let a = self.valuer.score_cards(&mut after, &cards).max(1.0);
            (a / b).log2() * (self.strat.target_hands[ht.index()] + 0.03) * self.cfg.planet_scale
        };
        ev.level_cache[ht.index()] = Some(gain);
        gain
    }

    /// Mean log2 best of `base_game` with `j` inserted at its best slot, minus `ev.base`.
    fn place_gain(&mut self, ev: &Eval, base_game: &Game, j: &JokerInstance) -> f64 {
        let len = base_game.state().jokers.len();
        let mut positions: Vec<usize> = if is_copier(j.id) {
            (0..=len).collect()
        } else {
            vec![0, len]
        };
        positions.dedup();
        let mut best = f64::MIN;
        for p in positions {
            let mut g = valuer::with_joker(base_game, *j, p);
            best = best.max(self.valuer.mean_log2(&mut g, &ev.samples));
        }
        best - ev.base
    }

    fn add_gain(&mut self, ev: &Eval, j: &JokerInstance) -> f64 {
        self.place_gain(ev, &ev.sim, j) + jokers::prior(j.id) + edition_prior(j.edition)
    }

    fn swap_gain(&mut self, ev: &Eval, j: &JokerInstance, w: usize) -> f64 {
        let old = ev.sim.state().jokers[w];
        let without = valuer::without_joker(&ev.sim, w);
        self.place_gain(ev, &without, j) + jokers::prior(j.id) + edition_prior(j.edition)
            - jokers::prior(old.id)
            - edition_prior(old.edition)
    }

    /// Price penalty of spending `price` now.
    fn penalty(&self, s: &GameState, price: i32) -> f64 {
        let lam = match self.strat.econ {
            EconMode::Build => self.cfg.lambda_build,
            EconMode::Bank => self.cfg.lambda_bank,
        } * (1.0 / (self.pressure * self.pressure).max(1e-6))
            .clamp(self.cfg.min_lambda_scale, 1.0);
        let now = economy::interest(s.money, s.interest_cap, 0);
        let after = economy::interest(s.money - price, s.interest_cap, 0);
        let lost = f64::from((now - after).max(0));
        lam * (f64::from(price.max(0)) + self.cfg.interest_rounds * lost)
    }

    /// Best way to acquire joker `j` for `price`: `(net value, joker to sell first)`.
    fn joker_option(
        &mut self,
        ev: &Eval,
        s: &GameState,
        j: &JokerInstance,
        price: i32,
    ) -> Option<(f64, Option<usize>)> {
        let gain = self.add_gain(ev, j);
        if packs::joker_fits(s, j) {
            return Some((gain - self.penalty(s, price), None));
        }
        if gain < 0.1 {
            return None;
        }
        let mut best: Option<(f64, usize)> = None;
        for (w, old) in s.jokers.iter().enumerate() {
            if old.stickers.eternal
                || (old.edition == Edition::Negative && j.edition != Edition::Negative)
            {
                continue;
            }
            let eff = price - old.sell_value;
            if !s.can_afford(eff) {
                continue;
            }
            let net = self.swap_gain(ev, j, w) - self.penalty(s, eff.max(0));
            if best.is_none_or(|(b, _)| net > b) {
                best = Some((net, w));
            }
        }
        best.map(|(n, w)| (n, Some(w)))
    }

    // ----- joker order ------------------------------------------------------------------------

    /// One `MoveJoker` that improves the simulated score, or `None` once the order is settled.
    pub(super) fn arrange_action(&mut self, game: &Game) -> Option<Action> {
        let s = game.state();
        let n = s.jokers.len();
        let mut bytes = Vec::with_capacity(n * 4);
        for j in &s.jokers {
            bytes.extend_from_slice(&j.uid.to_le_bytes());
        }
        let sig = fnv1a64(&bytes);
        if n < 2 || self.arranged_sig == Some(sig) {
            self.arranged_sig = Some(sig);
            return None;
        }
        let key = (u64::from(s.stats.shops_entered) << 32) | u64::from(s.round);
        if self.arrange_moves.0 != key {
            self.arrange_moves = (key, 0);
        }
        if self.arrange_moves.1 >= 8 {
            self.arranged_sig = Some(sig);
            return None;
        }
        let size = s.hand_size() as usize;
        let samples = Self::cached_samples(
            &mut self.arrange_samples,
            key,
            game,
            &mut self.rng,
            self.cfg.arrange_samples,
            size,
        );
        let mut sim = Valuer::sim_game(game);
        let base = self.valuer.mean_log2(&mut sim, &samples);
        let mut best: Option<(f64, usize, usize)> = None;
        for from in 0..n {
            for to in 0..n {
                if from == to {
                    continue;
                }
                let mut g = valuer::with_move(&sim, from, to);
                let v = self.valuer.mean_log2(&mut g, &samples) - base;
                if best.is_none_or(|(b, _, _)| v > b) {
                    best = Some((v, from, to));
                }
            }
        }
        match best {
            Some((gain, from, to)) if gain > self.cfg.arrange_min_gain => {
                self.arrange_moves.1 += 1;
                Some(Action::MoveJoker {
                    from: from as u8,
                    to: to as u8,
                })
            }
            _ => {
                self.arranged_sig = Some(sig);
                None
            }
        }
    }

    // ----- blind select -----------------------------------------------------------------------

    pub(super) fn blind_action(&mut self, game: &Game) -> Action {
        let s = game.state();
        let step = self.tick(s);
        if step < 12 {
            let ctx = Ctx {
                dominant: self.strat.dominant,
                best_mask: None,
            };
            if let Some(a) = consumables::best_held_action(game, &ctx) {
                return a;
            }
            if let Some(a) = self.arrange_action(game) {
                return a;
            }
        }
        if s.blind == BlindKind::Boss
            && matches!(
                s.boss,
                Some(BossId::TheWall | BossId::TheNeedle | BossId::VioletVessel | BossId::TheWater)
            )
            && step < 12
            && s.money >= 10 + self.cfg.boss_reroll_reserve
            && game.validate(Action::RerollBoss).is_ok()
        {
            return Action::RerollBoss;
        }
        if s.blind != BlindKind::Boss {
            let idx = usize::from(s.blind == BlindKind::Big);
            let v = jokers::tag_value(
                s.blind_tags[idx],
                s.money,
                s.stats.hands_played,
                s.stats.run.blinds_skipped,
                s.has_joker_room(),
            );
            if v >= self.cfg.skip_threshold {
                return Action::SkipBlind;
            }
        }
        Action::SelectBlind
    }

    // ----- shop -------------------------------------------------------------------------------

    fn pack_value(&mut self, ev: &mut Eval, s: &GameState, spec: PackSpec) -> f64 {
        let m = match spec.size {
            PackSize::Normal => 1.0,
            PackSize::Jumbo => 1.35,
            PackSize::Mega => 1.9,
        };
        match spec.kind {
            PackKind::Buffoon => 0.55 * m * if s.has_joker_room() { 1.0 } else { 0.35 },
            PackKind::Celestial => {
                // Average gain of the three hands the build leans on most.
                let mut order: Vec<HandType> = HandType::ALL
                    .into_iter()
                    .filter(|h| !h.is_secret())
                    .collect();
                order.sort_by(|a, b| {
                    self.strat.target_hands[b.index()]
                        .partial_cmp(&self.strat.target_hands[a.index()])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<HandType> = order.into_iter().take(3).collect();
                let sum: f64 = top.iter().map(|&h| self.level_gain(ev, h)).sum();
                (sum / top.len().max(1) as f64).max(0.0) * 1.3 * m
            }
            PackKind::Arcana => 0.15 * m,
            PackKind::Spectral => 0.15 * m,
            PackKind::Standard => 0.05 * m,
        }
    }

    pub(super) fn shop_action(&mut self, game: &Game) -> Action {
        let s = game.state();
        let step = self.tick(s);
        if step > 60 {
            return Action::EndShop;
        }
        if step < 30 {
            let ctx = Ctx {
                dominant: self.strat.dominant,
                best_mask: None,
            };
            if let Some(a) = consumables::best_held_action(game, &ctx) {
                return a;
            }
        }
        let mut ev = self.market_eval(game);
        // (net value, action to take now)
        let mut best: Option<(f64, Action)> = None;
        let thr = self.cfg.buy_threshold;
        let consider = |net: f64, a: Action, best: &mut Option<(f64, Action)>| {
            if net > thr && best.is_none_or(|(b, _)| net > b) {
                *best = Some((net, a));
            }
        };
        let cards: Vec<(usize, crate::shop::ShopCard)> = s
            .shop
            .cards
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
            .collect();
        for (i, c) in cards {
            let Ok(slot) = u8::try_from(i) else { continue };
            let price = shop::card_price(s, &c);
            match c.kind {
                ShopCardKind::Joker(j) => {
                    if let Some((net, sell)) = self.joker_option(&ev, s, &j, price) {
                        match sell {
                            None => {
                                if game.validate(Action::Buy(slot)).is_ok() {
                                    consider(net, Action::Buy(slot), &mut best);
                                }
                            }
                            Some(w) => {
                                if let Ok(w) = u8::try_from(w) {
                                    consider(net, Action::SellJoker(w), &mut best);
                                }
                            }
                        }
                    }
                }
                ShopCardKind::Consumable(id) => match id.kind() {
                    ConsumableKind::Planet => {
                        let Some(ht) = id.planet_hand() else { continue };
                        let net = self.level_gain(&mut ev, ht) - self.penalty(s, price);
                        let a = if game.validate(Action::BuyAndUse(slot)).is_ok() {
                            Action::BuyAndUse(slot)
                        } else {
                            Action::Buy(slot)
                        };
                        if game.validate(a).is_ok() {
                            consider(net, a, &mut best);
                        }
                    }
                    ConsumableKind::Tarot => {
                        let v = tarot_shop_value(id, s, self.strat.dominant);
                        let net = v - self.penalty(s, price);
                        if game.validate(Action::Buy(slot)).is_ok() {
                            consider(net, Action::Buy(slot), &mut best);
                        }
                    }
                    ConsumableKind::Spectral => {}
                },
                ShopCardKind::PlayingCard(_) => {}
            }
        }
        let packs_v: Vec<(usize, crate::shop::ShopPack)> = s
            .shop
            .packs
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (i, p)))
            .collect();
        for (i, p) in packs_v {
            let Ok(slot) = u8::try_from(i) else { continue };
            let price = shop::pack_price_of(s, &p);
            let v = self.pack_value(&mut ev, s, p.spec);
            let net = v - self.penalty(s, price);
            if game.validate(Action::OpenPack(slot)).is_ok() {
                consider(net, Action::OpenPack(slot), &mut best);
            }
        }
        for (i, v) in s.shop.vouchers.iter().enumerate() {
            let (Some(v), Ok(slot)) = (v, u8::try_from(i)) else {
                continue;
            };
            let price = shop::voucher_price(s, v.id);
            let net = voucher_value(v.id) - self.penalty(s, price);
            if game.validate(Action::BuyVoucher(slot)).is_ok() {
                consider(net, Action::BuyVoucher(slot), &mut best);
            }
        }
        if let Some((_, a)) = best {
            return a;
        }

        // Reroll with surplus money (or when it is free).
        let price = shop::reroll_price(s);
        let rerolls = s.shop.rerolls + s.shop.free_rerolls_used;
        let floor = if self.pressure > 1.0 {
            0
        } else if s.has_joker_room() {
            self.cfg.reroll_floor_room
        } else {
            self.cfg.reroll_floor_full.min(s.interest_cap * 5)
        };
        if rerolls < self.cfg.max_rerolls
            && (price == 0 || s.money - price >= floor)
            && game.validate(Action::Reroll).is_ok()
        {
            return Action::Reroll;
        }
        if let Some(a) = self.arrange_action(game) {
            return a;
        }
        Action::EndShop
    }

    // ----- packs ------------------------------------------------------------------------------

    pub(super) fn pack_action(&mut self, game: &Game) -> Action {
        let s = game.state();
        let step = self.tick(s);
        let Some(pack) = &s.pack else {
            return Action::SkipPack;
        };
        if step > 14 {
            return Action::SkipPack;
        }
        let items: Vec<(u8, PackItem)> = pack
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, it)| Some((u8::try_from(i).ok()?, (*it)?)))
            .collect();
        match pack.spec.kind {
            PackKind::Buffoon => {
                let ev = self.market_eval(game);
                let mut best: Option<(f64, Action)> = None;
                for (index, it) in items {
                    let PackItem::Joker(j) = it else { continue };
                    let gain = self.add_gain(&ev, &j);
                    if packs::joker_fits(s, &j) {
                        if gain > self.cfg.pack_joker_min && best.is_none_or(|(b, _)| gain > b) {
                            best = Some((gain, Action::PickPackItem { index, targets: 0 }));
                        }
                    } else if gain > self.cfg.pack_joker_min + 0.1 {
                        for (w, old) in s.jokers.iter().enumerate() {
                            if old.stickers.eternal
                                || (old.edition == Edition::Negative
                                    && j.edition != Edition::Negative)
                            {
                                continue;
                            }
                            let v = self.swap_gain(&ev, &j, w) - 0.1;
                            if v > self.cfg.pack_joker_min
                                && best.is_none_or(|(b, _)| v > b)
                                && let Ok(w) = u8::try_from(w)
                            {
                                best = Some((v, Action::SellJoker(w)));
                            }
                        }
                    }
                }
                best.map_or(Action::SkipPack, |(_, a)| a)
            }
            PackKind::Celestial => {
                let mut ev = self.market_eval(game);
                let mut best: Option<(f64, u8)> = None;
                for (index, it) in items {
                    let PackItem::Consumable(id) = it else {
                        continue;
                    };
                    let v = match id {
                        ConsumableId::BlackHole => 2.0,
                        _ => match id.planet_hand() {
                            Some(h) => self.level_gain(&mut ev, h),
                            None => 0.0,
                        },
                    };
                    if best.is_none_or(|(b, _)| v > b) {
                        best = Some((v, index));
                    }
                }
                best.map_or(Action::SkipPack, |(_, index)| Action::PickPackItem {
                    index,
                    targets: 0,
                })
            }
            PackKind::Arcana | PackKind::Spectral => {
                let ctx = Ctx {
                    dominant: self.strat.dominant,
                    best_mask: None,
                };
                let mut best: Option<(f64, Action)> = None;
                for (index, it) in items {
                    let PackItem::Consumable(id) = it else {
                        continue;
                    };
                    let Some((targets, v)) = consumables::plan(id, game, &ctx) else {
                        continue;
                    };
                    let a = Action::PickPackItem { index, targets };
                    if game.validate(a).is_ok() && best.is_none_or(|(b, _)| v > b) {
                        best = Some((v, a));
                    }
                }
                best.map_or(Action::SkipPack, |(_, a)| a)
            }
            PackKind::Standard => {
                let mut best: Option<(f64, u8)> = None;
                for (index, it) in items {
                    let PackItem::Card(c) = it else { continue };
                    let v = standard_card_value(&c, self.strat.dominant);
                    if v >= 0.9 && best.is_none_or(|(b, _)| v > b) {
                        best = Some((v, index));
                    }
                }
                best.map_or(Action::SkipPack, |(_, index)| Action::PickPackItem {
                    index,
                    targets: 0,
                })
            }
        }
    }
}
