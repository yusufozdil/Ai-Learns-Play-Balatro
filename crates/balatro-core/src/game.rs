//! `Game`: the phase state machine over a [`GameState`] (prd.md §3.1).
//!
//! ```text
//! BlindSelect --SelectBlind--> Play --(score >= target)--> [round end + cash out] --> Shop
//!     |  SkipBlind (Small/Big): tag, next blind     | (no hands left / no cards, score < target)
//!     v                                             v
//!   BlindSelect                                  GameOver
//! Shop --EndShop--> BlindSelect          Ante-8 boss beaten --> Won (unless endless)
//! ```
//!
//! `step` validates first ([`actions::validate`]) and only then mutates, so a rejected action
//! leaves the state untouched. Joker events are fired at the moments listed in
//! `docs/design/hooks.md` §15; their [`SideEffect`]s are applied by [`apply_side_effects`].

use crate::actions::{self, Action, ActionError};
use crate::blinds::BlindKind;
use crate::boss::{self, DrawKind};
use crate::card::Card;
use crate::consumable::{self, ConsumableInstance, ConsumableKind, UseCtx};
use crate::content::ids::ConsumableId;
use crate::content::try_consumable_def;
use crate::data::run_rules::{MAX_EFFECT_CASCADE, WIN_ANTE};
use crate::deck::Pile;
use crate::economy::{self, CashOut, CashOutInput};
use crate::enums::{Edition, HandType, Phase, Seal};
use crate::hand_eval;
use crate::hooks::{
    self, DiscardInfo, EventInput, EventResult, RoundEndInfo, SellTarget, SideEffect, SoldItem,
};
use crate::joker::JokerInstance;
use crate::rng::RngKey;
use crate::scoring::{ScoreInput, Scorer};
use crate::state::{GameConfig, GameState, Outcome, ShopState, TagId, transition_allowed};

/// Result of one played hand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandReport {
    /// Hand type.
    pub hand_type: HandType,
    /// Level used for the base values.
    pub level: u32,
    /// Final chips.
    pub chips: f64,
    /// Final mult.
    pub mult: f64,
    /// Hand score (`floor(chips * mult)`).
    pub score: f64,
    /// The boss rejected the hand (score 0).
    pub invalid: bool,
}

/// What happened when a round ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundReport {
    /// The blind counts as beaten (by score, or saved by Mr. Bones).
    pub won: bool,
    /// Saved by `SideEffect::PreventDeath` (no blind reward).
    pub saved: bool,
    /// Lost because no cards were left to play.
    pub out_of_cards: bool,
    /// Ante of the round.
    pub ante: i32,
    /// Blind of the round.
    pub blind: BlindKind,
    /// Chips scored.
    pub round_score: f64,
    /// Chips required.
    pub target: f64,
    /// Money from cards held at round end (Gold cards; already in money before the cash-out,
    /// so it counts for interest).
    pub held_money: i32,
    /// Itemised cash-out (`None` when the run was lost).
    pub cash_out: Option<CashOut>,
}

/// Summary of one successful `step`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepInfo {
    /// The action taken.
    pub action: Action,
    /// Phase before the action.
    pub phase_before: Phase,
    /// Phase after the action.
    pub phase_after: Phase,
    /// Money before the action.
    pub money_before: i32,
    /// Money after the action.
    pub money_after: i32,
    /// Set by `Play`.
    pub hand: Option<HandReport>,
    /// Set when a round ended during this step.
    pub round: Option<RoundReport>,
    /// Tag gained by `SkipBlind`.
    pub tag: Option<TagId>,
    /// Price of a sold joker or consumable.
    pub sold_for: Option<i32>,
    /// Consumable used by `UseConsumable`.
    pub used: Option<ConsumableId>,
    /// Playing cards destroyed during the step.
    pub cards_destroyed: u32,
    /// Playing cards added during the step.
    pub cards_added: u32,
}

/// Outcome of applying a side-effect queue.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EffectReport {
    /// Playing cards destroyed.
    pub destroyed: u32,
    /// Playing cards added.
    pub added: u32,
    /// A `PreventDeath` was applied.
    pub prevent_death: bool,
    /// Effects dropped (no room, not implemented yet, eternal target, cascade cap).
    pub dropped: u32,
}

/// One run: the state plus reusable scratch buffers.
#[derive(Debug)]
pub struct Game {
    state: GameState,
    scorer: Scorer,
    fx: Vec<SideEffect>,
}

impl Clone for Game {
    /// Clones the state; scratch buffers start empty.
    fn clone(&self) -> Self {
        Game::from_state(self.state.clone())
    }
}

impl Game {
    /// New run from a seed and options.
    pub fn new(seed: u64, config: GameConfig) -> Self {
        Game::from_state(GameState::new(seed, config))
    }

    /// Resume from a state (rollouts, replays, tests).
    pub fn from_state(state: GameState) -> Self {
        Game {
            state,
            scorer: Scorer::new(),
            fx: Vec::new(),
        }
    }

    /// Read access to the state.
    pub fn state(&self) -> &GameState {
        &self.state
    }

    /// Write access to the state, for tests and tools that set up positions (e.g. jump to the
    /// ante-8 boss). Keep it consistent: `step` checks invariants in debug builds.
    pub fn state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }

    /// Consume the game, returning the state.
    pub fn into_state(self) -> GameState {
        self.state
    }

    /// The run is over.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// How the run ended (`None` while running).
    pub fn outcome(&self) -> Option<Outcome> {
        self.state.outcome
    }

    /// All legal actions in the current state.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut v = Vec::new();
        actions::legal_actions(&self.state, &mut v);
        v
    }

    /// [`Game::legal_actions`] into a reusable buffer (cleared first).
    pub fn legal_actions_into(&self, out: &mut Vec<Action>) {
        actions::legal_actions(&self.state, out);
    }

    /// Check an action without applying it.
    pub fn validate(&self, action: Action) -> Result<(), ActionError> {
        actions::validate(&self.state, action)
    }

    /// Score `Play(mask)` without changing the game: the result `step` would report for the
    /// same state (jokers and the RNG are cloned, so probabilistic effects resolve exactly as
    /// the real play would). Errors like `validate(Action::Play(mask))`.
    pub fn preview_play(&self, mask: u16) -> Result<HandReport, ActionError> {
        self.preview_play_with(&mut Scorer::new(), mask)
    }

    /// [`Game::preview_play`] with a caller-owned [`Scorer`] (reuse it across candidates to
    /// avoid per-call allocations).
    pub fn preview_play_with(
        &self,
        scorer: &mut Scorer,
        mask: u16,
    ) -> Result<HandReport, ActionError> {
        actions::validate(&self.state, Action::Play(mask))?;
        let s = &self.state;
        let n = s.deck.hand.len();
        let mut played: Vec<Card> = Vec::with_capacity(5);
        let mut held: Vec<Card> = Vec::with_capacity(n);
        for (i, c) in s.deck.hand.iter().enumerate() {
            if i < 16 && (mask >> i) & 1 == 1 {
                played.push(*c);
            } else {
                held.push(*c);
            }
        }
        let eval = hand_eval::evaluate(&played, &s.mods.rules);
        let ht = eval.hand_type;
        // Same counters `play` has paid by the time scoring runs.
        let mut view = s.run_view_with(&s.consumables);
        view.hands_left = view.hands_left.saturating_sub(1);
        view.hands_played_round += 1;
        view.hands_played_run += 1;
        view.hand_plays_run[ht.index()] += 1;
        view.hand_plays_round[ht.index()] += 1;
        let mut jokers = s.jokers.clone();
        let mut rng = s.rng.clone();
        // Boss pre-play effects on copies, exactly as `play` applies them to the state.
        if s.boss_in_effect() {
            let mut bs = s.boss_state;
            let mut hooked = Vec::new();
            boss::pre_play(
                s.boss,
                &played,
                ht,
                &mut held,
                &mut hooked,
                &mut view.money,
                &mut view.hand_levels,
                &view.hand_plays_round,
                &mut bs,
                &mut rng,
            );
            view.boss.triggered = bs.triggered;
            view.boss.hand_invalid = bs.hand_invalid;
            if !hooked.is_empty() {
                let d = DiscardInfo {
                    cards: &hooked,
                    hand_type: hand_eval::evaluate(&hooked, &s.mods.rules).hand_type,
                    forced: true,
                };
                let mut fx = Vec::new();
                let r = hooks::fire_discard(
                    EventInput {
                        jokers: &mut jokers,
                        run: &view,
                        deck: &s.deck,
                        rng: &mut rng,
                        effects: &mut fx,
                    },
                    &d,
                );
                view.money = view.money.saturating_add(r.money_delta);
                // Jokers the discard destroyed (Ramen) are gone before scoring in `play`.
                jokers.retain(|j| {
                    j.stickers.eternal
                        || !fx
                            .iter()
                            .any(|e| matches!(e, SideEffect::DestroyJoker { uid } if *uid == j.uid))
                });
            }
        }
        let r = scorer.score_hand(ScoreInput {
            played: &mut played,
            held: &held,
            eval,
            jokers: &mut jokers,
            run: &view,
            rng: &mut rng,
        });
        Ok(HandReport {
            hand_type: r.hand_type,
            level: r.hand_level,
            chips: r.chips,
            mult: r.mult,
            score: r.score,
            invalid: r.invalid,
        })
    }

    /// Apply one action. On `Err` the state is unchanged.
    pub fn step(&mut self, action: Action) -> Result<StepInfo, ActionError> {
        actions::validate(&self.state, action)?;
        let s = &self.state;
        let (destroyed0, added0) = (s.stats.cards_destroyed, s.stats.cards_added);
        let mut info = StepInfo {
            action,
            phase_before: s.phase,
            phase_after: s.phase,
            money_before: s.money,
            money_after: s.money,
            hand: None,
            round: None,
            tag: None,
            sold_for: None,
            used: None,
            cards_destroyed: 0,
            cards_added: 0,
        };
        match action {
            Action::SelectBlind => self.select_blind(&mut info),
            Action::SkipBlind => self.skip_blind(&mut info),
            Action::Play(mask) => self.play(mask, &mut info)?,
            Action::Discard(mask) => self.discard(mask, &mut info)?,
            Action::SellJoker(i) => self.sell_joker(usize::from(i), &mut info),
            Action::UseConsumable { slot, targets } => {
                self.use_consumable(usize::from(slot), targets, &mut info)
            }
            Action::SellConsumable(i) => self.sell_consumable(usize::from(i), &mut info),
            Action::EndShop => self.end_shop(),
            // `validate` rejects these; nothing has been mutated yet.
            other => {
                return Err(ActionError::NotImplemented {
                    action: other,
                    phase_of_plan: 5,
                });
            }
        }
        if self.state.phase == Phase::Play {
            // Debuffs follow suit/rank/joker changes made by the action (hooks.md §15).
            boss::refresh_debuffs(&mut self.state);
        }
        let s = &self.state;
        info.phase_after = s.phase;
        info.money_after = s.money;
        info.cards_destroyed = s.stats.cards_destroyed - destroyed0;
        info.cards_added = s.stats.cards_added - added0;
        debug_assert!(
            transition_allowed(info.phase_before, info.phase_after),
            "illegal phase transition {:?} -> {:?}",
            info.phase_before,
            info.phase_after
        );
        #[cfg(debug_assertions)]
        if let Err(e) = s.check_invariants() {
            panic!("invariant violated after {action:?}: {e}");
        }
        Ok(info)
    }

    /// Add a joker (shop purchase, pack, tests). Its uid is reassigned; `on_acquire` fires.
    /// Returns the slot, or `None` if there is no room (Negative jokers always fit).
    pub fn add_joker(&mut self, joker: JokerInstance) -> Option<usize> {
        let s = &mut self.state;
        if joker.edition != Edition::Negative && !s.has_joker_room() {
            return None;
        }
        let mut j = joker;
        j.uid = s.alloc_joker_uid();
        s.jokers.push(j);
        s.refresh_modifiers();
        let slot = s.jokers.len() - 1;
        fire_paid(s, &mut self.fx, |inp| hooks::fire_acquire(inp, slot));
        self.resolve_fx();
        if self.state.phase == Phase::Play {
            boss::refresh_debuffs(&mut self.state);
        }
        Some(slot)
    }

    /// Add a consumable (packs/shop in Phase 5, tests). Returns its slot, or `None` without room
    /// (Negative ones always fit).
    pub fn add_consumable(&mut self, c: ConsumableInstance) -> Option<usize> {
        let s = &mut self.state;
        if !c.negative && !s.has_consumable_room() {
            return None;
        }
        s.consumables.push(c);
        Some(s.consumables.len() - 1)
    }

    // ----- actions --------------------------------------------------------------------------

    fn select_blind(&mut self, info: &mut StepInfo) {
        let s = &mut self.state;
        s.round += 1;
        s.round_score = 0.0;
        s.hands_played_round = 0;
        s.discards_used_round = 0;
        s.hand_plays_round = [0; HandType::COUNT];
        s.hands_left = s.round_hands();
        s.discards_left = s.round_discards();
        s.refresh_target();
        s.deck.reset_for_round(&mut s.rng);
        // The round is on from here (`in_round`), so the boss is in effect for its set-up and
        // for joker `on_blind_select` (Chicot's disable reverts the set-up).
        s.phase = Phase::Play;
        boss::blind_start(s);
        fire_paid(s, &mut self.fx, hooks::fire_blind_select);
        self.resolve_fx();
        let s = &mut self.state;
        boss::refresh_debuffs(s);
        boss::draw(s, DrawKind::RoundStart);
        fire_paid(s, &mut self.fx, hooks::fire_first_hand_drawn);
        self.resolve_fx();
        self.settle(info, DrawKind::Refill);
    }

    fn skip_blind(&mut self, info: &mut StepInfo) {
        let s = &mut self.state;
        let idx = match s.blind {
            BlindKind::Small => 0,
            BlindKind::Big => 1,
            BlindKind::Boss => return, // rejected by `validate`
        };
        let tag = s.blind_tags[idx];
        s.tags.push(tag);
        info.tag = Some(tag);
        // TODO(Phase 5C): tag effects (Economy, Speed, Handy, Garbage, Orbital, Double, pack
        // tags -> PackOpen, next-shop tags).
        s.stats.run.blinds_skipped += 1;
        fire_paid(s, &mut self.fx, hooks::fire_blind_skip);
        self.resolve_fx();
        advance_blind(&mut self.state);
    }

    fn play(&mut self, mask: u16, info: &mut StepInfo) -> Result<(), ActionError> {
        let s = &mut self.state;
        let hand_len = s.deck.hand.len();
        // First mutation; the hand is unchanged if it fails (it cannot after `validate`).
        let played = s
            .deck
            .take_from_hand(mask)
            .map_err(|_| ActionError::SelectionOutOfRange { mask, hand_len })?;
        s.deck.played = played;
        for c in &mut s.deck.played {
            c.played_ante = true; // The Pillar
        }
        // Pay the hand (RunView conventions, hooks.md §5.3).
        s.hands_left -= 1;
        s.hands_played_round += 1;
        s.stats.hands_played += 1;
        let eval = hand_eval::evaluate(&s.deck.played, &s.mods.rules);
        let ht = eval.hand_type;
        s.hand_plays_run[ht.index()] += 1;
        s.hand_plays_round[ht.index()] += 1;
        s.last_hand = Some(ht);

        // Boss pre-play effects (validity, The Hook, Tooth, Ox, Arm; `Game::preview_play`
        // mirrors this on copies).
        let boss_id = if s.boss_in_effect() { s.boss } else { None };
        let mut hooked = Vec::new();
        boss::pre_play(
            boss_id,
            &s.deck.played,
            ht,
            &mut s.deck.hand,
            &mut hooked,
            &mut s.money,
            &mut s.hand_levels,
            &s.hand_plays_round,
            &mut s.boss_state,
            &mut s.rng,
        );
        if !hooked.is_empty() {
            self.forced_discard(hooked);
        }

        let s = &mut self.state;
        let view = s.run_view_with(&s.consumables);
        let r = self.scorer.score_hand(ScoreInput {
            played: &mut s.deck.played,
            held: &s.deck.hand,
            eval,
            jokers: &mut s.jokers,
            run: &view,
            rng: &mut s.rng,
        });
        s.money = s.money.saturating_add(r.money_delta);
        s.round_score += r.score;
        if r.score > s.stats.best_hand {
            s.stats.best_hand = r.score;
        }
        info.hand = Some(HandReport {
            hand_type: r.hand_type,
            level: r.hand_level,
            chips: r.chips,
            mult: r.mult,
            score: r.score,
            invalid: r.invalid,
        });
        s.refresh_modifiers();
        self.fx.extend_from_slice(self.scorer.effects());
        self.resolve_fx();
        let s = &mut self.state;
        s.deck.move_played_to_discard();
        if s.round_score < s.target && s.hands_left > 0 {
            boss::after_play(s); // e.g. Crimson Heart picks the joker for the next hand
        }
        self.settle(info, DrawKind::AfterPlay);
        Ok(())
    }

    /// Cards discarded by a boss before scoring (The Hook): to the discard pile, then a forced
    /// discard event (no discard is used) and Purple Seals.
    fn forced_discard(&mut self, cards: Vec<Card>) {
        let s = &mut self.state;
        s.deck.discard_pile.extend_from_slice(&cards);
        let d = DiscardInfo {
            cards: &cards,
            hand_type: hand_eval::evaluate(&cards, &s.mods.rules).hand_type,
            forced: true,
        };
        fire_paid(s, &mut self.fx, |inp| hooks::fire_discard(inp, &d));
        push_purple_seals(&cards, &mut self.fx);
        self.resolve_fx();
    }

    fn discard(&mut self, mask: u16, info: &mut StepInfo) -> Result<(), ActionError> {
        let s = &mut self.state;
        let hand_len = s.deck.hand.len();
        let cards = s
            .deck
            .discard_from_hand(mask)
            .map_err(|_| ActionError::SelectionOutOfRange { mask, hand_len })?;
        s.discards_left -= 1;
        s.discards_used_round += 1;
        s.stats.discards_used += 1;
        let hand_type = hand_eval::evaluate(&cards, &s.mods.rules).hand_type;
        let d = DiscardInfo {
            cards: &cards,
            hand_type,
            forced: false,
        };
        fire_paid(s, &mut self.fx, |inp| hooks::fire_discard(inp, &d));
        push_purple_seals(&cards, &mut self.fx);
        self.resolve_fx();
        self.settle(info, DrawKind::AfterDiscard);
        Ok(())
    }

    fn sell_joker(&mut self, slot: usize, info: &mut StepInfo) {
        let s = &mut self.state;
        let Some(&sold) = s.jokers.get(slot) else {
            return; // rejected by `validate`
        };
        let price = sold.sell_value;
        // Hooks run before the joker is removed (hooks.md §15).
        fire_paid(s, &mut self.fx, |inp| hooks::fire_sell_self(inp, slot));
        fire_paid(s, &mut self.fx, |inp| {
            hooks::fire_sell_other(inp, &SoldItem::Joker(sold), Some(slot))
        });
        if let Some(pos) = s.jokers.iter().position(|j| j.uid == sold.uid) {
            s.jokers.remove(pos);
        }
        s.money = s.money.saturating_add(price);
        s.stats.jokers_sold += 1;
        s.refresh_modifiers();
        info.sold_for = Some(price);
        self.resolve_fx();
        boss::joker_sold(&mut self.state); // Verdant Leaf
        // A sale cannot normally end a round, but effects (e.g. AddHands(-n)) could.
        if self.state.phase == Phase::Play {
            self.settle_without_draw(info);
        }
    }

    /// Use a consumable (validated): out of its slot, effect (immediate writes + queued side
    /// effects, applied with their follow-up events), run counters and The Fool's record, then
    /// `on_consumable_used` (hooks see the counters including this use).
    fn use_consumable(&mut self, slot: usize, targets: u16, info: &mut StepInfo) {
        let s = &mut self.state;
        if slot >= s.consumables.len() {
            return; // rejected by `validate`
        }
        let used = s.consumables.remove(slot);
        let Some(def) = try_consumable_def(used.id) else {
            return; // rejected by `validate`
        };
        let mut ctx = UseCtx::new(s, &mut self.fx, used, targets);
        (def.effect)(&mut ctx);
        s.refresh_modifiers(); // joker editions may have changed
        self.resolve_fx();
        let s = &mut self.state;
        record_use(s, used.id);
        info.used = Some(used.id);
        fire_paid(s, &mut self.fx, |inp| {
            hooks::fire_consumable_used(inp, &used)
        });
        self.resolve_fx();
        let s = &mut self.state;
        if s.phase == Phase::Play {
            s.deck.sort_hand(); // ranks may have changed (Strength, Death, Ouija)
            if s.deck.hand.is_empty() {
                self.settle(info, DrawKind::Refill);
            } else {
                self.settle_without_draw(info);
            }
        }
    }

    fn sell_consumable(&mut self, slot: usize, info: &mut StepInfo) {
        let s = &mut self.state;
        let Some(&sold) = s.consumables.get(slot) else {
            return; // rejected by `validate`
        };
        fire_paid(s, &mut self.fx, |inp| {
            hooks::fire_sell_other(inp, &SoldItem::Consumable(sold), None)
        });
        // Event effects are only queued, so `slot` still names the sold card.
        s.consumables.remove(slot);
        s.money = s.money.saturating_add(sold.sell_value);
        info.sold_for = Some(sold.sell_value);
        self.resolve_fx();
        if self.state.phase == Phase::Play {
            self.settle_without_draw(info);
        }
    }

    fn end_shop(&mut self) {
        let s = &mut self.state;
        fire_paid(s, &mut self.fx, hooks::fire_shop_exit);
        self.resolve_fx();
        let s = &mut self.state;
        s.shop = ShopState::default();
        s.phase = Phase::BlindSelect;
    }

    // ----- round flow -----------------------------------------------------------------------

    /// After a Play phase action: end the round if it is decided, else refill the hand (boss
    /// draw rules for `kind`) and end it if nothing is left to play (blinds.md §6.4-§6.6).
    fn settle(&mut self, info: &mut StepInfo, kind: DrawKind) {
        if self.settle_without_draw(info) {
            return;
        }
        let s = &mut self.state;
        boss::draw(s, kind);
        if s.deck.hand.is_empty() {
            self.finish_round(false, true, info);
        }
    }

    /// End the round if the score is reached or no hand is left. Returns true if it ended.
    fn settle_without_draw(&mut self, info: &mut StepInfo) -> bool {
        let s = &self.state;
        if s.round_score >= s.target {
            self.finish_round(true, false, info);
            true
        } else if s.hands_left == 0 {
            self.finish_round(false, false, info);
            true
        } else if s.deck.hand.is_empty() && s.deck.draw_pile.is_empty() {
            self.finish_round(false, true, info);
            true
        } else {
            false
        }
    }

    /// Round end: held-card effects, joker end-of-round event, then cash-out and Shop, or game
    /// over (unless `PreventDeath` saves a lost round; never when out of cards, Mr. Bones wiki).
    fn finish_round(&mut self, won: bool, out_of_cards: bool, info: &mut StepInfo) {
        let s = &mut self.state;
        let last = s.last_hand.unwrap_or(HandType::HighCard);
        // 1. Cards held at round end: Gold card money, Blue Seal planets (Red Seal / Mime
        //    retrigger them). Paid now, so it counts for interest (notes.md §4.1c).
        let view = s.run_view_with(&s.consumables);
        let held_money =
            self.scorer
                .round_end_held(&s.deck.hand, &mut s.jokers, &view, &mut s.rng, last);
        s.money = s.money.saturating_add(held_money);
        self.fx.extend_from_slice(self.scorer.effects());
        self.resolve_fx();

        // 2. Jokers' end of round (also on a loss: Mr. Bones). Their money is a cash-out item.
        let s = &mut self.state;
        let mut end = RoundEndInfo {
            won,
            saved: false,
            round_score: s.round_score,
            target: s.target,
        };
        // A lost round may be saved by a joker (Mr. Bones). Hooks at this event (Campfire,
        // Rocket) must see the outcome, so probe it on a copy of the jokers and the RNG first.
        if !won && !out_of_cards {
            end.saved = round_end_saves(s, &end);
        }
        let joker_money = fire(s, &mut self.fx, |inp| hooks::fire_round_end(inp, &end));
        let rep = self.resolve_fx();
        let saved = !won && !out_of_cards && rep.prevent_death;
        debug_assert_eq!(saved, end.saved);

        let s = &mut self.state;
        // The boss's joker debuff lasted through the round end (Crimson Heart, bosses.md).
        boss::round_end(s);
        let mut report = RoundReport {
            won: won || saved,
            saved,
            out_of_cards,
            ante: s.ante,
            blind: s.blind,
            round_score: s.round_score,
            target: s.target,
            held_money,
            cash_out: None,
        };
        if !(won || saved) {
            s.money = s.money.saturating_add(joker_money);
            s.phase = Phase::GameOver;
            s.outcome = Some(Outcome::Loss);
            info.round = Some(report);
            return;
        }

        // 3. Cash out (economy::cash_out; Gold cards were already paid in step 1).
        let co = economy::cash_out(&CashOutInput {
            money: s.money,
            gold_cards_held: 0,
            gold_red_seal_cards_held: 0,
            kind: s.blind,
            is_showdown: s.is_showdown(),
            stake_no_small_reward: s.config.stake.no_small_reward(),
            no_blind_reward: saved,
            hands_left: i32::try_from(s.hands_left).unwrap_or(i32::MAX),
            interest_cap: s.interest_cap,
            interest_per_5_bonus: s.mods.interest_per_5,
            no_interest: false,
            joker_payout: joker_money,
        });
        s.money = co.final_money;
        report.cash_out = Some(co);
        info.round = Some(report);
        s.stats.rounds_won += 1;
        if s.blind == BlindKind::Boss {
            s.stats.run.bosses_defeated += 1;
        }
        gather_deck(s);

        // 4. Win, or advance to the next blind and open the shop.
        if s.blind == BlindKind::Boss && s.ante >= WIN_ANTE && !s.won {
            s.won = true;
            if !s.config.endless {
                s.phase = Phase::Won;
                s.outcome = Some(Outcome::Win);
                return;
            }
        }
        advance_blind(s);
        s.shop = ShopState::default();
        s.phase = Phase::Shop;
        // TODO(Phase 5A): generate the shop here (RngKey::Shop).
        fire_paid(s, &mut self.fx, hooks::fire_shop_enter);
        self.resolve_fx();
    }

    /// Apply the queued effects in `self.fx` (and their follow-up events).
    fn resolve_fx(&mut self) -> EffectReport {
        if self.fx.is_empty() {
            return EffectReport::default();
        }
        let batch = std::mem::take(&mut self.fx);
        let rep = apply_side_effects(&mut self.state, &batch);
        self.fx = batch;
        self.fx.clear();
        rep
    }
}

/// Would the round-end event of a lost round queue a `PreventDeath`? Runs the event on copies
/// of the jokers and the RNG, so nothing of the real state changes.
fn round_end_saves(s: &GameState, end: &RoundEndInfo) -> bool {
    let mut jokers = s.jokers.clone();
    let mut rng = s.rng.clone();
    let mut fx = Vec::new();
    let view = s.run_view_with(&s.consumables);
    hooks::fire_round_end(
        EventInput {
            jokers: &mut jokers,
            run: &view,
            deck: &s.deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        end,
    );
    fx.iter().any(|e| matches!(e, SideEffect::PreventDeath))
}

/// Fire one joker event with the current [`RunView`]; side effects are appended to `fx`. Returns
/// the money delta (not applied). Refreshes the modifier cache (hooks may change joker state).
fn fire(
    s: &mut GameState,
    fx: &mut Vec<SideEffect>,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> i32 {
    let view = s.run_view_with(&s.consumables);
    let r = f(EventInput {
        jokers: &mut s.jokers,
        run: &view,
        deck: &s.deck,
        rng: &mut s.rng,
        effects: fx,
    });
    s.refresh_modifiers();
    r.money_delta
}

/// [`fire`] and add the money delta to the run's money.
fn fire_paid(
    s: &mut GameState,
    fx: &mut Vec<SideEffect>,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) {
    let d = fire(s, fx, f);
    s.money = s.money.saturating_add(d);
}

/// Next blind: Small -> Big -> Boss -> next ante's Small (with a fresh ante set-up).
fn advance_blind(s: &mut GameState) {
    match s.blind {
        BlindKind::Small => s.blind = BlindKind::Big,
        BlindKind::Big => s.blind = BlindKind::Boss,
        BlindKind::Boss => {
            s.ante += 1;
            s.start_ante();
        }
    }
    s.refresh_target();
}

/// Return every card to the draw pile (end of round), clearing round-only flags. Canonical uid
/// order; the next `reset_for_round` shuffles.
fn gather_deck(s: &mut GameState) {
    let d = &mut s.deck;
    let mut hand = std::mem::take(&mut d.hand);
    let mut played = std::mem::take(&mut d.played);
    let mut disc = std::mem::take(&mut d.discard_pile);
    d.draw_pile.append(&mut hand);
    d.draw_pile.append(&mut played);
    d.draw_pile.append(&mut disc);
    for c in &mut d.draw_pile {
        c.debuffed = false;
        c.face_down = false;
    }
    d.draw_pile.sort_by_key(|c| c.uid);
}

/// Apply a side-effect queue to the state, in order (hooks.md §6, §15), then fire the follow-up
/// events (`on_cards_destroyed` with pre-removal snapshots, `on_cards_added`, `on_acquire` for
/// created jokers) and apply what they queue, up to `MAX_EFFECT_CASCADE` rounds.
///
/// Rules: eternal jokers are never destroyed; creations that need a slot are dropped when there
/// is no room; effects whose systems do not exist yet are dropped and counted in
/// `stats.effects_dropped` (random jokers: Phase 5). Never panics.
pub fn apply_side_effects(s: &mut GameState, effects: &[SideEffect]) -> EffectReport {
    let mut rep = EffectReport::default();
    let mut pending: Vec<SideEffect> = effects.to_vec();
    let mut depth = 0;
    while !pending.is_empty() {
        if depth >= MAX_EFFECT_CASCADE {
            let n = pending.len() as u32;
            rep.dropped += n;
            s.stats.effects_dropped += n;
            break;
        }
        depth += 1;
        let batch = std::mem::take(&mut pending);
        let mut destroyed: Vec<Card> = Vec::new();
        let mut added: Vec<Card> = Vec::new();
        let mut new_jokers: Vec<u32> = Vec::new();
        for &e in &batch {
            apply_one(s, e, &mut destroyed, &mut added, &mut new_jokers, &mut rep);
        }
        s.refresh_modifiers();
        if !destroyed.is_empty() {
            fire_paid(s, &mut pending, |inp| {
                hooks::fire_cards_destroyed(inp, &destroyed)
            });
        }
        if !added.is_empty() {
            fire_paid(s, &mut pending, |inp| hooks::fire_cards_added(inp, &added));
        }
        for uid in new_jokers {
            if let Some(slot) = s.jokers.iter().position(|j| j.uid == uid) {
                fire_paid(s, &mut pending, |inp| hooks::fire_acquire(inp, slot));
            }
        }
    }
    rep
}

fn drop_effect(s: &mut GameState, rep: &mut EffectReport) {
    rep.dropped += 1;
    s.stats.effects_dropped += 1;
}

fn apply_one(
    s: &mut GameState,
    e: SideEffect,
    destroyed: &mut Vec<Card>,
    added: &mut Vec<Card>,
    new_jokers: &mut Vec<u32>,
    rep: &mut EffectReport,
) {
    match e {
        SideEffect::LevelUpHand { hand, levels } => {
            let l = &mut s.hand_levels[hand.index()];
            *l = (i64::from(*l) + i64::from(levels)).clamp(1, i64::from(u32::MAX)) as u32;
        }
        SideEffect::CreateConsumable { kind, negative } => {
            if !negative && !s.has_consumable_room() {
                return drop_effect(s, rep);
            }
            let id = consumable::random_consumable(
                kind,
                &s.consumables,
                &s.hand_plays_run,
                s.mods.allow_duplicates,
                &mut s.rng,
            );
            s.consumables
                .push(ConsumableInstance::new(id).with_negative(negative));
        }
        SideEffect::AddConsumable { id, negative } => {
            if !negative && !s.has_consumable_room() {
                return drop_effect(s, rep);
            }
            s.consumables
                .push(ConsumableInstance::new(id).with_negative(negative));
        }
        SideEffect::CreatePlanetFor { hand } => {
            if !s.has_consumable_room() {
                return drop_effect(s, rep);
            }
            s.consumables
                .push(ConsumableInstance::new(ConsumableId::planet_for(hand)));
        }
        SideEffect::CopyConsumable { index, negative } => {
            let Some(src) = s.consumables.get(index).copied() else {
                return drop_effect(s, rep);
            };
            if !negative && !s.has_consumable_room() {
                return drop_effect(s, rep);
            }
            s.consumables
                .push(ConsumableInstance::new(src.id).with_negative(negative));
        }
        // TODO(Phase 5): random joker of a rarity (shop pools, Showman, Gros Michel flag).
        SideEffect::CreateJoker { .. } => drop_effect(s, rep),
        SideEffect::CopyJoker {
            uid,
            strip_negative,
        } => {
            let Some(src) = s.jokers.iter().find(|j| j.uid == uid).copied() else {
                return drop_effect(s, rep);
            };
            let mut copy = src;
            if strip_negative && copy.edition == Edition::Negative {
                copy.edition = Edition::Base;
            }
            if copy.edition != Edition::Negative && !s.has_joker_room() {
                return drop_effect(s, rep);
            }
            copy.uid = s.alloc_joker_uid();
            s.jokers.push(copy);
            new_jokers.push(copy.uid);
        }
        SideEffect::DestroyJoker { uid } => {
            match s.jokers.iter().position(|j| j.uid == uid) {
                Some(pos) if !s.jokers[pos].stickers.eternal => {
                    s.jokers.remove(pos);
                }
                Some(_) => drop_effect(s, rep), // eternal
                None => {}                      // already gone: idempotent
            }
        }
        SideEffect::DestroyCard { uid } => {
            if let Some(c) = s.deck.remove_card(uid) {
                destroyed.push(c);
                s.stats.cards_destroyed += 1;
                rep.destroyed += 1;
            }
        }
        SideEffect::AddCard { card, to } => {
            let mut c = card;
            c.uid = s.deck.alloc_uid();
            // Outside a round every card lives in the draw pile; `Played` is transient.
            let pile = match (s.phase, to) {
                (Phase::Play, Pile::Played) => Pile::Discard,
                (Phase::Play, p) => p,
                _ => Pile::Draw,
            };
            match pile {
                // Shuffled into the draw pile (Marble Joker, packs): a uniformly random position
                // on `RngKey::Deck`, not the top, so the card is not guaranteed to be drawn next.
                Pile::Draw => {
                    let n = s.deck.draw_pile.len();
                    let pos = s.rng.range(RngKey::Deck, 0, n + 1);
                    s.deck.draw_pile.insert(pos, c);
                }
                Pile::Hand => {
                    s.deck.add_card(c, pile);
                    s.deck.sort_hand();
                }
                Pile::Played | Pile::Discard => s.deck.add_card(c, pile),
            }
            added.push(c);
            s.stats.cards_added += 1;
            rep.added += 1;
        }
        SideEffect::AddSellValue { target, amount } => match target {
            SellTarget::AllJokers => {
                for j in &mut s.jokers {
                    j.sell_value = j.sell_value.saturating_add(amount);
                }
            }
            SellTarget::AllConsumables => {
                for c in &mut s.consumables {
                    c.sell_value = c.sell_value.saturating_add(amount);
                }
            }
        },
        SideEffect::AddHands(n) => s.hands_left = add_clamped(s.hands_left, n),
        SideEffect::AddDiscards(n) => s.discards_left = add_clamped(s.discards_left, n),
        SideEffect::ClearDiscards => s.discards_left = 0,
        SideEffect::CreateTag { id } => s.tags.push(id),
        // Reverts the boss set-up mid-round; Wall/Violet Vessel fall back to x2 (Needle keeps
        // x1) and the target is recomputed.
        SideEffect::DisableBoss => boss::disable(s),
        SideEffect::PreventDeath => rep.prevent_death = true,
        SideEffect::SetRunFlag(f) => {
            if !s.run_flags.contains(&f) {
                s.run_flags.push(f);
            }
        }
    }
}

/// Purple Seal: each discarded (non-debuffed) Purple Seal card creates a Tarot (cards.md).
fn push_purple_seals(cards: &[Card], fx: &mut Vec<SideEffect>) {
    for c in cards {
        if c.seal == Seal::Purple && !c.debuffed {
            fx.push(SideEffect::CreateConsumable {
                kind: ConsumableKind::Tarot,
                negative: false,
            });
        }
    }
}

/// Run counters after a use: per-id count, Tarot/Planet/unique-Planet stats (Fortune Teller,
/// Satellite) and The Fool's last Tarot/Planet (never The Fool, never a Spectral).
fn record_use(s: &mut GameState, id: ConsumableId) {
    let first = s.consumable_uses(id) == 0;
    if let Some(n) = s.consumable_use_counts.get_mut(id.index()) {
        *n += 1;
    }
    match id.kind() {
        ConsumableKind::Tarot => s.stats.run.tarots_used += 1,
        ConsumableKind::Planet => {
            s.stats.run.planets_used += 1;
            if first {
                s.stats.run.unique_planets_used += 1;
            }
        }
        ConsumableKind::Spectral => {}
    }
    if id.kind() != ConsumableKind::Spectral && id != ConsumableId::TheFool {
        s.last_tarot_planet = Some(id);
    }
}

fn add_clamped(v: u32, n: i32) -> u32 {
    (i64::from(v) + i64::from(n)).clamp(0, i64::from(u32::MAX)) as u32
}
