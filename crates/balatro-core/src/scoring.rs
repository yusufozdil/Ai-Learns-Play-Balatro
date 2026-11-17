//! Scoring pipeline with a fixed event order (`prd.md` §3.2, `docs/mechanics/scoring.md`,
//! decisions in `notes.md` §4.1), and [`ScoreCtx`], the context every scoring hook receives.
//!
//! Usage (game layer): evaluate the played cards with `hand_eval::evaluate`, fill a
//! [`RunView`], then call [`Scorer::score_hand`]. Apply `ScoreResult::money_delta` and the
//! queued [`Scorer::effects`] afterwards. Played cards and joker states are mutated in place.
//!
//! The [`Scorer`] owns reusable buffers: after warm-up, scoring does not allocate when tracing
//! is off.

use crate::card::Card;
use crate::content::jokers::try_def;
use crate::data::card_effects as fx;
use crate::data::hand_levels::chips_mult;
use crate::enums::{Edition, Enhancement, HandType, Seal, Suit};
use crate::hand_eval::HandEval;
use crate::hooks::{
    CallInfo, CardFn, DeckCounts, Hooks, JokerFn, RetriggerFn, RunView, SideEffect,
};
use crate::joker::{JokerDef, JokerInstance};
use crate::rng::{GameRng, RngKey};

/// The single rounding rule (notes.md §4.1): chips and mult stay `f64` with no intermediate
/// rounding; the hand score is `floor(chips * mult)`, never negative, NaN -> 0.
pub fn round_score(chips: f64, mult: f64) -> f64 {
    let s = (chips * mult).floor();
    if s.is_nan() || s < 0.0 { 0.0 } else { s }
}

/// The Flint (notes.md §4.1): `ceil(x / 2)`, applied to base chips and base mult.
pub fn halve_round_up(x: f64) -> f64 {
    (x / 2.0).ceil()
}

/// Everything one `score_hand` call needs.
pub struct ScoreInput<'a> {
    /// Played cards, left to right (the game's played pile). Mutated in place (Hiker perma
    /// bonus, Midas Mask, Vampire).
    pub played: &'a mut [Card],
    /// Cards still held in hand, left to right.
    pub held: &'a [Card],
    /// `hand_eval::evaluate(played, &run.rules)` (computed by the caller).
    pub eval: HandEval,
    /// Joker area, left to right. Joker states are mutated in place.
    pub jokers: &'a mut [JokerInstance],
    /// Run facts.
    pub run: &'a RunView<'a>,
    /// RNG: card effects use `RngKey::Cards`, jokers `RngKey::Jokers`.
    pub rng: &'a mut GameRng,
}

/// Outcome of one hand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreResult {
    /// `round_score(chips, mult)`: add to the round score.
    pub score: f64,
    /// Final chips.
    pub chips: f64,
    /// Final mult.
    pub mult: f64,
    /// Money gained during the hand (Gold Seal, Lucky, jokers). Apply to the run.
    pub money_delta: i32,
    /// Hand type scored.
    pub hand_type: HandType,
    /// Hand level used for the base values (after `before_scoring` level-ups).
    pub hand_level: u32,
    /// The boss rejected the hand (score 0).
    pub invalid: bool,
}

/// Pipeline step currently running (`ctx.step()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Step 0, invalid hand (`on_invalid_hand`).
    Invalid,
    /// Step 2, `before_scoring`.
    Before,
    /// Step 4, played cards.
    Cards,
    /// Step 5, held cards.
    Held,
    /// Step 6, joker phase.
    Jokers,
    /// Step 7, Observatory / Plasma.
    Late,
    /// Step 9, after scoring.
    After,
    /// End-of-round held-card effects ([`Scorer::round_end_held`]).
    RoundEnd,
}

/// Where a traced change came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceSource {
    /// Hand base values.
    Base,
    /// Boss blind (The Flint).
    Boss,
    /// A played card's own modifiers (rank, perma bonus, enhancement, seal, edition).
    Card {
        /// Index in the played cards.
        index: u8,
        /// 0 = first trigger, 1.. = retriggers.
        trigger: u8,
    },
    /// A held card's own modifiers (Steel, Gold, Blue Seal).
    Held {
        /// Index in the held cards.
        index: u8,
        /// 0 = first trigger.
        trigger: u8,
    },
    /// A joker hook.
    Joker {
        /// Slot where the hook executed.
        slot: u8,
        /// Slot whose ability ran (differs for Blueprint/Brainstorm).
        src: u8,
    },
    /// A joker's edition.
    JokerEdition {
        /// Joker slot.
        slot: u8,
    },
    /// Observatory, for the consumable at this index.
    Observatory {
        /// Consumable index.
        consumable: u8,
    },
    /// Plasma Deck balancing.
    Plasma,
}

/// What changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceOp {
    /// Base values set from the hand level.
    Base {
        /// Hand type.
        hand: HandType,
        /// Level used.
        level: u32,
    },
    /// The Flint halved the base.
    HalveBase,
    /// `+chips`.
    AddChips(f64),
    /// `+mult`.
    AddMult(f64),
    /// `xmult`.
    XMult(f64),
    /// Money gained.
    Money(i32),
    /// Extra triggers granted to a card (by its Red Seal or by a joker).
    Retriggers {
        /// Card index (played or held).
        card: u8,
        /// The card is held (not played).
        held: bool,
        /// Extra triggers.
        count: u32,
    },
    /// Lucky card rolls of one trigger.
    LuckyRoll {
        /// +20 mult hit.
        mult_hit: bool,
        /// $20 hit.
        money_hit: bool,
    },
    /// Glass break roll after scoring (once per Glass card).
    GlassRoll {
        /// Card index.
        card: u8,
        /// Broke.
        broke: bool,
    },
    /// Hand level-up requested.
    LevelUp {
        /// Hand type.
        hand: HandType,
        /// Levels.
        levels: i32,
    },
    /// A debuffed scoring card was skipped.
    Debuffed {
        /// Card index.
        card: u8,
    },
    /// Plasma Deck balanced chips and mult.
    Balance,
    /// A side effect was queued.
    Effect(SideEffect),
}

/// One trace entry: the change and the chips/mult right after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceEvent {
    /// Source.
    pub source: TraceSource,
    /// Change.
    pub op: TraceOp,
    /// Chips after the change.
    pub chips: f64,
    /// Mult after the change.
    pub mult: f64,
}

/// Resolved hook table of one joker slot for the current call.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Slot whose ability runs here (copy target for copiers).
    src: usize,
    /// `None` = inert (debuffed, unknown, or a copier with no valid target).
    hooks: Option<&'static Hooks>,
}

/// The card currently being processed.
#[derive(Debug, Clone, Copy, Default)]
struct CardInfo {
    index: usize,
    pos: usize,
    trigger: u32,
    lucky: bool,
}

fn small(i: usize) -> u8 {
    u8::try_from(i).unwrap_or(u8::MAX)
}

/// Reusable scorer: owns the side-effect queue, the trace and scratch buffers.
#[derive(Debug, Default)]
pub struct Scorer {
    effects: Vec<SideEffect>,
    trace: Vec<TraceEvent>,
    slots: Vec<Slot>,
    trace_enabled: bool,
}

impl Scorer {
    /// New scorer, tracing off.
    pub fn new() -> Self {
        Self::default()
    }

    /// New scorer with tracing on/off.
    pub fn with_trace(on: bool) -> Self {
        Scorer {
            trace_enabled: on,
            ..Self::default()
        }
    }

    /// Turn tracing on/off.
    pub fn set_trace(&mut self, on: bool) {
        self.trace_enabled = on;
    }

    /// Side effects queued by the last call, in order. Apply them, then fire follow-up events.
    pub fn effects(&self) -> &[SideEffect] {
        &self.effects
    }

    /// Trace of the last call (empty when tracing is off).
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    fn resolve(&mut self, jokers: &[JokerInstance]) {
        self.slots.clear();
        for slot in 0..jokers.len() {
            let res = crate::hooks::resolve_source(jokers, slot)
                .and_then(|src| try_def(jokers[src].id).map(|d| (src, &d.hooks)));
            self.slots.push(match res {
                Some((src, hooks)) => Slot {
                    src,
                    hooks: Some(hooks),
                },
                None => Slot {
                    src: slot,
                    hooks: None,
                },
            });
        }
    }

    /// Score one played hand (steps 0-9 of docs/mechanics/scoring.md).
    pub fn score_hand(&mut self, inp: ScoreInput<'_>) -> ScoreResult {
        self.effects.clear();
        self.trace.clear();
        self.resolve(inp.jokers);
        let level = inp.run.hand_level(inp.eval.hand_type);
        let mut ctx = ScoreCtx {
            run: inp.run,
            eval: inp.eval,
            played: inp.played,
            held: inp.held,
            jokers: inp.jokers,
            slots: &self.slots,
            rng: inp.rng,
            effects: &mut self.effects,
            trace: &mut self.trace,
            trace_on: self.trace_enabled,
            chips: 0.0,
            mult: 0.0,
            money_delta: 0,
            level,
            deck: inp.run.deck,
            step: Step::Before,
            call: CallInfo::default(),
            card: CardInfo::default(),
            source: TraceSource::Base,
            score: 0.0,
            invalid: false,
        };
        ctx.run_hand();
        ScoreResult {
            score: ctx.score,
            chips: ctx.chips,
            mult: ctx.mult,
            money_delta: ctx.money_delta,
            hand_type: ctx.eval.hand_type,
            hand_level: ctx.level,
            invalid: ctx.invalid,
        }
    }

    /// End-of-round effects of cards still held in hand: Gold card ($3) and Blue Seal (Planet
    /// of `last_hand`), each repeated by Red Seal and `held_retriggers` jokers (Mime and its
    /// copies). Debuffed cards do nothing. Returns the money gained; the Planets are queued in
    /// [`Scorer::effects`]. Call before the joker `on_round_end` event.
    pub fn round_end_held(
        &mut self,
        held: &[Card],
        jokers: &mut [JokerInstance],
        run: &RunView<'_>,
        rng: &mut GameRng,
        last_hand: HandType,
    ) -> i32 {
        self.effects.clear();
        self.trace.clear();
        self.resolve(jokers);
        let mut no_played: [Card; 0] = [];
        let mut ctx = ScoreCtx {
            run,
            eval: HandEval {
                hand_type: last_hand,
                scoring_mask: 0,
                contains: 0,
            },
            played: &mut no_played,
            held,
            jokers,
            slots: &self.slots,
            rng,
            effects: &mut self.effects,
            trace: &mut self.trace,
            trace_on: self.trace_enabled,
            chips: 0.0,
            mult: 0.0,
            money_delta: 0,
            level: run.hand_level(last_hand),
            deck: run.deck,
            step: Step::RoundEnd,
            call: CallInfo::default(),
            card: CardInfo::default(),
            source: TraceSource::Base,
            score: 0.0,
            invalid: false,
        };
        for (idx, &card) in held.iter().enumerate() {
            let gold = card.enhancement == Enhancement::Gold;
            let blue = card.seal == Seal::Blue;
            if card.debuffed || !(gold || blue) {
                continue;
            }
            ctx.card = CardInfo {
                index: idx,
                ..CardInfo::default()
            };
            let triggers = ctx.trigger_count(&card, true);
            for t in 0..triggers {
                ctx.card.trigger = t;
                ctx.source = TraceSource::Held {
                    index: small(idx),
                    trigger: small(t as usize),
                };
                if gold {
                    ctx.add_money(fx::GOLD_CARD_MONEY);
                }
                if blue {
                    ctx.push_effect(SideEffect::CreatePlanetFor { hand: last_hand });
                }
            }
        }
        ctx.money_delta
    }
}

/// Context of a scoring hook. Read: cards, hand, jokers, run facts, running chips/mult/money.
/// Write: chips/mult (add / x), money, played-card mutations, side effects. A hook's own joker
/// state is the `j` parameter, not `ctx.jokers()`.
pub struct ScoreCtx<'a> {
    run: &'a RunView<'a>,
    eval: HandEval,
    played: &'a mut [Card],
    held: &'a [Card],
    jokers: &'a mut [JokerInstance],
    slots: &'a [Slot],
    rng: &'a mut GameRng,
    effects: &'a mut Vec<SideEffect>,
    trace: &'a mut Vec<TraceEvent>,
    trace_on: bool,
    chips: f64,
    mult: f64,
    money_delta: i32,
    level: u32,
    deck: DeckCounts,
    step: Step,
    call: CallInfo,
    card: CardInfo,
    source: TraceSource,
    score: f64,
    invalid: bool,
}

// ----- hook-facing API -----
impl<'a> ScoreCtx<'a> {
    /// Run facts (money before this hand, counts, levels, rules, ...).
    pub fn run(&self) -> &RunView<'a> {
        self.run
    }

    /// Hand evaluation (hand type, `contains`, scoring mask).
    pub fn eval(&self) -> &HandEval {
        &self.eval
    }

    /// Played hand type.
    pub fn hand_type(&self) -> HandType {
        self.eval.hand_type
    }

    /// The played cards contain `ht` (Jolly Joker, The Duo, ...).
    pub fn contains(&self, ht: HandType) -> bool {
        self.eval.contains(ht)
    }

    /// Played cards, left to right (all of them, scoring or not).
    pub fn played(&self) -> &[Card] {
        self.played
    }

    /// Indices of the scoring played cards, left to right (debuffed ones included).
    pub fn scoring_indices(&self) -> impl Iterator<Item = usize> + use<> {
        let mask = self.eval.scoring_mask;
        (0..self.played.len().min(8)).filter(move |&i| mask & (1 << i) != 0)
    }

    /// Number of scoring played cards.
    pub fn scoring_count(&self) -> usize {
        self.scoring_indices().count()
    }

    /// Cards held in hand, left to right.
    pub fn held(&self) -> &[Card] {
        self.held
    }

    /// All jokers (own state may be stale while its hook runs; use `j`).
    pub fn jokers(&self) -> &[JokerInstance] {
        self.jokers
    }

    /// Definition of the joker at `slot`.
    pub fn joker_def(&self, slot: usize) -> Option<&'static JokerDef> {
        self.jokers.get(slot).and_then(|j| try_def(j.id))
    }

    /// Slot of the joker whose ability runs (the copy target for copies).
    pub fn slot(&self) -> usize {
        self.call.slot
    }

    /// Slot where the hook executes (the copier's slot for copies).
    pub fn exec_slot(&self) -> usize {
        self.call.exec_slot
    }

    /// The hook runs on behalf of Blueprint/Brainstorm. State changes to `j` are discarded
    /// automatically; check this before effects that must not be duplicated (card mutations,
    /// destroying jokers).
    pub fn copied(&self) -> bool {
        self.call.copied()
    }

    /// Pipeline step currently running.
    pub fn step(&self) -> Step {
        self.step
    }

    /// Index of the current card: into `played()` (scored hooks) or `held()` (held hooks).
    pub fn card_index(&self) -> usize {
        self.card.index
    }

    /// Position of the current card among the scoring cards (0 = first scoring card).
    pub fn scoring_pos(&self) -> usize {
        self.card.pos
    }

    /// Trigger number of the current card (0 = first trigger, 1.. = retriggers).
    pub fn trigger(&self) -> u32 {
        self.card.trigger
    }

    /// The current trigger of a Lucky card hit at least one of its rolls (Lucky Cat).
    pub fn lucky_triggered(&self) -> bool {
        self.card.lucky
    }

    /// Current chips.
    pub fn chips(&self) -> f64 {
        self.chips
    }

    /// Current mult.
    pub fn mult(&self) -> f64 {
        self.mult
    }

    /// Level of the played hand used for the base (raised by `level_up_hand` in `before_scoring`).
    pub fn hand_level(&self) -> u32 {
        self.level
    }

    /// Money including gains made earlier in this hand (Bull, Bootstraps, Vagabond).
    pub fn money(&self) -> i32 {
        self.run.money.saturating_add(self.money_delta)
    }

    /// Full-deck counts, updated for enhancement changes made during this hand.
    pub fn deck(&self) -> &DeckCounts {
        &self.deck
    }

    /// The boss rejected this hand (only `on_invalid_hand` and `after_scoring` run).
    pub fn hand_invalid(&self) -> bool {
        self.invalid
    }

    /// Face card under joker semantics (see [`RunView::is_face`]).
    pub fn is_face(&self, card: &Card) -> bool {
        self.run.is_face(card)
    }

    /// Suit test under the active rules (Wild, Smeared, Stone).
    pub fn has_suit(&self, card: &Card, suit: Suit) -> bool {
        self.run.has_suit(card, suit)
    }

    /// `+chips`.
    pub fn add_chips(&mut self, x: f64) {
        self.chips += x;
        self.rec(TraceOp::AddChips(x));
    }

    /// `+mult`.
    pub fn add_mult(&mut self, x: f64) {
        self.mult += x;
        self.rec(TraceOp::AddMult(x));
    }

    /// `xmult`.
    ///
    /// Bit-exact with the real game: SMODS does not multiply, it adds `mult * (x - 1)`
    /// (`smods/src/game_object.lua` Scoring_Parameter 'mult' `calc_effect`, line ~3825:
    /// `self:modify(mult * (amount - 1))`, with `modify` = `current + amount`). In doubles
    /// `m + m * (x - 1)` differs from `m * x` by an ulp for non-dyadic `x` (Ramen 1.93:
    /// 20 + 20 * 0.93 = 38.599999999999994 -> 20 * that = 771.99.. -> 771, not 772). `x == 1`
    /// is skipped there (a no-op here too).
    pub fn x_mult(&mut self, x: f64) {
        self.mult += self.mult * (x - 1.0);
        self.rec(TraceOp::XMult(x));
    }

    /// Gain (or lose) money now; visible to later hooks through `money()`.
    pub fn add_money(&mut self, dollars: i32) {
        self.money_delta = self.money_delta.saturating_add(dollars);
        self.rec(TraceOp::Money(dollars));
    }

    /// Queue a deferred side effect.
    pub fn push_effect(&mut self, e: SideEffect) {
        self.effects.push(e);
        self.rec(TraceOp::Effect(e));
    }

    /// Side effects queued so far in this hand.
    pub fn effects(&self) -> &[SideEffect] {
        self.effects
    }

    /// Level up a hand type. Persisted through a `LevelUpHand` side effect; when called during
    /// `before_scoring` for the played hand type it also raises this hand's base (Space Joker).
    pub fn level_up_hand(&mut self, hand: HandType, levels: i32) {
        if self.step == Step::Before && hand == self.eval.hand_type {
            self.level =
                (i64::from(self.level) + i64::from(levels)).clamp(1, i64::from(u32::MAX)) as u32;
        }
        self.rec(TraceOp::LevelUp { hand, levels });
        self.effects.push(SideEffect::LevelUpHand { hand, levels });
    }

    /// Destroy the joker whose ability runs (after the hand). No-op for copies.
    pub fn destroy_self(&mut self) {
        if !self.copied()
            && let Some(j) = self.jokers.get(self.call.slot)
        {
            let e = SideEffect::DestroyJoker { uid: j.uid };
            self.push_effect(e);
        }
    }

    /// Probability roll with the run's probability multiplier, on `RngKey::Jokers`.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.rng
            .chance(RngKey::Jokers, numerator, denominator, self.run.prob_mult)
    }

    /// Uniform integer in `[lo, hi_exclusive)` on `RngKey::Jokers` (Misprint).
    pub fn rng_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        self.rng.range(RngKey::Jokers, lo, hi_exclusive)
    }

    /// Raw RNG access. Joker hooks must only use `RngKey::Jokers`.
    pub fn rng(&mut self) -> &mut GameRng {
        self.rng
    }

    /// Mutable access to a played card (Midas Mask, Vampire). For enhancement changes prefer
    /// [`ScoreCtx::set_enhancement`], which keeps `deck()` counts right.
    pub fn played_mut(&mut self, index: usize) -> Option<&mut Card> {
        self.played.get_mut(index)
    }

    /// Change a played card's enhancement (Midas Mask -> Gold, Vampire -> None) and update
    /// the full-deck counts.
    pub fn set_enhancement(&mut self, index: usize, e: Enhancement) {
        if let Some(c) = self.played.get_mut(index) {
            let old = *c;
            c.enhancement = e;
            let new = *c;
            self.deck.remove(&old);
            self.deck.add(&new);
        }
    }

    /// Permanently add chips to a played card (Hiker). Later triggers of the card see it.
    pub fn add_perma_bonus(&mut self, index: usize, chips: i32) {
        if let Some(c) = self.played.get_mut(index) {
            c.perma_bonus = c.perma_bonus.saturating_add(chips);
        }
    }
}

// ----- pipeline internals -----
impl ScoreCtx<'_> {
    fn rec(&mut self, op: TraceOp) {
        if self.trace_on {
            self.trace.push(TraceEvent {
                source: self.source,
                op,
                chips: self.chips,
                mult: self.mult,
            });
        }
    }

    fn enter(&mut self, exec_slot: usize, src: usize) {
        self.call = CallInfo {
            slot: src,
            exec_slot,
        };
        self.source = TraceSource::Joker {
            slot: small(exec_slot),
            src: small(src),
        };
    }

    /// Run a no-card hook on every joker that has it, left to right.
    fn run_joker_hooks(&mut self, get: impl Fn(&Hooks) -> Option<JokerFn>) {
        let slots = self.slots;
        for (slot, s) in slots.iter().enumerate() {
            if let Some(h) = s.hooks
                && let Some(f) = get(h)
            {
                self.call_joker(slot, s.src, f);
            }
        }
    }

    fn call_joker(&mut self, slot: usize, src: usize, f: JokerFn) {
        let mut inst = self.jokers[src];
        self.enter(slot, src);
        f(&mut inst, self);
        if src == slot {
            self.jokers[src] = inst;
        }
    }

    /// Run a per-card hook on every joker that has it, left to right.
    fn run_card_hooks(&mut self, card: &Card, get: impl Fn(&Hooks) -> Option<CardFn>) {
        let slots = self.slots;
        for (slot, s) in slots.iter().enumerate() {
            if let Some(h) = s.hooks
                && let Some(f) = get(h)
            {
                let mut inst = self.jokers[s.src];
                self.enter(slot, s.src);
                f(&mut inst, self, card);
                if s.src == slot {
                    self.jokers[s.src] = inst;
                }
            }
        }
    }

    /// Total triggers of a card: 1 + Red Seal + joker retriggers (left to right), capped.
    fn trigger_count(&mut self, card: &Card, held: bool) -> u32 {
        let index = small(self.card.index);
        let mut extra: u32 = 0;
        if card.seal == Seal::Red {
            self.source = if held {
                TraceSource::Held { index, trigger: 0 }
            } else {
                TraceSource::Card { index, trigger: 0 }
            };
            extra = 1;
            self.rec(TraceOp::Retriggers {
                card: index,
                held,
                count: 1,
            });
        }
        let get: fn(&Hooks) -> Option<RetriggerFn> = if held {
            |h| h.held_retriggers
        } else {
            |h| h.card_retriggers
        };
        let slots = self.slots;
        for (slot, s) in slots.iter().enumerate() {
            if let Some(h) = s.hooks
                && let Some(f) = get(h)
            {
                self.enter(slot, s.src);
                let n = f(&self.jokers[s.src], self, card);
                if n > 0 {
                    extra = extra.saturating_add(n);
                    self.rec(TraceOp::Retriggers {
                        card: index,
                        held,
                        count: n,
                    });
                }
            }
        }
        extra.saturating_add(1).min(fx::MAX_TRIGGERS_PER_CARD)
    }

    fn apply_edition(&mut self, e: Edition) {
        match e {
            Edition::Foil => self.add_chips(fx::FOIL_CHIPS),
            Edition::Holographic => self.add_mult(fx::HOLO_MULT),
            Edition::Polychrome => self.x_mult(fx::POLYCHROME_XMULT),
            Edition::Base | Edition::Negative => {}
        }
    }

    fn run_hand(&mut self) {
        // Step 0: boss validation.
        if self.run.boss.hand_invalid {
            self.invalid = true;
            self.step = Step::Invalid;
            self.run_joker_hooks(|h| h.on_invalid_hand);
            self.score = 0.0;
            self.step = Step::After;
            self.run_joker_hooks(|h| h.after_scoring);
            return;
        }
        // Step 2: "on played" jokers (may level up the current hand).
        self.step = Step::Before;
        self.run_joker_hooks(|h| h.before_scoring);
        // Steps 1 + 3: base values of the current level, then The Flint.
        self.read_base();
        // Step 4: scoring cards.
        self.step = Step::Cards;
        self.score_played();
        // Step 5: held cards.
        self.step = Step::Held;
        self.score_held();
        // Step 6: jokers.
        self.step = Step::Jokers;
        self.score_jokers();
        // Step 7: Observatory, Plasma.
        self.step = Step::Late;
        self.late_modifiers();
        // Step 8: final score.
        self.score = round_score(self.chips, self.mult);
        // Step 9: Glass breaks, after-scoring hooks.
        self.step = Step::After;
        self.glass_rolls();
        self.run_joker_hooks(|h| h.after_scoring);
    }

    fn read_base(&mut self) {
        let hand = self.eval.hand_type;
        let (chips, mult) = chips_mult(hand, self.level);
        self.chips = chips;
        self.mult = mult;
        self.source = TraceSource::Base;
        self.rec(TraceOp::Base {
            hand,
            level: self.level,
        });
        if self.run.boss.halve_base {
            self.chips = halve_round_up(self.chips);
            self.mult = halve_round_up(self.mult);
            self.source = TraceSource::Boss;
            self.rec(TraceOp::HalveBase);
        }
    }

    fn score_played(&mut self) {
        let n = self.played.len().min(8);
        let mut pos = 0;
        for idx in 0..n {
            if !self.eval.scores(idx) {
                continue;
            }
            self.card = CardInfo {
                index: idx,
                pos,
                trigger: 0,
                lucky: false,
            };
            pos += 1;
            let card = self.played[idx];
            if card.debuffed {
                self.source = TraceSource::Card {
                    index: small(idx),
                    trigger: 0,
                };
                self.rec(TraceOp::Debuffed { card: small(idx) });
                continue;
            }
            let triggers = self.trigger_count(&card, false);
            for t in 0..triggers {
                self.card.trigger = t;
                self.trigger_played(idx);
            }
        }
    }

    /// One trigger of a played scoring card: steps 4a-4e.
    fn trigger_played(&mut self, idx: usize) {
        let card = self.played[idx];
        self.card.lucky = false;
        self.source = TraceSource::Card {
            index: small(idx),
            trigger: small(self.card.trigger as usize),
        };
        // a. rank chips + permanent bonus
        let rank_chips = card.base_chips();
        if rank_chips != 0 {
            self.add_chips(f64::from(rank_chips));
        }
        if card.perma_bonus != 0 {
            self.add_chips(f64::from(card.perma_bonus));
        }
        // b. enhancement
        match card.enhancement {
            Enhancement::Bonus => self.add_chips(fx::BONUS_CHIPS),
            Enhancement::Stone => self.add_chips(fx::STONE_CHIPS),
            Enhancement::Mult => self.add_mult(fx::MULT_CARD_MULT),
            Enhancement::Glass => self.x_mult(fx::GLASS_XMULT),
            Enhancement::Lucky => self.lucky_trigger(),
            Enhancement::None | Enhancement::Wild | Enhancement::Steel | Enhancement::Gold => {}
        }
        // c. seal
        if card.seal == Seal::Gold {
            self.add_money(fx::GOLD_SEAL_MONEY);
        }
        // d. edition
        self.apply_edition(card.edition);
        // e. jokers
        self.run_card_hooks(&card, |h| h.on_card_scored);
    }

    fn lucky_trigger(&mut self) {
        let pm = self.run.prob_mult;
        let (n, d) = fx::LUCKY_MULT_ODDS;
        let mult_hit = self.rng.chance(RngKey::Cards, n, d, pm);
        let (n, d) = fx::LUCKY_MONEY_ODDS;
        let money_hit = self.rng.chance(RngKey::Cards, n, d, pm);
        self.rec(TraceOp::LuckyRoll {
            mult_hit,
            money_hit,
        });
        if mult_hit {
            self.add_mult(fx::LUCKY_MULT);
        }
        if money_hit {
            self.add_money(fx::LUCKY_MONEY);
        }
        self.card.lucky = mult_hit || money_hit;
    }

    fn score_held(&mut self) {
        let slots = self.slots;
        let any_held_hook = slots
            .iter()
            .any(|s| s.hooks.is_some_and(|h| h.on_held.is_some()));
        let held = self.held;
        for (idx, card) in held.iter().enumerate() {
            let steel = card.enhancement == Enhancement::Steel;
            if card.debuffed || !(steel || any_held_hook) {
                continue;
            }
            self.card = CardInfo {
                index: idx,
                ..CardInfo::default()
            };
            let triggers = self.trigger_count(card, true);
            for t in 0..triggers {
                self.card.trigger = t;
                if steel {
                    self.source = TraceSource::Held {
                        index: small(idx),
                        trigger: small(t as usize),
                    };
                    self.x_mult(fx::STEEL_XMULT);
                }
                self.run_card_hooks(card, |h| h.on_held);
            }
        }
    }

    fn score_jokers(&mut self) {
        let slots = self.slots;
        let any_other = slots
            .iter()
            .any(|s| s.hooks.is_some_and(|h| h.on_other_joker.is_some()));
        for i in 0..slots.len() {
            let me = self.jokers[i];
            // a. Foil / Holo
            if !me.debuffed && matches!(me.edition, Edition::Foil | Edition::Holographic) {
                self.source = TraceSource::JokerEdition { slot: small(i) };
                self.apply_edition(me.edition);
            }
            // b. own effect (copiers run their target's)
            if let Some(h) = slots[i].hooks
                && let Some(f) = h.joker_main
            {
                self.call_joker(i, slots[i].src, f);
            }
            // c. joker-on-joker (Baseball Card counts debuffed jokers too)
            if any_other {
                for (k, s) in slots.iter().enumerate() {
                    if k == i {
                        continue;
                    }
                    if let Some(h) = s.hooks
                        && let Some(f) = h.on_other_joker
                    {
                        let mut inst = self.jokers[s.src];
                        self.enter(k, s.src);
                        f(&mut inst, self, i);
                        if s.src == k {
                            self.jokers[s.src] = inst;
                        }
                    }
                }
            }
            // d. Polychrome
            if !me.debuffed && me.edition == Edition::Polychrome {
                self.source = TraceSource::JokerEdition { slot: small(i) };
                self.apply_edition(me.edition);
            }
        }
    }

    fn late_modifiers(&mut self) {
        let run = self.run;
        if run.observatory {
            let ht = self.eval.hand_type;
            for (i, c) in run.consumables.iter().enumerate() {
                if c.planet_hand() == Some(ht) {
                    self.source = TraceSource::Observatory {
                        consumable: small(i),
                    };
                    self.x_mult(fx::OBSERVATORY_XMULT);
                }
            }
        }
        if run.plasma_deck {
            // back.lua `Plasma Deck` final_scoring_step: both become `math.floor((chips + mult) / 2)`
            // (dump/back.lua:157-161). `mod_mult(F)` then stores `mult + (F - mult)`
            // (SMODS `modify` with `skip`, game_object.lua mult `modify`), which can differ from F
            // by an ulp when the old mult is not an integer; chips are integral so exact.
            let avg = ((self.chips + self.mult) / 2.0).floor();
            self.chips = avg;
            self.mult += avg - self.mult;
            self.source = TraceSource::Plasma;
            self.rec(TraceOp::Balance);
        }
    }

    fn glass_rolls(&mut self) {
        let n = self.played.len().min(8);
        let (num, den) = fx::GLASS_BREAK_ODDS;
        for idx in 0..n {
            let card = self.played[idx];
            if !self.eval.scores(idx) || card.debuffed || card.enhancement != Enhancement::Glass {
                continue;
            }
            let broke = self.rng.chance(RngKey::Cards, num, den, self.run.prob_mult);
            self.source = TraceSource::Card {
                index: small(idx),
                trigger: 0,
            };
            self.rec(TraceOp::GlassRoll {
                card: small(idx),
                broke,
            });
            if broke {
                self.push_effect(SideEffect::DestroyCard { uid: card.uid });
            }
        }
    }
}
