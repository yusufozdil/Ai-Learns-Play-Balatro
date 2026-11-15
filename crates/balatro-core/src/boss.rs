//! Boss blinds: the static [`BossDef`] (min ante, requirement multiplier, static rule changes,
//! [`BossHooks`]), per-ante/per-round [`BossState`], the hook contexts, boss selection, and the
//! game-layer glue that runs the hooks (`blind_start`, `draw`, `pre_play`, `disable`, ...).
//!
//! Content guide: `docs/design/consumables_bosses.md`. Ids live in
//! [`crate::content::ids::BossId`], definitions in `content/bosses.rs`, the registry in
//! [`crate::content`].
//!
//! A boss is **in effect** while `GameState::boss_in_effect()` holds: the current blind is the
//! boss, a boss was chosen, it is not disabled (Chicot/Luchador/Verdant Leaf sale) and a round
//! is being played. Hooks are only called while the boss is in effect.

use crate::blinds::MULT_BOSS_DEFAULT;
use crate::card::{Card, RuleFlags};
use crate::content::ids::BossId;
use crate::content::try_boss_def;
use crate::enums::{HandType, Phase};
use crate::joker::JokerInstance;
use crate::rng::{GameRng, RngKey};
use crate::state::GameState;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

/// Boss bookkeeping. Generic fields; each boss documents which ones it uses.
///
/// Lifetime: the whole struct is reset at ante start (then `on_ante_start` runs); the per-round
/// fields (`card_uid`, `joker_uid`, `removed`, `jokers_hidden`, `draws`) are reset when a blind
/// is selected; `triggered`/`hand_invalid` are reset before every played hand.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BossState {
    /// A hand type fixed by the boss for the whole ante (The Ox: the most played hand at ante
    /// start).
    pub hand: Option<HandType>,
    /// A tracked playing card (Cerulean Bell: the force-selected card). Play/Discard masks must
    /// include it while it is in hand (`actions::validate`).
    pub card_uid: Option<u32>,
    /// The joker this boss currently debuffs (Crimson Heart). Set only through
    /// [`BossCtx::set_debuffed_joker`]; the game restores it at round end and on disable.
    pub joker_uid: Option<u32>,
    /// Hands or discards taken away at blind start, given back by `on_disable` (Needle, Water).
    pub removed: u32,
    /// Jokers are face down (Amber Acorn; hidden from the observation, Phase 8).
    pub jokers_hidden: bool,
    /// Draws done this round (0 before the opening draw).
    pub draws: u32,
    /// The boss ability affected the current hand (Matador). Per hand.
    pub triggered: bool,
    /// The boss rejects the current hand (Psychic/Eye/Mouth): it scores 0. Per hand.
    pub hand_invalid: bool,
}

// ---------------------------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------------------------

/// Why cards are being drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawKind {
    /// The opening hand of the round (The House).
    RoundStart,
    /// Refill after a played hand (The Fish, The Serpent).
    AfterPlay,
    /// Refill after a discard (The Serpent).
    AfterDiscard,
    /// Refill for another reason (a consumable emptied the hand).
    Refill,
}

/// Event hook with full (controlled) state access: ante start, blind start, after draw, after
/// play, joker sold, disable.
pub type BossFn = fn(ctx: &mut BossCtx<'_>);
/// Card debuff predicate. `card` is passed with `debuffed = false`. Pure.
pub type DebuffFn = fn(card: &Card, rules: &RuleFlags) -> bool;
/// Face-down decision for one card being drawn (rolls allowed through `ctx`).
pub type FaceDownFn = fn(ctx: &mut DrawCtx<'_>, card: &Card) -> bool;
/// Number of cards to draw (`default` = up to the hand size). The result is capped by the draw
/// pile and by `MAX_HAND_SIZE` cards in hand.
pub type DrawCountFn = fn(kind: DrawKind, default: usize) -> usize;
/// Before a played hand is scored (preview-safe: works on copies inside `Game::preview_play`).
pub type PlayFn = fn(ctx: &mut PlayCtx<'_>);

/// Hook table of one boss. Every field is optional; use `..BossHooks::NONE`.
///
/// | hook | when | ctx | examples |
/// |---|---|---|---|
/// | `on_ante_start` | the boss was chosen for a new ante | Boss | The Ox (fix the hand) |
/// | `on_blind_start` | boss blind selected: after the deck reset and round counters, before joker `on_blind_select` and the first draw | Boss | Water, Needle, Amber Acorn, Crimson Heart |
/// | `debuffs_card` | whenever debuffs are recomputed (blind start, after every Play-phase action) | pure | Club/Goad/Window/Head, Plant, Pillar, Verdant Leaf |
/// | `face_down` | per card, as it is drawn | Draw | House, Wheel, Fish, Mark |
/// | `draw_count` | before each draw | pure | Serpent |
/// | `after_draw` | after each draw | Boss | Cerulean Bell |
/// | `before_play` | after the hand is paid and evaluated, before jokers score it | Play | Psychic, Eye, Mouth, Hook, Tooth, Ox, Arm |
/// | `after_play` | after the hand was scored and its effects applied, before the refill | Boss | Crimson Heart |
/// | `on_joker_sold` | a joker was sold during the round | Boss | Verdant Leaf |
/// | `on_disable` | the boss is being disabled mid-round (still in effect during the call) | Boss | Water, Needle (give back) |
#[derive(Debug, Clone, Copy)]
pub struct BossHooks {
    /// The boss was chosen for a new ante.
    pub on_ante_start: Option<BossFn>,
    /// The boss blind was selected.
    pub on_blind_start: Option<BossFn>,
    /// Which cards the boss debuffs.
    pub debuffs_card: Option<DebuffFn>,
    /// Which drawn cards are face down.
    pub face_down: Option<FaceDownFn>,
    /// How many cards a draw takes.
    pub draw_count: Option<DrawCountFn>,
    /// After a draw.
    pub after_draw: Option<BossFn>,
    /// Before a played hand is scored.
    pub before_play: Option<PlayFn>,
    /// After a played hand, before the refill.
    pub after_play: Option<BossFn>,
    /// A joker was sold during the round.
    pub on_joker_sold: Option<BossFn>,
    /// The boss is being disabled mid-round.
    pub on_disable: Option<BossFn>,
}

impl BossHooks {
    /// No hooks.
    pub const NONE: BossHooks = BossHooks {
        on_ante_start: None,
        on_blind_start: None,
        debuffs_card: None,
        face_down: None,
        draw_count: None,
        after_draw: None,
        before_play: None,
        after_play: None,
        on_joker_sold: None,
        on_disable: None,
    };
}

/// Static definition of one boss blind. Build with [`BossDef::new`] and struct update.
#[derive(Debug, Clone, Copy)]
pub struct BossDef {
    /// Id (exactly one def per id).
    pub id: BossId,
    /// Display name.
    pub name: &'static str,
    /// Earliest ante it can appear at (bosses.md "Min ante"; Any = 1, Showdown = 8).
    pub min_ante: i32,
    /// Requirement multiplier (default x2; Wall x4, Violet Vessel x6, Needle x1).
    pub mult: f64,
    /// Keep `mult` when disabled (The Needle stays x1; Wall/Vessel fall back to x2).
    pub keep_mult_when_disabled: bool,
    /// Halve base chips and mult, rounding up (The Flint; scoring step 1).
    pub halve_base: bool,
    /// Hand size delta while in effect (The Manacle -1).
    pub hand_size: i32,
    /// Hooks.
    pub hooks: BossHooks,
}

impl BossDef {
    /// Def with x2 requirement, no static rule change, no hooks.
    pub const fn new(id: BossId, name: &'static str, min_ante: i32) -> Self {
        BossDef {
            id,
            name,
            min_ante,
            mult: MULT_BOSS_DEFAULT,
            keep_mult_when_disabled: false,
            halve_base: false,
            hand_size: 0,
            hooks: BossHooks::NONE,
        }
    }

    /// Showdown boss (`id.is_showdown()`).
    pub const fn showdown(&self) -> bool {
        self.id.is_showdown()
    }
}

// ---------------------------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------------------------

/// Context of [`BossFn`] hooks: read the whole state, write through the methods.
/// Randomness on `RngKey::Bosses`.
pub struct BossCtx<'a> {
    state: &'a mut GameState,
    disable: bool,
}

impl<'a> BossCtx<'a> {
    /// Build a context (game layer and tests).
    pub fn new(state: &'a mut GameState) -> Self {
        BossCtx {
            state,
            disable: false,
        }
    }

    /// The run (read-only).
    pub fn state(&self) -> &GameState {
        self.state
    }

    /// Boss bookkeeping.
    pub fn boss(&self) -> &BossState {
        &self.state.boss_state
    }

    /// Boss bookkeeping (write).
    pub fn boss_mut(&mut self) -> &mut BossState {
        &mut self.state.boss_state
    }

    /// Cards in hand.
    pub fn hand(&self) -> &[Card] {
        &self.state.deck.hand
    }

    /// Jokers.
    pub fn jokers(&self) -> &[JokerInstance] {
        &self.state.jokers
    }

    /// Set hands left this round (The Needle: 1).
    pub fn set_hands_left(&mut self, n: u32) {
        self.state.hands_left = n;
    }

    /// Set discards left this round (The Water: 0).
    pub fn set_discards_left(&mut self, n: u32) {
        self.state.discards_left = n;
    }

    /// Add hands this round (give back on disable).
    pub fn add_hands(&mut self, n: u32) {
        self.state.hands_left = self.state.hands_left.saturating_add(n);
    }

    /// Add discards this round (give back on disable).
    pub fn add_discards(&mut self, n: u32) {
        self.state.discards_left = self.state.discards_left.saturating_add(n);
    }

    /// Debuff the joker with `uid` (Crimson Heart), restoring the one debuffed before; `None`
    /// only restores. Modifiers are refreshed here.
    pub fn set_debuffed_joker(&mut self, uid: Option<u32>) {
        set_debuffed_joker(self.state, uid);
    }

    /// Shuffle the joker order (Amber Acorn), `RngKey::Bosses`.
    pub fn shuffle_jokers(&mut self) {
        let s = &mut *self.state;
        s.rng.shuffle(RngKey::Bosses, &mut s.jokers);
        s.refresh_modifiers();
    }

    /// Uniform integer in `[lo, hi_exclusive)` on `RngKey::Bosses`.
    pub fn rng_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        self.state.rng.range(RngKey::Bosses, lo, hi_exclusive)
    }

    /// Probability roll (Oops! All 6s applies) on `RngKey::Bosses`.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        let pm = self.state.mods.prob_mult;
        self.state
            .rng
            .chance(RngKey::Bosses, numerator, denominator, pm)
    }

    /// Ask the game layer to disable this boss after the hook (Verdant Leaf on a joker sale).
    pub fn disable(&mut self) {
        self.disable = true;
    }
}

/// Context of a [`FaceDownFn`].
pub struct DrawCtx<'a> {
    kind: DrawKind,
    rules: RuleFlags,
    prob_mult: u32,
    rng: &'a mut GameRng,
}

impl DrawCtx<'_> {
    /// Why cards are drawn.
    pub fn kind(&self) -> DrawKind {
        self.kind
    }

    /// Face card under the active rules (Pareidolia), ignoring debuffs (The Mark).
    pub fn is_face(&self, card: &Card) -> bool {
        let mut c = *card;
        c.debuffed = false;
        c.is_face(&self.rules)
    }

    /// Probability roll (Oops! All 6s applies, notes.md §4.1c) on `RngKey::Bosses`.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.rng
            .chance(RngKey::Bosses, numerator, denominator, self.prob_mult)
    }
}

/// Context of a [`PlayFn`]. Works on plain references so `Game::preview_play` can run the same
/// hook on copies. Counters already include the hand being played (hooks.md §5.3).
pub struct PlayCtx<'a> {
    played: &'a [Card],
    hand_type: HandType,
    held: &'a mut Vec<Card>,
    hooked: &'a mut Vec<Card>,
    money: &'a mut i32,
    hand_levels: &'a mut [u32; HandType::COUNT],
    hand_plays_round: &'a [u32; HandType::COUNT],
    boss: &'a mut BossState,
    rng: &'a mut GameRng,
}

impl PlayCtx<'_> {
    /// Played cards, left to right.
    pub fn played(&self) -> &[Card] {
        self.played
    }

    /// Evaluated hand type.
    pub fn hand_type(&self) -> HandType {
        self.hand_type
    }

    /// Cards still held in hand.
    pub fn held(&self) -> &[Card] {
        self.held
    }

    /// Money now.
    pub fn money(&self) -> i32 {
        *self.money
    }

    /// Times `ht` was played this round, including this hand.
    pub fn plays_round(&self, ht: HandType) -> u32 {
        self.hand_plays_round[ht.index()]
    }

    /// Level of `ht` now.
    pub fn level(&self, ht: HandType) -> u32 {
        self.hand_levels[ht.index()]
    }

    /// Boss bookkeeping (The Ox reads `hand`).
    pub fn boss(&self) -> &BossState {
        self.boss
    }

    /// Reject the hand (scores 0, only `on_invalid_hand` + `after_scoring` run). Also marks
    /// the boss as triggered (Matador).
    pub fn invalidate(&mut self) {
        self.boss.hand_invalid = true;
        self.boss.triggered = true;
    }

    /// The boss ability affected this hand (Matador).
    pub fn trigger(&mut self) {
        self.boss.triggered = true;
    }

    /// Gain or lose money (The Tooth: -1 per card; may go negative, notes.md §4.1c).
    pub fn add_money(&mut self, dollars: i32) {
        *self.money = self.money.saturating_add(dollars);
    }

    /// Set money (The Ox: $0).
    pub fn set_money(&mut self, dollars: i32) {
        *self.money = dollars;
    }

    /// Change a hand's level, never below 1 (The Arm: -1). Returns the levels actually changed.
    pub fn add_levels(&mut self, ht: HandType, delta: i32) -> i32 {
        let l = &mut self.hand_levels[ht.index()];
        let old = *l;
        *l = (i64::from(old) + i64::from(delta)).clamp(1, i64::from(u32::MAX)) as u32;
        (i64::from(*l) - i64::from(old)) as i32
    }

    /// Discard up to `n` random held cards (The Hook), `RngKey::Bosses`. The game moves them to
    /// the discard pile and fires a forced discard event (`DiscardInfo::forced`). Returns how
    /// many were discarded.
    pub fn discard_random_held(&mut self, n: usize) -> usize {
        let mut k = 0;
        while k < n && !self.held.is_empty() {
            let i = self.rng.range(RngKey::Bosses, 0, self.held.len());
            self.hooked.push(self.held.remove(i));
            k += 1;
        }
        k
    }
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

/// Choose the boss of `ante` (notes.md §4.1c): among DEFINED bosses of the right pool
/// (Showdown iff `ante > 0 && ante % interval == 0`; if that pool has no defined boss, the
/// other pool) whose `min_ante <= max(ante, 1)`, keep those seen the fewest times, pick one
/// uniformly on `RngKey::Bosses`, and count it as seen. `None` if no boss is defined.
pub fn choose_boss(
    ante: i32,
    showdown_interval: i32,
    seen: &mut [u32; BossId::COUNT],
    rng: &mut GameRng,
) -> Option<BossId> {
    choose_boss_excluding(ante, showdown_interval, seen, rng, None)
}

/// [`choose_boss`] that never returns `exclude` (boss rerolls). If `exclude` is the only
/// eligible boss it is returned anyway.
pub fn choose_boss_excluding(
    ante: i32,
    showdown_interval: i32,
    seen: &mut [u32; BossId::COUNT],
    rng: &mut GameRng,
    exclude: Option<BossId>,
) -> Option<BossId> {
    let showdown = ante > 0 && showdown_interval > 0 && ante % showdown_interval == 0;
    let eff_ante = ante.max(1);
    let eligible = |want_showdown: bool| {
        BossId::ALL.into_iter().filter(move |&id| {
            try_boss_def(id).is_some_and(|d| d.min_ante <= eff_ante)
                && id.is_showdown() == want_showdown
                && Some(id) != exclude
        })
    };
    let pool_showdown = if eligible(showdown).next().is_some() {
        showdown
    } else if eligible(!showdown).next().is_some() {
        !showdown
    } else {
        return exclude.filter(|&id| try_boss_def(id).is_some());
    };
    let min_seen = eligible(pool_showdown)
        .map(|id| seen[id.index()])
        .min()
        .unwrap_or(0);
    let least = |id: &BossId| seen[id.index()] == min_seen;
    let n = eligible(pool_showdown).filter(least).count();
    let k = rng.range(RngKey::Bosses, 0, n);
    let id = eligible(pool_showdown).filter(least).nth(k)?;
    seen[id.index()] += 1;
    Some(id)
}

// ---------------------------------------------------------------------------------------------
// Game-layer glue (called by `state.rs` / `game.rs`; content never calls these)
// ---------------------------------------------------------------------------------------------

/// Run a [`BossFn`] and apply a requested disable.
fn run_hook(s: &mut GameState, f: BossFn) {
    let mut ctx = BossCtx::new(s);
    f(&mut ctx);
    if ctx.disable {
        disable(s);
    }
}

fn active_def(s: &GameState) -> Option<&'static BossDef> {
    if s.boss_in_effect() {
        s.boss.and_then(try_boss_def)
    } else {
        None
    }
}

/// Reroll the current ante's boss (Boss tag, Director's Cut, Retcon): a different boss from the
/// same pool (the old one counts as unseen again), boss state reset, requirement recomputed.
/// Only meant for the phases between rounds. Returns true if the boss changed.
pub(crate) fn reroll_boss(s: &mut GameState) -> bool {
    let Some(old) = s.boss else {
        return false;
    };
    let seen = &mut s.boss_seen[old.index()];
    *seen = seen.saturating_sub(1);
    let new = choose_boss_excluding(
        s.ante,
        crate::data::run_rules::SHOWDOWN_ANTE_INTERVAL,
        &mut s.boss_seen,
        &mut s.rng,
        Some(old),
    )
    .unwrap_or(old);
    s.boss = Some(new);
    s.boss_disabled = false;
    ante_start(s);
    s.refresh_target();
    new != old
}

/// Ante start: reset the boss state and run `on_ante_start` of the chosen boss (it runs even
/// though no round is being played).
pub(crate) fn ante_start(s: &mut GameState) {
    s.boss_state = BossState::default();
    if let Some(f) = s
        .boss
        .and_then(try_boss_def)
        .and_then(|d| d.hooks.on_ante_start)
    {
        run_hook(s, f);
    }
}

/// Blind selected (phase already `Play`, deck reset, round counters set): reset the per-round
/// state, run `on_blind_start`, recompute debuffs.
pub(crate) fn blind_start(s: &mut GameState) {
    let b = &mut s.boss_state;
    b.card_uid = None;
    b.joker_uid = None;
    b.removed = 0;
    b.jokers_hidden = false;
    b.draws = 0;
    b.triggered = false;
    b.hand_invalid = false;
    if let Some(f) = active_def(s).and_then(|d| d.hooks.on_blind_start) {
        run_hook(s, f);
    }
    refresh_debuffs(s);
}

/// Recompute every card's boss debuff (all piles). Cheap; the game calls it after every
/// Play-phase action.
pub(crate) fn refresh_debuffs(s: &mut GameState) {
    let pred = active_def(s).and_then(|d| d.hooks.debuffs_card);
    let rules = s.mods.rules;
    let d = &mut s.deck;
    for c in d
        .draw_pile
        .iter_mut()
        .chain(d.hand.iter_mut())
        .chain(d.played.iter_mut())
        .chain(d.discard_pile.iter_mut())
    {
        c.debuffed = match pred {
            Some(f) => {
                let mut probe = *c;
                probe.debuffed = false;
                f(&probe, &rules)
            }
            None => false,
        };
    }
}

/// Draw for `kind`: the boss may change the count (Serpent) and flip cards face down as they
/// are drawn (rolls in draw order, top card first); then `after_draw` runs. Returns the number
/// of cards drawn.
pub(crate) fn draw(s: &mut GameState, kind: DrawKind) -> usize {
    let def = active_def(s);
    let max_hand = crate::data::run_rules::MAX_HAND_SIZE as usize;
    let default = (s.hand_size() as usize).saturating_sub(s.deck.hand.len());
    let mut n = match def.and_then(|d| d.hooks.draw_count) {
        Some(f) => f(kind, default).min(max_hand.saturating_sub(s.deck.hand.len())),
        None => default,
    };
    n = n.min(s.deck.draw_pile.len());
    if let Some(f) = def.and_then(|d| d.hooks.face_down) {
        let (rules, prob_mult) = (s.mods.rules, s.mods.prob_mult);
        let pile_len = s.deck.draw_pile.len();
        let mut ctx = DrawCtx {
            kind,
            rules,
            prob_mult,
            rng: &mut s.rng,
        };
        for c in s.deck.draw_pile[pile_len - n..].iter_mut().rev() {
            c.face_down = f(&mut ctx, c);
        }
    }
    let drawn = s.deck.draw(n);
    s.boss_state.draws += 1;
    if let Some(f) = def.and_then(|d| d.hooks.after_draw) {
        run_hook(s, f);
    }
    drawn
}

/// Reset the per-hand flags and run `before_play` on the given pieces. Shared by `Game::play`
/// (real state) and `Game::preview_play` (copies). Also sets `triggered` when a played card is
/// debuffed by the boss or the boss halves the base (Matador, hooks.md §15).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pre_play(
    boss: Option<BossId>,
    played: &[Card],
    hand_type: HandType,
    held: &mut Vec<Card>,
    hooked: &mut Vec<Card>,
    money: &mut i32,
    hand_levels: &mut [u32; HandType::COUNT],
    hand_plays_round: &[u32; HandType::COUNT],
    state: &mut BossState,
    rng: &mut GameRng,
) {
    state.triggered = false;
    state.hand_invalid = false;
    let Some(def) = boss.and_then(try_boss_def) else {
        return;
    };
    if def.halve_base || (def.hooks.debuffs_card.is_some() && played.iter().any(|c| c.debuffed)) {
        state.triggered = true;
    }
    if let Some(f) = def.hooks.before_play {
        let mut ctx = PlayCtx {
            played,
            hand_type,
            held,
            hooked,
            money,
            hand_levels,
            hand_plays_round,
            boss: state,
            rng,
        };
        f(&mut ctx);
    }
}

/// After a played hand (scored, effects applied), before the refill.
pub(crate) fn after_play(s: &mut GameState) {
    if let Some(f) = active_def(s).and_then(|d| d.hooks.after_play) {
        run_hook(s, f);
    }
}

/// A joker was sold (call after it was removed).
pub(crate) fn joker_sold(s: &mut GameState) {
    if let Some(f) = active_def(s).and_then(|d| d.hooks.on_joker_sold) {
        run_hook(s, f);
    }
}

/// Disable the current boss (Chicot, Luchador, Verdant Leaf). While a round is played the boss
/// first runs `on_disable`; then generic clean-up: joker debuff restored, forced card cleared,
/// jokers revealed, cards un-flipped, debuffs and the requirement recomputed. Outside a round it
/// only sets the flag. Idempotent.
pub(crate) fn disable(s: &mut GameState) {
    if s.boss_disabled {
        return;
    }
    if let Some(f) = active_def(s).and_then(|d| d.hooks.on_disable) {
        let mut ctx = BossCtx::new(s);
        f(&mut ctx);
    }
    s.boss_disabled = true;
    set_debuffed_joker(s, None);
    s.boss_state.card_uid = None;
    s.boss_state.jokers_hidden = false;
    if s.phase == Phase::Play {
        for c in &mut s.deck.hand {
            c.face_down = false;
        }
    }
    refresh_debuffs(s);
    s.refresh_target();
}

/// Round over (won or saved): restore the boss's joker debuff and clear per-round flags.
pub(crate) fn round_end(s: &mut GameState) {
    set_debuffed_joker(s, None);
    s.boss_state.card_uid = None;
    s.boss_state.jokers_hidden = false;
}

fn set_debuffed_joker(s: &mut GameState, uid: Option<u32>) {
    if let Some(old) = s.boss_state.joker_uid.take()
        && let Some(j) = s.jokers.iter_mut().find(|j| j.uid == old)
    {
        // An expired Perishable joker stays debuffed.
        j.debuffed = j.stickers.perishable && j.stickers.perish_rounds_left == 0;
    }
    if let Some(new) = uid
        && let Some(j) = s.jokers.iter_mut().find(|j| j.uid == new)
    {
        j.debuffed = true;
        s.boss_state.joker_uid = Some(new);
    }
    s.refresh_modifiers();
}

/// Hand position of the force-selected card (Cerulean Bell), if the boss is in effect and the
/// card is in hand.
pub fn forced_position(s: &GameState) -> Option<usize> {
    if !s.boss_in_effect() {
        return None;
    }
    let uid = s.boss_state.card_uid?;
    s.deck.hand.iter().position(|c| c.uid == uid)
}

/// Definition of the current boss (any phase, disabled or not).
pub fn current_def(s: &GameState) -> Option<&'static BossDef> {
    s.boss.and_then(try_boss_def)
}
