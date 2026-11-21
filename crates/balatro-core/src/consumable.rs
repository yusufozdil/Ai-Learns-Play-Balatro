//! Consumables (Tarot / Planet / Spectral): the per-run [`ConsumableInstance`], the static
//! [`ConsumableDef`] (targets, usability, effect), and the contexts an effect is written
//! against: [`UseCheck`] (pure usability test) and [`UseCtx`] (applying the effect).
//!
//! Content guide: `docs/design/consumables_bosses.md`. Ids live in
//! [`crate::content::ids::ConsumableId`], definitions in `content/{planets,tarots,spectrals}.rs`,
//! the registry in [`crate::content`]. The game layer (`game.rs`) validates, removes the card
//! from its slot, runs the effect, applies the queued [`SideEffect`]s, updates the run counters
//! and finally fires `hooks::fire_consumable_used`.

use crate::card::Card;
use crate::content::ids::ConsumableId;
use crate::content::try_consumable_def;
use crate::data::prices::{PLANET, SPECTRAL, TAROT};
use crate::data::run_rules::MAX_SELECTABLE_HAND;
use crate::deck::Pile;
use crate::economy;
use crate::enums::{Edition, Enhancement, HandType, Rank, Rarity, Seal, Suit};
use crate::hooks::SideEffect;
use crate::joker::JokerInstance;
use crate::rng::{GameRng, RngKey};
use crate::state::GameState;
use serde::{Deserialize, Serialize};

/// Kind of consumable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsumableKind {
    /// Tarot card.
    Tarot,
    /// Planet card.
    Planet,
    /// Spectral card.
    Spectral,
}

impl ConsumableKind {
    /// Base shop price (consumables.md: Tarot $3, Planet $3, Spectral $4; The Soul and Black
    /// Hole use the Spectral price).
    pub const fn base_cost(self) -> i32 {
        match self {
            ConsumableKind::Tarot => TAROT,
            ConsumableKind::Planet => PLANET,
            ConsumableKind::Spectral => SPECTRAL,
        }
    }
}

/// One held consumable. `Copy`, 8 bytes. This is also what hooks see
/// (`hooks::ConsumableView` is an alias).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumableInstance {
    /// Which card.
    pub id: ConsumableId,
    /// Negative edition: +1 consumable slot while held (the only edition a consumable can have).
    pub negative: bool,
    /// Current sell value in dollars (Gift Card raises it).
    pub sell_value: i32,
}

impl ConsumableInstance {
    /// A fresh, non-Negative card with the default sell value `max(1, floor(cost / 2))`.
    pub fn new(id: ConsumableId) -> Self {
        ConsumableInstance {
            id,
            negative: false,
            sell_value: economy::sell_value(id.kind().base_cost(), 0),
        }
    }

    /// Builder: set the Negative edition.
    pub fn with_negative(mut self, negative: bool) -> Self {
        self.negative = negative;
        self
    }

    /// Tarot / Planet / Spectral.
    pub fn kind(&self) -> ConsumableKind {
        self.id.kind()
    }

    /// For Planets: the hand type it levels (Observatory, Constellation).
    pub fn planet_hand(&self) -> Option<HandType> {
        self.id.planet_hand()
    }

    /// The definition, `None` while no content file defines this id.
    pub fn def(&self) -> Option<&'static ConsumableDef> {
        try_consumable_def(self.id)
    }
}

// ---------------------------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------------------------

/// Pure usability test (beyond the generic target-count and hand checks). Must not mutate or
/// roll: it runs inside `actions::validate` and `legal_actions`.
pub type CanUseFn = fn(c: &UseCheck<'_>) -> bool;
/// The effect. Writes through [`UseCtx`] only.
pub type EffectFn = fn(ctx: &mut UseCtx<'_>);

/// Static definition of one consumable. Build with [`ConsumableDef::new`] and struct update:
///
/// ```ignore
/// ConsumableDef {
///     min_targets: 1,
///     max_targets: 2,
///     ..ConsumableDef::new(ConsumableId::TheMagician, "The Magician", magician)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConsumableDef {
    /// Id (exactly one def per id across all `DEFS` tables).
    pub id: ConsumableId,
    /// Display name (English, for logs/debug).
    pub name: &'static str,
    /// Fewest selected hand cards (0 = untargeted).
    pub min_targets: u8,
    /// Most selected hand cards (0 = untargeted: `targets` must be 0). At most
    /// [`MAX_CONSUMABLE_TARGETS`].
    pub max_targets: u8,
    /// Needs a drawn hand even without targets (Sigil, Ouija, Familiar, Grim, Incantation,
    /// Immolate). Implied when `max_targets > 0`. Such cards are usable only while a hand is
    /// dealt: `Phase::Play` and Arcana/Spectral packs.
    pub needs_hand: bool,
    /// Extra usability condition (free slot, eligible joker, money, ...). `None` = always.
    pub can_use: Option<CanUseFn>,
    /// The effect.
    pub effect: EffectFn,
}

/// Upper bound for `ConsumableDef::max_targets` (the base game's maximum is 3). Keeps
/// `legal_actions` bounded: per targeted card, all `C(n,k)` selections for `k` in
/// `min..=max` (hand of 8, 1-3 targets: 92 masks).
pub const MAX_CONSUMABLE_TARGETS: u8 = 3;

impl ConsumableDef {
    /// Untargeted, always usable (unless `can_use` is set), no hand needed.
    pub const fn new(id: ConsumableId, name: &'static str, effect: EffectFn) -> Self {
        ConsumableDef {
            id,
            name,
            min_targets: 0,
            max_targets: 0,
            needs_hand: false,
            can_use: None,
            effect,
        }
    }

    /// Kind of the card.
    pub const fn kind(&self) -> ConsumableKind {
        self.id.kind()
    }

    /// Usable only while a hand is drawn.
    pub const fn requires_hand(&self) -> bool {
        self.needs_hand || self.max_targets > 0
    }
}

// ---------------------------------------------------------------------------------------------
// Targets and the usability check
// ---------------------------------------------------------------------------------------------

/// The selected hand cards of a use (bitmask over hand positions; bit i = hand card i).
#[derive(Debug, Clone, Copy)]
pub struct Targets<'a> {
    hand: &'a [Card],
    mask: u32,
}

impl<'a> Targets<'a> {
    /// Selection `mask` over `hand` (bits past the hand are ignored).
    pub fn new(hand: &'a [Card], mask: u32) -> Self {
        Targets { hand, mask }
    }

    /// The raw mask.
    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// Number of selected cards that exist in the hand.
    pub fn len(&self) -> usize {
        self.positions().count()
    }

    /// No card selected.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Selected hand positions, left to right.
    pub fn positions(&self) -> impl Iterator<Item = usize> + 'a {
        let (mask, n) = (self.mask, self.hand.len().min(MAX_SELECTABLE_HAND));
        (0..n).filter(move |&i| (mask >> i) & 1 == 1)
    }

    /// Selected cards, left to right.
    pub fn cards(&self) -> impl Iterator<Item = &'a Card> + 'a {
        let hand = self.hand;
        self.positions().map(move |i| &hand[i])
    }

    /// The `k`-th selected card (0 = leftmost).
    pub fn get(&self, k: usize) -> Option<&'a Card> {
        self.cards().nth(k)
    }
}

/// What a [`CanUseFn`] sees: the state before the use, the selection, and the slot the card is
/// used from (`None` for a card picked from a pack or bought with Buy & Use).
pub struct UseCheck<'a> {
    state: &'a GameState,
    targets: Targets<'a>,
    slot: Option<usize>,
}

impl<'a> UseCheck<'a> {
    /// Build a check (the game layer and tests).
    pub fn new(state: &'a GameState, targets: u32, slot: Option<usize>) -> Self {
        UseCheck {
            state,
            targets: Targets::new(&state.deck.hand, targets),
            slot,
        }
    }

    /// The run, unchanged.
    pub fn state(&self) -> &'a GameState {
        self.state
    }

    /// Selected hand cards.
    pub fn targets(&self) -> Targets<'a> {
        self.targets
    }

    /// Cards in hand (empty outside a round).
    pub fn hand(&self) -> &'a [Card] {
        &self.state.deck.hand
    }

    /// Jokers.
    pub fn jokers(&self) -> &'a [JokerInstance] {
        &self.state.jokers
    }

    /// Free consumable slots once the used card has left its slot (The Fool, High Priestess,
    /// Emperor: a full area still has room for one card after using one of its cards).
    pub fn free_consumable_slots(&self) -> u32 {
        let s = self.state;
        let (mut len, mut cap) = (s.consumables.len() as u32, s.consumable_capacity());
        if let Some(c) = self.slot.and_then(|i| s.consumables.get(i)) {
            len -= 1;
            if c.negative {
                cap = cap.saturating_sub(1);
            }
        }
        cap.saturating_sub(len)
    }

    /// Free joker slots (Judgement, Wraith, The Soul, Ankh).
    pub fn free_joker_slots(&self) -> u32 {
        let s = self.state;
        s.joker_capacity().saturating_sub(s.jokers.len() as u32)
    }

    /// Times `id` was used this run before this use.
    pub fn uses_of(&self, id: ConsumableId) -> u32 {
        self.state.consumable_uses(id)
    }
}

// ---------------------------------------------------------------------------------------------
// Effect context
// ---------------------------------------------------------------------------------------------

/// What an [`EffectFn`] writes through.
///
/// **Immediate** (applied to the state at once): selected/held card fields (`set_*`,
/// `edit_card`, `copy_card`), money, permanent hand size, joker editions.
/// **Deferred** (queued [`SideEffect`]s, applied in order after the effect, with follow-up joker
/// events `on_cards_destroyed` / `on_cards_added` / `on_acquire`): destroying/adding playing
/// cards, hand levels, creating/destroying/copying jokers, creating consumables.
/// Creations that need a slot are dropped by the game layer when there is no room.
///
/// Randomness: only `chance` / `rng_range` / `weighted_index` (all on `RngKey::Consumables`).
pub struct UseCtx<'a> {
    state: &'a mut GameState,
    fx: &'a mut Vec<SideEffect>,
    used: ConsumableInstance,
    mask: u32,
}

impl<'a> UseCtx<'a> {
    /// Build a context (the game layer and tests). `targets` = selection mask over the hand.
    pub fn new(
        state: &'a mut GameState,
        fx: &'a mut Vec<SideEffect>,
        used: ConsumableInstance,
        targets: u32,
    ) -> Self {
        UseCtx {
            state,
            fx,
            used,
            mask: targets,
        }
    }

    /// The run (read-only; the used card is already out of its slot).
    pub fn state(&self) -> &GameState {
        self.state
    }

    /// The card being used.
    pub fn used(&self) -> ConsumableInstance {
        self.used
    }

    /// Cards in hand, left to right.
    pub fn hand(&self) -> &[Card] {
        &self.state.deck.hand
    }

    /// Jokers, left to right.
    pub fn jokers(&self) -> &[JokerInstance] {
        &self.state.jokers
    }

    /// Selected hand positions, left to right (positions stay valid during the effect: deferred
    /// additions/destructions happen afterwards).
    pub fn targets(&self) -> Targets<'_> {
        Targets::new(&self.state.deck.hand, self.mask)
    }

    /// Hand position of the `k`-th selected card (0 = leftmost).
    pub fn target_pos(&self, k: usize) -> Option<usize> {
        self.targets().positions().nth(k)
    }

    /// Number of selected cards.
    pub fn target_count(&self) -> usize {
        self.targets().len()
    }

    /// Times `id` was used this run before this use (Ectoplasm's growing penalty).
    pub fn uses_of(&self, id: ConsumableId) -> u32 {
        self.state.consumable_uses(id)
    }

    /// Free consumable slots now (the used card has already left its slot): High Priestess /
    /// Emperor create `min(2, free)`.
    pub fn free_consumable_slots(&self) -> u32 {
        let s = &*self.state;
        s.consumable_capacity()
            .saturating_sub(s.consumables.len() as u32)
    }

    /// Free joker slots now.
    pub fn free_joker_slots(&self) -> u32 {
        let s = &*self.state;
        s.joker_capacity().saturating_sub(s.jokers.len() as u32)
    }

    // ----- immediate: playing cards in hand ---------------------------------------------------

    /// Change the hand card at `pos` in place. `uid` is kept whatever `f` does; debuffs are
    /// recomputed by the game layer after the use. No-op for a bad position.
    pub fn edit_card(&mut self, pos: usize, f: impl FnOnce(&mut Card)) {
        if let Some(c) = self.state.deck.hand.get_mut(pos) {
            let uid = c.uid;
            f(c);
            c.uid = uid;
        }
    }

    /// Set the enhancement (Magician, Empress, Hierophant, Lovers, Chariot, Justice, Devil,
    /// Tower). Replaces any existing enhancement.
    pub fn set_enhancement(&mut self, pos: usize, e: Enhancement) {
        self.edit_card(pos, |c| c.enhancement = e);
    }

    /// Set the printed suit (Star, Moon, Sun, World, Sigil).
    pub fn set_suit(&mut self, pos: usize, s: Suit) {
        self.edit_card(pos, |c| c.suit = s);
    }

    /// Set the printed rank (Strength, Ouija).
    pub fn set_rank(&mut self, pos: usize, r: Rank) {
        self.edit_card(pos, |c| c.rank = r);
    }

    /// Set the seal (Talisman, Deja Vu, Trance, Medium). Replaces any existing seal.
    pub fn set_seal(&mut self, pos: usize, s: Seal) {
        self.edit_card(pos, |c| c.seal = s);
    }

    /// Set the edition (Aura).
    pub fn set_edition(&mut self, pos: usize, e: Edition) {
        self.edit_card(pos, |c| c.edition = e);
    }

    /// The card at `dst` becomes a copy of the card at `src`: rank, suit, enhancement, edition,
    /// seal and `perma_bonus` (Death). `uid` and the round flags of `dst`
    /// (debuff, face-down, played-this-ante) are kept.
    pub fn copy_card(&mut self, dst: usize, src: usize) {
        let Some(&from) = self.state.deck.hand.get(src) else {
            return;
        };
        self.edit_card(dst, |c| {
            c.rank = from.rank;
            c.suit = from.suit;
            c.enhancement = from.enhancement;
            c.edition = from.edition;
            c.seal = from.seal;
            c.perma_bonus = from.perma_bonus;
        });
    }

    // ----- deferred: playing cards --------------------------------------------------------------

    /// Destroy the hand card at `pos` (Hanged Man, Familiar, Immolate). Queued `DestroyCard`.
    pub fn destroy_card(&mut self, pos: usize) {
        if let Some(c) = self.state.deck.hand.get(pos) {
            self.fx.push(SideEffect::DestroyCard { uid: c.uid });
        }
    }

    /// Add a playing card (Familiar/Grim/Incantation/Cryptid: `Pile::Hand`). The uid is
    /// reassigned. Queued `AddCard`; the hand may exceed the hand size.
    pub fn add_card(&mut self, card: Card, to: Pile) {
        self.fx.push(SideEffect::AddCard { card, to });
    }

    // ----- money, hands, hand size ----------------------------------------------------------------

    /// Current money.
    pub fn money(&self) -> i32 {
        self.state.money
    }

    /// Gain (or lose) money (Hermit, Temperance, Immolate).
    pub fn add_money(&mut self, dollars: i32) {
        self.state.money = self.state.money.saturating_add(dollars);
    }

    /// Set money (Wraith: $0).
    pub fn set_money(&mut self, dollars: i32) {
        self.state.money = dollars;
    }

    /// Raise (or lower) a hand's level. Queued `LevelUpHand` (floor level 1).
    pub fn level_up(&mut self, hand: HandType, levels: i32) {
        self.fx.push(SideEffect::LevelUpHand { hand, levels });
    }

    /// Permanently change the base hand size (Ouija -1, Ectoplasm -n). Clamped at 0 (the
    /// effective size is clamped to `MIN_HAND_SIZE..=MAX_HAND_SIZE`).
    pub fn add_hand_size(&mut self, delta: i32) {
        let v = i64::from(self.state.base_hand_size) + i64::from(delta);
        self.state.base_hand_size = v.clamp(0, i64::from(u32::MAX)) as u32;
    }

    // ----- jokers -------------------------------------------------------------------------------

    /// Set a joker's edition (Wheel of Fortune, Ectoplasm, Hex). A Negative edition adds a joker
    /// slot. Modifiers are refreshed by the game layer.
    pub fn set_joker_edition(&mut self, slot: usize, e: Edition) {
        if let Some(j) = self.state.jokers.get_mut(slot) {
            j.edition = e;
        }
    }

    /// Destroy the joker at `slot` (Ankh, Hex). Eternal jokers are refused (returns false, no
    /// effect queued). Queued `DestroyJoker`.
    pub fn destroy_joker(&mut self, slot: usize) -> bool {
        match self.state.jokers.get(slot) {
            Some(j) if !j.stickers.eternal => {
                self.fx.push(SideEffect::DestroyJoker { uid: j.uid });
                true
            }
            _ => false,
        }
    }

    /// Copy the joker at `slot` (Ankh: `strip_negative = true`). Queued `CopyJoker`; needs a
    /// slot at application time.
    pub fn copy_joker(&mut self, slot: usize, strip_negative: bool) {
        if let Some(j) = self.state.jokers.get(slot) {
            self.fx.push(SideEffect::CopyJoker {
                uid: j.uid,
                strip_negative,
            });
        }
    }

    /// Create a random joker (Judgement: `None` = shop odds, Wraith: Rare, The Soul:
    /// Legendary). Queued `CreateJoker`; needs a free joker slot when applied (shop pools).
    pub fn create_joker(&mut self, rarity: Option<Rarity>) {
        self.fx.push(SideEffect::CreateJoker { rarity });
    }

    // ----- consumables ------------------------------------------------------------------------

    /// Create a random consumable of `kind` (High Priestess, Emperor). Queued; pool rules in
    /// [`random_consumable`].
    pub fn create_consumable(&mut self, kind: ConsumableKind) {
        self.fx.push(SideEffect::CreateConsumable {
            kind,
            negative: false,
        });
    }

    /// Create a specific consumable (The Fool). Queued `AddConsumable`.
    pub fn add_consumable(&mut self, id: ConsumableId) {
        self.fx.push(SideEffect::AddConsumable {
            id,
            negative: false,
        });
    }

    /// Queue any other side effect.
    pub fn push_effect(&mut self, e: SideEffect) {
        self.fx.push(e);
    }

    // ----- randomness ---------------------------------------------------------------------------

    /// Probability roll with the run's probability multiplier (Oops! All 6s), on
    /// `RngKey::Consumables`. Always consumes one roll.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        let pm = self.state.mods.prob_mult;
        self.state
            .rng
            .chance(RngKey::Consumables, numerator, denominator, pm)
    }

    /// Uniform integer in `[lo, hi_exclusive)` on `RngKey::Consumables`.
    pub fn rng_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        self.state.rng.range(RngKey::Consumables, lo, hi_exclusive)
    }

    /// Weighted index on `RngKey::Consumables` (`None` if no positive weight).
    pub fn weighted_index(&mut self, weights: &[f64]) -> Option<usize> {
        self.state.rng.weighted_index(RngKey::Consumables, weights)
    }
}

// ---------------------------------------------------------------------------------------------
// Generation pools
// ---------------------------------------------------------------------------------------------

/// Fallback when every candidate of a kind is excluded (consumables.md §5: Tarot -> Strength,
/// Planet -> Pluto, Spectral -> Incantation).
pub const fn fallback(kind: ConsumableKind) -> ConsumableId {
    match kind {
        ConsumableKind::Tarot => ConsumableId::Strength,
        ConsumableKind::Planet => ConsumableId::Pluto,
        ConsumableKind::Spectral => ConsumableId::Incantation,
    }
}

/// Part of the random pool of its kind: not The Soul / Black Hole (packs only), and a secret
/// Planet only once its hand was played this run (consumables.md §1/§5).
pub fn in_random_pool(id: ConsumableId, hand_plays_run: &[u32; HandType::COUNT]) -> bool {
    match id {
        ConsumableId::TheSoul | ConsumableId::BlackHole => false,
        ConsumableId::PlanetX | ConsumableId::Ceres | ConsumableId::Eris => id
            .planet_hand()
            .is_some_and(|h| hand_plays_run[h.index()] > 0),
        _ => true,
    }
}

/// A random consumable of `kind` for effects that create one (8 Ball, Cartomancer, Séance,
/// Purple Seal, High Priestess, Emperor, ...): uniform over [`in_random_pool`], excluding ids
/// currently held unless `allow_duplicates` (Showman); [`fallback`] if nothing is left. One
/// roll on `RngKey::Consumables` when the pool is non-empty.
pub fn random_consumable(
    kind: ConsumableKind,
    held: &[ConsumableInstance],
    hand_plays_run: &[u32; HandType::COUNT],
    allow_duplicates: bool,
    rng: &mut GameRng,
) -> ConsumableId {
    let ok = |id: &&ConsumableId| {
        in_random_pool(**id, hand_plays_run)
            && (allow_duplicates || !held.iter().any(|c| c.id == **id))
    };
    let pool = ConsumableId::of_kind(kind);
    let n = pool.iter().filter(ok).count();
    if n == 0 {
        return fallback(kind);
    }
    let k = rng.range(RngKey::Consumables, 0, n);
    pool.iter()
        .filter(ok)
        .nth(k)
        .copied()
        .unwrap_or(fallback(kind))
}

/// Like [`random_consumable`] but the exclusion list is explicit and the roll key is the
/// caller's (shop: `RngKey::Shop`, packs: `RngKey::Packs`). Uniform over [`in_random_pool`] ids
/// not in `excluded` (everything allowed when `allow_duplicates`); [`fallback`] if none is left.
/// One roll when the pool is non-empty.
pub fn random_consumable_excluding(
    kind: ConsumableKind,
    excluded: &[ConsumableId],
    hand_plays_run: &[u32; HandType::COUNT],
    allow_duplicates: bool,
    key: RngKey,
    rng: &mut GameRng,
) -> ConsumableId {
    let ok = |id: &&ConsumableId| {
        in_random_pool(**id, hand_plays_run) && (allow_duplicates || !excluded.contains(*id))
    };
    let pool = ConsumableId::of_kind(kind);
    let n = pool.iter().filter(ok).count();
    if n == 0 {
        return fallback(kind);
    }
    let k = rng.range(key, 0, n);
    pool.iter()
        .filter(ok)
        .nth(k)
        .copied()
        .unwrap_or(fallback(kind))
}
