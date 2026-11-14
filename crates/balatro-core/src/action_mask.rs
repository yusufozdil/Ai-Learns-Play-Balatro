//! Structured action space for the RL env (Phase 8A): an action is `[kind, arg0, arg1, cards]`
//! (notes.md §3, format B) and the legal choices are described level by level instead of as a
//! flat list, so the model can pick the kind, then the argument, then the cards one at a time.
//!
//! ```text
//! kind              arg0                          arg1          cards
//! SelectBlind ..... -                             -             -
//! SkipBlind ....... -                             -             -
//! Play ............ -                             -             1..=5 hand cards
//! Discard ......... -                             -             1..=5 hand cards
//! SellJoker ....... joker slot                    -             -
//! EndShop ......... -                             -             -
//! UseConsumable ... consumable slot               -             0 or min..=max hand cards
//! SellConsumable .. consumable slot               -             -
//! MoveJoker ....... from joker slot               to joker slot -
//! Buy ............. shop card slot                -             -
//! BuyAndUse ....... shop card slot                -             -
//! OpenPack ........ shop pack slot                -             -
//! BuyVoucher ...... shop voucher slot             -             -
//! Reroll .......... -                             -             -
//! PickPackItem .... pack item                     -             0 or min..=max hand cards
//! SkipPack ........ -                             -             -
//! RerollBoss ...... -                             -             -
//! ```
//!
//! * [`structured_mask`] answers "which kinds, and for each kind which `arg0`" (and, for
//!   `MoveJoker`, [`StructuredMask::move_to`] answers `arg1`). It is one pass over the state:
//!   no allocation, a few dozen `validate` calls.
//! * [`card_select`] answers "which hand positions, how many, and which one is mandatory" for a
//!   `(kind, arg0)` that takes cards, as a [`CardSelect`] with the sequential-selection helpers
//!   [`CardSelect::next_candidates`] and [`CardSelect::can_finish`].
//! * The final action is built with [`Action::from_parts`] and goes through `validate` as always.
//!
//! **Consistency contract** (property-tested in `tests/action_mask.rs`): every action of
//! `legal_actions` is [`StructuredMask::permits`]-ed, and every action the masks permit passes
//! `validate`, with one documented exception class: rules that depend on card *combinations*
//! of a multi-card consumable target. None exists today (see the list below).
//!
//! # Rules the masks express only approximately
//!
//! 1. **Aura** (needs an edition-less card) and any other single-target consumable with a
//!    `can_use` check: expressed exactly, as a per-position filter of [`CardSelect::candidates`]
//!    (the single-card selection of each position is validated).
//! 2. A consumable with `min_targets >= 2` or `max_targets >= 2` **and** a card-dependent
//!    `can_use`: the mask would be a superset of the legal sets (all hand positions, count
//!    range only). No such card exists; the property test fails if one is added, which is the
//!    cue to extend [`card_select`]. The env must treat such an action as invalid (no-op plus a
//!    flag), exactly like any action rejected by `validate`.
//! 3. Hand positions >= 32 cannot be selected ([`MAX_SELECTABLE_HAND`]); a hand that long
//!    needs 16 cards added on top of the hand size and does not occur in practice.
//! 4. Consumables with `max_targets > 3` would be clamped to 3 (`MAX_CONSUMABLE_TARGETS`), as in
//!    `legal_actions` (no such card exists).
//! 5. Slots >= 256 do not exist (`u8` operands), same as `legal_actions`.
//!
//! Everything else (money, room, phase, eternal stickers, hands/discards left, boss reroll
//! limits, Cerulean Bell's forced card, Psychic/Hook are not legality rules) is state
//! dependent only and is exact.

use crate::actions::{Action, MaskIter, target_range, validate, validate_pick, validate_use_def};
use crate::boss;
use crate::consumable::ConsumableDef;
use crate::content::try_consumable_def;
use crate::data::run_rules::{MAX_SELECTABLE_HAND, MAX_SELECTED_CARDS};
use crate::packs::PackItem;
use crate::state::GameState;

/// The type of an [`Action`] (its variant, without operands). `repr(u8)` indices are the `kind`
/// values of the structured action and are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionKind {
    /// [`Action::SelectBlind`].
    SelectBlind = 0,
    /// [`Action::SkipBlind`].
    SkipBlind = 1,
    /// [`Action::Play`].
    Play = 2,
    /// [`Action::Discard`].
    Discard = 3,
    /// [`Action::SellJoker`].
    SellJoker = 4,
    /// [`Action::EndShop`].
    EndShop = 5,
    /// [`Action::UseConsumable`].
    UseConsumable = 6,
    /// [`Action::SellConsumable`].
    SellConsumable = 7,
    /// [`Action::MoveJoker`].
    MoveJoker = 8,
    /// [`Action::Buy`].
    Buy = 9,
    /// [`Action::BuyAndUse`].
    BuyAndUse = 10,
    /// [`Action::OpenPack`].
    OpenPack = 11,
    /// [`Action::BuyVoucher`].
    BuyVoucher = 12,
    /// [`Action::Reroll`].
    Reroll = 13,
    /// [`Action::PickPackItem`].
    PickPackItem = 14,
    /// [`Action::SkipPack`].
    SkipPack = 15,
    /// [`Action::RerollBoss`].
    RerollBoss = 16,
}

impl ActionKind {
    /// Every kind, in index order.
    pub const ALL: [ActionKind; 17] = [
        ActionKind::SelectBlind,
        ActionKind::SkipBlind,
        ActionKind::Play,
        ActionKind::Discard,
        ActionKind::SellJoker,
        ActionKind::EndShop,
        ActionKind::UseConsumable,
        ActionKind::SellConsumable,
        ActionKind::MoveJoker,
        ActionKind::Buy,
        ActionKind::BuyAndUse,
        ActionKind::OpenPack,
        ActionKind::BuyVoucher,
        ActionKind::Reroll,
        ActionKind::PickPackItem,
        ActionKind::SkipPack,
        ActionKind::RerollBoss,
    ];
    /// Number of kinds.
    pub const COUNT: usize = Self::ALL.len();

    /// The `kind` value of the structured action.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Kind from its index (`None` past the last).
    pub fn from_index(i: usize) -> Option<ActionKind> {
        Self::ALL.get(i).copied()
    }

    /// The kind takes an `arg0` (a slot, an item): everything but the operand-free actions and
    /// `Play`/`Discard`.
    pub const fn has_arg0(self) -> bool {
        matches!(
            self,
            ActionKind::SellJoker
                | ActionKind::UseConsumable
                | ActionKind::SellConsumable
                | ActionKind::MoveJoker
                | ActionKind::Buy
                | ActionKind::BuyAndUse
                | ActionKind::OpenPack
                | ActionKind::BuyVoucher
                | ActionKind::PickPackItem
        )
    }

    /// The kind takes an `arg1` (only `MoveJoker`: the destination slot).
    pub const fn has_arg1(self) -> bool {
        matches!(self, ActionKind::MoveJoker)
    }

    /// The kind can take hand cards (`Play`, `Discard` always; `UseConsumable` and
    /// `PickPackItem` for targeted cards; ask [`card_select`] per `arg0`).
    pub const fn may_take_cards(self) -> bool {
        matches!(
            self,
            ActionKind::Play
                | ActionKind::Discard
                | ActionKind::UseConsumable
                | ActionKind::PickPackItem
        )
    }
}

impl Action {
    /// The kind of this action.
    pub fn kind(&self) -> ActionKind {
        match self {
            Action::SelectBlind => ActionKind::SelectBlind,
            Action::SkipBlind => ActionKind::SkipBlind,
            Action::Play(_) => ActionKind::Play,
            Action::Discard(_) => ActionKind::Discard,
            Action::SellJoker(_) => ActionKind::SellJoker,
            Action::EndShop => ActionKind::EndShop,
            Action::UseConsumable { .. } => ActionKind::UseConsumable,
            Action::SellConsumable(_) => ActionKind::SellConsumable,
            Action::MoveJoker { .. } => ActionKind::MoveJoker,
            Action::Buy(_) => ActionKind::Buy,
            Action::BuyAndUse(_) => ActionKind::BuyAndUse,
            Action::OpenPack(_) => ActionKind::OpenPack,
            Action::BuyVoucher(_) => ActionKind::BuyVoucher,
            Action::Reroll => ActionKind::Reroll,
            Action::PickPackItem { .. } => ActionKind::PickPackItem,
            Action::SkipPack => ActionKind::SkipPack,
            Action::RerollBoss => ActionKind::RerollBoss,
        }
    }

    /// The structured form `(kind, arg0, arg1, cards)`. Unused operands are 0.
    pub fn parts(&self) -> (ActionKind, u8, u8, u32) {
        let kind = self.kind();
        match *self {
            Action::Play(m) | Action::Discard(m) => (kind, 0, 0, m),
            Action::SellJoker(i)
            | Action::SellConsumable(i)
            | Action::Buy(i)
            | Action::BuyAndUse(i)
            | Action::OpenPack(i)
            | Action::BuyVoucher(i) => (kind, i, 0, 0),
            Action::UseConsumable { slot, targets } => (kind, slot, 0, targets),
            Action::MoveJoker { from, to } => (kind, from, to, 0),
            Action::PickPackItem { index, targets } => (kind, index, 0, targets),
            Action::SelectBlind
            | Action::SkipBlind
            | Action::EndShop
            | Action::Reroll
            | Action::SkipPack
            | Action::RerollBoss => (kind, 0, 0, 0),
        }
    }

    /// Build the action of a structured choice; operands the kind does not use are ignored.
    /// Inverse of [`Action::parts`] on well-formed actions.
    pub fn from_parts(kind: ActionKind, arg0: u8, arg1: u8, cards: u32) -> Action {
        match kind {
            ActionKind::SelectBlind => Action::SelectBlind,
            ActionKind::SkipBlind => Action::SkipBlind,
            ActionKind::Play => Action::Play(cards),
            ActionKind::Discard => Action::Discard(cards),
            ActionKind::SellJoker => Action::SellJoker(arg0),
            ActionKind::EndShop => Action::EndShop,
            ActionKind::UseConsumable => Action::UseConsumable {
                slot: arg0,
                targets: cards,
            },
            ActionKind::SellConsumable => Action::SellConsumable(arg0),
            ActionKind::MoveJoker => Action::MoveJoker {
                from: arg0,
                to: arg1,
            },
            ActionKind::Buy => Action::Buy(arg0),
            ActionKind::BuyAndUse => Action::BuyAndUse(arg0),
            ActionKind::OpenPack => Action::OpenPack(arg0),
            ActionKind::BuyVoucher => Action::BuyVoucher(arg0),
            ActionKind::Reroll => Action::Reroll,
            ActionKind::PickPackItem => Action::PickPackItem {
                index: arg0,
                targets: cards,
            },
            ActionKind::SkipPack => Action::SkipPack,
            ActionKind::RerollBoss => Action::RerollBoss,
        }
    }
}

/// A set of [`ActionKind`]s (bit `kind.index()`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindSet(pub u32);

impl KindSet {
    /// The kind is in the set.
    #[inline]
    pub fn contains(self, k: ActionKind) -> bool {
        self.0 >> k.index() & 1 == 1
    }

    #[inline]
    fn insert(&mut self, k: ActionKind) {
        self.0 |= 1 << k.index();
    }

    /// Number of kinds.
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// No kind.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The kinds, in index order.
    pub fn iter(self) -> impl Iterator<Item = ActionKind> {
        ActionKind::ALL
            .into_iter()
            .filter(move |&k| self.contains(k))
    }
}

/// A set of slots/items `0..=255` (an `arg0` or `arg1` domain).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotSet(pub [u64; 4]);

impl SlotSet {
    /// The empty set.
    pub const EMPTY: SlotSet = SlotSet([0; 4]);

    /// The slot is in the set.
    #[inline]
    pub fn contains(&self, i: u8) -> bool {
        self.0[usize::from(i >> 6)] >> (i & 63) & 1 == 1
    }

    /// Add a slot.
    #[inline]
    pub fn insert(&mut self, i: u8) {
        self.0[usize::from(i >> 6)] |= 1 << (i & 63);
    }

    /// Number of slots.
    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    /// No slot.
    pub fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }

    /// The slots, ascending.
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u8).filter(move |&i| self.contains(i))
    }

    /// `{0}` (the `arg0` domain of a kind without operands).
    pub fn zero() -> SlotSet {
        let mut s = SlotSet::EMPTY;
        s.insert(0);
        s
    }

    /// Every slot in `0..n` (`n` capped at 256).
    pub fn below(n: usize) -> SlotSet {
        let mut s = SlotSet::EMPTY;
        for i in 0..n.min(256) {
            s.insert(i as u8);
        }
        s
    }
}

/// Which hand cards a `(kind, arg0)` may take: a set of positions to choose from, how many, and
/// which are mandatory. A selection `S` is legal iff [`CardSelect::is_valid_set`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardSelect {
    /// Hand positions (bit i = hand card i) that can be part of the selection.
    pub candidates: u32,
    /// Positions that must be part of it (Cerulean Bell's forced card); a subset of
    /// `candidates`.
    pub required: u32,
    /// Fewest cards (>= 1).
    pub min: u8,
    /// Most cards (<= the number of candidates).
    pub max: u8,
}

impl CardSelect {
    /// `set` is a legal complete selection.
    pub fn is_valid_set(&self, set: u32) -> bool {
        let k = set.count_ones();
        set & !self.candidates == 0
            && set & self.required == self.required
            && k >= u32::from(self.min)
            && k <= u32::from(self.max)
    }

    /// Positions that may be added to the partial selection `partial` (a subset of
    /// `candidates`) so that a legal completion still exists: none once `max` cards are
    /// selected; only the missing mandatory cards when they fill the remaining capacity.
    pub fn next_candidates(&self, partial: u32) -> u32 {
        let k = partial.count_ones();
        if k >= u32::from(self.max) {
            return 0;
        }
        let pool = self.candidates & !partial;
        let missing = self.required & !partial;
        if missing != 0 && u32::from(self.max) - k == missing.count_ones() {
            missing
        } else {
            pool
        }
    }

    /// The partial selection is already a legal complete one (the model may stop).
    pub fn can_finish(&self, partial: u32) -> bool {
        self.is_valid_set(partial)
    }

    /// Every legal selection, ascending, by enumeration (tests and small hands; use the
    /// sequential helpers in the env).
    pub fn iter_sets(&self) -> impl Iterator<Item = u32> + '_ {
        let n = 32 - self.candidates.leading_zeros() as usize;
        MaskIter::new(n, u32::from(self.max), self.required).filter(|&m| self.is_valid_set(m))
    }
}

/// Low `n` bits set (`n` capped at 32).
#[inline]
fn low_bits(n: usize) -> u32 {
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

/// The legal choices of the current state at the kind and operand level; see the module docs.
/// Built by [`structured_mask`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuredMask {
    /// Kinds with at least one legal action.
    pub kinds: KindSet,
    /// Legal `SellJoker` slots.
    pub sell_joker: SlotSet,
    /// Legal `MoveJoker` source slots (every slot when at least two jokers are owned).
    pub move_from: SlotSet,
    /// Number of jokers (a `MoveJoker` destination is any other slot below it).
    pub joker_count: u8,
    /// Legal `UseConsumable` slots (targeted ones: enough eligible cards exist).
    pub use_consumable: SlotSet,
    /// Legal `SellConsumable` slots.
    pub sell_consumable: SlotSet,
    /// Legal `Buy` shop card slots.
    pub buy: SlotSet,
    /// Legal `BuyAndUse` shop card slots.
    pub buy_and_use: SlotSet,
    /// Legal `OpenPack` shop pack slots.
    pub open_pack: SlotSet,
    /// Legal `BuyVoucher` shop voucher slots.
    pub buy_voucher: SlotSet,
    /// Legal `PickPackItem` items (targeted ones: enough eligible cards exist).
    pub pick_pack_item: SlotSet,
}

impl StructuredMask {
    /// The kind has a legal action.
    #[inline]
    pub fn allows_kind(&self, k: ActionKind) -> bool {
        self.kinds.contains(k)
    }

    /// Legal `arg0` values of `kind` (empty for an illegal kind). Kinds without an `arg0`
    /// return `{0}` so that a codec can treat every kind alike.
    pub fn arg0(&self, kind: ActionKind) -> SlotSet {
        if !self.allows_kind(kind) {
            return SlotSet::EMPTY;
        }
        match kind {
            ActionKind::SellJoker => self.sell_joker,
            ActionKind::MoveJoker => self.move_from,
            ActionKind::UseConsumable => self.use_consumable,
            ActionKind::SellConsumable => self.sell_consumable,
            ActionKind::Buy => self.buy,
            ActionKind::BuyAndUse => self.buy_and_use,
            ActionKind::OpenPack => self.open_pack,
            ActionKind::BuyVoucher => self.buy_voucher,
            ActionKind::PickPackItem => self.pick_pack_item,
            _ => SlotSet::zero(),
        }
    }

    /// Legal `MoveJoker` destinations for source slot `from`: every other joker slot.
    pub fn move_to(&self, from: u8) -> SlotSet {
        if !self.move_from.contains(from) {
            return SlotSet::EMPTY;
        }
        let mut s = SlotSet::below(usize::from(self.joker_count));
        s.0[usize::from(from >> 6)] &= !(1u64 << (from & 63));
        s
    }

    /// The structure of `action` is allowed: kind, `arg0`, `arg1` (`MoveJoker`) and the card
    /// set (`card_select` of this state; no cards where none are taken). Does not run
    /// `validate`; by the consistency contract `permits(a)` implies `validate(a).is_ok()`.
    pub fn permits(&self, state: &GameState, action: Action) -> bool {
        let (kind, a0, a1, cards) = action.parts();
        if !self.allows_kind(kind) {
            return false;
        }
        if kind.has_arg0() && !self.arg0(kind).contains(a0) {
            return false;
        }
        if kind.has_arg1() && !self.move_to(a0).contains(a1) {
            return false;
        }
        match card_select(state, kind, a0) {
            Some(sel) => sel.is_valid_set(cards),
            None => cards == 0,
        }
    }
}

/// The legal structure of the current state. Pure, allocation-free, one pass of `validate`
/// calls (no per-card-set enumeration).
pub fn structured_mask(state: &GameState) -> StructuredMask {
    let mut m = StructuredMask::default();
    if state.is_terminal() {
        return m;
    }
    let slots = |len: usize| len.min(256) as u16;
    // Phase actions without operands.
    for (kind, action) in [
        (ActionKind::SelectBlind, Action::SelectBlind),
        (ActionKind::SkipBlind, Action::SkipBlind),
        (ActionKind::EndShop, Action::EndShop),
        (ActionKind::Reroll, Action::Reroll),
        (ActionKind::SkipPack, Action::SkipPack),
        (ActionKind::RerollBoss, Action::RerollBoss),
    ] {
        if validate(state, action).is_ok() {
            m.kinds.insert(kind);
        }
    }
    // Play / Discard: the probe selection is the mandatory card, or card 0.
    let probe = boss::forced_position(state)
        .filter(|&p| p < MAX_SELECTABLE_HAND)
        .map_or(1u32, |p| 1u32 << p);
    if validate(state, Action::Play(probe)).is_ok() {
        m.kinds.insert(ActionKind::Play);
    }
    if validate(state, Action::Discard(probe)).is_ok() {
        m.kinds.insert(ActionKind::Discard);
    }
    // Jokers.
    for i in 0..slots(state.jokers.len()) {
        if validate(state, Action::SellJoker(i as u8)).is_ok() {
            m.sell_joker.insert(i as u8);
        }
    }
    // `legal_actions` caps joker operands at 255 slots for moves.
    let movable = state.jokers.len().min(usize::from(u8::MAX));
    m.joker_count = movable as u8;
    if movable >= 2 && validate(state, Action::MoveJoker { from: 0, to: 1 }).is_ok() {
        m.move_from = SlotSet::below(movable);
    }
    // Consumables held.
    for i in 0..slots(state.consumables.len()) {
        let slot = i as u8;
        if validate(state, Action::SellConsumable(slot)).is_ok() {
            m.sell_consumable.insert(slot);
        }
        if use_consumable_possible(state, slot) {
            m.use_consumable.insert(slot);
        }
    }
    // Market.
    for i in 0..slots(state.shop.cards.len()) {
        let slot = i as u8;
        if validate(state, Action::Buy(slot)).is_ok() {
            m.buy.insert(slot);
        }
        if validate(state, Action::BuyAndUse(slot)).is_ok() {
            m.buy_and_use.insert(slot);
        }
    }
    for i in 0..slots(state.shop.packs.len()) {
        if validate(state, Action::OpenPack(i as u8)).is_ok() {
            m.open_pack.insert(i as u8);
        }
    }
    for i in 0..slots(state.shop.vouchers.len()) {
        if validate(state, Action::BuyVoucher(i as u8)).is_ok() {
            m.buy_voucher.insert(i as u8);
        }
    }
    // Pack items.
    if let Some(pack) = &state.pack {
        for i in 0..slots(pack.items.len()) {
            if pick_possible(state, i as u8) {
                m.pick_pack_item.insert(i as u8);
            }
        }
    }
    for (kind, set) in [
        (ActionKind::SellJoker, &m.sell_joker),
        (ActionKind::MoveJoker, &m.move_from),
        (ActionKind::UseConsumable, &m.use_consumable),
        (ActionKind::SellConsumable, &m.sell_consumable),
        (ActionKind::Buy, &m.buy),
        (ActionKind::BuyAndUse, &m.buy_and_use),
        (ActionKind::OpenPack, &m.open_pack),
        (ActionKind::BuyVoucher, &m.buy_voucher),
        (ActionKind::PickPackItem, &m.pick_pack_item),
    ] {
        if !set.is_empty() {
            m.kinds.insert(kind);
        }
    }
    m
}

/// The hand cards a `(kind, arg0)` takes, or `None` when the kind takes no cards there (or the
/// choice is not legal at all: check [`StructuredMask::arg0`] first). `Play`/`Discard`: 1..=5
/// cards, the forced card (Cerulean Bell) mandatory. A targeted consumable (`UseConsumable`,
/// or a `PickPackItem` consumable): `min_targets..=max_targets` cards (at most 3) among the
/// eligible positions. Pure, allocation-free.
pub fn card_select(state: &GameState, kind: ActionKind, arg0: u8) -> Option<CardSelect> {
    match kind {
        ActionKind::Play | ActionKind::Discard => {
            let n = state.deck.hand.len().min(MAX_SELECTABLE_HAND);
            if n == 0 {
                return None;
            }
            let required = boss::forced_position(state)
                .filter(|&p| p < n)
                .map_or(0, |p| 1u32 << p);
            Some(CardSelect {
                candidates: low_bits(n),
                required,
                min: 1,
                max: MAX_SELECTED_CARDS.min(n as u32) as u8,
            })
        }
        ActionKind::UseConsumable => {
            let c = state.consumables.get(usize::from(arg0))?;
            let def = try_consumable_def(c.id)?;
            target_select(state, def, |mask| {
                validate_use_def(state, def, mask, Some(usize::from(arg0))).is_ok()
            })
        }
        ActionKind::PickPackItem => {
            let pack = state.pack.as_ref()?;
            let item = pack.items.get(usize::from(arg0))?.as_ref()?;
            let PackItem::Consumable(id) = item else {
                return None;
            };
            let def = try_consumable_def(*id)?;
            target_select(state, def, |mask| validate_pick(state, arg0, mask).is_ok())
        }
        _ => None,
    }
}

/// Card selection of a targeted consumable: `None` for an untargeted one. A single-target card
/// with a `can_use` check is filtered per position through `probe` (exact for Aura).
fn target_select(
    state: &GameState,
    def: &ConsumableDef,
    probe: impl Fn(u32) -> bool,
) -> Option<CardSelect> {
    if def.max_targets == 0 {
        return None;
    }
    let n = state.deck.hand.len().min(MAX_SELECTABLE_HAND);
    let (lo, hi) = target_range(def);
    let mut candidates = low_bits(n);
    if def.max_targets == 1 && def.can_use.is_some() {
        candidates = (0..n)
            .filter(|&p| probe(1u32 << p))
            .fold(0u32, |acc, p| acc | (1u32 << p));
    }
    let eligible = candidates.count_ones();
    let min = lo.max(1);
    if eligible < min {
        return None;
    }
    Some(CardSelect {
        candidates,
        required: 0,
        min: min as u8,
        max: hi.min(eligible) as u8,
    })
}

/// The smallest legal selection of `sel` (the lowest `min` candidates, plus the mandatory
/// cards): a witness that the choice has at least one legal card set.
fn witness(sel: &CardSelect) -> u32 {
    let mut set = sel.required;
    let mut rest = sel.candidates & !set;
    while set.count_ones() < u32::from(sel.min) && rest != 0 {
        let low = rest.isolate_lowest_one();
        set |= low;
        rest &= !low;
    }
    set
}

/// Some `UseConsumable` action with this slot is legal: an untargeted card validates with no
/// cards; a targeted one needs a card selection that validates (its witness).
fn use_consumable_possible(state: &GameState, slot: u8) -> bool {
    let Some(c) = state.consumables.get(usize::from(slot)) else {
        return false;
    };
    let Some(def) = try_consumable_def(c.id) else {
        return false;
    };
    let targets = if def.max_targets == 0 {
        0
    } else {
        match card_select(state, ActionKind::UseConsumable, slot) {
            Some(sel) => witness(&sel),
            None => return false,
        }
    };
    validate(state, Action::UseConsumable { slot, targets }).is_ok()
}

/// Some `PickPackItem` with this item is legal (same witness rule).
fn pick_possible(state: &GameState, index: u8) -> bool {
    let targets = match card_select(state, ActionKind::PickPackItem, index) {
        Some(sel) => witness(&sel),
        None => {
            // No card selection: either an item that takes no cards, or a targeted consumable
            // without enough eligible cards.
            let targeted = state
                .pack
                .as_ref()
                .and_then(|p| p.items.get(usize::from(index)))
                .and_then(|i| i.as_ref())
                .is_some_and(|item| {
                    matches!(item, PackItem::Consumable(id)
                        if try_consumable_def(*id).is_some_and(|d| d.max_targets > 0))
                });
            if targeted {
                return false;
            }
            0
        }
    };
    validate(state, Action::PickPackItem { index, targets }).is_ok()
}
