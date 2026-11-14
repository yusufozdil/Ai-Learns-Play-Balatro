//! [`Action`] (card selections are `u16` bitmasks over hand positions, bit i = hand card i),
//! [`ActionError`], action validation and per-phase legal-action generation.
//!
//! Contract: [`validate`] is pure and is the single gate of `Game::step`; an action that passes
//! it is executed without further failure, and an action that fails it leaves the state
//! untouched. [`legal_actions`] lists exactly the candidate actions that pass [`validate`]
//! (tested). Phase 2C replaces the enumeration with a fixed-size action codec and masks.

use crate::blinds::BlindKind;
use crate::boss;
use crate::consumable::{ConsumableDef, MAX_CONSUMABLE_TARGETS, UseCheck};
use crate::content::try_consumable_def;
use crate::data::run_rules::MAX_SELECTED_CARDS;
use crate::data::shop_weights::BOSS_REROLL_COST;
use crate::enums::Phase;
use crate::packs::{self, PackItem};
use crate::shop::{self, ShopCard, ShopCardKind};
use crate::state::GameState;
use crate::voucher::BossReroll;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Every decision a player can make (prd.md §3.1). Indices are positions (hand position, joker
/// slot, consumable slot, shop slot, pack item).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// BlindSelect: play the current blind.
    SelectBlind,
    /// BlindSelect: skip the Small/Big blind for its tag (never the Boss).
    SkipBlind,
    /// Play: play the selected hand cards (1-5), left to right in hand order.
    Play(u16),
    /// Play: discard the selected hand cards (1-5).
    Discard(u16),
    /// Any phase but game over: sell the joker in this slot.
    SellJoker(u8),
    /// Shop: leave the shop.
    EndShop,
    /// Any phase but game over: use the consumable in `slot` on the selected hand cards (cards
    /// that need a hand only work while one is dealt: `Play`, Arcana/Spectral packs).
    UseConsumable {
        /// Consumable slot.
        slot: u8,
        /// Selected hand cards (bitmask), 0 if the consumable has no targets.
        targets: u16,
    },
    /// Any phase but game over: sell the consumable in this slot.
    SellConsumable(u8),
    /// Any phase but game over: move a joker from slot `from` to slot `to` (the others shift).
    /// Joker order changes the score.
    MoveJoker {
        /// Source slot.
        from: u8,
        /// Destination slot.
        to: u8,
    },
    /// Shop: buy the card in this shop card slot (joker, consumable, playing card).
    Buy(u8),
    /// Shop: buy the consumable in this shop card slot and use it at once without taking a
    /// slot. Only for cards usable with no hand and no targets (Planets, Hermit, ...): the shop
    /// has no hand to target.
    BuyAndUse(u8),
    /// Shop: buy and open the booster pack in this slot (`PackOpen`).
    OpenPack(u8),
    /// Shop: buy the voucher in this slot.
    BuyVoucher(u8),
    /// Shop: reroll the card slots.
    Reroll,
    /// PackOpen: take item `index` (with hand-card targets for Tarots/Spectrals).
    PickPackItem {
        /// Pack item.
        index: u8,
        /// Selected hand cards (bitmask).
        targets: u16,
    },
    /// PackOpen: skip the rest of the pack.
    SkipPack,
    /// BlindSelect: reroll the ante's boss for $10 (Director's Cut once per ante, Retcon
    /// unlimited).
    RerollBoss,
}

/// Why an action was rejected. The state is unchanged when `step` returns one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ActionError {
    /// The run is over.
    #[error("the run is over")]
    GameOver,
    /// The action is not available in this phase.
    #[error("{action:?} is not allowed in phase {phase:?}")]
    WrongPhase {
        /// The rejected action.
        action: Action,
        /// Current phase.
        phase: Phase,
    },
    /// The action exists in the action space but its phase of the plan is not implemented yet.
    #[error("{action:?} is not implemented yet (Phase {phase_of_plan})")]
    NotImplemented {
        /// The rejected action.
        action: Action,
        /// The plan phase that implements it.
        phase_of_plan: u8,
    },
    /// Boss blinds cannot be skipped.
    #[error("the boss blind cannot be skipped")]
    CannotSkipBoss,
    /// The card selection is empty.
    #[error("empty card selection")]
    EmptySelection,
    /// More cards selected than allowed.
    #[error("{selected} cards selected, at most {max}")]
    TooManyCards {
        /// Selected cards.
        selected: u32,
        /// Maximum.
        max: u32,
    },
    /// The selection names a position past the end of the hand.
    #[error("selection {mask:#b} exceeds hand of {hand_len} cards")]
    SelectionOutOfRange {
        /// Offending mask.
        mask: u16,
        /// Hand length.
        hand_len: usize,
    },
    /// No hands left this round.
    #[error("no hands left")]
    NoHandsLeft,
    /// No discards left this round.
    #[error("no discards left")]
    NoDiscardsLeft,
    /// No joker in that slot.
    #[error("no joker in slot {index} ({len} jokers)")]
    InvalidJoker {
        /// Requested slot.
        index: u8,
        /// Jokers owned.
        len: usize,
    },
    /// Eternal jokers cannot be sold.
    #[error("eternal jokers cannot be sold")]
    EternalJoker,
    /// No consumable in that slot.
    #[error("no consumable in slot {index} ({len} held)")]
    InvalidConsumable {
        /// Requested slot.
        index: u8,
        /// Consumables held.
        len: usize,
    },
    /// No item in that shop slot (empty or past the end).
    #[error("no item in shop slot {index}")]
    InvalidShopSlot {
        /// Requested slot.
        index: u8,
    },
    /// Not enough money (money minus `cost` would go below the debt floor).
    #[error("cannot afford ${cost} with ${money}")]
    CannotAfford {
        /// Price.
        cost: i32,
        /// Money held.
        money: i32,
    },
    /// No free joker slot.
    #[error("no free joker slot")]
    NoJokerRoom,
    /// No free consumable slot.
    #[error("no free consumable slot")]
    NoConsumableRoom,
    /// The shop slot does not hold a consumable usable on the spot.
    #[error("this shop item cannot be bought and used")]
    NotUsableOnBuy,
    /// No item at that pack position (already taken or past the end).
    #[error("no item at pack position {index}")]
    InvalidPackItem {
        /// Requested position.
        index: u8,
    },
    /// A joker move needs two different existing slots.
    #[error("cannot move joker {from} to {to} ({len} jokers)")]
    InvalidJokerMove {
        /// Source slot.
        from: u8,
        /// Destination slot.
        to: u8,
        /// Jokers owned.
        len: usize,
    },
    /// The boss cannot be rerolled now (no voucher, limit reached, no boss).
    #[error("the boss blind cannot be rerolled now")]
    BossRerollUnavailable,
    /// The consumable has no definition yet (it can only be sold).
    #[error("this consumable is not implemented yet")]
    UnknownConsumable,
    /// The consumable needs a drawn hand (only in `Phase::Play`).
    #[error("this consumable needs cards in hand")]
    NeedsHand,
    /// Wrong number of selected hand cards for the consumable.
    #[error("{selected} target cards selected, need {min}..={max}")]
    TargetCount {
        /// Selected cards.
        selected: u32,
        /// Minimum.
        min: u8,
        /// Maximum.
        max: u8,
    },
    /// The consumable's own condition fails (no free slot, no eligible joker, ...).
    #[error("this consumable cannot be used now")]
    CannotUse,
    /// The boss forces a card to be selected (Cerulean Bell) and the selection lacks it.
    #[error("the force-selected card at position {position} must be selected")]
    ForcedCardMissing {
        /// Hand position of the forced card.
        position: usize,
    },
}

/// Validate a card selection over a hand of `hand_len` cards: 1..=5 bits, all inside the hand.
pub fn validate_selection(mask: u16, hand_len: usize) -> Result<(), ActionError> {
    if mask == 0 {
        return Err(ActionError::EmptySelection);
    }
    let selected = mask.count_ones();
    if selected > MAX_SELECTED_CARDS {
        return Err(ActionError::TooManyCards {
            selected,
            max: MAX_SELECTED_CARDS,
        });
    }
    if hand_len < 16 && (mask >> hand_len) != 0 {
        return Err(ActionError::SelectionOutOfRange { mask, hand_len });
    }
    Ok(())
}

/// Check `action` against `state` without changing anything.
pub fn validate(state: &GameState, action: Action) -> Result<(), ActionError> {
    if state.is_terminal() {
        return Err(ActionError::GameOver);
    }
    let phase = state.phase;
    let wrong = || ActionError::WrongPhase { action, phase };
    match action {
        Action::SelectBlind => {
            if phase != Phase::BlindSelect {
                return Err(wrong());
            }
        }
        Action::SkipBlind => {
            if phase != Phase::BlindSelect {
                return Err(wrong());
            }
            if state.blind == BlindKind::Boss {
                return Err(ActionError::CannotSkipBoss);
            }
        }
        Action::Play(mask) => {
            if phase != Phase::Play {
                return Err(wrong());
            }
            if state.hands_left == 0 {
                return Err(ActionError::NoHandsLeft);
            }
            validate_selection(mask, state.deck.hand.len())?;
            check_forced(state, mask)?;
        }
        Action::Discard(mask) => {
            if phase != Phase::Play {
                return Err(wrong());
            }
            if state.discards_left == 0 {
                return Err(ActionError::NoDiscardsLeft);
            }
            validate_selection(mask, state.deck.hand.len())?;
            check_forced(state, mask)?;
        }
        Action::SellJoker(i) => {
            if !can_manage(phase) {
                return Err(wrong());
            }
            let j = state
                .jokers
                .get(usize::from(i))
                .ok_or(ActionError::InvalidJoker {
                    index: i,
                    len: state.jokers.len(),
                })?;
            if j.stickers.eternal {
                return Err(ActionError::EternalJoker);
            }
        }
        Action::EndShop => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
        }
        Action::UseConsumable { slot, targets } => {
            if !can_manage(phase) {
                return Err(wrong());
            }
            validate_use(state, slot, targets)?;
        }
        Action::SellConsumable(i) => {
            if !can_manage(phase) {
                return Err(wrong());
            }
            if usize::from(i) >= state.consumables.len() {
                return Err(ActionError::InvalidConsumable {
                    index: i,
                    len: state.consumables.len(),
                });
            }
        }
        Action::MoveJoker { from, to } => {
            if !can_manage(phase) {
                return Err(wrong());
            }
            let len = state.jokers.len();
            if from == to || usize::from(from) >= len || usize::from(to) >= len {
                return Err(ActionError::InvalidJokerMove { from, to, len });
            }
        }
        Action::Buy(i) => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
            let c = shop_card(state, i)?;
            pay_check(state, shop::card_price(state, c))?;
            match c.kind {
                ShopCardKind::Joker(j) if !packs::joker_fits(state, &j) => {
                    return Err(ActionError::NoJokerRoom);
                }
                ShopCardKind::Consumable(_) if !state.has_consumable_room() => {
                    return Err(ActionError::NoConsumableRoom);
                }
                _ => {}
            }
        }
        Action::BuyAndUse(i) => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
            let c = shop_card(state, i)?;
            pay_check(state, shop::card_price(state, c))?;
            let ShopCardKind::Consumable(id) = c.kind else {
                return Err(ActionError::NotUsableOnBuy);
            };
            let def = try_consumable_def(id).ok_or(ActionError::NotUsableOnBuy)?;
            if def.requires_hand()
                || def.max_targets != 0
                || def
                    .can_use
                    .is_some_and(|f| !f(&UseCheck::new(state, 0, None)))
            {
                return Err(ActionError::NotUsableOnBuy);
            }
        }
        Action::OpenPack(i) => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
            let p = state
                .shop
                .packs
                .get(usize::from(i))
                .and_then(|p| p.as_ref())
                .ok_or(ActionError::InvalidShopSlot { index: i })?;
            pay_check(state, shop::pack_price_of(state, p))?;
        }
        Action::BuyVoucher(i) => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
            let v = state
                .shop
                .vouchers
                .get(usize::from(i))
                .and_then(|v| v.as_ref())
                .ok_or(ActionError::InvalidShopSlot { index: i })?;
            pay_check(state, shop::voucher_price(state, v.id))?;
        }
        Action::Reroll => {
            if phase != Phase::Shop {
                return Err(wrong());
            }
            pay_check(state, shop::reroll_price(state))?;
        }
        Action::PickPackItem { index, targets } => {
            if phase != Phase::PackOpen {
                return Err(wrong());
            }
            validate_pick(state, index, targets)?;
        }
        Action::SkipPack => {
            if phase != Phase::PackOpen {
                return Err(wrong());
            }
        }
        Action::RerollBoss => {
            if phase != Phase::BlindSelect {
                return Err(wrong());
            }
            if !boss_reroll_available(state) {
                return Err(ActionError::BossRerollUnavailable);
            }
            pay_check(state, BOSS_REROLL_COST)?;
        }
    }
    Ok(())
}

/// Selling, moving jokers and using consumables are possible in every phase of a live run.
fn can_manage(phase: Phase) -> bool {
    matches!(
        phase,
        Phase::BlindSelect | Phase::Play | Phase::Shop | Phase::PackOpen
    )
}

fn shop_card(state: &GameState, i: u8) -> Result<&ShopCard, ActionError> {
    state
        .shop
        .cards
        .get(usize::from(i))
        .and_then(|c| c.as_ref())
        .ok_or(ActionError::InvalidShopSlot { index: i })
}

/// Spending `cost` keeps money at or above the debt floor.
fn pay_check(state: &GameState, cost: i32) -> Result<(), ActionError> {
    if state.can_afford(cost) {
        Ok(())
    } else {
        Err(ActionError::CannotAfford {
            cost,
            money: state.money,
        })
    }
}

/// A paid boss reroll is allowed by the vouchers and the ante's count (price not checked).
pub fn boss_reroll_available(state: &GameState) -> bool {
    state.boss.is_some()
        && match state.shop_mods.boss_reroll {
            BossReroll::None => false,
            BossReroll::OncePerAnte => state.boss_rerolls_ante == 0,
            BossReroll::Unlimited => true,
        }
}

/// Validate taking pack item `index` with hand-card `targets`: the item exists; a consumable
/// follows the same target rules as `validate_use` (slot `None`, see `UseCheck`); a joker needs a
/// free slot (Negative excepted); a playing card always fits. Pure.
pub fn validate_pick(state: &GameState, index: u8, targets: u16) -> Result<(), ActionError> {
    let pack = state.pack.as_ref().ok_or(ActionError::WrongPhase {
        action: Action::PickPackItem { index, targets },
        phase: state.phase,
    })?;
    let item = pack
        .items
        .get(usize::from(index))
        .and_then(|i| i.as_ref())
        .ok_or(ActionError::InvalidPackItem { index })?;
    match item {
        PackItem::Consumable(id) => {
            let def = try_consumable_def(*id).ok_or(ActionError::UnknownConsumable)?;
            validate_use_def(state, def, targets, None)
        }
        PackItem::Joker(j) => {
            if targets != 0 {
                return Err(ActionError::TargetCount {
                    selected: targets.count_ones(),
                    min: 0,
                    max: 0,
                });
            }
            if packs::joker_fits(state, j) {
                Ok(())
            } else {
                Err(ActionError::NoJokerRoom)
            }
        }
        PackItem::Card(_) => {
            if targets != 0 {
                return Err(ActionError::TargetCount {
                    selected: targets.count_ones(),
                    min: 0,
                    max: 0,
                });
            }
            Ok(())
        }
    }
}

/// Cerulean Bell: a Play/Discard selection must contain the force-selected card.
fn check_forced(state: &GameState, mask: u16) -> Result<(), ActionError> {
    match boss::forced_position(state) {
        Some(p) if p < 16 && (mask >> p) & 1 == 0 => {
            Err(ActionError::ForcedCardMissing { position: p })
        }
        _ => Ok(()),
    }
}

/// Validate using the consumable in `slot` with `targets` (phase already checked): the card
/// exists and is defined; cards needing a hand only in `Phase::Play`; untargeted cards need
/// `targets == 0`, targeted ones `min..=max` selected cards inside the hand; then the card's
/// own `can_use`. Pure.
pub fn validate_use(state: &GameState, slot: u8, targets: u16) -> Result<(), ActionError> {
    let c = state
        .consumables
        .get(usize::from(slot))
        .ok_or(ActionError::InvalidConsumable {
            index: slot,
            len: state.consumables.len(),
        })?;
    let def = try_consumable_def(c.id).ok_or(ActionError::UnknownConsumable)?;
    validate_use_def(state, def, targets, Some(usize::from(slot)))
}

/// The checks of [`validate_use`] for a definition used from `slot` (`None` = picked from a
/// pack): a hand is needed for `requires_hand` cards, target counts, then `can_use`.
pub fn validate_use_def(
    state: &GameState,
    def: &ConsumableDef,
    targets: u16,
    slot: Option<usize>,
) -> Result<(), ActionError> {
    if def.requires_hand() && state.deck.hand.is_empty() {
        return Err(ActionError::NeedsHand);
    }
    let selected = targets.count_ones();
    if def.max_targets == 0 {
        if targets != 0 {
            return Err(ActionError::TargetCount {
                selected,
                min: 0,
                max: 0,
            });
        }
    } else {
        let hand_len = state.deck.hand.len();
        if hand_len < 16 && (targets >> hand_len) != 0 {
            return Err(ActionError::SelectionOutOfRange {
                mask: targets,
                hand_len,
            });
        }
        let max = def.max_targets.min(MAX_CONSUMABLE_TARGETS);
        if selected < u32::from(def.min_targets) || selected > u32::from(max) {
            return Err(ActionError::TargetCount {
                selected,
                min: def.min_targets,
                max,
            });
        }
    }
    if let Some(f) = def.can_use
        && !f(&UseCheck::new(state, targets, slot))
    {
        return Err(ActionError::CannotUse);
    }
    Ok(())
}

/// Every card selection with 1..=5 cards over a hand of `hand_len` cards, ascending.
fn push_selections(hand_len: usize, out: &mut Vec<u16>) {
    let n = hand_len.min(16);
    if n == 0 {
        return;
    }
    let limit: u32 = 1 << n;
    for m in 1..limit {
        if m.count_ones() <= MAX_SELECTED_CARDS {
            out.push(m as u16);
        }
    }
}

/// Append every legal action of `state` to `out` (cleared first). Order: phase actions, then
/// `Play` masks ascending, `Discard` masks ascending, shop purchases (`Buy`/`BuyAndUse` per card
/// slot, `OpenPack`, `BuyVoucher`, `Reroll`) or pack picks (`PickPackItem` per item, targets
/// ascending, then `SkipPack`), `SellJoker` slots ascending, `MoveJoker` pairs, then per
/// consumable slot (ascending) its `UseConsumable` actions (targets ascending) followed by its
/// `SellConsumable`.
///
/// Target enumeration rule: an untargeted card gets one action (`targets = 0`); a targeted card
/// gets every selection whose size is in `min_targets..=max_targets` (at most
/// `MAX_CONSUMABLE_TARGETS` = 3: 92 masks for 8 cards) that passes `validate`. Cerulean Bell
/// restricts Play/Discard masks to those containing the forced card (not consumable targets).
/// Skeleton: exhaustive enumeration (218 Play + 218 Discard masks for 8 cards); Phase 8
/// provides the fixed-size codec/masks for the env.
pub fn legal_actions(state: &GameState, out: &mut Vec<Action>) {
    out.clear();
    if state.is_terminal() {
        return;
    }
    let sell = |out: &mut Vec<Action>| {
        for (i, j) in state.jokers.iter().enumerate() {
            if !j.stickers.eternal
                && let Ok(i) = u8::try_from(i)
            {
                out.push(Action::SellJoker(i));
            }
        }
    };
    let moves = |out: &mut Vec<Action>| {
        let n = state.jokers.len().min(usize::from(u8::MAX));
        for from in 0..n {
            for to in 0..n {
                if from != to {
                    out.push(Action::MoveJoker {
                        from: from as u8,
                        to: to as u8,
                    });
                }
            }
        }
    };
    match state.phase {
        Phase::BlindSelect => {
            out.push(Action::SelectBlind);
            if state.blind != BlindKind::Boss {
                out.push(Action::SkipBlind);
            }
            if validate(state, Action::RerollBoss).is_ok() {
                out.push(Action::RerollBoss);
            }
            sell(out);
            moves(out);
            consumable_actions(state, &[], out);
        }
        Phase::Play => {
            let mut masks = Vec::new();
            push_selections(state.deck.hand.len(), &mut masks);
            let forced = boss::forced_position(state).filter(|&p| p < 16);
            let ok = |m: u16| forced.is_none_or(|p| (m >> p) & 1 == 1);
            if state.hands_left > 0 {
                out.extend(masks.iter().filter(|&&m| ok(m)).map(|&m| Action::Play(m)));
            }
            if state.discards_left > 0 {
                out.extend(
                    masks
                        .iter()
                        .filter(|&&m| ok(m))
                        .map(|&m| Action::Discard(m)),
                );
            }
            sell(out);
            moves(out);
            consumable_actions(state, &masks, out);
        }
        Phase::Shop => {
            out.push(Action::EndShop);
            let n = |len: usize| u8::try_from(len).unwrap_or(u8::MAX);
            let push_if = |a: Action, out: &mut Vec<Action>| {
                if validate(state, a).is_ok() {
                    out.push(a);
                }
            };
            for i in 0..n(state.shop.cards.len()) {
                push_if(Action::Buy(i), out);
                push_if(Action::BuyAndUse(i), out);
            }
            for i in 0..n(state.shop.packs.len()) {
                push_if(Action::OpenPack(i), out);
            }
            for i in 0..n(state.shop.vouchers.len()) {
                push_if(Action::BuyVoucher(i), out);
            }
            push_if(Action::Reroll, out);
            sell(out);
            moves(out);
            consumable_actions(state, &[], out);
        }
        Phase::PackOpen => {
            let mut masks = Vec::new();
            push_selections(state.deck.hand.len(), &mut masks);
            if let Some(pack) = &state.pack {
                for (i, item) in pack.items.iter().enumerate() {
                    let (Some(item), Ok(index)) = (item, u8::try_from(i)) else {
                        continue;
                    };
                    match item {
                        PackItem::Consumable(id) => {
                            let Some(def) = try_consumable_def(*id) else {
                                continue;
                            };
                            if def.max_targets == 0 {
                                if validate_pick(state, index, 0).is_ok() {
                                    out.push(Action::PickPackItem { index, targets: 0 });
                                }
                            } else {
                                let (lo, hi) = (
                                    u32::from(def.min_targets),
                                    u32::from(def.max_targets.min(MAX_CONSUMABLE_TARGETS)),
                                );
                                for &m in &masks {
                                    let k = m.count_ones();
                                    if k >= lo && k <= hi && validate_pick(state, index, m).is_ok()
                                    {
                                        out.push(Action::PickPackItem { index, targets: m });
                                    }
                                }
                            }
                        }
                        _ => {
                            if validate_pick(state, index, 0).is_ok() {
                                out.push(Action::PickPackItem { index, targets: 0 });
                            }
                        }
                    }
                }
            }
            out.push(Action::SkipPack);
            sell(out);
            moves(out);
            consumable_actions(state, &masks, out);
        }
        Phase::GameOver | Phase::Won => {}
    }
}

/// Per consumable slot: legal `UseConsumable` actions (see [`legal_actions`]) then
/// `SellConsumable`. `masks` = all 1..=5 card selections of the hand (ascending).
fn consumable_actions(state: &GameState, masks: &[u16], out: &mut Vec<Action>) {
    for (i, c) in state.consumables.iter().enumerate() {
        let Ok(slot) = u8::try_from(i) else {
            break;
        };
        if let Some(def) = try_consumable_def(c.id) {
            if def.max_targets == 0 {
                if validate_use(state, slot, 0).is_ok() {
                    out.push(Action::UseConsumable { slot, targets: 0 });
                }
            } else {
                let (lo, hi) = (
                    u32::from(def.min_targets),
                    u32::from(def.max_targets.min(MAX_CONSUMABLE_TARGETS)),
                );
                for &m in masks {
                    let k = m.count_ones();
                    if k >= lo && k <= hi && validate_use(state, slot, m).is_ok() {
                        out.push(Action::UseConsumable { slot, targets: m });
                    }
                }
            }
        }
        out.push(Action::SellConsumable(slot));
    }
}
