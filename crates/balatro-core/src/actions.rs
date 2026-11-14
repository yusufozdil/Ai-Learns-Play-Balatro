//! [`Action`] (card selections are `u16` bitmasks over hand positions, bit i = hand card i),
//! [`ActionError`], action validation and per-phase legal-action generation.
//!
//! Contract: [`validate`] is pure and is the single gate of `Game::step`; an action that passes
//! it is executed without further failure, and an action that fails it leaves the state
//! untouched. [`legal_actions`] lists exactly the candidate actions that pass [`validate`]
//! (tested). Phase 2C replaces the enumeration with a fixed-size action codec and masks.

use crate::blinds::BlindKind;
use crate::data::run_rules::MAX_SELECTED_CARDS;
use crate::enums::Phase;
use crate::state::GameState;
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
    /// Play/Shop: sell the joker in this slot.
    SellJoker(u8),
    /// Shop: leave the shop.
    EndShop,
    /// Play/Shop: use the consumable in `slot` on the selected hand cards. Phase 4.
    UseConsumable {
        /// Consumable slot.
        slot: u8,
        /// Selected hand cards (bitmask), 0 if the consumable has no targets.
        targets: u16,
    },
    /// Play/Shop: sell the consumable in this slot. Phase 4.
    SellConsumable(u8),
    /// Play/Shop: move a joker from one slot to another. Phase 5.
    MoveJoker {
        /// Source slot.
        from: u8,
        /// Destination slot.
        to: u8,
    },
    /// Shop: buy the card in this shop slot. Phase 5.
    Buy(u8),
    /// Shop: buy the consumable in this shop slot and use it at once. Phase 5.
    BuyAndUse(u8),
    /// Shop: buy and open the booster pack in this slot. Phase 5.
    OpenPack(u8),
    /// Shop: buy the voucher in this slot. Phase 5.
    BuyVoucher(u8),
    /// Shop: reroll the shop. Phase 5.
    Reroll,
    /// PackOpen: take item `index` (with hand-card targets for Tarots/Spectrals). Phase 5.
    PickPackItem {
        /// Pack item.
        index: u8,
        /// Selected hand cards (bitmask).
        targets: u16,
    },
    /// PackOpen: skip the rest of the pack. Phase 5.
    SkipPack,
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
}

/// Plan phase that implements each not-yet-available action (for `NotImplemented`).
fn unimplemented_phase(a: Action) -> Option<u8> {
    match a {
        Action::UseConsumable { .. } | Action::SellConsumable(_) => Some(4),
        Action::MoveJoker { .. }
        | Action::Buy(_)
        | Action::BuyAndUse(_)
        | Action::OpenPack(_)
        | Action::BuyVoucher(_)
        | Action::Reroll
        | Action::PickPackItem { .. }
        | Action::SkipPack => Some(5),
        _ => None,
    }
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
    if let Some(p) = unimplemented_phase(action) {
        return Err(ActionError::NotImplemented {
            action,
            phase_of_plan: p,
        });
    }
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
        }
        Action::Discard(mask) => {
            if phase != Phase::Play {
                return Err(wrong());
            }
            if state.discards_left == 0 {
                return Err(ActionError::NoDiscardsLeft);
            }
            validate_selection(mask, state.deck.hand.len())?;
        }
        Action::SellJoker(i) => {
            if !matches!(phase, Phase::Play | Phase::Shop) {
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
        // Handled by `unimplemented_phase` above.
        Action::UseConsumable { .. }
        | Action::SellConsumable(_)
        | Action::MoveJoker { .. }
        | Action::Buy(_)
        | Action::BuyAndUse(_)
        | Action::OpenPack(_)
        | Action::BuyVoucher(_)
        | Action::Reroll
        | Action::PickPackItem { .. }
        | Action::SkipPack => {
            return Err(ActionError::NotImplemented {
                action,
                phase_of_plan: 5,
            });
        }
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
/// `Play` masks ascending, `Discard` masks ascending, `SellJoker` slots ascending.
/// Skeleton: exhaustive enumeration (218 Play + 218 Discard masks for 8 cards); Phase 2C
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
    match state.phase {
        Phase::BlindSelect => {
            out.push(Action::SelectBlind);
            if state.blind != BlindKind::Boss {
                out.push(Action::SkipBlind);
            }
        }
        Phase::Play => {
            let mut masks = Vec::new();
            push_selections(state.deck.hand.len(), &mut masks);
            if state.hands_left > 0 {
                out.extend(masks.iter().map(|&m| Action::Play(m)));
            }
            if state.discards_left > 0 {
                out.extend(masks.iter().map(|&m| Action::Discard(m)));
            }
            sell(out);
        }
        Phase::Shop => {
            out.push(Action::EndShop);
            sell(out);
        }
        // TODO(Phase 5): pack items.
        Phase::PackOpen => {}
        Phase::GameOver | Phase::Won => {}
    }
}
