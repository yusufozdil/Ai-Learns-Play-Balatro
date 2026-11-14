//! Pure money math: interest, hands money, sell value, shop prices, reroll cost, cash-out.
//! Sources: docs/mechanics/economy.md, blinds.md §4/§6/§8, shop.md §6-§8; notes.md §4.1c/§4.1d.

use crate::blinds::{BlindKind, blind_reward};
use crate::data::prices::{
    CLEARANCE_PERCENT, DISCARD_REWARD, GOLD_CARD_PAYOUT, GOLD_CARD_RED_SEAL_PAYOUT, HAND_REWARD,
    INTEREST_STEP, LIQUIDATION_PERCENT, REROLL_INCREMENT,
};

/// Shop discount from vouchers (shop.md §6). Applies to cards, packs and vouchers (notes.md §4.1d).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Discount {
    /// No discount.
    #[default]
    None,
    /// Clearance Sale, 25% off.
    Clearance,
    /// Liquidation, 50% off.
    Liquidation,
}

impl Discount {
    /// Discount in percent (0, 25, 50).
    pub fn percent(self) -> i32 {
        match self {
            Discount::None => 0,
            Discount::Clearance => CLEARANCE_PERCENT,
            Discount::Liquidation => LIQUIDATION_PERCENT,
        }
    }
}

/// Interest: $1 per full $5 held, capped at `cap` dollars; money <= 0 pays nothing.
/// `per_5_bonus` (To the Moon: `TO_THE_MOON_BONUS`) adds that many dollars per $5 and is subject
/// to the SAME step cap, i.e. `min(money/5, cap) * (1 + per_5_bonus)`. This matches economy.md:
/// base interest is capped at `cap`, then an equal extra amount is added (cap 20 + To the Moon = 40).
pub fn interest(money: i32, cap: i32, per_5_bonus: i32) -> i32 {
    if money <= 0 || cap <= 0 {
        return 0;
    }
    let steps = (money / INTEREST_STEP).min(cap);
    steps.saturating_mul(1 + per_5_bonus.max(0))
}

/// Money for unused hands, $1 each (blinds.md §4). Unused discards pay `DISCARD_REWARD` = $0.
pub fn hands_reward(hands_left: i32) -> i32 {
    hands_left.max(0).saturating_mul(HAND_REWARD)
}

/// Money for unused discards (always $0 in the base game, economy.md).
pub fn discards_reward(discards_left: i32) -> i32 {
    discards_left.max(0).saturating_mul(DISCARD_REWARD)
}

/// Sell value: `max(1, floor(cost_paid / 2)) + extra` (Egg/Gift Card bonus). `cost_paid` is the
/// discounted price at purchase (notes.md §4.1d).
pub fn sell_value(cost_paid: i32, extra: i32) -> i32 {
    (cost_paid / 2).max(1).saturating_add(extra)
}

/// Shop price: `max(1, round_half_down((base + edition_surcharge) * (100 - disc) / 100))`
/// (shop.md §6), computed in integers as `floor((2x + 1) * (100 - disc) / 200)`.
pub fn shop_price(base: i32, edition_surcharge: i32, discount: Discount) -> i32 {
    let x = i64::from(base) + i64::from(edition_surcharge);
    let pct = i64::from(100 - discount.percent());
    let price = ((2 * x + 1) * pct).div_euclid(200);
    price.clamp(1, i64::from(i32::MAX)) as i32
}

/// Reroll cost: `base + rerolls_this_shop`, floored at $0 (shop.md §7). `base` is
/// `REROLL_BASE`, `REROLL_SURPLUS_BASE`, `REROLL_GLUT_BASE`, or 0 (D6 Tag).
/// Free rerolls (Chaos the Clown) do not advance the counter; the caller handles that.
pub fn reroll_cost(rerolls_this_shop: u32, base: i32) -> i32 {
    let inc = i64::from(rerolls_this_shop) * i64::from(REROLL_INCREMENT);
    (i64::from(base) + inc).clamp(0, i64::from(i32::MAX)) as i32
}

/// Inputs of [`cash_out`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CashOutInput {
    /// Money when the round ended (before any cash-out payout).
    pub money: i32,
    /// Gold-enhanced cards held in hand without a Red Seal.
    pub gold_cards_held: u32,
    /// Gold-enhanced cards held in hand with a Red Seal.
    pub gold_red_seal_cards_held: u32,
    /// Blind just beaten.
    pub kind: BlindKind,
    /// Boss is a showdown blind.
    pub is_showdown: bool,
    /// Red Stake or above (Small pays $0).
    pub stake_no_small_reward: bool,
    /// Blind saved by Mr. Bones: no blind reward (blinds.md §4).
    pub no_blind_reward: bool,
    /// Unused hands.
    pub hands_left: i32,
    /// Interest cap in dollars (`INTEREST_CAP_BASE` etc.).
    pub interest_cap: i32,
    /// To the Moon bonus per $5 (0 if absent).
    pub interest_per_5_bonus: i32,
    /// No interest at all (Green Deck, Omelette, Mad World).
    pub no_interest: bool,
    /// Joker cash-out payouts (Golden Joker, Rocket, ...), summed by the caller.
    pub joker_payout: i32,
}

/// Itemised end-of-round payout. The game layer applies `total` to money.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CashOut {
    /// Gold cards held.
    pub gold_cards: i32,
    /// Balance interest was computed on (money + gold cards only).
    pub interest_base: i32,
    /// Interest earned.
    pub interest: i32,
    /// Blind reward.
    pub blind_reward: i32,
    /// Unused hands money.
    pub hands: i32,
    /// Joker payouts.
    pub jokers: i32,
    /// Sum of all items.
    pub total: i32,
    /// `money + total`.
    pub final_money: i32,
}

/// Compute the cash-out in the order fixed by notes.md §4.1c: gold cards held -> interest on that
/// balance (money + gold) -> blind reward -> hands money -> joker payouts. Interest therefore
/// ignores the blind reward, hands money and joker cash-out money.
pub fn cash_out(input: &CashOutInput) -> CashOut {
    let gold_cards = (input.gold_cards_held as i32)
        .saturating_mul(GOLD_CARD_PAYOUT)
        .saturating_add(
            (input.gold_red_seal_cards_held as i32).saturating_mul(GOLD_CARD_RED_SEAL_PAYOUT),
        );
    let interest_base = input.money.saturating_add(gold_cards);
    let interest = if input.no_interest {
        0
    } else {
        self::interest(
            interest_base,
            input.interest_cap,
            input.interest_per_5_bonus,
        )
    };
    let blind = if input.no_blind_reward {
        0
    } else {
        blind_reward(input.kind, input.is_showdown, input.stake_no_small_reward)
    };
    let hands = hands_reward(input.hands_left);
    let jokers = input.joker_payout;
    let total = gold_cards
        .saturating_add(interest)
        .saturating_add(blind)
        .saturating_add(hands)
        .saturating_add(jokers);
    CashOut {
        gold_cards,
        interest_base,
        interest,
        blind_reward: blind,
        hands,
        jokers,
        total,
        final_money: input.money.saturating_add(total),
    }
}
