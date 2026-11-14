//! Vouchers: the static [`VoucherDef`] (price, passive shop modifiers, one-shot effect) and the
//! contexts effects are written against. Ids: [`crate::content::ids::VoucherId`], definitions:
//! `content/vouchers.rs`, registry: [`crate::content`]. Guide:
//! `docs/design/shop_packs_vouchers_tags.md`.
//!
//! A voucher has two kinds of effects:
//! - **passive** ([`ShopMods`]): read by the shop/pack/economy code while the voucher is owned.
//!   `GameState::shop_mods` is the fold of all owned vouchers (recomputed on redeem).
//! - **one-shot** (`on_redeem`, through [`VoucherCtx`]): permanent writes to the run (hands,
//!   discards, hand size, slots, interest cap, ante) done once when it is bought.

use crate::data::prices::VOUCHER;
use crate::economy::Discount;
use crate::state::GameState;
use serde::{Deserialize, Serialize};

/// How often the boss blind may be rerolled for money (Director's Cut / Retcon).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum BossReroll {
    /// Not available.
    #[default]
    None,
    /// Once per ante (Director's Cut).
    OncePerAnte,
    /// Without limit (Retcon).
    Unlimited,
}

/// Passive modifiers of the owned vouchers. Start from [`ShopMods::NONE`] in a def with struct
/// update syntax; [`ShopMods::merge`] folds two (sums for deltas, best-of for the rest).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShopMods {
    /// Extra random card slots in the shop (Overstock +1, Overstock Plus +1 more).
    pub card_slots: i32,
    /// Price discount on cards, packs and vouchers (Clearance Sale / Liquidation).
    pub discount: Discount,
    /// Scale of the Foil/Holo/Poly odds (Hone 2, Glow Up 4; 1 = none; Negative is not scaled).
    pub edition_mult: u32,
    /// Starting reroll price, absolute (Reroll Surplus 3, Reroll Glut 1); `None` = $5.
    pub reroll_base: Option<i32>,
    /// Tarot weight override (Tarot Merchant 9.6, Tycoon 32); `None` = 4.
    pub tarot_weight: Option<f64>,
    /// Planet weight override (Planet Merchant 9.6, Tycoon 32); `None` = 4.
    pub planet_weight: Option<f64>,
    /// Playing cards appear in the shop (Magic Trick).
    pub playing_cards: bool,
    /// Shop playing cards may be enhanced / have an edition / a seal (Illusion).
    pub illusion: bool,
    /// Celestial packs always hold the Planet of the most played hand (Telescope).
    pub telescope: bool,
    /// Arcana packs may hold Spectrals (Omen Globe).
    pub omen_globe: bool,
    /// Held Planets give x1.5 Mult for their hand (Observatory; read by scoring).
    pub observatory: bool,
    /// Paid boss rerolls (Director's Cut / Retcon).
    pub boss_reroll: BossReroll,
}

impl ShopMods {
    /// No effect.
    pub const NONE: ShopMods = ShopMods {
        card_slots: 0,
        discount: Discount::None,
        edition_mult: 1,
        reroll_base: None,
        tarot_weight: None,
        planet_weight: None,
        playing_cards: false,
        illusion: false,
        telescope: false,
        omen_globe: false,
        observatory: false,
        boss_reroll: BossReroll::None,
    };

    /// Fold `o` into `self`: `card_slots` add up; discount, `edition_mult`, weights, booleans and
    /// boss rerolls take the better value; `reroll_base` the lower one.
    pub fn merge(&mut self, o: &ShopMods) {
        self.card_slots += o.card_slots;
        self.discount = self.discount.max(o.discount);
        self.edition_mult = self.edition_mult.max(o.edition_mult);
        self.reroll_base = match (self.reroll_base, o.reroll_base) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.tarot_weight = max_opt(self.tarot_weight, o.tarot_weight);
        self.planet_weight = max_opt(self.planet_weight, o.planet_weight);
        self.playing_cards |= o.playing_cards;
        self.illusion |= o.illusion;
        self.telescope |= o.telescope;
        self.omen_globe |= o.omen_globe;
        self.observatory |= o.observatory;
        self.boss_reroll = self.boss_reroll.max(o.boss_reroll);
    }
}

impl Default for ShopMods {
    fn default() -> Self {
        ShopMods::NONE
    }
}

fn max_opt(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, y) => x.or(y),
    }
}

/// One-shot effect written against [`VoucherCtx`].
pub type RedeemFn = fn(&mut VoucherCtx<'_>);

/// Static definition of one voucher. Build with [`VoucherDef::new`] and struct update syntax:
///
/// ```ignore
/// VoucherDef {
///     passive: ShopMods { card_slots: 1, ..ShopMods::NONE },
///     ..VoucherDef::new(VoucherId::Overstock, "Overstock")
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VoucherDef {
    /// Id (exactly one def per id across all `DEFS` tables).
    pub id: crate::content::ids::VoucherId,
    /// Display name (English, for logs/debug).
    pub name: &'static str,
    /// Base price in dollars (every voucher costs $10).
    pub cost: i32,
    /// Passive modifiers.
    pub passive: ShopMods,
    /// One-shot effect when bought.
    pub on_redeem: Option<RedeemFn>,
}

impl VoucherDef {
    /// $10, no effect.
    pub const fn new(id: crate::content::ids::VoucherId, name: &'static str) -> Self {
        VoucherDef {
            id,
            name,
            cost: VOUCHER,
            passive: ShopMods::NONE,
            on_redeem: None,
        }
    }
}

/// What a [`RedeemFn`] writes through: permanent run changes. All clamped to the engine limits
/// (hand size 1..=16, hands >= 1, discards >= 0).
pub struct VoucherCtx<'a> {
    state: &'a mut GameState,
}

impl<'a> VoucherCtx<'a> {
    /// Build a context (the game layer and tests).
    pub fn new(state: &'a mut GameState) -> Self {
        VoucherCtx { state }
    }

    /// The run (read-only).
    pub fn state(&self) -> &GameState {
        self.state
    }

    /// Change the hands per round (Grabber +1, Hieroglyph -1).
    pub fn add_hands(&mut self, delta: i32) {
        self.state.base_hands = add_u32(self.state.base_hands, delta, 1);
    }

    /// Change the discards per round (Wasteful +1, Petroglyph -1).
    pub fn add_discards(&mut self, delta: i32) {
        self.state.base_discards = add_u32(self.state.base_discards, delta, 0);
    }

    /// Change the base hand size (Paint Brush, Palette +1).
    pub fn add_hand_size(&mut self, delta: i32) {
        self.state.base_hand_size = add_u32(self.state.base_hand_size, delta, 1);
    }

    /// Change the joker slots (Antimatter +1).
    pub fn add_joker_slots(&mut self, delta: i32) {
        self.state.joker_slots = add_u32(self.state.joker_slots, delta, 0);
    }

    /// Change the consumable slots (Crystal Ball +1).
    pub fn add_consumable_slots(&mut self, delta: i32) {
        self.state.consumable_slots = add_u32(self.state.consumable_slots, delta, 0);
    }

    /// Raise the interest cap to at least `cap` (Seed Money, Money Tree).
    pub fn raise_interest_cap(&mut self, cap: i32) {
        self.state.interest_cap = self.state.interest_cap.max(cap);
    }

    /// Change the current ante (Hieroglyph/Petroglyph -1) and recompute the blind target. The
    /// ante's boss and tags stay as rolled.
    pub fn add_ante(&mut self, delta: i32) {
        self.state.ante = self.state.ante.saturating_add(delta);
        self.state.refresh_target();
    }
}

fn add_u32(v: u32, delta: i32, lo: u32) -> u32 {
    (i64::from(v) + i64::from(delta)).clamp(i64::from(lo), i64::from(u32::MAX)) as u32
}

/// Redeem `id`: record it, refold `GameState::shop_mods`, run its one-shot effect, re-price the
/// owned items if the discount changed, and fill a card slot the voucher opened in a running
/// shop (Overstock). Does nothing for an undefined voucher. Payment and slot bookkeeping are the
/// caller's (`Game::step(BuyVoucher)`).
pub fn redeem(s: &mut GameState, id: crate::content::ids::VoucherId) {
    let Some(def) = crate::content::try_voucher_def(id) else {
        return;
    };
    let old_discount = s.shop_mods.discount;
    s.vouchers.push(id);
    s.shop_mods = crate::state::fold_shop_mods(&s.vouchers);
    if let Some(f) = def.on_redeem {
        f(&mut VoucherCtx::new(s));
    }
    crate::shop::rescale_sell_values(s, old_discount, s.shop_mods.discount);
    if s.phase == crate::enums::Phase::Shop {
        crate::shop::fill_card_slots(s);
    }
}
