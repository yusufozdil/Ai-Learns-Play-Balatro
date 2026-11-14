//! Skip tags (24). Phase 5.0 defines the prototypes (Economy, Investment, Charm, Coupon, Double,
//! Boss); Phase 5B adds the other 18 here. Guide: `docs/design/shop_packs_vouchers_tags.md`.
//! Source: https://balatrowiki.org/w/Tags

use crate::content::ids::TagId as T;
use crate::data::shop_weights::{TAG_ECONOMY_CAP, TAG_INVESTMENT};
use crate::hooks::PackKind;
use crate::packs::{PackSize, PackSpec};
use crate::tag::{TagCtx, TagDef};

/// Definitions of this file, listed in `content::TAG_SOURCES`.
pub static DEFS: &[TagDef] = &[
    TagDef {
        on_gain: Some(economy_on_gain),
        ..TagDef::new(T::Economy, "Economy Tag", 1)
    },
    TagDef {
        on_boss_defeated: Some(investment_on_boss),
        ..TagDef::new(T::Investment, "Investment Tag", 1)
    },
    TagDef {
        on_gain: Some(charm_on_gain),
        ..TagDef::new(T::Charm, "Charm Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(coupon_on_shop),
        ..TagDef::new(T::Coupon, "Coupon Tag", 1)
    },
    // Double has no hook of its own: `tag::gain` copies the next tag gained.
    TagDef::new(T::Double, "Double Tag", 1),
    TagDef {
        on_gain: Some(boss_on_gain),
        ..TagDef::new(T::Boss, "Boss Tag", 1)
    },
];

// --- Economy Tag: https://balatrowiki.org/w/Tags --- doubles the money, gain capped at $40.
fn economy_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    let gain = ctx.money().clamp(0, TAG_ECONOMY_CAP);
    ctx.add_money(gain);
    true
}

// --- Investment Tag --- $25 after the next Boss Blind is beaten (after the cash-out).
fn investment_on_boss(ctx: &mut TagCtx<'_>) -> bool {
    ctx.add_money(TAG_INVESTMENT);
    true
}

// --- Charm Tag --- a free Mega Arcana Pack right away.
fn charm_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    ctx.queue_pack(PackSpec {
        kind: PackKind::Arcana,
        size: PackSize::Mega,
    });
    true
}

// --- Coupon Tag --- the cards and packs of the next shop are free (not the voucher).
fn coupon_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_make_free();
    true
}

// --- Boss Tag --- rerolls the ante's Boss Blind.
fn boss_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    ctx.reroll_boss();
    true
}
