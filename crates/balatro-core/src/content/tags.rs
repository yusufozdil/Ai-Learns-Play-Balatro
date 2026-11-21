//! Skip tags (24). Phase 5.0 defines the prototypes (Economy, Investment, Charm, Coupon, Double,
//! Boss); Phase 5B adds the other 18 here. Guide: `docs/design/shop_packs_vouchers_tags.md`.
//! Source: https://balatrowiki.org/w/Tags

use crate::content::ids::TagId as T;
use crate::data::shop_weights::{
    TAG_ECONOMY_CAP, TAG_INVESTMENT, TAG_JUGGLE_HAND_SIZE, TAG_ORBITAL_LEVELS, TAG_SPEED_PER_SKIP,
    TAG_TOP_UP_JOKERS,
};
use crate::enums::{Edition, Rarity};
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
    TagDef {
        on_shop_enter: Some(uncommon_on_shop),
        ..TagDef::new(T::Uncommon, "Uncommon Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(rare_on_shop),
        ..TagDef::new(T::Rare, "Rare Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(negative_on_shop),
        ..TagDef::new(T::Negative, "Negative Tag", 2)
    },
    TagDef {
        on_shop_enter: Some(foil_on_shop),
        ..TagDef::new(T::Foil, "Foil Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(holographic_on_shop),
        ..TagDef::new(T::Holographic, "Holographic Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(polychrome_on_shop),
        ..TagDef::new(T::Polychrome, "Polychrome Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(voucher_on_shop),
        ..TagDef::new(T::Voucher, "Voucher Tag", 1)
    },
    TagDef {
        on_gain: Some(standard_on_gain),
        ..TagDef::new(T::Standard, "Standard Tag", 2)
    },
    TagDef {
        on_gain: Some(meteor_on_gain),
        ..TagDef::new(T::Meteor, "Meteor Tag", 2)
    },
    TagDef {
        on_gain: Some(buffoon_on_gain),
        ..TagDef::new(T::Buffoon, "Buffoon Tag", 2)
    },
    TagDef {
        on_gain: Some(handy_on_gain),
        ..TagDef::new(T::Handy, "Handy Tag", 2)
    },
    TagDef {
        on_gain: Some(garbage_on_gain),
        ..TagDef::new(T::Garbage, "Garbage Tag", 2)
    },
    TagDef {
        on_gain: Some(ethereal_on_gain),
        ..TagDef::new(T::Ethereal, "Ethereal Tag", 2)
    },
    TagDef {
        on_round_start: Some(juggle_on_round_start),
        ..TagDef::new(T::Juggle, "Juggle Tag", 1)
    },
    TagDef {
        on_shop_enter: Some(d6_on_shop),
        ..TagDef::new(T::D6, "D6 Tag", 1)
    },
    TagDef {
        on_gain: Some(top_up_on_gain),
        ..TagDef::new(T::TopUp, "Top-up Tag", 2)
    },
    TagDef {
        on_gain: Some(speed_on_gain),
        ..TagDef::new(T::Speed, "Speed Tag", 1)
    },
    TagDef {
        on_gain: Some(orbital_on_gain),
        ..TagDef::new(T::Orbital, "Orbital Tag", 2)
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

// --- Uncommon / Rare Tag: https://balatrowiki.org/w/Tags --- a free extra joker in the next shop.
fn uncommon_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_add_free_joker(Rarity::Uncommon)
}

fn rare_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_add_free_joker(Rarity::Rare)
}

// --- Negative / Foil / Holographic / Polychrome Tag --- the first base-edition shop joker gets
// the edition and is free; the tag waits for the next shop if there is none.
fn negative_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_edition_first_base_joker(Edition::Negative)
}

fn foil_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_edition_first_base_joker(Edition::Foil)
}

fn holographic_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_edition_first_base_joker(Edition::Holographic)
}

fn polychrome_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_edition_first_base_joker(Edition::Polychrome)
}

// --- Voucher Tag --- one extra voucher in the next shop.
fn voucher_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_add_voucher()
}

fn open_mega(ctx: &mut TagCtx<'_>, kind: PackKind) -> bool {
    ctx.queue_pack(PackSpec {
        kind,
        size: PackSize::Mega,
    });
    true
}

// --- Standard / Meteor / Buffoon Tag --- a free Mega pack right away.
fn standard_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    open_mega(ctx, PackKind::Standard)
}

fn meteor_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    open_mega(ctx, PackKind::Celestial)
}

fn buffoon_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    open_mega(ctx, PackKind::Buffoon)
}

// --- Handy Tag --- $1 per hand played this run.
fn handy_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    let n = i32::try_from(ctx.state().stats.hands_played).unwrap_or(i32::MAX);
    ctx.add_money(n);
    true
}

// --- Garbage Tag --- $1 per unused discard this run.
fn garbage_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    let n = i32::try_from(ctx.state().stats.unused_discards).unwrap_or(i32::MAX);
    ctx.add_money(n);
    true
}

// --- Ethereal Tag --- a free Spectral Pack (Normal size).
fn ethereal_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    ctx.queue_pack(PackSpec {
        kind: PackKind::Spectral,
        size: PackSize::Normal,
    });
    true
}

// --- Juggle Tag --- +3 hand size for the next round.
fn juggle_on_round_start(ctx: &mut TagCtx<'_>) -> bool {
    ctx.add_round_hand_size(TAG_JUGGLE_HAND_SIZE);
    true
}

// --- D6 Tag --- rerolls start at $0 in the next shop.
fn d6_on_shop(ctx: &mut TagCtx<'_>) -> bool {
    ctx.shop_set_reroll_base(0);
    true
}

// --- Top-up Tag --- up to two Common jokers (as many as fit).
fn top_up_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    for _ in 0..TAG_TOP_UP_JOKERS {
        ctx.create_joker(Rarity::Common);
    }
    true
}

// --- Speed Tag --- $5 per blind skipped this run (counts its own skip).
fn speed_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    let n = i32::try_from(ctx.state().stats.run.blinds_skipped).unwrap_or(i32::MAX);
    ctx.add_money(TAG_SPEED_PER_SKIP.saturating_mul(n));
    true
}

// --- Orbital Tag: https://balatrowiki.org/w/Orbital_Tag --- the hand picked when it was offered
// gains three levels.
fn orbital_on_gain(ctx: &mut TagCtx<'_>) -> bool {
    if let Some(h) = ctx.aux_hand() {
        ctx.level_up(h, TAG_ORBITAL_LEVELS);
    }
    true
}
