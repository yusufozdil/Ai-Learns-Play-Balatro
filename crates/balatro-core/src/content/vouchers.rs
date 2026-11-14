//! Vouchers (16 base + 16 upgraded). all 32 are defined here (Phase 5.0 prototypes + Phase 5C).
//! Guide: `docs/design/shop_packs_vouchers_tags.md`. Source: https://balatrowiki.org/w/Vouchers

use crate::content::ids::VoucherId as V;
use crate::data::prices::{
    INTEREST_CAP_MONEY_TREE, INTEREST_CAP_SEED_MONEY, REROLL_GLUT_BASE, REROLL_SURPLUS_BASE,
};
use crate::data::shop_weights::{WEIGHT_MERCHANT, WEIGHT_TYCOON};
use crate::economy::Discount;
use crate::voucher::{BossReroll, ShopMods, VoucherCtx, VoucherDef};

/// Definitions of this file, listed in `content::VOUCHER_SOURCES`.
pub static DEFS: &[VoucherDef] = &[
    VoucherDef {
        passive: ShopMods {
            card_slots: 1,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Overstock, "Overstock")
    },
    VoucherDef {
        passive: ShopMods {
            discount: Discount::Clearance,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::ClearanceSale, "Clearance Sale")
    },
    VoucherDef {
        on_redeem: Some(grabber),
        ..VoucherDef::new(V::Grabber, "Grabber")
    },
    VoucherDef {
        on_redeem: Some(seed_money),
        ..VoucherDef::new(V::SeedMoney, "Seed Money")
    },
    VoucherDef {
        on_redeem: Some(crystal_ball),
        ..VoucherDef::new(V::CrystalBall, "Crystal Ball")
    },
    VoucherDef {
        passive: ShopMods {
            telescope: true,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Telescope, "Telescope")
    },
    VoucherDef {
        passive: ShopMods {
            boss_reroll: BossReroll::OncePerAnte,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::DirectorsCut, "Director's Cut")
    },
    VoucherDef {
        passive: ShopMods {
            boss_reroll: BossReroll::Unlimited,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Retcon, "Retcon")
    },
    VoucherDef {
        passive: ShopMods {
            card_slots: 1,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::OverstockPlus, "Overstock Plus")
    },
    VoucherDef {
        passive: ShopMods {
            discount: Discount::Liquidation,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Liquidation, "Liquidation")
    },
    VoucherDef {
        passive: ShopMods {
            edition_mult: 2,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Hone, "Hone")
    },
    VoucherDef {
        passive: ShopMods {
            edition_mult: 4,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::GlowUp, "Glow Up")
    },
    VoucherDef {
        passive: ShopMods {
            reroll_base: Some(REROLL_SURPLUS_BASE),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::RerollSurplus, "Reroll Surplus")
    },
    VoucherDef {
        passive: ShopMods {
            reroll_base: Some(REROLL_GLUT_BASE),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::RerollGlut, "Reroll Glut")
    },
    VoucherDef {
        passive: ShopMods {
            omen_globe: true,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::OmenGlobe, "Omen Globe")
    },
    VoucherDef {
        passive: ShopMods {
            observatory: true,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Observatory, "Observatory")
    },
    VoucherDef {
        on_redeem: Some(nacho_tong),
        ..VoucherDef::new(V::NachoTong, "Nacho Tong")
    },
    VoucherDef {
        on_redeem: Some(wasteful),
        ..VoucherDef::new(V::Wasteful, "Wasteful")
    },
    VoucherDef {
        on_redeem: Some(recyclomancy),
        ..VoucherDef::new(V::Recyclomancy, "Recyclomancy")
    },
    VoucherDef {
        passive: ShopMods {
            tarot_weight: Some(WEIGHT_MERCHANT),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::TarotMerchant, "Tarot Merchant")
    },
    VoucherDef {
        passive: ShopMods {
            tarot_weight: Some(WEIGHT_TYCOON),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::TarotTycoon, "Tarot Tycoon")
    },
    VoucherDef {
        passive: ShopMods {
            planet_weight: Some(WEIGHT_MERCHANT),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::PlanetMerchant, "Planet Merchant")
    },
    VoucherDef {
        passive: ShopMods {
            planet_weight: Some(WEIGHT_TYCOON),
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::PlanetTycoon, "Planet Tycoon")
    },
    VoucherDef {
        on_redeem: Some(money_tree),
        ..VoucherDef::new(V::MoneyTree, "Money Tree")
    },
    VoucherDef::new(V::Blank, "Blank"),
    VoucherDef {
        on_redeem: Some(antimatter),
        ..VoucherDef::new(V::Antimatter, "Antimatter")
    },
    VoucherDef {
        passive: ShopMods {
            playing_cards: true,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::MagicTrick, "Magic Trick")
    },
    VoucherDef {
        passive: ShopMods {
            illusion: true,
            ..ShopMods::NONE
        },
        ..VoucherDef::new(V::Illusion, "Illusion")
    },
    VoucherDef {
        on_redeem: Some(hieroglyph),
        ..VoucherDef::new(V::Hieroglyph, "Hieroglyph")
    },
    VoucherDef {
        on_redeem: Some(petroglyph),
        ..VoucherDef::new(V::Petroglyph, "Petroglyph")
    },
    VoucherDef {
        on_redeem: Some(paint_brush),
        ..VoucherDef::new(V::PaintBrush, "Paint Brush")
    },
    VoucherDef {
        on_redeem: Some(palette),
        ..VoucherDef::new(V::Palette, "Palette")
    },
];

// --- Grabber: https://balatrowiki.org/w/Vouchers ---
const GRABBER_HANDS: i32 = 1;

fn grabber(ctx: &mut VoucherCtx<'_>) {
    ctx.add_hands(GRABBER_HANDS);
}

// --- Seed Money: https://balatrowiki.org/w/Vouchers ---
fn seed_money(ctx: &mut VoucherCtx<'_>) {
    ctx.raise_interest_cap(INTEREST_CAP_SEED_MONEY);
}

// --- Crystal Ball: https://balatrowiki.org/w/Vouchers ---
const CRYSTAL_BALL_SLOTS: i32 = 1;

fn crystal_ball(ctx: &mut VoucherCtx<'_>) {
    ctx.add_consumable_slots(CRYSTAL_BALL_SLOTS);
}

// --- Nacho Tong: https://balatrowiki.org/w/Vouchers ---
fn nacho_tong(ctx: &mut VoucherCtx<'_>) {
    ctx.add_hands(1);
}

// --- Wasteful / Recyclomancy: https://balatrowiki.org/w/Vouchers ---
fn wasteful(ctx: &mut VoucherCtx<'_>) {
    ctx.add_discards(1);
}

fn recyclomancy(ctx: &mut VoucherCtx<'_>) {
    ctx.add_discards(1);
}

// --- Money Tree: https://balatrowiki.org/w/Vouchers ---
fn money_tree(ctx: &mut VoucherCtx<'_>) {
    ctx.raise_interest_cap(INTEREST_CAP_MONEY_TREE);
}

// --- Antimatter: https://balatrowiki.org/w/Vouchers ---
fn antimatter(ctx: &mut VoucherCtx<'_>) {
    ctx.add_joker_slots(1);
}

// --- Hieroglyph / Petroglyph: https://balatrowiki.org/w/Vouchers ---
fn hieroglyph(ctx: &mut VoucherCtx<'_>) {
    ctx.add_ante(-1);
    ctx.add_hands(-1);
}

fn petroglyph(ctx: &mut VoucherCtx<'_>) {
    ctx.add_ante(-1);
    ctx.add_discards(-1);
}

// --- Paint Brush / Palette: https://balatrowiki.org/w/Vouchers ---
fn paint_brush(ctx: &mut VoucherCtx<'_>) {
    ctx.add_hand_size(1);
}

fn palette(ctx: &mut VoucherCtx<'_>) {
    ctx.add_hand_size(1);
}
