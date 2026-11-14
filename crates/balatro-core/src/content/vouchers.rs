//! Vouchers (16 base + 16 upgraded). Phase 5.0 defines the prototypes (Overstock, Clearance
//! Sale, Grabber, Seed Money, Crystal Ball, Telescope, Director's Cut, Retcon); Phase 5C adds
//! the other 24 here.
//! Guide: `docs/design/shop_packs_vouchers_tags.md`. Source: https://balatrowiki.org/w/Vouchers

use crate::content::ids::VoucherId as V;
use crate::data::prices::INTEREST_CAP_SEED_MONEY;
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
