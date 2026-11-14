//! Typed ids of consumables ([`ConsumableId`], 52) and boss blinds ([`BossId`], 28).
//!
//! Orchestrator-owned. Hand-written, fixed order: variant order defines `index()` (observation
//! encoding), so append new variants only, never reorder. Sources: docs/mechanics/consumables.md
//! (Planets in hand-type order, Tarots in wiki number order 0-21, Spectrals in table order) and
//! docs/mechanics/bosses.md (23 regular bosses in table order, then the 5 Showdown bosses).

use crate::consumable::ConsumableKind;
use crate::enums::HandType;
use serde::{Deserialize, Serialize};

macro_rules! typed_ids {
    (@one $v:ident) => { 1 };
    ($(#[$tm:meta])* $name:ident { $($(#[$m:meta])* $v:ident),+ $(,)? }) => {
        $(#[$tm])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[repr(u8)]
        pub enum $name {
            $($(#[$m])* $v),+
        }

        impl $name {
            /// All ids in declaration order.
            pub const ALL: [$name; 0 $(+ typed_ids!(@one $v))+] = [$($name::$v),+];
            /// Number of ids.
            pub const COUNT: usize = Self::ALL.len();
            /// Stable 0-based index (position in `ALL`).
            pub const fn index(self) -> usize {
                self as usize
            }
        }
    };
}

typed_ids! {
    /// One consumable kind: 12 Planets, then 22 Tarots, then 18 Spectrals.
    ConsumableId {
        // --- Planets (12), in `HandType` order: `index() == hand.index()` ---
        /// Planet: High Card.
        Pluto,
        /// Planet: Pair.
        Mercury,
        /// Planet: Two Pair.
        Uranus,
        /// Planet: Three of a Kind.
        Venus,
        /// Planet: Straight.
        Saturn,
        /// Planet: Flush.
        Jupiter,
        /// Planet: Full House.
        Earth,
        /// Planet: Four of a Kind.
        Mars,
        /// Planet: Straight Flush.
        Neptune,
        /// Planet: Five of a Kind (secret).
        PlanetX,
        /// Planet: Flush House (secret).
        Ceres,
        /// Planet: Flush Five (secret).
        Eris,
        // --- Tarots (22), wiki numbers 0..21 ---
        /// Tarot 0.
        TheFool,
        /// Tarot 1.
        TheMagician,
        /// Tarot 2.
        TheHighPriestess,
        /// Tarot 3.
        TheEmpress,
        /// Tarot 4.
        TheEmperor,
        /// Tarot 5.
        TheHierophant,
        /// Tarot 6.
        TheLovers,
        /// Tarot 7.
        TheChariot,
        /// Tarot 8.
        Justice,
        /// Tarot 9.
        TheHermit,
        /// Tarot 10.
        TheWheelOfFortune,
        /// Tarot 11.
        Strength,
        /// Tarot 12.
        TheHangedMan,
        /// Tarot 13.
        Death,
        /// Tarot 14.
        Temperance,
        /// Tarot 15.
        TheDevil,
        /// Tarot 16.
        TheTower,
        /// Tarot 17.
        TheStar,
        /// Tarot 18.
        TheMoon,
        /// Tarot 19.
        TheSun,
        /// Tarot 20.
        Judgement,
        /// Tarot 21.
        TheWorld,
        // --- Spectrals (18) ---
        /// Spectral.
        Familiar,
        /// Spectral.
        Grim,
        /// Spectral.
        Incantation,
        /// Spectral.
        Talisman,
        /// Spectral.
        Aura,
        /// Spectral.
        Wraith,
        /// Spectral.
        Sigil,
        /// Spectral.
        Ouija,
        /// Spectral.
        Ectoplasm,
        /// Spectral.
        Immolate,
        /// Spectral.
        Ankh,
        /// Spectral.
        DejaVu,
        /// Spectral.
        Hex,
        /// Spectral.
        Trance,
        /// Spectral.
        Medium,
        /// Spectral.
        Cryptid,
        /// Spectral (packs only, 0.3%).
        TheSoul,
        /// Spectral (packs only, 0.3%).
        BlackHole,
    }
}

const FIRST_TAROT: usize = ConsumableId::TheFool.index();
const FIRST_SPECTRAL: usize = ConsumableId::Familiar.index();

impl ConsumableId {
    /// The 12 Planets (`PLANETS[h.index()]` levels hand `h`).
    pub const PLANETS: &'static [ConsumableId] = Self::ALL.split_at(FIRST_TAROT).0;
    /// The 22 Tarots.
    pub const TAROTS: &'static [ConsumableId] =
        Self::ALL.split_at(FIRST_SPECTRAL).0.split_at(FIRST_TAROT).1;
    /// The 18 Spectrals (The Soul and Black Hole included).
    pub const SPECTRALS: &'static [ConsumableId] = Self::ALL.split_at(FIRST_SPECTRAL).1;

    /// Tarot / Planet / Spectral.
    pub const fn kind(self) -> ConsumableKind {
        let i = self.index();
        if i < FIRST_TAROT {
            ConsumableKind::Planet
        } else if i < FIRST_SPECTRAL {
            ConsumableKind::Tarot
        } else {
            ConsumableKind::Spectral
        }
    }

    /// The hand a Planet levels (`None` for Tarots and Spectrals).
    pub const fn planet_hand(self) -> Option<HandType> {
        let i = self.index();
        if i < FIRST_TAROT {
            Some(HandType::ALL[i])
        } else {
            None
        }
    }

    /// The Planet of `hand` (Blue Seal, Telescope, `SideEffect::CreatePlanetFor`).
    pub const fn planet_for(hand: HandType) -> ConsumableId {
        Self::PLANETS[hand as usize]
    }

    /// Ids of one kind.
    pub const fn of_kind(kind: ConsumableKind) -> &'static [ConsumableId] {
        match kind {
            ConsumableKind::Planet => Self::PLANETS,
            ConsumableKind::Tarot => Self::TAROTS,
            ConsumableKind::Spectral => Self::SPECTRALS,
        }
    }
}

typed_ids! {
    /// One boss blind: 23 regular bosses, then 5 Showdown (finisher) bosses.
    BossId {
        /// Discards 2 random held cards per hand played.
        TheHook,
        /// Playing the ante's most played hand sets money to $0 (min ante 6).
        TheOx,
        /// First hand drawn face down (min ante 2).
        TheHouse,
        /// Requirement x4 (min ante 2).
        TheWall,
        /// Each drawn card 1 in 7 face down (min ante 2).
        TheWheel,
        /// Played hand loses a level (min ante 2).
        TheArm,
        /// Clubs debuffed.
        TheClub,
        /// Cards drawn face down after each hand played (min ante 2).
        TheFish,
        /// Must play exactly 5 cards.
        ThePsychic,
        /// Spades debuffed.
        TheGoad,
        /// Start with 0 discards (min ante 2).
        TheWater,
        /// Diamonds debuffed.
        TheWindow,
        /// Hand size -1.
        TheManacle,
        /// No repeated hand types this round (min ante 3).
        TheEye,
        /// Only one hand type this round (min ante 2).
        TheMouth,
        /// Face cards debuffed (min ante 4).
        ThePlant,
        /// Always draw 3 cards after play/discard (min ante 5).
        TheSerpent,
        /// Cards played this ante debuffed.
        ThePillar,
        /// Only 1 hand; requirement x1 (min ante 2).
        TheNeedle,
        /// Hearts debuffed.
        TheHead,
        /// Lose $1 per card played (min ante 3).
        TheTooth,
        /// Base chips and mult halved (min ante 2).
        TheFlint,
        /// Face cards drawn face down (min ante 2).
        TheMark,
        // --- Showdown (ante % 8 == 0) ---
        /// Jokers flipped and shuffled.
        AmberAcorn,
        /// All cards debuffed until a joker is sold.
        VerdantLeaf,
        /// Requirement x6.
        VioletVessel,
        /// One random joker disabled every hand.
        CrimsonHeart,
        /// One card always force-selected.
        CeruleanBell,
    }
}

impl BossId {
    /// Showdown (finisher) boss: only at antes divisible by 8.
    pub const fn is_showdown(self) -> bool {
        self.index() >= BossId::AmberAcorn.index()
    }
}

typed_ids! {
    /// One voucher: the 16 base vouchers each followed by their upgrade, in the wiki table order
    /// (docs/mechanics/vouchers.md). `requires()` names the base of an upgrade.
    VoucherId {
        /// +1 shop card slot.
        Overstock,
        /// +1 more shop card slot (needs Overstock).
        OverstockPlus,
        /// 25% off.
        ClearanceSale,
        /// 50% off (needs Clearance Sale).
        Liquidation,
        /// Foil/Holo/Poly 2x as likely.
        Hone,
        /// Foil/Holo/Poly 4x as likely (needs Hone).
        GlowUp,
        /// Rerolls start at $3.
        RerollSurplus,
        /// Rerolls start at $1 (needs Reroll Surplus).
        RerollGlut,
        /// +1 consumable slot.
        CrystalBall,
        /// Spectrals may appear in Arcana packs (needs Crystal Ball).
        OmenGlobe,
        /// Celestial packs hold the Planet of the most played hand.
        Telescope,
        /// Planets held give x1.5 Mult for their hand (needs Telescope).
        Observatory,
        /// +1 hand per round.
        Grabber,
        /// +1 more hand per round (needs Grabber).
        NachoTong,
        /// +1 discard per round.
        Wasteful,
        /// +1 more discard per round (needs Wasteful).
        Recyclomancy,
        /// Tarot weight 9.6.
        TarotMerchant,
        /// Tarot weight 32 (needs Tarot Merchant).
        TarotTycoon,
        /// Planet weight 9.6.
        PlanetMerchant,
        /// Planet weight 32 (needs Planet Merchant).
        PlanetTycoon,
        /// Interest cap $10.
        SeedMoney,
        /// Interest cap $20 (needs Seed Money).
        MoneyTree,
        /// Does nothing.
        Blank,
        /// +1 joker slot (needs Blank).
        Antimatter,
        /// Playing cards in the shop.
        MagicTrick,
        /// Shop playing cards may be enhanced/edition/sealed (needs Magic Trick).
        Illusion,
        /// -1 ante, -1 hand per round.
        Hieroglyph,
        /// -1 ante, -1 discard per round (needs Hieroglyph).
        Petroglyph,
        /// Reroll the boss once per ante for $10.
        DirectorsCut,
        /// Reroll the boss without limit for $10 each (needs Director's Cut).
        Retcon,
        /// +1 hand size.
        PaintBrush,
        /// +1 more hand size (needs Paint Brush).
        Palette,
    }
}

impl VoucherId {
    /// The base voucher an upgrade needs (`None` for the 16 base vouchers). Upgrades are the
    /// odd positions of the table, so the base is the id just before.
    pub const fn requires(self) -> Option<VoucherId> {
        if self.index() % 2 == 1 {
            Some(Self::ALL[self.index() - 1])
        } else {
            None
        }
    }

    /// An upgrade of a base voucher.
    pub const fn is_upgrade(self) -> bool {
        self.requires().is_some()
    }
}

typed_ids! {
    /// One skip tag, in the wiki table order (docs/mechanics/tags.md). `Double` is index 17.
    TagId {
        /// Free Uncommon Joker in the next shop.
        Uncommon,
        /// Free Rare Joker in the next shop.
        Rare,
        /// Next base-edition shop Joker is free and Negative (min ante 2).
        Negative,
        /// Same, Foil.
        Foil,
        /// Same, Holographic.
        Holographic,
        /// Same, Polychrome.
        Polychrome,
        /// +$25 after the next Boss Blind.
        Investment,
        /// One extra voucher in the next shop.
        Voucher,
        /// Rerolls the Boss Blind.
        Boss,
        /// Free Mega Standard Pack (min ante 2).
        Standard,
        /// Free Mega Arcana Pack.
        Charm,
        /// Free Mega Celestial Pack (min ante 2).
        Meteor,
        /// Free Mega Buffoon Pack (min ante 2).
        Buffoon,
        /// $1 per hand played this run (min ante 2).
        Handy,
        /// $1 per unused discard this run (min ante 2).
        Garbage,
        /// Free Spectral Pack (min ante 2).
        Ethereal,
        /// Cards and packs of the next shop are free.
        Coupon,
        /// Copies the next tag gained.
        Double,
        /// +3 hand size next round.
        Juggle,
        /// Rerolls start at $0 in the next shop.
        D6,
        /// Create up to 2 Common Jokers (min ante 2).
        TopUp,
        /// $5 per blind skipped this run.
        Speed,
        /// Upgrade a random hand by 3 levels (min ante 2).
        Orbital,
        /// Double the money, gain at most $40.
        Economy,
    }
}
