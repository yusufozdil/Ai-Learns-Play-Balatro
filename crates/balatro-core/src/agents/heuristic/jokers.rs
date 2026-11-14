//! Static joker priors: the part of a joker's value that a scoring simulation cannot see.
//!
//! The market expert values a joker as `sim_gain + prior`. `sim_gain` is the gain in
//! `log2(best hand score)` measured by previewing sampled hands with and without the joker
//! (`valuer.rs`); it already covers every +chips/+mult/xmult/retrigger/rule effect. `prior` (same
//! unit: doublings of the score) covers what previews cannot show:
//!
//! * economy and utility (money, slots, hand size, discards, consumable/joker generation);
//! * scaling jokers that are weak today but grow (the preview only sees today's value);
//! * effects tied to a state the sampled hands do not have (final hand, empty discards, bosses);
//! * jokers whose cost the simulation cannot see (they destroy other jokers, shrink the hand).
//!
//! Numbers are hand-set from the effect text in `docs/mechanics/jokers.csv` and are tuned by the
//! evaluation runs, not derived.

use crate::content::jokers::JokerId;
use crate::enums::{Edition, HandType};

/// Prior value in doublings of the score (0 = nothing beyond what the preview shows).
pub fn prior(id: JokerId) -> f64 {
    use JokerId::*;
    match id {
        // --- economy ---
        GoldenJoker => 0.60,
        Rocket => 0.45,
        Cloud9 => 0.30,
        ToTheMoon => 0.30,
        Satellite => 0.30,
        Egg => 0.20,
        DelayedGratification => 0.15,
        BusinessCard => 0.15,
        ReservedParking => 0.15,
        MailInRebate => 0.10,
        Matador => 0.10,
        CreditCard => 0.20,
        Perkeo => 0.50,
        TradingCard => 0.05,
        GiftCard => 0.15,
        RoughGem => 0.10,
        // --- slots, hand size, discards, hands ---
        Juggler => 0.45,
        Drunkard => 0.35,
        TurtleBean => 0.45,
        Troubadour => 0.05,
        MerryAndy => 0.0,
        Burglar => 0.20,
        Stuntman => -0.45,
        // --- generators and utility ---
        Cartomancer => 0.15,
        RiffRaff => 0.25,
        EightBall => 0.15,
        Hallucination => 0.10,
        Superposition => 0.10,
        Vagabond => 0.10,
        Seance => 0.10,
        SixthSense => 0.10,
        MarbleJoker => 0.25,
        Certificate => 0.10,
        DNA => 0.25,
        Hiker => 0.55,
        SpaceJoker => 0.55,
        BurntJoker => 0.55,
        Astronomer => 0.30,
        ChaosTheClown => 0.20,
        DietCola => 0.10,
        InvisibleJoker => 0.35,
        Luchador => 0.10,
        MrBones => 0.55,
        Chicot => 1.20,
        Showman => 0.05,
        OopsAll6s => 0.10,
        MidasMask => 0.15,
        // --- scalers (value today is small, growth is the point) ---
        GreenJoker => 0.50,
        RideTheBus => 0.40,
        Runner => 0.50,
        SquareJoker => 0.50,
        WeeJoker => 0.55,
        FlashCard => 0.30,
        RedCard => 0.20,
        SpareTrousers => 0.50,
        FortuneTeller => 0.30,
        Supernova => 0.30,
        Constellation => 0.90,
        Vampire => 0.80,
        Hologram => 0.70,
        LuckyCat => 0.60,
        Campfire => 0.50,
        Obelisk => 0.70,
        GlassJoker => 0.35,
        Canio => 0.90,
        Yorick => 0.90,
        Ramen => 0.30,
        Throwback => 0.25,
        Popcorn => 0.0,
        Castle => 0.35,
        // --- conditional states the samples do not have ---
        Acrobat => 0.70,
        LoyaltyCard => 0.55,
        Dusk => 0.40,
        MysticSummit => 0.25,
        Banner => 0.0,
        // --- harmful side effects ---
        CeremonialDagger => -0.50,
        Madness => -0.60,
        // Blueprint/Brainstorm are valued by the preview at their best slot.
        _ => 0.0,
    }
}

/// Extra value of an edition that the preview cannot show: Negative is a free joker slot.
pub fn edition_prior(e: Edition) -> f64 {
    match e {
        Edition::Negative => 0.70,
        _ => 0.0,
    }
}

/// Copiers copy a neighbour: their preview value depends on the slot, so the market tries every
/// slot for them.
pub fn is_copier(id: JokerId) -> bool {
    matches!(id, JokerId::Blueprint | JokerId::Brainstorm)
}

/// Hand types a joker builds around, used for the strategy's target hand weights.
pub fn target_hands(id: JokerId) -> &'static [HandType] {
    use HandType::*;
    use JokerId::*;
    match id {
        JollyJoker | SlyJoker | TheDuo | SpareTrousers => &[Pair, TwoPair, ThreeOfAKind, FullHouse],
        ZanyJoker | WilyJoker | TheTrio => &[ThreeOfAKind, FullHouse, FourOfAKind],
        MadJoker | CleverJoker => &[TwoPair, FullHouse],
        CrazyJoker | DeviousJoker | TheOrder | Runner | Shortcut => &[Straight, StraightFlush],
        DrollJoker | CraftyJoker | TheTribe | FlowerPot | Arrowhead | OnyxAgate | Bloodstone
        | Blackboard | SmearedJoker => &[Flush, StraightFlush],
        TheFamily => &[FourOfAKind],
        HalfJoker => &[HighCard, Pair],
        _ => &[],
    }
}

/// Value of a skip tag, in doublings (see `market.rs::blind_action`).
pub fn tag_value(
    id: crate::content::ids::TagId,
    money: i32,
    hands_played: u32,
    skipped: u32,
    joker_room: bool,
) -> f64 {
    use crate::content::ids::TagId::*;
    match id {
        Investment => 0.9,
        Economy => f64::from(money.clamp(0, 40)) * 0.05,
        Negative => 1.0,
        Rare => 0.9,
        Polychrome => 0.7,
        Holographic => 0.5,
        Foil => 0.4,
        Uncommon => 0.4,
        Voucher => 0.7,
        Buffoon => {
            if joker_room {
                0.9
            } else {
                0.4
            }
        }
        Meteor => 0.8,
        Charm => 0.4,
        Standard => 0.15,
        Ethereal => 0.3,
        Orbital => 0.6,
        Speed => 0.15 * f64::from(skipped + 1) + 0.2,
        Handy => 0.04 * f64::from(hands_played),
        Garbage => 0.1,
        Juggle => 0.1,
        TopUp => 0.35,
        D6 => 0.15,
        Coupon => 0.35,
        Boss => 0.05,
        Double => 0.2,
    }
}
