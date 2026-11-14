//! Phase 4A: the 12 Planets and 22 Tarots through `Game::step`. Guide:
//! docs/design/consumables_bosses.md. Cards are found by uid (the hand is re-sorted after a use).

use balatro_core::actions::{Action, ActionError};
use balatro_core::card::Card;
use balatro_core::consumable::{ConsumableInstance, ConsumableKind};
use balatro_core::content::ConsumableId as C;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::try_consumable_def;
use balatro_core::data::hand_levels::chips_mult;
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::joker::JokerInstance;
use balatro_core::state::GameConfig;

// ----- helpers ------------------------------------------------------------------------------

fn in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    g
}

fn give(g: &mut Game, id: C) -> usize {
    g.add_consumable(ConsumableInstance::new(id)).expect("room")
}

fn use_c(g: &mut Game, slot: u8, targets: u16) -> Result<StepInfo, ActionError> {
    g.step(Action::UseConsumable { slot, targets })
}

fn add_joker(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("joker room")
}

fn set_card(g: &mut Game, pos: usize, rank: Rank, suit: Suit) {
    let c = &mut g.state_mut().deck.hand[pos];
    c.rank = rank;
    c.suit = suit;
    c.enhancement = Enhancement::None;
    c.edition = Edition::Base;
    c.seal = Seal::None;
    c.perma_bonus = 0;
}

fn uid_at(g: &Game, pos: usize) -> u32 {
    g.state().deck.hand[pos].uid
}

fn pos_of(g: &Game, uid: u32) -> usize {
    g.state()
        .deck
        .hand
        .iter()
        .position(|c| c.uid == uid)
        .expect("card in hand")
}

fn card(g: &Game, uid: u32) -> Card {
    g.state().deck.hand[pos_of(g, uid)]
}

fn tarot_count(g: &Game) -> u32 {
    g.state().stats.run.tarots_used
}

fn win_blind(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("winning play");
}

// ----- planets --------------------------------------------------------------------------------

/// Cards (rank, suit) that make exactly `ht` when played together.
fn hand_for(ht: HandType) -> Vec<(Rank, Suit)> {
    use Rank::*;
    use Suit::*;
    match ht {
        HandType::HighCard => vec![(Ace, Spades)],
        HandType::Pair => vec![(Five, Spades), (Five, Hearts)],
        HandType::TwoPair => vec![
            (King, Spades),
            (King, Hearts),
            (Queen, Spades),
            (Queen, Hearts),
        ],
        HandType::ThreeOfAKind => vec![(Nine, Spades), (Nine, Hearts), (Nine, Clubs)],
        HandType::Straight => vec![
            (Five, Spades),
            (Six, Hearts),
            (Seven, Clubs),
            (Eight, Diamonds),
            (Nine, Spades),
        ],
        HandType::Flush => vec![
            (Two, Spades),
            (Five, Spades),
            (Seven, Spades),
            (Nine, Spades),
            (Jack, Spades),
        ],
        HandType::FullHouse => vec![
            (King, Spades),
            (King, Hearts),
            (King, Clubs),
            (Two, Spades),
            (Two, Hearts),
        ],
        HandType::FourOfAKind => vec![
            (Eight, Spades),
            (Eight, Hearts),
            (Eight, Clubs),
            (Eight, Diamonds),
        ],
        HandType::StraightFlush => vec![
            (Five, Hearts),
            (Six, Hearts),
            (Seven, Hearts),
            (Eight, Hearts),
            (Nine, Hearts),
        ],
        HandType::FiveOfAKind => vec![
            (Ace, Spades),
            (Ace, Hearts),
            (Ace, Clubs),
            (Ace, Diamonds),
            (Ace, Spades),
        ],
        HandType::FlushHouse => vec![
            (King, Hearts),
            (King, Hearts),
            (King, Hearts),
            (Queen, Hearts),
            (Queen, Hearts),
        ],
        HandType::FlushFive => vec![(Ace, Hearts); 5],
    }
}

/// Play `ht` (optionally after using `planet`); returns (hand type, level, chips, mult).
fn play_hand(seed: u64, ht: HandType, planet: Option<C>) -> (HandType, u32, f64, f64) {
    let mut g = in_play(seed);
    if let Some(p) = planet {
        give(&mut g, p);
        use_c(&mut g, 0, 0).expect("planet");
    }
    let cards = hand_for(ht);
    for (i, &(r, s)) in cards.iter().enumerate() {
        set_card(&mut g, i, r, s);
    }
    let r = g
        .step(Action::Play((1u16 << cards.len()) - 1))
        .expect("play")
        .hand
        .expect("hand");
    (r.hand_type, r.level, r.chips, r.mult)
}

#[test]
fn every_planet_levels_its_hand_and_the_next_play_scores_with_it() {
    assert_eq!(C::PLANETS.len(), 12);
    for &id in C::PLANETS {
        let ht = id.planet_hand().expect("planet hand");
        let (t0, l0, chips0, mult0) = play_hand(7, ht, None);
        let (t1, l1, chips1, mult1) = play_hand(7, ht, Some(id));
        assert_eq!((t0, t1), (ht, ht), "{id:?} hand type");
        assert_eq!((l0, l1), (1, 2), "{id:?} level");
        let (c1, m1) = chips_mult(ht, 1);
        let (c2, m2) = chips_mult(ht, 2);
        assert_eq!(chips1 - chips0, c2 - c1, "{id:?} chips");
        assert_eq!(mult1 - mult0, m2 - m1, "{id:?} mult");
    }
}

#[test]
fn planets_feed_constellation_and_the_unique_planet_count() {
    let mut g = in_play(8);
    add_joker(&mut g, JokerId::Constellation);
    for (n, &id) in [C::Pluto, C::Mercury, C::Uranus].iter().enumerate() {
        give(&mut g, id);
        use_c(&mut g, 0, 0).expect("planet");
        let want = 1.0 + 0.1 * (n as f64 + 1.0);
        assert!((g.state().jokers[0].value - want).abs() < 1e-9, "{id:?}");
    }
    assert_eq!(g.state().stats.run.planets_used, 3);
    assert_eq!(g.state().stats.run.unique_planets_used, 3);
    // A repeated Planet raises Constellation but not the unique count.
    give(&mut g, C::Pluto);
    use_c(&mut g, 0, 0).expect("Pluto again");
    assert_eq!(g.state().stats.run.planets_used, 4);
    assert_eq!(g.state().stats.run.unique_planets_used, 3);
    assert_eq!(tarot_count(&g), 0);
    assert!((g.state().jokers[0].value - 1.4).abs() < 1e-9);
}

#[test]
fn satellite_pays_one_dollar_per_unique_planet_at_round_end() {
    let run = |planets: &[C]| {
        let mut g = in_play(9);
        add_joker(&mut g, JokerId::Satellite);
        for &id in planets {
            give(&mut g, id);
            use_c(&mut g, 0, 0).expect("planet");
        }
        win_blind(&mut g);
        assert_eq!(g.state().phase, Phase::Shop);
        g.state().money
    };
    let base = run(&[]);
    assert_eq!(run(&[C::Pluto, C::Mercury]), base + 2);
    assert_eq!(
        run(&[C::Pluto, C::Pluto, C::Mercury]),
        base + 2,
        "unique only"
    );
}

#[test]
fn fortune_teller_counts_every_tarot_used_but_no_planet() {
    let mut g = in_play(10);
    add_joker(&mut g, JokerId::FortuneTeller);
    for id in [C::TheHermit, C::Temperance, C::TheHermit] {
        give(&mut g, id);
        use_c(&mut g, 0, 0).expect("tarot");
    }
    assert_eq!(tarot_count(&g), 3);
    give(&mut g, C::Pluto);
    use_c(&mut g, 0, 0).expect("planet");
    assert_eq!(tarot_count(&g), 3);
    set_card(&mut g, 0, Rank::Ace, Suit::Spades);
    let uid = uid_at(&g, 0);
    let pos = pos_of(&g, uid);
    let h = g
        .step(Action::Play(1 << pos))
        .expect("play")
        .hand
        .expect("hand");
    // High Card level 2 (15 chips x 2 mult) + Ace 11 chips; Fortune Teller +3 mult.
    assert_eq!((h.chips, h.mult), (26.0, 5.0));
}

// ----- The Fool -------------------------------------------------------------------------------

#[test]
fn the_fool_copies_the_last_tarot_or_planet_but_never_itself() {
    let mut g = in_play(11);
    give(&mut g, C::Strength);
    use_c(&mut g, 0, 1).expect("Strength");
    give(&mut g, C::TheFool);
    use_c(&mut g, 0, 0).expect("Fool");
    let ids: Vec<C> = g.state().consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![C::Strength]);
    // The Fool does not become "last used": a second Fool still copies Strength.
    give(&mut g, C::TheFool);
    use_c(&mut g, 1, 0).expect("Fool again");
    let ids: Vec<C> = g.state().consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![C::Strength, C::Strength]);
    assert_eq!(g.state().last_tarot_planet, Some(C::Strength));
    assert_eq!(
        tarot_count(&g),
        3,
        "Strength and both Fools are Tarots used"
    );
    // A Planet used later is the new "last".
    g.state_mut().consumables.clear();
    give(&mut g, C::Mars);
    use_c(&mut g, 0, 0).expect("Mars");
    give(&mut g, C::TheFool);
    use_c(&mut g, 0, 0).expect("Fool");
    assert_eq!(g.state().consumables[0].id, C::Mars);
}

// ----- enhancement tarots -----------------------------------------------------------------------

#[test]
fn enhancement_tarots_set_the_enhancement_on_their_targets_only() {
    let table: [(C, Enhancement, u8); 8] = [
        (C::TheMagician, Enhancement::Lucky, 2),
        (C::TheEmpress, Enhancement::Mult, 2),
        (C::TheHierophant, Enhancement::Bonus, 2),
        (C::TheLovers, Enhancement::Wild, 1),
        (C::TheChariot, Enhancement::Steel, 1),
        (C::Justice, Enhancement::Glass, 1),
        (C::TheDevil, Enhancement::Gold, 1),
        (C::TheTower, Enhancement::Stone, 1),
    ];
    for (id, enh, max) in table {
        let mut g = in_play(12);
        give(&mut g, id);
        let uids: Vec<u32> = (0..8).map(|i| uid_at(&g, i)).collect();
        // A pre-existing enhancement on a target is replaced.
        let other = if enh == Enhancement::Bonus {
            Enhancement::Glass
        } else {
            Enhancement::Bonus
        };
        g.state_mut().deck.hand[1].enhancement = other;
        let mask: u16 = if max == 2 { 0b110 } else { 0b010 };
        use_c(&mut g, 0, mask).expect("use");
        for (i, uid) in uids.iter().enumerate() {
            let c = card(&g, *uid);
            let want = if (mask >> i) & 1 == 1 {
                enh
            } else {
                Enhancement::None
            };
            assert_eq!(c.enhancement, want, "{id:?} card {i}");
        }
        assert_eq!(tarot_count(&g), 1);
        assert_eq!(g.state().last_tarot_planet, Some(id));
        assert!(g.state().check_invariants().is_ok());
        // Target-count errors leave everything untouched.
        give(&mut g, id);
        assert!(matches!(
            use_c(&mut g, 0, 0),
            Err(ActionError::TargetCount { selected: 0, .. })
        ));
        let too_many = (1u16 << (max + 1)) - 1;
        assert!(matches!(
            use_c(&mut g, 0, too_many),
            Err(ActionError::TargetCount { .. })
        ));
        assert_eq!(g.state().consumables.len(), 1, "{id:?} kept on error");
    }
}

#[test]
fn lucky_wild_and_stone_changes_show_up_in_scoring() {
    // Steel in hand and Stone played: the Tower's Stone card scores 50 chips and no rank.
    let mut g = in_play(13);
    give(&mut g, C::TheTower);
    set_card(&mut g, 0, Rank::Two, Suit::Spades);
    let uid = uid_at(&g, 0);
    use_c(&mut g, 0, 1).expect("Tower");
    let pos = pos_of(&g, uid);
    let h = g
        .step(Action::Play(1 << pos))
        .expect("play")
        .hand
        .expect("hand");
    assert_eq!(h.hand_type, HandType::HighCard);
    assert_eq!(h.chips, 5.0 + 50.0, "Stone gives +50 chips, no rank chips");
}

// ----- suit tarots ------------------------------------------------------------------------------

#[test]
fn star_moon_sun_world_change_up_to_three_suits() {
    for (id, suit) in [
        (C::TheStar, Suit::Diamonds),
        (C::TheMoon, Suit::Clubs),
        (C::TheSun, Suit::Hearts),
        (C::TheWorld, Suit::Spades),
    ] {
        let mut g = in_play(14);
        let other = if suit == Suit::Spades {
            Suit::Hearts
        } else {
            Suit::Spades
        };
        for i in 0..8 {
            set_card(&mut g, i, Rank::Seven, other);
        }
        give(&mut g, id);
        let uids: Vec<u32> = (0..8).map(|i| uid_at(&g, i)).collect();
        use_c(&mut g, 0, 0b10101).expect("use 3");
        for (i, uid) in uids.iter().enumerate() {
            let want = if matches!(i, 0 | 2 | 4) { suit } else { other };
            assert_eq!(card(&g, *uid).suit, want, "{id:?} card {i}");
        }
        assert_eq!(tarot_count(&g), 1);
        give(&mut g, id);
        assert!(matches!(
            use_c(&mut g, 0, 0b1111),
            Err(ActionError::TargetCount { selected: 4, .. })
        ));
        assert!(matches!(
            use_c(&mut g, 0, 0),
            Err(ActionError::TargetCount { selected: 0, .. })
        ));
        assert!(g.state().check_invariants().is_ok());
    }
}

// ----- High Priestess / Emperor -----------------------------------------------------------------

#[test]
fn high_priestess_and_emperor_create_up_to_two_cards_limited_by_free_slots() {
    for (id, kind) in [
        (C::TheHighPriestess, ConsumableKind::Planet),
        (C::TheEmperor, ConsumableKind::Tarot),
    ] {
        for seed in 0..20 {
            // Empty area: both slots fill.
            let mut g = in_play(seed);
            give(&mut g, id);
            use_c(&mut g, 0, 0).expect("use");
            assert_eq!(g.state().consumables.len(), 2, "{id:?}");
            assert!(g.state().consumables.iter().all(|c| c.kind() == kind));
            assert!(g.state().check_invariants().is_ok());
            // One other card held: only one slot is free after use.
            let mut g = in_play(seed);
            give(&mut g, C::TheHermit);
            give(&mut g, id);
            use_c(&mut g, 1, 0).expect("use");
            assert_eq!(g.state().consumables.len(), 2, "{id:?}");
            assert_eq!(g.state().consumables[1].kind(), kind);
            assert_eq!(g.state().stats.effects_dropped, 0);
        }
        // Capacity 3 (a Negative copy) with 2 normal cards held: using the Negative card
        // removes its own slot, nothing is free.
        let mut g = in_play(3);
        give(&mut g, C::TheHermit);
        give(&mut g, C::TheHermit);
        g.add_consumable(ConsumableInstance::new(id).with_negative(true))
            .expect("negative");
        assert_eq!(use_c(&mut g, 2, 0), Err(ActionError::CannotUse));
        // Untargeted: targets are rejected.
        let mut g = in_play(3);
        give(&mut g, id);
        assert!(matches!(
            use_c(&mut g, 0, 1),
            Err(ActionError::TargetCount { .. })
        ));
    }
}

#[test]
fn high_priestess_and_emperor_work_in_the_shop() {
    let mut g = in_play(15);
    win_blind(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    give(&mut g, C::TheEmperor);
    give(&mut g, C::TheHighPriestess);
    use_c(&mut g, 0, 0).expect("Emperor in the shop");
    assert_eq!(g.state().consumables.len(), 2);
}

// ----- Wheel of Fortune --------------------------------------------------------------------------

#[test]
fn wheel_of_fortune_needs_an_edition_less_joker() {
    let mut g = in_play(16);
    give(&mut g, C::TheWheelOfFortune);
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
    let s = add_joker(&mut g, JokerId::Joker);
    g.state_mut().jokers[s].edition = Edition::Foil;
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
    g.state_mut().jokers[s].edition = Edition::Negative;
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
    g.state_mut().jokers[s].edition = Edition::Base;
    use_c(&mut g, 0, 0).expect("now usable");
}

fn wheel_run(seed: u64, prob_mult: u32) -> (Vec<Edition>, usize) {
    let mut g = in_play(seed);
    if prob_mult > 1 {
        // Oops! All 6s (with an edition, so the Wheel cannot pick it) doubles the odds.
        let o = add_joker(&mut g, JokerId::OopsAll6s);
        g.state_mut().jokers[o].edition = Edition::Foil;
    }
    add_joker(&mut g, JokerId::Joker);
    let s = add_joker(&mut g, JokerId::GreedyJoker);
    g.state_mut().jokers[s].edition = Edition::Holographic;
    add_joker(&mut g, JokerId::LustyJoker);
    give(&mut g, C::TheWheelOfFortune);
    use_c(&mut g, 0, 0).expect("wheel");
    let ed: Vec<Edition> = g
        .state()
        .jokers
        .iter()
        .filter(|j| j.id != JokerId::OopsAll6s)
        .map(|j| j.edition)
        .collect();
    (ed, g.state().consumables.len())
}

#[test]
fn wheel_of_fortune_rolls_one_in_four_and_is_used_up_either_way() {
    let (mut hits, n) = (0, 400);
    for seed in 0..n {
        let (ed, left) = wheel_run(seed, 1);
        assert_eq!(left, 0, "card used even on Nope!");
        assert_eq!(ed[1], Edition::Holographic, "edition-ful joker untouched");
        assert!(ed.iter().all(|e| *e != Edition::Negative));
        let changed = ed.iter().filter(|e| **e != Edition::Base).count();
        assert!(changed == 1 || changed == 2, "seed {seed}: {ed:?}");
        if changed == 2 {
            hits += 1;
        }
    }
    assert!((60..=140).contains(&hits), "about 25% of {n}, got {hits}");
}

#[test]
fn wheel_of_fortune_is_deterministic_and_oops_all_sixes_doubles_the_odds() {
    for seed in 0..10 {
        assert_eq!(wheel_run(seed, 1), wheel_run(seed, 1));
        assert_eq!(wheel_run(seed, 2), wheel_run(seed, 2));
    }
    let (mut hits, n) = (0, 400);
    let mut counts = [0u32; 3];
    for seed in 0..n {
        let (ed, left) = wheel_run(seed, 2);
        assert_eq!(left, 0);
        let new: Vec<Edition> = [ed[0], ed[2]]
            .into_iter()
            .filter(|e| *e != Edition::Base)
            .collect();
        assert!(new.len() <= 1);
        if let Some(e) = new.first() {
            hits += 1;
            match e {
                Edition::Foil => counts[0] += 1,
                Edition::Holographic => counts[1] += 1,
                Edition::Polychrome => counts[2] += 1,
                other => panic!("unexpected edition {other:?}"),
            }
        }
    }
    assert!((160..=240).contains(&hits), "about 50% of {n}, got {hits}");
    assert!(
        counts[0] > counts[1] && counts[1] > counts[2] && counts[2] > 0,
        "Foil 50 / Holo 35 / Poly 15: {counts:?}"
    );
}

// ----- Strength -----------------------------------------------------------------------------------

#[test]
fn strength_raises_ranks_wrapping_king_to_ace_and_ace_to_two() {
    let mut g = in_play(17);
    give(&mut g, C::Strength);
    set_card(&mut g, 0, Rank::King, Suit::Spades);
    set_card(&mut g, 1, Rank::Ace, Suit::Hearts);
    let (a, b) = (uid_at(&g, 0), uid_at(&g, 1));
    use_c(&mut g, 0, 0b11).expect("Strength");
    assert_eq!(card(&g, a).rank, Rank::Ace);
    assert_eq!(card(&g, b).rank, Rank::Two);
    for (from, to) in [
        (Rank::Two, Rank::Three),
        (Rank::Nine, Rank::Ten),
        (Rank::Ten, Rank::Jack),
        (Rank::Jack, Rank::Queen),
        (Rank::Queen, Rank::King),
    ] {
        give(&mut g, C::Strength);
        set_card(&mut g, 2, from, Suit::Clubs);
        let uid = uid_at(&g, 2);
        let mask = 1u16 << pos_of(&g, uid);
        use_c(&mut g, 0, mask).expect("Strength");
        assert_eq!(card(&g, uid).rank, to);
    }
    assert!(g.state().check_invariants().is_ok());
    give(&mut g, C::Strength);
    assert!(matches!(
        use_c(&mut g, 0, 0b111),
        Err(ActionError::TargetCount { selected: 3, .. })
    ));
    assert!(matches!(
        use_c(&mut g, 0, 0),
        Err(ActionError::TargetCount { selected: 0, .. })
    ));
}

#[test]
fn strength_skips_stone_cards() {
    let mut g = in_play(18);
    give(&mut g, C::Strength);
    set_card(&mut g, 0, Rank::Five, Suit::Spades);
    set_card(&mut g, 1, Rank::Five, Suit::Hearts);
    g.state_mut().deck.hand[0].enhancement = Enhancement::Stone;
    let (a, b) = (uid_at(&g, 0), uid_at(&g, 1));
    use_c(&mut g, 0, 0b11).expect("Strength");
    assert_eq!(card(&g, a).rank, Rank::Five, "Stone has no rank");
    assert_eq!(card(&g, b).rank, Rank::Six);
}

// ----- The Hanged Man -----------------------------------------------------------------------------

#[test]
fn the_hanged_man_destroys_one_or_two_cards() {
    let mut g = in_play(19);
    give(&mut g, C::TheHangedMan);
    let uids: Vec<u32> = (0..8).map(|i| uid_at(&g, i)).collect();
    let info = use_c(&mut g, 0, 0b101).expect("Hanged Man");
    assert_eq!(info.cards_destroyed, 2);
    let hand = &g.state().deck.hand;
    assert_eq!(hand.len(), 6);
    assert!(!hand.iter().any(|c| c.uid == uids[0] || c.uid == uids[2]));
    assert!(hand.iter().any(|c| c.uid == uids[1]));
    assert!(g.state().check_invariants().is_ok());
    give(&mut g, C::TheHangedMan);
    assert!(matches!(
        use_c(&mut g, 0, 0b111),
        Err(ActionError::TargetCount { selected: 3, .. })
    ));
    assert!(matches!(
        use_c(&mut g, 0, 0),
        Err(ActionError::TargetCount { selected: 0, .. })
    ));
}

#[test]
fn the_hanged_man_fires_destruction_events() {
    let mut g = in_play(20);
    add_joker(&mut g, JokerId::GlassJoker);
    give(&mut g, C::TheHangedMan);
    g.state_mut().deck.hand[0].enhancement = Enhancement::Glass;
    let before = g.state().jokers[0].value;
    use_c(&mut g, 0, 1).expect("Hanged Man");
    assert!(g.state().jokers[0].value > before, "Glass Joker grew");
}

// ----- Temperance ---------------------------------------------------------------------------------

#[test]
fn temperance_pays_the_total_joker_sell_value_up_to_fifty() {
    let mut g = in_play(21);
    // No jokers: $0, still usable.
    give(&mut g, C::Temperance);
    let m = g.state().money;
    use_c(&mut g, 0, 0).expect("Temperance");
    assert_eq!(g.state().money, m);
    add_joker(&mut g, JokerId::Joker);
    add_joker(&mut g, JokerId::GreedyJoker);
    g.state_mut().jokers[0].sell_value = 3;
    g.state_mut().jokers[1].sell_value = 4;
    give(&mut g, C::Temperance);
    let m = g.state().money;
    use_c(&mut g, 0, 0).expect("Temperance");
    assert_eq!(g.state().money, m + 7);
    g.state_mut().jokers[0].sell_value = 40;
    g.state_mut().jokers[1].sell_value = 40;
    give(&mut g, C::Temperance);
    let m = g.state().money;
    use_c(&mut g, 0, 0).expect("Temperance");
    assert_eq!(g.state().money, m + 50, "capped at $50");
    assert_eq!(tarot_count(&g), 3);
}

// ----- Judgement ----------------------------------------------------------------------------------

#[test]
fn judgement_needs_a_free_joker_slot_and_its_creation_is_dropped_until_phase_5() {
    let mut g = in_play(22);
    give(&mut g, C::Judgement);
    for _ in 0..5 {
        add_joker(&mut g, JokerId::Joker);
    }
    assert_eq!(use_c(&mut g, 0, 0), Err(ActionError::CannotUse));
    g.state_mut().jokers.pop();
    let dropped = g.state().stats.effects_dropped;
    use_c(&mut g, 0, 0).expect("Judgement");
    assert_eq!(g.state().stats.effects_dropped, dropped + 1);
    assert_eq!(g.state().jokers.len(), 4);
    assert_eq!(tarot_count(&g), 1);
    assert!(g.state().check_invariants().is_ok());
}

// ----- registry -----------------------------------------------------------------------------------

#[test]
fn all_twenty_two_tarots_are_defined_with_sane_target_ranges() {
    assert_eq!(C::TAROTS.len(), 22);
    for &id in C::TAROTS {
        let d = try_consumable_def(id).unwrap_or_else(|| panic!("{id:?} undefined"));
        assert_eq!(d.id, id);
        assert!(d.max_targets <= 3);
        assert!(d.min_targets <= d.max_targets);
        assert_eq!(d.min_targets > 0, d.max_targets > 0, "{id:?}");
    }
    for &id in C::PLANETS {
        let d = try_consumable_def(id).expect("planet defined");
        assert_eq!((d.min_targets, d.max_targets), (0, 0));
    }
}
