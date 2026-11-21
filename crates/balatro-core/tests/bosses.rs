//! Phase 4C: the 28 boss blinds at `Game::step` level. Each boss: its effect, the case where it
//! does not trigger / is not in effect, disabling (Chicot owned before the blind and added
//! mid-round), Matador (`triggered`), seed determinism for the random ones, plus selection over
//! the complete boss pool. Preview/step mirroring and random-game smoke tests for every boss
//! live in `boss_contract.rs`.
//! Rules: docs/mechanics/bosses.md; contract: docs/design/consumables_bosses.md.

use balatro_core::actions::{Action, ActionError};
use balatro_core::blinds::{self, BlindKind, MULT_BOSS_DEFAULT, StakeScaling};
use balatro_core::boss::{BossCtx, choose_boss};
use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{BossId as B, boss_def_count, try_boss_def};
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rank, Seal, Suit};
use balatro_core::game::{Game, HandReport};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::state::GameConfig;

const MATADOR_PAY: i32 = 8;

// ----- helpers ------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn win_round(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let n = g.state().deck.hand.len().min(5);
    let mask = (1u32 << n) - 1; // 5 cards: valid under The Psychic
    let info = g.step(Action::Play(mask)).expect("winning play");
    assert!(info.round.expect("round ended").won);
    if g.state().phase == Phase::Shop {
        g.step(Action::EndShop).expect("end shop");
    }
}

/// Win the round with a (valid) Pair play; for bosses that reject repeated hand types.
fn win_with_pair(g: &mut Game) {
    set_plain_hand(g, true);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let info = g.step(Action::Play(0b11)).expect("winning pair");
    assert!(info.round.expect("round ended").won);
    if g.state().phase == Phase::Shop {
        g.step(Action::EndShop).expect("end shop");
    }
}

/// Beat Small and Big, add `jokers`, force `boss` and select it.
fn at_boss_with(seed: u64, boss: B, jokers: &[JokerId]) -> Game {
    let mut g = new_game(seed);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    assert_eq!(g.state().blind, BlindKind::Boss);
    for &j in jokers {
        add(&mut g, j);
    }
    g.state_mut().boss = Some(boss);
    g.step(Action::SelectBlind).expect("select boss");
    g
}

fn at_boss(seed: u64, boss: B) -> Game {
    let g = at_boss_with(seed, boss, &[]);
    assert!(g.state().boss_in_effect());
    g
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("room")
}

fn money(g: &Game) -> i32 {
    g.state().money
}

fn ok(g: &Game) {
    g.state().check_invariants().expect("invariants");
}

const PLAIN: [(Rank, Suit); 8] = [
    (Rank::Two, Suit::Spades),
    (Rank::Three, Suit::Hearts),
    (Rank::Four, Suit::Clubs),
    (Rank::Five, Suit::Diamonds),
    (Rank::Six, Suit::Spades),
    (Rank::Seven, Suit::Hearts),
    (Rank::Eight, Suit::Clubs),
    (Rank::Nine, Suit::Diamonds),
];

fn plain(c: &mut Card, rank: Rank, suit: Suit) {
    c.rank = rank;
    c.suit = suit;
    c.enhancement = Enhancement::None;
    c.edition = Edition::Base;
    c.seal = Seal::None;
    c.perma_bonus = 0;
}

/// Overwrite the hand with distinct plain ranks (no pair, 5-card hands may form straights, so
/// tests play at most 3 cards). `pair`: cards 0 and 1 are a pair of Twos.
fn set_plain_hand(g: &mut Game, pair: bool) {
    for (c, &(r, s)) in g.state_mut().deck.hand.iter_mut().zip(PLAIN.iter()) {
        plain(c, r, s);
    }
    if pair {
        plain(&mut g.state_mut().deck.hand[1], Rank::Two, Suit::Hearts);
    }
}

fn play(g: &mut Game, mask: u32) -> HandReport {
    g.step(Action::Play(mask))
        .expect("play")
        .hand
        .expect("hand report")
}

/// Disable the boss mid-round with Chicot.
fn chicot_now(g: &mut Game) {
    add(g, JokerId::Chicot);
    assert!(g.state().boss_disabled);
}

fn base_chips() -> f64 {
    blinds::base_chips(1, StakeScaling::White)
}

fn hook(f: Option<fn(&mut BossCtx<'_>)>, g: &mut Game) {
    (f.expect("hook"))(&mut BossCtx::new(g.state_mut()));
}

fn uids(g: &Game) -> Vec<u32> {
    g.state().deck.hand.iter().map(|c| c.uid).collect()
}

// ----- registry ----------------------------------------------------------------------------

#[test]
fn all_28_bosses_are_defined_once_with_the_documented_min_ante() {
    let table: [(B, i32); 28] = [
        (B::TheHook, 1),
        (B::TheOx, 6),
        (B::TheHouse, 2),
        (B::TheWall, 2),
        (B::TheWheel, 2),
        (B::TheArm, 2),
        (B::TheClub, 1),
        (B::TheFish, 2),
        (B::ThePsychic, 1),
        (B::TheGoad, 1),
        (B::TheWater, 2),
        (B::TheWindow, 1),
        (B::TheManacle, 1),
        (B::TheEye, 3),
        (B::TheMouth, 2),
        (B::ThePlant, 4),
        (B::TheSerpent, 5),
        (B::ThePillar, 1),
        (B::TheNeedle, 2),
        (B::TheHead, 1),
        (B::TheTooth, 3),
        (B::TheFlint, 2),
        (B::TheMark, 2),
        (B::AmberAcorn, 8),
        (B::VerdantLeaf, 8),
        (B::VioletVessel, 8),
        (B::CrimsonHeart, 8),
        (B::CeruleanBell, 8),
    ];
    assert_eq!(table.len(), B::COUNT);
    for (id, min) in table {
        assert_eq!(boss_def_count(id), 1, "{id:?}");
        assert_eq!(try_boss_def(id).expect("defined").min_ante, min, "{id:?}");
    }
}

// ----- selection with the complete pool ---------------------------------------------------

#[test]
fn selection_over_the_full_pool_respects_min_ante_and_showdown() {
    let mut rng = GameRng::new(77);
    for ante in 1..=24 {
        for _ in 0..40 {
            let mut seen = [0u32; B::COUNT];
            let b = choose_boss(ante, 8, &mut seen, &mut rng).expect("boss");
            assert!(
                try_boss_def(b).expect("def").min_ante <= ante,
                "{b:?} @{ante}"
            );
            assert_eq!(b.is_showdown(), ante % 8 == 0, "{b:?} @{ante}");
        }
    }
}

#[test]
fn ante_one_pool_is_exactly_the_eight_any_ante_bosses() {
    let mut rng = GameRng::new(5);
    let mut got = std::collections::BTreeSet::new();
    for _ in 0..400 {
        let mut seen = [0u32; B::COUNT];
        got.insert(choose_boss(1, 8, &mut seen, &mut rng).expect("boss"));
    }
    let want: std::collections::BTreeSet<B> = [
        B::TheHook,
        B::TheClub,
        B::ThePsychic,
        B::TheGoad,
        B::TheWindow,
        B::TheManacle,
        B::ThePillar,
        B::TheHead,
    ]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}

#[test]
fn the_five_showdown_bosses_rotate_at_ante_eight() {
    let mut rng = GameRng::new(6);
    let mut seen = [0u32; B::COUNT];
    let mut got: Vec<B> = (0..5)
        .map(|_| choose_boss(8, 8, &mut seen, &mut rng).expect("boss"))
        .collect();
    got.sort();
    got.dedup();
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|b| b.is_showdown()));
}

// ----- suit bosses -----------------------------------------------------------------------

const SUIT_BOSSES: [(B, Suit); 4] = [
    (B::TheClub, Suit::Clubs),
    (B::TheGoad, Suit::Spades),
    (B::TheWindow, Suit::Diamonds),
    (B::TheHead, Suit::Hearts),
];

#[test]
fn suit_bosses_debuff_exactly_their_suit() {
    for (boss, suit) in SUIT_BOSSES {
        let g = at_boss(1, boss);
        let mut n = 0;
        for c in g.state().deck.iter_all() {
            assert_eq!(c.debuffed, c.suit == suit, "{boss:?} {c:?}");
            n += usize::from(c.debuffed);
        }
        assert_eq!(n, 13, "{boss:?}: 13 cards of a suit in a standard deck");
        // Not in effect elsewhere.
        let g = at_boss(1, B::TheWall);
        assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
        ok(&g);
    }
}

#[test]
fn suit_bosses_debuff_wild_cards_and_follow_smeared() {
    for (boss, suit) in SUIT_BOSSES {
        let mut g = at_boss(2, boss);
        let other = if suit == Suit::Spades {
            Suit::Hearts
        } else {
            Suit::Spades
        };
        // A Wild card of another suit is every suit: debuffed.
        let wild_other = if suit == Suit::Hearts {
            Suit::Clubs
        } else {
            Suit::Hearts
        };
        {
            let c = &mut g.state_mut().deck.hand[0];
            plain(c, Rank::Ace, wild_other);
            c.enhancement = Enhancement::Wild;
            let c = &mut g.state_mut().deck.hand[1];
            plain(c, Rank::King, other);
            // Pick a suit that is not `suit` and not smeared with it: the opposite colour.
        }
        add(&mut g, JokerId::Joker); // refreshes debuffs
        assert!(
            g.state().deck.hand[0].debuffed,
            "{boss:?}: Wild is debuffed"
        );
        // Smeared: Hearts == Diamonds, Spades == Clubs.
        add(&mut g, JokerId::SmearedJoker);
        for c in g.state().deck.iter_all() {
            if c.enhancement == Enhancement::Wild {
                continue;
            }
            assert_eq!(
                c.debuffed,
                c.suit.is_red() == suit.is_red(),
                "{boss:?} {c:?}"
            );
        }
        assert!(g.state().deck.iter_all().filter(|c| c.debuffed).count() >= 26);
    }
}

#[test]
fn suit_bosses_are_disabled_by_chicot_and_trigger_matador_only_for_debuffed_plays() {
    for (boss, suit) in SUIT_BOSSES {
        // Matador pays only if a played card is debuffed.
        let mut g = at_boss_with(3, boss, &[JokerId::Matador]);
        let other = if suit.is_red() {
            Suit::Spades
        } else {
            Suit::Hearts
        };
        let other = if other == suit { Suit::Clubs } else { other };
        set_plain_hand(&mut g, false);
        plain(&mut g.state_mut().deck.hand[0], Rank::Two, other);
        plain(&mut g.state_mut().deck.hand[1], Rank::Three, suit);
        add(&mut g, JokerId::Joker); // refresh debuffs after the manual edit
        let m0 = money(&g);
        let h = play(&mut g, 0b01); // safe card
        assert!(h.score > 0.0);
        assert_eq!(money(&g), m0, "{boss:?}: no trigger");
        set_plain_hand(&mut g, false);
        plain(&mut g.state_mut().deck.hand[0], Rank::Three, suit);
        add_refresh(&mut g);
        let m1 = money(&g);
        let h = play(&mut g, 0b01); // debuffed card: scores nothing but base, Matador pays
        assert_eq!(h.chips, 5.0, "{boss:?}: High Card base only");
        assert_eq!(money(&g), m1 + MATADOR_PAY, "{boss:?}: trigger");
        ok(&g);

        // Chicot mid-round clears every debuff; Chicot before the blind: never any.
        let mut g = at_boss(3, boss);
        chicot_now(&mut g);
        assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
        let g = at_boss_with(3, boss, &[JokerId::Chicot]);
        assert!(g.state().boss_disabled);
        assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
    }
}

/// Re-run the debuff recomputation (any Play-phase action does it); selling nothing, so a
/// cheap trick: discard-free `refresh` is not public, `add_joker` is.
fn add_refresh(g: &mut Game) {
    if g.state().jokers.len() < 5 {
        add(g, JokerId::Joker);
    }
}

// ----- The Ox ----------------------------------------------------------------------------

fn ox_ante_hook() -> Option<fn(&mut BossCtx<'_>)> {
    try_boss_def(B::TheOx).expect("def").hooks.on_ante_start
}

#[test]
fn the_ox_fixes_the_most_played_hand_and_ties_go_to_the_highest_tier() {
    let mut g = at_boss(1, B::TheOx);
    let mut plays = [0u32; HandType::COUNT];
    plays[HandType::Pair.index()] = 5;
    plays[HandType::Flush.index()] = 3;
    g.state_mut().hand_plays_run = plays;
    hook(ox_ante_hook(), &mut g);
    assert_eq!(g.state().boss_state.hand, Some(HandType::Pair));
    // Tie Pair / Flush / Straight -> highest tier (Flush).
    plays[HandType::Flush.index()] = 5;
    plays[HandType::Straight.index()] = 5;
    g.state_mut().hand_plays_run = plays;
    hook(ox_ante_hook(), &mut g);
    assert_eq!(g.state().boss_state.hand, Some(HandType::Flush));
    // Nothing played yet: fallback High Card.
    g.state_mut().hand_plays_run = [0; HandType::COUNT];
    hook(ox_ante_hook(), &mut g);
    assert_eq!(g.state().boss_state.hand, Some(HandType::HighCard));
}

#[test]
fn the_ox_zeroes_money_on_its_hand_only_and_matador_pays_afterwards() {
    let mut g = at_boss_with(2, B::TheOx, &[JokerId::Matador]);
    g.state_mut().boss_state.hand = Some(HandType::Pair);
    g.state_mut().money = 25;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1); // High Card: not the Ox hand
    assert_eq!(money(&g), 25);
    set_plain_hand(&mut g, true);
    let h = play(&mut g, 0b11);
    assert_eq!(h.hand_type, HandType::Pair);
    assert_eq!(money(&g), MATADOR_PAY, "$0, then Matador pays");
    ok(&g);
    // $0 stays $0; the boss still triggers.
    g.state_mut().money = 0;
    set_plain_hand(&mut g, true);
    play(&mut g, 0b11);
    assert_eq!(money(&g), MATADOR_PAY);
}

#[test]
fn the_ox_raises_negative_money_to_zero_and_is_disabled_by_chicot() {
    let mut g = at_boss(3, B::TheOx);
    g.state_mut().boss_state.hand = Some(HandType::HighCard);
    g.state_mut().money = -3;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(money(&g), 0);
    chicot_now(&mut g);
    g.state_mut().money = 17;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(money(&g), 17, "disabled: money untouched");
}

#[test]
fn the_ox_ignores_hands_that_only_contain_its_hand() {
    // Two Pair contains a Pair but is not "Pair".
    let mut g = at_boss(4, B::TheOx);
    g.state_mut().boss_state.hand = Some(HandType::Pair);
    g.state_mut().money = 9;
    set_plain_hand(&mut g, true);
    plain(&mut g.state_mut().deck.hand[2], Rank::Three, Suit::Spades);
    plain(&mut g.state_mut().deck.hand[3], Rank::Three, Suit::Clubs);
    let h = play(&mut g, 0b1111);
    assert_eq!(h.hand_type, HandType::TwoPair);
    assert_eq!(money(&g), 9);
}

// ----- The House / Wheel / Fish ------------------------------------------------------------

#[test]
fn the_house_draws_only_the_opening_hand_face_down() {
    let mut g = at_boss(1, B::TheHouse);
    assert_eq!(g.state().deck.hand.len(), 8);
    assert!(g.state().deck.hand.iter().all(|c| c.face_down));
    let before = uids(&g);
    g.step(Action::Discard(0b11)).expect("discard");
    for c in &g.state().deck.hand {
        assert_eq!(c.face_down, before.contains(&c.uid), "refills are face up");
    }
    // Face-down cards can be played and score normally.
    let h = play(&mut g, 0b1);
    assert!(h.score > 0.0);
    ok(&g);
    // Not the boss: nothing face down. Disabled: turned face up.
    let g = at_boss(1, B::TheWall);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
    let mut g = at_boss(1, B::TheHouse);
    chicot_now(&mut g);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
    let g = at_boss_with(1, B::TheHouse, &[JokerId::Chicot]);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
}

fn face_down_count(seed_range: std::ops::Range<u64>, jokers: &[JokerId]) -> (usize, usize) {
    let (mut down, mut total) = (0, 0);
    for seed in seed_range {
        let g = at_boss_with(seed, B::TheWheel, jokers);
        down += g.state().deck.hand.iter().filter(|c| c.face_down).count();
        total += g.state().deck.hand.len();
    }
    (down, total)
}

#[test]
fn the_wheel_flips_about_one_in_seven_and_oops_all_6s_doubles_it() {
    let (down, total) = face_down_count(0..250, &[]);
    let p = down as f64 / total as f64;
    assert!((p - 1.0 / 7.0).abs() < 0.03, "{down}/{total} = {p}");
    let (down, total) = face_down_count(0..250, &[JokerId::OopsAll6s]);
    let p = down as f64 / total as f64;
    assert!((p - 2.0 / 7.0).abs() < 0.04, "Oops!: {down}/{total} = {p}");
}

#[test]
fn the_wheel_is_deterministic_per_seed_and_rolls_on_refills_too() {
    let pattern = |seed: u64| {
        let mut g = at_boss(seed, B::TheWheel);
        let mut v: Vec<bool> = g.state().deck.hand.iter().map(|c| c.face_down).collect();
        for _ in 0..3 {
            g.step(Action::Discard(0b1111)).ok();
            v.extend(g.state().deck.hand.iter().map(|c| c.face_down));
        }
        v
    };
    assert_eq!(pattern(5), pattern(5));
    assert!(
        (0..20u64).any(|s| pattern(s) != pattern(s + 1)),
        "seed matters"
    );
    // Over many refills some cards beyond the opening hand are face down.
    let mut flipped_refill = 0;
    for seed in 0..20 {
        let mut g = at_boss(seed, B::TheWheel);
        for _ in 0..3 {
            let before = uids(&g);
            g.step(Action::Discard(0b1_1111)).expect("discard");
            flipped_refill += g
                .state()
                .deck
                .hand
                .iter()
                .filter(|c| c.face_down && !before.contains(&c.uid))
                .count();
        }
        ok(&g);
    }
    assert!(flipped_refill > 0, "refills roll too");
}

#[test]
fn the_wheel_disabled_by_chicot_turns_cards_up() {
    for seed in 0..10 {
        let mut g = at_boss(seed, B::TheWheel);
        chicot_now(&mut g);
        assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
    }
}

#[test]
fn the_fish_flips_cards_drawn_after_a_play_but_not_after_a_discard() {
    let mut g = at_boss(1, B::TheFish);
    assert!(
        g.state().deck.hand.iter().all(|c| !c.face_down),
        "opening hand up"
    );
    g.step(Action::Discard(0b111)).expect("discard");
    assert!(
        g.state().deck.hand.iter().all(|c| !c.face_down),
        "discard refill up"
    );
    let before2 = uids(&g);
    play(&mut g, 0b11);
    for c in &g.state().deck.hand {
        assert_eq!(c.face_down, !before2.contains(&c.uid), "{c:?}");
    }
    assert_eq!(
        g.state().deck.hand.iter().filter(|c| c.face_down).count(),
        2
    );
    ok(&g);
    // Disabled: turned up, and later refills are normal.
    chicot_now(&mut g);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
    play(&mut g, 0b1);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
    // Chicot owned before the blind.
    let mut g = at_boss_with(1, B::TheFish, &[JokerId::Chicot]);
    play(&mut g, 0b1);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
}

// ----- The Arm ------------------------------------------------------------------------------

#[test]
fn the_arm_lowers_the_played_hand_level_and_scores_at_the_lowered_level() {
    let mut g = at_boss(1, B::TheArm);
    g.state_mut().hand_levels[HandType::Pair.index()] = 3;
    set_plain_hand(&mut g, true);
    let h = play(&mut g, 0b11);
    assert_eq!(g.state().hand_levels[HandType::Pair.index()], 2);
    assert_eq!(h.level, 2);
    // Same hand without the Arm at level 2 scores identically.
    let mut c = at_boss(1, B::TheWall);
    c.state_mut().hand_levels[HandType::Pair.index()] = 2;
    set_plain_hand(&mut c, true);
    let hc = play(&mut c, 0b11);
    assert_eq!((h.chips, h.mult, h.score), (hc.chips, hc.mult, hc.score));
    // Other hands keep their level.
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 1);
    ok(&g);
}

#[test]
fn the_arm_never_goes_below_level_one_and_only_triggers_on_a_real_drop() {
    let mut g = at_boss_with(2, B::TheArm, &[JokerId::Matador]);
    set_plain_hand(&mut g, false);
    let m0 = money(&g);
    let h = play(&mut g, 0b1);
    assert_eq!(h.level, 1);
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 1);
    assert_eq!(money(&g), m0, "level 1: nothing changed, no trigger");
    g.state_mut().hand_levels[HandType::HighCard.index()] = 2;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 1);
    assert_eq!(money(&g), m0 + MATADOR_PAY);
}

#[test]
fn the_arm_does_nothing_when_disabled() {
    let mut g = at_boss(3, B::TheArm);
    g.state_mut().hand_levels[HandType::HighCard.index()] = 4;
    chicot_now(&mut g);
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 4);
    let mut g = at_boss_with(3, B::TheArm, &[JokerId::Chicot]);
    g.state_mut().hand_levels[HandType::HighCard.index()] = 4;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 4);
}

// ----- The Water ----------------------------------------------------------------------------

#[test]
fn the_water_removes_all_discards_and_chicot_gives_them_back() {
    let mut g = at_boss(1, B::TheWater);
    let n = g.state().round_discards();
    assert!(n > 0);
    assert_eq!(g.state().discards_left, 0);
    assert_eq!(g.state().boss_state.removed, n);
    assert_eq!(
        g.validate(Action::Discard(0b1)),
        Err(ActionError::NoDiscardsLeft)
    );
    assert!(
        g.legal_actions()
            .iter()
            .all(|a| !matches!(a, Action::Discard(_)))
    );
    chicot_now(&mut g);
    assert_eq!(g.state().discards_left, n);
    g.step(Action::Discard(0b1)).expect("discard works again");
    assert_eq!(g.state().discards_left, n - 1);
    ok(&g);
}

#[test]
fn the_water_chicot_owned_before_the_blind_never_takes_discards() {
    let g = at_boss_with(1, B::TheWater, &[JokerId::Chicot]);
    assert!(g.state().boss_disabled);
    assert_eq!(g.state().discards_left, g.state().round_discards());
    // A normal blind afterwards also has its discards (no stale state).
    let mut g = at_boss(1, B::TheWater);
    win_round(&mut g);
    g.step(Action::SelectBlind).expect("next blind");
    assert_eq!(g.state().discards_left, g.state().round_discards());
}

// ----- The Manacle --------------------------------------------------------------------------

#[test]
fn the_manacle_reduces_hand_size_by_one_until_disabled_or_the_round_ends() {
    let mut g = at_boss(1, B::TheManacle);
    assert_eq!(g.state().hand_size(), 7);
    assert_eq!(g.state().deck.hand.len(), 7);
    play(&mut g, 0b1);
    assert_eq!(g.state().deck.hand.len(), 7, "refills to the reduced size");
    chicot_now(&mut g);
    assert_eq!(g.state().hand_size(), 8);
    play(&mut g, 0b1);
    assert_eq!(g.state().deck.hand.len(), 8);
    let g = at_boss_with(1, B::TheManacle, &[JokerId::Chicot]);
    assert_eq!(g.state().deck.hand.len(), 8);
    // Restored for the next round.
    let mut g = at_boss(2, B::TheManacle);
    win_round(&mut g);
    g.step(Action::SelectBlind).expect("next ante");
    assert_eq!(g.state().hand_size(), 8);
    assert_eq!(g.state().deck.hand.len(), 8);
    ok(&g);
}

// ----- The Eye / The Mouth -------------------------------------------------------------------

#[test]
fn the_eye_rejects_a_repeated_hand_type_and_matador_pays() {
    let mut g = at_boss_with(1, B::TheEye, &[JokerId::Matador]);
    let m0 = money(&g);
    set_plain_hand(&mut g, false);
    let h = play(&mut g, 0b1);
    assert!(!h.invalid && h.score > 0.0);
    assert_eq!(money(&g), m0);
    set_plain_hand(&mut g, false);
    let h = play(&mut g, 0b1);
    assert!(h.invalid);
    assert_eq!(h.score, 0.0);
    assert_eq!(money(&g), m0 + MATADOR_PAY);
    assert_eq!(
        g.state().hands_left,
        g.state().round_hands() - 2,
        "hand is spent"
    );
    // A new type is fine; the rejected repeat still counts, a third High Card is rejected.
    set_plain_hand(&mut g, true);
    let h = play(&mut g, 0b11);
    assert!(!h.invalid);
    assert_eq!(h.hand_type, HandType::Pair);
    set_plain_hand(&mut g, false);
    ok(&g);
}

#[test]
fn the_eye_compares_the_hand_type_not_contained_hands() {
    // Pair first, then Two Pair (contains a Pair): allowed.
    let mut g = at_boss(2, B::TheEye);
    set_plain_hand(&mut g, true);
    assert!(!play(&mut g, 0b11).invalid);
    set_plain_hand(&mut g, true);
    plain(&mut g.state_mut().deck.hand[2], Rank::Three, Suit::Spades);
    plain(&mut g.state_mut().deck.hand[3], Rank::Three, Suit::Clubs);
    let h = play(&mut g, 0b1111);
    assert_eq!(h.hand_type, HandType::TwoPair);
    assert!(!h.invalid);
}

#[test]
fn the_eye_is_per_round_and_off_when_disabled_or_not_the_boss() {
    // Per round: the next round starts with a clean slate.
    let mut g = at_boss(3, B::TheEye);
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    win_with_pair(&mut g);
    g.step(Action::SelectBlind).expect("next");
    assert!(g.state().hand_plays_round.iter().all(|&n| n == 0));
    // Disabled mid-round: repeats are fine.
    let mut g = at_boss(3, B::TheEye);
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    chicot_now(&mut g);
    set_plain_hand(&mut g, false);
    assert!(!play(&mut g, 0b1).invalid);
    // Not the boss.
    let mut g = at_boss(3, B::TheWall);
    for _ in 0..2 {
        set_plain_hand(&mut g, false);
        assert!(!play(&mut g, 0b1).invalid);
    }
}

#[test]
fn the_mouth_allows_only_the_first_hand_type_of_the_round() {
    let mut g = at_boss_with(1, B::TheMouth, &[JokerId::Matador]);
    let m0 = money(&g);
    set_plain_hand(&mut g, false);
    assert!(!play(&mut g, 0b1).invalid); // High Card sets the type
    set_plain_hand(&mut g, true);
    let h = play(&mut g, 0b11); // Pair: rejected
    assert!(h.invalid);
    assert_eq!(h.score, 0.0);
    assert_eq!(money(&g), m0 + MATADOR_PAY);
    // The rejected Pair does not become the allowed type: High Card is still fine.
    set_plain_hand(&mut g, false);
    let h = play(&mut g, 0b1);
    assert!(!h.invalid && h.score > 0.0);
    assert_eq!(money(&g), m0 + MATADOR_PAY, "valid hand: no trigger");
    ok(&g);
}

#[test]
fn the_mouth_resets_each_round_and_is_off_when_disabled() {
    let mut g = at_boss(2, B::TheMouth);
    set_plain_hand(&mut g, true);
    assert!(!play(&mut g, 0b11).invalid); // Pair is the type
    chicot_now(&mut g);
    set_plain_hand(&mut g, false);
    assert!(!play(&mut g, 0b1).invalid, "disabled: any type");
    // Next round (after the boss) has no stale type.
    let mut g = at_boss(2, B::TheMouth);
    win_with_pair(&mut g);
    assert_eq!(g.state().ante, 2);
    // A fresh Mouth round in the next ante starts with no allowed type.
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    g.state_mut().boss = Some(B::TheMouth);
    g.step(Action::SelectBlind).expect("boss");
    set_plain_hand(&mut g, false);
    assert!(!play(&mut g, 0b1).invalid);
}

// ----- The Serpent ---------------------------------------------------------------------------

#[test]
fn the_serpent_draws_exactly_three_after_a_play_or_discard() {
    let mut g = at_boss(1, B::TheSerpent);
    assert_eq!(g.state().deck.hand.len(), 8, "opening draw is normal");
    play(&mut g, 0b1);
    assert_eq!(g.state().deck.hand.len(), 7 + 3);
    g.step(Action::Discard(0b11)).expect("discard");
    assert_eq!(g.state().deck.hand.len(), 8 + 3);
    play(&mut g, 0b111);
    assert_eq!(g.state().deck.hand.len(), 8 + 3);
    ok(&g);
}

#[test]
fn the_serpent_is_capped_by_the_draw_pile_and_off_when_disabled() {
    let mut g = at_boss(2, B::TheSerpent);
    {
        let d = &mut g.state_mut().deck;
        let keep = d.draw_pile.len() - 2;
        let moved: Vec<Card> = d.draw_pile.drain(..keep).collect();
        d.discard_pile.extend(moved);
    }
    play(&mut g, 0b1);
    assert_eq!(g.state().deck.hand.len(), 9, "only 2 cards left to draw");
    ok(&g);
    let mut g = at_boss(2, B::TheSerpent);
    chicot_now(&mut g);
    play(&mut g, 0b1);
    assert_eq!(g.state().deck.hand.len(), 8);
    let mut g = at_boss_with(2, B::TheSerpent, &[JokerId::Chicot]);
    g.step(Action::Discard(0b11)).expect("discard");
    assert_eq!(g.state().deck.hand.len(), 8);
}

// ----- The Pillar ----------------------------------------------------------------------------

#[test]
fn the_pillar_debuffs_cards_played_earlier_this_ante() {
    let mut g = at_boss_with(1, B::ThePillar, &[JokerId::Matador]);
    let played_before = g.state().deck.iter_all().filter(|c| c.played_ante).count();
    assert!(played_before >= 5, "Small and Big played cards");
    for c in g.state().deck.iter_all() {
        assert_eq!(c.debuffed, c.played_ante, "{c:?}");
    }
    // A fresh card scores and does not trigger; an earlier-played one does.
    let m0 = money(&g);
    if let Some(i) = g.state().deck.hand.iter().position(|c| !c.played_ante) {
        let h = play(&mut g, 1 << i);
        assert!(h.chips > 5.0 || h.score > 0.0);
        assert_eq!(money(&g), m0);
        // The card played now is debuffed from then on.
        let discarded = g.state().deck.discard_pile.iter().filter(|c| c.played_ante);
        assert!(discarded.clone().count() >= 1);
        assert!(discarded.into_iter().all(|c| c.debuffed));
    }
    if let Some(i) = g.state().deck.hand.iter().position(|c| c.debuffed) {
        let m1 = money(&g);
        let h = play(&mut g, 1 << i);
        assert_eq!(h.chips, 5.0);
        assert_eq!(money(&g), m1 + MATADOR_PAY);
    }
    ok(&g);
}

#[test]
fn the_pillar_flag_is_ignored_outside_the_boss_and_when_disabled() {
    let g = at_boss(1, B::TheWall);
    assert!(g.state().deck.iter_all().any(|c| c.played_ante));
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
    let mut g = at_boss(1, B::ThePillar);
    assert!(g.state().deck.iter_all().any(|c| c.debuffed));
    chicot_now(&mut g);
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
    let g = at_boss_with(1, B::ThePillar, &[JokerId::Chicot]);
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
}

// ----- The Tooth -----------------------------------------------------------------------------

#[test]
fn the_tooth_takes_a_dollar_per_played_card_before_scoring() {
    let mut g = at_boss_with(1, B::TheTooth, &[JokerId::Matador]);
    g.state_mut().money = 20;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b111);
    assert_eq!(money(&g), 20 - 3 + MATADOR_PAY);
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(money(&g), 20 - 3 + MATADOR_PAY - 1 + MATADOR_PAY);
    ok(&g);
}

#[test]
fn the_tooth_can_push_money_negative_and_is_off_elsewhere() {
    let mut g = at_boss(2, B::TheTooth);
    g.state_mut().money = 1;
    set_plain_hand(&mut g, false);
    play(&mut g, 0b111);
    assert_eq!(money(&g), -2);
    ok(&g);
    let mut g = at_boss(2, B::TheTooth);
    chicot_now(&mut g);
    let m = money(&g);
    play(&mut g, 0b111);
    assert_eq!(money(&g), m);
    let mut g = at_boss(2, B::TheWall);
    let m = money(&g);
    play(&mut g, 0b111);
    assert_eq!(money(&g), m);
}

// ----- Showdown bosses -----------------------------------------------------------------------

#[test]
fn violet_vessel_is_x6_and_x2_when_disabled() {
    let mut g = at_boss(1, B::VioletVessel);
    assert_eq!(g.state().target, base_chips() * 6.0);
    chicot_now(&mut g);
    assert_eq!(g.state().target, base_chips() * MULT_BOSS_DEFAULT);
    let g = at_boss_with(1, B::VioletVessel, &[JokerId::Chicot]);
    assert_eq!(g.state().target, base_chips() * MULT_BOSS_DEFAULT);
}

#[test]
fn amber_acorn_shuffles_and_hides_the_jokers_until_disabled() {
    let js = [
        JokerId::Joker,
        JokerId::GreedyJoker,
        JokerId::LustyJoker,
        JokerId::WrathfulJoker,
        JokerId::GluttonousJoker,
    ];
    let mut reordered = 0;
    for seed in 0..20u64 {
        let mut g = new_game(seed);
        for _ in 0..2 {
            g.step(Action::SelectBlind).expect("select");
            win_round(&mut g);
        }
        for &j in &js[..4] {
            add(&mut g, j);
        }
        let before: Vec<u32> = g.state().jokers.iter().map(|j| j.uid).collect();
        g.state_mut().boss = Some(B::AmberAcorn);
        g.step(Action::SelectBlind).expect("boss");
        let after: Vec<u32> = g.state().jokers.iter().map(|j| j.uid).collect();
        let (mut a, mut b) = (before.clone(), after.clone());
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "same jokers");
        reordered += usize::from(before != after);
        assert!(g.state().boss_state.jokers_hidden);
        ok(&g);
        if seed == 0 {
            chicot_now(&mut g);
            assert!(!g.state().boss_state.jokers_hidden);
        }
        // Round end reveals them.
        let mut g = at_boss_with(seed, B::AmberAcorn, &js);
        assert!(g.state().boss_state.jokers_hidden);
        win_round(&mut g);
        assert!(!g.state().boss_state.jokers_hidden);
    }
    assert!(reordered > 10, "shuffled in {reordered}/20 seeds");
}

#[test]
fn amber_acorn_is_deterministic_and_does_nothing_without_it() {
    let order = |seed: u64, boss: B| -> Vec<u32> {
        let g = at_boss_with(
            seed,
            boss,
            &[
                JokerId::Joker,
                JokerId::GreedyJoker,
                JokerId::LustyJoker,
                JokerId::JollyJoker,
            ],
        );
        g.state().jokers.iter().map(|j| j.uid).collect()
    };
    assert_eq!(order(4, B::AmberAcorn), order(4, B::AmberAcorn));
    let plain_order = order(4, B::TheWall);
    assert_eq!(plain_order, {
        let mut v = plain_order.clone();
        v.sort_unstable();
        v
    });
    let g = at_boss_with(4, B::TheWall, &[JokerId::Joker]);
    assert!(!g.state().boss_state.jokers_hidden);
}

fn debuffed_jokers(g: &Game) -> Vec<u32> {
    g.state()
        .jokers
        .iter()
        .filter(|j| j.debuffed)
        .map(|j| j.uid)
        .collect()
}

#[test]
fn crimson_heart_debuffs_one_joker_per_hand_never_the_same_twice() {
    let js = [JokerId::Joker, JokerId::GreedyJoker, JokerId::LustyJoker];
    for seed in 0..15u64 {
        let mut g = at_boss_with(seed, B::CrimsonHeart, &js);
        let mut prev = {
            let d = debuffed_jokers(&g);
            assert_eq!(d.len(), 1, "one debuffed at blind start");
            assert_eq!(g.state().boss_state.joker_uid, Some(d[0]));
            d[0]
        };
        for _ in 0..3 {
            if g.state().phase != Phase::Play || g.state().hands_left <= 1 {
                break;
            }
            set_plain_hand(&mut g, false);
            play(&mut g, 0b1);
            let d = debuffed_jokers(&g);
            assert_eq!(d.len(), 1);
            assert_ne!(d[0], prev, "seed {seed}: never the same joker twice");
            prev = d[0];
            ok(&g);
        }
        // Round end restores everything.
        win_round(&mut g);
        assert!(debuffed_jokers(&g).is_empty());
    }
}

#[test]
fn crimson_heart_with_a_single_joker_repeats_it_and_the_debuff_stops_its_effect() {
    let score = |boss: B, jokers: &[JokerId]| {
        let mut g = at_boss_with(3, boss, jokers);
        set_plain_hand(&mut g, false);
        play(&mut g, 0b1).score
    };
    let none = score(B::TheWall, &[]);
    let with = score(B::TheWall, &[JokerId::Joker]);
    assert!(with > none);
    assert_eq!(
        score(B::CrimsonHeart, &[JokerId::Joker]),
        none,
        "debuffed joker adds nothing"
    );
    let mut g = at_boss_with(3, B::CrimsonHeart, &[JokerId::Joker]);
    set_plain_hand(&mut g, false);
    play(&mut g, 0b1);
    assert_eq!(
        debuffed_jokers(&g).len(),
        1,
        "the only joker is chosen again"
    );
    // No jokers: nothing happens.
    let g = at_boss(3, B::CrimsonHeart);
    assert!(debuffed_jokers(&g).is_empty());
    assert_eq!(g.state().boss_state.joker_uid, None);
}

#[test]
fn crimson_heart_disable_restores_the_joker_and_is_deterministic() {
    let mut g = at_boss_with(2, B::CrimsonHeart, &[JokerId::Joker, JokerId::GreedyJoker]);
    assert_eq!(debuffed_jokers(&g).len(), 1);
    chicot_now(&mut g);
    assert!(debuffed_jokers(&g).is_empty());
    assert_eq!(g.state().boss_state.joker_uid, None);
    play(&mut g, 0b1);
    assert!(
        debuffed_jokers(&g).is_empty(),
        "no more picks once disabled"
    );
    let picks = |seed: u64| {
        let mut g = at_boss_with(
            seed,
            B::CrimsonHeart,
            &[
                JokerId::Joker,
                JokerId::GreedyJoker,
                JokerId::LustyJoker,
                JokerId::JollyJoker,
            ],
        );
        let mut v = vec![g.state().boss_state.joker_uid];
        for _ in 0..2 {
            set_plain_hand(&mut g, false);
            play(&mut g, 0b1);
            v.push(g.state().boss_state.joker_uid);
        }
        v
    };
    assert_eq!(picks(9), picks(9));
    assert!((0..10u64).any(|s| picks(s) != picks(s + 1)));
}

#[test]
fn cerulean_bell_forces_one_hand_card_and_repicks_after_every_draw() {
    for seed in 0..10u64 {
        let mut g = at_boss(seed, B::CeruleanBell);
        let uid = g.state().boss_state.card_uid.expect("forced card");
        let pos = g
            .state()
            .deck
            .hand
            .iter()
            .position(|c| c.uid == uid)
            .expect("in hand");
        assert_eq!(
            g.validate(Action::Play(if pos == 0 { 0b10 } else { 0b01 })),
            Err(ActionError::ForcedCardMissing { position: pos })
        );
        assert_eq!(
            g.validate(Action::Discard(if pos == 0 { 0b10 } else { 0b01 })),
            Err(ActionError::ForcedCardMissing { position: pos })
        );
        assert!(g.validate(Action::Play(1 << pos)).is_ok());
        assert!(g.legal_actions().iter().all(|a| match *a {
            Action::Play(m) | Action::Discard(m) => (m >> pos) & 1 == 1,
            _ => true,
        }));
        // Discarding the forced card is allowed; a new one is picked from the new hand.
        g.step(Action::Discard(1 << pos)).expect("discard forced");
        let new_uid = g.state().boss_state.card_uid.expect("re-picked");
        assert_ne!(new_uid, uid);
        assert!(g.state().deck.hand.iter().any(|c| c.uid == new_uid));
        let pos = g
            .state()
            .deck
            .hand
            .iter()
            .position(|c| c.uid == new_uid)
            .expect("pos");
        play(&mut g, 1 << pos);
        assert!(
            g.state()
                .deck
                .hand
                .iter()
                .any(|c| Some(c.uid) == g.state().boss_state.card_uid)
        );
        ok(&g);
    }
}

#[test]
fn cerulean_bell_keeps_a_card_that_is_still_in_hand_and_repicks_a_missing_one() {
    let after_draw = try_boss_def(B::CeruleanBell).expect("def").hooks.after_draw;
    let mut g = at_boss(1, B::CeruleanBell);
    let keep = g.state().deck.hand[5].uid;
    g.state_mut().boss_state.card_uid = Some(keep);
    for _ in 0..10 {
        hook(after_draw, &mut g);
        assert_eq!(g.state().boss_state.card_uid, Some(keep));
    }
    g.state_mut().boss_state.card_uid = Some(u32::MAX);
    hook(after_draw, &mut g);
    let uid = g.state().boss_state.card_uid.expect("re-picked");
    assert!(g.state().deck.hand.iter().any(|c| c.uid == uid));
    g.state_mut().boss_state.card_uid = None;
    hook(after_draw, &mut g);
    assert!(g.state().boss_state.card_uid.is_some());
    // Empty hand: no forced card.
    g.state_mut().deck.hand.clear();
    hook(after_draw, &mut g);
    assert_eq!(g.state().boss_state.card_uid, None);
}

#[test]
fn cerulean_bell_is_off_when_disabled_or_not_the_boss_and_deterministic() {
    let mut g = at_boss(1, B::CeruleanBell);
    assert!(g.state().boss_state.card_uid.is_some());
    chicot_now(&mut g);
    assert_eq!(g.state().boss_state.card_uid, None);
    assert!(g.validate(Action::Play(0b1)).is_ok());
    g.step(Action::Discard(0b1)).expect("discard");
    assert_eq!(
        g.state().boss_state.card_uid,
        None,
        "no re-pick once disabled"
    );
    let g = at_boss_with(1, B::CeruleanBell, &[JokerId::Chicot]);
    assert_eq!(g.state().boss_state.card_uid, None);
    let g = at_boss(1, B::TheWall);
    assert_eq!(g.state().boss_state.card_uid, None);
    let forced = |seed: u64| at_boss(seed, B::CeruleanBell).state().boss_state.card_uid;
    assert_eq!(forced(12), forced(12));
    // The forced card is also required by the Play/Discard mask when the round starts, and
    // cleared at round end.
    let mut g = at_boss(2, B::CeruleanBell);
    let pos = g
        .state()
        .deck
        .hand
        .iter()
        .position(|c| Some(c.uid) == g.state().boss_state.card_uid)
        .expect("pos");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let info = g.step(Action::Play(1 << pos)).expect("win");
    assert!(info.round.expect("round").won);
    assert_eq!(g.state().boss_state.card_uid, None);
}

// ----- boss effects are in effect only during the boss round -------------------------------

#[test]
fn boss_state_does_not_leak_into_the_next_round() {
    for boss in [
        B::TheWater,
        B::TheManacle,
        B::TheHouse,
        B::TheFish,
        B::TheWheel,
        B::AmberAcorn,
        B::CrimsonHeart,
        B::CeruleanBell,
    ] {
        let mut g = at_boss_with(6, boss, &[JokerId::Joker, JokerId::GreedyJoker]);
        win_round(&mut g);
        g.step(Action::SelectBlind)
            .expect("small blind of the next ante");
        let s = g.state();
        assert_eq!(s.deck.hand.len(), 8, "{boss:?}");
        assert_eq!(s.discards_left, s.round_discards(), "{boss:?}");
        assert!(
            s.deck.iter_all().all(|c| !c.face_down && !c.debuffed),
            "{boss:?}"
        );
        assert!(s.jokers.iter().all(|j| !j.debuffed), "{boss:?}");
        assert!(!s.boss_state.jokers_hidden, "{boss:?}");
        assert_eq!(s.boss_state.card_uid, None, "{boss:?}");
        ok(&g);
    }
}

#[test]
fn every_new_boss_is_deterministic_per_seed() {
    let run = |seed: u64, boss: B| {
        let mut g = at_boss_with(
            seed,
            boss,
            &[JokerId::Joker, JokerId::GreedyJoker, JokerId::Matador],
        );
        for _ in 0..3 {
            if g.state().phase != Phase::Play {
                break;
            }
            let forced = g
                .state()
                .boss_state
                .card_uid
                .and_then(|u| g.state().deck.hand.iter().position(|c| c.uid == u));
            let mask: u32 = forced.map_or(0b1, |p| 1 << p);
            if g.step(Action::Play(mask)).is_err() {
                break;
            }
        }
        g.into_state()
    };
    for boss in B::ALL {
        assert_eq!(run(21, boss), run(21, boss), "{boss:?}");
    }
}
