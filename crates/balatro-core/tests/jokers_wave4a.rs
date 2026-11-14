//! Phase 3 package wave4a: Marble Joker, 8 Ball, Dusk, Pareidolia, Burglar, DNA, Splash, Sixth
//! Sense, Superposition, Seance, Riff-Raff, Shortcut, Vagabond, Midas Mask.
//!
//! Scorer-level tests check trigger / non-trigger / seeds / Oops! All 6s / Blueprint. Creators
//! are checked at the effect level (the consumable and random-joker systems come in Phases 4/5,
//! so the game layer drops those effects today); card creators and rule changers are also tested
//! through `Game::step`. Every expected number is derived in a comment.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::card::Card;
use balatro_core::content::jokers::{JokerId, def_count, try_def};
use balatro_core::deck::{Deck, Pile};
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Rarity, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::hand_eval::{HandEval, evaluate};
use balatro_core::hooks::{
    ConsumableKind, EventInput, RunView, SideEffect, collect_modifiers, fire_blind_select,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer};
use balatro_core::state::GameConfig;

use JokerId as J;
use Rank::*;
use Suit::*;

// ----- helpers -----

fn c(uid: u32, rank: Rank, suit: Suit) -> Card {
    Card::new(uid, rank, suit)
}

fn enh(mut card: Card, e: Enhancement) -> Card {
    card.enhancement = e;
    card
}

fn seal(mut card: Card, s: Seal) -> Card {
    card.seal = s;
    card
}

fn debuff(mut card: Card) -> Card {
    card.debuffed = true;
    card
}

fn ev(hand_type: HandType, scoring_mask: u8, contains: &[HandType]) -> HandEval {
    let mut bits = 1u16 << hand_type.index();
    for h in contains {
        bits |= 1 << h.index();
    }
    HandEval {
        hand_type,
        scoring_mask,
        contains: bits,
    }
}

fn jk(uid: u32, id: JokerId) -> JokerInstance {
    JokerInstance::new(uid, id)
}

/// Jokers with uids 100, 101, ... in the given order.
fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| jk(100 + i as u32, id))
        .collect()
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
    played: Vec<Card>,
}

fn score_seed(
    played: &[Card],
    held: &[Card],
    eval: HandEval,
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> Out {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::with_trace(true);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    });
    Out {
        res,
        effects: scorer.effects().to_vec(),
        played,
    }
}

fn score_with(played: &[Card], eval: HandEval, js: &mut [JokerInstance], run: &RunView<'_>) -> Out {
    score_seed(played, &[], eval, js, run, 11)
}

fn score(played: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> Out {
    score_with(played, eval, js, &RunView::default())
}

fn run_pm(prob_mult: u32) -> RunView<'static> {
    RunView {
        prob_mult,
        ..RunView::default()
    }
}

/// `RunView` for a hand played with `hands_left` hands remaining.
fn run_hands_left(hands_left: u32) -> RunView<'static> {
    RunView {
        hands_left,
        ..RunView::default()
    }
}

/// First hand of the round (`hands_played_round == 1`).
fn first_hand() -> RunView<'static> {
    RunView {
        hands_played_round: 1,
        ..RunView::default()
    }
}

fn tarot() -> SideEffect {
    SideEffect::CreateConsumable {
        kind: ConsumableKind::Tarot,
        negative: false,
    }
}

fn spectral() -> SideEffect {
    SideEffect::CreateConsumable {
        kind: ConsumableKind::Spectral,
        negative: false,
    }
}

/// Fire `on_blind_select` for `blind` and return the queued effects.
fn blind_select(
    js: &mut [JokerInstance],
    blind: BlindKind,
    seed: u64,
) -> (Vec<SideEffect>, GameRng) {
    let run = RunView {
        blind,
        ..RunView::default()
    };
    let deck = Deck::standard();
    let mut rng = GameRng::new(seed);
    let mut fx = Vec::new();
    fire_blind_select(EventInput {
        jokers: js,
        run: &run,
        deck: &deck,
        rng: &mut rng,
        effects: &mut fx,
    });
    (fx, rng)
}

fn fx_of(js: &mut [JokerInstance]) -> Vec<SideEffect> {
    blind_select(js, BlindKind::Small, 5).0
}

// ----- Game-level helpers -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn select(g: &mut Game) -> StepInfo {
    g.step(Action::SelectBlind).expect("select blind")
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id))
        .expect("room for joker")
}

/// Overwrite hand card `i` in place (the uid stays, so deck invariants hold).
fn set_hand(g: &mut Game, i: usize, rank: Rank, suit: Suit) {
    let c = &mut g.state_mut().deck.hand[i];
    c.rank = rank;
    c.suit = suit;
    c.enhancement = Enhancement::None;
    c.edition = Edition::Base;
    c.seal = Seal::None;
}

fn play(g: &mut Game, mask: u16) -> StepInfo {
    let info = g.step(Action::Play(mask)).expect("play");
    g.state().check_invariants().expect("invariants");
    info
}

fn find_card(g: &Game, uid: u32) -> Option<Card> {
    g.state().deck.iter_all().find(|c| c.uid == uid).copied()
}

// ----- registry / flags -----

const WAVE: [JokerId; 14] = [
    J::MarbleJoker,
    J::EightBall,
    J::Dusk,
    J::Pareidolia,
    J::Burglar,
    J::DNA,
    J::Splash,
    J::SixthSense,
    J::Superposition,
    J::Seance,
    J::RiffRaff,
    J::Shortcut,
    J::Vagabond,
    J::MidasMask,
];

#[test]
fn all_fourteen_are_defined_once_with_csv_stats() {
    // (id, rarity, cost, blueprint-compatible) from docs/mechanics/jokers.csv.
    let table = [
        (J::MarbleJoker, Rarity::Uncommon, 6, true),
        (J::EightBall, Rarity::Common, 5, true),
        (J::Dusk, Rarity::Uncommon, 5, true),
        (J::Pareidolia, Rarity::Uncommon, 5, false),
        (J::Burglar, Rarity::Uncommon, 6, true),
        (J::DNA, Rarity::Rare, 8, true),
        (J::Splash, Rarity::Common, 3, false),
        (J::SixthSense, Rarity::Uncommon, 6, false),
        (J::Superposition, Rarity::Common, 4, true),
        (J::Seance, Rarity::Uncommon, 6, true),
        (J::RiffRaff, Rarity::Common, 6, true),
        (J::Shortcut, Rarity::Uncommon, 7, false),
        (J::Vagabond, Rarity::Rare, 8, true),
        (J::MidasMask, Rarity::Uncommon, 7, false),
    ];
    assert_eq!(table.len(), WAVE.len());
    for (id, rarity, cost, bp) in table {
        assert_eq!(def_count(id), 1, "{id:?}");
        let d = try_def(id).expect("def");
        assert_eq!(
            (d.rarity, d.base_cost, d.blueprint_compatible),
            (rarity, cost, bp)
        );
        assert!(d.eternal_compatible && d.perishable_compatible, "{id:?}");
    }
}

// ----- Marble Joker -----

#[test]
fn marble_adds_a_stone_card_to_the_draw_pile() {
    for blind in [BlindKind::Small, BlindKind::Big, BlindKind::Boss] {
        let (fx, _) = blind_select(&mut jokers(&[J::MarbleJoker]), blind, 3);
        assert_eq!(fx.len(), 1, "{blind:?}");
        let SideEffect::AddCard { card, to } = fx[0] else {
            panic!("expected AddCard, got {:?}", fx[0]);
        };
        assert_eq!(to, Pile::Draw);
        assert_eq!(card.enhancement, Enhancement::Stone);
        assert!(!card.debuffed && card.edition == Edition::Base && card.seal == Seal::None);
    }
}

#[test]
fn marble_is_seed_deterministic_and_varies_its_front() {
    let a = blind_select(&mut jokers(&[J::MarbleJoker]), BlindKind::Small, 9).0;
    let b = blind_select(&mut jokers(&[J::MarbleJoker]), BlindKind::Small, 9).0;
    assert_eq!(a, b);
    let mut fronts = std::collections::BTreeSet::new();
    for seed in 0..40 {
        let fx = blind_select(&mut jokers(&[J::MarbleJoker]), BlindKind::Small, seed).0;
        if let SideEffect::AddCard { card, .. } = fx[0] {
            fronts.insert((card.rank as u8, card.suit.index()));
        }
    }
    assert!(fronts.len() > 5, "front looks constant: {fronts:?}");
}

#[test]
fn marble_debuffed_is_inert_and_blueprint_adds_another() {
    let mut js = vec![jk(1, J::MarbleJoker).with_debuff(true)];
    assert!(fx_of(&mut js).is_empty());
    let fx = fx_of(&mut jokers(&[J::Blueprint, J::MarbleJoker]));
    assert_eq!(fx.len(), 2);
    assert!(fx.iter().all(|e| matches!(e, SideEffect::AddCard { .. })));
    // Two Marbles: two cards.
    assert_eq!(
        fx_of(&mut jokers(&[J::MarbleJoker, J::MarbleJoker])).len(),
        2
    );
}

#[test]
fn marble_game_adds_a_stone_card_and_keeps_the_card_count_invariant() {
    let mut g = new_game(70);
    add(&mut g, J::MarbleJoker);
    let info = select(&mut g);
    let s = g.state();
    assert_eq!(info.cards_added, 1);
    assert_eq!(s.deck.len(), 53);
    assert_eq!(s.stats.cards_added, 1);
    assert_eq!(
        s.deck.iter_all().filter(|c| c.is_stone()).count(),
        1,
        "exactly one Stone card"
    );
    s.check_invariants().expect("invariants");
    // Blueprint copy: two Stone cards per blind (round 1 here, fresh game).
    let mut g = new_game(71);
    add(&mut g, J::Blueprint);
    add(&mut g, J::MarbleJoker);
    select(&mut g);
    assert_eq!(g.state().deck.len(), 54);
    assert_eq!(
        g.state().deck.iter_all().filter(|c| c.is_stone()).count(),
        2
    );
    g.state().check_invariants().expect("invariants");
}

// ----- 8 Ball -----

#[test]
fn eight_ball_creates_a_tarot_per_scoring_eight() {
    let run = run_pm(1000); // always
    // 8♠ 8♥ Pair: 10 + 8 + 8 = 26 x 2 = 52; two scoring 8s -> two Tarots.
    let played = [c(1, Eight, Spades), c(2, Eight, Hearts)];
    let o = score_with(
        &played,
        ev(HandType::Pair, 0b11, &[]),
        &mut jokers(&[J::EightBall]),
        &run,
    );
    assert_eq!(o.res.score, 52.0);
    assert_eq!(o.effects, vec![tarot(), tarot()]);
    // One 8 and one 7 scoring (two pair stand-in): only the 8 rolls.
    let played = [c(1, Eight, Spades), c(2, Seven, Hearts)];
    let o = score_with(
        &played,
        ev(HandType::Pair, 0b11, &[]),
        &mut jokers(&[J::EightBall]),
        &run,
    );
    assert_eq!(o.effects, vec![tarot()]);
}

#[test]
fn eight_ball_ignores_unscored_stone_debuffed_and_other_ranks() {
    let run = run_pm(1000);
    let mut js = jokers(&[J::EightBall]);
    // The kicker 8 does not score.
    let played = [c(1, Seven, Spades), c(2, Seven, Hearts), c(3, Eight, Clubs)];
    let o = score_with(&played, ev(HandType::Pair, 0b011, &[]), &mut js, &run);
    assert!(o.effects.is_empty());
    // Stone card with a printed 8 has no rank.
    let played = [enh(c(1, Eight, Spades), Enhancement::Stone)];
    let o = score_with(&played, ev(HandType::HighCard, 0b1, &[]), &mut js, &run);
    assert!(o.effects.is_empty());
    // Debuffed 8 never triggers.
    let played = [debuff(c(1, Eight, Spades))];
    let o = score_with(&played, ev(HandType::HighCard, 0b1, &[]), &mut js, &run);
    assert!(o.effects.is_empty());
    // Other ranks.
    let played = [c(1, Nine, Spades)];
    let o = score_with(&played, ev(HandType::HighCard, 0b1, &[]), &mut js, &run);
    assert!(o.effects.is_empty());
    // Debuffed joker.
    let mut dead = vec![jk(1, J::EightBall).with_debuff(true)];
    let played = [c(1, Eight, Spades)];
    let o = score_with(&played, ev(HandType::HighCard, 0b1, &[]), &mut dead, &run);
    assert!(o.effects.is_empty());
    // Probability 0 never rolls true.
    let o = score_with(
        &played,
        ev(HandType::HighCard, 0b1, &[]),
        &mut js,
        &run_pm(0),
    );
    assert!(o.effects.is_empty());
}

#[test]
fn eight_ball_rolls_again_on_retriggers_and_blueprint_copies_it() {
    let run = run_pm(1000);
    // Red Seal 8: two triggers -> two rolls -> two Tarots.
    let played = [seal(c(1, Eight, Spades), Seal::Red)];
    let o = score_with(
        &played,
        ev(HandType::HighCard, 0b1, &[]),
        &mut jokers(&[J::EightBall]),
        &run,
    );
    assert_eq!(o.effects.len(), 2);
    // Blueprint + 8 Ball: each scored 8 creates one from each -> two for a single trigger.
    let played = [c(1, Eight, Spades)];
    let o = score_with(
        &played,
        ev(HandType::HighCard, 0b1, &[]),
        &mut jokers(&[J::Blueprint, J::EightBall]),
        &run,
    );
    assert_eq!(o.effects, vec![tarot(), tarot()]);
}

#[test]
fn eight_ball_odds_are_seeded_and_oops_all_6s_raises_them() {
    let played = [c(1, Eight, Spades)];
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let hits = |pm: u32| -> usize {
        (0..400u64)
            .filter(|&seed| {
                let run = run_pm(pm);
                !score_seed(&played, &[], eval, &mut jokers(&[J::EightBall]), &run, seed)
                    .effects
                    .is_empty()
            })
            .count()
    };
    let (base, oops, sure) = (hits(1), hits(2), hits(4));
    // 1/4 of 400 = 100 (+-35); Oops! doubles to 1/2 = 200 (+-40); x4 is certain.
    assert!((65..=135).contains(&base), "base {base}");
    assert!((160..=240).contains(&oops), "oops {oops}");
    assert_eq!(sure, 400);
    // Same seed, same outcome.
    for seed in 0..20 {
        let a = score_seed(
            &played,
            &[],
            eval,
            &mut jokers(&[J::EightBall]),
            &run_pm(1),
            seed,
        );
        let b = score_seed(
            &played,
            &[],
            eval,
            &mut jokers(&[J::EightBall]),
            &run_pm(1),
            seed,
        );
        assert_eq!(a.effects, b.effects);
    }
    // The real Oops! joker raises the multiplier the game feeds into RunView.
    let m = collect_modifiers(&jokers(&[J::OopsAll6s, J::EightBall]));
    assert_eq!(m.prob_mult, 2);
}

// ----- Dusk -----

#[test]
fn dusk_retriggers_scoring_cards_on_the_final_hand_only() {
    // 7♠ 7♥ Pair (+ kicker 2♣ that does not score).
    let played = [c(1, Seven, Spades), c(2, Seven, Hearts), c(3, Two, Clubs)];
    let eval = ev(HandType::Pair, 0b011, &[]);
    // Final hand (hands_left 0): each 7 scores twice: 10 + 4 x 7 = 38 x 2 = 76.
    let o = score_with(&played, eval, &mut jokers(&[J::Dusk]), &run_hands_left(0));
    assert_eq!(o.res.score, 76.0);
    // Not the final hand: 10 + 14 = 24 x 2 = 48.
    let o = score_with(&played, eval, &mut jokers(&[J::Dusk]), &run_hands_left(1));
    assert_eq!(o.res.score, 48.0);
    // Debuffed Dusk: nothing.
    let mut dead = vec![jk(1, J::Dusk).with_debuff(true)];
    let o = score_with(&played, eval, &mut dead, &run_hands_left(0));
    assert_eq!(o.res.score, 48.0);
    // Debuffed cards stay skipped.
    let dp = [debuff(c(1, Seven, Spades)), c(2, Seven, Hearts)];
    let o = score_with(
        &dp,
        ev(HandType::Pair, 0b11, &[]),
        &mut jokers(&[J::Dusk]),
        &run_hands_left(0),
    );
    // 10 + 2 x 7 = 24 x 2 = 48.
    assert_eq!(o.res.score, 48.0);
}

#[test]
fn dusk_stacks_with_red_seal_and_blueprint() {
    let eval = ev(HandType::Pair, 0b11, &[]);
    let run = run_hands_left(0);
    // Red Seal 7 (1 + 1 + 1 = 3 triggers) + plain 7 (2): 10 + 21 + 14 = 45 x 2 = 90.
    let played = [seal(c(1, Seven, Spades), Seal::Red), c(2, Seven, Hearts)];
    let o = score_with(&played, eval, &mut jokers(&[J::Dusk]), &run);
    assert_eq!(o.res.score, 90.0);
    // Blueprint + Dusk: each card triggers 3 times: 10 + 6 x 7 = 52 x 2 = 104.
    let played = [c(1, Seven, Spades), c(2, Seven, Hearts)];
    let o = score_with(&played, eval, &mut jokers(&[J::Blueprint, J::Dusk]), &run);
    assert_eq!(o.res.score, 104.0);
}

// ----- Rule changers: Pareidolia, Splash, Shortcut -----

#[test]
fn rule_jokers_set_only_their_flag_and_are_not_copied() {
    let flags = |ids: &[JokerId]| collect_modifiers(&jokers(ids)).rules;
    let r = flags(&[J::Pareidolia]);
    assert!(r.pareidolia && !r.splash && !r.shortcut && !r.four_fingers && !r.smeared);
    let r = flags(&[J::Splash]);
    assert!(r.splash && !r.pareidolia && !r.shortcut);
    let r = flags(&[J::Shortcut]);
    assert!(r.shortcut && !r.splash && !r.pareidolia);
    let r = flags(&[J::Pareidolia, J::Splash, J::Shortcut, J::FourFingers]);
    assert!(r.pareidolia && r.splash && r.shortcut && r.four_fingers);
    // Blueprint copies none of them; a debuffed one gives nothing.
    for id in [J::Pareidolia, J::Splash, J::Shortcut] {
        assert!(!try_def(id).expect("def").blueprint_compatible);
    }
    let dead = vec![jk(1, J::Splash).with_debuff(true)];
    assert!(!collect_modifiers(&dead).rules.splash);
    let o = score(
        &[c(1, Seven, Spades)],
        ev(HandType::HighCard, 0b1, &[]),
        &mut jokers(&[J::Blueprint, J::Splash]),
    );
    assert_eq!(o.res.score, 12.0); // 5 + 7 = 12 x 1; Blueprint gives nothing
}

#[test]
fn shortcut_makes_gapped_straights_in_hand_eval() {
    let rules = collect_modifiers(&jokers(&[J::Shortcut])).rules;
    let none = collect_modifiers(&[]).rules;
    // 10 8 7 6 5, mixed suits.
    let gap = [
        c(1, Ten, Spades),
        c(2, Eight, Hearts),
        c(3, Seven, Clubs),
        c(4, Six, Diamonds),
        c(5, Five, Spades),
    ];
    let e = evaluate(&gap, &rules);
    assert_eq!(e.hand_type, HandType::Straight);
    assert_eq!(e.scoring_mask, 0b11111);
    assert_eq!(evaluate(&gap, &none).hand_type, HandType::HighCard);
    // A gap of two ranks is still not a straight: 10 7 6 5 4.
    let wide = [
        c(1, Ten, Spades),
        c(2, Seven, Hearts),
        c(3, Six, Clubs),
        c(4, Five, Diamonds),
        c(5, Four, Spades),
    ];
    assert_eq!(evaluate(&wide, &rules).hand_type, HandType::HighCard);
    // No wrap-around: Q K A 2 3.
    let wrap = [
        c(1, Queen, Spades),
        c(2, King, Hearts),
        c(3, Ace, Clubs),
        c(4, Two, Diamonds),
        c(5, Three, Spades),
    ];
    assert_eq!(evaluate(&wrap, &rules).hand_type, HandType::HighCard);
}

#[test]
fn shortcut_game_scores_a_gapped_straight() {
    // Straight (level 1): 30 chips x 4 mult; 10 + 8 + 7 + 6 + 5 = 36 card chips.
    // (30 + 36) x 4 = 264. Without Shortcut it is High Card: (5 + 10) x 1 = 15.
    for (with, expect_type, expect_score) in [
        (true, HandType::Straight, 264.0),
        (false, HandType::HighCard, 15.0),
    ] {
        let mut g = new_game(80);
        if with {
            add(&mut g, J::Shortcut);
            assert!(g.state().mods.rules.shortcut);
        }
        select(&mut g);
        for (i, (r, s)) in [
            (Ten, Spades),
            (Eight, Hearts),
            (Seven, Clubs),
            (Six, Diamonds),
            (Five, Spades),
        ]
        .into_iter()
        .enumerate()
        {
            set_hand(&mut g, i, r, s);
        }
        let h = play(&mut g, 0b11111).hand.expect("hand");
        assert_eq!(h.hand_type, expect_type);
        assert_eq!(h.score, expect_score);
    }
}

#[test]
fn splash_scores_every_played_card_in_game() {
    // 7♠ 7♥ 2♣ 3♦ K♠: Pair (10 x 2) + scoring 7 7 only: (10 + 14) x 2 = 48.
    // With Splash all five score: (10 + 7 + 7 + 2 + 3 + 10) x 2 = 78.
    for (with, expect) in [(true, 78.0), (false, 48.0)] {
        let mut g = new_game(81);
        if with {
            add(&mut g, J::Splash);
        }
        select(&mut g);
        for (i, (r, s)) in [
            (Seven, Spades),
            (Seven, Hearts),
            (Two, Clubs),
            (Three, Diamonds),
            (King, Spades),
        ]
        .into_iter()
        .enumerate()
        {
            set_hand(&mut g, i, r, s);
        }
        let h = play(&mut g, 0b11111).hand.expect("hand");
        assert_eq!(h.hand_type, HandType::Pair);
        assert_eq!(h.score, expect, "splash {with}");
    }
}

#[test]
fn splash_is_seen_by_scoring_hooks_of_other_jokers() {
    // Splash makes the unpaired King score, so Midas Mask (face cards) turns it Gold; without
    // Splash it stays a plain King.
    let rules = collect_modifiers(&jokers(&[J::Splash])).rules;
    let cards = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Two, Clubs),
        c(4, Three, Diamonds),
        c(5, King, Spades),
    ];
    let eval = evaluate(&cards, &rules);
    assert_eq!(eval.scoring_mask, 0b11111);
    let run = RunView {
        rules,
        ..RunView::default()
    };
    let o = score_with(&cards, eval, &mut jokers(&[J::MidasMask]), &run);
    assert_eq!(o.played[4].enhancement, Enhancement::Gold);
}

#[test]
fn pareidolia_makes_every_card_a_face_in_game() {
    // Ride the Bus: +1 mult per hand without a scoring face card, reset by a face card.
    // High Card 7: (5 + 7) x (1 + 1) = 24 normally; with Pareidolia the 7 is a face card, the
    // counter stays 0: (5 + 7) x 1 = 12.
    for (with, expect) in [(false, 24.0), (true, 12.0)] {
        let mut g = new_game(82);
        if with {
            add(&mut g, J::Pareidolia);
            assert!(g.state().mods.rules.pareidolia);
        }
        add(&mut g, J::RideTheBus);
        select(&mut g);
        set_hand(&mut g, 0, Seven, Spades);
        let h = play(&mut g, 0b1).hand.expect("hand");
        assert_eq!(h.score, expect, "pareidolia {with}");
    }
}

// ----- Burglar -----

#[test]
fn burglar_adds_hands_and_clears_discards_on_every_blind() {
    for blind in [BlindKind::Small, BlindKind::Big, BlindKind::Boss] {
        let (fx, _) = blind_select(&mut jokers(&[J::Burglar]), blind, 1);
        assert_eq!(
            fx,
            vec![SideEffect::AddHands(3), SideEffect::ClearDiscards],
            "{blind:?}"
        );
    }
    let mut dead = vec![jk(1, J::Burglar).with_debuff(true)];
    assert!(fx_of(&mut dead).is_empty());
    // Blueprint doubles the hands (and the idempotent clear).
    let fx = fx_of(&mut jokers(&[J::Blueprint, J::Burglar]));
    let hands: i32 = fx
        .iter()
        .map(|e| {
            if let SideEffect::AddHands(n) = e {
                *n
            } else {
                0
            }
        })
        .sum();
    assert_eq!(hands, 6);
}

#[test]
fn burglar_game_gives_seven_hands_and_no_discards() {
    let mut g = new_game(83);
    add(&mut g, J::Burglar);
    select(&mut g);
    let s = g.state();
    assert_eq!(s.hands_left, 4 + 3);
    assert_eq!(s.discards_left, 0);
    assert!(
        g.legal_actions()
            .iter()
            .all(|a| !matches!(a, Action::Discard(_))),
        "no discard is legal"
    );
    s.check_invariants().expect("invariants");
    // With Blueprint to its left: +6 hands.
    let mut g = new_game(84);
    add(&mut g, J::Blueprint);
    add(&mut g, J::Burglar);
    select(&mut g);
    assert_eq!(g.state().hands_left, 4 + 6);
    assert_eq!(g.state().discards_left, 0);
}

// ----- DNA -----

#[test]
fn dna_copies_a_single_card_on_the_first_hand() {
    let mut card = seal(c(7, Ace, Spades), Seal::Red);
    card.edition = Edition::Foil;
    card.perma_bonus = 5;
    card.enhancement = Enhancement::Bonus;
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let o = score_with(&[card], eval, &mut jokers(&[J::DNA]), &first_hand());
    assert_eq!(
        o.effects,
        vec![SideEffect::AddCard {
            card,
            to: Pile::Hand
        }]
    );
    // A debuffed card is copied as a fresh, undebuffed card.
    let mut sick = debuff(card);
    sick.face_down = true;
    let o = score_with(&[sick], eval, &mut jokers(&[J::DNA]), &first_hand());
    assert_eq!(
        o.effects,
        vec![SideEffect::AddCard {
            card,
            to: Pile::Hand
        }]
    );
}

#[test]
fn dna_needs_first_hand_and_exactly_one_card() {
    let one = [c(1, Ace, Spades)];
    let eval1 = ev(HandType::HighCard, 0b1, &[]);
    // Second hand of the round.
    let second = RunView {
        hands_played_round: 2,
        ..RunView::default()
    };
    let o = score_with(&one, eval1, &mut jokers(&[J::DNA]), &second);
    assert!(o.effects.is_empty());
    // Two cards on the first hand.
    let two = [c(1, Seven, Spades), c(2, Seven, Hearts)];
    let o = score_with(
        &two,
        ev(HandType::Pair, 0b11, &[]),
        &mut jokers(&[J::DNA]),
        &first_hand(),
    );
    assert!(o.effects.is_empty());
    // Debuffed DNA.
    let mut dead = vec![jk(1, J::DNA).with_debuff(true)];
    let o = score_with(&one, eval1, &mut dead, &first_hand());
    assert!(o.effects.is_empty());
    // A hand rejected by the boss triggers nothing.
    let mut run = first_hand();
    run.boss.hand_invalid = true;
    let o = score_with(&one, eval1, &mut jokers(&[J::DNA]), &run);
    assert!(o.effects.is_empty());
}

#[test]
fn dna_blueprint_adds_a_second_copy() {
    let o = score_with(
        &[c(1, King, Hearts)],
        ev(HandType::HighCard, 0b1, &[]),
        &mut jokers(&[J::Blueprint, J::DNA]),
        &first_hand(),
    );
    assert_eq!(o.effects.len(), 2);
    assert!(
        o.effects
            .iter()
            .all(|e| matches!(e, SideEffect::AddCard { to: Pile::Hand, .. }))
    );
}

#[test]
fn dna_game_adds_a_permanent_copy_to_hand_once() {
    let mut g = new_game(85);
    add(&mut g, J::DNA);
    select(&mut g);
    set_hand(&mut g, 0, Ace, Spades);
    g.state_mut().deck.hand[0].edition = Edition::Foil;
    g.state_mut().deck.hand[0].seal = Seal::Red;
    let original = g.state().deck.hand[0].uid;
    let info = play(&mut g, 0b1);
    assert_eq!(info.cards_added, 1);
    let s = g.state();
    assert_eq!(s.deck.len(), 53);
    assert_eq!((s.stats.cards_added, s.stats.cards_destroyed), (1, 0));
    let copy = s
        .deck
        .iter_all()
        .find(|c| c.uid >= 52)
        .copied()
        .expect("copy in the deck");
    assert_eq!((copy.rank, copy.suit), (Ace, Spades));
    assert_eq!((copy.edition, copy.seal), (Edition::Foil, Seal::Red));
    assert_ne!(copy.uid, original);
    assert!(
        s.deck.hand.iter().any(|c| c.uid == copy.uid),
        "the copy is drawn into the hand"
    );
    // Second hand: no further copy.
    let info = play(&mut g, 0b1);
    assert_eq!(info.cards_added, 0);
    assert_eq!(g.state().deck.len(), 53);
}

// ----- Sixth Sense -----

#[test]
fn sixth_sense_destroys_a_single_first_hand_six() {
    let six = c(9, Six, Hearts);
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let o = score_with(&[six], eval, &mut jokers(&[J::SixthSense]), &first_hand());
    assert_eq!(
        o.effects,
        vec![SideEffect::DestroyCard { uid: 9 }, spectral()]
    );
    // Works even if the 6 is debuffed.
    let o = score_with(
        &[debuff(six)],
        eval,
        &mut jokers(&[J::SixthSense]),
        &first_hand(),
    );
    assert_eq!(o.effects.len(), 2);
}

#[test]
fn sixth_sense_requires_first_hand_one_card_and_a_six() {
    let mut js = jokers(&[J::SixthSense]);
    let eval = ev(HandType::HighCard, 0b1, &[]);
    // Not a 6.
    let o = score_with(&[c(1, Seven, Hearts)], eval, &mut js, &first_hand());
    assert!(o.effects.is_empty());
    // Stone card printed as a 6.
    let stone = enh(c(1, Six, Hearts), Enhancement::Stone);
    let o = score_with(&[stone], eval, &mut js, &first_hand());
    assert!(o.effects.is_empty());
    // Two cards.
    let o = score_with(
        &[c(1, Six, Hearts), c(2, Six, Spades)],
        ev(HandType::Pair, 0b11, &[]),
        &mut js,
        &first_hand(),
    );
    assert!(o.effects.is_empty());
    // Second hand.
    let second = RunView {
        hands_played_round: 2,
        ..RunView::default()
    };
    let o = score_with(&[c(1, Six, Hearts)], eval, &mut js, &second);
    assert!(o.effects.is_empty());
    // Boss rejected the hand.
    let mut run = first_hand();
    run.boss.hand_invalid = true;
    let o = score_with(&[c(1, Six, Hearts)], eval, &mut js, &run);
    assert!(o.effects.is_empty());
    // Debuffed joker, and Blueprint cannot copy it.
    let mut dead = vec![jk(1, J::SixthSense).with_debuff(true)];
    let o = score_with(&[c(1, Six, Hearts)], eval, &mut dead, &first_hand());
    assert!(o.effects.is_empty());
    // Blueprint cannot copy it (blueprint_compatible: false): the effects are those of the
    // Sixth Sense itself, not doubled.
    let o = score_with(
        &[c(1, Six, Hearts)],
        eval,
        &mut jokers(&[J::Blueprint, J::SixthSense]),
        &first_hand(),
    );
    assert_eq!(o.effects.len(), 2);
}

/// Consumables are dropped until Phase 4 and created afterwards: either way each requested
/// consumable is accounted for exactly once.
fn consumable_accounting(g: &Game, expected: u32) {
    let s = g.state();
    assert_eq!(
        s.consumables.len() as u32 + s.stats.effects_no_room,
        expected,
        "consumables created + dropped"
    );
}

#[test]
fn sixth_sense_game_destroys_the_card_for_good() {
    let mut g = new_game(86);
    add(&mut g, J::SixthSense);
    select(&mut g);
    set_hand(&mut g, 0, Six, Clubs);
    let uid = g.state().deck.hand[0].uid;
    let info = play(&mut g, 0b1);
    assert_eq!(info.cards_destroyed, 1);
    let s = g.state();
    assert_eq!(s.deck.len(), 51);
    assert_eq!(s.stats.cards_destroyed, 1);
    assert!(find_card(&g, uid).is_none());
    consumable_accounting(&g, 1);
    // Second hand: another single 6 does nothing.
    set_hand(&mut g, 0, Six, Hearts);
    let info = play(&mut g, 0b1);
    assert_eq!(info.cards_destroyed, 0);
    assert_eq!(g.state().deck.len(), 51);
    consumable_accounting(&g, 1);
}

#[test]
fn sixth_sense_and_dna_together_farm_a_spectral_per_round() {
    // DNA copies the 6 before scoring; Sixth Sense then destroys the original: net 52 cards.
    let mut g = new_game(87);
    add(&mut g, J::SixthSense);
    add(&mut g, J::DNA);
    select(&mut g);
    set_hand(&mut g, 0, Six, Clubs);
    let original = g.state().deck.hand[0].uid;
    let info = play(&mut g, 0b1);
    assert_eq!((info.cards_added, info.cards_destroyed), (1, 1));
    let s = g.state();
    assert_eq!(s.deck.len(), 52);
    assert!(find_card(&g, original).is_none());
    let copy = s.deck.iter_all().find(|c| c.uid >= 52).expect("copy");
    assert_eq!((copy.rank, copy.suit), (Six, Clubs));
    consumable_accounting(&g, 1);
}

// ----- Superposition -----

#[test]
fn superposition_needs_a_scoring_ace_and_a_straight() {
    let straight = |ranks: [Rank; 5]| {
        let suits = [Spades, Hearts, Clubs, Diamonds, Spades];
        let cards: Vec<Card> = ranks
            .iter()
            .zip(suits)
            .enumerate()
            .map(|(i, (&r, s))| c(i as u32 + 1, r, s))
            .collect();
        let eval = ev(HandType::Straight, 0b11111, &[HandType::Straight]);
        score(&cards, eval, &mut jokers(&[J::Superposition])).effects
    };
    // Ace high and Ace low straights.
    assert_eq!(straight([Ten, Jack, Queen, King, Ace]), vec![tarot()]);
    assert_eq!(straight([Ace, Two, Three, Four, Five]), vec![tarot()]);
    // Straight without an Ace.
    assert!(straight([Five, Six, Seven, Eight, Nine]).is_empty());
}

#[test]
fn superposition_edge_cases() {
    let mut js = jokers(&[J::Superposition]);
    // Ace but no Straight: Pair of Aces.
    let aces = [c(1, Ace, Spades), c(2, Ace, Hearts)];
    assert!(
        score(&aces, ev(HandType::Pair, 0b11, &[]), &mut js)
            .effects
            .is_empty()
    );
    // Four Fingers straight 6 7 8 9 with the Ace as a non-scoring fifth card.
    let cards = [
        c(1, Six, Spades),
        c(2, Seven, Hearts),
        c(3, Eight, Clubs),
        c(4, Nine, Diamonds),
        c(5, Ace, Spades),
    ];
    let eval = ev(HandType::Straight, 0b01111, &[HandType::Straight]);
    assert!(score(&cards, eval, &mut js).effects.is_empty());
    // With Splash every card scores, so the Ace counts.
    let eval = ev(HandType::Straight, 0b11111, &[HandType::Straight]);
    assert_eq!(score(&cards, eval, &mut js).effects, vec![tarot()]);
    // Straight Flush contains a Straight.
    let sf = [
        c(1, Ten, Hearts),
        c(2, Jack, Hearts),
        c(3, Queen, Hearts),
        c(4, King, Hearts),
        c(5, Ace, Hearts),
    ];
    let eval = ev(
        HandType::StraightFlush,
        0b11111,
        &[HandType::Straight, HandType::Flush],
    );
    assert_eq!(score(&sf, eval, &mut js).effects, vec![tarot()]);
    // Debuffed joker; Blueprint copy doubles.
    let mut dead = vec![jk(1, J::Superposition).with_debuff(true)];
    assert!(score(&sf, eval, &mut dead).effects.is_empty());
    let o = score(&sf, eval, &mut jokers(&[J::Blueprint, J::Superposition]));
    assert_eq!(o.effects, vec![tarot(), tarot()]);
}

// ----- Seance -----

#[test]
fn seance_creates_a_spectral_on_a_straight_flush() {
    let sf = [
        c(1, Five, Clubs),
        c(2, Six, Clubs),
        c(3, Seven, Clubs),
        c(4, Eight, Clubs),
        c(5, Nine, Clubs),
    ];
    let eval = ev(
        HandType::StraightFlush,
        0b11111,
        &[HandType::Straight, HandType::Flush],
    );
    let o = score(&sf, eval, &mut jokers(&[J::Seance]));
    assert_eq!(o.effects, vec![spectral()]);
    // Plain Straight and plain Flush do not trigger it.
    let o = score(
        &sf,
        ev(HandType::Straight, 0b11111, &[]),
        &mut jokers(&[J::Seance]),
    );
    assert!(o.effects.is_empty());
    let o = score(
        &sf,
        ev(HandType::Flush, 0b11111, &[]),
        &mut jokers(&[J::Seance]),
    );
    assert!(o.effects.is_empty());
    // Flush Five / Flush House are not Straight Flushes.
    let o = score(
        &sf,
        ev(HandType::FlushHouse, 0b11111, &[HandType::Flush]),
        &mut jokers(&[J::Seance]),
    );
    assert!(o.effects.is_empty());
    // Debuffed joker; Blueprint doubles.
    let mut dead = vec![jk(1, J::Seance).with_debuff(true)];
    assert!(score(&sf, eval, &mut dead).effects.is_empty());
    let o = score(&sf, eval, &mut jokers(&[J::Blueprint, J::Seance]));
    assert_eq!(o.effects, vec![spectral(), spectral()]);
}

// ----- Riff-Raff -----

#[test]
fn riff_raff_requests_two_common_jokers_per_blind() {
    for blind in [BlindKind::Small, BlindKind::Big, BlindKind::Boss] {
        let (fx, _) = blind_select(&mut jokers(&[J::RiffRaff]), blind, 1);
        let common = SideEffect::CreateJoker {
            rarity: Some(Rarity::Common),
        };
        assert_eq!(fx, vec![common, common], "{blind:?}");
    }
    let mut dead = vec![jk(1, J::RiffRaff).with_debuff(true)];
    assert!(fx_of(&mut dead).is_empty());
    assert_eq!(fx_of(&mut jokers(&[J::Blueprint, J::RiffRaff])).len(), 4);
}

#[test]
fn riff_raff_game_never_overfills_the_joker_area() {
    // Phase 5 will create the jokers; until then the game layer drops the request. Either way
    // the number of jokers grows by at most two and every request is accounted for.
    let mut g = new_game(88);
    add(&mut g, J::RiffRaff);
    select(&mut g);
    let s = g.state();
    assert!(s.jokers.len() <= 3);
    assert_eq!(s.jokers.len() as u32 - 1 + s.stats.effects_no_room, 2);
    s.check_invariants().expect("invariants");
    // A full joker area: nothing is added.
    let mut g = new_game(89);
    for _ in 0..4 {
        add(&mut g, J::Joker);
    }
    add(&mut g, J::RiffRaff);
    select(&mut g);
    assert_eq!(g.state().jokers.len(), 5);
    g.state().check_invariants().expect("invariants");
}

// ----- Vagabond -----

#[test]
fn vagabond_triggers_at_four_dollars_or_less() {
    let played = [c(1, Seven, Spades)];
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let at = |money: i32| {
        let run = RunView {
            money,
            ..RunView::default()
        };
        score_with(&played, eval, &mut jokers(&[J::Vagabond]), &run).effects
    };
    assert_eq!(at(0), vec![tarot()]);
    assert_eq!(at(4), vec![tarot()]);
    assert_eq!(at(-3), vec![tarot()]); // debt counts as "$4 or less"
    assert!(at(5).is_empty());
    assert!(at(100).is_empty());
    // Debuffed joker.
    let mut dead = vec![jk(1, J::Vagabond).with_debuff(true)];
    assert!(score(&played, eval, &mut dead).effects.is_empty());
}

#[test]
fn vagabond_sees_money_earned_earlier_in_the_hand_and_blueprint_copies() {
    // $1 + a Gold Seal card scoring (+$3) = $4: still triggers; $2 + $3 = $5: does not.
    let gold = [seal(c(1, Seven, Spades), Seal::Gold)];
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let run = |money| RunView {
        money,
        ..RunView::default()
    };
    let o = score_with(&gold, eval, &mut jokers(&[J::Vagabond]), &run(1));
    assert_eq!(o.effects, vec![tarot()]);
    let o = score_with(&gold, eval, &mut jokers(&[J::Vagabond]), &run(2));
    assert!(o.effects.is_empty());
    // Blueprint + Vagabond.
    let o = score_with(
        &[c(1, Seven, Spades)],
        eval,
        &mut jokers(&[J::Blueprint, J::Vagabond]),
        &run(0),
    );
    assert_eq!(o.effects, vec![tarot(), tarot()]);
}

// ----- Midas Mask -----

#[test]
fn midas_turns_scoring_faces_gold_and_keeps_edition_and_seal() {
    let mut king = enh(c(1, King, Spades), Enhancement::Glass);
    king.edition = Edition::Foil;
    king.seal = Seal::Red;
    let played = [
        king,
        c(2, Queen, Hearts),
        c(3, Jack, Clubs),
        c(4, Seven, Clubs),
    ];
    // Pair stand-in scoring the K and the Q only; the J is a non-scoring face, the 7 not a face.
    let o = score(
        &played,
        ev(HandType::Pair, 0b0011, &[]),
        &mut jokers(&[J::MidasMask]),
    );
    assert_eq!(o.played[0].enhancement, Enhancement::Gold);
    assert_eq!(o.played[0].edition, Edition::Foil);
    assert_eq!(o.played[0].seal, Seal::Red);
    assert_eq!(o.played[1].enhancement, Enhancement::Gold);
    assert_eq!(o.played[2].enhancement, Enhancement::None, "unscored face");
    assert_eq!(o.played[3].enhancement, Enhancement::None, "not a face");
    // The Glass x2 was replaced by Gold before scoring:
    // Pair base 10 x 2; Q 10; Red Seal K triggers twice: 2 x (10 + Foil 50) = 120; chips
    // 10 + 10 + 120 = 140 x 2 = 280 (no Glass x2: the enhancement was overwritten).
    assert_eq!(o.res.score, 280.0);
}

#[test]
fn midas_ignores_debuffed_faces_and_respects_pareidolia() {
    let eval = ev(HandType::HighCard, 0b1, &[]);
    let o = score(
        &[debuff(c(1, King, Spades))],
        eval,
        &mut jokers(&[J::MidasMask]),
    );
    assert_eq!(o.played[0].enhancement, Enhancement::None);
    // A 7 is not a face... until Pareidolia: then even a Stone card becomes Gold.
    let seven = [c(1, Seven, Spades)];
    let o = score(&seven, eval, &mut jokers(&[J::MidasMask]));
    assert_eq!(o.played[0].enhancement, Enhancement::None);
    let run = RunView {
        rules: collect_modifiers(&jokers(&[J::Pareidolia])).rules,
        ..RunView::default()
    };
    let o = score_with(&seven, eval, &mut jokers(&[J::MidasMask]), &run);
    assert_eq!(o.played[0].enhancement, Enhancement::Gold);
    let stone = [enh(c(1, Two, Spades), Enhancement::Stone)];
    let o = score_with(&stone, eval, &mut jokers(&[J::MidasMask]), &run);
    assert_eq!(o.played[0].enhancement, Enhancement::Gold);
    // Debuffed joker does nothing (Blueprint cannot copy Midas: see the flags test).
    let king = [c(1, King, Spades)];
    let mut dead = vec![jk(1, J::MidasMask).with_debuff(true)];
    assert_eq!(
        score(&king, eval, &mut dead).played[0].enhancement,
        Enhancement::None
    );
}

#[test]
fn midas_game_makes_the_card_gold_permanently() {
    let mut g = new_game(90);
    add(&mut g, J::MidasMask);
    select(&mut g);
    set_hand(&mut g, 0, King, Spades);
    let uid = g.state().deck.hand[0].uid;
    play(&mut g, 0b1);
    let card = find_card(&g, uid).expect("card still in the deck");
    assert_eq!(card.enhancement, Enhancement::Gold);
    // Pareidolia + Midas: a plain 7 turns Gold as well.
    let mut g = new_game(91);
    add(&mut g, J::Pareidolia);
    add(&mut g, J::MidasMask);
    select(&mut g);
    set_hand(&mut g, 0, Seven, Clubs);
    let uid = g.state().deck.hand[0].uid;
    play(&mut g, 0b1);
    assert_eq!(
        find_card(&g, uid).expect("card").enhancement,
        Enhancement::Gold
    );
}

// ----- whole-wave soak -----

fn soak(seed: u64) -> Game {
    let mut g = new_game(seed);
    // Five of the fourteen, rotating with the seed.
    for k in 0..5 {
        let id = WAVE[(seed as usize * 3 + k) % WAVE.len()];
        let _ = g.add_joker(JokerInstance::new(0, id));
    }
    let mut policy = GameRng::new(seed ^ 0xABCD);
    let mut legal = Vec::new();
    for _ in 0..400 {
        g.legal_actions_into(&mut legal);
        if legal.is_empty() {
            break;
        }
        let a = legal[policy.range(RngKey::Misc, 0, legal.len())];
        g.step(a).expect("legal action"); // debug builds re-check every invariant
        g.state().check_invariants().expect("invariants");
    }
    g
}

#[test]
fn random_play_with_wave4a_jokers_is_deterministic_and_keeps_invariants() {
    for seed in 0..28 {
        let a = soak(seed);
        let b = soak(seed);
        assert_eq!(a.state(), b.state(), "seed {seed}");
    }
}
