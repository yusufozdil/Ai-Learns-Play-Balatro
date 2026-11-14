//! Phase 3 package wave4b: Turtle Bean, Hallucination, Juggler, Drunkard, Diet Cola, Seltzer,
//! Sock and Buskin, Troubadour, Certificate, Hanging Chad, Merry Andy, Cartomancer, Perkeo.
//! Scoring jokers are tested through `Scorer`, event jokers through the `fire_*` dispatchers,
//! passive jokers and Certificate additionally through `Game`. Every expected number is derived
//! in a comment.

use balatro_core::actions::Action;
use balatro_core::card::{Card, RuleFlags};
use balatro_core::content::BossId;
use balatro_core::content::ConsumableId;
use balatro_core::content::TagId;
use balatro_core::content::jokers::{JokerId, def_count, try_def};
use balatro_core::deck::{Deck, Pile};
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Rarity, Seal, Suit};
use balatro_core::game::Game;
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{
    BlindKind, BossView, ConsumableKind, ConsumableView, EventInput, EventResult, PackKind,
    RoundEndInfo, RunView, SideEffect, collect_modifiers, fire_blind_select, fire_first_hand_drawn,
    fire_pack_open, fire_round_end, fire_sell_self, fire_shop_exit,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer};
use balatro_core::state::GameConfig;

use JokerId as J;
use Rank::*;
use Suit::*;

const WAVE: [JokerId; 13] = [
    J::TurtleBean,
    J::Hallucination,
    J::Juggler,
    J::Drunkard,
    J::DietCola,
    J::Seltzer,
    J::SockAndBuskin,
    J::Troubadour,
    J::Certificate,
    J::HangingChad,
    J::MerryAndy,
    J::Cartomancer,
    J::Perkeo,
];

// ----- helpers -----

fn c(uid: u32, rank: Rank, suit: Suit) -> Card {
    Card::new(uid, rank, suit)
}

fn seal(mut card: Card, s: Seal) -> Card {
    card.seal = s;
    card
}

fn debuff(mut card: Card) -> Card {
    card.debuffed = true;
    card
}

fn ev(hand_type: HandType, scoring_mask: u8) -> HandEval {
    HandEval {
        hand_type,
        scoring_mask,
        contains: 1 << hand_type.index(),
    }
}

/// Jokers with uids 100, 101, ... in the given order.
fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| JokerInstance::new(100 + i as u32, id))
        .collect()
}

/// 7♠ 7♥ Pair: 10 + 7 + 7 = 24 x 2 = 48 with no jokers.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11),
    )
}

/// K♠ K♥ Pair: 10 + 10 + 10 = 30 x 2 = 60 with no jokers.
fn pair_k() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, King, Spades), c(2, King, Hearts)],
        ev(HandType::Pair, 0b11),
    )
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
}

fn score_with(played: &[Card], eval: HandEval, js: &mut [JokerInstance], run: &RunView<'_>) -> Out {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(11);
    let mut scorer = Scorer::with_trace(true);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held: &[],
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    });
    Out {
        res,
        effects: scorer.effects().to_vec(),
    }
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

/// Run one event dispatcher on `js` and return the queued side effects.
fn fire(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> Vec<SideEffect> {
    let deck = Deck::standard();
    let mut rng = GameRng::new(seed);
    let mut fx = Vec::new();
    f(EventInput {
        jokers: js,
        run,
        deck: &deck,
        rng: &mut rng,
        effects: &mut fx,
    });
    fx
}

fn tarot() -> SideEffect {
    SideEffect::CreateConsumable {
        kind: ConsumableKind::Tarot,
        negative: false,
    }
}

fn cons(id: u16) -> ConsumableView {
    ConsumableView::new(ConsumableId::TAROTS[usize::from(id)])
}

fn round_end_info() -> RoundEndInfo {
    RoundEndInfo {
        won: true,
        saved: false,
        round_score: 400.0,
        target: 300.0,
    }
}

// ----- Game helpers -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id))
        .expect("room for joker")
}

fn select(g: &mut Game) {
    // Boss blinds get The Wall: no ability that could stop a one-card win (The Psychic).
    if g.state().blind == BlindKind::Boss {
        g.state_mut().boss = Some(BossId::TheWall);
    }
    g.step(Action::SelectBlind).expect("select blind");
}

/// Beat the current blind with one hand: round score one chip below the target, then play the
/// leftmost card (any hand scores at least 7).
fn beat_blind(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play");
}

// ----- registry / definitions -----

#[test]
fn all_13_jokers_are_defined_exactly_once() {
    for id in WAVE {
        assert_eq!(def_count(id), 1, "{id:?}");
        assert_eq!(try_def(id).expect("def").id, id);
    }
}

#[test]
fn definitions_match_the_csv() {
    // (id, name, rarity, cost, blueprint-copyable, eternal-compatible)
    let table = [
        (
            J::TurtleBean,
            "Turtle Bean",
            Rarity::Uncommon,
            6,
            false,
            false,
        ),
        (
            J::Hallucination,
            "Hallucination",
            Rarity::Common,
            4,
            true,
            true,
        ),
        (J::Juggler, "Juggler", Rarity::Common, 4, false, true),
        (J::Drunkard, "Drunkard", Rarity::Common, 4, false, true),
        (J::DietCola, "Diet Cola", Rarity::Uncommon, 6, true, false),
        (J::Seltzer, "Seltzer", Rarity::Uncommon, 6, true, false),
        (
            J::SockAndBuskin,
            "Sock and Buskin",
            Rarity::Uncommon,
            6,
            true,
            true,
        ),
        (
            J::Troubadour,
            "Troubadour",
            Rarity::Uncommon,
            6,
            false,
            true,
        ),
        (
            J::Certificate,
            "Certificate",
            Rarity::Uncommon,
            6,
            true,
            true,
        ),
        (
            J::HangingChad,
            "Hanging Chad",
            Rarity::Common,
            4,
            true,
            true,
        ),
        (J::MerryAndy, "Merry Andy", Rarity::Uncommon, 7, false, true),
        (
            J::Cartomancer,
            "Cartomancer",
            Rarity::Uncommon,
            6,
            true,
            true,
        ),
        (J::Perkeo, "Perkeo", Rarity::Legendary, 20, true, true),
    ];
    for (id, name, rarity, cost, bp, eternal) in table {
        let d = try_def(id).expect("def");
        assert_eq!(d.name, name);
        assert_eq!(d.rarity, rarity, "{name}");
        assert_eq!(d.base_cost, cost, "{name}");
        assert_eq!(d.blueprint_compatible, bp, "{name} blueprint");
        assert_eq!(d.eternal_compatible, eternal, "{name} eternal");
    }
    assert_eq!(try_def(J::Seltzer).expect("def").init_counter, 10);
    assert_eq!(try_def(J::TurtleBean).expect("def").init_value, 5.0);
}

// ----- passive modifiers (collect_modifiers) -----

#[test]
fn passive_modifiers_values() {
    let m = |id| collect_modifiers(&jokers(&[id]));
    assert_eq!(m(J::Juggler).hand_size, 1);
    assert_eq!(m(J::Drunkard).discards_per_round, 1);
    let t = m(J::Troubadour);
    assert_eq!((t.hand_size, t.hands_per_round), (2, -1));
    let a = m(J::MerryAndy);
    assert_eq!((a.discards_per_round, a.hand_size), (3, -1));
    assert_eq!(m(J::TurtleBean).hand_size, 5);
    // They stack: Juggler +1, Troubadour +2, Merry Andy -1 = +2 hand size; hands -1; discards +3.
    let all = collect_modifiers(&jokers(&[J::Juggler, J::Troubadour, J::MerryAndy]));
    assert_eq!(all.hand_size, 2);
    assert_eq!(all.hands_per_round, -1);
    assert_eq!(all.discards_per_round, 3);
}

#[test]
fn passives_ignore_blueprint_and_debuff() {
    // Blueprint never copies passives (and these are copy-incompatible anyway).
    for id in [
        J::Juggler,
        J::Drunkard,
        J::Troubadour,
        J::MerryAndy,
        J::TurtleBean,
    ] {
        let m = collect_modifiers(&jokers(&[J::Blueprint, id]));
        let single = collect_modifiers(&jokers(&[id]));
        assert_eq!(m, single, "{id:?}");
        // A debuffed joker gives nothing.
        let mut js = jokers(&[id]);
        js[0].debuffed = true;
        assert_eq!(collect_modifiers(&js), collect_modifiers(&[]), "{id:?}");
    }
}

// ----- passive modifiers at Game level (start of round) -----

fn round_stats(ids: &[JokerId]) -> (usize, u32, u32) {
    let mut g = new_game(5);
    for &id in ids {
        add(&mut g, id);
    }
    select(&mut g);
    let s = g.state();
    assert!(s.check_invariants().is_ok());
    (s.deck.hand.len(), s.hands_left, s.discards_left)
}

#[test]
fn game_baseline_round_stats() {
    // 8 cards, 4 hands, 3 + 1 (Red Deck) discards.
    assert_eq!(round_stats(&[]), (8, 4, 4));
}

#[test]
fn juggler_adds_one_hand_size() {
    assert_eq!(round_stats(&[J::Juggler]), (9, 4, 4));
}

#[test]
fn drunkard_adds_one_discard() {
    assert_eq!(round_stats(&[J::Drunkard]), (8, 4, 5));
}

#[test]
fn troubadour_adds_two_hand_size_and_removes_a_hand() {
    assert_eq!(round_stats(&[J::Troubadour]), (10, 3, 4));
}

#[test]
fn merry_andy_adds_three_discards_and_removes_hand_size() {
    assert_eq!(round_stats(&[J::MerryAndy]), (7, 4, 7));
}

#[test]
fn turtle_bean_adds_five_hand_size() {
    assert_eq!(round_stats(&[J::TurtleBean]), (13, 4, 4));
}

#[test]
fn passives_stack_at_game_level() {
    // hand size 8 + 1 + 2 - 1 = 10; hands 4 - 1 = 3; discards 4 + 1 + 3 = 8.
    assert_eq!(
        round_stats(&[J::Juggler, J::Troubadour, J::MerryAndy, J::Drunkard]),
        (10, 3, 8)
    );
}

#[test]
fn selling_a_passive_joker_removes_its_effect_next_round() {
    let mut g = new_game(6);
    add(&mut g, J::Juggler);
    select(&mut g);
    assert_eq!(g.state().deck.hand.len(), 9);
    // The modifier cache follows the sale at once: hand size is back to 8.
    g.step(Action::SellJoker(0)).expect("sell");
    assert_eq!(g.state().hand_size(), 8);
    assert!(g.state().check_invariants().is_ok());
    beat_blind(&mut g);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    assert_eq!(g.state().deck.hand.len(), 8);
}

// ----- Turtle Bean -----

#[test]
fn turtle_bean_decays_each_round_and_is_eaten_after_five() {
    let mut g = new_game(7);
    add(&mut g, J::TurtleBean);
    // Round 1 starts with +5; after each round the bonus shrinks by 1.
    // hand sizes of rounds 1..=5: 13, 12, 11, 10, 9; round 6: 8 (the joker is gone).
    let expected_hand = [13usize, 12, 11, 10, 9];
    for (round, &hand) in expected_hand.iter().enumerate() {
        select(&mut g);
        assert_eq!(g.state().deck.hand.len(), hand, "round {}", round + 1);
        assert_eq!(g.state().jokers.len(), 1);
        beat_blind(&mut g);
        // value after the round: 4, 3, 2, 1, 0 (destroyed at 0).
        let left = 4 - round as i32;
        if left > 0 {
            assert_eq!(g.state().jokers[0].value, f64::from(left));
        } else {
            assert!(g.state().jokers.is_empty(), "eaten at 0");
        }
        assert!(g.state().check_invariants().is_ok());
        g.step(Action::EndShop).expect("end shop");
    }
    assert_eq!(g.state().mods.hand_size, 0);
    select(&mut g);
    assert_eq!(g.state().deck.hand.len(), 8);
    assert_eq!(g.state().stats.effects_no_room, 0);
}

#[test]
fn turtle_bean_round_end_event() {
    // value 5 -> 4: no destruction.
    let mut js = jokers(&[J::TurtleBean]);
    let fx = fire(&mut js, &RunView::default(), 1, |inp| {
        fire_round_end(inp, &round_end_info())
    });
    assert_eq!(js[0].value, 4.0);
    assert!(fx.is_empty());
    // value 1 -> 0: destroyed.
    let mut js = jokers(&[J::TurtleBean]);
    js[0].value = 1.0;
    let fx = fire(&mut js, &RunView::default(), 1, |inp| {
        fire_round_end(inp, &round_end_info())
    });
    assert_eq!(js[0].value, 0.0);
    assert_eq!(fx, vec![SideEffect::DestroyJoker { uid: 100 }]);
    // A bean at 0 gives no hand size.
    assert_eq!(collect_modifiers(&js).hand_size, 0);
}

#[test]
fn turtle_bean_end_of_round_is_not_copied() {
    // Blueprint cannot copy Turtle Bean (incompatible): the bean decays once, no extra effects.
    let mut js = jokers(&[J::Blueprint, J::TurtleBean]);
    let fx = fire(&mut js, &RunView::default(), 1, |inp| {
        fire_round_end(inp, &round_end_info())
    });
    assert_eq!(js[1].value, 4.0);
    assert!(fx.is_empty());
}

// ----- Sock and Buskin -----

#[test]
fn sock_and_buskin_retriggers_face_cards() {
    // Pair of Kings, each twice: 10 + 4 x 10 = 50 x 2 = 100.
    let (p, e) = pair_k();
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        100.0
    );
}

#[test]
fn sock_and_buskin_ignores_non_faces() {
    // Sevens untouched: 48.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        48.0
    );
    // Ace is not a face card: pair of aces 10 + 22 = 32 x 2 = 64.
    let p = [c(1, Ace, Spades), c(2, Ace, Hearts)];
    let e = ev(HandType::Pair, 0b11);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        64.0
    );
}

#[test]
fn sock_and_buskin_only_scoring_cards() {
    // K♠ K♥ pair plus an unscored K♣ kicker: only the pair retriggers: 100.
    let p = [c(1, King, Spades), c(2, King, Hearts), c(3, King, Clubs)];
    // Three Kings would be a trio; use a scoring mask of the first two to model a kicker.
    let e = ev(HandType::Pair, 0b011);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        100.0
    );
}

#[test]
fn sock_and_buskin_pareidolia_makes_every_card_a_face() {
    // 7♠ 7♥ with Pareidolia: both twice: 10 + 4 x 7 = 38 x 2 = 76.
    let (p, e) = pair7();
    let run = RunView {
        rules: RuleFlags {
            pareidolia: true,
            ..RuleFlags::default()
        },
        ..RunView::default()
    };
    assert_eq!(
        score_with(&p, e, &mut jokers(&[J::SockAndBuskin]), &run)
            .res
            .score,
        76.0
    );
}

#[test]
fn sock_and_buskin_debuffed_face_is_skipped() {
    // K♠ debuffed (scores nothing, no retrigger); K♥ twice: 10 + 20 = 30 x 2 = 60.
    let (mut p, e) = pair_k();
    p[0] = debuff(p[0]);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        60.0
    );
}

#[test]
fn sock_and_buskin_stacks_with_red_seal() {
    // Red Seal K♠: 1 + 1 (seal) + 1 (joker) = 3 triggers: 30; K♥ twice: 20; 10 + 50 = 60 x 2 = 120.
    let (mut p, e) = pair_k();
    p[0] = seal(p[0], Seal::Red);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin])).res.score,
        120.0
    );
}

#[test]
fn sock_and_buskin_gold_seal_pays_per_trigger() {
    // Gold Seal K♠ with Sock and Buskin: 2 triggers = $6.
    let (mut p, e) = pair_k();
    p[0] = seal(p[0], Seal::Gold);
    let o = score(&p, e, &mut jokers(&[J::SockAndBuskin]));
    assert_eq!((o.res.score, o.res.money_delta), (100.0, 6));
}

#[test]
fn sock_and_buskin_blueprint_copy_adds_another_retrigger() {
    // Blueprint + Sock and Buskin: each King 1 + 1 + 1 = 3 triggers: 10 + 6 x 10 = 70 x 2 = 140.
    let (p, e) = pair_k();
    let mut js = jokers(&[J::Blueprint, J::SockAndBuskin]);
    assert_eq!(score(&p, e, &mut js).res.score, 140.0);
    // Two Sock and Buskin do the same.
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SockAndBuskin, J::SockAndBuskin]))
            .res
            .score,
        140.0
    );
}

#[test]
fn sock_and_buskin_debuffed_joker_does_nothing() {
    let (p, e) = pair_k();
    let mut js = jokers(&[J::SockAndBuskin]);
    js[0].debuffed = true;
    assert_eq!(score(&p, e, &mut js).res.score, 60.0);
}

// ----- Hanging Chad -----

#[test]
fn hanging_chad_retriggers_first_scoring_card_twice() {
    // First 7 three times: 10 + 21 + 7 = 38 x 2 = 76.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut jokers(&[J::HangingChad])).res.score, 76.0);
}

#[test]
fn hanging_chad_uses_scoring_position_not_card_index() {
    // A♣ is played but does not score (mask 0b110): the first SCORING card is 7♠ (index 1).
    // 10 + 7 x 3 + 7 = 38 x 2 = 76.
    let p = [c(1, Ace, Clubs), c(2, Seven, Spades), c(3, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b110);
    assert_eq!(score(&p, e, &mut jokers(&[J::HangingChad])).res.score, 76.0);
}

#[test]
fn hanging_chad_with_red_seal_gives_four_triggers() {
    // Red Seal 7♠: 1 + 1 + 2 = 4 triggers (the seal is not retriggered): 10 + 28 + 7 = 45 x 2 = 90.
    let (mut p, e) = pair7();
    p[0] = seal(p[0], Seal::Red);
    assert_eq!(score(&p, e, &mut jokers(&[J::HangingChad])).res.score, 90.0);
    // Red Seal on the SECOND card only gets its own +1: 10 + 21 + 14 = 45 x 2 = 90.
    let (mut p, e) = pair7();
    p[1] = seal(p[1], Seal::Red);
    assert_eq!(score(&p, e, &mut jokers(&[J::HangingChad])).res.score, 90.0);
}

#[test]
fn hanging_chad_debuffed_first_card_gives_nothing() {
    // The first scoring card is debuffed: it is skipped, and the second is not "first".
    // 10 + 0 + 7 = 17 x 2 = 34.
    let (mut p, e) = pair7();
    p[0] = debuff(p[0]);
    assert_eq!(score(&p, e, &mut jokers(&[J::HangingChad])).res.score, 34.0);
}

#[test]
fn hanging_chad_blueprint_and_stacking() {
    // Blueprint + Hanging Chad: first card 1 + 2 + 2 = 5 triggers: 10 + 35 + 7 = 52 x 2 = 104.
    let (p, e) = pair7();
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::HangingChad]))
            .res
            .score,
        104.0
    );
    // Hanging Chad + Seltzer: first card 1 + 2 + 1 = 4 triggers (28), second 2 triggers (14):
    // 10 + 42 = 52 x 2 = 104.
    assert_eq!(
        score(&p, e, &mut jokers(&[J::HangingChad, J::Seltzer]))
            .res
            .score,
        104.0
    );
}

#[test]
fn hanging_chad_pays_gold_seal_per_trigger() {
    // Gold Seal first card: 3 triggers = $9; score as without the seal: 76.
    let (mut p, e) = pair7();
    p[0] = seal(p[0], Seal::Gold);
    let o = score(&p, e, &mut jokers(&[J::HangingChad]));
    assert_eq!((o.res.score, o.res.money_delta), (76.0, 9));
}

// ----- Seltzer -----

#[test]
fn seltzer_retriggers_all_scoring_cards() {
    // Both sevens twice: 10 + 28 = 38 x 2 = 76. Counter 10 -> 9.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Seltzer]);
    let o = score(&p, e, &mut js);
    assert_eq!(o.res.score, 76.0);
    assert_eq!(js[0].counter, 9);
    assert!(o.effects.is_empty());
}

#[test]
fn seltzer_last_hand_is_retriggered_then_destroyed() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Seltzer]);
    js[0].counter = 1;
    let o = score(&p, e, &mut js);
    assert_eq!(o.res.score, 76.0, "the 10th hand still gets the retrigger");
    assert_eq!(o.effects, vec![SideEffect::DestroyJoker { uid: 100 }]);
    assert_eq!(js[0].counter, 0);
}

#[test]
fn seltzer_counts_down_after_a_rejected_hand() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Seltzer]);
    let run = RunView {
        boss: BossView {
            hand_invalid: true,
            ..BossView::default()
        },
        ..RunView::default()
    };
    let o = score_with(&p, e, &mut js, &run);
    assert_eq!(o.res.score, 0.0);
    assert_eq!(js[0].counter, 9);
}

#[test]
fn seltzer_red_seal_triggers_three_times() {
    // Red Seal 7♠: 3 triggers (21); 7♥ twice (14): 10 + 35 = 45 x 2 = 90.
    let (mut p, e) = pair7();
    p[0] = seal(p[0], Seal::Red);
    assert_eq!(score(&p, e, &mut jokers(&[J::Seltzer])).res.score, 90.0);
}

#[test]
fn seltzer_blueprint_copy_retriggers_but_counts_down_once() {
    // Blueprint + Seltzer: every card 3 triggers: 10 + 42 = 52 x 2 = 104.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::Seltzer]);
    let o = score(&p, e, &mut js);
    assert_eq!(o.res.score, 104.0);
    assert_eq!(js[1].counter, 9, "only the real Seltzer counts down");
    // On its last hand exactly one DestroyJoker (of the Seltzer) is queued.
    let mut js = jokers(&[J::Blueprint, J::Seltzer]);
    js[1].counter = 1;
    let o = score(&p, e, &mut js);
    assert_eq!(o.effects, vec![SideEffect::DestroyJoker { uid: 101 }]);
}

#[test]
fn seltzer_debuffed_card_not_retriggered() {
    // 7♠ debuffed; 7♥ twice: 10 + 14 = 24 x 2 = 48.
    let (mut p, e) = pair7();
    p[0] = debuff(p[0]);
    assert_eq!(score(&p, e, &mut jokers(&[J::Seltzer])).res.score, 48.0);
}

#[test]
fn seltzer_is_destroyed_after_ten_hands_at_game_level() {
    let mut g = new_game(8);
    // Many hands per round so that the round survives ten plays: use the state to refill.
    g.add_joker(JokerInstance::new(0, J::Seltzer).with_counter(2))
        .expect("room");
    select(&mut g);
    g.step(Action::Play(0b1)).expect("hand 1");
    assert_eq!(g.state().jokers.len(), 1);
    assert_eq!(g.state().jokers[0].counter, 1);
    g.step(Action::Play(0b1)).expect("hand 2");
    assert!(g.state().jokers.is_empty(), "drank after the last hand");
    assert!(g.state().check_invariants().is_ok());
    assert_eq!(g.state().stats.effects_no_room, 0);
}

#[test]
fn seltzer_eternal_is_not_destroyed_by_game_layer() {
    // Eternal is incompatible, but if a test forces it the game layer refuses to destroy it.
    let mut g = new_game(9);
    let mut j = JokerInstance::new(0, J::Seltzer).with_counter(1);
    j.stickers.eternal = true;
    g.add_joker(j).expect("room");
    select(&mut g);
    g.step(Action::Play(0b1)).expect("hand");
    assert_eq!(g.state().jokers.len(), 1);
}

// ----- Hallucination -----

#[test]
fn hallucination_creates_a_tarot_on_any_pack() {
    for pack in [
        PackKind::Arcana,
        PackKind::Celestial,
        PackKind::Spectral,
        PackKind::Standard,
        PackKind::Buffoon,
    ] {
        let mut js = jokers(&[J::Hallucination]);
        let fx = fire(&mut js, &run_pm(1000), 1, |inp| fire_pack_open(inp, pack));
        assert_eq!(fx, vec![tarot()], "{pack:?}");
    }
}

#[test]
fn hallucination_never_with_zero_probability() {
    let mut js = jokers(&[J::Hallucination]);
    for seed in 0..20 {
        let fx = fire(&mut js, &run_pm(0), seed, |inp| {
            fire_pack_open(inp, PackKind::Arcana)
        });
        assert!(fx.is_empty());
    }
}

#[test]
fn hallucination_is_one_in_two_and_deterministic() {
    let run = RunView::default();
    let roll = |seed| {
        let mut js = jokers(&[J::Hallucination]);
        let mut rng = GameRng::new(seed);
        let deck = Deck::standard();
        let mut fx = Vec::new();
        let mut hits = Vec::new();
        for _ in 0..200 {
            let before = fx.len();
            fire_pack_open(
                EventInput {
                    jokers: &mut js,
                    run: &run,
                    deck: &deck,
                    rng: &mut rng,
                    effects: &mut fx,
                },
                PackKind::Buffoon,
            );
            hits.push(fx.len() > before);
        }
        hits
    };
    let a = roll(42);
    assert_eq!(a, roll(42), "same seed, same rolls");
    assert_ne!(a, roll(43), "different seed, different rolls");
    let n = a.iter().filter(|&&h| h).count();
    assert!(
        (70..=130).contains(&n),
        "about half of 200 rolls hit, got {n}"
    );
}

#[test]
fn hallucination_with_oops_all_6s_always_triggers() {
    // 1 in 2 doubled = 2 in 2.
    let mut js = jokers(&[J::Hallucination, J::OopsAll6s]);
    let run = run_pm(collect_modifiers(&js).prob_mult);
    assert_eq!(run.prob_mult, 2);
    for seed in 0..30 {
        let fx = fire(&mut js, &run, seed, |inp| {
            fire_pack_open(inp, PackKind::Celestial)
        });
        assert_eq!(fx, vec![tarot()], "seed {seed}");
    }
}

#[test]
fn hallucination_blueprint_gives_a_second_roll() {
    let mut js = jokers(&[J::Blueprint, J::Hallucination]);
    let fx = fire(&mut js, &run_pm(1000), 1, |inp| {
        fire_pack_open(inp, PackKind::Arcana)
    });
    assert_eq!(fx, vec![tarot(), tarot()]);
}

// ----- Cartomancer -----

#[test]
fn cartomancer_creates_a_tarot_on_every_blind_kind() {
    for blind in [BlindKind::Small, BlindKind::Big, BlindKind::Boss] {
        let mut js = jokers(&[J::Cartomancer]);
        let run = RunView {
            blind,
            ..RunView::default()
        };
        let fx = fire(&mut js, &run, 1, fire_blind_select);
        assert_eq!(fx, vec![tarot()], "{blind:?}");
    }
}

#[test]
fn cartomancer_blueprint_and_debuff() {
    let mut js = jokers(&[J::Blueprint, J::Cartomancer]);
    let fx = fire(&mut js, &RunView::default(), 1, fire_blind_select);
    assert_eq!(fx, vec![tarot(), tarot()]);
    let mut js = jokers(&[J::Cartomancer]);
    js[0].debuffed = true;
    assert!(fire(&mut js, &RunView::default(), 1, fire_blind_select).is_empty());
}

#[test]
fn cartomancer_through_the_game_keeps_the_run_consistent() {
    // Consumables arrive in Phase 4: the Tarot request is currently dropped by the game layer
    // (no consumable system yet). The run must stay consistent either way.
    let mut g = new_game(10);
    add(&mut g, J::Cartomancer);
    let before = g.state().consumables.len();
    select(&mut g);
    let s = g.state();
    assert!(s.check_invariants().is_ok());
    assert!(s.consumables.len() <= before + 1);
}

// ----- Perkeo -----

#[test]
fn perkeo_copies_a_random_consumable_as_negative() {
    let held = [cons(1), cons(2)];
    let run = RunView {
        consumables: &held,
        ..RunView::default()
    };
    let mut seen = [false; 2];
    for seed in 0..40 {
        let mut js = jokers(&[J::Perkeo]);
        let fx = fire(&mut js, &run, seed, fire_shop_exit);
        assert_eq!(fx.len(), 1);
        match fx[0] {
            SideEffect::CopyConsumable { index, negative } => {
                assert!(negative);
                assert!(index < 2);
                seen[index] = true;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(seen, [true, true], "both consumables get picked");
}

#[test]
fn perkeo_is_deterministic_per_seed() {
    let held = [cons(1), cons(2), cons(3)];
    let run = RunView {
        consumables: &held,
        ..RunView::default()
    };
    let go = |seed| fire(&mut jokers(&[J::Perkeo]), &run, seed, fire_shop_exit);
    assert_eq!(go(5), go(5));
}

#[test]
fn perkeo_with_no_consumable_does_nothing_and_draws_nothing() {
    let mut js = jokers(&[J::Perkeo]);
    let deck = Deck::standard();
    let mut rng = GameRng::new(3);
    let reference = rng.clone();
    let mut fx = Vec::new();
    fire_shop_exit(EventInput {
        jokers: &mut js,
        run: &RunView::default(),
        deck: &deck,
        rng: &mut rng,
        effects: &mut fx,
    });
    assert!(fx.is_empty());
    assert_eq!(rng, reference, "no RNG draw without a target");
}

#[test]
fn perkeo_single_consumable_and_blueprint() {
    let held = [cons(7)];
    let run = RunView {
        consumables: &held,
        ..RunView::default()
    };
    let copy = SideEffect::CopyConsumable {
        index: 0,
        negative: true,
    };
    assert_eq!(
        fire(&mut jokers(&[J::Perkeo]), &run, 1, fire_shop_exit),
        vec![copy]
    );
    // Blueprint left of Perkeo triggers it a second time.
    assert_eq!(
        fire(
            &mut jokers(&[J::Blueprint, J::Perkeo]),
            &run,
            1,
            fire_shop_exit
        ),
        vec![copy, copy]
    );
}

#[test]
fn perkeo_through_the_game_keeps_the_run_consistent() {
    let mut g = new_game(11);
    add(&mut g, J::Perkeo);
    g.state_mut().consumables.push(cons(1));
    select(&mut g);
    beat_blind(&mut g);
    g.step(Action::EndShop).expect("end shop");
    let s = g.state();
    assert!(s.check_invariants().is_ok());
    assert!(s.consumables.len() <= s.consumable_capacity() as usize);
}

// ----- Diet Cola -----

const DOUBLE_TAG: SideEffect = SideEffect::CreateTag { id: TagId::Double };

#[test]
fn diet_cola_creates_a_double_tag_when_sold() {
    let mut js = jokers(&[J::DietCola]);
    let fx = fire(&mut js, &RunView::default(), 1, |inp| {
        fire_sell_self(inp, 0)
    });
    assert_eq!(fx, vec![DOUBLE_TAG]);
}

#[test]
fn diet_cola_does_not_trigger_when_another_joker_is_sold() {
    let mut js = jokers(&[J::Joker, J::DietCola]);
    let fx = fire(&mut js, &RunView::default(), 1, |inp| {
        fire_sell_self(inp, 0)
    });
    assert!(fx.is_empty());
}

#[test]
fn diet_cola_blueprint_sold_copies_the_tag() {
    // Selling a Blueprint that copies Diet Cola creates the tag (wiki), selling Diet Cola itself
    // once creates one.
    let mut js = jokers(&[J::Blueprint, J::DietCola]);
    assert_eq!(
        fire(&mut js, &RunView::default(), 1, |inp| fire_sell_self(
            inp, 0
        )),
        vec![DOUBLE_TAG]
    );
    assert_eq!(
        fire(&mut js, &RunView::default(), 1, |inp| fire_sell_self(
            inp, 1
        )),
        vec![DOUBLE_TAG]
    );
}

#[test]
fn diet_cola_sale_at_game_level() {
    let mut g = new_game(12);
    add(&mut g, J::DietCola);
    select(&mut g);
    let money = g.state().money;
    let tags = g.state().tags.len();
    let info = g.step(Action::SellJoker(0)).expect("sell");
    // Sell value of a cost-6 joker is $3; one Double Tag appears; the joker is gone.
    assert_eq!(info.sold_for, Some(3));
    assert_eq!(g.state().money, money + 3);
    assert_eq!(g.state().tags.len(), tags + 1);
    assert_eq!(g.state().tags.last(), Some(&TagId::Double));
    assert!(g.state().jokers.is_empty());
    assert!(g.state().check_invariants().is_ok());
}

// ----- Certificate -----

fn certificate_card(fx: &[SideEffect]) -> Card {
    match fx {
        [SideEffect::AddCard { card, to }] => {
            assert_eq!(*to, Pile::Hand);
            *card
        }
        other => panic!("expected exactly one AddCard, got {other:?}"),
    }
}

#[test]
fn certificate_adds_a_random_sealed_card_to_hand() {
    let mut js = jokers(&[J::Certificate]);
    let fx = fire(&mut js, &RunView::default(), 1, fire_first_hand_drawn);
    let card = certificate_card(&fx);
    assert_ne!(card.seal, Seal::None);
    assert_eq!(card.enhancement, Enhancement::None);
    assert_eq!(card.edition, Edition::Base);
}

#[test]
fn certificate_is_deterministic_and_covers_every_seal_rank_and_suit() {
    let go = |seed| {
        certificate_card(&fire(
            &mut jokers(&[J::Certificate]),
            &RunView::default(),
            seed,
            fire_first_hand_drawn,
        ))
    };
    assert_eq!(go(9), go(9));
    let mut seals = Vec::new();
    let mut ranks = Vec::new();
    let mut suits = Vec::new();
    for seed in 0..400 {
        let c = go(seed);
        if !seals.contains(&c.seal) {
            seals.push(c.seal);
        }
        if !ranks.contains(&c.rank) {
            ranks.push(c.rank);
        }
        if !suits.contains(&c.suit) {
            suits.push(c.suit);
        }
    }
    assert_eq!(seals.len(), 4, "Gold/Red/Blue/Purple, never None");
    assert!(!seals.contains(&Seal::None));
    assert_eq!(ranks.len(), 13);
    assert_eq!(suits.len(), 4);
}

#[test]
fn certificate_is_not_affected_by_probability_modifiers() {
    // No chance roll: prob_mult 0 still creates the card.
    let mut js = jokers(&[J::Certificate]);
    let fx = fire(&mut js, &run_pm(0), 1, fire_first_hand_drawn);
    certificate_card(&fx);
}

#[test]
fn certificate_blueprint_adds_two_cards() {
    let mut js = jokers(&[J::Blueprint, J::Certificate]);
    let fx = fire(&mut js, &RunView::default(), 1, fire_first_hand_drawn);
    assert_eq!(fx.len(), 2);
    assert!(
        fx.iter()
            .all(|e| matches!(e, SideEffect::AddCard { to: Pile::Hand, .. }))
    );
}

#[test]
fn certificate_debuffed_does_nothing() {
    let mut js = jokers(&[J::Certificate]);
    js[0].debuffed = true;
    assert!(fire(&mut js, &RunView::default(), 1, fire_first_hand_drawn).is_empty());
}

#[test]
fn certificate_at_game_level_adds_a_card_to_hand_and_deck() {
    for seed in 0..10 {
        let mut g = new_game(seed);
        add(&mut g, J::Certificate);
        select(&mut g);
        let s = g.state();
        // 8 drawn + 1 Certificate card (beyond the hand size); the deck grows to 53.
        assert_eq!(s.deck.hand.len(), 9, "seed {seed}");
        assert_eq!(s.deck.len(), 53);
        assert_eq!(s.stats.cards_added, 1);
        assert_eq!(s.stats.effects_no_room, 0);
        let sealed: Vec<&Card> = s
            .deck
            .hand
            .iter()
            .filter(|c| c.seal != Seal::None)
            .collect();
        assert_eq!(sealed.len(), 1);
        assert!(s.check_invariants().is_ok());
        // The card is real: it has a fresh uid (52) and the hand is still sorted/legal.
        assert_eq!(sealed[0].uid, 52);
        assert!(g.legal_actions().contains(&Action::Play(0b1_1111)));
    }
}

#[test]
fn certificate_card_stays_in_the_deck_for_later_rounds() {
    let mut g = new_game(21);
    add(&mut g, J::Certificate);
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().deck.len(), 53);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    // The second round adds another card: 54 cards, 2 sealed cards were created in total.
    let s = g.state();
    assert_eq!(s.deck.len(), 54);
    assert_eq!(s.stats.cards_added, 2);
    assert_eq!(s.deck.hand.len(), 9);
    assert!(s.check_invariants().is_ok());
}

#[test]
fn certificate_with_blueprint_at_game_level() {
    let mut g = new_game(22);
    add(&mut g, J::Blueprint);
    add(&mut g, J::Certificate);
    select(&mut g);
    let s = g.state();
    assert_eq!(s.deck.hand.len(), 10);
    assert_eq!(s.deck.len(), 54);
    assert!(s.check_invariants().is_ok());
}

#[test]
fn certificate_with_a_maximum_hand_does_not_break_legal_actions() {
    // Hand size 8 + 5 (Turtle Bean) + 1 (Juggler) + 2 (Troubadour) = 16 = the cap; Certificate
    // adds a 17th card. The game must stay consistent and playable.
    let mut g = new_game(23);
    for id in [J::TurtleBean, J::Juggler, J::Troubadour, J::Certificate] {
        add(&mut g, id);
    }
    select(&mut g);
    assert_eq!(g.state().deck.hand.len(), 17);
    assert!(g.state().check_invariants().is_ok());
    let acts = g.legal_actions();
    assert!(!acts.is_empty());
    let first_play = acts
        .iter()
        .copied()
        .find(|a| matches!(a, Action::Play(_)))
        .expect("a legal play");
    g.step(first_play).expect("play with a 17-card hand");
    assert!(g.state().check_invariants().is_ok());
}

// ----- whole package smoke / determinism -----

/// Deterministic pseudo-random action choice (no `rand`): xorshift on a seed.
fn pick(state: &mut u64, n: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state % n as u64) as usize
}

fn run_all_jokers(seed: u64) -> Game {
    let mut g = new_game(seed);
    for id in WAVE {
        g.add_joker(JokerInstance::new(0, id).with_edition(Edition::Negative))
            .expect("negative jokers always fit");
    }
    let mut r = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for _ in 0..400 {
        if g.is_terminal() {
            break;
        }
        let acts = g.legal_actions();
        let a = acts[pick(&mut r, acts.len())];
        g.step(a).expect("legal action");
    }
    g
}

#[test]
fn all_wave4b_jokers_together_are_stable_and_deterministic() {
    for seed in 0..12 {
        let a = run_all_jokers(seed);
        let b = run_all_jokers(seed);
        assert_eq!(a.state(), b.state(), "seed {seed}");
        assert!(a.state().check_invariants().is_ok(), "seed {seed}");
    }
}
