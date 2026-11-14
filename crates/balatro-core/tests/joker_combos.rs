//! Phase 3.R: hand-computed multi-joker scenarios that cross joker packages. Each test writes
//! the expected chips/mult step by step (pipeline order: docs/design/hooks.md §4,
//! docs/mechanics/scoring.md). Scorer-level tests evaluate the played cards with the real
//! `hand_eval::evaluate` under the jokers' rule flags; game-level tests go through `Game::step`.
//!
//! Base values used below (level 1): High Card 5 x 1, Pair 10 x 2, Straight 30 x 4,
//! Straight Flush 100 x 8. Card chips: 2-10 face value, J/Q/K 10, A 11.

use balatro_core::actions::Action;
use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::hand_eval::evaluate;
use balatro_core::hooks::{RunView, SideEffect, collect_modifiers};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreResult, Scorer, TraceOp};
use balatro_core::state::GameConfig;

use JokerId as J;
use Rank::*;
use Suit::*;

// ----- helpers -----------------------------------------------------------------------------

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

fn ed(mut card: Card, e: Edition) -> Card {
    card.edition = e;
    card
}

fn jk(id: JokerId) -> JokerInstance {
    JokerInstance::new(0, id)
}

/// Jokers with uids 100, 101, ... in the given order.
fn jokers(ids: &[JokerId]) -> Vec<JokerInstance> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| JokerInstance::new(100 + i as u32, id))
        .collect()
}

/// Re-number uids 100, 101, ... (for hand-built joker lists).
fn uids(mut js: Vec<JokerInstance>) -> Vec<JokerInstance> {
    for (i, j) in js.iter_mut().enumerate() {
        j.uid = 100 + i as u32;
    }
    js
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
    played: Vec<Card>,
    trace: Vec<balatro_core::scoring::TraceEvent>,
}

/// Score `played` (held: `held`) with the jokers' passive rules applied to both the hand
/// evaluation and the run view (`rules`, `prob_mult`), as the game layer does.
fn score_run(
    played: &[Card],
    held: &[Card],
    js: &mut [JokerInstance],
    base: RunView<'_>,
    seed: u64,
) -> Out {
    let mods = collect_modifiers(js);
    let run = RunView {
        rules: mods.rules,
        prob_mult: mods.prob_mult,
        ..base
    };
    let mut played = played.to_vec();
    let eval = evaluate(&played, &run.rules);
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::with_trace(true);
    let res = scorer.score_hand(balatro_core::scoring::ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: js,
        run: &run,
        rng: &mut rng,
    });
    Out {
        res,
        effects: scorer.effects().to_vec(),
        played,
        trace: scorer.trace().to_vec(),
    }
}

fn score(played: &[Card], held: &[Card], js: &mut [JokerInstance]) -> Out {
    score_run(played, held, js, RunView::default(), 1)
}

/// Pair of sevens: 10 + 7 + 7 = 24 chips x 2 mult before jokers.
fn pair7() -> [Card; 2] {
    [c(1, Seven, Spades), c(2, Seven, Hearts)]
}

/// Total triggers of played card `index` (1 + every `Retriggers` entry for it).
fn triggers_of(o: &Out, index: u8) -> u32 {
    1 + o
        .trace
        .iter()
        .filter_map(|e| match e.op {
            TraceOp::Retriggers {
                card,
                held: false,
                count,
            } if card == index => Some(count),
            _ => None,
        })
        .sum::<u32>()
}

// ----- game helpers ------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn add(g: &mut Game, j: JokerInstance) -> usize {
    g.add_joker(j).expect("room for joker")
}

/// Overwrite hand card `i` in place (uid kept, so the deck invariants hold).
fn set_hand(g: &mut Game, i: usize, card: Card) {
    let h = &mut g.state_mut().deck.hand[i];
    let uid = h.uid;
    *h = Card { uid, ..card };
}

fn play(g: &mut Game, mask: u16) -> StepInfo {
    let info = g.step(Action::Play(mask)).expect("play");
    g.state().check_invariants().expect("invariants");
    info
}

fn joker_value(g: &Game, id: JokerId) -> f64 {
    g.state()
        .jokers
        .iter()
        .find(|j| j.id == id)
        .map(|j| j.value)
        .expect("joker present")
}

// ===========================================================================================
// 1. Blueprint + xmult joker (Blueprint runs the copy at its own slot, with its own edition)
// ===========================================================================================

#[test]
fn blueprint_copies_an_xmult_joker_at_its_own_slot() {
    // [Joker, Blueprint -> Cavendish, Cavendish], Pair 7s: 24 x 2.
    //   Joker +4 -> 6; Blueprint (Cavendish) x3 -> 18; Cavendish x3 -> 54.  24 x 54 = 1296.
    let mut js = jokers(&[J::Joker, J::Blueprint, J::Cavendish]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (24.0, 54.0, 1296.0));

    // [Blueprint -> Cavendish, Cavendish, Joker]: x3 -> 6, x3 -> 18, +4 -> 22. 24 x 22 = 528.
    let mut js = jokers(&[J::Blueprint, J::Cavendish, J::Joker]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 528.0);

    // Blueprint with Polychrome: its OWN edition applies after the copied effect.
    //   Joker +4 -> 6; Blueprint x3 -> 18, Poly x1.5 -> 27; Cavendish x3 -> 81. 24 x 81 = 1944.
    let mut js = uids(vec![
        jk(J::Joker),
        jk(J::Blueprint).with_edition(Edition::Polychrome),
        jk(J::Cavendish),
    ]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.mult, o.res.score), (81.0, 1944.0));
}

// ===========================================================================================
// 2. Brainstorm + scaling joker (the copy reads the scaled value, never scales twice)
// ===========================================================================================

#[test]
fn brainstorm_copies_a_scaling_joker_without_double_counting() {
    // [Ride the Bus (+3 so far), Brainstorm -> slot 0], Pair 7s (no face card).
    //   before: Ride the Bus 3 -> 4 (owner); the copy's +1 goes to a scratch copy (discarded).
    //   main: Ride the Bus +4 -> 6; Brainstorm (reads 4) +4 -> 10.  24 x 10 = 240.
    let mut js = uids(vec![jk(J::RideTheBus).with_value(3.0), jk(J::Brainstorm)]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.mult, o.res.score), (10.0, 240.0));
    assert_eq!(js[0].value, 4.0, "scaled once");
    // Next hand: 4 -> 5, both pay 5: 2 + 10 = 12. 24 x 12 = 288.
    let o = score(&pair7(), &[], &mut js);
    assert_eq!(o.res.score, 288.0);
    assert_eq!(js[0].value, 5.0);

    // Lucky Cat scales per trigger in `on_card_scored`; Brainstorm's copy must not add more.
    // Four Oops! All 6s -> prob_mult 16 >= 15, so both Lucky rolls always hit.
    // [Lucky Cat, Brainstorm, Oops x4], Pair of Lucky 7s:
    //   7S: +7 chips, +20 mult -> 22; Cat 1.0 -> 1.25.  7H: +7, +20 -> 42; Cat -> 1.5.
    //   main: Cat x1.5 -> 63; Brainstorm (reads 1.5) x1.5 -> 94.5.  24 x 94.5 = 2268.
    let mut js = jokers(&[
        J::LuckyCat,
        J::Brainstorm,
        J::OopsAll6s,
        J::OopsAll6s,
        J::OopsAll6s,
        J::OopsAll6s,
    ]);
    let played = [
        enh(c(1, Seven, Spades), Enhancement::Lucky),
        enh(c(2, Seven, Hearts), Enhancement::Lucky),
    ];
    let o = score(&played, &[], &mut js);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (24.0, 94.5, 2268.0));
    assert_eq!(js[0].value, 1.5);
    assert_eq!(o.res.money_delta, 40, "two Lucky $20 hits");
}

// ===========================================================================================
// 3. Copy chains: Blueprint -> Blueprint -> X, Brainstorm -> Blueprint -> X, broken chains
// ===========================================================================================

#[test]
fn copier_chains_resolve_and_stop_at_incompatible_targets() {
    // [Blueprint, Blueprint, The Duo], Pair: x2 three times: 2 -> 16. 24 x 16 = 384.
    let mut js = jokers(&[J::Blueprint, J::Blueprint, J::TheDuo]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.mult, o.res.score), (16.0, 384.0));

    // [Blueprint -> The Duo, The Duo, Brainstorm -> Blueprint -> The Duo]: 2 -> 4 -> 8 -> 16.
    let mut js = jokers(&[J::Blueprint, J::TheDuo, J::Brainstorm]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 384.0);

    // Chain ending at Four Fingers (not copyable): both Blueprints give nothing.
    // [Blueprint, Blueprint, Four Fingers, Joker]: only Joker +4 -> 6. 24 x 6 = 144.
    let mut js = jokers(&[J::Blueprint, J::Blueprint, J::FourFingers, J::Joker]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 144.0);

    // A debuffed middle link breaks the chain for the first Blueprint only.
    // [Blueprint, Blueprint (debuffed), The Duo]: slot 0 -> nothing, slot 1 inert, Duo x2.
    //   2 -> 4. 24 x 4 = 96.
    let mut js = uids(vec![
        jk(J::Blueprint),
        jk(J::Blueprint).with_debuff(true),
        jk(J::TheDuo),
    ]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 96.0);

    // Brainstorm in the leftmost slot would copy itself: inert.
    // [Brainstorm, Joker]: slot 0 targets itself -> nothing; Joker +4. 24 x 6 = 144.
    let mut js = jokers(&[J::Brainstorm, J::Joker]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 144.0);
}

// ===========================================================================================
// 4. Baseball Card + several Uncommons (applied at each Uncommon's turn, wherever Baseball is)
// ===========================================================================================

#[test]
fn baseball_card_multiplies_at_each_uncommon_turn() {
    // [Joker (C), Acrobat (U), Baseball Card (R), Bootstraps (U), Hack (U)], $10, final hand.
    // Pair 7s: 24 x 2.
    //   Joker +4 -> 6 (Common: no Baseball).
    //   Acrobat x3 -> 18, Baseball x1.5 -> 27.
    //   Baseball itself (Rare): nothing.
    //   Bootstraps +2 * floor(10/5) = +4 -> 31, Baseball x1.5 -> 46.5.
    //   Hack (no own scoring effect, still Uncommon): Baseball x1.5 -> 69.75.
    //   24 x 69.75 = 1674.
    let ids = [
        J::Joker,
        J::Acrobat,
        J::BaseballCard,
        J::Bootstraps,
        J::Hack,
    ];
    let run = RunView {
        money: 10,
        hands_left: 0,
        ..RunView::default()
    };
    let o = score_run(&pair7(), &[], &mut jokers(&ids), run, 1);
    assert_eq!((o.res.mult, o.res.score), (69.75, 1674.0));

    // Baseball's own slot does not matter: leftmost or rightmost gives the same product
    // (each x1.5 lands right after its Uncommon, and no additive effect sits between).
    for ids in [
        [
            J::BaseballCard,
            J::Joker,
            J::Acrobat,
            J::Bootstraps,
            J::Hack,
        ],
        [
            J::Joker,
            J::Acrobat,
            J::Bootstraps,
            J::Hack,
            J::BaseballCard,
        ],
    ] {
        let o = score_run(&pair7(), &[], &mut jokers(&ids), run, 1);
        assert_eq!(o.res.score, 1674.0, "{ids:?}");
    }

    // ...but it does matter relative to additive jokers: an Uncommon's x1.5 lands before a
    // later +mult. [Acrobat, Baseball, Joker]: x3 -> 6, x1.5 -> 9, +4 -> 13. 24 x 13 = 312.
    let o = score_run(
        &pair7(),
        &[],
        &mut jokers(&[J::Acrobat, J::BaseballCard, J::Joker]),
        run,
        1,
    );
    assert_eq!(o.res.score, 312.0);

    // A debuffed Uncommon is still counted; Blueprint copying Baseball doubles the bonus.
    // [Blueprint -> Baseball, Baseball, Hack (debuffed)], Pair: 2 x1.5 x1.5 = 4.5. 24 x 4.5 = 108.
    let mut js = uids(vec![
        jk(J::Blueprint),
        jk(J::BaseballCard),
        jk(J::Hack).with_debuff(true),
    ]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.mult, o.res.score), (4.5, 108.0));
}

// ===========================================================================================
// 5. Mime + Baron + Steel + Red Seal (held-card retriggers)
// ===========================================================================================

#[test]
fn mime_baron_steel_and_red_seal_on_held_cards() {
    // Held: K♠ Steel + Red Seal, K♥, Q♣. Jokers [Baron, Mime]. Pair 7s: 24 x 2.
    //   K♠: 1 + Red Seal + Mime = 3 triggers, each Steel x1.5 then Baron x1.5 -> x2.25^3.
    //   K♥: 1 + Mime = 2 triggers, Baron x1.5 each -> x2.25.
    //   Q♣: 2 triggers, nothing.
    //   mult = 2 * 2.25^3 * 2.25 = 2 * 11.390625 * 2.25 = 51.2578125.
    //   24 x 51.2578125 = 1230.1875 -> 1230.
    let held = [
        seal(enh(c(10, King, Spades), Enhancement::Steel), Seal::Red),
        c(11, King, Hearts),
        c(12, Queen, Clubs),
    ];
    let mut js = jokers(&[J::Baron, J::Mime]);
    let o = score(&pair7(), &held, &mut js);
    assert_eq!((o.res.mult, o.res.score), (51.2578125, 1230.0));

    // Blueprint copying Mime adds one more held retrigger per card:
    //   K♠ 4 triggers -> 2.25^4 = 25.62890625; K♥ 3 -> 1.5^3 = 3.375.
    //   2 * 25.62890625 * 3.375 = 172.99511719 (exact: 172.9951171875).
    //   24 x 172.9951171875 = 4151.8828125 -> 4151.
    let mut js = jokers(&[J::Baron, J::Blueprint, J::Mime]);
    let o = score(&pair7(), &held, &mut js);
    assert_eq!((o.res.mult, o.res.score), (172.9951171875, 4151.0));

    // Steel Joker reads the full deck, not the held cards: 1 Steel card -> x1.2 at its turn.
    // [Baron, Mime, Steel Joker] with deck counts holding that one Steel card:
    //   51.2578125 x 1.2 = 61.509375. 24 x 61.509375 = 1476.225 -> 1476.
    let mut deck = balatro_core::hooks::DeckCounts::default();
    deck.add(&held[0]);
    let run = RunView {
        deck,
        ..RunView::default()
    };
    let mut js = jokers(&[J::Baron, J::Mime, J::SteelJoker]);
    let o = score_run(&pair7(), &held, &mut js, run, 1);
    assert_eq!(o.res.score, 1476.0);
}

// ===========================================================================================
// 6. Retrigger chain on one card: Red Seal + Hack + Hanging Chad + Sock and Buskin + Seltzer
//    + Dusk (Pareidolia makes the 2 a face card for Sock and Buskin)
// ===========================================================================================

#[test]
fn retrigger_chain_sums_every_source() {
    // Seven jokers (two Negative, so the board is legal). Pair 2♠ (Red Seal) 2♥, final hand.
    //   2♠ (first scoring card): 1 + Red Seal 1 + Hack 1 + Hanging Chad 2 + Sock and Buskin 1
    //        + Seltzer 1 + Dusk 1 = 8 triggers.
    //   2♥: 1 + Hack 1 + S&B 1 + Seltzer 1 + Dusk 1 = 5 triggers.
    //   chips = 10 + 8 x 2 + 5 x 2 = 36, mult 2 -> 72.
    let ids = [
        J::Pareidolia,
        J::Hack,
        J::HangingChad,
        J::SockAndBuskin,
        J::Seltzer,
        J::Dusk,
    ];
    let mut js: Vec<JokerInstance> = jokers(&ids);
    js[0].edition = Edition::Negative;
    js[5].edition = Edition::Negative;
    let played = [seal(c(1, Two, Spades), Seal::Red), c(2, Two, Hearts)];
    let run = RunView {
        hands_left: 0,
        ..RunView::default()
    };
    let o = score_run(&played, &[], &mut js, run, 1);
    assert_eq!(triggers_of(&o, 0), 8);
    assert_eq!(triggers_of(&o, 1), 5);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (36.0, 2.0, 72.0));
    assert_eq!(js[4].counter, 9, "Seltzer counted down once");

    // Not the final hand: Dusk drops out (7 and 4 triggers). chips 10 + 14 + 8 = 32 -> 64.
    let run = RunView {
        hands_left: 2,
        ..RunView::default()
    };
    let o = score_run(&played, &[], &mut jokers(&ids), run, 1);
    assert_eq!((triggers_of(&o, 0), triggers_of(&o, 1)), (7, 4));
    assert_eq!(o.res.score, 64.0);

    // Per-trigger joker effects repeat on every trigger: add Fibonacci (+8 mult per 2).
    //   12 + 1... : 13 triggers x +8 = +104 mult -> 106. 36 x 106 = 3816 (final hand).
    let mut ids7 = ids.to_vec();
    ids7.push(J::Fibonacci);
    let run = RunView {
        hands_left: 0,
        ..RunView::default()
    };
    let o = score_run(&played, &[], &mut jokers(&ids7), run, 1);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score),
        (36.0, 106.0, 3816.0)
    );
}

// ===========================================================================================
// 7. Edition order: card editions per trigger; joker Foil/Holo -> effect -> Polychrome
// ===========================================================================================

#[test]
fn edition_order_on_cards_and_jokers() {
    // Played 7♠ Foil, 7♥ Polychrome (Pair 10 x 2):
    //   7♠: +7 -> 17, Foil +50 -> 67.   7♥: +7 -> 74, Poly x1.5 -> mult 3.
    // Jokers [Cavendish Holo, Joker Poly, Joker Foil, Joker Holo (debuffed)]:
    //   Cavendish: Holo +10 -> 13, x3 -> 39.
    //   Joker:     +4 -> 43, Poly x1.5 -> 64.5   (Polychrome AFTER the joker's own effect).
    //   Joker:     Foil +50 chips -> 124, +4 -> 68.5.
    //   debuffed:  nothing (no edition either).
    //   124 x 68.5 = 8494.
    let played = [
        ed(c(1, Seven, Spades), Edition::Foil),
        ed(c(2, Seven, Hearts), Edition::Polychrome),
    ];
    let mut js = uids(vec![
        jk(J::Cavendish).with_edition(Edition::Holographic),
        jk(J::Joker).with_edition(Edition::Polychrome),
        jk(J::Joker).with_edition(Edition::Foil),
        jk(J::Joker)
            .with_edition(Edition::Holographic)
            .with_debuff(true),
    ]);
    let o = score(&played, &[], &mut js);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score),
        (124.0, 68.5, 8494.0)
    );

    // Card editions repeat on retriggers: Red Seal on the Polychrome 7♥ -> x1.5 twice.
    //   chips 10 + 7 + 50 + 7 + 7 = 81; mult 2 x 1.5 x 1.5 = 4.5. No jokers: 81 x 4.5 = 364.5.
    let played = [
        ed(c(1, Seven, Spades), Edition::Foil),
        seal(ed(c(2, Seven, Hearts), Edition::Polychrome), Seal::Red),
    ];
    let o = score(&played, &[], &mut []);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (81.0, 4.5, 364.0));
}

// ===========================================================================================
// 8. Pareidolia + face jokers (Scary Face, Smiley, Photograph, Sock and Buskin, Midas Mask)
// ===========================================================================================

#[test]
fn pareidolia_turns_every_card_into_a_face_for_all_face_jokers() {
    // [Pareidolia, Midas Mask, Scary Face, Smiley Face, Photograph, Sock and Buskin]
    // Pair 7♠ 7♥ (both faces under Pareidolia). Midas turns both Gold before scoring (Gold
    // does nothing when scored). S&B: 2 triggers each. Photograph: only the first face (7♠).
    //   7♠ t1: +7 +30 -> 47; +5 -> 7; x2 -> 14.
    //   7♠ t2: +7 +30 -> 84; +5 -> 19; x2 -> 38.
    //   7♥ t1: +7 +30 -> 121; +5 -> 43.
    //   7♥ t2: +7 +30 -> 158; +5 -> 48.
    //   158 x 48 = 7584.
    let mut js = jokers(&[
        J::Pareidolia,
        J::MidasMask,
        J::ScaryFace,
        J::SmileyFace,
        J::Photograph,
        J::SockAndBuskin,
    ]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score),
        (158.0, 48.0, 7584.0)
    );
    assert!(
        o.played.iter().all(|c| c.enhancement == Enhancement::Gold),
        "Midas Mask gilds every 'face' card"
    );

    // Without Pareidolia nothing is a face card: 24 x 2 = 48, cards untouched.
    let mut js = jokers(&[
        J::MidasMask,
        J::ScaryFace,
        J::SmileyFace,
        J::Photograph,
        J::SockAndBuskin,
    ]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!(o.res.score, 48.0);
    assert!(o.played.iter().all(|c| c.enhancement == Enhancement::None));

    // Real faces, Photograph picks the first SCORING face: High Card... use Pair K♠ K♥ with a
    // non-scoring Q♣ kicker on the left. [Photograph, Sock and Buskin]:
    //   K♠ t1: +10 -> 20, x2 -> 4;  t2: +10 -> 30, x2 -> 8.  K♥ t1/t2: +10 +10 -> 50.
    //   50 x 8 = 400 (Q♣ does not score, so it is not "first").
    let played = [c(3, Queen, Clubs), c(1, King, Spades), c(2, King, Hearts)];
    let mut js = jokers(&[J::Photograph, J::SockAndBuskin]);
    let o = score(&played, &[], &mut js);
    assert_eq!((o.res.chips, o.res.mult, o.res.score), (50.0, 8.0, 400.0));
}

// ===========================================================================================
// 9. Four Fingers + Shortcut + Splash together
// ===========================================================================================

#[test]
fn four_fingers_shortcut_and_splash_together() {
    // 10♥ 8♥ 6♥ 4♥ K♠: four hearts (Four Fingers flush) that skip one rank each
    // (Shortcut straight of four) -> Straight Flush; Splash makes the K♠ score too.
    // [Four Fingers, Shortcut, Splash, Crazy Joker, Droll Joker]:
    //   Straight Flush 100 x 8. Cards: 10 + 8 + 6 + 4 + 10 = 38 -> 138 chips.
    //   Crazy (contains Straight) +12 -> 20; Droll (contains Flush) +10 -> 30.
    //   138 x 30 = 4140.
    let played = [
        c(1, Ten, Hearts),
        c(2, Eight, Hearts),
        c(3, Six, Hearts),
        c(4, Four, Hearts),
        c(5, King, Spades),
    ];
    let ids = [
        J::FourFingers,
        J::Shortcut,
        J::Splash,
        J::CrazyJoker,
        J::DrollJoker,
    ];
    let o = score(&played, &[], &mut jokers(&ids));
    assert_eq!(o.res.hand_type, HandType::StraightFlush);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score),
        (138.0, 30.0, 4140.0)
    );

    // Without Splash the K♠ does not score: 128 x 30 = 3840.
    let ids = [J::FourFingers, J::Shortcut, J::CrazyJoker, J::DrollJoker];
    let o = score(&played, &[], &mut jokers(&ids));
    assert_eq!(o.res.hand_type, HandType::StraightFlush);
    assert_eq!(o.res.score, 3840.0);

    // Without Shortcut it is only a (four-card) Flush: 35 x 4, cards 28 (hearts only),
    // Droll +10 -> 14; Crazy nothing. 63 x 14 = 882.
    let ids = [J::FourFingers, J::CrazyJoker, J::DrollJoker];
    let o = score(&played, &[], &mut jokers(&ids));
    assert_eq!(o.res.hand_type, HandType::Flush);
    assert_eq!(o.res.score, 882.0);

    // Without Four Fingers nothing of size four counts: High Card K♠ (5 + 10) x 1 = 15.
    let ids = [J::Shortcut, J::Splash, J::CrazyJoker, J::DrollJoker];
    let o = score(&played, &[], &mut jokers(&ids));
    assert_eq!(o.res.hand_type, HandType::HighCard);
    // Splash: all five score: 5 + 10 + 8 + 6 + 4 + 10 = 43 x 1 = 43.
    assert_eq!(o.res.score, 43.0);
}

// ===========================================================================================
// 10. Vampire and Midas Mask: `before_scoring` runs left to right
// ===========================================================================================

#[test]
fn vampire_and_midas_mask_resolve_left_to_right() {
    // Pair K♠ (Mult card) K♥: 10 + 10 + 10 = 30 chips.
    let played = [
        enh(c(1, King, Spades), Enhancement::Mult),
        c(2, King, Hearts),
    ];
    // [Midas Mask, Vampire]: Midas gilds both (Mult overwritten), Vampire eats two Gold
    //   enhancements: 1.0 + 0.2 = 1.2. Cards score plain: mult 2 x 1.2 = 2.4. 30 x 2.4 = 72.
    let mut js = jokers(&[J::MidasMask, J::Vampire]);
    let o = score(&played, &[], &mut js);
    assert!((js[1].value - 1.2).abs() < 1e-12);
    assert!(o.played.iter().all(|c| c.enhancement == Enhancement::None));
    assert_eq!(o.res.score, (30.0 * (2.0 * js[1].value)).floor());
    assert!((o.res.mult - 2.4).abs() < 1e-12);

    // [Vampire, Midas Mask]: Vampire eats the Mult card first (1.1), then Midas gilds both.
    //   The K♠ lost its +4 before scoring, Gold scores nothing: mult 2 x 1.1 = 2.2.
    //   30 x 2.2 = 66.
    let mut js = jokers(&[J::Vampire, J::MidasMask]);
    let o = score(&played, &[], &mut js);
    assert!((js[0].value - 1.1).abs() < 1e-12);
    assert!(o.played.iter().all(|c| c.enhancement == Enhancement::Gold));
    assert_eq!(o.res.score, 66.0);

    // No Vampire: the Mult card would have scored +4, unless Midas overwrote it first.
    //   [Midas Mask] alone: both Gold, mult 2 -> 60.
    let o = score(&played, &[], &mut jokers(&[J::MidasMask]));
    assert_eq!(o.res.score, 60.0);
}

// ===========================================================================================
// 11. Space Joker: the level-up applies to the hand being scored
// ===========================================================================================

#[test]
fn space_joker_level_up_counts_for_the_current_hand() {
    // Two Oops! All 6s: prob_mult 4 -> 1 in 4 becomes certain.
    // [Oops, Oops, Space Joker], Pair 7s: Pair level 1 -> 2 = 25 x 3. 25 + 14 = 39 x 3 = 117.
    let mut js = jokers(&[J::OopsAll6s, J::OopsAll6s, J::SpaceJoker]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.hand_level, o.res.score), (2, 117.0));
    assert_eq!(
        o.effects,
        vec![SideEffect::LevelUpHand {
            hand: HandType::Pair,
            levels: 1
        }]
    );

    // Blueprint copying Space: two level-ups -> level 3 = 40 x 4. 54 x 4 = 216.
    let mut js = jokers(&[J::OopsAll6s, J::OopsAll6s, J::Blueprint, J::SpaceJoker]);
    let o = score(&pair7(), &[], &mut js);
    assert_eq!((o.res.hand_level, o.res.score), (3, 216.0));

    // End to end: the level persists in the run.
    let mut g = new_game(5);
    for id in [J::OopsAll6s, J::OopsAll6s, J::SpaceJoker] {
        add(&mut g, jk(id));
    }
    g.step(Action::SelectBlind).expect("select");
    set_hand(&mut g, 0, c(0, Seven, Spades));
    set_hand(&mut g, 1, c(0, Seven, Hearts));
    let info = play(&mut g, 0b11);
    let h = info.hand.expect("hand report");
    assert_eq!((h.hand_type, h.level, h.score), (HandType::Pair, 2, 117.0));
    assert_eq!(g.state().hand_levels[HandType::Pair.index()], 2);
}

// ===========================================================================================
// 12. Glass break + Glass Joker + Canio in the same hand (game level)
// ===========================================================================================

#[test]
fn glass_break_feeds_glass_joker_and_canio_after_the_hand() {
    // Two Oops! All 6s make the 1 in 4 Glass break certain.
    // Hand: K♠ Glass, K♥ Glass (Pair 10 x 2): chips 10 + 10 + 10 = 30, mult 2 x2 x2 = 8.
    //   Glass Joker / Canio start at x1: nothing this hand. 30 x 8 = 240.
    // After: both Kings break -> Glass Joker 1 + 2 x 0.75 = 2.5, Canio 1 + 2 x 1 = 3.
    let mut g = new_game(11);
    for id in [J::OopsAll6s, J::OopsAll6s, J::GlassJoker, J::Canio] {
        add(&mut g, jk(id));
    }
    g.step(Action::SelectBlind).expect("select");
    set_hand(&mut g, 0, enh(c(0, King, Spades), Enhancement::Glass));
    set_hand(&mut g, 1, enh(c(0, King, Hearts), Enhancement::Glass));
    let glass_uids = [g.state().deck.hand[0].uid, g.state().deck.hand[1].uid];
    let info = play(&mut g, 0b11);
    assert_eq!(info.hand.expect("hand").score, 240.0);
    assert_eq!(info.cards_destroyed, 2);
    assert!(
        g.state()
            .deck
            .iter_all()
            .all(|c| !glass_uids.contains(&c.uid))
    );
    assert_eq!(g.state().deck.len(), 50);
    assert_eq!(joker_value(&g, J::GlassJoker), 2.5);
    assert_eq!(joker_value(&g, J::Canio), 3.0);

    // Next hand pays both: Pair 7s 24 x 2, Glass Joker x2.5 -> 5, Canio x3 -> 15. 24 x 15 = 360.
    assert_eq!(
        g.state().phase,
        balatro_core::enums::Phase::Play,
        "240 < 300"
    );
    set_hand(&mut g, 0, c(0, Seven, Spades));
    set_hand(&mut g, 1, c(0, Seven, Hearts));
    let info = play(&mut g, 0b11);
    assert_eq!(info.hand.expect("hand").score, 360.0);
}

// ===========================================================================================
// 13. Lucky cards + Lucky Cat + Oops! All 6s (+ Red Seal retrigger rolls again)
// ===========================================================================================

#[test]
fn lucky_cards_lucky_cat_and_oops() {
    // Four Oops! -> prob_mult 16: Lucky's 1/5 mult and 1/15 money both always hit.
    // [Oops x4, Lucky Cat], Pair 7♠ Lucky + Red Seal, 7♥ Lucky:
    //   7♠ t1: +7 -> 17, +20 -> 22, $20, Cat 1.25.  t2: +7 -> 24, +20 -> 42, $20, Cat 1.5.
    //   7♥:    +7 -> 31, +20 -> 62, $20, Cat 1.75.
    //   Cat x1.75 -> 108.5. 31 x 108.5 = 3363.5 -> 3363. Money +$60.
    let mut js = jokers(&[
        J::OopsAll6s,
        J::OopsAll6s,
        J::OopsAll6s,
        J::OopsAll6s,
        J::LuckyCat,
    ]);
    let played = [
        seal(enh(c(1, Seven, Spades), Enhancement::Lucky), Seal::Red),
        enh(c(2, Seven, Hearts), Enhancement::Lucky),
    ];
    let o = score(&played, &[], &mut js);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score, o.res.money_delta),
        (31.0, 108.5, 3363.0, 60)
    );
    assert_eq!(js[4].value, 1.75);

    // Without Oops nothing is forced; with prob_mult 0 (a run view override) nothing hits and
    // Lucky Cat does not grow: 31 x 2 = 62.
    let mut js = jokers(&[J::LuckyCat]);
    let mut played2 = played.to_vec();
    let eval = evaluate(&played2, &Default::default());
    let run = RunView {
        prob_mult: 0,
        ..RunView::default()
    };
    let mut rng = GameRng::new(3);
    let r = Scorer::new().score_hand(balatro_core::scoring::ScoreInput {
        played: &mut played2,
        held: &[],
        eval,
        jokers: &mut js,
        run: &run,
        rng: &mut rng,
    });
    assert_eq!((r.score, r.money_delta), (62.0, 0));
    assert_eq!(js[0].value, 1.0);
}

// ===========================================================================================
// 14. Gros Michel / Cavendish and xmult ordering
// ===========================================================================================

#[test]
fn bananas_and_xmult_order() {
    // [Gros Michel, Cavendish]: +15 -> 17, x3 -> 51. 24 x 51 = 1224.
    let o = score(&pair7(), &[], &mut jokers(&[J::GrosMichel, J::Cavendish]));
    assert_eq!(o.res.score, 1224.0);
    // [Cavendish, Gros Michel]: x3 -> 6, +15 -> 21. 24 x 21 = 504.
    let o = score(&pair7(), &[], &mut jokers(&[J::Cavendish, J::GrosMichel]));
    assert_eq!(o.res.score, 504.0);
    // [Gros Michel, Blueprint -> Cavendish, Cavendish (Polychrome)]:
    //   +15 -> 17, x3 -> 51, x3 -> 153, Poly x1.5 -> 229.5. 24 x 229.5 = 5508.
    let mut js = uids(vec![
        jk(J::GrosMichel),
        jk(J::Blueprint),
        jk(J::Cavendish).with_edition(Edition::Polychrome),
    ]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 5508.0);
}

// ===========================================================================================
// 15. Joker Stencil + empty slots + Negative jokers (game level: slots come from the state)
// ===========================================================================================

#[test]
fn stencil_counts_empty_slots_with_negatives() {
    // Pair 7s preview: 24 x 2 base.
    let preview = |js: &[JokerInstance]| {
        let mut g = new_game(3);
        for &j in js {
            add(&mut g, j);
        }
        g.step(Action::SelectBlind).expect("select");
        set_hand(&mut g, 0, c(0, Seven, Spades));
        set_hand(&mut g, 1, c(0, Seven, Hearts));
        g.preview_play(0b11).expect("preview").score
    };
    // [Stencil, Joker]: 5 slots, 2 jokers, 1 Stencil -> 4 empty -> x4 -> 8, +4 -> 12. 288.
    assert_eq!(preview(&[jk(J::JokerStencil), jk(J::Joker)]), 288.0);
    // + a Negative Joker: 6 slots, 3 jokers -> still 4 empty. x4 -> 8, +4, +4 -> 16. 384.
    assert_eq!(
        preview(&[
            jk(J::JokerStencil),
            jk(J::Joker),
            jk(J::Joker).with_edition(Edition::Negative),
        ]),
        384.0
    );
    // Negative Stencil: 6 slots, 2 jokers, 1 Stencil -> 5 empty -> x5 -> 10, +4 -> 14. 336.
    assert_eq!(
        preview(&[
            jk(J::JokerStencil).with_edition(Edition::Negative),
            jk(J::Joker),
        ]),
        336.0
    );
    // Two Stencils: 5 slots, 2 jokers, 2 Stencils -> 5 empty each -> x25 -> 50. 1200.
    assert_eq!(preview(&[jk(J::JokerStencil), jk(J::JokerStencil)]), 1200.0);
    // Full board: [Stencil, Joker x4]: 5 - 5 + 1 = 1 -> x1 (no-op), +16 -> 18. 432.
    assert_eq!(
        preview(&[
            jk(J::JokerStencil),
            jk(J::Joker),
            jk(J::Joker),
            jk(J::Joker),
            jk(J::Joker),
        ]),
        432.0
    );
    // Blueprint copying Stencil reads the same board: [Blueprint, Stencil]: 5 - 2 + 1 = 4 each
    //   -> x4 x4 = 32. 24 x 32 = 768.
    assert_eq!(preview(&[jk(J::Blueprint), jk(J::JokerStencil)]), 768.0);
}

// ===========================================================================================
// 16. Money jokers (Bull, Bootstraps) see money earned earlier in the same hand
// ===========================================================================================

#[test]
fn money_jokers_see_gold_seal_and_golden_ticket_money() {
    // $3 before the hand. Pair 7♠ (Gold Seal), 7♥ (Gold card + Gold Seal).
    // [Golden Ticket, Bull, Bootstraps]:
    //   7♠: +7 -> 17, Gold Seal $3 (money 6).
    //   7♥: +7 -> 24, Gold Seal $3 (9), Golden Ticket (Gold card scored) $4 (13).
    //   Bull +2 x 13 = +26 -> 50 chips. Bootstraps +2 x floor(13/5) = +4 -> 6.
    //   50 x 6 = 300; money +$10.
    let played = [
        seal(c(1, Seven, Spades), Seal::Gold),
        seal(enh(c(2, Seven, Hearts), Enhancement::Gold), Seal::Gold),
    ];
    let run = RunView {
        money: 3,
        ..RunView::default()
    };
    let mut js = jokers(&[J::GoldenTicket, J::Bull, J::Bootstraps]);
    let o = score_run(&played, &[], &mut js, run, 1);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score, o.res.money_delta),
        (50.0, 6.0, 300.0, 10)
    );

    // Retriggers repeat the seal money before the money jokers read it. Hanging Chad gives
    // the FIRST scoring card (7♠) 3 triggers; [Bull, Bootstraps, Golden Ticket, Hanging Chad]:
    //   7♠ x3: $9, chips +21; 7♥: $3 + $4. money 3 + 16 = 19, chips 10 + 21 + 7 = 38.
    //   Bull +38 -> 76; Bootstraps +2 x 3 = +6 -> 8. 76 x 8 = 608; money +$16.
    let mut js = jokers(&[J::Bull, J::Bootstraps, J::GoldenTicket, J::HangingChad]);
    let o = score_run(&played, &[], &mut js, run, 1);
    assert_eq!(
        (o.res.chips, o.res.mult, o.res.score, o.res.money_delta),
        (76.0, 8.0, 608.0, 16)
    );

    // End to end: the hand's money reaches the run.
    let mut g = new_game(9);
    for id in [J::GoldenTicket, J::Bull, J::Bootstraps] {
        add(&mut g, jk(id));
    }
    g.step(Action::SelectBlind).expect("select");
    g.state_mut().money = 3;
    g.state_mut().target = 1e9; // keep the round going: money_after is the hand's money only
    set_hand(&mut g, 0, played[0]);
    set_hand(&mut g, 1, played[1]);
    let info = play(&mut g, 0b11);
    assert_eq!(info.hand.expect("hand").score, 300.0);
    assert_eq!(info.money_after, 13);
}

// ===========================================================================================
// 17. Hiker + retriggers + Blueprint: perma chips land before the next trigger
// ===========================================================================================

#[test]
fn hiker_with_retriggers_and_blueprint() {
    // [Blueprint -> Hiker, Hiker, Hanging Chad], High Card A♠ (5 x 1):
    //   3 triggers. Each: rank 11 + perma bonus so far; then Blueprint-Hiker +5, Hiker +5.
    //   t1: 11 + 0 = 11; perma 10.  t2: 11 + 10 = 21; perma 20.  t3: 11 + 20 = 31; perma 30.
    //   chips 5 + 11 + 21 + 31 = 68, mult 1 -> 68. The card keeps +30.
    let played = [c(1, Ace, Spades)];
    let mut js = jokers(&[J::Blueprint, J::Hiker, J::HangingChad]);
    let o = score(&played, &[], &mut js);
    assert_eq!((o.res.chips, o.res.score), (68.0, 68.0));
    assert_eq!(o.played[0].perma_bonus, 30);
}

// ===========================================================================================
// 18. Swashbuckler / Abstract count Blueprint as another joker
// ===========================================================================================

#[test]
fn swashbuckler_and_abstract_with_copies() {
    // Sell values: Swashbuckler (cost 4) $2, Blueprint (cost 10) $5, Joker (cost 2) $1.
    // [Swashbuckler, Blueprint -> Joker, Joker], Pair 7s 24 x 2:
    //   Swashbuckler: others = 5 + 1 = +6 -> 8. Blueprint (Joker) +4 -> 12. Joker +4 -> 16.
    //   24 x 16 = 384.
    let mut js = jokers(&[J::Swashbuckler, J::Blueprint, J::Joker]);
    assert_eq!(js[1].sell_value, 5);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 384.0);

    // [Blueprint -> Swashbuckler, Swashbuckler, Abstract]: "self" is Swashbuckler's slot, so
    // both exclude slot 1: others = Blueprint 5 + Abstract (cost 4) 2 = 7.
    //   +7 -> 9, +7 -> 16, Abstract +3 x 3 jokers = +9 -> 25. 24 x 25 = 600.
    let mut js = jokers(&[J::Blueprint, J::Swashbuckler, J::AbstractJoker]);
    assert_eq!(score(&pair7(), &[], &mut js).res.score, 600.0);
}

// ===========================================================================================
// 19. Marble Joker's Stone card is shuffled into the draw pile, not put on top (3.R fix)
// ===========================================================================================

#[test]
fn marble_stone_card_lands_at_a_random_deck_position() {
    // Blind select: reset + shuffle, then Marble adds a Stone card, then 8 cards are drawn.
    // Top-of-pile insertion made the Stone card always the first card drawn; a uniform
    // position over 53 slots puts it in the first hand with probability 8/53 (~15%).
    let mut in_first_hand = 0;
    let mut positions = std::collections::BTreeSet::new();
    const GAMES: u64 = 400;
    for seed in 0..GAMES {
        let mut g = new_game(seed);
        add(&mut g, jk(J::MarbleJoker));
        let info = g.step(Action::SelectBlind).expect("select");
        assert_eq!(info.cards_added, 1);
        let s = g.state();
        s.check_invariants().expect("invariants");
        assert_eq!(s.deck.iter_all().filter(|c| c.is_stone()).count(), 1);
        if s.deck.hand.iter().any(|c| c.is_stone()) {
            in_first_hand += 1;
        } else {
            let pos = s
                .deck
                .draw_pile
                .iter()
                .position(|c| c.is_stone())
                .expect("stone card in the draw pile");
            positions.insert(pos);
        }
    }
    // Expected ~60 of 400; far from "always" (400) and from "never".
    assert!(
        (25..=110).contains(&in_first_hand),
        "Stone card drawn first in {in_first_hand}/{GAMES} games"
    );
    assert!(positions.len() > 20, "positions look fixed: {positions:?}");

    // Deterministic per seed.
    let run = |seed| {
        let mut g = new_game(seed);
        add(&mut g, jk(J::MarbleJoker));
        g.step(Action::SelectBlind).expect("select");
        g.into_state()
    };
    assert_eq!(run(42), run(42));
}

// ===========================================================================================
// 20. Discard jokers on one discard: debuffed cards activate nothing card-specific (3.R fix
//     for Mail-In Rebate and Hit the Road), hand-level jokers (Burnt) still work
// ===========================================================================================

#[test]
fn discard_jokers_ignore_debuffed_cards() {
    use balatro_core::deck::Deck;
    use balatro_core::hooks::{DiscardInfo, EventInput, fire_discard};
    // balatrowiki.org/w/Debuffed: jokers that trigger on a discarded card do not activate from
    // a debuffed card; Burnt Joker (hand level) still does. Hit the Road: The Plant (debuffs
    // faces) "renders Hit the Road useless"; Mail-In Rebate: "debuffed cards do not have a rank".
    let mut jd = c(3, Jack, Diamonds);
    jd.debuffed = true;
    let cards = [c(1, Jack, Spades), c(2, Jack, Hearts), jd];
    // [Mail-In (target Jack), Hit the Road, Castle (target Hearts), Burnt Joker, Faceless].
    let mut js = jokers(&[
        J::MailInRebate,
        J::HitTheRoad,
        J::Castle,
        J::BurntJoker,
        J::FacelessJoker,
    ]);
    js[0].set_aux_rank(Some(Jack));
    js[2].set_aux_suit(Some(Hearts));
    let run = RunView {
        discards_used_round: 1,
        ..RunView::default()
    };
    let deck = Deck::standard();
    let mut rng = GameRng::new(1);
    let mut fx = Vec::new();
    let info = DiscardInfo {
        cards: &cards,
        hand_type: HandType::ThreeOfAKind,
        forced: false,
    };
    let r = fire_discard(
        EventInput {
            jokers: &mut js,
            run: &run,
            deck: &deck,
            rng: &mut rng,
            effects: &mut fx,
        },
        &info,
    );
    // Mail-In: 2 live Jacks x $5 = $10. Faceless: only 2 live faces (< 3) -> $0.
    assert_eq!(r.money_delta, 10);
    // Hit the Road: 1 + 2 x 0.5 = 2.0 (not 2.5).
    assert_eq!(js[1].value, 2.0);
    // Castle: one live Heart -> +3 chips.
    assert_eq!(js[2].value, 3.0);
    // Burnt Joker levels the discarded hand type regardless.
    assert_eq!(
        fx,
        vec![SideEffect::LevelUpHand {
            hand: HandType::ThreeOfAKind,
            levels: 1
        }]
    );
}

// ===========================================================================================
// 21. DNA + Sixth Sense + Hologram on one first-hand single 6 (game level)
// ===========================================================================================

#[test]
fn dna_and_sixth_sense_on_the_same_single_six() {
    // First hand of the round, one 6♠ played. [DNA, Sixth Sense, Hologram]:
    //   before: DNA queues a copy of the 6♠ into the hand.
    //   score: High Card 5 + 6 = 11 x 1 (Hologram still x1) = 11.
    //   after: Sixth Sense destroys the played 6♠ and queues a Spectral (dropped until Phase 4).
    //   effects: +1 card (Hologram 1.25), -1 card -> deck stays at 52.
    let mut g = new_game(21);
    for id in [J::DNA, J::SixthSense, J::Hologram] {
        add(&mut g, jk(id));
    }
    g.step(Action::SelectBlind).expect("select");
    set_hand(&mut g, 0, c(0, Six, Spades));
    let six_uid = g.state().deck.hand[0].uid;
    let dropped0 = g.state().stats.effects_dropped;
    let info = play(&mut g, 0b1);
    assert_eq!(info.hand.expect("hand").score, 11.0);
    assert_eq!((info.cards_added, info.cards_destroyed), (1, 1));
    let s = g.state();
    assert_eq!(s.deck.len(), 52);
    assert!(
        s.deck.iter_all().all(|c| c.uid != six_uid),
        "original destroyed"
    );
    assert!(
        s.deck
            .hand
            .iter()
            .any(|c| c.rank == Six && c.suit == Spades && c.uid != six_uid),
        "DNA copy is in the hand"
    );
    assert_eq!(joker_value(&g, J::Hologram), 1.25);
    // The Spectral is created (Phase 4 implements consumables).
    assert_eq!(s.stats.effects_dropped, dropped0);
    assert_eq!(s.consumables.len(), 1);
    assert_eq!(
        s.consumables[0].kind(),
        balatro_core::consumable::ConsumableKind::Spectral
    );
}
