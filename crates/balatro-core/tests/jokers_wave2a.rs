//! Phase 3 `wave2a`: Ceremonial Dagger, Loyalty Card, Gros Michel, Runner, Constellation, Green
//! Joker, Card Sharp, Red Card, Madness, Square Joker, Vampire, Hologram, Obelisk, Fortune
//! Teller, Lucky Cat. Per joker: trigger, non-trigger, scaling over several events, reset or
//! decay, Blueprint (copy reads the value, never changes it) and a `Game`-level case that proves
//! the event fires at the right moment. Every expected number is derived in a comment.
//!
//! Base values used throughout: Pair = 10 chips x 2 mult, High Card = 5 x 1, Straight = 30 x 4,
//! Straight Flush = 100 x 8 (level 1). A pair of 7s: 10 + 7 + 7 = 24 chips x 2 mult = 48.
//! Constellation, Fortune Teller and Red Card depend on systems of Phase 4/5 (consumables,
//! packs): their events are fired directly with `fire_consumable_used` / `fire_pack_skip`.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::card::Card;
use balatro_core::content::ConsumableId;
use balatro_core::content::jokers::{JokerId, try_def};
use balatro_core::deck::{Deck, Pile};
use balatro_core::enums::{Enhancement, HandType, Rank, Rarity, Seal, Suit};
use balatro_core::game::{Game, StepInfo, apply_side_effects};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{
    ConsumableView, DiscardInfo, EventInput, EventResult, PackKind, RoundEndInfo, RunFlag,
    RunStats, RunView, SideEffect, fire_acquire, fire_blind_select, fire_cards_added,
    fire_consumable_used, fire_discard, fire_pack_open, fire_pack_skip, fire_round_end,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer, TraceEvent, TraceOp};
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

/// 7♠ 7♥ Pair: 24 chips x 2 mult = 48 with no jokers.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

/// 3-4-5-6-7 mixed suits: Straight, 30 + 3 + 4 + 5 + 6 + 7 = 55 chips x 4 mult = 220.
fn straight() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Three, Spades),
            c(2, Four, Hearts),
            c(3, Five, Clubs),
            c(4, Six, Diamonds),
            c(5, Seven, Spades),
        ],
        ev(HandType::Straight, 0b11111, &[]),
    )
}

struct Out {
    res: ScoreResult,
    effects: Vec<SideEffect>,
    trace: Vec<TraceEvent>,
    played: Vec<Card>,
}

fn score_with(
    played: &[Card],
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
        held: &[],
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    });
    Out {
        res,
        effects: scorer.effects().to_vec(),
        trace: scorer.trace().to_vec(),
        played,
    }
}

fn score(played: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> Out {
    score_with(played, eval, js, &RunView::default(), 11)
}

fn run_pm(prob_mult: u32) -> RunView<'static> {
    RunView {
        prob_mult,
        ..RunView::default()
    }
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Run an event dispatcher against `js` with a fresh standard deck and `seed`.
fn event(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    seed: u64,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> (i32, Vec<SideEffect>) {
    let deck = Deck::standard();
    let mut rng = GameRng::new(seed);
    let mut fx = Vec::new();
    let r = f(EventInput {
        jokers: js,
        run,
        deck: &deck,
        rng: &mut rng,
        effects: &mut fx,
    });
    (r.money_delta, fx)
}

fn blind_select(js: &mut [JokerInstance], blind: BlindKind, seed: u64) -> Vec<SideEffect> {
    let run = RunView {
        blind,
        ..RunView::default()
    };
    event(js, &run, seed, fire_blind_select).1
}

fn destroyed_uids(fx: &[SideEffect]) -> Vec<u32> {
    fx.iter()
        .filter_map(|e| match e {
            SideEffect::DestroyJoker { uid } => Some(*uid),
            _ => None,
        })
        .collect()
}

fn round_end(js: &mut [JokerInstance], run: &RunView<'_>, seed: u64) -> Vec<SideEffect> {
    let info = RoundEndInfo {
        won: true,
        saved: false,
        round_score: 400.0,
        target: 300.0,
    };
    event(js, run, seed, |inp| fire_round_end(inp, &info)).1
}

fn planet() -> ConsumableView {
    ConsumableView::new(ConsumableId::planet_for(HandType::Pair))
}

fn tarot() -> ConsumableView {
    ConsumableView::new(ConsumableId::TAROTS[0])
}

fn spectral() -> ConsumableView {
    ConsumableView::new(ConsumableId::SPECTRALS[0])
}

// ----- game-level helpers -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn select(g: &mut Game) -> StepInfo {
    g.step(Action::SelectBlind).expect("select blind")
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("joker room")
}

/// Put the round score one chip below the target, then play card 0 (any hand scores >= 7).
fn beat_blind(g: &mut Game) -> StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn set_hand(g: &mut Game, cards: &[(Rank, Suit)]) {
    for (i, &(r, s)) in cards.iter().enumerate() {
        let h = &mut g.state_mut().deck.hand[i];
        h.rank = r;
        h.suit = s;
        h.enhancement = Enhancement::None;
    }
}

fn play(g: &mut Game, mask: u16) -> StepInfo {
    g.step(Action::Play(mask)).expect("play")
}

// ----- definitions -----

#[test]
fn definitions_match_the_wiki() {
    // (id, rarity, cost, eternal ok, perishable ok, init value)
    let table: [(JokerId, Rarity, u32, bool, bool, f64); 15] = [
        (J::CeremonialDagger, Rarity::Uncommon, 6, true, false, 0.0),
        (J::LoyaltyCard, Rarity::Uncommon, 5, true, true, 0.0),
        (J::GrosMichel, Rarity::Common, 5, false, true, 0.0),
        (J::Runner, Rarity::Common, 5, true, false, 0.0),
        (J::Constellation, Rarity::Uncommon, 6, true, false, 1.0),
        (J::GreenJoker, Rarity::Common, 4, true, false, 0.0),
        (J::CardSharp, Rarity::Uncommon, 6, true, true, 0.0),
        (J::RedCard, Rarity::Common, 5, true, false, 0.0),
        (J::Madness, Rarity::Uncommon, 7, true, false, 1.0),
        (J::SquareJoker, Rarity::Common, 4, true, false, 0.0),
        (J::Vampire, Rarity::Uncommon, 7, true, false, 1.0),
        (J::Hologram, Rarity::Uncommon, 7, true, false, 1.0),
        (J::Obelisk, Rarity::Rare, 8, true, false, 1.0),
        (J::FortuneTeller, Rarity::Common, 6, true, true, 0.0),
        (J::LuckyCat, Rarity::Uncommon, 6, true, false, 1.0),
    ];
    for (id, rarity, cost, eternal, perishable, init) in table {
        let d = try_def(id).unwrap_or_else(|| panic!("{id:?} has no def"));
        assert_eq!(d.id, id);
        assert_eq!((d.rarity, d.base_cost), (rarity, cost), "{id:?}");
        assert!(d.blueprint_compatible, "{id:?} is Blueprint compatible");
        assert_eq!(d.eternal_compatible, eternal, "{id:?} eternal");
        assert_eq!(d.perishable_compatible, perishable, "{id:?} perishable");
        assert_eq!(d.init_value, init, "{id:?} init value");
        assert!(d.copies.is_none());
    }
}

// ----- Ceremonial Dagger -----

#[test]
fn dagger_destroys_right_neighbour_and_adds_double_sell_value() {
    // Mime (cost 5) with sell value forced to 4: Dagger gains 2 x 4 = +8 mult.
    let mut js = jokers(&[J::CeremonialDagger, J::Mime]);
    js[1].sell_value = 4;
    let fx = blind_select(&mut js, BlindKind::Small, 1);
    assert_eq!(fx, vec![SideEffect::DestroyJoker { uid: 101 }]);
    assert_eq!(js[0].value, 8.0);
    // joker_main: 24 x (2 + 8) = 240.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js[..1]).res.score, 240.0);
}

#[test]
fn dagger_triggers_on_boss_blind_too() {
    let mut js = jokers(&[J::CeremonialDagger, J::Mime]);
    js[1].sell_value = 3;
    let fx = blind_select(&mut js, BlindKind::Boss, 1);
    assert_eq!(destroyed_uids(&fx), vec![101]);
    assert_eq!(js[0].value, 6.0);
}

#[test]
fn dagger_nothing_when_rightmost_or_alone() {
    let mut js = jokers(&[J::Mime, J::CeremonialDagger]);
    assert!(blind_select(&mut js, BlindKind::Small, 1).is_empty());
    assert_eq!(js[1].value, 0.0);
    let mut alone = jokers(&[J::CeremonialDagger]);
    assert!(blind_select(&mut alone, BlindKind::Big, 1).is_empty());
    assert_eq!(alone[0].value, 0.0);
    // No scaling yet: score unchanged (48).
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut alone).res.score, 48.0);
}

#[test]
fn dagger_spares_eternal_and_gains_nothing() {
    let mut js = jokers(&[J::CeremonialDagger, J::Mime]);
    js[1].stickers.eternal = true;
    js[1].sell_value = 5;
    assert!(blind_select(&mut js, BlindKind::Small, 1).is_empty());
    assert_eq!(js[0].value, 0.0);
}

#[test]
fn dagger_accumulates_over_blinds() {
    // Blind 1: right neighbour Mime sell 4 -> +8. Blind 2: new neighbour Hack sell 3 -> +6.
    let mut js = jokers(&[J::CeremonialDagger, J::Mime]);
    js[1].sell_value = 4;
    blind_select(&mut js, BlindKind::Small, 1);
    js.truncate(1); // the game layer removed Mime
    js.push({
        let mut h = jk(102, J::Hack);
        h.sell_value = 3;
        h
    });
    let fx = blind_select(&mut js, BlindKind::Big, 2);
    assert_eq!(destroyed_uids(&fx), vec![102]);
    assert_eq!(js[0].value, 14.0);
}

#[test]
fn dagger_skips_a_joker_that_is_already_doomed() {
    // Dagger A destroys Dagger B (sell 3 -> +6); B is doomed so it never fires; X survives.
    let mut js = jokers(&[J::CeremonialDagger, J::CeremonialDagger, J::Mime]);
    let fx = blind_select(&mut js, BlindKind::Small, 1);
    assert_eq!(destroyed_uids(&fx), vec![101]);
    assert_eq!(js[0].value, 6.0);
    assert_eq!(js[1].value, 0.0);
}

#[test]
fn dagger_blueprint_does_not_destroy_or_scale_twice() {
    // [Blueprint, Dagger, Mime]: Blueprint copies the Dagger but copies never destroy.
    let mut js = jokers(&[J::Blueprint, J::CeremonialDagger, J::Mime]);
    js[2].sell_value = 4;
    let fx = blind_select(&mut js, BlindKind::Small, 1);
    assert_eq!(fx, vec![SideEffect::DestroyJoker { uid: 102 }]);
    assert_eq!(js[1].value, 8.0, "gained once");
    assert_eq!(js[0].value, 0.0, "Blueprint state untouched");
    // Both Blueprint and Dagger pay +8: 24 x (2 + 8 + 8) = 432.
    js.truncate(2);
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).res.score, 432.0);
    assert_eq!(js[1].value, 8.0);
}

#[test]
fn game_dagger_destroys_at_blind_select_and_scales_play() {
    let mut g = new_game(1);
    add(&mut g, J::CeremonialDagger);
    add(&mut g, J::Hack); // cost 6 -> sell value 3
    assert_eq!(g.state().jokers[1].sell_value, 3);
    select(&mut g);
    let s = g.state();
    assert_eq!(
        s.jokers.len(),
        1,
        "Hack destroyed when the blind was selected"
    );
    assert_eq!(s.jokers[0].value, 6.0); // 2 x 3
    // High Card mult 1 + 6 = 7.
    assert_eq!(play(&mut g, 0b1).hand.expect("hand").mult, 7.0);
    // The next blind has nothing to the right: no more growth.
    beat_blind(&mut g);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    assert_eq!(g.state().jokers[0].value, 6.0);
}

#[test]
fn game_dagger_keeps_eternal_neighbour() {
    let mut g = new_game(2);
    add(&mut g, J::CeremonialDagger);
    add(&mut g, J::Hack);
    g.state_mut().jokers[1].stickers.eternal = true;
    select(&mut g);
    assert_eq!(g.state().jokers.len(), 2);
    assert_eq!(g.state().jokers[0].value, 0.0);
}

// ----- Loyalty Card -----

fn loyalty_run(offset: u32, k: u32) -> RunView<'static> {
    // k-th hand since acquisition, including the current hand.
    RunView {
        hands_played_run: offset + k,
        ..RunView::default()
    }
}

#[test]
fn loyalty_acquire_records_the_current_hand_count() {
    let mut js = jokers(&[J::LoyaltyCard]);
    let run = RunView {
        hands_played_run: 3,
        ..RunView::default()
    };
    event(&mut js, &run, 1, |inp| fire_acquire(inp, 0));
    assert_eq!(js[0].counter, 3);
    // A duplicate (Ankh/Invisible) fires on_acquire again but keeps the recorded offset.
    let run9 = RunView {
        hands_played_run: 9,
        ..RunView::default()
    };
    event(&mut js, &run9, 1, |inp| fire_acquire(inp, 0));
    assert_eq!(js[0].counter, 3);
}

#[test]
fn loyalty_every_sixth_hand_gives_x4() {
    // Acquired when 3 hands had been played. Hand k since acquisition: x4 only at k % 6 == 0.
    let (p, e) = pair7();
    let mut js = jokers(&[J::LoyaltyCard]);
    js[0].counter = 3;
    for k in 1..=13 {
        let run = loyalty_run(3, k);
        let o = score_with(&p, e, &mut js, &run, 1);
        // 24 x 2 = 48; the 6th and 12th hand: 24 x (2 x 4) = 192.
        let want = if k % 6 == 0 { 192.0 } else { 48.0 };
        assert_eq!(o.res.score, want, "hand {k}");
    }
    assert_eq!(js[0].counter, 3, "the offset never changes while scoring");
}

#[test]
fn loyalty_blueprint_uses_targets_offset() {
    // Loyalty acquired at offset 0; hand 6. Blueprint copy and the Loyalty Card both give x4:
    // 24 x (2 x 4 x 4) = 768. Neither state changes.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::LoyaltyCard]);
    let o = score_with(&p, e, &mut js, &loyalty_run(0, 6), 1);
    assert_eq!(o.res.score, 768.0);
    assert_eq!((js[0].counter, js[1].counter), (0, 0));
    // Hand 5: neither fires.
    let o5 = score_with(&p, e, &mut js, &loyalty_run(0, 5), 1);
    assert_eq!(o5.res.score, 48.0);
}

#[test]
fn game_loyalty_counts_hands_since_purchase() {
    let mut g = new_game(5);
    select(&mut g);
    // Two hands before buying: hands_played_run = 2 at acquisition -> offset 2.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    add(&mut g, J::LoyaltyCard);
    assert_eq!(g.state().jokers[0].counter, 2);
    // Hands 1 and 2 since purchase (the second wins the blind).
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    // Hands 3, 4, 5 since purchase: nothing; hand 6 (this round's 4th hand): High Card x4.
    for _ in 0..3 {
        assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    }
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let h = play(&mut g, 0b1).hand.expect("h");
    assert_eq!(h.mult, 4.0, "1 mult x4 on the 6th hand since purchase");
}

// ----- Gros Michel -----

#[test]
fn gros_michel_adds_fifteen_mult() {
    // 24 x (2 + 15) = 408.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut jokers(&[J::GrosMichel])).res.score, 408.0);
    // Blueprint copies the +15: 24 x (2 + 30) = 768.
    assert_eq!(
        score(&p, e, &mut jokers(&[J::Blueprint, J::GrosMichel]))
            .res
            .score,
        768.0
    );
}

#[test]
fn gros_michel_round_end_extinction_roll() {
    // prob_mult 1000 -> always: destroys itself and sets the flag.
    let mut js = jokers(&[J::GrosMichel]);
    let fx = round_end(&mut js, &run_pm(1000), 1);
    assert_eq!(
        fx,
        vec![
            SideEffect::DestroyJoker { uid: 100 },
            SideEffect::SetRunFlag(RunFlag::GrosMichelExtinct)
        ]
    );
    // prob_mult 0 -> never.
    assert!(round_end(&mut js, &run_pm(0), 1).is_empty());
    // Not copied: Blueprint adds no second roll or destruction.
    let mut bp = jokers(&[J::Blueprint, J::GrosMichel]);
    let fx = round_end(&mut bp, &run_pm(1000), 1);
    assert_eq!(destroyed_uids(&fx), vec![101]);
    assert_eq!(fx.len(), 2);
}

#[test]
fn gros_michel_odds_are_one_in_six() {
    // Over 6000 rolls with default prob_mult the hit rate is ~1/6 (generous bounds).
    let hits = (0..6000u64)
        .filter(|&s| {
            let mut js = jokers(&[J::GrosMichel]);
            !round_end(&mut js, &RunView::default(), s).is_empty()
        })
        .count();
    assert!((850..1150).contains(&hits), "hits = {hits}");
    // Oops! All 6s (prob_mult 2) doubles it to 2 in 6.
    let hits2 = (0..6000u64)
        .filter(|&s| {
            let mut js = jokers(&[J::GrosMichel]);
            !round_end(&mut js, &run_pm(2), s).is_empty()
        })
        .count();
    assert!((1850..2150).contains(&hits2), "hits2 = {hits2}");
}

#[test]
fn game_gros_michel_pays_then_goes_extinct_at_round_end() {
    let (mut gone, mut kept) = (false, false);
    for seed in 0..64 {
        let mut g = new_game(seed);
        add(&mut g, J::GrosMichel);
        select(&mut g);
        // High Card: 1 + 15 = 16 mult while it lives.
        assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 16.0);
        beat_blind(&mut g);
        let s = g.state();
        if s.jokers.is_empty() {
            assert!(s.has_flag(RunFlag::GrosMichelExtinct));
            gone = true;
        } else {
            assert!(!s.has_flag(RunFlag::GrosMichelExtinct));
            kept = true;
        }
        if gone && kept {
            return;
        }
    }
    panic!("Gros Michel never both survived and went extinct in 64 seeds (1 in 6)");
}

// ----- Runner -----

#[test]
fn runner_scales_on_straights_only() {
    let mut js = jokers(&[J::Runner]);
    let (st, se) = straight();
    let (pp, pe) = pair7();
    // Straight: +15 chips immediately: (55 + 15) x 4 = 280.
    assert_eq!(score(&st, se, &mut js).res.score, 280.0);
    assert_eq!(js[0].value, 15.0);
    // Pair: no gain, keeps +15 chips: (24 + 15) x 2 = 78.
    assert_eq!(score(&pp, pe, &mut js).res.score, 78.0);
    assert_eq!(js[0].value, 15.0);
    // Second Straight: value 30: (55 + 30) x 4 = 340.
    assert_eq!(score(&st, se, &mut js).res.score, 340.0);
    assert_eq!(js[0].value, 30.0);
}

#[test]
fn runner_counts_hands_that_contain_a_straight() {
    // Straight Flush (contains Straight): +15 -> (100 + 25 + 15) x 8 = 1120.
    let (st, _) = straight();
    let e = ev(HandType::StraightFlush, 0b11111, &[HandType::Straight]);
    let mut js = jokers(&[J::Runner]);
    assert_eq!(score(&st, e, &mut js).res.score, 1120.0);
    assert_eq!(js[0].value, 15.0);
}

#[test]
fn runner_blueprint_reads_but_does_not_scale() {
    // Runner scales once to 15; Blueprint (before it) also pays 15: (55 + 15 + 15) x 4 = 340.
    let (st, se) = straight();
    let mut js = jokers(&[J::Blueprint, J::Runner]);
    assert_eq!(score(&st, se, &mut js).res.score, 340.0);
    assert_eq!((js[0].value, js[1].value), (0.0, 15.0));
}

// ----- Constellation -----

fn use_consumable(js: &mut [JokerInstance], used: ConsumableView) {
    event(js, &RunView::default(), 1, |inp| {
        fire_consumable_used(inp, &used)
    });
}

#[test]
fn constellation_gains_per_planet_only() {
    let mut js = jokers(&[J::Constellation]);
    assert_eq!(js[0].value, 1.0);
    use_consumable(&mut js, tarot());
    use_consumable(&mut js, spectral()); // Black Hole is Spectral
    assert_eq!(js[0].value, 1.0);
    for _ in 0..3 {
        use_consumable(&mut js, planet());
    }
    // 1 + 0.1 + 0.1 + 0.1 (plain f64 additions, like the original).
    assert!(approx(js[0].value, 1.3));
    // Pair: 24 chips x (2 x 1.3 = 2.6) = 62.4 -> floor 62.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).res.score, 62.0);
}

#[test]
fn constellation_unscaled_is_a_noop_and_blueprint_does_not_double_count() {
    let (p, e) = pair7();
    let mut fresh = jokers(&[J::Constellation]);
    assert_eq!(score(&p, e, &mut fresh).res.score, 48.0);
    // [Blueprint, Constellation] + one Planet: only the Constellation scales (1.1).
    let mut js = jokers(&[J::Blueprint, J::Constellation]);
    use_consumable(&mut js, planet());
    assert!(approx(js[1].value, 1.1));
    assert_eq!(js[0].value, 0.0);
    // Both multiply: 2 x 1.1 x 1.1 = 2.42; 24 x 2.42 = 58.08 -> 58.
    assert_eq!(score(&p, e, &mut js).res.score, 58.0);
}

// ----- Green Joker -----

fn discard_event(js: &mut [JokerInstance], n_cards: usize) {
    let cards: Vec<Card> = (0..n_cards).map(|i| c(i as u32 + 1, Two, Hearts)).collect();
    let info = DiscardInfo {
        cards: &cards,
        hand_type: HandType::HighCard,
        forced: false,
    };
    event(js, &RunView::default(), 1, |inp| fire_discard(inp, &info));
}

#[test]
fn green_joker_plus_one_per_hand_minus_one_per_discard() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::GreenJoker]);
    // Hand 1: +1 before scoring, so it already pays: 24 x (2 + 1) = 72.
    assert_eq!(score(&p, e, &mut js).res.score, 72.0);
    // Hand 2: value 2 -> 24 x 4 = 96.
    assert_eq!(score(&p, e, &mut js).res.score, 96.0);
    assert_eq!(js[0].value, 2.0);
    // A 3-card discard counts once (-1, not -3).
    discard_event(&mut js, 3);
    assert_eq!(js[0].value, 1.0);
    // Floor at 0.
    discard_event(&mut js, 1);
    discard_event(&mut js, 1);
    discard_event(&mut js, 1);
    assert_eq!(js[0].value, 0.0);
    // Next hand: +1 -> 24 x 3 = 72.
    assert_eq!(score(&p, e, &mut js).res.score, 72.0);
}

#[test]
fn green_joker_blueprint_copy_neither_gains_nor_loses() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::GreenJoker]);
    js[1].value = 3.0;
    discard_event(&mut js, 1);
    assert_eq!((js[0].value, js[1].value), (0.0, 2.0), "-1 once");
    // Hand: Green +1 = 3 once; Blueprint reads 3: 24 x (2 + 3 + 3) = 192.
    assert_eq!(score(&p, e, &mut js).res.score, 192.0);
    assert_eq!(js[1].value, 3.0);
}

#[test]
fn game_green_joker_events_in_real_flow() {
    let mut g = new_game(7);
    add(&mut g, J::GreenJoker);
    select(&mut g);
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 2.0); // 1 + 1
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 3.0); // 1 + 2
    g.step(Action::Discard(0b111)).expect("discard");
    assert_eq!(g.state().jokers[0].value, 1.0);
    g.step(Action::Discard(0b1)).expect("discard");
    g.step(Action::Discard(0b1)).expect("discard");
    assert_eq!(
        g.state().jokers[0].value,
        0.0,
        "floor 0 (a third discard would go below)"
    );
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 2.0); // 0 + 1 -> 1 + 1
}

// ----- Card Sharp -----

fn run_round_plays(ht: HandType, n: u32) -> RunView<'static> {
    let mut r = RunView::default();
    r.hand_plays_round[ht.index()] = n;
    r
}

#[test]
fn card_sharp_needs_the_same_hand_type_earlier_this_round() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::CardSharp]);
    // First Pair of the round (count 1 includes this hand): 48.
    let first = score_with(&p, e, &mut js, &run_round_plays(HandType::Pair, 1), 1);
    assert_eq!(first.res.score, 48.0);
    // Second Pair: x3 -> 24 x 6 = 144.
    let second = score_with(&p, e, &mut js, &run_round_plays(HandType::Pair, 2), 1);
    assert_eq!(second.res.score, 144.0);
    // A different type played twice does not count: first Pair with Flush played twice.
    let mut run = run_round_plays(HandType::Flush, 2);
    run.hand_plays_round[HandType::Pair.index()] = 1;
    assert_eq!(score_with(&p, e, &mut js, &run, 1).res.score, 48.0);
}

#[test]
fn card_sharp_blueprint_multiplies_twice() {
    // x3 from Card Sharp and x3 from its Blueprint copy: 24 x (2 x 9) = 432.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::CardSharp]);
    let o = score_with(&p, e, &mut js, &run_round_plays(HandType::Pair, 2), 1);
    assert_eq!(o.res.score, 432.0);
}

#[test]
fn game_card_sharp_resets_each_round() {
    let mut g = new_game(9);
    add(&mut g, J::CardSharp);
    select(&mut g);
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0); // first High Card
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 3.0); // second: x3
    assert_eq!(beat_blind(&mut g).hand.expect("h").mult, 3.0); // third
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    assert_eq!(
        play(&mut g, 0b1).hand.expect("h").mult,
        1.0,
        "new round, first again"
    );
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 3.0);
}

// ----- Red Card -----

fn skip_pack(js: &mut [JokerInstance], kind: PackKind) {
    event(js, &RunView::default(), 1, |inp| fire_pack_skip(inp, kind));
}

#[test]
fn red_card_gains_three_per_skipped_pack() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::RedCard]);
    assert_eq!(score(&p, e, &mut js).res.score, 48.0); // starts +0
    skip_pack(&mut js, PackKind::Arcana);
    assert_eq!(js[0].value, 3.0);
    // 24 x (2 + 3) = 120.
    assert_eq!(score(&p, e, &mut js).res.score, 120.0);
    // One skip event per pack, whatever its size or kind (a Mega pack is one `fire_pack_skip`).
    skip_pack(&mut js, PackKind::Celestial);
    skip_pack(&mut js, PackKind::Buffoon);
    assert_eq!(js[0].value, 9.0);
    // 24 x (2 + 9) = 264.
    assert_eq!(score(&p, e, &mut js).res.score, 264.0);
}

#[test]
fn red_card_ignores_opening_a_pack_and_blueprint_does_not_double_gain() {
    let mut js = jokers(&[J::Blueprint, J::RedCard]);
    event(&mut js, &RunView::default(), 1, |inp| {
        fire_pack_open(inp, PackKind::Standard)
    });
    assert_eq!(js[1].value, 0.0);
    skip_pack(&mut js, PackKind::Standard);
    assert_eq!((js[0].value, js[1].value), (0.0, 3.0));
    // Blueprint reads 3 too: 24 x (2 + 3 + 3) = 192.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).res.score, 192.0);
}

// ----- Madness -----

#[test]
fn madness_gains_and_destroys_a_random_other_joker() {
    let mut picked = [false; 2];
    for seed in 0..64 {
        let mut js = jokers(&[J::Madness, J::Mime, J::Hack]);
        let fx = blind_select(&mut js, BlindKind::Small, seed);
        assert_eq!(js[0].value, 1.5, "1 + 0.5");
        let d = destroyed_uids(&fx);
        assert_eq!(d.len(), 1, "seed {seed}");
        assert!(d[0] == 101 || d[0] == 102);
        picked[(d[0] - 101) as usize] = true;
        // Deterministic for a seed.
        let mut js2 = jokers(&[J::Madness, J::Mime, J::Hack]);
        assert_eq!(blind_select(&mut js2, BlindKind::Small, seed), fx);
    }
    assert_eq!(picked, [true, true], "both candidates get picked sometimes");
}

#[test]
fn madness_big_blind_triggers_boss_does_not() {
    let mut js = jokers(&[J::Madness, J::Mime]);
    let fx = blind_select(&mut js, BlindKind::Big, 3);
    assert_eq!(destroyed_uids(&fx), vec![101]);
    assert_eq!(js[0].value, 1.5);
    let mut boss = jokers(&[J::Madness, J::Mime]);
    assert!(blind_select(&mut boss, BlindKind::Boss, 3).is_empty());
    assert_eq!(boss[0].value, 1.0, "no gain on a Boss Blind");
}

#[test]
fn madness_never_picks_eternal_and_gains_even_without_a_victim() {
    for seed in 0..32 {
        let mut js = jokers(&[J::Madness, J::Mime, J::Hack]);
        js[1].stickers.eternal = true;
        let fx = blind_select(&mut js, BlindKind::Small, seed);
        assert_eq!(destroyed_uids(&fx), vec![102], "only Hack is eligible");
    }
    // Only an Eternal neighbour: gain, no destruction.
    let mut js = jokers(&[J::Madness, J::Mime]);
    js[1].stickers.eternal = true;
    assert!(blind_select(&mut js, BlindKind::Small, 1).is_empty());
    assert_eq!(js[0].value, 1.5);
    // Alone: gain, nothing to destroy.
    let mut alone = jokers(&[J::Madness]);
    assert!(blind_select(&mut alone, BlindKind::Small, 1).is_empty());
    assert_eq!(alone[0].value, 1.5);
    // Scaling over two blinds: 1 + 0.5 + 0.5 = 2 -> 24 x (2 x 2) = 96.
    blind_select(&mut alone, BlindKind::Big, 1);
    assert_eq!(alone[0].value, 2.0);
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut alone).res.score, 96.0);
}

#[test]
fn madness_can_destroy_another_madness_but_not_itself() {
    // Two Madness: the left one runs first and can only pick the right one, which is then doomed
    // and does not fire (no gain for it).
    let mut js = jokers(&[J::Madness, J::Madness]);
    let fx = blind_select(&mut js, BlindKind::Small, 1);
    assert_eq!(destroyed_uids(&fx), vec![101]);
    assert_eq!((js[0].value, js[1].value), (1.5, 1.0));
}

#[test]
fn madness_and_dagger_never_destroy_the_same_joker_twice() {
    for seed in 0..64 {
        let mut js = jokers(&[J::Madness, J::CeremonialDagger, J::Mime]);
        let fx = blind_select(&mut js, BlindKind::Small, seed);
        let d = destroyed_uids(&fx);
        let mut dedup = d.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(d.len(), dedup.len(), "seed {seed}: {d:?}");
        assert!(!d.is_empty());
    }
}

#[test]
fn madness_blueprint_copy_does_nothing() {
    // [Blueprint, Madness, Mime]: Madness (slot 1) picks among Blueprint and Mime; the Blueprint
    // copy at slot 0 must add neither a second destruction nor a gain.
    for seed in 0..32 {
        let mut js = jokers(&[J::Blueprint, J::Madness, J::Mime]);
        let fx = blind_select(&mut js, BlindKind::Small, seed);
        assert_eq!(destroyed_uids(&fx).len(), 1, "seed {seed}");
        assert_eq!(js[1].value, 1.5, "gained once");
        assert_eq!(js[0].value, 0.0);
    }
    // joker_main through the copy: Blueprint x2 and Madness x2: 24 x (2 x 2 x 2) = 192.
    let mut js = jokers(&[J::Blueprint, J::Madness]);
    js[1].value = 2.0;
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).res.score, 192.0);
}

#[test]
fn game_madness_on_select_blind_not_on_boss() {
    let mut g = new_game(11);
    add(&mut g, J::Madness);
    add(&mut g, J::Hack);
    select(&mut g);
    assert_eq!(
        g.state().jokers.len(),
        1,
        "Hack destroyed on Small blind select"
    );
    assert_eq!(g.state().jokers[0].id, J::Madness);
    // High Card: 1 x 1.5 = 1.5.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.5);

    // Boss blind: nothing destroyed, no gain.
    let mut b = new_game(12);
    add(&mut b, J::Madness);
    add(&mut b, J::Hack);
    b.state_mut().blind = BlindKind::Boss;
    select(&mut b);
    assert_eq!(b.state().jokers.len(), 2);
    assert_eq!(b.state().jokers[0].value, 1.0);
}

#[test]
fn game_madness_spares_eternal() {
    let mut g = new_game(13);
    add(&mut g, J::Madness);
    add(&mut g, J::Hack);
    g.state_mut().jokers[1].stickers.eternal = true;
    select(&mut g);
    assert_eq!(g.state().jokers.len(), 2);
    assert_eq!(g.state().jokers[0].value, 1.5);
}

// ----- Square Joker -----

#[test]
fn square_joker_scales_on_exactly_four_played_cards() {
    // 7♠ 7♥ 2♣ 3♦: Pair scores the two 7s, the kickers do not score but count as played.
    let four = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Two, Clubs),
        c(4, Three, Diamonds),
    ];
    let e = ev(HandType::Pair, 0b0011, &[]);
    let mut js = jokers(&[J::SquareJoker]);
    // +4 chips this hand: (24 + 4) x 2 = 56.
    assert_eq!(score(&four, e, &mut js).res.score, 56.0);
    // Second four-card hand: +8 total: (24 + 8) x 2 = 64.
    assert_eq!(score(&four, e, &mut js).res.score, 64.0);
    assert_eq!(js[0].value, 8.0);
    // Three or five cards: no gain, still pays +8: pair7 (2 cards) -> (24 + 8) x 2 = 64.
    let (p, pe) = pair7();
    assert_eq!(score(&p, pe, &mut js).res.score, 64.0);
    let five = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Two, Clubs),
        c(4, Three, Diamonds),
        c(5, Nine, Clubs),
    ];
    score(&five, ev(HandType::Pair, 0b00011, &[]), &mut js);
    assert_eq!(js[0].value, 8.0);
    // Debuffed cards still count as played: all four debuffed -> still +4 (nothing scores, 0 x 1
    // is irrelevant here, only the value matters).
    let mut deb = jokers(&[J::SquareJoker]);
    let dfour = four.map(debuff);
    score(&dfour, e, &mut deb);
    assert_eq!(deb[0].value, 4.0);
}

#[test]
fn square_joker_blueprint_reads_without_scaling() {
    let four = [
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Two, Clubs),
        c(4, Three, Diamonds),
    ];
    let e = ev(HandType::Pair, 0b0011, &[]);
    let mut js = jokers(&[J::Blueprint, J::SquareJoker]);
    // Square +4 once; Blueprint pays 4 too: (24 + 4 + 4) x 2 = 64.
    assert_eq!(score(&four, e, &mut js).res.score, 64.0);
    assert_eq!((js[0].value, js[1].value), (0.0, 4.0));
}

#[test]
fn game_square_joker_four_card_play() {
    let mut g = new_game(15);
    add(&mut g, J::SquareJoker);
    select(&mut g);
    set_hand(
        &mut g,
        &[
            (Seven, Spades),
            (Seven, Hearts),
            (Two, Clubs),
            (Three, Diamonds),
        ],
    );
    // Pair of 7s: (10 + 14 + 4) x 2 = 56.
    assert_eq!(play(&mut g, 0b1111).hand.expect("h").score, 56.0);
    assert_eq!(g.state().jokers[0].value, 4.0);
    set_hand(&mut g, &[(Ace, Spades)]);
    // One card: High Card (5 + 11 + 4) x 1 = 20; no gain.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").score, 20.0);
    assert_eq!(g.state().jokers[0].value, 4.0);
}

// ----- Vampire -----

fn enhanced_pair() -> Vec<Card> {
    // 7♠ Mult (+4 mult), 7♥ Bonus (+30 chips).
    vec![
        enh(c(1, Seven, Spades), Enhancement::Mult),
        enh(c(2, Seven, Hearts), Enhancement::Bonus),
    ]
}

#[test]
fn vampire_strips_scoring_enhancements_before_they_apply() {
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut js = jokers(&[J::Vampire]);
    let o = score(&enhanced_pair(), e, &mut js);
    // Stripped before scoring: chips 10 + 7 + 7 = 24 (no +30), mult 2 (no +4). Two cards eaten:
    // value 1 + 0.1 x 2 = 1.2 -> mult 2.4; 24 x 2.4 = 57.6 -> 57.
    assert_eq!(o.res.score, 57.0);
    assert!(approx(js[0].value, 1.2));
    // The mutation is on the played cards themselves (persistent in the game).
    assert!(o.played.iter().all(|c| c.enhancement == Enhancement::None));
    // Second hand with the (now plain) cards: nothing to eat, value stays 1.2: 24 x 2.4 -> 57.
    let o2 = score(&o.played, e, &mut js);
    assert_eq!(o2.res.score, 57.0);
    assert!(approx(js[0].value, 1.2));
}

#[test]
fn vampire_leaves_unscored_debuffed_plain_cards_editions_and_seals() {
    let e = ev(HandType::Pair, 0b011, &[]);
    // Pair scores the two 7s; the Glass kicker does not score -> keeps Glass.
    let cards = vec![
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        enh(c(3, Two, Clubs), Enhancement::Glass),
    ];
    let mut js = jokers(&[J::Vampire]);
    let o = score(&cards, e, &mut js);
    assert_eq!(js[0].value, 1.0);
    assert_eq!(o.played[2].enhancement, Enhancement::Glass);
    assert_eq!(o.res.score, 48.0, "no gain, x1 skipped");

    // A debuffed enhanced scoring card is untouched and gives no gain.
    let deb = vec![
        debuff(enh(c(1, Seven, Spades), Enhancement::Mult)),
        c(2, Seven, Hearts),
    ];
    let mut js2 = jokers(&[J::Vampire]);
    let o2 = score(&deb, ev(HandType::Pair, 0b11, &[]), &mut js2);
    assert_eq!(js2[0].value, 1.0);
    assert_eq!(o2.played[0].enhancement, Enhancement::Mult);

    // Seal and edition survive the strip: Red Seal Bonus card is stripped but keeps the seal.
    let sealed = vec![
        seal(enh(c(1, Seven, Spades), Enhancement::Bonus), Seal::Red),
        c(2, Seven, Hearts),
    ];
    let mut js3 = jokers(&[J::Vampire]);
    let o3 = score(&sealed, ev(HandType::Pair, 0b11, &[]), &mut js3);
    assert_eq!(o3.played[0].enhancement, Enhancement::None);
    assert_eq!(o3.played[0].seal, Seal::Red);
    assert!(approx(js3[0].value, 1.1));
    // Red Seal retrigger without the Bonus: chips 10 + 7 + 7 + 7 = 31; mult 2 x 1.1 = 2.2 -> 68.2.
    assert_eq!(o3.res.score, 68.0);
}

#[test]
fn vampire_eats_stone_and_prevents_glass_roll() {
    // Scoring Stone (printed A♠): stripped -> a normal Ace of Spades: 5 + 11 = 16 chips,
    // mult 1 x 1.1: 17.6 -> 17. (Unstripped it would be 5 + 0 + 50 = 55.)
    let stone = vec![enh(c(1, Ace, Spades), Enhancement::Stone)];
    let mut js = jokers(&[J::Vampire]);
    let o = score(&stone, ev(HandType::HighCard, 0b1, &[]), &mut js);
    assert_eq!(o.res.score, 17.0);
    // A scoring Glass card never gets its x2 and never breaks, for any seed.
    for seed in 0..40 {
        let glass = vec![
            enh(c(1, Seven, Spades), Enhancement::Glass),
            c(2, Seven, Hearts),
        ];
        let mut vj = jokers(&[J::Vampire]);
        let o = score_with(
            &glass,
            ev(HandType::Pair, 0b11, &[]),
            &mut vj,
            &RunView::default(),
            seed,
        );
        assert!(
            !o.effects
                .iter()
                .any(|e| matches!(e, SideEffect::DestroyCard { .. })),
            "seed {seed}"
        );
        // 24 chips x (2 x 1.1 = 2.2) = 52.8 -> 52.
        assert_eq!(o.res.score, 52.0);
    }
}

#[test]
fn vampire_only_the_leftmost_vampire_eats() {
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut js = jokers(&[J::Vampire, J::Vampire]);
    score(&enhanced_pair(), e, &mut js);
    assert!(approx(js[0].value, 1.2));
    assert_eq!(js[1].value, 1.0);
}

#[test]
fn vampire_blueprint_copy_neither_strips_nor_scales() {
    let e = ev(HandType::Pair, 0b11, &[]);
    // Blueprint left of Vampire runs its copied before_scoring first: it must not strip, or the
    // real Vampire would find nothing.
    let mut js = jokers(&[J::Blueprint, J::Vampire]);
    let o = score(&enhanced_pair(), e, &mut js);
    assert!(approx(js[1].value, 1.2), "Vampire ate both cards itself");
    assert_eq!(js[0].value, 0.0);
    // Blueprint x1.2 and Vampire x1.2: mult 2 x 1.44 = 2.88; 24 x 2.88 = 69.12 -> 69.
    assert_eq!(o.res.score, 69.0);
}

#[test]
fn game_vampire_strip_persists_in_the_deck() {
    let mut g = new_game(21);
    add(&mut g, J::Vampire);
    select(&mut g);
    set_hand(&mut g, &[(Seven, Spades), (Seven, Hearts)]);
    let (u0, u1) = {
        let h = &mut g.state_mut().deck.hand;
        h[0].enhancement = Enhancement::Mult;
        h[1].enhancement = Enhancement::Bonus;
        (h[0].uid, h[1].uid)
    };
    let h = play(&mut g, 0b11).hand.expect("h");
    // 24 chips x 2.4 mult = 57.6 -> 57.
    assert_eq!(h.score, 57.0);
    for uid in [u0, u1] {
        let card = g.state().deck.iter_all().find(|c| c.uid == uid).copied();
        assert_eq!(
            card.map(|c| c.enhancement),
            Some(Enhancement::None),
            "uid {uid}"
        );
    }
    assert!(approx(g.state().jokers[0].value, 1.2));
    assert!(g.state().check_invariants().is_ok());
}

// ----- Hologram -----

fn added(js: &mut [JokerInstance], n: usize) {
    let cards: Vec<Card> = (0..n).map(|i| c(200 + i as u32, Ace, Spades)).collect();
    event(js, &RunView::default(), 1, |inp| {
        fire_cards_added(inp, &cards)
    });
}

#[test]
fn hologram_gains_per_card_added() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Hologram]);
    assert_eq!(score(&p, e, &mut js).res.score, 48.0);
    added(&mut js, 1);
    assert_eq!(js[0].value, 1.25);
    added(&mut js, 3); // per card, not per event
    assert_eq!(js[0].value, 2.0);
    added(&mut js, 0);
    assert_eq!(js[0].value, 2.0);
    // 24 x (2 x 2.0) = 96.
    assert_eq!(score(&p, e, &mut js).res.score, 96.0);
}

#[test]
fn hologram_blueprint_reads_without_double_counting() {
    let mut js = jokers(&[J::Blueprint, J::Hologram]);
    added(&mut js, 2);
    assert_eq!((js[0].value, js[1].value), (0.0, 1.5));
    // Both x1.5: 24 x (2 x 2.25 = 4.5) = 108.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).res.score, 108.0);
}

#[test]
fn game_hologram_fires_when_cards_enter_the_deck() {
    let mut g = new_game(31);
    add(&mut g, J::Hologram);
    select(&mut g);
    let fx = [
        SideEffect::AddCard {
            card: c(0, Ace, Hearts),
            to: Pile::Draw,
        },
        SideEffect::AddCard {
            card: c(0, King, Hearts),
            to: Pile::Hand,
        },
    ];
    let rep = apply_side_effects(g.state_mut(), &fx);
    assert_eq!(rep.added, 2);
    assert_eq!(g.state().jokers[0].value, 1.5); // 1 + 2 x 0.25
    // High Card with Hologram: mult 1 x 1.5 = 1.5.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.5);
}

// ----- Obelisk -----

/// Play counts for one hand: `(hand type, count)` pairs, all others 0 (current hand included).
fn plays(counts: &[(HandType, u32)]) -> RunView<'static> {
    let mut r = RunView::default();
    for &(h, n) in counts {
        r.hand_plays_run[h.index()] = n;
    }
    r
}

#[test]
fn obelisk_gains_unless_strictly_most_played() {
    let (p, e) = pair7();
    // Pair is the only hand ever played (1 > 0 everywhere): strictly most played -> reset/no gain.
    let mut js = jokers(&[J::Obelisk]);
    js[0].value = 1.4;
    let o = score_with(&p, e, &mut js, &plays(&[(HandType::Pair, 1)]), 1);
    assert_eq!(js[0].value, 1.0);
    assert_eq!(o.res.score, 48.0, "reset happens before this hand scores");

    // Another hand was played more often: +0.2 -> 1.2: 24 x (2 x 1.2 = 2.4) = 57.6 -> 57.
    let mut js2 = jokers(&[J::Obelisk]);
    let o2 = score_with(
        &p,
        e,
        &mut js2,
        &plays(&[(HandType::Pair, 1), (HandType::HighCard, 2)]),
        1,
    );
    assert!(approx(js2[0].value, 1.2));
    assert_eq!(o2.res.score, 57.0);
}

#[test]
fn obelisk_tie_does_not_reset() {
    // Pair 2, High Card 2: the current hand is tied for most played, not strictly: gains.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Obelisk]);
    js[0].value = 1.2;
    score_with(
        &p,
        e,
        &mut js,
        &plays(&[(HandType::Pair, 2), (HandType::HighCard, 2)]),
        1,
    );
    assert!(approx(js[0].value, 1.4));
}

#[test]
fn obelisk_sequence_of_hands() {
    // Cumulative counts include the current hand. P = Pair, H = High Card, F = Flush.
    //   1. P: P=1                -> strictly most -> 1.0
    //   2. H: H=1, P=1           -> tie           -> 1.2
    //   3. F: F=1, P=1, H=1      -> tie           -> 1.4
    //   4. F: F=2 > 1            -> reset         -> 1.0
    //   5. P: P=2, F=2           -> tie           -> 1.2
    //   6. H: H=2, P=2, F=2      -> tie           -> 1.4
    let (p, pe) = pair7();
    let hc = [c(1, Ace, Spades)];
    let he = ev(HandType::HighCard, 0b1, &[]);
    let fl = [
        c(1, Two, Hearts),
        c(2, Five, Hearts),
        c(3, Seven, Hearts),
        c(4, Nine, Hearts),
        c(5, King, Hearts),
    ];
    let fe = ev(HandType::Flush, 0b11111, &[]);
    let mut js = jokers(&[J::Obelisk]);
    let (mut np, mut nh, mut nf) = (0, 0, 0);
    let steps = [
        ('P', 1.0),
        ('H', 1.2),
        ('F', 1.4),
        ('F', 1.0),
        ('P', 1.2),
        ('H', 1.4),
    ];
    for (i, (hand, want)) in steps.into_iter().enumerate() {
        match hand {
            'P' => np += 1,
            'H' => nh += 1,
            _ => nf += 1,
        }
        let run = plays(&[
            (HandType::Pair, np),
            (HandType::HighCard, nh),
            (HandType::Flush, nf),
        ]);
        let (cards, eval): (&[Card], HandEval) = match hand {
            'P' => (&p, pe),
            'H' => (&hc, he),
            _ => (&fl, fe),
        };
        score_with(cards, eval, &mut js, &run, 1);
        assert!(approx(js[0].value, want), "step {}: {}", i + 1, js[0].value);
    }
}

#[test]
fn obelisk_blueprint_reads_without_scaling() {
    // Pair 1, High Card 2: Obelisk 1.0 -> 1.2 once; Blueprint reads 1.2 as well:
    // mult 2 x 1.44 = 2.88; 24 x 2.88 = 69.12 -> 69.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::Obelisk]);
    let run = plays(&[(HandType::Pair, 1), (HandType::HighCard, 2)]);
    let o = score_with(&p, e, &mut js, &run, 1);
    assert!(approx(js[1].value, 1.2));
    assert_eq!(js[0].value, 0.0);
    assert_eq!(o.res.score, 69.0);
}

#[test]
fn game_obelisk_follows_play_counts() {
    let mut g = new_game(41);
    add(&mut g, J::Obelisk);
    select(&mut g);
    // 1. High Card (HC=1, strictly most): mult 1.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 1.0);
    // 2. Pair (P=1, HC=1 tie): gain -> 1.2; Pair mult 2 x 1.2 = 2.4.
    set_hand(&mut g, &[(Seven, Spades), (Seven, Hearts), (Two, Clubs)]);
    let h = play(&mut g, 0b11).hand.expect("h");
    assert_eq!(h.hand_type, HandType::Pair);
    assert!(approx(h.mult, 2.4), "mult {}", h.mult);
    // 3. Pair again (P=2 > HC=1): strictly most -> reset: mult 2.0.
    set_hand(&mut g, &[(Seven, Spades), (Seven, Hearts), (Two, Clubs)]);
    let h = play(&mut g, 0b11).hand.expect("h");
    assert_eq!(h.mult, 2.0);
    assert_eq!(g.state().jokers[0].value, 1.0);
}

// ----- Fortune Teller -----

fn tarot_run(n: u32) -> RunView<'static> {
    RunView {
        stats: RunStats {
            tarots_used: n,
            ..RunStats::default()
        },
        ..RunView::default()
    }
}

#[test]
fn fortune_teller_plus_one_mult_per_tarot_used_this_run() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::FortuneTeller]);
    // No Tarot used: 48.
    assert_eq!(score_with(&p, e, &mut js, &tarot_run(0), 1).res.score, 48.0);
    // 3 Tarots: 24 x (2 + 3) = 120 (retroactive: the count is the run's, not the joker's).
    assert_eq!(
        score_with(&p, e, &mut js, &tarot_run(3), 1).res.score,
        120.0
    );
    // 10 Tarots: 24 x 12 = 288.
    assert_eq!(
        score_with(&p, e, &mut js, &tarot_run(10), 1).res.score,
        288.0
    );
    // Planets used do not count.
    let planets = RunView {
        stats: RunStats {
            planets_used: 5,
            ..RunStats::default()
        },
        ..RunView::default()
    };
    assert_eq!(score_with(&p, e, &mut js, &planets, 1).res.score, 48.0);
}

#[test]
fn fortune_teller_blueprint_and_debuff() {
    let (p, e) = pair7();
    // Blueprint copy pays +3 as well: 24 x (2 + 3 + 3) = 192.
    let mut js = jokers(&[J::Blueprint, J::FortuneTeller]);
    assert_eq!(
        score_with(&p, e, &mut js, &tarot_run(3), 1).res.score,
        192.0
    );
    assert_eq!(js[1].value, 0.0, "no own state to change");
    // Debuffed: nothing.
    let mut deb = vec![jk(1, J::FortuneTeller).with_debuff(true)];
    assert_eq!(
        score_with(&p, e, &mut deb, &tarot_run(3), 1).res.score,
        48.0
    );
    // fire_consumable_used with a Tarot changes no joker state (the count lives in RunStats).
    let mut fresh = jokers(&[J::FortuneTeller]);
    let before = fresh[0];
    use_consumable(&mut fresh, tarot());
    assert_eq!(fresh[0], before);
}

#[test]
fn game_fortune_teller_reads_run_stats() {
    let mut g = new_game(51);
    add(&mut g, J::FortuneTeller);
    select(&mut g);
    g.state_mut().stats.run.tarots_used = 4;
    // High Card: 1 + 4 = 5 mult.
    assert_eq!(play(&mut g, 0b1).hand.expect("h").mult, 5.0);
}

// ----- Lucky Cat -----

fn lucky_pair() -> Vec<Card> {
    vec![
        enh(c(1, Seven, Spades), Enhancement::Lucky),
        c(2, Seven, Hearts),
    ]
}

#[test]
fn lucky_cat_scales_on_successful_lucky_triggers() {
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut js = jokers(&[J::LuckyCat]);
    // prob_mult 1000: both rolls hit (+20 mult, +$20), counted as ONE trigger: 1.25.
    // chips 24, mult 2 + 20 = 22, x1.25 = 27.5; 24 x 27.5 = 660. Money +20.
    let o = score_with(&lucky_pair(), e, &mut js, &run_pm(1000), 1);
    assert_eq!(js[0].value, 1.25);
    assert_eq!(o.res.score, 660.0);
    assert_eq!(o.res.money_delta, 20);
    // Never triggers (prob_mult 0): nothing changes. 24 x (2 x 1.25) = 60.
    let o2 = score_with(&lucky_pair(), e, &mut js, &run_pm(0), 1);
    assert_eq!(js[0].value, 1.25);
    assert_eq!(o2.res.score, 60.0);
}

#[test]
fn lucky_cat_counts_each_retrigger_and_each_card() {
    let e = ev(HandType::Pair, 0b11, &[]);
    // Red Seal Lucky card triggers twice: both hit -> +0.5 = 1.5.
    // chips 10 + 7 + 7 + 7 = 31; mult 2 + 20 + 20 = 42 x 1.5 = 63; 31 x 63 = 1953.
    let red = vec![
        seal(enh(c(1, Seven, Spades), Enhancement::Lucky), Seal::Red),
        c(2, Seven, Hearts),
    ];
    let mut js = jokers(&[J::LuckyCat]);
    let o = score_with(&red, e, &mut js, &run_pm(1000), 1);
    assert_eq!(js[0].value, 1.5);
    assert_eq!(o.res.score, 1953.0);
    // Two Lucky cards: +0.5. chips 24; mult 2 + 40 = 42 x 1.5 = 63; 24 x 63 = 1512.
    let two = vec![
        enh(c(1, Seven, Spades), Enhancement::Lucky),
        enh(c(2, Seven, Hearts), Enhancement::Lucky),
    ];
    let mut js2 = jokers(&[J::LuckyCat]);
    let o2 = score_with(&two, e, &mut js2, &run_pm(1000), 1);
    assert_eq!(js2[0].value, 1.5);
    assert_eq!(o2.res.score, 1512.0);
}

#[test]
fn lucky_cat_ignores_non_lucky_and_debuffed_cards() {
    let e = ev(HandType::Pair, 0b11, &[]);
    let (p, _) = pair7();
    let mut js = jokers(&[J::LuckyCat]);
    let o = score_with(&p, e, &mut js, &run_pm(1000), 1);
    assert_eq!(js[0].value, 1.0);
    assert_eq!(o.res.score, 48.0);
    let deb = vec![
        debuff(enh(c(1, Seven, Spades), Enhancement::Lucky)),
        c(2, Seven, Hearts),
    ];
    score_with(&deb, e, &mut js, &run_pm(1000), 1);
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn lucky_cat_money_only_and_mult_only_both_count() {
    // Search seeds for each roll outcome and check the cat against the trace.
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut seen = [false; 4]; // none, mult only, money only, both
    for seed in 0..400u64 {
        let mut js = jokers(&[J::LuckyCat]);
        let o = score_with(&lucky_pair(), e, &mut js, &RunView::default(), seed);
        let (mut m, mut d) = (false, false);
        for t in &o.trace {
            if let TraceOp::LuckyRoll {
                mult_hit,
                money_hit,
            } = t.op
            {
                (m, d) = (mult_hit, money_hit);
            }
        }
        let want = if m || d { 1.25 } else { 1.0 };
        assert_eq!(js[0].value, want, "seed {seed} (mult {m}, money {d})");
        seen[usize::from(m) + 2 * usize::from(d)] = true;
    }
    assert_eq!(seen, [true; 4], "every roll outcome was exercised");
}

#[test]
fn lucky_cat_blueprint_reads_without_scaling() {
    // Cat scales once to 1.25; Blueprint reads it at its slot (after the scoring cards).
    // mult 22 x 1.25 x 1.25 = 34.375; 24 x 34.375 = 825.
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut js = jokers(&[J::Blueprint, J::LuckyCat]);
    let o = score_with(&lucky_pair(), e, &mut js, &run_pm(1000), 1);
    assert_eq!((js[0].value, js[1].value), (0.0, 1.25));
    assert_eq!(o.res.score, 825.0);
}

#[test]
fn game_lucky_cat_scales_when_the_lucky_card_hits() {
    let (mut hit, mut miss) = (false, false);
    for seed in 0..80 {
        let mut g = new_game(seed);
        add(&mut g, J::LuckyCat);
        select(&mut g);
        set_hand(&mut g, &[(Seven, Spades), (Seven, Hearts), (Two, Clubs)]);
        g.state_mut().deck.hand[0].enhancement = Enhancement::Lucky;
        let info = play(&mut g, 0b11);
        let h = info.hand.expect("h");
        // A hit pays $20 or +20 mult (the cat then multiplies a mult above 2).
        let hit_now = info.money_after - info.money_before >= 20 || h.mult > 3.0;
        let v = g.state().jokers[0].value;
        if hit_now {
            assert_eq!(v, 1.25, "seed {seed}");
            hit = true;
        } else {
            assert_eq!(v, 1.0, "seed {seed}");
            miss = true;
        }
        if hit && miss {
            return;
        }
    }
    panic!("no Lucky hit and miss in 80 seeds");
}
