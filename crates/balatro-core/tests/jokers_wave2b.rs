//! Phase 3 package wave2b: Flash Card, Popcorn, Spare Trousers, Ancient Joker, Ramen, Castle,
//! Campfire, Throwback, Glass Joker, Wee Joker, The Idol, Hit the Road, Canio, Yorick.
//!
//! For each joker: trigger, non-trigger, scaling/state after several events, decay/reset where
//! it exists, and (for copyable jokers) a Blueprint case proving that the copy neither doubles
//! nor loses the counter. Event-driven jokers are additionally driven through the real `Game`
//! (`step`) so the events fire at the right moment. Every expected number is derived in a
//! comment.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::card::{Card, RuleFlags};
use balatro_core::content::jokers::{JokerId, try_def};
use balatro_core::deck::Deck;
use balatro_core::enums::{Enhancement, HandType, Phase, Rank, Rarity, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::{
    ConsumableKind, ConsumableView, DiscardInfo, EventInput, EventResult, RoundEndInfo, RunStats,
    RunView, SideEffect, SoldItem, fire_acquire, fire_cards_destroyed, fire_discard, fire_reroll,
    fire_round_end, fire_sell_other,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::scoring::{ScoreInput, Scorer};
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

/// 7S 7H Pair: base 10 x 2, +7 +7 -> 24 x 2 = 48 with no jokers.
fn pair7() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Seven, Spades), c(2, Seven, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

/// 2S 2H Pair: 10 + 2 + 2 = 14 x 2 = 28 with no jokers.
fn pair2() -> (Vec<Card>, HandEval) {
    (
        vec![c(1, Two, Spades), c(2, Two, Hearts)],
        ev(HandType::Pair, 0b11, &[]),
    )
}

/// 7S 7H 9C 9D Two Pair: base 20 x 2, +7 +7 +9 +9 -> 52 x 2 = 104 with no jokers.
fn two_pair() -> (Vec<Card>, HandEval) {
    (
        vec![
            c(1, Seven, Spades),
            c(2, Seven, Hearts),
            c(3, Nine, Clubs),
            c(4, Nine, Diamonds),
        ],
        ev(HandType::TwoPair, 0b1111, &[HandType::Pair]),
    )
}

struct Out {
    score: f64,
}

fn score_with(played: &[Card], eval: HandEval, js: &mut [JokerInstance], run: &RunView<'_>) -> Out {
    let mut played = played.to_vec();
    let mut rng = GameRng::new(11);
    let mut scorer = Scorer::with_trace(false);
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held: &[],
        eval,
        jokers: js,
        run,
        rng: &mut rng,
    });
    Out { score: res.score }
}

fn score(played: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> Out {
    score_with(played, eval, js, &RunView::default())
}

/// Fire one event with a fresh `GameRng::new(seed)`; returns (money delta, queued effects).
fn fire_with(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    deck: &Deck,
    seed: u64,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> (i32, Vec<SideEffect>) {
    let mut rng = GameRng::new(seed);
    let mut fx = Vec::new();
    let r = f(EventInput {
        jokers: js,
        run,
        deck,
        rng: &mut rng,
        effects: &mut fx,
    });
    (r.money_delta, fx)
}

fn fire(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> Vec<SideEffect> {
    fire_with(js, run, &Deck::standard(), 3, f).1
}

fn reroll(js: &mut [JokerInstance]) {
    fire(js, &RunView::default(), fire_reroll);
}

fn discard(js: &mut [JokerInstance], cards: &[Card]) -> Vec<SideEffect> {
    let info = DiscardInfo {
        cards,
        hand_type: HandType::HighCard,
    };
    fire(js, &RunView::default(), |inp| fire_discard(inp, &info))
}

fn destroyed(js: &mut [JokerInstance], cards: &[Card], run: &RunView<'_>) {
    fire(js, run, |inp| fire_cards_destroyed(inp, cards));
}

fn round_end_with(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    deck: &Deck,
    seed: u64,
    won: bool,
) -> Vec<SideEffect> {
    let info = RoundEndInfo {
        won,
        saved: false,
        round_score: 400.0,
        target: 300.0,
    };
    fire_with(js, run, deck, seed, |inp| fire_round_end(inp, &info)).1
}

fn round_end(js: &mut [JokerInstance]) -> Vec<SideEffect> {
    round_end_with(js, &RunView::default(), &Deck::standard(), 3, true)
}

fn sell_other(js: &mut [JokerInstance], item: SoldItem, sold_slot: Option<usize>) {
    fire(js, &RunView::default(), |inp| {
        fire_sell_other(inp, &item, sold_slot)
    });
}

/// The joker at `slot` is sold: every other joker hears `on_sell_other`.
fn sell_joker_other(js: &mut [JokerInstance], slot: usize) {
    let item = SoldItem::Joker(js[slot]);
    sell_other(js, item, Some(slot));
}

fn tarot() -> SoldItem {
    SoldItem::Consumable(ConsumableView {
        kind: ConsumableKind::Tarot,
        id: 0,
        planet_hand: None,
        negative: false,
    })
}

/// A deck holding exactly these cards (all in the draw pile).
fn deck_of(cards: &[Card]) -> Deck {
    let mut d = Deck::standard();
    d.draw_pile = cards.to_vec();
    d.hand.clear();
    d.played.clear();
    d.discard_pile.clear();
    d
}

fn acquire(js: &mut [JokerInstance], slot: usize, deck: &Deck, seed: u64) {
    fire_with(js, &RunView::default(), deck, seed, |inp| {
        fire_acquire(inp, slot)
    });
}

fn destroy_of(uid: u32) -> SideEffect {
    SideEffect::DestroyJoker { uid }
}

// ----- Game-level helpers -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn select(g: &mut Game) -> StepInfo {
    g.step(Action::SelectBlind).expect("select blind")
}

/// Beat the current blind: round score one chip below the target, then play card 0.
fn beat_blind(g: &mut Game) -> StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("room")
}

/// Overwrite hand card `i` with a plain card of this rank and suit.
fn set_hand(g: &mut Game, i: usize, rank: Rank, suit: Suit) {
    let h = &mut g.state_mut().deck.hand[i];
    h.rank = rank;
    h.suit = suit;
    h.enhancement = Enhancement::None;
}

fn play0(g: &mut Game) -> (f64, f64) {
    let h = g
        .step(Action::Play(0b1))
        .expect("play")
        .hand
        .expect("hand report");
    (h.chips, h.mult)
}

// =============================================================================================
// Registry: definitions
// =============================================================================================

#[test]
fn all_fourteen_defs_exist_with_wiki_stats() {
    // (id, rarity, cost, eternal ok, perishable ok), balatrowiki.org pages / jokers.csv.
    let table = [
        (J::FlashCard, Rarity::Uncommon, 5, true, false),
        (J::Popcorn, Rarity::Common, 5, false, true),
        (J::SpareTrousers, Rarity::Uncommon, 6, true, false),
        (J::AncientJoker, Rarity::Rare, 8, true, true),
        (J::Ramen, Rarity::Uncommon, 6, false, true),
        (J::Castle, Rarity::Uncommon, 6, true, false),
        (J::Campfire, Rarity::Rare, 9, true, true),
        (J::Throwback, Rarity::Uncommon, 6, true, true),
        (J::GlassJoker, Rarity::Uncommon, 6, true, false),
        (J::WeeJoker, Rarity::Rare, 8, true, false),
        (J::TheIdol, Rarity::Uncommon, 6, true, true),
        (J::HitTheRoad, Rarity::Rare, 8, true, true),
        (J::Canio, Rarity::Legendary, 20, true, true),
        (J::Yorick, Rarity::Legendary, 20, true, true),
    ];
    for (id, rarity, cost, eternal, perishable) in table {
        let d = try_def(id).unwrap_or_else(|| panic!("{id:?} has no def"));
        assert_eq!(d.id, id);
        assert_eq!(d.rarity, rarity, "{id:?} rarity");
        assert_eq!(d.base_cost, cost, "{id:?} cost");
        assert_eq!(d.eternal_compatible, eternal, "{id:?} eternal");
        assert_eq!(d.perishable_compatible, perishable, "{id:?} perishable");
        assert!(d.blueprint_compatible, "{id:?} is copyable (jokers.csv)");
        assert!(d.copies.is_none());
    }
}

#[test]
fn initial_state_matches_wiki() {
    let init = |id| jk(0, id);
    assert_eq!(init(J::FlashCard).value, 0.0);
    assert_eq!(init(J::Popcorn).value, 20.0);
    assert_eq!(init(J::SpareTrousers).value, 0.0);
    assert_eq!(init(J::Ramen).value, 2.0);
    assert_eq!(init(J::Castle).value, 0.0);
    assert_eq!(init(J::Campfire).value, 1.0);
    assert_eq!(init(J::GlassJoker).value, 1.0);
    assert_eq!(init(J::WeeJoker).value, 0.0);
    assert_eq!(init(J::HitTheRoad).value, 1.0);
    assert_eq!(init(J::Canio).value, 1.0);
    let y = init(J::Yorick);
    assert_eq!((y.value, y.counter), (1.0, 23));
    // Sell value = floor(cost / 2): Campfire $9 -> 4, Canio $20 -> 10.
    assert_eq!(init(J::Campfire).sell_value, 4);
    assert_eq!(init(J::Canio).sell_value, 10);
}

// =============================================================================================
// Flash Card
// =============================================================================================

#[test]
fn flash_card_scales_two_per_reroll() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::FlashCard]);
    // No reroll yet: +0 mult -> 24 x 2 = 48.
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    // Three rerolls: +6 mult -> 24 x (2 + 6) = 192.
    for _ in 0..3 {
        reroll(&mut js);
    }
    assert_eq!(js[0].value, 6.0);
    assert_eq!(score(&p, e, &mut js).score, 192.0);
    // Scoring does not change the counter.
    assert_eq!(js[0].value, 6.0);
}

#[test]
fn flash_card_ignores_other_events_and_debuffed_is_inert() {
    let mut js = jokers(&[J::FlashCard]);
    discard(&mut js, &[c(1, Two, Hearts)]);
    round_end(&mut js);
    sell_other(&mut js, tarot(), None);
    assert_eq!(js[0].value, 0.0);
    let mut js = vec![jk(1, J::FlashCard).with_debuff(true)];
    reroll(&mut js);
    assert_eq!(js[0].value, 0.0);
}

#[test]
fn flash_card_blueprint_copy_neither_doubles_nor_loses() {
    // [Blueprint, Flash Card]: one reroll. The copy's write is discarded -> value 2 (not 4).
    let mut js = jokers(&[J::Blueprint, J::FlashCard]);
    reroll(&mut js);
    assert_eq!(js[1].value, 2.0);
    // Both pay the target's value: mult 2 + 2 + 2 = 6 -> 24 x 6 = 144.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 144.0);
    assert_eq!(js[1].value, 2.0);
}

// =============================================================================================
// Popcorn
// =============================================================================================

#[test]
fn popcorn_pays_20_and_decays_4_per_round_until_eaten() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Popcorn]);
    // +20 mult: 24 x 22 = 528.
    assert_eq!(score(&p, e, &mut js).score, 528.0);
    // Round ends: 20 -> 16 -> 12 -> 8 -> 4, none destroys it.
    for expect in [16.0, 12.0, 8.0, 4.0] {
        let fx = round_end(&mut js);
        assert_eq!(js[0].value, expect);
        assert!(fx.is_empty(), "no destruction at {expect}");
    }
    // At 4: 24 x (2 + 4) = 144.
    assert_eq!(score(&p, e, &mut js).score, 144.0);
    // Fifth round end: 4 - 4 = 0 -> eaten.
    let fx = round_end(&mut js);
    assert_eq!(js[0].value, 0.0);
    assert_eq!(fx, vec![destroy_of(100)]);
}

#[test]
fn popcorn_not_affected_by_hands_or_discards_and_not_eternal() {
    let mut js = jokers(&[J::Popcorn]);
    discard(&mut js, &[c(1, Two, Hearts)]);
    reroll(&mut js);
    let (p, e) = pair7();
    score(&p, e, &mut js);
    assert_eq!(js[0].value, 20.0);
    assert!(!try_def(J::Popcorn).expect("def").eternal_compatible);
}

#[test]
fn popcorn_blueprint_copies_value_not_decay() {
    // [Blueprint, Popcorn]: 2 + 20 + 20 = 42 -> 24 x 42 = 1008.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::Popcorn]);
    assert_eq!(score(&p, e, &mut js).score, 1008.0);
    // The round end is not copied: Popcorn decays once (16), the copy sees 16 afterwards,
    // and Blueprint itself is untouched. 2 + 16 + 16 = 34 -> 24 x 34 = 816.
    let fx = round_end(&mut js);
    assert!(fx.is_empty());
    assert_eq!(js[1].value, 16.0);
    assert_eq!(score(&p, e, &mut js).score, 816.0);
}

#[test]
fn popcorn_through_the_game_lives_exactly_five_rounds() {
    let mut g = new_game(21);
    add(&mut g, J::Popcorn);
    for (round, expect) in [16.0, 12.0, 8.0, 4.0].into_iter().enumerate() {
        if g.state().phase == Phase::Shop {
            g.step(Action::EndShop).expect("end shop");
        }
        select(&mut g);
        let info = beat_blind(&mut g);
        if round == 0 {
            // High Card 1 mult + 20 (the hand is paid before the round-end decay).
            assert_eq!(info.hand.expect("hand").mult, 21.0);
        }
        assert!(info.round.expect("round ended").won);
        assert_eq!(
            g.state().jokers[0].value,
            expect,
            "after round {}",
            round + 1
        );
    }
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    beat_blind(&mut g);
    // 4 - 4 = 0: eaten at the fifth round end (Small, Big, Boss, Small, Big).
    assert!(g.state().jokers.is_empty());
    assert_eq!(g.state().stats.rounds_won, 5);
}

// =============================================================================================
// Spare Trousers
// =============================================================================================

#[test]
fn spare_trousers_scales_on_two_pair_and_pays_the_same_hand() {
    let (tp, e) = two_pair();
    let mut js = jokers(&[J::SpareTrousers]);
    // 1st Two Pair: value 0 -> 2 before scoring: 52 x (2 + 2) = 208.
    assert_eq!(score(&tp, e, &mut js).score, 208.0);
    assert_eq!(js[0].value, 2.0);
    // 2nd: value 4: 52 x 6 = 312.
    assert_eq!(score(&tp, e, &mut js).score, 312.0);
    assert_eq!(js[0].value, 4.0);
    // A Pair does not grow it but pays the stored +4: 24 x (2 + 4) = 144.
    let (p, pe) = pair7();
    assert_eq!(score(&p, pe, &mut js).score, 144.0);
    assert_eq!(js[0].value, 4.0);
}

#[test]
fn spare_trousers_counts_hands_that_contain_two_pair() {
    // Full House 7S 7H 7C 9D 9S contains Two Pair: base 40 x 4, +21 +18 = 79 chips.
    // value 0 -> 2: mult 4 + 2 = 6 -> 79 x 6 = 474.
    let fh = vec![
        c(1, Seven, Spades),
        c(2, Seven, Hearts),
        c(3, Seven, Clubs),
        c(4, Nine, Diamonds),
        c(5, Nine, Spades),
    ];
    let e = ev(
        HandType::FullHouse,
        0b11111,
        &[HandType::Pair, HandType::TwoPair, HandType::ThreeOfAKind],
    );
    let mut js = jokers(&[J::SpareTrousers]);
    assert_eq!(score(&fh, e, &mut js).score, 474.0);
    assert_eq!(js[0].value, 2.0);

    // A Flush without two pairs, a plain Pair and High Card never grow it.
    let flush = vec![
        c(1, Two, Hearts),
        c(2, Five, Hearts),
        c(3, Seven, Hearts),
        c(4, Nine, Hearts),
        c(5, King, Hearts),
    ];
    let mut js = jokers(&[J::SpareTrousers]);
    score(&flush, ev(HandType::Flush, 0b11111, &[]), &mut js);
    let (p, pe) = pair7();
    score(&p, pe, &mut js);
    assert_eq!(js[0].value, 0.0);
}

#[test]
fn spare_trousers_blueprint_copy_scales_once() {
    // [Blueprint, Spare Trousers] on Two Pair: before_scoring runs on the copy (discarded) and
    // on the real joker (0 -> 2). Both pay +2: 52 x (2 + 2 + 2) = 312; the counter is 2, not 4.
    let (tp, e) = two_pair();
    let mut js = jokers(&[J::Blueprint, J::SpareTrousers]);
    assert_eq!(score(&tp, e, &mut js).score, 312.0);
    assert_eq!(js[1].value, 2.0);
}

// =============================================================================================
// Ancient Joker
// =============================================================================================

fn ancient_with(suit: Suit) -> JokerInstance {
    let mut j = jk(100, J::AncientJoker);
    j.set_aux_suit(Some(suit));
    j
}

#[test]
fn ancient_joker_xmult_on_matching_scoring_cards() {
    // Target Hearts. 7S 7H: only the Heart gives x1.5: 24 x (2 x 1.5) = 72.
    let (p, e) = pair7();
    let mut js = vec![ancient_with(Hearts)];
    assert_eq!(score(&p, e, &mut js).score, 72.0);
    // Target Diamonds: neither card matches -> 48.
    let mut js = vec![ancient_with(Diamonds)];
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    // Target Spades: a Wild 7C matches every suit, the 7H does not -> 24 x (2 x 1.5) = 72;
    // two Wild 7s -> 24 x (2 x 1.5 x 1.5) = 108.
    let one_wild = vec![
        enh(c(1, Seven, Clubs), Enhancement::Wild),
        c(2, Seven, Hearts),
    ];
    let mut js = vec![ancient_with(Spades)];
    assert_eq!(score(&one_wild, e, &mut js).score, 72.0);
    let two_wild = vec![
        enh(c(1, Seven, Clubs), Enhancement::Wild),
        enh(c(2, Seven, Hearts), Enhancement::Wild),
    ];
    assert_eq!(score(&two_wild, e, &mut js).score, 108.0);
}

#[test]
fn ancient_joker_skips_debuffed_cards_and_retriggers() {
    // Debuffed 7H gives nothing (and no chips): 7S only. Target Spades: chips 10 + 7 = 17,
    // mult 2 x 1.5 = 3 -> 51.
    let (p, e) = pair7();
    let dp = vec![p[0], debuff(p[1])];
    let mut js = vec![ancient_with(Spades)];
    assert_eq!(score(&dp, e, &mut js).score, 51.0);
    // Hack retriggers 2s: 2H 2H, target Hearts: each card triggers twice -> x1.5 four times.
    // chips 10 + 4 x 2 = 18; mult 2 x 1.5^4 = 10.125 -> floor(182.25) = 182.
    let twos = vec![c(1, Two, Hearts), c(2, Two, Hearts)];
    let mut js = vec![jk(101, J::Hack), ancient_with(Hearts)];
    assert_eq!(
        score(&twos, ev(HandType::Pair, 0b11, &[]), &mut js).score,
        182.0
    );
}

#[test]
fn ancient_joker_picks_any_suit_on_acquire() {
    let deck = Deck::standard();
    let mut seen = [false; 4];
    for seed in 0..64 {
        let mut js = jokers(&[J::AncientJoker]);
        acquire(&mut js, 0, &deck, seed);
        seen[js[0].aux_suit().expect("a suit was chosen").index()] = true;
    }
    assert_eq!(seen, [true; 4], "all four suits reachable on acquire");
}

#[test]
fn ancient_joker_changes_to_a_different_suit_every_round() {
    let deck = Deck::standard();
    let mut reached = [[false; 4]; 4]; // reached[from][to]
    for seed in 0..64 {
        let mut js = jokers(&[J::AncientJoker]);
        acquire(&mut js, 0, &deck, seed);
        let mut cur = js[0].aux_suit().expect("suit");
        for round in 0..4 {
            round_end_with(&mut js, &RunView::default(), &deck, seed * 7 + round, true);
            let next = js[0].aux_suit().expect("suit");
            assert_ne!(next, cur, "the suit never repeats (seed {seed})");
            reached[cur.index()][next.index()] = true;
            cur = next;
        }
    }
    // Every transition to one of the other 3 suits occurs.
    for (from, row) in reached.iter().enumerate() {
        for (to, &hit) in row.iter().enumerate() {
            assert_eq!(hit, from != to, "{from} -> {to}");
        }
    }
}

#[test]
fn ancient_joker_choice_is_deterministic_per_seed() {
    let deck = Deck::standard();
    let run_once = |seed| {
        let mut js = jokers(&[J::AncientJoker]);
        acquire(&mut js, 0, &deck, seed);
        let a = js[0].aux_suit();
        round_end_with(&mut js, &RunView::default(), &deck, seed, true);
        (a, js[0].aux_suit())
    };
    for seed in 0..16 {
        assert_eq!(run_once(seed), run_once(seed));
    }
}

#[test]
fn ancient_joker_blueprint_does_not_change_the_target() {
    // Acquire of the Blueprint (slot 0) does not touch the Ancient Joker behind it, and a
    // second acquire of an already-targeted Ancient keeps its suit.
    let deck = Deck::standard();
    let mut js = vec![jk(100, J::Blueprint), ancient_with(Hearts)];
    acquire(&mut js, 0, &deck, 5);
    assert_eq!(js[1].aux_suit(), Some(Hearts));
    acquire(&mut js, 1, &deck, 5);
    assert_eq!(
        js[1].aux_suit(),
        Some(Hearts),
        "acquire keeps a chosen target"
    );
    // The copy gives the same x1.5 on Hearts: 7S 7H -> 24 x (2 x 1.5 x 1.5) = 108.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 108.0);
    // Round end re-rolls only the real joker; the Blueprint has no suit of its own.
    round_end_with(&mut js, &RunView::default(), &deck, 9, true);
    assert_ne!(js[1].aux_suit(), Some(Hearts));
    assert_eq!(js[0].aux_suit(), None);
}

#[test]
fn ancient_joker_in_the_game_rerolls_each_round_and_pays_x1_5() {
    let mut g = new_game(41);
    add(&mut g, J::AncientJoker);
    let mut cur = g.state().jokers[0].aux_suit().expect("chosen on acquire");
    for _ in 0..3 {
        if g.state().phase == Phase::Shop {
            g.step(Action::EndShop).expect("end shop");
        }
        select(&mut g);
        let other = Suit::ALL
            .into_iter()
            .find(|s| *s != cur)
            .expect("other suit");
        // A single card is High Card (5 + chips, mult 1): matching suit -> x1.5, else x1.
        set_hand(&mut g, 0, Ace, other);
        let (_, mult) = {
            let mut probe = g.clone();
            play0(&mut probe)
        };
        assert_eq!(mult, 1.0);
        set_hand(&mut g, 0, Ace, cur);
        assert_eq!(play0(&mut g).1, 1.5);
        let info = beat_blind(&mut g);
        assert!(info.round.expect("round").won);
        let next = g.state().jokers[0].aux_suit().expect("suit");
        assert_ne!(next, cur);
        cur = next;
    }
}

// =============================================================================================
// Ramen
// =============================================================================================

fn cards(n: usize) -> Vec<Card> {
    (0..n as u32).map(|i| c(i, Two, Hearts)).collect()
}

#[test]
fn ramen_starts_x2_and_loses_001_per_discarded_card() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Ramen]);
    // x2: 24 x (2 x 2) = 96.
    assert_eq!(score(&p, e, &mut js).score, 96.0);
    // Discard 5 cards: 2.00 - 0.05 = 1.95 -> 24 x (2 x 1.95) = 93.6 -> 93.
    let fx = discard(&mut js, &cards(5));
    assert!(fx.is_empty());
    assert_eq!(js[0].value, 1.95);
    assert_eq!(score(&p, e, &mut js).score, 93.0);
    // Per card, not per discard action: one more card -> 1.94.
    discard(&mut js, &cards(1));
    assert_eq!(js[0].value, 1.94);
}

#[test]
fn ramen_is_eaten_when_the_100th_card_would_bring_it_to_x1() {
    let mut js = jokers(&[J::Ramen]);
    // 99 cards: 2.00 - 0.99 = 1.01 (exactly, no float drift), still alive.
    let fx = discard(&mut js, &cards(99));
    assert!(fx.is_empty());
    assert_eq!(js[0].value, 1.01);
    // The 100th card would give 1.00 -> destroyed (value untouched).
    let fx = discard(&mut js, &cards(1));
    assert_eq!(fx, vec![destroy_of(100)]);
    assert_eq!(js[0].value, 1.01);
    // 100 cards in one discard destroy it just the same.
    let mut js = jokers(&[J::Ramen]);
    assert_eq!(discard(&mut js, &cards(100)), vec![destroy_of(100)]);
}

#[test]
fn ramen_ignores_non_discard_events_and_is_not_eternal() {
    let mut js = jokers(&[J::Ramen]);
    reroll(&mut js);
    round_end(&mut js);
    sell_other(&mut js, tarot(), None);
    let (p, e) = pair7();
    score(&p, e, &mut js);
    assert_eq!(js[0].value, 2.0);
    assert!(!try_def(J::Ramen).expect("def").eternal_compatible);
    // A debuffed Ramen neither pays nor decays.
    let mut js = vec![jk(1, J::Ramen).with_debuff(true)];
    discard(&mut js, &cards(5));
    assert_eq!(js[0].value, 2.0);
    assert_eq!(score(&p, e, &mut js).score, 48.0);
}

#[test]
fn ramen_blueprint_copy_decays_once_and_never_destroys_its_target_twice() {
    let mut js = jokers(&[J::Blueprint, J::Ramen]);
    // 5 discarded cards: the copy's write is discarded, the real Ramen drops once to 1.95.
    discard(&mut js, &cards(5));
    assert_eq!(js[1].value, 1.95);
    // Copy pays too: mult 2 x 1.95 x 1.95 = 7.605 -> 24 x 7.605 = 182.52 -> 182.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 182.0);
    // Fatal discard: exactly one DestroyJoker (the real Ramen's), none from the copy.
    js[1].value = 1.01;
    let fx = discard(&mut js, &cards(1));
    assert_eq!(fx, vec![destroy_of(101)]);
}

#[test]
fn ramen_in_the_game_decays_on_discard_and_pays_next_hand() {
    let mut g = new_game(51);
    add(&mut g, J::Ramen);
    select(&mut g);
    // 5 cards discarded (a real step): 2.00 -> 1.95; High Card mult 1 x 1.95.
    g.step(Action::Discard(0b1_1111)).expect("discard");
    assert_eq!(g.state().jokers[0].value, 1.95);
    // Playing cards does not decay it (unscored played cards are not discards).
    assert_eq!(play0(&mut g).1, 1.95);
    assert_eq!(g.state().jokers[0].value, 1.95);
}

// =============================================================================================
// Castle
// =============================================================================================

fn castle_with(suit: Suit) -> JokerInstance {
    let mut j = jk(100, J::Castle);
    j.set_aux_suit(Some(suit));
    j
}

#[test]
fn castle_gains_3_chips_per_discarded_card_of_the_target_suit() {
    let mut js = vec![castle_with(Clubs)];
    // Discard 2C 5C 9H: two Clubs -> +6.
    discard(
        &mut js,
        &[c(1, Two, Clubs), c(2, Five, Clubs), c(3, Nine, Hearts)],
    );
    assert_eq!(js[0].value, 6.0);
    // Pair of 7s: (10 + 14 + 6) x 2 = 60.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 60.0);
    // One more Club later: 6 + 3 = 9 -> (24 + 9) x 2 = 66.
    discard(&mut js, &[c(4, King, Clubs)]);
    assert_eq!(js[0].value, 9.0);
    assert_eq!(score(&p, e, &mut js).score, 66.0);
}

#[test]
fn castle_ignores_other_suits_debuffed_cards_and_other_events() {
    let mut js = vec![castle_with(Clubs)];
    discard(&mut js, &[c(1, Two, Hearts), c(2, Five, Spades)]);
    discard(&mut js, &[debuff(c(3, Two, Clubs))]);
    reroll(&mut js);
    assert_eq!(js[0].value, 0.0);
    // Wild counts as the target suit.
    discard(&mut js, &[enh(c(4, Two, Hearts), Enhancement::Wild)]);
    assert_eq!(js[0].value, 3.0);
    // Playing cards in a hand never counts as discarding.
    let (p, e) = pair7();
    score(&p, e, &mut js);
    assert_eq!(js[0].value, 3.0);
}

#[test]
fn castle_target_follows_the_deck_composition() {
    // Acquire from a deck of only Diamonds / only Stone cards (-> Spades fallback).
    let diamonds = deck_of(&[c(0, Two, Diamonds), c(1, Nine, Diamonds)]);
    let stone = deck_of(&[enh(c(0, Two, Hearts), Enhancement::Stone)]);
    for seed in 0..8 {
        let mut js = jokers(&[J::Castle]);
        acquire(&mut js, 0, &diamonds, seed);
        assert_eq!(js[0].aux_suit(), Some(Diamonds));
        let mut js = jokers(&[J::Castle]);
        acquire(&mut js, 0, &stone, seed);
        assert_eq!(js[0].aux_suit(), Some(Spades));
    }
    // Weighted: 3 Clubs : 1 Spade -> Clubs about 75% of the time, Spades sometimes.
    let mixed = deck_of(&[
        c(0, Two, Clubs),
        c(1, Three, Clubs),
        c(2, Four, Clubs),
        c(3, Five, Spades),
    ]);
    let (mut clubs, mut spades) = (0, 0);
    for seed in 0..200 {
        let mut js = jokers(&[J::Castle]);
        acquire(&mut js, 0, &mixed, seed);
        match js[0].aux_suit() {
            Some(Clubs) => clubs += 1,
            Some(Spades) => spades += 1,
            other => panic!("suit {other:?} is not in the deck"),
        }
    }
    assert!(
        spades > 20 && clubs > 2 * spades,
        "{clubs} clubs / {spades} spades"
    );
}

#[test]
fn castle_rerolls_target_each_round_and_keeps_chips() {
    let diamonds = deck_of(&[c(0, Two, Diamonds)]);
    let mut js = vec![castle_with(Clubs)];
    discard(&mut js, &[c(1, Two, Clubs)]);
    round_end_with(&mut js, &RunView::default(), &diamonds, 1, true);
    assert_eq!(js[0].aux_suit(), Some(Diamonds));
    assert_eq!(js[0].value, 3.0, "chips are permanent");
}

#[test]
fn castle_blueprint_copy_counts_discards_once() {
    // [Blueprint, Castle(Clubs)]: two Clubs discarded -> +6 (not +12).
    let mut bp = jokers(&[J::Blueprint]);
    bp.push(castle_with(Clubs));
    discard(&mut bp, &[c(1, Two, Clubs), c(2, Five, Clubs)]);
    assert_eq!(bp[1].value, 6.0);
    // Both pay +6 chips: (10 + 14 + 6 + 6) x 2 = 72.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut bp).score, 72.0);
}

#[test]
fn castle_in_the_game_counts_real_discards() {
    let mut g = new_game(61);
    add(&mut g, J::Castle);
    let suit = g.state().jokers[0].aux_suit().expect("chosen on acquire");
    select(&mut g);
    let other = Suit::ALL
        .into_iter()
        .find(|s| *s != suit)
        .expect("other suit");
    for i in 0..3 {
        set_hand(&mut g, i, Five, suit);
    }
    set_hand(&mut g, 3, Five, other);
    // Discard hand[0..4]: three target-suit cards -> +9 (a real step fires the event).
    g.step(Action::Discard(0b1111)).expect("discard");
    assert_eq!(g.state().jokers[0].value, 9.0);
    // Next single-card hand: High Card 5 + card chips + 9 castle chips.
    let card = g.state().deck.hand[0];
    let (chips, mult) = play0(&mut g);
    assert_eq!(chips, 5.0 + f64::from(card.base_chips()) + 9.0);
    assert_eq!(mult, 1.0);
}

// =============================================================================================
// Campfire
// =============================================================================================

#[test]
fn campfire_gains_quarter_per_sold_card_and_resets_on_boss() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Campfire, J::Joker, J::Joker]);
    // Nothing sold: x1 -> 48.
    assert_eq!(score(&p, e, &mut js[..1]).score, 48.0);
    // Sell a joker (slot 1), then a consumable: 1 + 0.25 + 0.25 = 1.5 -> mult 2 x 1.5 = 3 -> 72.
    sell_joker_other(&mut js, 1);
    assert_eq!(js[0].value, 1.25);
    sell_other(&mut js, tarot(), None);
    assert_eq!(js[0].value, 1.5);
    assert_eq!(score(&p, e, &mut js[..1]).score, 72.0);
    // Two more sales: 2.0 -> mult 4 -> 96.
    sell_other(&mut js, tarot(), None);
    sell_other(&mut js, tarot(), None);
    assert_eq!(score(&p, e, &mut js[..1]).score, 96.0);

    // A non-boss round end (Small/Big) or a lost boss round keeps the stacks.
    for (blind, won) in [
        (BlindKind::Small, true),
        (BlindKind::Big, true),
        (BlindKind::Boss, false),
    ] {
        let run = RunView {
            blind,
            ..RunView::default()
        };
        round_end_with(&mut js, &run, &Deck::standard(), 1, won);
        assert_eq!(js[0].value, 2.0, "{blind:?} won={won}");
    }
    // Boss defeated: back to x1.
    let run = RunView {
        blind: BlindKind::Boss,
        ..RunView::default()
    };
    round_end_with(&mut js, &run, &Deck::standard(), 1, true);
    assert_eq!(js[0].value, 1.0);
    assert_eq!(score(&p, e, &mut js[..1]).score, 48.0);
}

#[test]
fn campfire_does_not_count_its_own_sale_and_ignores_other_events() {
    let mut js = jokers(&[J::Campfire, J::Joker]);
    // Selling the Campfire itself: it is the sold slot, so it is skipped.
    sell_joker_other(&mut js, 0);
    assert_eq!(js[0].value, 1.0);
    reroll(&mut js);
    discard(&mut js, &[c(1, Two, Hearts)]);
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn campfire_blueprint_copy_counts_each_sale_once() {
    // [Blueprint, Campfire, Joker]: sell the Joker (slot 2) -> Campfire 1.25 (not 1.5).
    let mut js = jokers(&[J::Blueprint, J::Campfire, J::Joker]);
    sell_joker_other(&mut js, 2);
    assert_eq!(js[1].value, 1.25);
    // Blueprint copies the xmult as well: mult 2 x 1.25 x 1.25 = 3.125 -> 24 x 3.125 = 75.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js[..2]).score, 75.0);
    // Boss reset is a round-end effect and is not copied; the real Campfire resets once.
    let run = RunView {
        blind: BlindKind::Boss,
        ..RunView::default()
    };
    round_end_with(&mut js, &run, &Deck::standard(), 1, true);
    assert_eq!(js[1].value, 1.0);
}

#[test]
fn campfire_in_the_game_sells_stack_and_boss_resets() {
    let mut g = new_game(71);
    add(&mut g, J::Campfire);
    add(&mut g, J::Joker);
    add(&mut g, J::Joker);
    select(&mut g);
    // Selling in Play: Joker (slot 1) -> 1.25, Joker (now slot 1) -> 1.5. The sold jokers
    // are removed, Campfire stays at slot 0.
    g.step(Action::SellJoker(1)).expect("sell");
    assert_eq!(g.state().jokers[0].value, 1.25);
    g.step(Action::SellJoker(1)).expect("sell");
    assert_eq!(g.state().jokers[0].value, 1.5);
    assert_eq!(g.state().jokers.len(), 1);
    // High Card mult 1 x 1.5.
    assert_eq!(play0(&mut g).1, 1.5);
    // Small and Big do not reset it.
    beat_blind(&mut g);
    assert_eq!(g.state().jokers[0].value, 1.5);
    g.step(Action::EndShop).expect("end shop");
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().jokers[0].value, 1.5);
    g.step(Action::EndShop).expect("end shop");
    // Boss round: still x1.5 while playing, reset when the boss is defeated.
    assert_eq!(g.state().blind, BlindKind::Boss);
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().jokers[0].value, 1.0);
    assert_eq!(g.state().stats.run.bosses_defeated, 1);
}

// =============================================================================================
// Throwback
// =============================================================================================

fn skipped(n: u32) -> RunView<'static> {
    RunView {
        stats: RunStats {
            blinds_skipped: n,
            ..RunStats::default()
        },
        ..RunView::default()
    }
}

#[test]
fn throwback_is_x_one_plus_quarter_per_skipped_blind() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Throwback]);
    // 0 skips: x1 -> 48. 2 skips: x1.5 -> mult 3 -> 72. 4 skips: x2 -> 96.
    assert_eq!(score_with(&p, e, &mut js, &skipped(0)).score, 48.0);
    assert_eq!(score_with(&p, e, &mut js, &skipped(2)).score, 72.0);
    assert_eq!(score_with(&p, e, &mut js, &skipped(4)).score, 96.0);
}

#[test]
fn throwback_blueprint_copy_reads_the_same_run_total() {
    // 2 skips: each copy x1.5 -> mult 2 x 1.5 x 1.5 = 4.5 -> 24 x 4.5 = 108.
    let (p, e) = pair7();
    let mut js = jokers(&[J::Blueprint, J::Throwback]);
    assert_eq!(score_with(&p, e, &mut js, &skipped(2)).score, 108.0);
}

#[test]
fn throwback_through_the_game_counts_skips_before_and_after_acquire() {
    let mut g = new_game(81);
    g.step(Action::SkipBlind).expect("skip small");
    assert_eq!(g.state().stats.run.blinds_skipped, 1);
    // Acquired after the first skip: it still counts (run total).
    add(&mut g, J::Throwback);
    g.step(Action::SkipBlind).expect("skip big");
    assert_eq!(g.state().stats.run.blinds_skipped, 2);
    assert_eq!(g.state().blind, BlindKind::Boss);
    select(&mut g);
    set_hand(&mut g, 0, Ace, Spades);
    // High Card: chips 5 + 11 = 16, mult 1 x (1 + 0.25 x 2) = 1.5 -> score 24.
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("hand");
    assert_eq!((h.chips, h.mult, h.score), (16.0, 1.5, 24.0));
}

// =============================================================================================
// Glass Joker
// =============================================================================================

fn glass(uid: u32, rank: Rank, suit: Suit) -> Card {
    enh(c(uid, rank, suit), Enhancement::Glass)
}

#[test]
fn glass_joker_gains_075_per_destroyed_glass_card_only() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::GlassJoker]);
    // Non-Glass cards destroyed: nothing. Score stays 48.
    destroyed(
        &mut js,
        &[
            c(1, King, Spades),
            enh(c(2, Two, Hearts), Enhancement::Steel),
        ],
        &RunView::default(),
    );
    assert_eq!(js[0].value, 1.0);
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    // Two Glass cards among three: 1 + 2 x 0.75 = 2.5 -> mult 2 x 2.5 = 5 -> 120.
    destroyed(
        &mut js,
        &[
            glass(3, Two, Spades),
            c(4, King, Hearts),
            glass(5, King, Clubs),
        ],
        &RunView::default(),
    );
    assert_eq!(js[0].value, 2.5);
    assert_eq!(score(&p, e, &mut js).score, 120.0);
}

#[test]
fn glass_joker_ignores_other_events() {
    let mut js = jokers(&[J::GlassJoker]);
    discard(&mut js, &[glass(1, Two, Spades)]);
    reroll(&mut js);
    round_end(&mut js);
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn glass_joker_blueprint_copy_counts_once() {
    // [Blueprint, Glass Joker]: one Glass destroyed -> 1.75 (not 2.5).
    let mut js = jokers(&[J::Blueprint, J::GlassJoker]);
    destroyed(&mut js, &[glass(1, Two, Spades)], &RunView::default());
    assert_eq!(js[1].value, 1.75);
    // Mult 2 x 1.75 x 1.75 = 6.125 -> 24 x 6.125 = 147.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 147.0);
}

// =============================================================================================
// Wee Joker
// =============================================================================================

#[test]
fn wee_joker_gains_8_chips_per_scoring_two_and_pays_the_same_hand() {
    let (p, e) = pair2();
    let mut js = jokers(&[J::WeeJoker]);
    // Each scoring 2 adds +8 as it scores (before the joker pays): chips 10 + 2 + 2 = 14 from
    // the pair, joker pays 16 -> 30 x 2 = 60.
    assert_eq!(score(&p, e, &mut js).score, 60.0);
    assert_eq!(js[0].value, 16.0);
    // Second hand: value 32 -> (14 + 32) x 2 = 92.
    assert_eq!(score(&p, e, &mut js).score, 92.0);
    assert_eq!(js[0].value, 32.0);
    // A hand without scoring 2s keeps the stack and pays it: (10 + 14 + 32) x 2 = 112.
    let (p7, e7) = pair7();
    assert_eq!(score(&p7, e7, &mut js).score, 112.0);
    assert_eq!(js[0].value, 32.0);
}

#[test]
fn wee_joker_counts_only_scoring_undebuffed_twos() {
    // A 2 kicker that does not score (mask 0b011 = the two 7s) gives nothing.
    let hand = vec![c(1, Seven, Spades), c(2, Seven, Hearts), c(3, Two, Clubs)];
    let mut js = jokers(&[J::WeeJoker]);
    assert_eq!(
        score(&hand, ev(HandType::Pair, 0b011, &[]), &mut js).score,
        48.0
    );
    assert_eq!(js[0].value, 0.0);
    // A debuffed 2 does not score: only 2H counts -> +8; chips 10 + 2 + 8 = 20 x 2 = 40.
    let (p, e) = pair2();
    let dp = vec![debuff(p[0]), p[1]];
    let mut js = jokers(&[J::WeeJoker]);
    assert_eq!(score(&dp, e, &mut js).score, 40.0);
    assert_eq!(js[0].value, 8.0);
}

#[test]
fn wee_joker_counts_retriggers() {
    // Hack retriggers 2s: 2S 2H each score twice = 4 triggers -> +32.
    // Chips: 10 + 4 x 2 (rank) + 32 = 50, mult 2 -> 100.
    let (p, e) = pair2();
    let mut js = jokers(&[J::Hack, J::WeeJoker]);
    assert_eq!(score(&p, e, &mut js).score, 100.0);
    assert_eq!(js[1].value, 32.0);
}

#[test]
fn wee_joker_blueprint_copy_neither_doubles_nor_loses_the_stack() {
    // [Blueprint, Wee]: the real joker ends at 16 (the copy's writes are discarded); both pay
    // 16: chips 10 + 2 + 2 + 16 + 16 = 46 x 2 = 92.
    let (p, e) = pair2();
    let mut js = jokers(&[J::Blueprint, J::WeeJoker]);
    assert_eq!(score(&p, e, &mut js).score, 92.0);
    assert_eq!(js[1].value, 16.0);
}

#[test]
fn wee_joker_in_the_game_scales_with_played_twos() {
    let mut g = new_game(91);
    add(&mut g, J::WeeJoker);
    select(&mut g);
    // High Card with a lone 2: scoring 2 -> +8 first, then the joker pays it:
    // chips 5 + 2 + 8 = 15.
    set_hand(&mut g, 0, Two, Hearts);
    assert_eq!(play0(&mut g).0, 15.0);
    assert_eq!(g.state().jokers[0].value, 8.0);
    // A non-2 card: chips 5 + 9 + 8 = 22.
    set_hand(&mut g, 0, Nine, Hearts);
    assert_eq!(play0(&mut g).0, 22.0);
    assert_eq!(g.state().jokers[0].value, 8.0);
}

// =============================================================================================
// The Idol
// =============================================================================================

fn idol_with(rank: Rank, suit: Suit) -> JokerInstance {
    let mut j = jk(100, J::TheIdol);
    j.set_aux_rank(Some(rank));
    j.set_aux_suit(Some(suit));
    j
}

#[test]
fn idol_xmult_on_scoring_cards_matching_rank_and_suit() {
    let (p, e) = pair7();
    // Target 7H: only 7H gives x2 -> 24 x (2 x 2) = 96.
    let mut js = vec![idol_with(Seven, Hearts)];
    assert_eq!(score(&p, e, &mut js).score, 96.0);
    // Target 7D: wrong suit; target 8H: wrong rank -> 48.
    let mut js = vec![idol_with(Seven, Diamonds)];
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    let mut js = vec![idol_with(Eight, Hearts)];
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    // A Wild 7 matches any suit: 7S(Wild) + 7H, target 7H -> both x2: 24 x (2 x 4) = 192.
    let wild = vec![enh(p[0], Enhancement::Wild), p[1]];
    let mut js = vec![idol_with(Seven, Hearts)];
    assert_eq!(score(&wild, e, &mut js).score, 192.0);
}

#[test]
fn idol_skips_debuffed_cards_and_retriggers() {
    // Debuffed 7H: no x2 (and no chips): chips 10 + 7, mult 2 -> 34.
    let (p, e) = pair7();
    let dp = vec![p[0], debuff(p[1])];
    let mut js = vec![idol_with(Seven, Hearts)];
    assert_eq!(score(&dp, e, &mut js).score, 34.0);
    // Two 2H with Hack: 4 triggers x2 = x16. chips 10 + 8 = 18, mult 2 x 16 = 32 -> 576.
    let twos = vec![c(1, Two, Hearts), c(2, Two, Hearts)];
    let mut js = vec![jk(101, J::Hack), idol_with(Two, Hearts)];
    assert_eq!(
        score(&twos, ev(HandType::Pair, 0b11, &[]), &mut js).score,
        576.0
    );
}

#[test]
fn idol_picks_a_random_non_stone_deck_card_from_every_pile() {
    // Cards only in the hand / discard pile are candidates too.
    let mut d = deck_of(&[]);
    d.hand = vec![c(0, King, Clubs)];
    d.discard_pile = vec![enh(c(1, Two, Hearts), Enhancement::Stone)];
    for seed in 0..8 {
        let mut js = jokers(&[J::TheIdol]);
        acquire(&mut js, 0, &d, seed);
        assert_eq!(
            (js[0].aux_rank(), js[0].aux_suit()),
            (Some(King), Some(Clubs))
        );
    }
    // A deck with only Stone cards (or empty): Ace of Spades.
    let stone = deck_of(&[enh(c(0, Two, Hearts), Enhancement::Stone)]);
    let mut js = jokers(&[J::TheIdol]);
    acquire(&mut js, 0, &stone, 1);
    assert_eq!(
        (js[0].aux_rank(), js[0].aux_suit()),
        (Some(Ace), Some(Spades))
    );
    let mut js = jokers(&[J::TheIdol]);
    acquire(&mut js, 0, &deck_of(&[]), 1);
    assert_eq!(
        (js[0].aux_rank(), js[0].aux_suit()),
        (Some(Ace), Some(Spades))
    );
}

#[test]
fn idol_target_is_weighted_by_the_deck_and_rerolled_each_round() {
    // 3 x AS : 1 x 2D -> Ace of Spades ~75%.
    let d = deck_of(&[
        c(0, Ace, Spades),
        c(1, Ace, Spades),
        c(2, Ace, Spades),
        c(3, Two, Diamonds),
    ]);
    let (mut aces, mut twos) = (0, 0);
    for seed in 0..200 {
        let mut js = jokers(&[J::TheIdol]);
        acquire(&mut js, 0, &d, seed);
        match (js[0].aux_rank(), js[0].aux_suit()) {
            (Some(Ace), Some(Spades)) => aces += 1,
            (Some(Two), Some(Diamonds)) => twos += 1,
            other => panic!("target {other:?} is not in the deck"),
        }
    }
    assert!(twos > 20 && aces > 2 * twos, "{aces} aces / {twos} twos");
    // Round end re-picks (here the deck only has 3C).
    let only = deck_of(&[c(0, Three, Clubs)]);
    let mut js = vec![idol_with(Seven, Hearts)];
    round_end_with(&mut js, &RunView::default(), &only, 4, true);
    assert_eq!(
        (js[0].aux_rank(), js[0].aux_suit()),
        (Some(Three), Some(Clubs))
    );
}

#[test]
fn idol_blueprint_copy_uses_the_target_and_never_changes_it() {
    // Acquiring keeps a chosen target; the Blueprint (slot 0) never rewrites it.
    let d = deck_of(&[c(0, Three, Clubs)]);
    let mut js = vec![jk(100, J::Blueprint), idol_with(Seven, Hearts)];
    acquire(&mut js, 0, &d, 1);
    acquire(&mut js, 1, &d, 1);
    assert_eq!(
        (js[1].aux_rank(), js[1].aux_suit()),
        (Some(Seven), Some(Hearts))
    );
    // Copy pays x2 as well: 24 x (2 x 2 x 2) = 192.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 192.0);
    // Round end re-rolls only the Idol itself.
    round_end_with(&mut js, &RunView::default(), &d, 1, true);
    assert_eq!(
        (js[1].aux_rank(), js[1].aux_suit()),
        (Some(Three), Some(Clubs))
    );
    assert_eq!((js[0].aux_rank(), js[0].aux_suit()), (None, None));
}

#[test]
fn idol_in_the_game_targets_a_deck_card_and_rerolls_each_round() {
    let mut g = new_game(101);
    add(&mut g, J::TheIdol);
    for _ in 0..3 {
        if g.state().phase == Phase::Shop {
            g.step(Action::EndShop).expect("end shop");
        }
        let (rank, suit) = {
            let j = &g.state().jokers[0];
            (j.aux_rank().expect("rank"), j.aux_suit().expect("suit"))
        };
        assert!(
            g.state()
                .deck
                .iter_all()
                .any(|c| c.rank == rank && c.suit == suit),
            "the target card exists in the deck"
        );
        select(&mut g);
        // High Card of the target card: mult 1 x 2; of another rank: x1.
        let wrong = if rank == Two { Three } else { Two };
        set_hand(&mut g, 0, wrong, suit);
        let mut probe = g.clone();
        assert_eq!(play0(&mut probe).1, 1.0);
        set_hand(&mut g, 0, rank, suit);
        assert_eq!(play0(&mut g).1, 2.0);
        beat_blind(&mut g);
    }
}

// =============================================================================================
// Hit the Road
// =============================================================================================

#[test]
fn hit_the_road_gains_half_per_jack_discarded_and_resets_each_round() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::HitTheRoad]);
    // Discard JS JH 5C: two Jacks -> 1 + 2 x 0.5 = 2.0 -> mult 2 x 2 = 4 -> 96.
    discard(
        &mut js,
        &[c(1, Jack, Spades), c(2, Jack, Hearts), c(3, Five, Clubs)],
    );
    assert_eq!(js[0].value, 2.0);
    assert_eq!(score(&p, e, &mut js).score, 96.0);
    // Another single Jack: 2.5 -> mult 5 -> 120.
    discard(&mut js, &[c(4, Jack, Diamonds)]);
    assert_eq!(js[0].value, 2.5);
    assert_eq!(score(&p, e, &mut js).score, 120.0);
    // Round end resets it (also after a loss) -> 48 again.
    let fx = round_end(&mut js);
    assert!(fx.is_empty());
    assert_eq!(js[0].value, 1.0);
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    discard(&mut js, &[c(5, Jack, Clubs)]);
    round_end_with(&mut js, &RunView::default(), &Deck::standard(), 1, false);
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn hit_the_road_only_counts_jacks() {
    let mut js = jokers(&[J::HitTheRoad]);
    // Queen, King, Ten, a Stone card with a printed Jack, and played cards: nothing.
    discard(
        &mut js,
        &[
            c(1, Queen, Spades),
            c(2, King, Hearts),
            c(3, Ten, Clubs),
            enh(c(4, Jack, Spades), Enhancement::Stone),
        ],
    );
    reroll(&mut js);
    let (p, e) = pair7();
    score(&p, e, &mut js);
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn hit_the_road_blueprint_copy_counts_once_and_reset_is_not_copied() {
    // [Blueprint, Hit the Road]: one Jack -> 1.5 (not 2.0). Copy pays too: mult
    // 2 x 1.5 x 1.5 = 4.5 -> 24 x 4.5 = 108.
    let mut js = jokers(&[J::Blueprint, J::HitTheRoad]);
    discard(&mut js, &[c(1, Jack, Spades)]);
    assert_eq!(js[1].value, 1.5);
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 108.0);
    round_end(&mut js);
    assert_eq!(js[1].value, 1.0);
}

#[test]
fn hit_the_road_in_the_game_resets_at_round_end() {
    let mut g = new_game(111);
    add(&mut g, J::HitTheRoad);
    select(&mut g);
    set_hand(&mut g, 0, Jack, Spades);
    set_hand(&mut g, 1, Jack, Hearts);
    g.step(Action::Discard(0b11)).expect("discard");
    assert_eq!(g.state().jokers[0].value, 2.0);
    // High Card mult 1 x 2.0.
    assert_eq!(play0(&mut g).1, 2.0);
    // Winning the round resets it (the winning hand itself was still paid x2).
    let info = beat_blind(&mut g);
    assert_eq!(info.hand.expect("hand").mult, 2.0);
    assert_eq!(g.state().jokers[0].value, 1.0);
}

// =============================================================================================
// Canio
// =============================================================================================

#[test]
fn canio_gains_one_per_destroyed_face_card() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Canio]);
    let run = RunView::default();
    // J, Q, K, 5: three faces -> 1 + 3 = 4.0 -> mult 2 x 4 = 8 -> 192. Non-faces: nothing.
    destroyed(
        &mut js,
        &[
            c(1, Jack, Spades),
            c(2, Queen, Hearts),
            c(3, King, Clubs),
            c(4, Five, Clubs),
        ],
        &run,
    );
    assert_eq!(js[0].value, 4.0);
    assert_eq!(score(&p, e, &mut js).score, 192.0);
    destroyed(&mut js, &[c(5, Ace, Spades), c(6, Ten, Hearts)], &run);
    assert_eq!(js[0].value, 4.0);
}

#[test]
fn canio_respects_pareidolia_stone_and_debuffs() {
    let mut js = jokers(&[J::Canio]);
    // Pareidolia: every card is a face card, Stone included -> +3.
    let pareidolia = RunView {
        rules: RuleFlags {
            pareidolia: true,
            ..RuleFlags::default()
        },
        ..RunView::default()
    };
    destroyed(
        &mut js,
        &[
            c(1, Two, Spades),
            c(2, Nine, Hearts),
            enh(c(3, Two, Clubs), Enhancement::Stone),
        ],
        &pareidolia,
    );
    assert_eq!(js[0].value, 4.0);
    // Without Pareidolia: a debuffed King and a Stone Jack are not faces.
    let mut js = jokers(&[J::Canio]);
    destroyed(
        &mut js,
        &[
            debuff(c(1, King, Spades)),
            enh(c(2, Jack, Hearts), Enhancement::Stone),
        ],
        &RunView::default(),
    );
    assert_eq!(js[0].value, 1.0);
}

#[test]
fn canio_and_glass_joker_both_count_a_glass_face_card() {
    let mut js = jokers(&[J::GlassJoker, J::Canio]);
    destroyed(&mut js, &[glass(1, King, Spades)], &RunView::default());
    assert_eq!((js[0].value, js[1].value), (1.75, 2.0));
    // A Glass non-face only feeds the Glass Joker.
    destroyed(&mut js, &[glass(2, Two, Spades)], &RunView::default());
    assert_eq!((js[0].value, js[1].value), (2.5, 2.0));
}

#[test]
fn canio_blueprint_copy_counts_once() {
    // [Blueprint, Canio]: one face destroyed -> 2.0 (not 3.0); both pay x2:
    // mult 2 x 2 x 2 = 8 -> 24 x 8 = 192.
    let mut js = jokers(&[J::Blueprint, J::Canio]);
    destroyed(&mut js, &[c(1, King, Spades)], &RunView::default());
    assert_eq!(js[1].value, 2.0);
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 192.0);
}

#[test]
fn glass_break_in_the_game_feeds_glass_joker_and_canio_next_hand() {
    // Two Oops! All 6s: prob_mult 4 -> the 1-in-4 Glass break is certain.
    let mut g = new_game(121);
    add(&mut g, J::GlassJoker);
    add(&mut g, J::Canio);
    add(&mut g, J::OopsAll6s);
    add(&mut g, J::OopsAll6s);
    select(&mut g);
    // A Glass Jack is played alone and breaks after scoring.
    {
        let h = &mut g.state_mut().deck.hand[0];
        h.rank = Jack;
        h.suit = Spades;
        h.enhancement = Enhancement::Glass;
    }
    let info = g.step(Action::Play(0b1)).expect("play");
    // The broken card did not benefit the jokers during its own hand: High Card mult
    // 1 x 2 (Glass) = 2, no joker bonus.
    assert_eq!(info.hand.expect("hand").mult, 2.0);
    assert_eq!(info.cards_destroyed, 1);
    let s = g.state();
    assert_eq!(s.deck.len(), 51);
    // Glass Joker 1 + 0.75; Canio (Jack is a face) 1 + 1.
    assert_eq!(s.jokers[0].value, 1.75);
    assert_eq!(s.jokers[1].value, 2.0);
    // The next hand pays both: High Card mult 1 x 1.75 x 2.0 = 3.5.
    assert_eq!(play0(&mut g).1, 3.5);
}

// =============================================================================================
// Yorick
// =============================================================================================

#[test]
fn yorick_gains_one_every_23_discarded_cards_and_carries_over() {
    let (p, e) = pair7();
    let mut js = jokers(&[J::Yorick]);
    // 20 cards (4 discards of 5): counter 23 -> 3, still x1 -> 48.
    for _ in 0..4 {
        discard(&mut js, &cards(5));
    }
    assert_eq!((js[0].value, js[0].counter), (1.0, 3));
    assert_eq!(score(&p, e, &mut js).score, 48.0);
    // 3 more cards complete the 23rd: value 2.0, counter back to 23 -> mult 4 -> 96.
    discard(&mut js, &cards(3));
    assert_eq!((js[0].value, js[0].counter), (2.0, 23));
    assert_eq!(score(&p, e, &mut js).score, 96.0);
    // A discard that crosses the boundary carries the remainder: 25 cards -> +1 and 2 left over.
    discard(&mut js, &cards(25));
    assert_eq!((js[0].value, js[0].counter), (3.0, 21));
    // 46 cards at once cross it twice. From counter 21: 21 cards -> +1 (counter 23), 23 more
    // -> +1 (counter 23), the last 2 -> counter 21.
    discard(&mut js, &cards(46));
    assert_eq!((js[0].value, js[0].counter), (5.0, 21));
    // x5: 24 x (2 x 5) = 240.
    assert_eq!(score(&p, e, &mut js).score, 240.0);
}

#[test]
fn yorick_ignores_other_events_and_empty_discards() {
    let mut js = jokers(&[J::Yorick]);
    discard(&mut js, &[]);
    reroll(&mut js);
    round_end(&mut js);
    destroyed(&mut js, &[c(1, King, Spades)], &RunView::default());
    assert_eq!((js[0].value, js[0].counter), (1.0, 23));
    // A fresh instance with a zeroed counter heals itself instead of underflowing.
    let mut js = vec![jk(1, J::Yorick).with_counter(0)];
    discard(&mut js, &cards(1));
    assert_eq!((js[0].value, js[0].counter), (1.0, 22));
}

#[test]
fn yorick_blueprint_copy_counts_cards_once() {
    // [Blueprint, Yorick]: 23 cards -> exactly one step (x2), counter 23; never x3.
    let mut js = jokers(&[J::Blueprint, J::Yorick]);
    discard(&mut js, &cards(23));
    assert_eq!((js[1].value, js[1].counter), (2.0, 23));
    // Both pay x2: mult 2 x 2 x 2 = 8 -> 24 x 8 = 192.
    let (p, e) = pair7();
    assert_eq!(score(&p, e, &mut js).score, 192.0);
}

#[test]
fn yorick_in_the_game_counts_discarded_cards_across_actions() {
    let mut g = new_game(131);
    add(&mut g, J::Yorick);
    select(&mut g);
    // 5 discards of 5 cards = 25 cards (the default 4 discards are topped up).
    g.state_mut().discards_left = 8;
    for _ in 0..5 {
        g.step(Action::Discard(0b1_1111)).expect("discard");
    }
    // 25 = 23 + 2: x2 and 21 left.
    let j = &g.state().jokers[0];
    assert_eq!((j.value, j.counter), (2.0, 21));
    // High Card mult 1 x 2.
    assert_eq!(play0(&mut g).1, 2.0);
}

// =============================================================================================
// Whole-package smoke: all 14 jokers together stay deterministic and panic-free
// =============================================================================================

const PACKAGE: [JokerId; 14] = [
    J::FlashCard,
    J::Popcorn,
    J::SpareTrousers,
    J::AncientJoker,
    J::Ramen,
    J::Castle,
    J::Campfire,
    J::Throwback,
    J::GlassJoker,
    J::WeeJoker,
    J::TheIdol,
    J::HitTheRoad,
    J::Canio,
    J::Yorick,
];

fn play_random_game(seed: u64, jokers_in: &[JokerId]) -> Game {
    let mut g = new_game(seed);
    for &id in jokers_in {
        let _ = g.add_joker(JokerInstance::new(0, id));
    }
    let mut policy = GameRng::new(seed ^ 0xA11);
    let mut legal = Vec::new();
    let mut steps = 0;
    while !g.is_terminal() && steps < 3000 {
        g.legal_actions_into(&mut legal);
        let a = legal[policy.range(RngKey::Misc, 0, legal.len())];
        if matches!(a, Action::Play(_)) && policy.range(RngKey::Misc, 0, 3) == 0 {
            // Often win the round so end-of-round hooks and later antes are exercised.
            let t = g.state().target;
            g.state_mut().round_score = t - 1.0;
        }
        g.step(a).expect("legal action");
        steps += 1;
    }
    assert!(g.state().check_invariants().is_ok(), "seed {seed}");
    g
}

#[test]
fn package_jokers_survive_random_games_deterministically() {
    for seed in 0..40u64 {
        // Five of the package at a time, rotating, so every joker is exercised.
        let start = (seed as usize * 3) % PACKAGE.len();
        let picked: Vec<JokerId> = (0..5)
            .map(|i| PACKAGE[(start + i) % PACKAGE.len()])
            .collect();
        let a = play_random_game(seed, &picked);
        let b = play_random_game(seed, &picked);
        assert_eq!(a.state(), b.state(), "seed {seed}: replay diverged");
    }
}
