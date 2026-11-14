//! Phase 3 package wave5: Luchador, Mr. Bones, Smeared Joker, Showman, Invisible Joker, Chicot.
//!
//! These jokers mostly act through the game layer (`DisableBoss`, `PreventDeath`, `CopyJoker`)
//! or through passive flags (`RuleFlags::smeared`, `Modifiers::allow_duplicates`). Each joker is
//! tested at the hook level (trigger / non-trigger / copier) and through the real `Game`
//! (`step`), so the events fire at the right moment and the game layer applies the effects.
//! Smeared Joker is tested end to end with hand evaluation and every suit-reading joker.
//! Every expected number is derived in a comment.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::card::{Card, RuleFlags};
use balatro_core::content::BossId;
use balatro_core::content::jokers::{JokerId, try_def};
use balatro_core::deck::Deck;
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rank, Rarity, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::hand_eval::{HandEval, evaluate};
use balatro_core::hooks::{
    BossView, DiscardInfo, EventInput, EventResult, RoundEndInfo, RunView, SideEffect,
    collect_modifiers, fire_acquire, fire_blind_select, fire_discard, fire_round_end,
    fire_sell_self,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, Scorer};
use balatro_core::state::{GameConfig, Outcome};

use JokerId as J;
use Rank::*;
use Suit::*;

// =============================================================================================
// Helpers
// =============================================================================================

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

fn smeared_rules() -> RuleFlags {
    RuleFlags {
        smeared: true,
        ..RuleFlags::default()
    }
}

/// Score a hand with the rules the jokers themselves provide (`collect_modifiers`), exactly as
/// the game layer feeds them: Smeared in `js` turns the flag on.
fn score_held(played: &[Card], held: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> f64 {
    let mods = collect_modifiers(js);
    let run = RunView {
        rules: mods.rules,
        prob_mult: mods.prob_mult,
        ..RunView::default()
    };
    let mut played = played.to_vec();
    let mut rng = GameRng::new(5);
    let mut scorer = Scorer::with_trace(false);
    scorer
        .score_hand(ScoreInput {
            played: &mut played,
            held,
            eval,
            jokers: js,
            run: &run,
            rng: &mut rng,
        })
        .score
}

fn score(played: &[Card], eval: HandEval, js: &mut [JokerInstance]) -> f64 {
    score_held(played, &[], eval, js)
}

/// Fire one event with a fresh `GameRng::new(seed)`; returns (money delta, queued effects).
fn fire_with(
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

fn fire(
    js: &mut [JokerInstance],
    run: &RunView<'_>,
    f: impl FnOnce(EventInput<'_>) -> EventResult,
) -> Vec<SideEffect> {
    fire_with(js, run, 3, f).1
}

/// A view of the current blind: `active` = a boss effect is live.
fn blind_view(blind: BlindKind, active: bool) -> RunView<'static> {
    RunView {
        blind,
        in_round: true,
        boss: BossView {
            id: active.then_some(BossId::TheWall),
            active,
            ..BossView::default()
        },
        ..RunView::default()
    }
}

fn sell(js: &mut [JokerInstance], slot: usize, run: &RunView<'_>) -> Vec<SideEffect> {
    fire(js, run, |inp| fire_sell_self(inp, slot))
}

fn sell_seeded(js: &mut [JokerInstance], slot: usize, seed: u64) -> Vec<SideEffect> {
    fire_with(js, &RunView::default(), seed, |inp| {
        fire_sell_self(inp, slot)
    })
    .1
}

fn round_end(
    js: &mut [JokerInstance],
    won: bool,
    round_score: f64,
    target: f64,
) -> Vec<SideEffect> {
    let info = RoundEndInfo {
        won,
        saved: false,
        round_score,
        target,
    };
    fire(js, &RunView::default(), |inp| fire_round_end(inp, &info))
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

fn end_shop(g: &mut Game) {
    g.step(Action::EndShop).expect("end shop");
}

/// From BlindSelect of a Small blind: beat Small and Big, stop at BlindSelect of the Boss.
fn advance_to_boss_select(g: &mut Game) {
    assert_eq!(g.state().blind, BlindKind::Small);
    select(g);
    beat_blind(g);
    end_shop(g);
    select(g);
    beat_blind(g);
    end_shop(g);
    assert_eq!(g.state().blind, BlindKind::Boss);
    assert_eq!(g.state().phase, Phase::BlindSelect);
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("room")
}

fn slot_of(g: &Game, id: JokerId) -> Option<usize> {
    g.state().jokers.iter().position(|j| j.id == id)
}

fn count_of(g: &Game, id: JokerId) -> usize {
    g.state().jokers.iter().filter(|j| j.id == id).count()
}

/// Overwrite hand card `i` with a plain card of this rank and suit.
fn set_hand(g: &mut Game, i: usize, rank: Rank, suit: Suit) {
    let h = &mut g.state_mut().deck.hand[i];
    h.rank = rank;
    h.suit = suit;
    h.enhancement = Enhancement::None;
    h.debuffed = false;
}

/// Red flush-shaped hand in slots 0..5: 2H 4D 6H 8D 10D (no straight, no pairs).
fn set_red_hand(g: &mut Game) {
    for (i, (r, s)) in [
        (Two, Hearts),
        (Four, Diamonds),
        (Six, Hearts),
        (Eight, Diamonds),
        (Ten, Diamonds),
    ]
    .into_iter()
    .enumerate()
    {
        set_hand(g, i, r, s);
    }
}

// =============================================================================================
// Registry: definitions
// =============================================================================================

#[test]
fn all_six_defs_exist_with_wiki_stats() {
    // (id, rarity, cost, copyable, eternal ok, perishable ok): jokers.csv + balatrowiki.org.
    let table = [
        (J::Luchador, Rarity::Uncommon, 5, true, false, true),
        (J::MrBones, Rarity::Uncommon, 5, false, false, true),
        (J::SmearedJoker, Rarity::Uncommon, 7, false, true, true),
        (J::Showman, Rarity::Uncommon, 5, false, true, true),
        (J::InvisibleJoker, Rarity::Rare, 8, false, false, true),
        (J::Chicot, Rarity::Legendary, 20, false, true, true),
    ];
    for (id, rarity, cost, copyable, eternal, perishable) in table {
        let d = try_def(id).unwrap_or_else(|| panic!("{id:?} has no def"));
        assert_eq!(d.id, id);
        assert_eq!(d.rarity, rarity, "{id:?} rarity");
        assert_eq!(d.base_cost, cost, "{id:?} cost");
        assert_eq!(d.blueprint_compatible, copyable, "{id:?} copyable");
        assert_eq!(d.eternal_compatible, eternal, "{id:?} eternal");
        assert_eq!(d.perishable_compatible, perishable, "{id:?} perishable");
        assert!(d.copies.is_none());
    }
}

#[test]
fn initial_state_and_sell_values_match_wiki() {
    // Sell = floor(cost / 2): Luchador/Mr. Bones/Showman $2, Smeared $3, Invisible $4, Chicot $10.
    for (id, sell) in [
        (J::Luchador, 2),
        (J::MrBones, 2),
        (J::SmearedJoker, 3),
        (J::Showman, 2),
        (J::InvisibleJoker, 4),
        (J::Chicot, 10),
    ] {
        let j = jk(0, id);
        assert_eq!(j.sell_value, sell, "{id:?}");
        assert_eq!((j.value, j.counter, j.aux), (0.0, 0, 0), "{id:?}");
    }
}

#[test]
fn wave5_jokers_never_touch_the_score() {
    // 7S 7H Pair: 10 + 7 + 7 = 24 chips x 2 = 48 with or without any wave5 joker.
    let p = [c(1, Seven, Spades), c(2, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b11, &[]);
    let mut js = jokers(&[
        J::Luchador,
        J::MrBones,
        J::SmearedJoker,
        J::Showman,
        J::InvisibleJoker,
        J::Chicot,
    ]);
    assert_eq!(score(&p, e, &mut js), 48.0);
}

// =============================================================================================
// Luchador
// =============================================================================================

#[test]
fn luchador_sold_during_an_active_boss_disables_it() {
    let mut js = jokers(&[J::Joker, J::Luchador]);
    let fx = sell(&mut js, 1, &blind_view(BlindKind::Boss, true));
    assert_eq!(fx, vec![SideEffect::DisableBoss]);
}

#[test]
fn luchador_does_nothing_outside_an_active_boss() {
    let mut js = jokers(&[J::Luchador]);
    // Small / Big blind.
    assert!(sell(&mut js, 0, &blind_view(BlindKind::Small, false)).is_empty());
    assert!(sell(&mut js, 0, &blind_view(BlindKind::Big, false)).is_empty());
    // Boss already disabled (Chicot / an earlier Luchador): once per round.
    assert!(sell(&mut js, 0, &blind_view(BlindKind::Boss, false)).is_empty());
    // Debuffed Luchador (Perishable, Crimson Heart): sells but does nothing.
    let mut js = vec![jk(1, J::Luchador).with_debuff(true)];
    assert!(sell(&mut js, 0, &blind_view(BlindKind::Boss, true)).is_empty());
    // Other events never disable anything.
    let mut js = jokers(&[J::Luchador]);
    assert!(
        fire(
            &mut js,
            &blind_view(BlindKind::Boss, true),
            fire_blind_select
        )
        .is_empty()
    );
}

#[test]
fn luchador_and_chicot_acquire_ignore_a_boss_outside_the_round() {
    // In the shop / at blind select `blind` already names the next blind: `in_round` is false.
    let shop = RunView {
        in_round: false,
        ..blind_view(BlindKind::Boss, true)
    };
    let mut js = jokers(&[J::Luchador]);
    assert!(sell(&mut js, 0, &shop).is_empty());
    let mut js = jokers(&[J::Chicot]);
    assert!(fire(&mut js, &shop, |inp| fire_acquire(inp, 0)).is_empty());
    // Chicot's blind-select path does not need `in_round` (the round is not running yet).
    assert_eq!(
        fire(&mut js, &shop, fire_blind_select),
        vec![SideEffect::DisableBoss]
    );
}

#[test]
fn luchador_copied_by_a_sold_blueprint_or_brainstorm_disables_the_boss() {
    // Wiki (Brainstorm): selling a copier that copies Luchador also disables the Boss Blind.
    let boss = blind_view(BlindKind::Boss, true);
    let mut js = jokers(&[J::Blueprint, J::Luchador]);
    assert_eq!(sell(&mut js, 0, &boss), vec![SideEffect::DisableBoss]);
    let mut js = jokers(&[J::Luchador, J::Brainstorm]);
    assert_eq!(sell(&mut js, 1, &boss), vec![SideEffect::DisableBoss]);
    // Selling Luchador itself fires only its own hook, not the copier's: one effect.
    let mut js = jokers(&[J::Blueprint, J::Luchador]);
    assert_eq!(sell(&mut js, 1, &boss), vec![SideEffect::DisableBoss]);
    // A copier aimed elsewhere does nothing.
    let mut js = jokers(&[J::Luchador, J::Blueprint, J::Joker]);
    assert!(sell(&mut js, 1, &boss).is_empty());
}

#[test]
fn luchador_in_the_game_disables_only_the_boss_being_played() {
    let mut g = new_game(41);
    add(&mut g, J::Luchador);
    add(&mut g, J::Luchador);
    // Selling during the Small blind: sold for $2, no effect.
    select(&mut g);
    let m0 = g.state().money;
    let info = g.step(Action::SellJoker(0)).expect("sell");
    assert_eq!(info.sold_for, Some(2));
    assert_eq!(g.state().money, m0 + 2);
    assert!(!g.state().boss_disabled);
    assert_eq!(g.state().phase, Phase::Play);
    beat_blind(&mut g);
    end_shop(&mut g);
    select(&mut g);
    beat_blind(&mut g);
    end_shop(&mut g);
    g.state_mut().boss = Some(BossId::TheWall);
    select(&mut g);
    assert!(g.state().run_view().boss.active);
    g.step(Action::SellJoker(0)).expect("sell during the boss");
    assert!(g.state().boss_disabled);
    assert!(!g.state().run_view().boss.active);
    assert_eq!(g.state().phase, Phase::Play);
    assert!(g.state().jokers.is_empty());
    // A new ante starts with the boss enabled again.
    beat_blind(&mut g);
    assert_eq!(g.state().ante, 2);
    assert!(!g.state().boss_disabled);
}

#[test]
fn luchador_sold_in_the_shop_before_the_boss_does_not_disable_it() {
    // Wiki: Luchador "will only take effect if sold after entering the Boss Blind". In the shop
    // after the Big blind `run.blind` is already Boss, so the hook cannot tell the difference.
    let mut g = new_game(42);
    add(&mut g, J::Luchador);
    select(&mut g);
    beat_blind(&mut g);
    end_shop(&mut g);
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().blind, BlindKind::Boss);
    g.state_mut().boss = Some(BossId::TheWall);
    g.step(Action::SellJoker(0)).expect("sell in the shop");
    assert!(!g.state().boss_disabled);
}

// =============================================================================================
// Mr. Bones
// =============================================================================================

#[test]
fn mr_bones_saves_a_lost_round_with_a_quarter_of_the_chips_and_destroys_itself() {
    let mut js = jokers(&[J::Joker, J::MrBones]);
    // 75 / 300 = 25% exactly: saved.
    assert_eq!(
        round_end(&mut js, false, 75.0, 300.0),
        vec![SideEffect::PreventDeath, destroy_of(101)]
    );
    // 299 / 300: saved.
    assert_eq!(
        round_end(&mut js, false, 299.0, 300.0),
        vec![SideEffect::PreventDeath, destroy_of(101)]
    );
}

#[test]
fn mr_bones_does_not_trigger_below_a_quarter_on_a_win_or_when_debuffed() {
    let mut js = jokers(&[J::MrBones]);
    // 74.9 / 300 < 25%.
    assert!(round_end(&mut js, false, 74.9, 300.0).is_empty());
    assert!(round_end(&mut js, false, 0.0, 300.0).is_empty());
    // Won rounds: nothing (Mr. Bones stays).
    assert!(round_end(&mut js, true, 400.0, 300.0).is_empty());
    let mut js = vec![jk(1, J::MrBones).with_debuff(true)];
    assert!(round_end(&mut js, false, 200.0, 300.0).is_empty());
}

#[test]
fn only_the_leftmost_of_two_mr_bones_is_spent() {
    let mut js = jokers(&[J::MrBones, J::Joker, J::MrBones]);
    assert_eq!(
        round_end(&mut js, false, 100.0, 300.0),
        vec![SideEffect::PreventDeath, destroy_of(100)]
    );
}

#[test]
fn mr_bones_is_not_copied_by_blueprint() {
    // Blueprint is not even asked for end-of-round hooks; Mr. Bones is also incompatible.
    let mut js = jokers(&[J::Blueprint, J::MrBones]);
    assert_eq!(
        round_end(&mut js, false, 100.0, 300.0),
        vec![SideEffect::PreventDeath, destroy_of(101)]
    );
    let mut js = jokers(&[J::Blueprint, J::Joker]);
    assert!(round_end(&mut js, false, 100.0, 300.0).is_empty());
}

/// Lose the current round on the final hand with `fraction` of the target already scored.
fn lose_final_hand(g: &mut Game, fraction: f64) -> StepInfo {
    let t = g.state().target;
    let s = g.state_mut();
    s.hands_left = 1;
    s.round_score = t * fraction;
    g.step(Action::Play(0b1)).expect("final play")
}

#[test]
fn mr_bones_in_the_game_saves_the_run_without_blind_reward() {
    let mut g = new_game(7);
    add(&mut g, J::Joker);
    add(&mut g, J::MrBones);
    select(&mut g);
    g.state_mut().money = 20;
    // 300-chip Small blind; 75 already scored + one High Card (<= (5 + 11) x 5 = 80) < 300.
    let info = lose_final_hand(&mut g, 0.25);
    let round = info.round.expect("round ended");
    assert!(round.saved && round.won && !round.out_of_cards);
    let co = round.cash_out.expect("cash-out happens");
    // No blind reward, no hands left; interest on $20 = $4 is still paid -> $24.
    assert_eq!(co.blind_reward, 0);
    assert_eq!(co.hands, 0);
    assert_eq!(co.interest, 4);
    assert_eq!(co.final_money, 24);
    assert_eq!(g.state().money, 24);
    // The run goes on: shop, next blind, Mr. Bones destroyed, the other joker kept.
    assert_eq!(g.outcome(), None);
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().blind, BlindKind::Big);
    assert_eq!(slot_of(&g, J::MrBones), None);
    assert_eq!(count_of(&g, J::Joker), 1);
    assert_eq!(g.state().stats.rounds_won, 1);
    // And the next round plays normally.
    end_shop(&mut g);
    select(&mut g);
    assert_eq!(g.state().phase, Phase::Play);
}

/// Reach the Boss blind round (boss id 1 = placeholder with an active effect).
fn start_boss_round(g: &mut Game) {
    advance_to_boss_select(g);
    g.state_mut().boss = Some(BossId::TheWall);
    select(g);
    assert_eq!(g.state().blind, BlindKind::Boss);
    assert_eq!(g.state().phase, Phase::Play);
}

#[test]
fn mr_bones_saved_boss_round_counts_as_a_defeated_boss_for_campfire_and_rocket() {
    // Real game: Mr. Bones turns the loss into a normal round end, and Campfire / Rocket only ask
    // "is the blind a boss". Joker order must not matter (Bones to the left or the right).
    for bones_first in [true, false] {
        let mut g = new_game(70);
        if bones_first {
            add(&mut g, J::MrBones);
        }
        let camp = add(&mut g, J::Campfire);
        let rocket = add(&mut g, J::Rocket);
        if !bones_first {
            add(&mut g, J::MrBones);
        }
        g.state_mut().jokers[camp].value = 2.5;
        g.state_mut().jokers[rocket].value = 3.0;
        start_boss_round(&mut g);
        let info = lose_final_hand(&mut g, 0.25);
        let round = info.round.expect("round ended");
        assert!(round.saved && round.won, "bones_first = {bones_first}");
        // Rocket grew by $2 before its payout: 3 + 2 = 5 paid in the cash-out.
        assert_eq!(round.cash_out.expect("cash-out").jokers, 5);
        assert_eq!(slot_of(&g, J::MrBones), None);
        let camp = slot_of(&g, J::Campfire).expect("campfire kept");
        let rocket = slot_of(&g, J::Rocket).expect("rocket kept");
        assert_eq!(g.state().jokers[camp].value, 1.0, "campfire reset");
        assert_eq!(g.state().jokers[rocket].value, 5.0, "rocket grew");
        assert_eq!(g.state().phase, Phase::Shop);
    }
}

#[test]
fn mr_bones_saved_small_round_does_not_reset_campfire_or_grow_rocket() {
    let mut g = new_game(71);
    let camp = add(&mut g, J::Campfire);
    let rocket = add(&mut g, J::Rocket);
    add(&mut g, J::MrBones);
    g.state_mut().jokers[camp].value = 2.5;
    g.state_mut().jokers[rocket].value = 3.0;
    select(&mut g);
    let info = lose_final_hand(&mut g, 0.25);
    let round = info.round.expect("round ended");
    assert!(round.saved);
    // Rocket still pays its current $3, but does not grow outside a Boss Blind.
    assert_eq!(round.cash_out.expect("cash-out").jokers, 3);
    assert_eq!(g.state().jokers[camp].value, 2.5);
    assert_eq!(g.state().jokers[rocket].value, 3.0);
}

#[test]
fn round_end_probe_does_not_change_the_state_of_an_unsaved_loss() {
    // Well below 25%: no save, the run ends; Campfire keeps its value (hooks saw `saved = false`).
    let mut g = new_game(72);
    let camp = add(&mut g, J::Campfire);
    add(&mut g, J::MrBones);
    g.state_mut().jokers[camp].value = 2.5;
    start_boss_round(&mut g);
    let info = lose_final_hand(&mut g, 0.1);
    assert!(!info.round.expect("round").won);
    assert_eq!(g.outcome(), Some(Outcome::Loss));
    assert_eq!(
        g.state().jokers[slot_of(&g, J::Campfire).unwrap()].value,
        2.5
    );
    assert!(slot_of(&g, J::MrBones).is_some());
}

#[test]
fn boss_defeated_reads_won_or_saved_on_a_boss_blind_only() {
    let boss = blind_view(BlindKind::Boss, true);
    let small = blind_view(BlindKind::Small, false);
    let info = |won, saved| RoundEndInfo {
        won,
        saved,
        round_score: 400.0,
        target: 300.0,
    };
    assert!(info(true, false).boss_defeated(&boss));
    assert!(info(false, true).boss_defeated(&boss));
    assert!(!info(false, false).boss_defeated(&boss));
    assert!(!info(true, false).boss_defeated(&small));
    assert!(!info(false, true).boss_defeated(&small));
}

#[test]
fn disable_boss_recomputes_the_target_with_the_standard_multiplier() {
    // Boss specials (Wall x4, Violet Vessel x6) arrive in Phase 4; today every boss is x2, so
    // the target is unchanged. The disabled boss must use the standard multiplier.
    let mut g = new_game(73);
    add(&mut g, J::Chicot);
    advance_to_boss_select(&mut g);
    g.state_mut().boss = Some(BossId::TheWall);
    let before = g.state().target;
    select(&mut g);
    assert!(g.state().boss_disabled);
    assert_eq!(
        g.state().boss_mult(),
        balatro_core::blinds::MULT_BOSS_DEFAULT
    );
    assert_eq!(g.state().target, before);
    assert_eq!(
        g.state().target,
        balatro_core::blinds::requirement(
            g.state().ante,
            BlindKind::Boss,
            balatro_core::blinds::MULT_BOSS_DEFAULT,
            g.state().config.stake.scaling()
        )
    );
}

#[test]
fn mr_bones_in_the_game_below_a_quarter_is_game_over_and_he_stays() {
    let mut g = new_game(8);
    add(&mut g, J::MrBones);
    select(&mut g);
    // 0 scored + one High Card (<= 16 chips) < 75 = 25% of 300.
    let info = lose_final_hand(&mut g, 0.0);
    let round = info.round.expect("round ended");
    assert!(!round.saved && !round.won);
    assert!(round.cash_out.is_none());
    assert_eq!(g.outcome(), Some(Outcome::Loss));
    assert_eq!(g.state().phase, Phase::GameOver);
    assert_eq!(slot_of(&g, J::MrBones), Some(0));
}

#[test]
fn mr_bones_in_the_game_cannot_save_a_run_that_ran_out_of_cards() {
    let mut g = new_game(9);
    add(&mut g, J::MrBones);
    select(&mut g);
    // Keep one card in hand; everything else goes to the discard pile (cards conserved).
    {
        let s = g.state_mut();
        let rest: Vec<Card> = s.deck.hand.drain(1..).collect();
        s.deck.discard_pile.extend(rest);
        let draw = std::mem::take(&mut s.deck.draw_pile);
        s.deck.discard_pile.extend(draw);
        s.round_score = s.target * 0.5;
    }
    let info = g.step(Action::Play(0b1)).expect("last card");
    let round = info.round.expect("round ended");
    // Hands remain, but no card can be drawn: lost, and Mr. Bones does not apply.
    assert!(round.out_of_cards);
    assert!(!round.saved && !round.won);
    assert_eq!(g.outcome(), Some(Outcome::Loss));
}

#[test]
fn mr_bones_save_on_the_ante_8_boss_wins_the_run() {
    let mut g = new_game(10);
    add(&mut g, J::MrBones);
    {
        let s = g.state_mut();
        s.ante = 8;
        s.blind = BlindKind::Boss;
    }
    select(&mut g);
    let info = lose_final_hand(&mut g, 0.3);
    let round = info.round.expect("round ended");
    assert!(round.saved);
    assert_eq!(round.cash_out.expect("cash-out").blind_reward, 0);
    assert_eq!(g.outcome(), Some(Outcome::Win));
    assert_eq!(g.state().phase, Phase::Won);
}

// =============================================================================================
// Smeared Joker
// =============================================================================================

#[test]
fn smeared_sets_the_rule_flag_only_while_active() {
    assert!(collect_modifiers(&jokers(&[J::SmearedJoker])).rules.smeared);
    assert!(!collect_modifiers(&jokers(&[J::Joker])).rules.smeared);
    // Debuffed: no passive.
    assert!(
        !collect_modifiers(&[jk(1, J::SmearedJoker).with_debuff(true)])
            .rules
            .smeared
    );
    // Blueprint never copies passives: a copier aimed at Smeared adds nothing new, and a
    // copier alone gives nothing.
    assert!(
        !collect_modifiers(&jokers(&[J::Blueprint, J::Joker]))
            .rules
            .smeared
    );
    let m = collect_modifiers(&jokers(&[J::Blueprint, J::SmearedJoker]));
    assert!(m.rules.smeared);
    assert_eq!(
        m,
        collect_modifiers(&jokers(&[J::SmearedJoker])),
        "the copier changes nothing"
    );
}

#[test]
fn smeared_suit_membership_with_wild_debuff_and_stone() {
    let on = smeared_rules();
    let off = RuleFlags::default();
    let heart = c(1, Seven, Hearts);
    assert!(heart.has_suit(Diamonds, &on) && heart.has_suit(Hearts, &on));
    assert!(!heart.has_suit(Spades, &on) && !heart.has_suit(Clubs, &on));
    assert!(!heart.has_suit(Diamonds, &off));
    let spade = c(2, Seven, Spades);
    assert!(spade.has_suit(Clubs, &on) && !spade.has_suit(Hearts, &on));
    // A debuffed plain card keeps its (smeared) suit.
    assert!(debuff(heart).has_suit(Diamonds, &on));
    // Wild: every suit; a debuffed Wild reverts to its printed suit, which is then smeared.
    let wild = enh(c(3, Seven, Hearts), Enhancement::Wild);
    for s in Suit::ALL {
        assert!(wild.has_suit(s, &on), "{s:?}");
    }
    let dwild = debuff(wild);
    assert!(dwild.has_suit(Hearts, &on) && dwild.has_suit(Diamonds, &on));
    assert!(!dwild.has_suit(Clubs, &on) && !dwild.has_suit(Spades, &on));
    assert!(!dwild.has_suit(Diamonds, &off));
    // Stone: no suit at all.
    let stone = enh(c(4, Seven, Hearts), Enhancement::Stone);
    for s in Suit::ALL {
        assert!(!stone.has_suit(s, &on), "{s:?}");
    }
}

#[test]
fn smeared_makes_mixed_colour_flushes_in_hand_evaluation() {
    let reds = [
        c(1, Two, Hearts),
        c(2, Four, Diamonds),
        c(3, Six, Hearts),
        c(4, Eight, Diamonds),
        c(5, Ten, Diamonds),
    ];
    let mods = collect_modifiers(&jokers(&[J::SmearedJoker]));
    assert_eq!(evaluate(&reds, &mods.rules).hand_type, HandType::Flush);
    assert_eq!(
        evaluate(&reds, &RuleFlags::default()).hand_type,
        HandType::HighCard
    );
    // Blacks: 3C 5S 7C 9S QS.
    let blacks = [
        c(1, Three, Clubs),
        c(2, Five, Spades),
        c(3, Seven, Clubs),
        c(4, Nine, Spades),
        c(5, Queen, Spades),
    ];
    assert_eq!(evaluate(&blacks, &mods.rules).hand_type, HandType::Flush);
    // One black card among four reds: no 5-card flush, even smeared.
    let mut mixed = reds;
    mixed[4] = c(5, Ten, Clubs);
    assert_eq!(evaluate(&mixed, &mods.rules).hand_type, HandType::HighCard);
    // ... unless Four Fingers: 4 reds flush, the Club does not score.
    let ff = RuleFlags {
        four_fingers: true,
        ..mods.rules
    };
    let e = evaluate(&mixed, &ff);
    assert_eq!(e.hand_type, HandType::Flush);
    assert_eq!(e.scoring_mask.count_ones(), 4);
    // Straight flush across colours: 5H 6D 7H 8D 9H.
    let sf = [
        c(1, Five, Hearts),
        c(2, Six, Diamonds),
        c(3, Seven, Hearts),
        c(4, Eight, Diamonds),
        c(5, Nine, Hearts),
    ];
    assert_eq!(
        evaluate(&sf, &mods.rules).hand_type,
        HandType::StraightFlush
    );
    // A Wild Spade completes a red flush with or without Smeared; a Stone never does.
    let mut wild = reds;
    wild[0] = enh(c(1, Two, Spades), Enhancement::Wild);
    assert_eq!(evaluate(&wild, &mods.rules).hand_type, HandType::Flush);
    let mut stone = reds;
    stone[0] = enh(c(1, Two, Hearts), Enhancement::Stone);
    assert_ne!(evaluate(&stone, &mods.rules).hand_type, HandType::Flush);
}

#[test]
fn smeared_with_greedy_lusty_wrathful_gluttonous() {
    // 7H 7D Pair: 10 + 7 + 7 = 24 chips, mult 2.
    let p = [c(1, Seven, Hearts), c(2, Seven, Diamonds)];
    let e = ev(HandType::Pair, 0b11, &[]);
    let ids = [
        J::GreedyJoker,
        J::LustyJoker,
        J::WrathfulJoker,
        J::GluttonousJoker,
    ];
    // Without Smeared: Greedy +3 (7D), Lusty +3 (7H) -> 24 x 8 = 192.
    assert_eq!(score(&p, e, &mut jokers(&ids)), 192.0);
    // With Smeared: Greedy and Lusty +3 per card each -> 2 + 12 = 14 -> 24 x 14 = 336.
    let mut with: Vec<JokerId> = ids.to_vec();
    with.push(J::SmearedJoker);
    assert_eq!(score(&p, e, &mut jokers(&with)), 336.0);
    // A debuffed 7D is never scored: 10 + 7 = 17 chips; 7H gives Greedy +3, Lusty +3 -> 8.
    let pd = [c(1, Seven, Hearts), debuff(c(2, Seven, Diamonds))];
    assert_eq!(score(&pd, e, &mut jokers(&with)), 17.0 * 8.0);
    // Blacks: 7S 7C -> Wrathful and Gluttonous +3 per card each -> 24 x 14 = 336.
    let pb = [c(1, Seven, Spades), c(2, Seven, Clubs)];
    assert_eq!(score(&pb, e, &mut jokers(&with)), 336.0);
}

#[test]
fn smeared_with_flower_pot_needs_two_of_each_colour() {
    // 7H 7S 9H 9S Two Pair: 20 + 7 + 7 + 9 + 9 = 52 chips, mult 2 -> 104.
    let p = [
        c(1, Seven, Hearts),
        c(2, Seven, Spades),
        c(3, Nine, Hearts),
        c(4, Nine, Spades),
    ];
    let e = ev(HandType::TwoPair, 0b1111, &[HandType::Pair]);
    assert_eq!(score(&p, e, &mut jokers(&[J::FlowerPot])), 104.0);
    // Smeared: Hearts fill H and D, Spades fill S and C -> X3 -> 52 x 6 = 312 (wiki: "two of
    // each color").
    assert_eq!(
        score(&p, e, &mut jokers(&[J::FlowerPot, J::SmearedJoker])),
        312.0
    );
    // Three reds + one black (7H 7D 9H 9S): the single black card covers Clubs or Spades, not
    // both -> no X3 -> 104.
    let p3 = [
        c(1, Seven, Hearts),
        c(2, Seven, Diamonds),
        c(3, Nine, Hearts),
        c(4, Nine, Spades),
    ];
    assert_eq!(
        score(&p3, e, &mut jokers(&[J::FlowerPot, J::SmearedJoker])),
        104.0
    );
}

#[test]
fn smeared_with_seeing_double_spade_counts_as_club() {
    // 7S 7H Pair: 24 chips x 2 = 48; X2 -> 96.
    let p = [c(1, Seven, Spades), c(2, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b11, &[]);
    assert_eq!(score(&p, e, &mut jokers(&[J::SeeingDouble])), 48.0);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SeeingDouble, J::SmearedJoker])),
        96.0
    );
    // 7S 7S (two black cards): wiki says any Club/Spade triggers with Smeared (bug), and the
    // pair-based implementation agrees here: one Spade is the Club, the other the non-Club.
    let pss = [c(1, Seven, Spades), c(2, Seven, Spades)];
    assert_eq!(score(&pss, e, &mut jokers(&[J::SeeingDouble])), 48.0);
    assert_eq!(
        score(&pss, e, &mut jokers(&[J::SeeingDouble, J::SmearedJoker])),
        96.0
    );
    // Two reds never trigger, smeared or not.
    let prr = [c(1, Seven, Hearts), c(2, Seven, Diamonds)];
    assert_eq!(
        score(&prr, e, &mut jokers(&[J::SeeingDouble, J::SmearedJoker])),
        48.0
    );
}

#[test]
fn smeared_with_seeing_double_single_spade_high_card() {
    // 7S High Card: 5 + 7 = 12 chips x 1 = 12; wiki: X2 -> 24.
    let p = [c(1, Seven, Spades)];
    let e = ev(HandType::HighCard, 0b1, &[]);
    assert_eq!(
        score(&p, e, &mut jokers(&[J::SeeingDouble, J::SmearedJoker])),
        24.0
    );
}

#[test]
fn smeared_with_blackboard_changes_nothing() {
    // Blackboard already accepts both black suits; reds stay red under Smeared.
    // 7S 7H Pair = 48 base.
    let p = [c(1, Seven, Spades), c(2, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b11, &[]);
    let black = [c(10, Two, Spades), c(11, Three, Clubs)];
    let red = [c(10, Two, Spades), c(11, Three, Diamonds)];
    for ids in [vec![J::Blackboard], vec![J::Blackboard, J::SmearedJoker]] {
        // All held black -> X3 -> 144.
        assert_eq!(score_held(&p, &black, e, &mut jokers(&ids)), 144.0);
        assert_eq!(score_held(&p, &red, e, &mut jokers(&ids)), 48.0);
    }
    // A debuffed Wild Diamond in hand reverts to Diamond: blocks Blackboard even smeared.
    let dwild = [debuff(enh(c(10, Two, Diamonds), Enhancement::Wild))];
    assert_eq!(
        score_held(
            &p,
            &dwild,
            e,
            &mut jokers(&[J::Blackboard, J::SmearedJoker])
        ),
        48.0
    );
}

#[test]
fn smeared_with_ancient_joker_target_covers_its_colour() {
    // Target Spades; 7C 7H Pair: 24 x 2 = 48; a smeared Club counts -> X1.5 -> 72.
    let p = [c(1, Seven, Clubs), c(2, Seven, Hearts)];
    let e = ev(HandType::Pair, 0b11, &[]);
    let ancient = |uid| {
        let mut j = jk(uid, J::AncientJoker);
        j.set_aux_suit(Some(Spades));
        j
    };
    assert_eq!(score(&p, e, &mut [ancient(100)]), 48.0);
    assert_eq!(
        score(&p, e, &mut [ancient(100), jk(101, J::SmearedJoker)]),
        72.0
    );
}

#[test]
fn smeared_with_castle_counts_the_whole_colour() {
    let castle = || {
        let mut j = jk(100, J::Castle);
        j.set_aux_suit(Some(Hearts));
        j
    };
    let cards = [
        c(1, Two, Diamonds),
        c(2, Three, Hearts),
        c(3, Four, Spades),
        debuff(c(4, Five, Diamonds)),
    ];
    let discard = |js: &mut [JokerInstance]| {
        let mods = collect_modifiers(js);
        let run = RunView {
            rules: mods.rules,
            ..RunView::default()
        };
        let info = DiscardInfo {
            cards: &cards,
            hand_type: HandType::HighCard,
            forced: false,
        };
        fire(js, &run, |inp| fire_discard(inp, &info));
    };
    // Without Smeared only 3H counts: +3.
    let mut js = [castle()];
    discard(&mut js);
    assert_eq!(js[0].value, 3.0);
    // Smeared: 2D and 3H count (the debuffed 5D never does): +6.
    let mut js = [castle(), jk(101, J::SmearedJoker)];
    discard(&mut js);
    assert_eq!(js[0].value, 6.0);
}

#[test]
fn smeared_in_the_game_flush_and_greedy_end_to_end() {
    let mut g = new_game(21);
    add(&mut g, J::SmearedJoker);
    add(&mut g, J::GreedyJoker);
    select(&mut g);
    g.state_mut().target = 1.0e9; // keep the round going for the second hand
    set_red_hand(&mut g);
    let h = g
        .step(Action::Play(0b1_1111))
        .expect("play")
        .hand
        .expect("hand report");
    // Flush L1 35 x 4; chips 35 + 2 + 4 + 6 + 8 + 10 = 65; Greedy +3 x 5 -> mult 19 -> 1235.
    assert_eq!(h.hand_type, HandType::Flush);
    assert_eq!((h.chips, h.mult, h.score), (65.0, 19.0, 1235.0));
    // Sell Smeared: the rule disappears at once (modifiers are refreshed on sale).
    let s = slot_of(&g, J::SmearedJoker).expect("smeared") as u8;
    g.step(Action::SellJoker(s)).expect("sell smeared");
    assert!(!g.state().mods.rules.smeared);
    set_red_hand(&mut g);
    let h = g
        .step(Action::Play(0b1_1111))
        .expect("play")
        .hand
        .expect("hand report");
    // High Card: 10D scores alone: 5 + 10 = 15 chips; Greedy +3 -> 15 x 4 = 60.
    assert_eq!(h.hand_type, HandType::HighCard);
    assert_eq!(h.score, 60.0);
}

#[test]
fn smeared_in_the_game_castle_discard_end_to_end() {
    let mut g = new_game(22);
    add(&mut g, J::SmearedJoker);
    let cs = add(&mut g, J::Castle);
    select(&mut g);
    g.state_mut().jokers[cs].set_aux_suit(Some(Clubs));
    set_hand(&mut g, 0, Two, Spades);
    set_hand(&mut g, 1, Three, Clubs);
    set_hand(&mut g, 2, Four, Hearts);
    g.step(Action::Discard(0b111)).expect("discard");
    // 2S and 3C count as Clubs, 4H does not: +3 x 2 = 6 chips.
    assert_eq!(g.state().jokers[cs].value, 6.0);
}

// =============================================================================================
// Showman
// =============================================================================================

#[test]
fn showman_allows_duplicates_while_active() {
    assert!(!collect_modifiers(&[]).allow_duplicates);
    assert!(!collect_modifiers(&jokers(&[J::Joker])).allow_duplicates);
    assert!(collect_modifiers(&jokers(&[J::Showman])).allow_duplicates);
    assert!(!collect_modifiers(&[jk(1, J::Showman).with_debuff(true)]).allow_duplicates);
    // A copier alone never provides the passive.
    assert!(!collect_modifiers(&jokers(&[J::Blueprint, J::Joker])).allow_duplicates);
    // Nothing else changes.
    let m = collect_modifiers(&jokers(&[J::Showman]));
    assert_eq!(m.rules, RuleFlags::default());
    assert_eq!((m.prob_mult, m.hand_size, m.free_rerolls), (1, 0, 0));
}

#[test]
fn showman_in_the_game_sets_the_cached_flag() {
    let mut g = new_game(31);
    assert!(!g.state().mods.allow_duplicates);
    add(&mut g, J::Showman);
    assert!(g.state().mods.allow_duplicates);
    select(&mut g);
    g.step(Action::SellJoker(0)).expect("sell");
    assert!(!g.state().mods.allow_duplicates);
}

// =============================================================================================
// Invisible Joker
// =============================================================================================

#[test]
fn invisible_counts_rounds_up_to_two() {
    let mut js = jokers(&[J::InvisibleJoker]);
    assert!(round_end(&mut js, true, 400.0, 300.0).is_empty());
    assert_eq!(js[0].counter, 1);
    // A round saved by Mr. Bones (won = false) still ends the round.
    round_end(&mut js, false, 100.0, 300.0);
    assert_eq!(js[0].counter, 2);
    round_end(&mut js, true, 400.0, 300.0);
    assert_eq!(js[0].counter, 2, "capped at 2");
}

#[test]
fn invisible_sold_before_two_rounds_does_nothing() {
    for rounds in [0, 1] {
        let mut js = jokers(&[J::Joker, J::InvisibleJoker]);
        js[1].counter = rounds;
        assert!(sell_seeded(&mut js, 1, 1).is_empty(), "{rounds} rounds");
    }
}

#[test]
fn invisible_sold_after_two_rounds_copies_another_joker_without_negative() {
    let mut js = jokers(&[J::Joker, J::InvisibleJoker]);
    js[1].counter = 2;
    assert_eq!(
        sell_seeded(&mut js, 1, 1),
        vec![SideEffect::CopyJoker {
            uid: 100,
            strip_negative: true
        }]
    );
    // Alone: "No other Jokers!", nothing happens (and no roll is needed).
    let mut js = jokers(&[J::InvisibleJoker]);
    js[0].counter = 2;
    assert!(sell_seeded(&mut js, 0, 1).is_empty());
    // Debuffed: sells, but does not duplicate.
    let mut js = jokers(&[J::Joker, J::InvisibleJoker]);
    js[1].counter = 2;
    js[1].debuffed = true;
    assert!(sell_seeded(&mut js, 1, 1).is_empty());
}

#[test]
fn invisible_picks_uniformly_among_the_others_deterministically() {
    let mut seen = [0u32; 4];
    for seed in 0..200 {
        let mut js = jokers(&[J::Joker, J::InvisibleJoker, J::JollyJoker, J::Joker]);
        js[1].counter = 2;
        let fx = sell_seeded(&mut js, 1, seed);
        let [SideEffect::CopyJoker { uid, .. }] = fx[..] else {
            panic!("expected one copy, got {fx:?}");
        };
        assert_ne!(uid, 101, "never itself");
        seen[(uid - 100) as usize] += 1;
        // Same seed, same pick.
        assert_eq!(sell_seeded(&mut js, 1, seed), fx);
    }
    assert_eq!(seen[1], 0);
    for i in [0, 2, 3] {
        assert!(seen[i] > 30, "slot {i} picked {} / 200", seen[i]);
    }
}

#[test]
fn invisible_is_not_copied_by_blueprint() {
    // Selling a Blueprint aimed at a ready Invisible Joker: nothing (Invisible is bp-incompatible).
    let mut js = jokers(&[J::Joker, J::Blueprint, J::InvisibleJoker]);
    js[2].counter = 2;
    assert!(sell_seeded(&mut js, 1, 1).is_empty());
    // Blueprint does not run its end-of-round counter either.
    let mut js = jokers(&[J::Blueprint, J::InvisibleJoker]);
    round_end(&mut js, true, 400.0, 300.0);
    assert_eq!(js[1].counter, 1);
}

#[test]
fn invisible_acquire_restarts_the_counter() {
    // A duplicated Invisible Joker carries counter 2 but starts again at 0/2 (wiki).
    let mut js = jokers(&[J::InvisibleJoker]);
    js[0].counter = 2;
    fire(&mut js, &RunView::default(), |inp| fire_acquire(inp, 0));
    assert_eq!(js[0].counter, 0);
}

/// Play two rounds (Small, Big) so held Invisible Jokers are ready, ending in the shop.
fn two_rounds(g: &mut Game) {
    select(g);
    beat_blind(g);
    end_shop(g);
    select(g);
    beat_blind(g);
    assert_eq!(g.state().phase, Phase::Shop);
}

#[test]
fn invisible_in_the_game_duplicates_with_state_after_two_rounds() {
    let mut g = new_game(51);
    let inv = add(&mut g, J::InvisibleJoker);
    // A scaling joker: its live state must be copied.
    let rb = add(&mut g, J::RideTheBus);
    g.state_mut().jokers[rb].value = 7.0;
    // Not ready after one round: selling now would waste it.
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().jokers[inv].counter, 1);
    end_shop(&mut g);
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().jokers[inv].counter, 2);
    // Ride the Bus kept scaling through the two winning plays: read the live value.
    let rb_value = g.state().jokers[rb].value;
    let m0 = g.state().money;
    let info = g.step(Action::SellJoker(inv as u8)).expect("sell");
    assert_eq!(info.sold_for, Some(4));
    assert_eq!(g.state().money, m0 + 4);
    assert_eq!(slot_of(&g, J::InvisibleJoker), None);
    assert_eq!(count_of(&g, J::RideTheBus), 2);
    let js = &g.state().jokers;
    assert_ne!(js[0].uid, js[1].uid);
    assert_eq!((js[0].value, js[1].value), (rb_value, rb_value));
}

#[test]
fn invisible_in_the_game_copy_fits_in_full_slots_and_loses_negative() {
    let mut g = new_game(52);
    add(&mut g, J::InvisibleJoker);
    g.add_joker(JokerInstance::new(0, J::Joker).with_edition(Edition::Negative))
        .expect("negative fits");
    for _ in 0..4 {
        add(&mut g, J::Joker);
    }
    // 6 jokers in 5 + 1 (Negative) slots: full.
    assert_eq!(g.state().jokers.len(), 6);
    assert!(!g.state().has_joker_room());
    two_rounds(&mut g);
    g.step(Action::SellJoker(0)).expect("sell");
    // Selling frees a slot, the copy takes it: still 6 jokers, never over capacity, and the
    // copy is never Negative (only one Negative remains).
    let js = &g.state().jokers;
    assert_eq!(js.len(), 6);
    assert_eq!(js.iter().filter(|j| j.id == J::Joker).count(), 6);
    assert_eq!(
        js.iter().filter(|j| j.edition == Edition::Negative).count(),
        1
    );
    assert!(js.len() as u32 <= g.state().joker_capacity());
}

#[test]
fn invisible_in_the_game_copy_of_an_invisible_restarts_via_on_acquire() {
    let mut g = new_game(53);
    add(&mut g, J::InvisibleJoker);
    add(&mut g, J::InvisibleJoker);
    two_rounds(&mut g);
    assert!(g.state().jokers.iter().all(|j| j.counter == 2));
    let kept_uid = g.state().jokers[1].uid;
    g.step(Action::SellJoker(0)).expect("sell");
    // The only other joker is the second Invisible: it is copied, and the copy's on_acquire
    // resets it to 0/2 while the original stays ready.
    let js = &g.state().jokers;
    assert_eq!(js.len(), 2);
    assert!(js.iter().all(|j| j.id == J::InvisibleJoker));
    assert_eq!(js[0].uid, kept_uid);
    assert_eq!((js[0].counter, js[1].counter), (2, 0));
}

#[test]
fn invisible_in_the_game_alone_just_sells() {
    let mut g = new_game(54);
    add(&mut g, J::InvisibleJoker);
    two_rounds(&mut g);
    g.step(Action::SellJoker(0)).expect("sell");
    assert!(g.state().jokers.is_empty());
    assert_eq!(g.state().stats.effects_dropped, 0);
}

// =============================================================================================
// Chicot
// =============================================================================================

#[test]
fn chicot_disables_every_boss_on_selection() {
    let mut js = jokers(&[J::Chicot]);
    let fx = fire(
        &mut js,
        &blind_view(BlindKind::Boss, true),
        fire_blind_select,
    );
    assert_eq!(fx, vec![SideEffect::DisableBoss]);
    // Small and Big blinds: nothing.
    for b in [BlindKind::Small, BlindKind::Big] {
        assert!(fire(&mut js, &blind_view(b, false), fire_blind_select).is_empty());
    }
    // Debuffed Chicot: nothing.
    let mut js = vec![jk(1, J::Chicot).with_debuff(true)];
    assert!(
        fire(
            &mut js,
            &blind_view(BlindKind::Boss, true),
            fire_blind_select
        )
        .is_empty()
    );
}

#[test]
fn chicot_acquired_during_an_active_boss_disables_it() {
    let mut js = jokers(&[J::Joker, J::Chicot]);
    let acq =
        |js: &mut [JokerInstance], run: &RunView<'_>| fire(js, run, |inp| fire_acquire(inp, 1));
    assert_eq!(
        acq(&mut js, &blind_view(BlindKind::Boss, true)),
        vec![SideEffect::DisableBoss]
    );
    assert!(acq(&mut js, &blind_view(BlindKind::Boss, false)).is_empty());
    assert!(acq(&mut js, &blind_view(BlindKind::Small, false)).is_empty());
}

#[test]
fn chicot_is_not_copied_by_blueprint() {
    // Blueprint aimed at Chicot: still exactly one DisableBoss (Chicot's own).
    let mut js = jokers(&[J::Blueprint, J::Chicot]);
    let fx = fire(
        &mut js,
        &blind_view(BlindKind::Boss, true),
        fire_blind_select,
    );
    assert_eq!(fx, vec![SideEffect::DisableBoss]);
    let mut js = jokers(&[J::Chicot, J::Brainstorm]);
    let fx = fire(
        &mut js,
        &blind_view(BlindKind::Boss, true),
        fire_blind_select,
    );
    assert_eq!(fx, vec![SideEffect::DisableBoss]);
}

#[test]
fn chicot_in_the_game_disables_each_boss_and_only_bosses() {
    let mut g = new_game(61);
    add(&mut g, J::Chicot);
    select(&mut g);
    assert!(!g.state().boss_disabled, "Small blind");
    beat_blind(&mut g);
    end_shop(&mut g);
    select(&mut g);
    assert!(!g.state().boss_disabled, "Big blind");
    beat_blind(&mut g);
    end_shop(&mut g);
    g.state_mut().boss = Some(BossId::TheWall);
    select(&mut g);
    assert!(g.state().boss_disabled);
    assert!(!g.state().run_view().boss.active);
    // Beat the boss: ante 2 starts with the flag cleared, and the next boss is disabled again.
    beat_blind(&mut g);
    assert_eq!(g.state().ante, 2);
    assert!(!g.state().boss_disabled);
    end_shop(&mut g);
    advance_to_boss_select(&mut g);
    select(&mut g);
    assert!(g.state().boss_disabled);
}

#[test]
fn chicot_in_the_game_added_in_the_shop_before_the_boss_does_not_disable_it() {
    let mut g = new_game(63);
    select(&mut g);
    beat_blind(&mut g);
    end_shop(&mut g);
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    assert_eq!(g.state().blind, BlindKind::Boss);
    g.state_mut().boss = Some(BossId::TheWall);
    add(&mut g, J::Chicot);
    assert!(!g.state().boss_disabled);
    // The blind-select path still disables it when the boss is selected.
    end_shop(&mut g);
    select(&mut g);
    assert!(g.state().boss_disabled);
}

#[test]
fn chicot_in_the_game_added_mid_boss_disables_it() {
    let mut g = new_game(62);
    advance_to_boss_select(&mut g);
    g.state_mut().boss = Some(BossId::TheWall);
    select(&mut g);
    assert!(g.state().run_view().boss.active);
    add(&mut g, J::Chicot);
    assert!(g.state().boss_disabled);
}
