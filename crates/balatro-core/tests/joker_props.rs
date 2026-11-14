//! Phase 3.R property tests: random boards of 0-5 DEFINED jokers (random editions, debuffs and
//! states) with random 1-5 card hands and random held cards (enhancements, seals, editions).
//! The hand is evaluated with the real `hand_eval::evaluate` under the jokers' rule flags, as
//! the game layer does. Properties: no panic, a finite non-negative integral score, identical
//! results for identical inputs, and well-formed side effects. A second property fires every
//! non-scoring event on random boards.

use balatro_core::card::Card;
use balatro_core::content::jokers::{JokerId, all_defs};
use balatro_core::deck::{Deck, Pile};
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::hand_eval::evaluate;
use balatro_core::hooks::{
    self, BlindKind, ConsumableKind, ConsumableView, DeckCounts, DiscardInfo, EventInput, PackKind,
    RoundEndInfo, RunView, SideEffect, SoldItem, collect_modifiers,
};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, ScoreResult, Scorer};
use proptest::prelude::*;

/// Every joker id that has a definition (the property must hold for implemented jokers).
fn defined_ids() -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs().map(|d| d.id).collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

fn arb_card() -> impl Strategy<Value = Card> {
    (
        0usize..Rank::COUNT,
        0usize..Suit::COUNT,
        0usize..Enhancement::COUNT,
        // Negative is never put on playing cards (notes.md §4.1).
        prop::sample::select(vec![
            Edition::Base,
            Edition::Base,
            Edition::Foil,
            Edition::Holographic,
            Edition::Polychrome,
        ]),
        0usize..Seal::COUNT,
        prop::bool::weighted(0.1),
        prop::sample::select(vec![0i32, 0, 5, 30]),
    )
        .prop_map(|(r, s, e, ed, se, deb, perma)| {
            let mut c = Card::new(0, Rank::ALL[r], Suit::ALL[s]);
            c.enhancement = Enhancement::ALL[e];
            c.edition = ed;
            c.seal = Seal::ALL[se];
            c.debuffed = deb;
            c.perma_bonus = perma;
            c
        })
}

/// (index into `defined_ids()`, edition, debuffed, Some(state) = randomised value/counter).
type JokerSpec = (usize, usize, bool, Option<(f64, u32)>);

fn arb_joker() -> impl Strategy<Value = JokerSpec> {
    (
        any::<usize>(),
        0usize..Edition::COUNT,
        prop::bool::weighted(0.1),
        prop::option::weighted(0.4, (0.0f64..8.0, 0u32..30)),
    )
}

fn build_jokers(specs: &[JokerSpec], ids: &[JokerId]) -> Vec<JokerInstance> {
    specs
        .iter()
        .enumerate()
        .map(|(i, &(k, ed, deb, state))| {
            let mut j = JokerInstance::new(100 + i as u32, ids[k % ids.len()]);
            j.edition = Edition::ALL[ed];
            j.debuffed = deb;
            if let Some((v, n)) = state {
                j.value = v;
                j.counter = n;
            }
            j
        })
        .collect()
}

fn number(cards: Vec<Card>, first_uid: u32) -> Vec<Card> {
    cards
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.uid = first_uid + i as u32;
            c
        })
        .collect()
}

type Snapshot = (ScoreResult, Vec<SideEffect>, Vec<JokerInstance>, Vec<Card>);

fn score_once(
    played: &[Card],
    held: &[Card],
    js: &[JokerInstance],
    run: &RunView<'_>,
    seed: u64,
) -> Snapshot {
    let mut played = played.to_vec();
    let mut js = js.to_vec();
    let eval = evaluate(&played, &run.rules);
    let mut rng = GameRng::new(seed);
    let mut scorer = Scorer::new();
    let res = scorer.score_hand(ScoreInput {
        played: &mut played,
        held,
        eval,
        jokers: &mut js,
        run,
        rng: &mut rng,
    });
    (res, scorer.effects().to_vec(), js, played)
}

/// Run facts derived from a few random knobs plus the board's passive modifiers.
#[allow(clippy::too_many_arguments)]
fn run_view<'a>(
    js: &[JokerInstance],
    played: &[Card],
    held: &[Card],
    consumables: &'a [ConsumableView],
    money: i32,
    hands_left: u32,
    discards_left: u32,
    levels: [u32; HandType::COUNT],
    plays: [u32; HandType::COUNT],
    flags: u8,
) -> RunView<'a> {
    let mods = collect_modifiers(js);
    let eval = evaluate(played, &mods.rules);
    let ht = eval.hand_type.index();
    // Play counters include the current hand.
    let mut plays_run = plays;
    plays_run[ht] += 1;
    let mut plays_round = [0; HandType::COUNT];
    plays_round[ht] = 1 + u32::from(flags & 1);
    let negatives = js.iter().filter(|j| j.edition == Edition::Negative).count() as u32;
    let mut run = RunView {
        money,
        ante: 1 + u32::from(flags >> 6),
        hands_left,
        discards_left,
        hands_played_round: 1 + u32::from(flags & 1),
        hands_played_run: 1 + plays.iter().sum::<u32>(),
        discards_used_round: u32::from(flags >> 1 & 1),
        hand_levels: levels,
        hand_plays_run: plays_run,
        hand_plays_round: plays_round,
        joker_slots: 5 + negatives,
        consumables,
        deck: DeckCounts::from_cards(played.iter().chain(held), 30, 52),
        rules: mods.rules,
        prob_mult: mods.prob_mult,
        ..RunView::default()
    };
    run.blind =
        [BlindKind::Small, BlindKind::Big, BlindKind::Boss][usize::from(flags >> 2 & 3) % 3];
    run.boss.halve_base = flags & 16 != 0;
    run.boss.triggered = flags & 32 != 0;
    run.stats.blinds_skipped = u32::from(flags & 3);
    run.stats.tarots_used = u32::from(flags >> 4 & 3);
    run
}

fn check_effects(
    effects: &[SideEffect],
    played: &[Card],
    held: &[Card],
    js: &[JokerInstance],
) -> Result<(), TestCaseError> {
    for e in effects {
        match *e {
            SideEffect::DestroyCard { uid } => prop_assert!(
                played.iter().chain(held).any(|c| c.uid == uid),
                "DestroyCard of unknown uid {uid}"
            ),
            SideEffect::DestroyJoker { uid } | SideEffect::CopyJoker { uid, .. } => prop_assert!(
                js.iter().any(|j| j.uid == uid),
                "joker effect on unknown uid {uid}"
            ),
            SideEffect::LevelUpHand { levels, .. } => prop_assert!(levels != 0),
            _ => {}
        }
    }
    Ok(())
}

const CONSUMABLES: [ConsumableView; 2] = [
    ConsumableView {
        kind: ConsumableKind::Planet,
        id: 1,
        planet_hand: Some(HandType::Pair),
        negative: false,
    },
    ConsumableView {
        kind: ConsumableKind::Tarot,
        id: 2,
        planet_hand: None,
        negative: false,
    },
];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn random_boards_score_finite_and_deterministic(
        played in prop::collection::vec(arb_card(), 1..=5),
        held in prop::collection::vec(arb_card(), 0..=8),
        specs in prop::collection::vec(arb_joker(), 0..=5),
        money in -20i32..120,
        hands_left in 0u32..4,
        discards_left in 0u32..5,
        levels in prop::array::uniform12(1u32..5),
        plays in prop::array::uniform12(0u32..6),
        flags in any::<u8>(),
        n_consumables in 0usize..=2,
        seed in any::<u64>(),
    ) {
        let ids = defined_ids();
        let played = number(played, 1);
        let held = number(held, 20);
        let js = build_jokers(&specs, &ids);
        let run = run_view(
            &js, &played, &held, &CONSUMABLES[..n_consumables], money, hands_left,
            discards_left, levels, plays, flags,
        );

        let a = score_once(&played, &held, &js, &run, seed);
        let b = score_once(&played, &held, &js, &run, seed);
        let (res, effects, js_after, played_after) = &a;
        prop_assert!(res.score.is_finite() && res.score >= 0.0, "score {:?}", res);
        prop_assert!(res.chips.is_finite() && res.mult.is_finite(), "{:?}", res);
        prop_assert!(!res.chips.is_nan() && !res.mult.is_nan());
        prop_assert_eq!(res.score, res.score.floor());
        prop_assert!(!res.invalid);
        prop_assert_eq!(&a, &b, "same input, different result");

        // Shape: scoring never adds or removes played cards or jokers, keeps uids and ids.
        prop_assert_eq!(played_after.len(), played.len());
        prop_assert!(played_after.iter().zip(&played).all(|(x, y)| x.uid == y.uid));
        prop_assert_eq!(js_after.len(), js.len());
        prop_assert!(js_after.iter().zip(&js).all(|(x, y)| x.uid == y.uid && x.id == y.id));
        // Copiers have no state of their own: a copy's writes are always discarded.
        for (x, y) in js_after.iter().zip(&js) {
            if matches!(x.id, JokerId::Blueprint | JokerId::Brainstorm) {
                prop_assert_eq!(x, y);
            }
        }
        // Debuffed jokers never change.
        for (x, y) in js_after.iter().zip(&js) {
            if y.debuffed {
                prop_assert_eq!(x, y);
            }
        }
        check_effects(effects, &played, &held, &js)?;
    }

    #[test]
    fn random_boards_survive_every_event(
        specs in prop::collection::vec(arb_joker(), 0..=5),
        discarded in prop::collection::vec(arb_card(), 1..=5),
        money in -20i32..120,
        flags in any::<u8>(),
        seed in any::<u64>(),
    ) {
        let ids = defined_ids();
        let js0 = build_jokers(&specs, &ids);
        let deck = Deck::standard();
        let discarded = number(discarded, 60);
        let run = run_view(
            &js0, &discarded, &[], &CONSUMABLES, money, 1, 1, [1; 12], [0; 12], flags,
        );

        let fire_all = |js: &mut Vec<JokerInstance>| {
            let mut rng = GameRng::new(seed);
            let mut fx: Vec<SideEffect> = Vec::new();
            let mut money_total = 0i32;
            macro_rules! inp {
                () => {
                    EventInput { jokers: js, run: &run, deck: &deck, rng: &mut rng, effects: &mut fx }
                };
            }
            for slot in 0..js.len() {
                money_total += hooks::fire_acquire(inp!(), slot).money_delta;
            }
            money_total += hooks::fire_blind_select(inp!()).money_delta;
            money_total += hooks::fire_first_hand_drawn(inp!()).money_delta;
            let d = DiscardInfo {
                cards: &discarded,
                hand_type: evaluate(&discarded, &run.rules).hand_type,
            };
            money_total += hooks::fire_discard(inp!(), &d).money_delta;
            money_total += hooks::fire_cards_destroyed(inp!(), &discarded).money_delta;
            money_total += hooks::fire_cards_added(inp!(), &discarded).money_delta;
            let end = RoundEndInfo { won: flags & 1 == 0, saved: false, round_score: 100.0, target: 300.0 };
            money_total += hooks::fire_round_end(inp!(), &end).money_delta;
            money_total += hooks::fire_shop_enter(inp!()).money_delta;
            money_total += hooks::fire_reroll(inp!()).money_delta;
            money_total += hooks::fire_pack_open(inp!(), PackKind::Arcana).money_delta;
            money_total += hooks::fire_pack_skip(inp!(), PackKind::Buffoon).money_delta;
            money_total += hooks::fire_consumable_used(inp!(), &CONSUMABLES[0]).money_delta;
            money_total += hooks::fire_shop_exit(inp!()).money_delta;
            money_total += hooks::fire_blind_skip(inp!()).money_delta;
            if !js.is_empty() {
                let sold = js[0];
                money_total += hooks::fire_sell_self(inp!(), 0).money_delta;
                money_total +=
                    hooks::fire_sell_other(inp!(), &SoldItem::Joker(sold), Some(0)).money_delta;
            }
            (money_total, fx, rng)
        };

        let mut a_js = js0.clone();
        let mut b_js = js0.clone();
        let a = fire_all(&mut a_js);
        let b = fire_all(&mut b_js);
        prop_assert_eq!(&a_js, &b_js);
        prop_assert_eq!(a.0, b.0);
        prop_assert_eq!(&a.1, &b.1);
        prop_assert_eq!(a.2, b.2);
        prop_assert_eq!(a_js.len(), js0.len());
        for (x, y) in a_js.iter().zip(&js0) {
            if matches!(x.id, JokerId::Blueprint | JokerId::Brainstorm) || y.debuffed {
                prop_assert_eq!(x, y);
            }
            prop_assert!(x.value.is_finite());
        }
        for e in &a.1 {
            match *e {
                SideEffect::DestroyCard { uid } => {
                    prop_assert!(discarded.iter().any(|c| c.uid == uid));
                }
                SideEffect::DestroyJoker { uid } | SideEffect::CopyJoker { uid, .. } => {
                    prop_assert!(js0.iter().any(|j| j.uid == uid));
                }
                SideEffect::AddCard { to, .. } => {
                    prop_assert!(matches!(to, Pile::Draw | Pile::Hand));
                }
                _ => {}
            }
        }
    }
}
