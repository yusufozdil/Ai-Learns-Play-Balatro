use balatro_core::blinds::BlindKind;
use balatro_core::data::prices::*;
use balatro_core::economy::*;

#[test]
fn interest_cases() {
    let b = INTEREST_CAP_BASE;
    for (m, e) in [
        (0, 0),
        (4, 0),
        (5, 1),
        (24, 4),
        (25, 5),
        (26, 5),
        (100, 5),
        (-10, 0),
    ] {
        assert_eq!(interest(m, b, 0), e, "money {m}");
    }
    assert_eq!(interest(50, INTEREST_CAP_SEED_MONEY, 0), 10);
    assert_eq!(interest(100, INTEREST_CAP_SEED_MONEY, 0), 10);
    assert_eq!(interest(100, INTEREST_CAP_MONEY_TREE, 0), 20);
    assert_eq!(interest(1000, INTEREST_CAP_MONEY_TREE, 0), 20);
    // To the Moon: bonus capped by the same step cap.
    assert_eq!(interest(24, b, TO_THE_MOON_BONUS), 8);
    assert_eq!(interest(100, b, TO_THE_MOON_BONUS), 10);
    assert_eq!(
        interest(100, INTEREST_CAP_MONEY_TREE, TO_THE_MOON_BONUS),
        40
    );
    assert_eq!(interest(-3, b, TO_THE_MOON_BONUS), 0);
}

#[test]
fn hands_and_sell() {
    assert_eq!(hands_reward(3), 3);
    assert_eq!(hands_reward(0), 0);
    assert_eq!(hands_reward(-1), 0);
    assert_eq!(discards_reward(4), 0);
    assert_eq!(sell_value(1, 0), 1);
    assert_eq!(sell_value(3, 0), 1);
    assert_eq!(sell_value(5, 0), 2);
    assert_eq!(sell_value(6, 0), 3);
    assert_eq!(sell_value(5, 3), 5);
    assert_eq!(sell_value(0, 0), 1);
}

#[test]
fn prices() {
    use Discount::*;
    assert_eq!(shop_price(6, 0, None), 6);
    assert_eq!(shop_price(6, 0, Clearance), 4); // 4.5 rounds half down
    assert_eq!(shop_price(6, 0, Liquidation), 3);
    assert_eq!(shop_price(5, 0, Clearance), 4); // 3.75
    assert_eq!(shop_price(1, 0, Liquidation), 1); // min $1
    assert_eq!(shop_price(1, 0, Clearance), 1);
    assert_eq!(shop_price(4, EDITION_FOIL, None), 6);
    assert_eq!(shop_price(4, EDITION_HOLO, Clearance), 5); // 5.25
    assert_eq!(shop_price(4, EDITION_POLY, Liquidation), 4); // 4.5 -> 4
    assert_eq!(shop_price(PACK_MEGA, 0, Liquidation), 4);
    assert_eq!(shop_price(VOUCHER, 0, Clearance), 7); // 7.5 -> 7
    assert_eq!(shop_price(TAROT, EDITION_NEGATIVE, None), 8);
    assert_eq!(shop_price(SPECTRAL, 0, Clearance), 3);
    assert_eq!(shop_price(PACK_NORMAL, 0, None), 4);
    assert_eq!(shop_price(PACK_JUMBO, 0, None), 6);
}

#[test]
fn rerolls() {
    let seq = |base| (0..4).map(|n| reroll_cost(n, base)).collect::<Vec<_>>();
    assert_eq!(seq(REROLL_BASE), [5, 6, 7, 8]);
    assert_eq!(seq(REROLL_SURPLUS_BASE), [3, 4, 5, 6]);
    assert_eq!(seq(REROLL_GLUT_BASE), [1, 2, 3, 4]);
    assert_eq!(seq(0), [0, 1, 2, 3]);
    assert_eq!(reroll_cost(0, -2), 0);
}

#[test]
fn cash_out_order() {
    let input = CashOutInput {
        money: 22,
        gold_cards_held: 1,
        gold_red_seal_cards_held: 0,
        kind: BlindKind::Boss,
        is_showdown: false,
        stake_no_small_reward: false,
        no_blind_reward: false,
        hands_left: 2,
        interest_cap: INTEREST_CAP_BASE,
        interest_per_5_bonus: 0,
        no_interest: false,
        joker_payout: 4,
    };
    let c = cash_out(&input);
    assert_eq!(c.gold_cards, 3);
    assert_eq!(c.interest_base, 25); // 22 + 3, not including the $5 reward
    assert_eq!(c.interest, 5);
    assert_eq!(c.blind_reward, 5);
    assert_eq!(c.hands, 2);
    assert_eq!(c.jokers, 4);
    assert_eq!(c.total, 19);
    assert_eq!(c.final_money, 41);

    let low = CashOutInput {
        money: 9,
        gold_cards_held: 0,
        joker_payout: 0,
        hands_left: 0,
        ..input
    };
    // Base 9 -> $1; had the $5 reward counted (14) it would be $2.
    assert_eq!(cash_out(&low).interest, 1);
    let nm = CashOutInput {
        no_blind_reward: true,
        no_interest: true,
        ..low
    };
    let c = cash_out(&nm);
    assert_eq!((c.interest, c.blind_reward, c.total), (0, 0, 0));
    let red = CashOutInput {
        gold_red_seal_cards_held: 2,
        ..low
    };
    assert_eq!(cash_out(&red).gold_cards, 12);
}
