use balatro_core::blinds::*;

#[test]
fn ante_table_and_low_antes() {
    let w = StakeScaling::White;
    let exp = [
        300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
    ];
    for (i, e) in exp.iter().enumerate() {
        assert_eq!(base_chips(i as i32 + 1, w), *e);
    }
    assert_eq!(base_chips(0, w), 100.0);
    assert_eq!(base_chips(-5, w), 100.0);
    assert_eq!(base_chips(-1, StakeScaling::Purple), 100.0);
    assert_eq!(base_chips(8, StakeScaling::Green), 100_000.0);
    assert_eq!(base_chips(5, StakeScaling::Purple), 25_000.0);
    assert_eq!(StakeScaling::default(), StakeScaling::White);
}

#[test]
fn endless_matches_published_table() {
    let t: [f64; 31] = [
        110_000.0,
        560_000.0,
        7_200_000.0,
        300_000_000.0,
        47_000_000_000.0,
        2.9e13,
        7.7e16,
        8.6e20,
        4.2e25,
        9.2e30,
        9.2e36,
        4.3e43,
        9.7e50,
        1.0e59,
        5.8e67,
        1.6e77,
        2.4e87,
        1.9e98,
        8.4e109,
        2.0e122,
        2.7e135,
        2.1e149,
        9.9e163,
        2.7e179,
        4.4e195,
        4.4e212,
        2.8e230,
        1.1e249,
        2.7e268,
        4.5e288,
        f64::INFINITY,
    ];
    for (i, e) in t.iter().enumerate() {
        let ante = 9 + i as i32;
        assert_eq!(base_chips(ante, StakeScaling::White), *e, "ante {ante}");
    }
    assert!(base_chips(100, StakeScaling::White).is_infinite());
}

#[test]
fn requirements() {
    let w = StakeScaling::White;
    assert_eq!(requirement(1, BlindKind::Small, 2.0, w), 300.0);
    assert_eq!(requirement(1, BlindKind::Big, 2.0, w), 450.0);
    assert_eq!(requirement(1, BlindKind::Boss, MULT_BOSS_DEFAULT, w), 600.0);
    assert_eq!(
        requirement(8, BlindKind::Boss, MULT_BOSS_DEFAULT, w),
        100_000.0
    );
    assert_eq!(requirement(2, BlindKind::Boss, MULT_WALL, w), 3200.0);
    assert_eq!(
        requirement(2, BlindKind::Boss, MULT_VIOLET_VESSEL, w),
        4800.0
    );
    assert_eq!(requirement(2, BlindKind::Boss, MULT_NEEDLE, w), 800.0);
    assert_eq!(requirement(0, BlindKind::Boss, 2.0, w), 200.0);
    // Endless: truncate base first, then multiply.
    assert_eq!(requirement(9, BlindKind::Big, 2.0, w), 165_000.0);
    assert_eq!(requirement(9, BlindKind::Boss, 2.0, w), 220_000.0);
}

#[test]
fn rewards() {
    use BlindKind::*;
    assert_eq!(blind_reward(Small, false, false), 3);
    assert_eq!(blind_reward(Big, false, false), 4);
    assert_eq!(blind_reward(Boss, false, false), 5);
    assert_eq!(blind_reward(Boss, true, false), 8);
    assert_eq!(blind_reward(Small, false, true), 0);
    assert_eq!(blind_reward(Big, false, true), 4);
}
