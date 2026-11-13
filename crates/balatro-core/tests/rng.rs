use balatro_core::rng::*;

fn take(r: &mut GameRng, k: RngKey, n: usize) -> Vec<u64> {
    (0..n).map(|_| r.next_u64(k)).collect()
}

#[test]
fn fnv_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
    assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
}

#[test]
fn same_seed_same_sequence() {
    let mut a = GameRng::new(42);
    let mut b = GameRng::new(42);
    for k in RngKey::ALL {
        assert_eq!(take(&mut a, k, 50), take(&mut b, k, 50));
    }
    let mut c = GameRng::new(43);
    let mut d = GameRng::new(42);
    assert_ne!(take(&mut c, RngKey::Deck, 5), take(&mut d, RngKey::Deck, 5));
    assert_eq!(GameRng::from_seed_str("abc").seed(), fnv1a64(b"abc"));
}

#[test]
fn different_keys_differ() {
    let mut r = GameRng::new(1);
    let seqs: Vec<_> = RngKey::ALL.iter().map(|&k| take(&mut r, k, 4)).collect();
    for i in 0..seqs.len() {
        for j in i + 1..seqs.len() {
            assert_ne!(seqs[i], seqs[j]);
        }
    }
}

#[test]
fn stream_independence() {
    let mut a = GameRng::new(9);
    let mut b = GameRng::new(9);
    for _ in 0..1000 {
        a.next_u64(RngKey::Shop);
    }
    assert_eq!(
        take(&mut a, RngKey::Deck, 20),
        take(&mut b, RngKey::Deck, 20)
    );
}

#[test]
fn json_roundtrip() {
    let mut a = GameRng::new(5);
    take(&mut a, RngKey::Cards, 17);
    let j = serde_json::to_string(&a).unwrap();
    let mut b: GameRng = serde_json::from_str(&j).unwrap();
    assert_eq!(a, b);
    for k in RngKey::ALL {
        assert_eq!(take(&mut a, k, 30), take(&mut b, k, 30));
    }
}

#[test]
fn chance_edges() {
    let mut r = GameRng::new(3);
    let k = RngKey::Cards;
    for _ in 0..1000 {
        assert!(!r.chance(k, 0, 5, 1));
        assert!(r.chance(k, 5, 5, 1));
        assert!(r.chance(k, 1, 2, 2)); // doubled -> 1.0
        assert!(r.chance(k, 1, 2, 100)); // capped
        assert!(!r.chance(k, 1, 0, 1));
    }
    let n = 100_000;
    let hits = (0..n).filter(|_| r.chance(k, 1, 4, 1)).count() as f64 / n as f64;
    assert!((hits - 0.25).abs() < 0.01);
    let hits = (0..n).filter(|_| r.chance(k, 1, 4, 2)).count() as f64 / n as f64;
    assert!((hits - 0.5).abs() < 0.01);
}

#[test]
fn weighted() {
    let mut r = GameRng::new(11);
    let k = RngKey::Shop;
    assert_eq!(r.weighted_index(k, &[]), None);
    assert_eq!(r.weighted_index(k, &[0.0, -1.0, f64::NAN]), None);
    assert_eq!(r.weighted_index(k, &[0.0, 2.0, -3.0]), Some(1));
    let w = [1.0, 0.0, 3.0, 6.0];
    let n = 100_000;
    let mut c = [0usize; 4];
    for _ in 0..n {
        c[r.weighted_index(k, &w).unwrap()] += 1;
    }
    assert_eq!(c[1], 0);
    for (i, p) in [0.1, 0.0, 0.3, 0.6].iter().enumerate() {
        assert!((c[i] as f64 / n as f64 - p).abs() < 0.01, "{i}: {c:?}");
    }
}

#[test]
fn shuffle_and_helpers() {
    let mut r = GameRng::new(8);
    let mut v: Vec<u32> = (0..52).collect();
    r.shuffle(RngKey::Deck, &mut v);
    assert_ne!(v, (0..52).collect::<Vec<_>>());
    v.sort();
    assert_eq!(v, (0..52).collect::<Vec<_>>());
    let mut e: [u8; 0] = [];
    r.shuffle(RngKey::Deck, &mut e);
    assert_eq!(r.choose_index(RngKey::Misc, 0), None);
    assert!(r.choose_index(RngKey::Misc, 3).unwrap() < 3);
    for _ in 0..1000 {
        let f = r.float(RngKey::Misc);
        assert!((0.0..1.0).contains(&f));
    }
    let mut counts = [0usize; 6];
    for _ in 0..60_000 {
        counts[r.range(RngKey::Misc, 0, 6)] += 1;
    }
    for c in counts {
        assert!((c as f64 - 10_000.0).abs() < 500.0);
    }
}
