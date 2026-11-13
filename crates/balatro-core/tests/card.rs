use balatro_core::card::*;
use balatro_core::enums::*;

#[test]
fn chips() {
    assert_eq!(Card::new(0, Rank::Ace, Suit::Spades).base_chips(), 11);
    assert_eq!(Card::new(0, Rank::Five, Suit::Spades).base_chips(), 5);
    let mut s = Card::new(0, Rank::Ace, Suit::Spades);
    s.enhancement = Enhancement::Stone;
    assert_eq!(s.base_chips(), 0);
    assert_eq!(s.effective_rank(), None);
    assert!(s.is_stone());
}

#[test]
fn face() {
    let none = RuleFlags::default();
    let pareidolia = RuleFlags {
        pareidolia: true,
        ..none
    };
    let k = Card::new(0, Rank::King, Suit::Hearts);
    let two = Card::new(1, Rank::Two, Suit::Hearts);
    let mut stone = k;
    stone.enhancement = Enhancement::Stone;
    assert!(k.is_face(&none) && !two.is_face(&none));
    assert!(k.is_face(&pareidolia) && two.is_face(&pareidolia));
    assert!(!stone.is_face(&none) && !stone.is_face(&pareidolia));
}

#[test]
fn suits() {
    let none = RuleFlags::default();
    let smeared = RuleFlags {
        smeared: true,
        ..none
    };
    let h = Card::new(0, Rank::Two, Suit::Hearts);
    assert!(h.has_suit(Suit::Hearts, &none));
    assert!(!h.has_suit(Suit::Diamonds, &none));
    assert!(h.has_suit(Suit::Diamonds, &smeared));
    assert!(!h.has_suit(Suit::Spades, &smeared));
    let s = Card::new(0, Rank::Two, Suit::Spades);
    assert!(s.has_suit(Suit::Clubs, &smeared));
    assert!(!s.has_suit(Suit::Hearts, &smeared));
    let mut w = h;
    w.enhancement = Enhancement::Wild;
    for suit in Suit::ALL {
        assert!(w.has_suit(suit, &none));
    }
    // Debuffed Wild reverts to its printed suit (h is Hearts).
    w.debuffed = true;
    assert!(w.has_suit(Suit::Hearts, &none));
    assert!(!w.has_suit(Suit::Spades, &none));
    assert!(w.has_suit(Suit::Diamonds, &smeared));
    let mut st = h;
    st.enhancement = Enhancement::Stone;
    for suit in Suit::ALL {
        assert!(!st.has_suit(suit, &smeared));
    }
}

#[test]
fn serde_roundtrip() {
    let mut c = Card::new(7, Rank::Queen, Suit::Clubs);
    c.enhancement = Enhancement::Glass;
    c.edition = Edition::Foil;
    c.seal = Seal::Red;
    c.perma_bonus = 6;
    let j = serde_json::to_string(&c).unwrap();
    assert_eq!(serde_json::from_str::<Card>(&j).unwrap(), c);
    let f = RuleFlags {
        smeared: true,
        ..Default::default()
    };
    let j = serde_json::to_string(&f).unwrap();
    assert_eq!(serde_json::from_str::<RuleFlags>(&j).unwrap(), f);
}
