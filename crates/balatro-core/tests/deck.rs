use balatro_core::card::Card;
use balatro_core::deck::*;
use balatro_core::enums::*;
use balatro_core::rng::GameRng;
use proptest::prelude::*;
use std::collections::HashSet;

fn uids(d: &Deck) -> Vec<u32> {
    let mut v: Vec<u32> = d.iter_all().map(|c| c.uid).collect();
    v.sort();
    v
}

#[test]
fn standard_unique() {
    let d = Deck::standard();
    assert_eq!(d.len(), 52);
    let s: HashSet<u32> = d.iter_all().map(|c| c.uid).collect();
    assert_eq!(s.len(), 52);
    let combos: HashSet<(Rank, Suit)> = d.iter_all().map(|c| (c.rank, c.suit)).collect();
    assert_eq!(combos.len(), 52);
    assert_eq!(d.next_uid(), 52);
}

#[test]
fn draw_short_pile() {
    let mut d = Deck::standard();
    d.draw_pile.truncate(5);
    assert_eq!(d.draw(8), 5);
    assert_eq!(d.hand.len(), 5);
    assert_eq!(d.draw(1), 0);
    assert_eq!(d.draw_to_hand_size(8), 0);
}

#[test]
fn draw_takes_from_end() {
    let mut d = Deck::standard();
    let top = d.draw_pile.last().unwrap().uid;
    d.draw(1);
    assert_eq!(d.hand[0].uid, top);
}

#[test]
fn mask_errors_and_order() {
    let mut d = Deck::standard();
    d.draw(8);
    assert_eq!(d.take_from_hand(0), Err(DeckError::EmptyMask));
    assert!(matches!(
        d.take_from_hand(1 << 8),
        Err(DeckError::MaskOutOfRange { .. })
    ));
    assert!(matches!(
        d.take_from_hand(0x8000),
        Err(DeckError::MaskOutOfRange { .. })
    ));
    assert_eq!(d.hand.len(), 8);
    let before: Vec<u32> = d.hand.iter().map(|c| c.uid).collect();
    let t = d.take_from_hand(0b1010_0001).unwrap();
    assert_eq!(
        t.iter().map(|c| c.uid).collect::<Vec<_>>(),
        vec![before[0], before[5], before[7]]
    );
    assert_eq!(d.hand.len(), 5);
    let x = d.discard_from_hand(0b1).unwrap();
    assert_eq!(d.discard_pile, x);
}

#[test]
fn sort_order() {
    let mut d = Deck::standard();
    d.draw_pile.clear();
    let mut stone = Card::new(100, Rank::Ace, Suit::Spades);
    stone.enhancement = Enhancement::Stone;
    d.hand = vec![
        Card::new(5, Rank::Two, Suit::Spades),
        stone,
        Card::new(4, Rank::King, Suit::Diamonds),
        Card::new(3, Rank::King, Suit::Spades),
        Card::new(2, Rank::King, Suit::Hearts),
        Card::new(1, Rank::King, Suit::Clubs),
        Card::new(9, Rank::King, Suit::Spades),
    ];
    d.sort_hand();
    let got: Vec<u32> = d.hand.iter().map(|c| c.uid).collect();
    assert_eq!(got, vec![3, 9, 2, 1, 4, 5, 100]);
}

#[test]
fn reset_deterministic_and_clears() {
    let mut a = Deck::standard();
    let mut b = Deck::standard();
    a.draw(8);
    a.hand[0].debuffed = true;
    a.hand[1].face_down = true;
    let (mut ra, mut rb) = (GameRng::new(7), GameRng::new(7));
    a.reset_for_round(&mut ra);
    b.reset_for_round(&mut rb);
    assert_eq!(a, b);
    assert!(a.iter_all().all(|c| !c.debuffed && !c.face_down));
    assert_eq!(a.draw_pile.len(), 52);
}

#[test]
fn remove_and_add() {
    let mut d = Deck::standard();
    d.draw(3);
    let uid = d.hand[1].uid;
    let c = d.remove_card(uid).unwrap();
    assert_eq!(d.len(), 51);
    assert!(d.remove_card(uid).is_none());
    let nu = d.alloc_uid();
    assert_eq!(nu, 52);
    d.add_card(Card::new(nu, Rank::Ace, Suit::Hearts), Pile::Draw);
    d.add_card(c, Pile::Discard);
    assert_eq!(d.len(), 53);
    assert_eq!(d.draw_pile.last().unwrap().uid, 52);
}

proptest! {
    #[test]
    fn conservation(ops in proptest::collection::vec((0u8..6, any::<u16>().prop_map(u32::from)), 0..60), seed in any::<u64>()) {
        let mut d = Deck::standard();
        let mut rng = GameRng::new(seed);
        d.reset_for_round(&mut rng);
        let orig = uids(&d);
        for (op, m) in ops {
            match op {
                0 => { d.draw_to_hand_size(8); }
                1 => { let _ = d.discard_from_hand(m); }
                2 => { if let Ok(t) = d.take_from_hand(m) { d.played.extend(t); } }
                3 => d.move_played_to_discard(),
                4 => d.reset_for_round(&mut rng),
                _ => { d.draw((m % 10) as usize); }
            }
            prop_assert_eq!(uids(&d), orig.clone());
            prop_assert!(d.hand.len() <= 52);
        }
    }
}
