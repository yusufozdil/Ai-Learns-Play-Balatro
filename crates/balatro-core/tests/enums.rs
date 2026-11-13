use balatro_core::enums::*;

fn bijection(idx: Vec<usize>, count: usize) {
    assert_eq!(idx.len(), count);
    for (i, v) in idx.iter().enumerate() {
        assert_eq!(*v, i);
    }
}

#[test]
fn index_bijections() {
    bijection(Suit::ALL.iter().map(|x| x.index()).collect(), Suit::COUNT);
    bijection(Rank::ALL.iter().map(|x| x.index()).collect(), Rank::COUNT);
    bijection(
        HandType::ALL.iter().map(|x| x.index()).collect(),
        HandType::COUNT,
    );
    bijection(
        Enhancement::ALL.iter().map(|x| x.index()).collect(),
        Enhancement::COUNT,
    );
    bijection(
        Edition::ALL.iter().map(|x| x.index()).collect(),
        Edition::COUNT,
    );
    bijection(Seal::ALL.iter().map(|x| x.index()).collect(), Seal::COUNT);
    bijection(Phase::ALL.iter().map(|x| x.index()).collect(), Phase::COUNT);
    bijection(
        Rarity::ALL.iter().map(|x| x.index()).collect(),
        Rarity::COUNT,
    );
    assert_eq!(HandType::COUNT, 12);
    assert_eq!(Rank::COUNT, 13);
}

#[test]
fn chip_values_and_from_value() {
    assert_eq!(Rank::Two.chip_value(), 2);
    assert_eq!(Rank::Ten.chip_value(), 10);
    assert_eq!(Rank::Jack.chip_value(), 10);
    assert_eq!(Rank::King.chip_value(), 10);
    assert_eq!(Rank::Ace.chip_value(), 11);
    assert!(Rank::Queen.is_face_rank() && !Rank::Ten.is_face_rank());
    assert_eq!(Rank::from_value(14), Some(Rank::Ace));
    assert_eq!(Rank::from_value(2), Some(Rank::Two));
    assert_eq!(Rank::from_value(1), None);
    assert_eq!(Rank::from_value(15), None);
    for r in Rank::ALL {
        assert_eq!(Rank::from_value(r as u8), Some(r));
    }
}

#[test]
fn ordering_and_defaults() {
    assert!(HandType::FlushFive > HandType::FiveOfAKind);
    assert!(HandType::Pair > HandType::HighCard);
    assert!(Rank::Ace > Rank::King);
    assert_eq!(Enhancement::default(), Enhancement::None);
    assert_eq!(Edition::default(), Edition::Base);
    assert!(Suit::Hearts.is_red() && Suit::Diamonds.is_red());
    assert!(!Suit::Spades.is_red() && !Suit::Clubs.is_red());
}
