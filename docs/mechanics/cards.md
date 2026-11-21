# Playing cards: chips, enhancements, editions, seals, deck

Sources:
- C1 https://balatrowiki.org/w/Card_Ranks
- C2 https://balatrowiki.org/w/Card_Modifiers
- C3 https://balatrowiki.org/w/Glass_Card
- C4 https://balatrowiki.org/w/Stone_Card
- C5 https://balatrowiki.org/w/Gold_Seal
- C6 https://balatrowiki.org/w/Wild_Card
- C7 https://balatrowiki.org/w/Debuffed
- C8 https://balatrowiki.org/w/Decks and https://balatrowiki.org/w/Red_Deck
- C9 https://balatrowiki.org/w/Jokers
- C10 https://balatrowiki.org/w/Blue_Deck (default 4 hands)
- C11 secondary guides for starting values: https://balatrocalc.pages.dev/blog/balatro-gameplay and https://blizzardwatch.com/2025/01/23/balatro-beginner-intermediate-tips/

## Chip values (C1)

| Rank | Chips |
|---|---|
| 2-10 | face value (2..10) |
| J, Q, K | 10 |
| A | 11 (counts as low or high in straights; always scores 11) |

Stone cards have no rank chip, they give +50 as their enhancement (C4).

## Card = one rank + one suit + at most one enhancement + one edition + one seal (C2)
Applying a new enhancement or seal replaces the old one; editions cannot be overwritten (C2).

## Enhancements (C2 unless noted)

| Enhancement | Effect | When |
|---|---|---|
| Bonus | +30 chips (in addition to rank chips) | when played and scored |
| Mult | +4 mult | when played and scored |
| Wild | counts as any suit (all four simultaneously, C6) | detection (suit tests, flushes) |
| Glass | x2 mult; 1 in 4 chance to be destroyed | x2 on each scoring trigger; destruction rolled once after all scoring, retriggers do not re-roll (C3); debuffed Glass neither scores x2 nor breaks (C3) |
| Steel | x1.5 mult | while the card is held in hand (not played) during scoring |
| Stone | +50 chips, no rank, no suit; always scores; counts for no hand type | when played (always in scoring set) (C4); no base rank chips |
| Gold | $3 | if held in hand at end of round |
| Lucky | 1 in 5: +20 mult; 1 in 15: win $20 (independent rolls) | when played and scored; per-trigger rerolls are `UNVERIFIED` |

Notes:
- Wild: a debuffed Wild reverts to its base suit (C6, C7). Changing a Stone card's enhancement restores its original rank/suit (C4).
- All probabilities go through the shared `chance(num, den)` (Oops! All 6s doubles numerators) (C3).
- `Bonus` +30 and `Stone` +50 chips are additive chip bonuses, not "base chips" for Hiker purposes; Hiker's perma bonus adds on top of that (C4).

## Editions on playing cards (C2)

| Edition | Effect |
|---|---|
| Base | none |
| Foil | +50 chips when scored |
| Holographic | +10 mult when scored |
| Polychrome | x1.5 mult when scored |
| Negative | +1 joker slot (jokers) / +1 consumable slot (consumables) (C2). Not a playing-card edition in normal play; the evaluator enum may keep it, but cards will never carry it unless a later task adds a source. `UNVERIFIED` whether any in-game effect gives a playing card Negative; recommended: do not generate. |

Appearance rates in packs: Foil 4%, Holographic 2.8%, Polychrome 1.2% (x2 Hone, x4 Glow Up); enhancement 40%, seal 20% (C2; the wiki table is ambiguous, `UNVERIFIED`; only needed for pack generation).

## Seals (C2, C5)

| Seal | Effect | Timing |
|---|---|---|
| Gold | $3 | when the card is played and scores; once per trigger (each retrigger pays again) (C5); debuffed pays $0 (C5) |
| Red | retrigger the card 1 time | played-and-scored; also retriggers held-in-hand effects (scoring.md); not itself multiplied by jokers like Hack/Mime (C2) |
| Blue | creates the Planet card for the last played hand type of the round, needs a free consumable slot | card held in hand at end of round (C2, C5) |
| Purple | creates a Tarot card, needs a free consumable slot | when the card is discarded (C2, C5) |

Seals can be re-triggered by jokers (Sock and Buskin retriggering a Gold Seal face card pays again) (C2). The simulator enum has 5 seals including none: Gold, Red, Blue, Purple + none.

## Debuffed cards (C7, C2)
- Keep base rank and suit, still count for hand detection and "contains" (hands.md).
- Contribute no chips, mult, enhancement, edition or seal effects when scored; no perma bonus; Gold Seal pays 0.
- Stone debuffed: still scores (in the scoring set) but has no rank/suit.
- Held: held-card effects (Steel, Baron etc.) are nullified; jokers keyed to that specific card do not fire; hand-type or held-set jokers like Blackboard still work.
- Discarded: seal/edition/enhancement discard effects nullified; jokers keyed to the discard itself (Burnt Joker) still fire.
- Changing rank/suit (Tarot/Spectral) clears a rank/suit-specific debuff; changing enhancement clears The Pillar-type debuff; edition/seal changes do not (C7).
- Debuffing sources: The Club (Clubs), The Goad (Spades), The Head (Hearts), The Window (Diamonds), The Plant (face cards), The Pillar (cards played earlier this ante), Verdant Leaf (all, until a joker is sold) (C7).

## Face cards (C9, C1)
- Face cards are Jack, Queen, King (Ace is not a face card). Stone cards are not face cards by default (C4).
- Pareidolia: every card counts as a face card, including Stone (C4, C9). Affects Midas Mask, Sock and Buskin, Photograph, Scary Face, Business Card, Reserved Parking etc.

## Standard deck and starting values
- Standard deck: 52 cards, 4 suits x 13 ranks, no enhancements/editions/seals (C8: Red Deck page, "the 52 cards of a normal deck").
- Red Deck: +1 discard every round (C8). So default 3 discards becomes 4.
- Default starting values (secondary sources C10, C11; the wiki Decks pages only show deltas, so mark `UNVERIFIED` as to "official" confirmation, but consistent across guides):
  - Hands per round 4 (C10 confirms 4).
  - Discards per round 3 (C11).
  - Hand size 8 (C11).
  - Starting money $4 (C11).
  - Joker slots 5 (C11).
  - Consumable slots 2 (C11).
- Deck deltas for reference (C8): Blue +1 hand, Black -1 hand/+1 joker slot, Painted +2 hand size/-1 joker slot, Yellow +$10, Nebula -1 consumable slot... (Nebula, Magic, etc. start with vouchers.)
