# Consumables: Planets, Tarots, Spectrals

Sources (own-words summaries; per-page URLs below):
- P  = https://balatrowiki.org/w/Planet_Cards
- T  = https://balatrowiki.org/w/Tarot_Cards
- S  = https://balatrowiki.org/w/Spectral_Cards
- C  = https://balatrowiki.org/w/Consumables
- B  = https://balatrowiki.org/w/Booster_Packs
- SH = https://balatrowiki.org/w/Shop
- Per-card pages: https://balatrowiki.org/w/<Card_Name> (Strength, Death, The_Fool, The_Soul, Ectoplasm, Familiar, Cryptid, Sigil, Wheel_of_Fortune, Immolate, Ankh, Ghost_Deck)

Items marked `UNVERIFIED` were not stated by the wiki pages read; the recommendation given matches
the original game's behaviour as best known and should be confirmed.

## 1. Planets (12)  [P, SH]

Base cost $3 for every Planet [SH]. Sell value = floor(cost/2), min $1 (= $1) [SH].
Using a Planet raises its hand by 1 level: add the listed chips and mult once. Increments cross-checked
against hands.md section 1: all 12 match exactly.

| Planet | Hand leveled | +Chips/level | +Mult/level | Cost | Secret? |
|---|---|---|---|---|---|
| Pluto | High Card | 10 | 1 | $3 | no |
| Mercury | Pair | 15 | 1 | $3 | no |
| Uranus | Two Pair | 20 | 1 | $3 | no |
| Venus | Three of a Kind | 20 | 2 | $3 | no |
| Saturn | Straight | 30 | 3 | $3 | no |
| Jupiter | Flush | 15 | 2 | $3 | no |
| Earth | Full House | 25 | 2 | $3 | no |
| Mars | Four of a Kind | 30 | 3 | $3 | no |
| Neptune | Straight Flush | 40 | 4 | $3 | no |
| Planet X | Five of a Kind | 35 | 3 | $3 | yes |
| Ceres | Flush House | 40 | 4 | $3 | yes |
| Eris | Flush Five | 50 | 3 | $3 | yes |

- No Royal Flush Planet; it levels with Straight Flush [P].
- Secret Planets (Planet X, Ceres, Eris) can appear in the shop only after that hand type has been played in the current run [P].
  (Whether packs also obey this is not stated by the wiki: UNVERIFIED; recommend same restriction for packs, which is the original behaviour.)
- Black Hole (Spectral, not a Planet): levels up all 12 hands by 1 [P, S]. See section 3.
- Planets need no target cards; usable any time (section 4).

## 2. Tarots (22)  [T]

Base cost $3 for all Tarots (changed from $4 in patch 0.9.3g) [T update history, SH]. Sell value $1.
"Hand targets" = the highlighted cards must be playing cards in the current hand (only possible when a hand is drawn: in a round, or in an Arcana/Spectral pack).
Enhancing/suit-changing keeps the card's rank and other properties (see cards.md); overwriting an existing enhancement replaces it.

| # | Card | Exact effect | Targets (hand) | Randomness | Edge cases |
|---|---|---|---|---|---|
| 0 | The Fool | Creates a copy of the last Tarot or Planet card used this run | none | none | Never copies itself; Spectrals (incl. Black Hole) never count as "last used". Needs a free consumable slot. If last used was The Fool, must use another Tarot/Planet first. Copy does not keep edition. If none used yet: UNVERIFIED (wiki silent; recommend the Fool cannot be used / does nothing). Whether the created copy becomes "last used": UNVERIFIED (original: it is not changed, since Fool is excluded; the last-used record only stores non-Fool cards). [The_Fool] |
| 1 | The Magician | Enhance targets to Lucky | 1-2 | none | |
| 2 | The High Priestess | Create up to 2 random Planet cards | none | random planets | Needs an empty consumable slot. Creates as many as fit, up to 2 (original: fills free slots, so 1 free slot -> 1 Planet). The exact "partial" rule is UNVERIFIED (wiki says "requires empty slot"). Created Planets may duplicate (Showman rule applies to pack/shop only: UNVERIFIED). |
| 3 | The Empress | Enhance targets to Mult | 1-2 | none | |
| 4 | The Emperor | Create up to 2 random Tarot cards | none | random tarots | Same slot rule as High Priestess. Can yield The Fool: UNVERIFIED. |
| 5 | The Hierophant | Enhance targets to Bonus | 1-2 | none | |
| 6 | The Lovers | Enhance target to Wild | exactly 1 | none | |
| 7 | The Chariot | Enhance target to Steel | exactly 1 | none | |
| 8 | Justice | Enhance target to Glass | exactly 1 | none | |
| 9 | The Hermit | Gain money equal to current money, max +$20 (i.e. doubles, capped at +$20) | none | none | Negative money (debt) gives 0: UNVERIFIED but follows "doubling" wording; recommend max(0, min(money, 20)). |
| 10 | The Wheel of Fortune | 1 in 4 chance: give a random Joker with no edition Foil (50%), Holographic (35%) or Polychrome (15%) | none (1 random Joker) | 25% trigger; edition weights 50/35/15 | Only Jokers with no edition (incl. not Negative) are eligible; if none, card cannot be used. Failure shows "Nope!" and the card is still consumed. Oops! All 6s doubles the 1-in-4 odds. Never produces Negative. [Wheel_of_Fortune] |
| 11 | Strength | Raise rank of targets by 1 | 1-2 | none | Wraps: K -> A and A -> 2 (so A is "after K" and 2 follows A), Jack follows 10 etc. Max 2 targets since 0.8.8. Stone card: UNVERIFIED (no rank; recommend not usable/ignored on Stone). [Strength] |
| 12 | The Hanged Man | Destroy targets | 1-2 | none | Destroyed cards leave the deck permanently. |
| 13 | Death | Left selected card becomes a copy of the right selected card | exactly 2 | none | "Left/right" = visual order of the two cards in hand (the hand's left-to-right order, player can rearrange). Left card is changed in place (rank, suit, enhancement, edition, seal, bonus chips copied); right card unchanged. Debuff status of the converted card follows the new card's state per the wiki (debuffed target stays debuffed). Copy fields list is partially UNVERIFIED (wiki says enhancements/editions/seals/Hiker chips carry over). [Death] |
| 14 | Temperance | Gain money equal to total sell value of all owned Jokers, max $50 | none | none | Counts only Jokers (not consumables). Joker sell values include Egg/Gift Card bonuses. If no Jokers, $0. |
| 15 | The Devil | Enhance target to Gold | exactly 1 | none | |
| 16 | The Tower | Enhance target to Stone | exactly 1 | none | |
| 17 | The Star | Change suit of targets to Diamonds | 1-3 | none | |
| 18 | The Moon | Change suit of targets to Clubs | 1-3 | none | |
| 19 | The Sun | Change suit of targets to Hearts | 1-3 | none | |
| 20 | Judgement | Create 1 random Joker | none | random rarity/joker | Needs an empty Joker slot (otherwise unusable). Rarity roll follows normal Joker rarity weights (common 70 / uncommon 25 / rare 5 [SH]); cannot create Legendary. Not told whether edition can be rolled: UNVERIFIED (original: yes, normal shop edition rolls). |
| 21 | The World | Change suit of targets to Spades | 1-3 | none | |

Number min for "up to N" is 1 (a Tarot needing targets cannot be used with 0 selected). Enhancement-class Tarots: "up to 2 / exactly 1" per [T].
Tarot ids in some sources are numbered Fool = 0, Magician = 1, ... World = 21 (order of the table above).
Source for all effects/targets: [T].

## 3. Spectrals (18)  [S]

Base cost $4 (Ankh, Immolate, Ectoplasm confirmed on their pages; others per wiki/shop page) [SH]. Sell value $2.
"Hand targets" as in section 2.

| Card | Exact effect | Targets | Randomness | Costs / side effects / edge cases |
|---|---|---|---|---|
| Familiar | Destroy 1 random card in hand, then add 3 random Enhanced face cards to hand | none | destroyed card random; added cards random rank (K/Q/J), random enhancement, random suit | Cannot be used with only 1 card in hand. No Stone cards added. Enhancement pool (excluding Stone) and random suit: UNVERIFIED (wiki silent; original: any of the 7 non-Stone enhancements, any suit). Added cards also go into the deck. [Familiar] |
| Grim | Destroy 1 random card in hand, add 2 random Enhanced Aces | none | as above (rank fixed to Ace) | same notes as Familiar |
| Incantation | Destroy 1 random card in hand, add 4 random Enhanced numbered cards | none | rank 2-10 (range implied, UNVERIFIED), enhancement/suit random | same notes. Also the fallback Spectral when all 18 are blocked. |
| Talisman | Add Gold Seal | exactly 1 | none | Replaces an existing seal. |
| Aura | Add Foil, Holographic or Polychrome to the card | exactly 1 | edition random among the 3 (never Negative); weights equal: UNVERIFIED | |
| Wraith | Create 1 random Rare Joker; set money to $0 | none | random Rare joker | Needs a free Joker slot (UNVERIFIED on wiki; recommend required). Money set to exactly $0 (even if negative: UNVERIFIED). |
| Sigil | Change all cards in hand to one random suit | none (all hand) | one random suit applied to all | Cannot be used with 1 card in hand. Whether the random suit may equal the existing one: UNVERIFIED (recommend uniform over 4 suits, equal allowed). |
| Ouija | Change all cards in hand to one random rank; -1 hand size permanently | none (all hand) | one random rank applied to all | Hand size -1 each use (flat, not cumulative scaling). Rank uniform over 2..A: UNVERIFIED re face/Stone handling (recommend Stone cards unaffected/ignored). |
| Ectoplasm | Give Negative to a random Joker; -1 hand size | none (1 random Joker) | random joker without an edition | Hand size penalty is CUMULATIVE: 1st use -1, 2nd -2, 3rd -3 ... (wiki, patch 0.9.3L). Only Jokers with no edition are eligible; unusable if none. Negative gives +1 Joker slot. [Ectoplasm] |
| Immolate | Destroy 5 random cards in hand; gain $20 | none | random cards | Cannot be used with 1 card in hand. Behaviour with 2-4 cards in hand: UNVERIFIED (recommend destroy min(5, hand size) cards, still pay $20). [Immolate] |
| Ankh | Create a copy of a random Joker; destroy all other Jokers | none | random joker | Copy never keeps Negative edition; keeps other editions, stickers, scaling state. Eternal Jokers survive destruction but can be chosen. Unusable when all Joker slots are full ("No Space"), (the copy needs a slot; recommend: slot check before destruction). [Ankh] |
| Deja Vu | Add Red Seal | exactly 1 | none | |
| Hex | Add Polychrome to a random Joker; destroy all other Jokers | none | random joker | Eternal Jokers survive. Whether only edition-less Jokers are eligible: UNVERIFIED (original: yes, like Ectoplasm/Wheel; recommend yes). Ghost Deck starts with one Hex. |
| Trance | Add Blue Seal | exactly 1 | none | |
| Medium | Add Purple Seal | exactly 1 | none | |
| Cryptid | Create 2 copies of the selected card | exactly 1 | none | Copies keep rank, suit, enhancement, edition, seal and bonus chips. Copies are added to the deck and hand; hand size limit is exceeded temporarily. [Cryptid] |
| The Soul | Create 1 Legendary Joker | none | random of the 5 Legendary | Needs a free Joker slot. If all 5 Legendary already owned (no Showman) a regular Joker is created instead. Never in normal pool. Cost $4: UNVERIFIED (original). [The_Soul] |
| Black Hole | Level up every poker hand by 1 | none | none | Never in normal pool. |

Wiki page does not list purchase costs for Spectrals; $4 is stated on Ankh, Ectoplasm and Immolate pages and the Shop page lists Spectral at $4 [SH]. The Soul / Black Hole have no shop price (only found in packs; rarely, $4 sell $2: UNVERIFIED).

## 4. Usage rules  [C, B]

- Consumable slots: 2 by default; Crystal Ball adds 1 [C]. Each consumable (any type) occupies one slot. When full, buying or generating a consumable is blocked [C].
- Negative edition on a consumable adds 1 extra consumable slot while held: UNVERIFIED on the consumable pages read (original behaviour; Negative consumables, e.g. from Perkeo, do not take a slot). Recommend implementing.
- Use vs sell: use consumes the card and applies the effect; sell removes it for floor(cost/2), min $1 [SH, C]. Selling is possible any time [C].
- When usable: Planets and non-targeting Tarots/Spectrals can be used outside a round (shop, blind select) if their own requirements are met; effects that need hand cards (enhancement, suit, seal, destroy, Death, Strength, Sigil, Ouija, Familiar etc.) need a drawn hand, i.e. during a round or in an Arcana/Spectral pack [C, B]. Packs deal a hand equal to the current hand size for this [B].
- Pack picks (Arcana/Celestial/Spectral) are used immediately on selection [B]. The wiki does not say whether a free consumable slot is needed to pick one; recommend: picked card is consumed at once so no slot is needed except for The Fool (which needs a free slot; UNVERIFIED).
- You can use held consumables while a pack is open [B].
- Skipping: pack costs and sizes (note only; later task) [B]:

| Pack | Cost | Cards shown | Picks |
|---|---|---|---|
| Arcana / Celestial | $4 | 3 | 1 |
| Jumbo Arcana/Celestial | $6 | 5 | 1 |
| Mega Arcana/Celestial | $8 | 5 | 2 |
| Spectral | $4 | 2 | 1 |
| Jumbo Spectral | $6 | 4 | 1 |
| Mega Spectral | $8 | 4 | 2 |

## 5. Generation facts  [SH, B, P, T, S]

- Shop card-slot weights: Joker 20, Tarot 4, Planet 4, Playing card 4 (only after Magic Trick), Spectral 2 (only with Ghost Deck; 2/(20+4+4+2) about 6.7%) [SH, Ghost_Deck]. Tarot Merchant x2, Tarot Tycoon x4 (weights 9.6 / 32 per wiki), Planet Merchant x2, Planet Tycoon x4.
- Spectrals appear in the shop only with the Ghost Deck [S].
- The Soul and Black Hole never appear in the shop [Ghost_Deck].
- The Soul: each card slot in an Arcana or Spectral pack has 0.3% (3/1000) chance to be The Soul, rolled independently [The_Soul]. Black Hole: same 0.3% in Celestial and Spectral packs [S, The_Soul]. Both can rarely appear in the same Spectral pack. Other 16 Spectrals equally likely [S].
- Spectral cards from Arcana packs need Omen Globe (to let Spectrals appear in Arcana packs) [S].
- Duplicates: Tarots, Planets and Spectrals do not repeat in the shop or packs unless Showman is owned [T, P, S]. The Booster Packs page says "duplicates can occur" inside a pack (CONFLICT): recommend no duplicates within a pack and none already seen... (UNVERIFIED which "already seen" set is used; the original blocks repeats only while the card is currently in shop/pack areas, not those already owned/used).
- Fallbacks when everything is blocked: Tarot -> Strength [T], Planet -> Pluto [P], Spectral -> Incantation [S]. (The Tarot fallback differs in memory-based sources, which say The Fool; use the wiki: Strength.)
- Séance (Straight Flush) and Sixth Sense create Spectrals that can never be The Soul/Black Hole [S].
- Telescope: Celestial packs always contain the Planet for the most played hand [P].
