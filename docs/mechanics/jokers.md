# Jokers inventory (companion to jokers.csv)

Source: https://balatrowiki.org/w/Jokers (150 jokers in collection order) plus per-joker wiki pages; effects paraphrased. Exact numbers for some entries were cross-checked from general game knowledge, not a second mirror (see UNVERIFIED).

## Counts
- Total 150. Rarity: Common 61, Uncommon 64, Rare 20, Legendary 5.
- Waves: 3.1 = 54, 3.2 = 33, 3.3 = 20, 3.4 = 33, 3.5 = 10 (correctness of category preferred over the 45/35/20/40/10 target).
- Not Blueprint/Brainstorm copyable (`copyable=no`): 29 (see csv); values follow each joker's wiki "Compatibility: Copyable" field.

## UNVERIFIED
- Joker Stencil: wiki says "X1 Mult for each empty Joker slot, Joker Stencil included"; behaviour with multiple Stencils / Negative slots still taken from game knowledge (X = free slots + number of Stencils).
- Blueprint/Brainstorm: wiki confirms Brainstorm copies what a leftmost Blueprint copies; Blueprint->Brainstorm chains not explicitly stated.
- Luchador and Diet Cola are marked copyable by the wiki but are sell-triggered, so Blueprint cannot meaningfully copy them; copyable=yes only records the wiki flag.
- Matador: exact boss-trigger list taken from the wiki page; wiki itself notes it is inconsistent ("oversight"), so implement per the listed blinds only.
- Mr. Bones: exact check timing (final hand) from wiki prose; no numbers beyond 25%.

## Interaction notes
- Legendary jokers appear only via The Soul (Spectral) / not in shops.
- Showman allows duplicate Jokers/Tarots/Planets/Spectrals in shop and packs; Jokers normally unique.
- Baseball Card is the only joker_on_joker effect (applies per Uncommon Joker owned; Baseball Card itself is Rare so does not count).
- Retriggers: Hack, Dusk, Seltzer, Hanging Chad, Sock and Buskin (cards); Mime (held). They stack with Red Seal.
- Rule flags: Four Fingers, Shortcut, Smeared, Pareidolia, Splash. Oops! All 6s doubles every listed probability (incl. Lucky card, Wheel of Fortune, Space Joker, 8 Ball, Gros Michel/Cavendish).
- Stuntman and other hand size / slot jokers are passive; hand size changes must update on add/remove.
- Self-destructing jokers: Gros Michel, Cavendish, Ice Cream, Popcorn, Ramen, Seltzer, Turtle Bean, Mr. Bones, Madness (others), Ceremonial Dagger (right neighbor).
- Eternal/Perishable/Rental stickers are out of scope here.
- Most "create a card" jokers require free consumable/joker slots.

## Verification log
Task 3.0v: every row 1-150 was compared against its own balatrowiki.org page (fetched in full; all 150 reachable). Rarity and buy price matched the CSV for all 150 rows (Legendary buy price 20, sell 10, confirmed on Canio/Triboulet/Yorick/Chicot/Perkeo pages). All base numbers in effect text (Mult/Chips/X values, odds, increments) matched. Changes made:
- copyable (set to the wiki Copyable flag): Ceremonial Dagger, Madness, Luchador, Diet Cola, Matador, Stuntman no->yes; Midas Mask, Trading Card yes->no.
- effect clarified from wiki prose: Ramen (destroyed at X1 or below, Hook discards count), Mr. Bones (no blind reward money, interest still paid, not saved by empty deck), Matador (boss-trigger list, no discards), Madness, Ceremonial Dagger, Blackboard (Stone blocks), Square Joker (4 played cards), Flower Pot (needs 4 scoring cards), Castle/Mail-In Rebate/The Idol target selection weighting, Erosion (Abandoned Deck 40), Rocket (increase before payout), Sixth Sense, Obelisk (reset only if strictly most played), Midas Mask (overwrites enhancement), Vampire, Red Card, Gros Michel/Cavendish spawn conditions, Chicot, Perkeo, Hiker.
- timing: Matador gained `boss_triggered`.
- Ramen X0.01 decay, Raised Fist rightmost-lowest tie rule, Ancient Joker new suit chosen from the other 3 suits: confirmed. Legendary cost 20: confirmed. Ceremonial Dagger/Madness/Certificate/Marble copyable: resolved from wiki.
