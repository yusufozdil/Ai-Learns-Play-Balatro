# Jokers inventory (companion to jokers.csv)

Source: https://balatrowiki.org/w/Jokers (150 jokers in collection order) plus per-joker wiki pages; effects paraphrased. Exact numbers for some entries were cross-checked from general game knowledge, not a second mirror (see UNVERIFIED).

## Counts
- Total 150. Rarity: Common 61, Uncommon 64, Rare 20, Legendary 5.
- Waves: 3.1 = 54, 3.2 = 33, 3.3 = 20, 3.4 = 33, 3.5 = 10 (correctness of category preferred over the 45/35/20/40/10 target).
- Not Blueprint/Brainstorm copyable (`copyable=no`): 33 (see csv).

## UNVERIFIED
- Matador: wiki says "earn $8 if played hand triggers the Boss Blind ability". Interpretation A: only when a played hand triggers a boss effect (e.g. The Tooth/Needle/Psychic-style); B: also discards (e.g. The Hook). Copyable set to `no` as a guess.
- Joker Stencil: whether its own slot counts as empty. Chosen: yes (X = free slots + number of Stencils, Negative slots add). Alternative: free slots only.
- Mr. Bones: restore semantics (round is won, blind cleared, Joker destroyed) taken from general knowledge.
- Ceremonial Dagger / Madness copyable=no and Certificate / Marble copyable=yes: from knowledge of Blueprint behaviour, wiki summary did not list them on the fetched table.
- Legendary `cost`: wiki shows N/A (not sold in shops); csv uses 20 (assumed internal base for sell value). Verify.
- Blueprint/Brainstorm: chained copying (Blueprint->Brainstorm, Brainstorm copying Blueprint) assumed to work.
- Raised Fist tie rule (rightmost lowest) and Ancient Joker / Mail-In target selection (new target differs from previous) assumed.
- Ramen decay: X0.01 per discarded card (wiki table); verify the figure.

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
