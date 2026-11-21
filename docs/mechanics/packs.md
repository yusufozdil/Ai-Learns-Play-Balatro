# Booster packs

Sources: BP = https://balatrowiki.org/w/Booster_Packs ; SH = https://balatrowiki.org/w/Shop ; ED = https://balatrowiki.org/w/Edition ; VO = https://balatrowiki.org/w/Vouchers . [mem] = recalled original behaviour, UNVERIFIED.

## 1. Table
Price: Normal $4, Jumbo $6, Mega $8 [BP]. Choices: Normal and Jumbo pick 1, Mega picks 2 [BP].
| Category | Normal cards | Jumbo cards | Mega cards | Contents |
|---|---|---|---|---|
| Arcana | 3 | 5 | 5 | Tarots, used immediately, hand of cards dealt [BP] |
| Celestial | 3 | 5 | 5 | Planets [BP] |
| Standard | 3 | 5 | 5 | Playing cards added to deck [BP] |
| Buffoon | 2 | 4 | 4 | Jokers [BP] |
| Spectral | 2 | 4 | 4 | Spectrals, used immediately [BP] |

## 2. Shop appearance weights [BP]
| Category | Normal | Jumbo | Mega | Total |
|---|---|---|---|---|
| Standard | 4 | 2 | 0.5 | 6.5 |
| Arcana | 4 | 2 | 0.5 | 6.5 |
| Celestial | 4 | 2 | 0.5 | 6.5 |
| Buffoon | 1.2 | 0.6 | 0.15 | 1.95 |
| Spectral | 0.6 | 0.3 | 0.07 | 0.97 |
Total weight 22.42 (Normal Arcana = 4/22.42 = 17.84%, Mega Spectral 0.31%) [BP]. Two packs are drawn per shop, possibly the same type; whether the second draw excludes the first is UNVERIFIED (wiki: "often but not always different"); recommend independent draws with replacement. First shop of the run: slot 1 is a Normal Buffoon pack [BP]. Packs are not refreshed by reroll [BP].

## 3. Opening and picking
- Buying a pack opens it immediately [BP]. You may use consumables from your slots while it is open [BP].
- Tarot/Spectral packs deal a hand of playing cards equal to current hand size to target effects; an emptied hand can softlock (Skip greyed out), Temperance/Hermit still work [BP]. Consumables from packs are used immediately and need no free slot (project decision); Planets may be placed into a slot (project decision). Buffoon picks require a free Joker slot [mem] UNVERIFIED (Negative edition grants the room).
- Pack closes after the last allowed pick (Mega: 2 picks) or on Skip, which may be done at any time [BP]. Playing cards from Standard are added to the deck permanently [BP].
- Skipping any pack (also a Mega after 1 pick) adds +3 Mult to Red Card [BP]. Note only.

## 4. Contents generation
- Pack contents are generated when the pack is opened, deterministic per seed [BP].
- Duplicates may occur within one pack [BP]; consumables.md recommends no duplicates unless Showman (CONFLICT, see there). Recommend: follow consumables.md/notes 4.1b: no repeats without Showman.
- Arcana: Tarots; The Soul 0.3% per slot [consumables.md]; Omen Globe lets slots turn into Spectrals (20% per slot [mem] UNVERIFIED) [VO].
- Celestial: Planets (cost-free to choose); Black Hole 0.3% [consumables.md]; secret hand planets only if the hand was already played (project decision). Telescope: pack always contains the Planet of the most played hand [VO]. Tie-break/zero plays UNVERIFIED; recommend: tie -> hand listed first in the natural hand order, no plays -> no forced card.
- Spectral: Spectrals; Soul/Black Hole 0.3% each [consumables.md].
- Standard: random rank/suit playing cards. Each card: Enhancement 40% (uniform among 8 non-Stone? incl. Stone [mem]), Seal 20% (uniform among 4), Edition Foil 4% / Holo 2.8% / Poly 1.2% (no Negative), doubled/quadrupled by Hone/Glow Up [ED, BP for edition; enhancement/seal probabilities UNVERIFIED]. The wiki says only "often" for enhancement/seal [BP].
- Buffoon: Jokers with the shop rarity weights Common 70 / Uncommon 25 / Rare 5, no Legendary [SH]; edition odds as shop Jokers (Negative 0.3%, Poly 0.3%, Holo 1.4%, Foil 2%, see shop.md) [mem] UNVERIFIED. Duplicates only with Showman (recommended).
- Challenges: Fragile bans Standard, Jokerless bans Buffoon [BP].
