# Shop

Sources: SH = https://balatrowiki.org/w/Shop ; BP = https://balatrowiki.org/w/Booster_Packs ; ED = https://balatrowiki.org/w/Edition ; VO = https://balatrowiki.org/w/Vouchers ; TG = https://balatrowiki.org/w/Tags.
Consumable costs, Soul/Black Hole odds and the Joker/Tarot/Planet/Spectral base weights are also in `consumables.md` (not repeated in detail). Packs: `packs.md`. Vouchers: `vouchers.md`. Tags: `tags.md`.
"[mem]" = recalled from the original game's behaviour, not confirmed on the wiki pages read; always paired with UNVERIFIED.

## 1. Layout
- Shop opens after beating a Small, Big or Boss blind [SH].
- Default: 2 random card slots, 2 booster-pack slots, 1 voucher slot [SH]. Overstock: 3 card slots, Overstock Plus: 4 [SH, VO]. Buying Overstock Plus also fills the new empty slot immediately [VO].
- Reroll replaces only the random card slots. Booster packs are fixed when the shop is entered and are not refreshed by rerolls [SH, BP]. Voucher: restocks only after the Boss Blind is defeated (i.e. one voucher per ante), or when the Voucher Tag is claimed [SH, VO]. An unbought voucher stays in the following shops of the same ante [mem] UNVERIFIED (wiki text says only "restock after boss"; recommended: keep it, it is the original behaviour).
- First shop of a run: the first booster slot is always a Normal Buffoon Pack (not in Jokerless challenge) [SH, BP]. The "+$3 booster price" is tutorial-only, ignore [SH].

## 2. Card-slot type weights
- Joker 20, Tarot 4, Planet 4 (total 28: 71.4% / 14.3% / 14.3%) [SH].
- Playing card weight 4 only after buying Magic Trick; Spectral weight 2 only with Ghost Deck [SH, consumables.md].
- Tarot Merchant sets Tarot weight to 9.6, Tarot Tycoon to 32 [SH]. Planet Merchant / Tycoon: "2x / 4x as often" in wiki text [SH]. Weights are therefore 9.6 and 32 for Tarot (2.4x / 8x of base, not 2x/4x). Planet likely identical (9.6 / 32) [mem] UNVERIFIED. Recommend 9.6/32 for both.
- Magic Trick / Illusion do not change the weight (still 4); Illusion changes the playing card's modifier odds (see 5).
- Per slot: roll type by weight, then roll the specific card (rarity for Jokers). Soul and Black Hole never appear in shop [consumables.md].

## 3. Jokers
- Rarity: Common 70%, Uncommon 25%, Rare 5% [SH]. Legendary never appears in the shop (only via The Soul) [SH].
- Duplicates: without Showman, a Joker already owned (or already in the shop) is not offered; Showman lifts this (also for Tarot/Planet/Spectral, see consumables.md) [consumables.md]. Exact "already owned vs in shop" scope UNVERIFIED; recommend: exclude owned + currently shown.
- Rental/Eternal/Perishable stickers: shop Jokers on higher stakes may carry them; White Stake has none. Rental: costs $1 to buy, -$3 at the end of each round [SH]. Eternal/Perishable chances (30% each at their stakes [mem]) UNVERIFIED; irrelevant for White Stake.

## 4. Edition odds (shop Jokers)
Per Joker, from ED (table garbled in fetch, reconstructed and checked against the original thresholds 0.3%/0.6%x/2%x/4%x [mem]):
| Voucher | Negative | Polychrome | Holographic | Foil | Total |
|---|---|---|---|---|---|
| none | 0.3% | 0.3% | 1.4% | 2% | 4.0% |
| Hone (x2) | 0.3% | 0.9% | 2.8% | 4% | 8.0% |
| Glow Up (x4) | 0.3% | 2.1% | 5.6% | 8% | 16.1% |
- Holo/Foil numbers are explicit on ED; Poly 0.3/0.9/2.1 and Negative constant 0.3% are UNVERIFIED (the wiki's table marks 0.9% and 2.1% in the Negative/Poly column pair; derivation: thresholds are 0.3% Negative, then Poly up to 0.6%*m, Holo up to 2%*m, Foil up to 4%*m minus the previous bands). Recommend the table above.
- Prices: edition surcharge Foil +$2, Holographic +$3, Polychrome +$5, Negative +$5 (applies to Jokers; playing cards: Foil/Holo/Poly use same surcharge) [SH]. Consumables can only be Negative via Perkeo, never in shop [SH].

## 5. Playing cards in shop (Magic Trick)
- Appear only after Magic Trick; base price $1 [SH]. Card is a random rank/suit; plain card unless Illusion [SH].
- With Illusion: Enhancement 40%, Edition 20% [SH]; Seal 20% [mem] UNVERIFIED. Wiki notes Illusion seals are bugged in v1.0.1o (shop cards cannot get seals) [VO]. Recommend: implement the intended 20% seal chance behind a flag defaulting to the intended behaviour; the shop expert is rarely affected.
- Edition rates for these cards: Foil 4%, Holo 2.8%, Poly 1.2% (x Hone/Glow Up), no Negative [ED playing-card table, listed for pack cards]. How this combines with Illusion's "Edition 20%" is UNVERIFIED; recommend: 20% chance to be edition-eligible is NOT modelled separately; use the ED bands for the edition roll and 40% enhancement / 20% seal, until a better source appears.

## 6. Price formula
- Base costs: Joker from `jokers.csv` (Common $1-6, Uncommon $4-8, Rare $7-10, Legendary $20 ranges [SH]); Planet $3, Tarot $3, Spectral $4, playing card $1 [SH]; packs Normal $4 / Jumbo $6 / Mega $8 [SH, BP]; voucher $10 [SH, VO].
- cost = max(1, round_half_down((base + edition surcharge) x discount)) [SH]. [mem] original formula floor((base+0.5) x (100-disc)/100), equivalent to half-down. Example: 6 x 0.75 = 4.5 -> 4.
- Clearance Sale 25% off, Liquidation 50% off, applying to all cards AND packs (also vouchers [mem] UNVERIFIED; wiki says "cards and packs") [SH, VO]. Liquidation also lowers sell values of current Jokers [VO].
- Inflation (challenge only): +$1 per purchase before discounts [SH]. Astronomer: Planets and Celestial Packs cost $0 [SH]. Coupon Tag: the cards and packs generated in the next shop are free (not the voucher) [SH, TG].
- Rental stakes: Rental joker buy cost $1 [SH].
- Buy-and-use: consumables bought from shop go to consumable slots; "Buy & Use" is available for non-targeting ones [mem] UNVERIFIED (consumables.md lists usage rules). Planets can be bought and used immediately outside a round.

## 7. Reroll
- Base $5, +$1 per reroll, reset on entering each new shop [SH].
- Reroll Surplus: start at $3; Reroll Glut: start at $1 [SH, VO]. Cost is floored at $0.
- Chaos the Clown: 1 free reroll per shop [SH]. D6 Tag: next shop's rerolls start at $0 [SH, TG]. Whether a free reroll increments the cost counter: UNVERIFIED; recommend: free rerolls do not advance the counter [mem].

## 8. Selling
- Sell value = max(1, floor(cost / 2)) + Egg/Gift Card bonuses [SH]. "Cost" here is the current price including edition surcharge and discount (e.g. Foil Joker base 4 -> cost 6 -> sell 3) [SH]. Consumable: Planet/Tarot sell $1, Spectral $2, see `consumables.md`.
- Items bought for $0 (Coupon/Astronomer/Couponed tag items) still sell for at least $1 [mem] UNVERIFIED.
