"""Loss analysis of a recorded policy (output of ``scripts/audit_agent.py``).

    # record (CPU is fine, ~15 min for the model, ~5 for the teacher)
    uv run --no-sync python scripts/audit_agent.py --mode det \
        --checkpoint runs/e2e_long_fix/latest.pt --out runs/analysis/ref_det.pkl
    uv run --no-sync python scripts/audit_agent.py --mode teacher --out runs/analysis/teacher.pkl
    # tables (markdown on stdout, raw numbers in --json)
    uv run --no-sync python scripts/analyze_losses.py report runs/analysis/ref_det.pkl \
        --teacher runs/analysis/teacher.pkl --json runs/analysis/stats.json
    # step-by-step log of one game (last K rounds)
    uv run --no-sync python scripts/analyze_losses.py narrate runs/analysis/ref_det.pkl 123 \
        --rounds 3
    # list the losing games, sorted by ante
    uv run --no-sync python scripts/analyze_losses.py losses runs/analysis/ref_det.pkl --max-ante 4

A *round* is one blind that was played (``rnd`` in the event log); shop events of the shop that
follows a round carry that round's ``rnd``. A round is *beaten* if its final score reached the
target (the engine ends the round on the first hand that does). Prices are not recorded; a shop
joker costs about ``2 * sell_value`` (sell = floor(cost / 2)).
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HAND_TYPES = [
    "HighCard", "Pair", "TwoPair", "ThreeKind", "Straight", "Flush", "FullHouse", "FourKind",
    "StraightFlush", "FiveKind", "FlushHouse", "FlushFive",
]  # fmt: skip
BLINDS = ["Small", "Big", "Boss"]
XMULT = {
    "Cavendish", "CardSharp", "LoyaltyCard", "SteelJoker", "AncientJoker", "Hologram", "Baron",
    "Photograph", "Vampire", "Constellation", "Madness", "LuckyCat", "GlassJoker", "Campfire",
    "Obelisk", "Ramen", "Throwback", "Bloodstone", "HitTheRoad", "TheDuo", "TheTrio",
    "TheFamily", "TheOrder", "TheTribe", "Acrobat", "SeeingDouble", "FlowerPot", "Blackboard",
    "DriversLicense", "Canio", "Triboulet", "Yorick", "TheIdol", "JokerStencil",
}  # fmt: skip
SCALING = {
    "RideTheBus", "GreenJoker", "SquareJoker", "Runner", "RedCard", "SpareTrousers",
    "Constellation", "Hologram", "Vampire", "Madness", "LuckyCat", "GlassJoker", "Campfire",
    "Obelisk", "FlashCard", "WeeJoker", "Castle", "Hiker", "Supernova", "Canio", "Yorick",
    "Throwback", "Erosion", "FortuneTeller", "Egg", "GiftCard", "Rocket", "Bull",
}  # fmt: skip
COPY = {"Blueprint", "Brainstorm"}
SHOWDOWN = {"VioletVessel", "CrimsonHeart", "AmberAcorn", "VerdantLeaf", "CeruleanBell"}
SHOP_SPEND = ("Buy", "BuyAndUse", "BuyVoucher", "OpenPack", "Reroll")


@dataclass
class Round:
    idx: int
    ante: int
    blind: str
    boss: str
    target: float
    money_start: int = 0
    jokers_start: list = field(default_factory=list)
    plays: list = field(
        default_factory=list
    )  # (hand, score, hands_left_before, discards_left_before)
    discards: int = 0
    final: float = 0.0
    hl_end: int = 0
    fatal: bool = False  # the run ended in this round (a loss); not beaten + not fatal = saved

    @property
    def beaten(self) -> bool:
        return self.final >= self.target

    @property
    def ratio(self) -> float:
        return self.final / self.target


@dataclass
class Game:
    i: int
    win: bool
    ante: int
    fin: dict
    ev: list
    money_curve: list
    rounds: list[Round]
    skips: int

    @property
    def last(self) -> Round:
        return self.rounds[-1]


def load(path: Path) -> list[Game]:
    d = pickle.loads(Path(path).read_bytes())
    out = []
    for i, (g, res) in enumerate(zip(d["games"], d["res"], strict=True)):
        rounds: dict[int, Round] = {}
        skips = 0
        for e in g["ev"]:
            if e["k"] == "SkipBlind":
                skips += 1
            if e["k"] not in ("Play", "Discard"):
                continue
            r = rounds.get(e["rnd"])
            if r is None:
                r = rounds[e["rnd"]] = Round(
                    e["rnd"], e["ante"], e["blind"], e["boss"], e["target"], e["money"], e["jk"]
                )
            if e["k"] == "Discard":
                r.discards += 1
            else:
                r.plays.append((e["hand"], e["score"], e["hl"], e["dl"]))
                r.final = e["rs"] + e["score"]
                r.hl_end = e["hl"] - 1
        if g["fin"]["outcome"] != "Win":
            rounds[max(rounds)].fatal = True
        out.append(
            Game(i, g["fin"]["outcome"] == "Win", g["fin"]["ante"], g["fin"], g["ev"],
                 g["money_curve"], [rounds[k] for k in sorted(rounds)], skips)
        )  # fmt: skip
        assert res[0] == i
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(head: list[str], rows: list[list]) -> str:
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def quant(a, qs=(10, 25, 50, 75, 90)) -> list[float]:
    return [float(np.percentile(a, q)) for q in qs] if len(a) else []


# ----------------------------------------------------------------------------- sections


def sec_overview(games: list[Game], name: str) -> tuple[str, dict]:
    n = len(games)
    hist = Counter(g.ante for g in games)
    losses = [g for g in games if not g.win]
    by_blind = Counter((g.ante, g.last.blind) for g in losses)
    mean_ante = np.mean([g.ante for g in games])
    s = f"### {name}: {n} games, {len(losses)} losses, mean ante {mean_ante:.2f}\n\n"
    s += "Ante histogram: " + ", ".join(f"{a}:{hist[a]}" for a in sorted(hist)) + "\n\n"
    # deaths / rounds entered by (ante, blind)
    entered: Counter = Counter()
    for g in games:
        for r in g.rounds:
            entered[(r.ante, r.blind)] += 1
    rows = []
    for a in range(1, 9):
        row = [a]
        for b in BLINDS:
            d, e = by_blind[(a, b)], entered[(a, b)]
            row.append(f"{d}/{e} ({pct(d / e) if e else '-'})")
        rows.append(row)
    s += "Deaths / rounds entered by ante and blind (death rate per entered round):\n\n"
    s += table(["ante", *BLINDS], rows) + "\n"
    out = {"hist": dict(hist), "by_blind": {f"{a}-{b}": v for (a, b), v in by_blind.items()}}
    return s, out


def sec_bosses(games: list[Game], name: str) -> tuple[str, dict]:
    enc: Counter = Counter()
    dead: Counter = Counter()
    saved: Counter = Counter()  # not beaten but the run went on (Mr. Bones)
    ratios = defaultdict(list)
    hands = defaultdict(list)
    antes = defaultdict(list)
    for g in games:
        for r in g.rounds:
            if r.blind != "Boss":
                continue
            enc[r.boss] += 1
            antes[r.boss].append(r.ante)
            if r.fatal:
                dead[r.boss] += 1
            elif not r.beaten:
                saved[r.boss] += 1
            else:
                ratios[r.boss].append(r.ratio)
                hands[r.boss].append(len(r.plays))
    rows = []
    for b in sorted(enc, key=lambda b: -dead[b] / enc[b]):
        rows.append([
            b, enc[b], dead[b], saved[b], pct(dead[b] / enc[b]), f"{np.mean(antes[b]):.1f}",
            f"{np.median(ratios[b]):.2f}" if ratios[b] else "-",
            f"{np.mean(hands[b]):.2f}" if hands[b] else "-",
        ])  # fmt: skip
    s = f"### {name}: death rate per boss (deaths / encounters)\n\n"
    head = ["boss", "enc", "deaths", "Mr. Bones saves", "death rate", "mean ante",
            "median final/target (beaten)", "mean hands used (beaten)"]  # fmt: skip
    s += table(head, rows)
    return s + "\n", {"enc": dict(enc), "dead": dict(dead)}


def sec_death_state(games: list[Game], name: str) -> tuple[str, dict]:
    losses = [g for g in games if not g.win]
    ratio = np.array([g.last.ratio for g in losses])
    s = f"### {name}: state at death ({len(losses)} losses)\n\n"
    qs = quant(ratio)
    s += (
        "final score / target in the lost round, quantiles 10/25/50/75/90: "
        + ", ".join(f"{q:.2f}" for q in qs)
        + "\n\n"
    )
    rows = []
    for lo, hi, lab in [(0, 0.25, "<0.25"), (0.25, 0.5, "0.25-0.5"), (0.5, 0.75, "0.5-0.75"),
                        (0.75, 0.9, "0.75-0.9"), (0.9, 1.0, "0.9-1.0")]:  # fmt: skip
        rows.append([lab, int(((ratio >= lo) & (ratio < hi)).sum())])
    s += table(["ratio bucket", "losses"], rows) + "\n"
    groups = [("ante 1-4", [g for g in losses if g.ante <= 4]),
              ("ante 5-7", [g for g in losses if 5 <= g.ante <= 7]),
              ("ante 8", [g for g in losses if g.ante == 8])]  # fmt: skip
    rows = []
    for lab, gs in groups:
        if not gs:
            continue
        rows.append([
            lab, len(gs),
            f"{np.median([g.last.ratio for g in gs]):.2f}",
            f"{np.mean([g.fin['discards_left'] for g in gs]):.2f}",
            f"{np.mean([len(g.last.plays) for g in gs]):.2f}",
            f"{np.mean([g.last.discards for g in gs]):.2f}",
            f"{np.mean([g.fin['money'] for g in gs]):.1f}",
            f"{np.mean([len(g.fin['jokers']) for g in gs]):.2f}",
            f"{np.mean([g.fin['joker_slots'] - len(g.fin['jokers']) for g in gs]):.2f}",
            f"{np.mean([len(g.fin['consumables']) > 0 for g in gs]):.2f}",
            f"{np.mean([g.fin['deck_size'] for g in gs]):.1f}",
            f"{np.mean([any(j in XMULT for j, _, _ in g.fin['jokers']) for g in gs]):.2f}",
        ])  # fmt: skip
    s += table(["group", "n", "median ratio", "discards left", "hands played", "discards used",
                "money", "jokers", "free joker slots", "unused consumable", "deck size",
                "has xMult"], rows) + "\n"  # fmt: skip
    # how many losses had discards left
    s += (
        f"Losses with >=1 discard left at death: {sum(g.fin['discards_left'] > 0 for g in losses)}"
        f" / {len(losses)}; with >= $10 unspent: {sum(g.fin['money'] >= 10 for g in losses)}"
        f"; with a free joker slot: "
        f"{sum(len(g.fin['jokers']) < g.fin['joker_slots'] for g in losses)}\n\n"
    )
    return s, {"ratio_q": qs}


def sec_margin(games: list[Game], name: str) -> tuple[str, dict]:
    wins = [g for g in games if g.win]
    s = f"### {name}: margin in winning runs ({len(wins)} wins)\n\n"
    rows, out = [], {}
    for ante in (6, 7, 8):
        for b in BLINDS:
            rs = [
                r for g in wins for r in g.rounds
                if r.ante == ante and r.blind == b and r.beaten
            ]  # fmt: skip
            if not rs:
                continue
            ra = np.array([r.ratio for r in rs])
            hl = np.array([r.hl_end for r in rs])
            first = np.array([r.plays[0][1] / r.target for r in rs])
            rows.append(
                [
                    ante, b, len(rs), f"{np.median(ra):.2f}", f"{np.percentile(ra, 10):.2f}",
                    f"{np.percentile(ra, 90):.2f}", pct(np.mean(ra < 1.2)),
                    f"{np.mean([len(r.plays) for r in rs]):.2f}", f"{np.mean(hl):.2f}",
                    pct(np.mean(hl == 0)), f"{np.median(first):.2f}",
                ]
            )  # fmt: skip
            out[f"{ante}-{b}"] = float(np.median(ra))
    s += table(
        ["ante", "blind", "n", "median final/target", "p10", "p90", "share < 1.2", "hands used",
         "hands left", "won on last hand", "median first-hand/target"],
        rows,
    )  # fmt: skip
    s += "\n"
    # the winning round (final boss) per boss id
    fin = [g.last for g in wins]
    ra = np.array([r.ratio for r in fin])
    s += (
        "Winning (final) round over all wins, final/target quantiles 10/25/50/75/90: "
        + ", ".join(f"{q:.2f}" for q in quant(ra))
        + "\n\n"
    )
    rows = []
    for boss in sorted({r.boss for r in fin}):
        rs = [r for r in fin if r.boss == boss]
        rows.append(
            [
                boss, len(rs), f"{rs[0].target:.0f}", f"{np.median([r.ratio for r in rs]):.2f}",
                pct(np.mean([r.hl_end == 0 for r in rs])),
                f"{np.mean([len(r.plays) for r in rs]):.2f}",
                f"{np.median([max(p[1] for p in r.plays) / r.target for r in rs]):.2f}",
            ]
        )  # fmt: skip
    s += table(
        ["final boss", "n wins", "target", "median final/target", "won on last hand",
         "hands used", "median best hand/target"],
        rows,
    )  # fmt: skip
    out["final_q"] = quant(ra)
    return s + "\n", out


def sec_money(games: list[Game], name: str) -> tuple[str, dict]:
    """Money at the start of the first round of each ante, joker count, xMult presence."""
    rows = []
    out = {}
    for a in range(1, 9):
        row = [a]
        for lab, sel in (("win", True), ("loss", False)):
            first = [
                next(r for r in g.rounds if r.ante == a)
                for g in games
                if g.win == sel and any(r.ante == a for r in g.rounds)
            ]
            if not first:
                row += ["-"] * 4
                continue
            row += [
                len(first),
                f"{np.mean([r.money_start for r in first]):.1f}",
                f"{np.mean([len(r.jokers_start) for r in first]):.2f}",
                f"{np.mean([any(j in XMULT for j, _ in r.jokers_start) for r in first]):.2f}",
            ]
            out[f"{lab}{a}"] = [r.money_start for r in first]
        rows.append(row)
    s = f"### {name}: state at the start of each ante (first round), wins vs losses\n\n"
    s += "(losses are those that reach the ante; a run dying in ante a counts in row a)\n\n"
    s += table(["ante", "n win", "money", "jokers", "has xMult", "n loss", "money", "jokers",
                "has xMult"], rows) + "\n"  # fmt: skip
    return s, {}


def sec_shop(games: list[Game], name: str) -> tuple[str, dict]:
    stat = defaultdict(lambda: defaultdict(list))  # (outcome, ante) -> metric -> per game values
    for g in games:
        per = defaultdict(Counter)
        end_money = {}
        for e in g.ev:
            if e["ph"] != "Shop":
                continue
            a = e["ante"]
            if e["k"] in SHOP_SPEND:
                per[a][e["k"]] += 1
                per[a]["spent"] += e["money"] - e["money_after"]
            elif e["k"] == "SellJoker":
                per[a]["sells"] += 1
            elif e["k"] == "EndShop":
                per[a]["shops"] += 1
                end_money.setdefault(a, []).append(e["money"])
        for a, c in per.items():
            key = ("win" if g.win else "loss", a)
            stat[key]["reroll"].append(c["Reroll"])
            stat[key]["buy"].append(c["Buy"] + c["BuyAndUse"])
            stat[key]["pack"].append(c["OpenPack"])
            stat[key]["spent"].append(c["spent"])
            stat[key]["sell"].append(c["sells"])
            stat[key]["end_money"].append(float(np.mean(end_money.get(a, [0]))))
    rows = []
    for a in range(1, 9):
        row = [a]
        for o in ("win", "loss"):
            m = stat.get((o, a))
            if not m:
                row += ["-"] * 6
                continue
            row += [len(m["reroll"])] + [
                f"{np.mean(m[k]):.2f}" for k in ("reroll", "buy", "pack", "sell", "end_money")
            ]
        rows.append(row)
    s = f"### {name}: shop behaviour per ante (per game, summed over the ante's shops)\n\n"
    hd = ["n", "rerolls", "buys", "packs", "sells", "money at EndShop"]
    s += table(["ante", *("win " + h for h in hd), *("loss " + h for h in hd)], rows) + "\n"
    return s, {}


def sec_unbought(games: list[Game], name: str) -> tuple[str, dict]:
    """EndShop events that left an affordable xMult joker with a free slot."""
    cnt = defaultdict(lambda: [0, 0, 0])  # outcome -> shops, shops with such a miss, deaths-ish
    ex = []
    for g in games:
        for e in g.ev:
            if e["k"] != "EndShop" or "shop_left" not in e:
                continue
            o = "win" if g.win else "loss"
            cnt[o][0] += 1
            slots_free = len(e["jk"]) < 5
            miss = [
                (x[1], x[2]) for x in e["shop_left"]
                if x[0] == "Joker" and x[1] in XMULT and x[2] is not None
                and e["money"] >= 2 * x[2] + 1 and slots_free
            ]  # fmt: skip
            if miss:
                cnt[o][1] += 1
                if len(ex) < 12 and not g.win:
                    ex.append((g.i, e["ante"], e["money"], miss))
    s = f"### {name}: shops ended with an affordable xMult joker and a free slot\n\n"
    s += (
        table(
            ["outcome", "EndShops", "with such a miss", "share"],
            [[o, v[0], v[1], pct(v[1] / max(v[0], 1))] for o, v in cnt.items()],
        )
        + "\n"
    )
    return s, {}


def sec_jokers(games: list[Game], name: str, at_ante: int = 5) -> tuple[str, dict]:
    reach = [(g, next(r for r in g.rounds if r.ante == at_ante))
             for g in games if any(r.ante == at_ante for r in g.rounds)]  # fmt: skip
    base = np.mean([g.win for g, _ in reach])
    s = f"### {name}: jokers held at the start of ante {at_ante} vs final outcome\n\n"
    s += f"Games reaching ante {at_ante}: {len(reach)}, win rate {pct(base)}.\n\n"
    cnt = Counter()
    wincnt = Counter()
    for g, r in reach:
        for j in {j for j, _ in r.jokers_start}:
            cnt[j] += 1
            wincnt[j] += g.win
    rows = []
    for j, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
        if n < 40:
            continue
        rows.append([j, n, pct(wincnt[j] / n), f"{100 * (wincnt[j] / n - base):+.1f}",
                     "X" if j in XMULT else "", "S" if j in SCALING else ""])  # fmt: skip
    s += table(["joker", "n", "win rate", "vs base (pts)", "xMult", "scaling"], rows[:40]) + "\n"

    # aggregate buckets
    def bucket(name_, pred):
        w = [g.win for g, r in reach if pred(r)]
        l_ = [not g.win for g, r in reach if pred(r)]
        return [name_, len(w), pct(np.mean(w)) if w else "-", sum(l_)]

    xm = lambda r: sum(j in XMULT for j, _ in r.jokers_start)  # noqa: E731
    nj = lambda r: len(r.jokers_start)  # noqa: E731
    rows = [bucket(f"xMult count = {k}", lambda r, k=k: min(xm(r), 3) == k) for k in range(4)]
    rows += [bucket(f"jokers = {k}", lambda r, k=k: min(nj(r), 5) == k) for k in range(0, 6)]
    rows += [bucket("has scaling joker", lambda r: any(j in SCALING for j, _ in r.jokers_start)),
             bucket("no scaling joker", lambda r: not any(j in SCALING for j, _ in r.jokers_start)),
             bucket("has copy joker", lambda r: any(j in COPY for j, _ in r.jokers_start)),
             bucket("money >= 25", lambda r: r.money_start >= 25),
             bucket("money < 10", lambda r: r.money_start < 10)]  # fmt: skip
    s += (
        f"Win rate by state at ante {at_ante} start:\n\n"
        + table(["bucket", "n", "win rate", "losses"], rows)
        + "\n"
    )
    # final joker sets
    for lab, sel in (("wins", True), ("losses", False)):
        c = Counter()
        gs = [g for g in games if g.win == sel]
        for g in gs:
            for j in {j for j, _, _ in g.fin["jokers"]}:
                c[j] += 1
        s += (
            f"Most common jokers in final state, {lab} (n={len(gs)}): "
            + ", ".join(f"{j} {pct(v / len(gs))}" for j, v in c.most_common(15))
            + "\n\n"
        )
    return s, {}


def sec_hands(games: list[Game], name: str) -> tuple[str, dict]:
    rows = []
    for lab, sel in (("win", True), ("loss", False)):
        gs = [g for g in games if g.win == sel]
        top = Counter(HAND_TYPES[int(np.argmax(g.fin["hand_plays_run"]))] for g in gs)
        lv = [g.fin["hand_levels"][int(np.argmax(g.fin["hand_plays_run"]))] for g in gs]
        mx = [max(g.fin["hand_levels"]) for g in gs]
        plan = [g.fin["stats"]["run"]["planets_used"] for g in gs]
        tops = ", ".join(f"{k} {pct(v / len(gs))}" for k, v in top.most_common(4))
        rows.append([lab, len(gs), tops,
                     f"{np.mean(lv):.1f}", f"{np.mean(mx):.1f}", f"{np.mean(plan):.1f}",
                     f"{np.mean([g.fin['deck_size'] for g in gs]):.1f}"])  # fmt: skip
    s = f"### {name}: main hand type, levels, deck\n\n"
    s += (
        table(
            [
                "outcome",
                "n",
                "most played hand type",
                "level of it",
                "max level",
                "planets used",
                "deck size",
            ],
            rows,
        )
        + "\n"
    )
    return s, {}


def _cause(g: Game) -> str:
    r = g.last
    if r.blind != "Boss":
        return f"{r.blind} blind"
    if r.boss == "VioletVessel":
        return "Violet Vessel (x6 base target)"
    if r.boss in SHOWDOWN:
        return "other showdown boss"
    if r.boss == "TheEye":
        return "The Eye (no repeat hand type)"
    if r.boss == "TheMouth":
        return "The Mouth (one hand type)"
    if r.boss in ("TheNeedle", "TheWall"):
        return f"{r.boss} (one hand / double target)"
    return "other boss, ante <= 4" if g.ante <= 4 else "other boss, ante >= 5"


def sec_causes(games: list[Game], name: str) -> tuple[str, dict]:
    losses = [g for g in games if not g.win]
    c = Counter(_cause(g) for g in losses)
    near = Counter(_cause(g) for g in losses if g.last.ratio >= 0.9)
    zero = Counter(_cause(g) for g in losses if any(p[1] == 0 for p in g.last.plays))
    hier = Counter(_cause(g) for g in losses if "Hieroglyph" in g.fin["vouchers"])
    rows = [
        [k, v, pct(v / len(losses)), near[k], zero[k], hier[k]]
        for k, v in sorted(c.items(), key=lambda kv: -kv[1])
    ]
    s = f"### {name}: cause of death (blind/boss of the lost round), {len(losses)} losses\n\n"
    s += table(
        ["cause", "losses", "share", "ratio >= 0.9", "has 0-score play", "owns Hieroglyph"],
        rows,
    )
    return s + "\n", dict(c)


def sec_blunders(games: list[Game], name: str) -> tuple[str, dict]:
    """Plays that scored 0: a repeated hand type under The Eye, a different type under The Mouth."""
    rows = []
    for boss in ("TheEye", "TheMouth", "ThePsychic"):
        rs = [r for g in games for r in g.rounds if r.blind == "Boss" and r.boss == boss]
        z = [r for r in rs if any(p[1] == 0 for p in r.plays)]
        nz = sum(p[1] == 0 for r in rs for p in r.plays)
        rows.append(
            [
                boss, len(rs), len(z), nz, sum(r.fatal for r in rs),
                sum(r.fatal for r in z),
                pct(np.mean([not r.fatal for r in z])) if z else "-",
                pct(np.mean([not r.fatal for r in rs if r not in z])),
            ]
        )  # fmt: skip
    s = f"### {name}: zero-score plays under boss restrictions\n\n"
    s += table(
        ["boss", "rounds", "rounds with 0-score play", "0-score plays", "deaths",
         "deaths with 0-score play", "survival rate with 0-score play", "survival rate without"],
        rows,
    )  # fmt: skip
    return s + "\n", {}


def sec_discards(games: list[Game], name: str) -> tuple[str, dict]:
    rows = []
    for lab, lo in (("ante 1-4", 1), ("ante 5-8", 5)):
        for res, sel in (("survived", False), ("lost", True)):
            rs = [
                r for g in games for r in g.rounds
                if r.fatal == sel and (r.ante >= 5) == (lo == 5)
            ]  # fmt: skip
            if rs:
                rows.append(
                    [
                        lab, res, len(rs), f"{np.mean([r.discards for r in rs]):.2f}",
                        pct(np.mean([r.discards == 0 for r in rs])),
                        f"{np.mean([len(r.plays) for r in rs]):.2f}",
                    ]
                )  # fmt: skip
    c = Counter(h for g in games for r in g.rounds if r.ante >= 5 for h, *_ in r.plays)
    tot = sum(c.values())
    s = f"### {name}: discard use and played hand types\n\n"
    s += table(
        ["group", "round", "n", "discards used", "share with 0 discards", "hands played"], rows
    )
    s += "\nHand types played at ante >= 5: " + ", ".join(
        f"{k} {pct(v / tot)}" for k, v in c.most_common(6)
    )
    return s + "\n\n", {}


def _vouchers_before(g: Game, ante: int) -> list[str] | None:
    v: list[str] = []
    for e in g.ev:
        if e["k"] == "Play" and e["ante"] == ante:
            return v
        if e["k"] == "BuyVoucher":
            v.append(e["item"][1]["id"])
    return None


def sec_vouchers(games: list[Game], name: str, at_ante: int = 5) -> tuple[str, dict]:
    reach = [(g, _vouchers_before(g, at_ante)) for g in games]
    reach = [(g, v) for g, v in reach if v is not None]
    base = np.mean([g.win for g, _ in reach])
    cnt: Counter = Counter()
    wins: Counter = Counter()
    for g, v in reach:
        for x in set(v):
            cnt[x] += 1
            wins[x] += g.win
    rows = []
    for x, n in cnt.most_common():
        if n < 20:
            continue
        wo = np.mean([g.win for g, v in reach if x not in v])
        rows.append([x, n, pct(wins[x] / n), pct(wo), f"{100 * (wins[x] / n - wo):+.1f}"])
    s = f"### {name}: vouchers bought before the first round of ante {at_ante}\n\n"
    s += f"{len(reach)} games, win rate {pct(base)} (observational, confounded).\n\n"
    s += table(["voucher", "n", "win rate with", "without", "diff (pts)"], rows) + "\n"
    hg = [g for g in games if "Hieroglyph" in g.fin["vouchers"]]
    s += (
        f"Games ending with Hieroglyph: {len(hg)} ({pct(len(hg) / len(games))}), win rate "
        f"{pct(np.mean([g.win for g in hg])) if hg else '-'}; rounds per game "
        f"{np.mean([len(g.rounds) for g in hg]) if hg else 0:.1f} vs "
        f"{np.mean([len(g.rounds) for g in games if g not in hg]):.1f} without.\n\n"
    )
    return s, {}


def sec_early(games: list[Game], name: str, max_ante: int = 4) -> tuple[str, dict]:
    es = [g for g in games if not g.win and g.ante <= max_ante]
    rows = []
    for g in es:
        r = g.last
        rows.append([
            g.i, g.ante, r.blind, r.boss if r.blind == "Boss" else "-", f"{r.target:.0f}",
            f"{r.final:.0f} ({r.ratio:.2f})", ",".join(h for h, *_ in r.plays),
            len(g.fin["jokers"]), ",".join(j for j, _, _ in g.fin["jokers"]),
            g.fin["money"], g.fin["discards_left"], g.skips,
        ])  # fmt: skip
    summ = (
        f"Summary: blinds {dict(Counter(g.last.blind for g in es))}; ratio >= 0.9: "
        f"{sum(g.last.ratio >= 0.9 for g in es)}, 0.75-0.9: "
        f"{sum(0.75 <= g.last.ratio < 0.9 for g in es)}, < 0.75: "
        f"{sum(g.last.ratio < 0.75 for g in es)}; <= 2 jokers: "
        f"{sum(len(g.fin['jokers']) <= 2 for g in es)}; >= $10 unspent: "
        f"{sum(g.fin['money'] >= 10 for g in es)}; >= 3 discards left: "
        f"{sum(g.fin['discards_left'] >= 3 for g in es)}\n\n"
    )
    s = f"### {name}: losses in ante <= {max_ante} ({len(es)} games)\n\n" + summ
    s += (
        table(
            [
                "game",
                "ante",
                "blind",
                "boss",
                "target",
                "final (ratio)",
                "hands played",
                "#jokers",
                "jokers",
                "money",
                "disc. left",
                "skips",
            ],
            rows,
        )
        + "\n"
    )
    return s, {}


# ----------------------------------------------------------------------------- narration


def narrate(g: Game, last_rounds: int = 3) -> str:
    first_rnd = g.rounds[max(0, len(g.rounds) - last_rounds)].idx
    lines = [
        f"game {g.i}: {'WIN' if g.win else 'LOSS'} at ante {g.ante}; final jokers "
        f"{[j for j, _, _ in g.fin['jokers']]}, money {g.fin['money']}, "
        f"vouchers {g.fin['vouchers']}"
    ]
    # money curve + joker set per ante start
    lines.append("ante starts: " + "; ".join(
        f"a{r.ante}/{r.blind} ${r.money_start} {[j for j, _ in r.jokers_start]}"
        for r in g.rounds if r.blind == "Small"))  # fmt: skip
    for e in g.ev:
        if e["k"] == "SelectBlind" and e["rnd"] + 1 < first_rnd:
            continue
        if e["rnd"] < first_rnd - 1 or (e["rnd"] == first_rnd - 1 and e["ph"] != "Shop"):
            continue
        k = e["k"]
        head = f"  a{e['ante']} {e['blind']:<5} r{e['rnd']} ${e['money']:<3}"
        if k == "Play":
            lines.append(f"{head} PLAY {e['hand']:<13} +{e['score']:.0f} "
                         f"(rs {e['rs']:.0f}/{e['target']:.0f}, hl {e['hl']} dl {e['dl']}) "
                         f"cards {e['cards']} from {e['hand_cards']}")  # fmt: skip
        elif k == "Discard":
            lines.append(f"{head} DISCARD {e['cards']} (hl {e['hl']} dl {e['dl']})")
        elif k == "EndShop":
            lines.append(f"{head} END_SHOP jokers {[j for j, _ in e['jk']]} left "
                         f"{e.get('shop_left')} packs {e.get('packs_left')}")  # fmt: skip
        elif "item" in e:
            lines.append(
                f"{head} {k} {e['item']} (-> ${e['money_after']})"
                + (f" ed={e['ed']}" if e.get("ed") not in (None, "Base") else "")
            )
        else:
            lines.append(f"{head} {k} {e['a'][1:]}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

SECTIONS = [
    sec_overview, sec_causes, sec_bosses, sec_blunders, sec_death_state, sec_margin,
    sec_money, sec_shop, sec_unbought, sec_discards, sec_jokers, sec_hands, sec_vouchers,
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("pkl", type=Path)
    r.add_argument("--teacher", type=Path, default=None)
    r.add_argument("--json", type=Path, default=None)
    r.add_argument("--early-ante", type=int, default=4)
    n = sub.add_parser("narrate")
    n.add_argument("pkl", type=Path)
    n.add_argument("game", type=int)
    n.add_argument("--rounds", type=int, default=3)
    lo = sub.add_parser("losses")
    lo.add_argument("pkl", type=Path)
    lo.add_argument("--max-ante", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "narrate":
        print(narrate(load(a.pkl)[a.game], a.rounds))
        return
    games = load(a.pkl)
    if a.cmd == "losses":
        print(sec_early(games, "losses", a.max_ante)[0])
        return
    stats = {}
    for fn in SECTIONS:
        text, st = fn(games, "reference")
        print(text)
        stats[fn.__name__] = st
    print(sec_early(games, "reference", a.early_ante)[0])
    if a.teacher:
        tg = load(a.teacher)
        for fn in (sec_overview, sec_causes, sec_bosses, sec_blunders, sec_discards, sec_margin):
            text, st = fn(tg, "teacher")
            print(text)
            stats["teacher_" + fn.__name__] = st
    if a.json:
        a.json.write_text(json.dumps(stats, default=float))


if __name__ == "__main__":
    main()
