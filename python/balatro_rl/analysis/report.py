"""Markdown report over a directory of per-game error-map JSONs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BARS = "▁▂▃▄▅▆▇█"


def load_games(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("game_*.json"))]


def eval_bar(curve: list[dict], width: int = 60) -> str:
    """ASCII P(win) curve over the decisions of a game (BlindSelect / Shop points)."""
    pts = [c["pwin"] for c in curve if c["phase"] in (0, 2)]
    if not pts:
        return ""
    if len(pts) > width:
        step = len(pts) / width
        pts = [pts[int(i * step)] for i in range(width)]
    return "".join(BARS[min(7, int(p * 8))] for p in pts)


def _cat(desc: str) -> str:
    t = desc.split()
    return f"{t[0]} {t[1]}" if t[0] in ("Buy", "BuyAndUse") and len(t) > 1 else t[0]


def mistake_type(m: dict) -> str:
    """Coarse type of a mistake, e.g. ``EndShop -> Buy Joker`` (left without buying a joker)."""
    return f"{_cat(m['played'])} -> {_cat(m['better'])}"


def describe_type(t: str) -> str:
    a, b = t.split(" -> ")
    if a == b:
        return f"{a}, but a different target was better"
    if a == "EndShop" and b.startswith(("Buy", "OpenPack")):
        return f"left the shop without {b.lower()}"
    if a == "Reroll" and b == "EndShop":
        return "rerolled instead of leaving"
    if a == "EndShop" and b == "Reroll":
        return "left instead of rerolling"
    if a.startswith("SellJoker"):
        return "sold a joker (better: " + b + ")"
    if a.startswith(("Buy", "OpenPack")) and b == "EndShop":
        return f"{a} instead of leaving"
    if a == "SelectBlind" and b == "SkipBlind":
        return "played the blind instead of skipping"
    if a == "SkipBlind" and b == "SelectBlind":
        return "skipped the blind instead of playing it"
    return f"{a} instead of {b}"


def narrate_mistake(m: dict, game: int) -> str:
    return (
        f"game {game} step {m['step']} (ante {m['ante']}, {m['phase']}): {m['context']}. "
        f"Played **{m['played']}** (policy p={m['played_prob']:.2f}); better **{m['better']}** "
        f"(p={m['better_prob']:.2f}). Q {m['q_played']:.3f} -> {m['q_better']:.3f}, "
        f"dP(win) = +{100 * m['delta']:.1f} +- {100 * m['se']:.1f} pts "
        f"(screen {100 * m['screen_delta']:+.1f}); P(win) before the decision "
        f"{100 * m['pwin_before']:.0f}%."
    )


def build_report(games: list[dict], timing: dict | None = None, top: int = 20) -> str:
    n = len(games)
    lines = [f"# Error-map report ({n} games)", ""]
    wins = sum(g["outcome"] == "Win" for g in games)
    cfg = games[0]["config"] if games else {}
    lines += [
        f"Games: {n} ({wins} wins, {n - wins} losses). Futures per alternative: screen "
        f"M={cfg.get('m_screen')}, confirm M={cfg.get('m_confirm')} on the top "
        f"{cfg.get('n_confirm')} suspects per game; horizon {cfg.get('horizon')} boss fight(s); "
        f"mistake = confirmed dP(win) > {100 * cfg.get('threshold', 0):.0f} pts and > "
        f"{cfg.get('z_min')} SE.",
        "",
    ]
    nd = sum(len(g["decisions"]) for g in games)
    nconf = sum(1 for g in games for d in g["decisions"] if "confirmed_best" in d)
    allm = [(g["game"], m) for g in games for m in g["mistakes"]]
    lines += [
        f"Analysed decisions: {nd} ({nconf} confirmed with the large M); confirmed mistakes: "
        f"{len(allm)} in {sum(1 for g in games if g['mistakes'])} games "
        f"({len(allm) / max(n, 1):.2f} per game).",
        "",
    ]
    if timing:
        lines += [
            f"Cost: {timing.get('seconds', 0):.0f} s, {timing.get('env_steps', 0):,} rollout steps "
            f"({timing.get('env_steps', 0) / max(n, 1):,.0f} per game).",
            "",
        ]
    lines += ["## Mistakes by phase and ante", ""]
    by_phase, by_ante = Counter(), Counter()
    dec_phase, dec_ante = Counter(), Counter()
    for g in games:
        for d in g["decisions"]:
            dec_phase[d["phase"]] += 1
            dec_ante[d["ante"]] += 1
    for _, m in allm:
        by_phase[m["phase"]] += 1
        by_ante[m["ante"]] += 1
    lines += ["| phase | analysed decisions | mistakes | rate |", "|---|---|---|---|"]
    for p in sorted(dec_phase):
        lines.append(f"| {p} | {dec_phase[p]} | {by_phase[p]} | {by_phase[p] / dec_phase[p]:.1%} |")
    lines += ["", "| ante | analysed decisions | mistakes | rate |", "|---|---|---|---|"]
    for a in sorted(dec_ante):
        lines.append(f"| {a} | {dec_ante[a]} | {by_ante[a]} | {by_ante[a] / dec_ante[a]:.1%} |")
    lines += ["", "## Most common mistake types", ""]
    types: dict[str, list] = defaultdict(list)
    for _, m in allm:
        types[mistake_type(m)].append(m["delta"])
    lines += ["| type | meaning | count | mean dP(win) |", "|---|---|---|---|"]
    for t, ds in sorted(types.items(), key=lambda kv: -len(kv[1]))[:12]:
        lines.append(f"| {t} | {describe_type(t)} | {len(ds)} | {100 * sum(ds) / len(ds):+.1f} pts |")
    items_left = Counter(m["better"] for _, m in allm if m["played_kind"] == "EndShop")
    if items_left:
        lines += ["", "Most frequent better actions where the shop was left:", ""]
        lines += [f"* {k} x{v}" for k, v in items_left.most_common(8)]
    lines += ["", f"## Top {top} blunders", ""]
    for i, (g, m) in enumerate(sorted(allm, key=lambda x: -x[1]["delta"])[:top], 1):
        lines.append(f"{i}. {narrate_mistake(m, g)}")
    lines += ["", "## Per game", ""]
    lines += ["| game | outcome | final ante | decisions | mistakes | eval bar (BlindSelect/Shop P(win)) |",
              "|---|---|---|---|---|---|"]
    for g in games:
        lines.append(
            f"| {g['game']} | {g['outcome']} | {g['final_ante']} | {len(g['decisions'])} | "
            f"{len(g['mistakes'])} | `{eval_bar(g['curve'])}` |"
        )
    return "\n".join(lines) + "\n"
