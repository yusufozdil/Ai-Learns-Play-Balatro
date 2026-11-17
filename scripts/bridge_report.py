"""Summarise bridge logs: win rate, ante reached, rejected actions and score parity.

    uv run --no-sync python scripts/bridge_report.py runs/bridge/run1.jsonl [more.jsonl ...]

Parity verdicts per played hand (``parity.compare``): ``exact`` (engine prediction == real chips),
``in_range`` (stochastic hand, real inside the engine's min..max over reseeded copies),
``mismatch`` (deterministic prediction differs), ``stochastic_miss`` (outside the range).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def factors(play: dict) -> set[str]:
    """Things present in a played hand that could explain a mismatch."""
    f = set()
    for j in play.get("jokers", []):
        f.add(f"joker:{j['id']}" + (f"/{j['ed']}" if j["ed"] != "Base" else ""))
    for c in play.get("cards", []):
        base, _, rest = c.partition("+")
        if rest:
            f.add("enh:" + rest.split("/")[0].split("#")[0])
        if "/" in c:
            f.add("edition:" + c.split("/")[1].split("#")[0])
        if "#" in c:
            f.add("seal:" + c.split("#")[1])
    if play.get("blind") == "Boss" and play.get("boss") and not play.get("boss_disabled"):
        f.add("boss:" + play["boss"])
    for fl in (play.get("pred") or {}).get("flags", []):
        f.add("flag:" + fl)
    if not (play.get("pred") or {}).get("deterministic", True):
        f.add("stochastic")
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--max-list", type=int, default=40, help="mismatches to list")
    a = ap.parse_args()
    recs = load(a.logs)

    ends = [r for r in recs if r["ev"] == "game_end"]
    print(f"== games: {len(ends)}")
    outcomes = Counter(r["outcome"] for r in ends)
    print("outcomes:", dict(outcomes))
    done = [r for r in ends if r["outcome"] in ("win", "loss")]
    if done:
        wins = sum(r["outcome"] == "win" for r in done)
        antes = [r["ante"] for r in done]
        print(
            f"completed games: {len(done)}  win rate: {100 * wins / len(done):.1f}%  "
            f"mean ante: {sum(antes) / len(antes):.2f}  antes: {sorted(antes)}"
        )
    endl = [r for r in ends if r.get("endless_ante") is not None]
    if endl:
        ea = [r["endless_ante"] for r in endl]
        mean = sum(ea) / len(ea)
        print(f"endless: {len(endl)} won runs continued, final ante mean {mean:.2f}, {sorted(ea)}")
    for r in ends:
        e = f"endless_ante={r['endless_ante']} " if r.get("endless_ante") is not None else ""
        print(
            f"  game {r['game']:>3} seed {r['seed']:<8} {r['outcome']:<5} ante {r['ante']} "
            f"decisions {r['decisions']:>4} plays {r['plays']:>3} exact {r['exact']:>3} "
            f"mismatch {r['mismatch']:>2} rejects {r['rejects']:>2} {r['seconds']:.0f}s "
            f"{e}{r['note']}"
        )

    rej = [r for r in recs if r["ev"] == "reject"]
    print(f"\n== rejected actions: {len(rej)}")
    for k, n in Counter(
        (r["action"]["kind"], r["err"].split(": ", 1)[-1][:70]) for r in rej
    ).most_common(15):
        print(f"  {n:>3} x {k[0]}: {k[1]}")

    warns = Counter(r.get("msg") for r in recs if r["ev"] == "warn")
    if warns:
        print("\n== warnings:", dict(warns))
        for r in [r for r in recs if r["ev"] == "warn" and r.get("msg") == "price_mismatch"][:8]:
            print("  price mismatch", r["items"])

    plays = [r for r in recs if r["ev"] == "play"]
    print(f"\n== parity: {len(plays)} played hands")
    if not plays:
        return
    st = Counter(p["verdict"]["status"] for p in plays)
    det = [p for p in plays if (p.get("pred") or {}).get("deterministic")]
    det_exact = sum(p["verdict"]["status"] == "exact" for p in det)
    print("verdicts:", dict(st))
    print(f"all hands exact or in range: {100 * (st['exact'] + st['in_range']) / len(plays):.1f}%")
    if det:
        print(
            f"deterministic hands exact: {det_exact}/{len(det)} = {100 * det_exact / len(det):.1f}%"
        )
    ht = [p for p in plays if p.get("real_hand") and (p["pred"] or {}).get("hand_type")]
    ht_ok = sum(p["real_hand"] == p["pred"]["hand_type"] for p in ht)
    print(f"hand type agrees: {ht_ok}/{len(ht)}")

    total: Counter = Counter()
    bad: Counter = Counter()
    for p in plays:
        bad_p = p["verdict"]["status"] in ("mismatch", "stochastic_miss")
        for f in factors(p):
            total[f] += 1
            bad[f] += bad_p
    rows = [(bad[f] / total[f], bad[f], total[f], f) for f in total if bad[f]]
    rows.sort(reverse=True)
    if rows:
        print("\nmismatch rate by factor present in the hand (rate, bad/total):")
        for rate, b, t, f in rows[:30]:
            print(f"  {100 * rate:5.1f}%  {b:>3}/{t:<4} {f}")

    bads = [p for p in plays if p["verdict"]["status"] in ("mismatch", "stochastic_miss")]
    print(f"\n== mismatches ({len(bads)}), first {a.max_list}:")
    groups = defaultdict(list)
    for p in bads:
        groups[
            tuple(
                sorted(
                    f
                    for f in factors(p)
                    if f.startswith(("joker:", "boss:", "enh:", "edition:", "seal:"))
                )
            )
        ].append(p)
    for p in bads[: a.max_list]:
        pr = p["pred"]
        print(
            f"  g{p['game']} a{p['ante']} {p['blind']}/{p['boss']}: pred {pr.get('pred')} "
            f"[{pr.get('min')}..{pr.get('max')}] real {p['real']} diff {p['verdict'].get('diff')} "
            f"hand {pr.get('hand_type')}/{p['real_hand']} cards {p['cards']} "
            f"jokers {[j['id'] for j in p['jokers']]}"
        )
    print(f"\n({len(groups)} distinct factor combinations among mismatches)")


if __name__ == "__main__":
    main()
