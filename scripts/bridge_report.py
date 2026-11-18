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


def endless_stats(ends: list[dict]) -> dict | None:
    """Endless-mode summary of the ``game_end`` records (None if no run continued).

    A won run counts as continued when it has ``endless_ante`` (older logs: no ``endless_end``,
    which then reads ``unknown``). ``endless_end`` is ``loss`` / ``cap`` for a regular end and
    ``crash`` / ``hang`` / ``error`` / ``max_decisions`` when the continuation broke down (the
    run is still a win).
    """
    endl = [r for r in ends if r.get("endless_ante") is not None and r["outcome"] == "win"]
    if not endl:
        return None
    ea = sorted(r["endless_ante"] for r in endl)
    return {
        "runs": len(endl),
        "mean_ante": sum(ea) / len(ea),
        "antes": ea,
        "ante_hist": dict(sorted(Counter(ea).items())),
        "ends": dict(Counter(r.get("endless_end") or "unknown" for r in endl)),
        # the ante counter reads win_ante + 1 as soon as the boss is beaten, so +2 means that the
        # first endless ante was cleared too
        "beyond_win": sum(r["endless_ante"] >= (r.get("won_ante") or 8) + 2 for r in endl),
    }


def print_endless(ends: list[dict]) -> None:
    st = endless_stats(ends)
    if st is None:
        return
    print(
        f"endless: {st['runs']} won runs, final ante mean {st['mean_ante']:.2f}, "
        f"{st['beyond_win']} cleared at least one endless ante; highest-ante histogram "
        f"{st['ante_hist']}"
    )
    broke = sum(n for k, n in st["ends"].items() if k not in ("loss", "cap", "unknown"))
    print(f"endless ends: {st['ends']}  (broken down by crash/hang/...: {broke})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--max-list", type=int, default=40, help="mismatches to list")
    a = ap.parse_args()
    recs = load(a.logs)

    ends = [r for r in recs if r["ev"] == "game_end"]
    print(f"== games: {len(ends)}")
    print("outcomes:", dict(Counter(r["outcome"] for r in ends)))
    done = [r for r in ends if r["outcome"] in ("win", "loss")]
    lost = [r for r in ends if r["outcome"] not in ("win", "loss")]
    if done:
        wins = sum(r["outcome"] == "win" for r in done)
        antes = [
            r.get("won_ante") or r["ante"] if r["outcome"] == "win" else r["ante"] for r in done
        ]
        print(
            f"completed games: {len(done)}  win rate: {100 * wins / len(done):.1f}%  "
            f"mean ante: {sum(antes) / len(antes):.2f}  antes: {sorted(antes)}"
        )
    if lost:
        print(
            f"excluded from the win rate (hang/error/max_decisions BEFORE a win): {len(lost)} "
            f"games, seeds {[r['seed'] for r in lost]}"
        )
    print_endless(ends)
    restarts = [r for r in recs if r["ev"] == "restart"]
    if restarts:
        ok = sum(bool(r.get("ok")) for r in restarts)
        print(f"game restarts: {len(restarts)} ({ok} ok, {len(restarts) - ok} failed)")
    for r in ends:
        e = f"endless_ante={r['endless_ante']} " if r.get("endless_ante") is not None else ""
        e += f"endless_end={r['endless_end']} " if r.get("endless_end") else ""
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
