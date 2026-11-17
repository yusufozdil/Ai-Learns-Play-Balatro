"""Error maps: counterfactual analysis of a reference game (``docs/design/error_maps.md``).

Pipeline (all batched, ``RolloutPool`` is the engine):

1. *Trunk*: replay the game (recorded actions, or the deterministic policy live) in a ``VecEnv``;
   at every step run the frozen policy (recurrent state included), score the state with the
   calibrated P(win) head (the "eval bar") and, at BlindSelect / Shop (optionally Pack) decisions,
   export the state, enumerate every legal mask-conforming action, score them with the policy's
   structured log-probability and keep the played action plus the top alternatives.
2. *Screen* (M futures): for each kept action, ``import_state`` of the decision state with a
   **full reseed** (all RNG streams, draw order, face-down cards; same seed for every alternative
   of a decision = common random numbers), apply the action, let the policy play on until the end of
   the next boss fight (``horizon`` bosses) and score the leaf with the P(win) head (terminal runs
   count 0 / 1). ``Q(a)`` is the mean over futures.
3. *Confirm* (M2 futures, fresh seeds): the most suspicious decisions of a game are re-run with
   many more futures and seeds that were not used in the screen (no winner's curse).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from balatro_rl import BalatroVecEnv, _engine
from balatro_rl.analysis.describe import describe_action, describe_state
from balatro_rl.analysis.features import (
    BLIND_BOSS,
    PHASE_BLIND,
    PHASE_PLAY,
    PHASE_SHOP,
    PolicyRunner,
    obs_ante,
    obs_blind,
    obs_phase,
)
from balatro_rl.model.inputs import prepare
from balatro_rl.select import KIND_NAMES, conforms

MASK64 = (1 << 64) - 1
STAGE_SCREEN, STAGE_CONFIRM = 1, 2


def mix_seed(*xs: int) -> int:
    """Deterministic 64-bit seed from integers (SplitMix64 rounds)."""
    z = 0x9E3779B97F4A7C15
    for x in xs:
        z = (z + (int(x) & MASK64) + 0x9E3779B97F4A7C15) & MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
    return z


@dataclass
class Config:
    k: int = 3  # played action + (k - 1) alternatives
    min_alt_prob: float = 0.01
    m_screen: int = 8
    m_confirm: int = 64
    n_confirm: int = 5  # decisions re-checked per game
    horizon: int = 1  # number of boss fights the futures run through
    threshold: float = 0.03  # mistake: confirmed dP(win) above this ...
    z_min: float = 2.0  # ... and above z_min standard errors
    phases: tuple[int, ...] = (PHASE_BLIND, PHASE_SHOP)
    deterministic: bool = True  # rollout policy (the det policy is what is analysed)
    max_rollout_steps: int = 300
    slots: int = 2048
    seed: int = 0xE44A  # salt of every future's seed


@dataclass
class Decision:
    game: int
    step: int
    phase: int
    ante: int
    state: str
    hidden_out: np.ndarray
    pwin: float
    cands: list  # [(action tuple, prob)], played first
    # filled by the analysis
    screen: np.ndarray | None = None  # (len(cands), M1)
    confirm: np.ndarray | None = None  # (len(cands), M2)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- rollouts


class RolloutPool:
    """Batched futures: slots of a ``VecEnv`` fed with jobs ``(state, hidden, action, seed)``."""

    def __init__(self, runner: PolicyRunner, head, cfg: Config, threads: int = 0):
        self.runner, self.head, self.cfg = runner, head, cfg
        self.env = BalatroVecEnv(
            cfg.slots, seed=1, num_threads=threads, max_episode_steps=3000
        )
        self.env.reset()
        self.env_steps = 0
        self.model_rows = 0

    def run(self, jobs: list) -> tuple[np.ndarray, dict]:
        """``jobs``: list of ``(state_json, hidden_post (mem,), action (4,), seed)``. Returns
        ``(values (n,), info)``: P(win) of each future's leaf (terminal: 0 / 1)."""
        cfg, env, runner = self.cfg, self.env, self.runner
        n, S = len(jobs), cfg.slots
        dev = runner.dev
        vals = np.zeros(n)
        kinds = np.zeros(n, np.int8)  # 0 leaf, 1 terminal, 2 step cap
        slot_job = np.full(S, -1)
        fresh = np.zeros(S, bool)
        forced = np.zeros((S, 4), np.int64)
        bosscnt = np.zeros(S, np.int64)
        inboss = np.zeros(S, bool)
        nsteps = np.zeros(S, np.int64)
        mem = jobs[0][1].shape[0] if n else 0
        hidden = torch.zeros(S, mem, device=dev)
        nxt = 0
        steps0 = 0
        while True:
            free = np.flatnonzero(slot_job < 0)[: n - nxt]
            if len(free):
                hs = np.empty((len(free), mem), np.float32)
                for t, s in enumerate(free):
                    state, h, act, seed = jobs[nxt]
                    env.import_state(int(s), state, reseed=int(seed))
                    slot_job[s] = nxt
                    forced[s] = act
                    hs[t] = h
                    nxt += 1
                fresh[free] = True
                bosscnt[free] = 0
                inboss[free] = False
                nsteps[free] = 0
                hidden[torch.from_numpy(free).to(dev)] = torch.from_numpy(hs).to(dev)
            live = slot_job >= 0
            if not live.any():
                break
            obs, masks = env.obs, env.masks
            act, h2, feats, _ = runner.run(obs, masks, hidden, cfg.deterministic)
            fr_t = torch.from_numpy(fresh).to(dev)
            hidden = torch.where(fr_t[:, None], hidden, h2)  # a fresh slot keeps the post-state
            act_np = act.cpu().numpy()
            act_np[fresh] = forced[fresh]
            ph, bl = obs_phase(obs), obs_blind(obs)
            rows = live & ~fresh
            boss_now = (ph == PHASE_PLAY) & (bl == BLIND_BOSS)
            bosscnt += rows & boss_now & ~inboss
            inboss = np.where(rows, boss_now, inboss)
            leaf = rows & (bosscnt >= cfg.horizon) & (ph == PHASE_BLIND)
            cap = rows & ~leaf & (nsteps >= cfg.max_rollout_steps)
            fin = np.flatnonzero(leaf | cap)
            if len(fin):
                p = self.head(feats[torch.from_numpy(fin).to(dev)]).cpu().numpy()
                jj = slot_job[fin]
                vals[jj] = p
                kinds[jj] = np.where(leaf[fin], 0, 2)
                slot_job[fin] = -1
            nsteps += rows
            steps0 += int(live.sum())
            _, _, _, done, _ = env.step(act_np)
            term = np.flatnonzero((done != 0) & (slot_job >= 0))
            if len(term):
                jj = slot_job[term]
                vals[jj] = (done[term] == 2).astype(np.float64)
                kinds[jj] = 1
                slot_job[term] = -1
            fresh[:] = False
        self.env_steps += steps0
        return vals, {
            "env_steps": steps0,
            "jobs": n,
            "leaf": int((kinds == 0).sum()),
            "terminal": int((kinds == 1).sum()),
            "capped": int((kinds == 2).sum()),
        }


# --------------------------------------------------------------------------- trunk


def _index_prepared(prepared, idx: torch.Tensor):
    o, m = prepared
    return {k: v[idx] for k, v in o.items()}, {k: v[idx] for k, v in m.items()}


def enumerate_candidates(state: str, masks_row: dict) -> list[tuple[int, ...]]:
    """Every legal action of the state that the policy can express (mask-conforming)."""
    g = _engine.Game.from_state(state)
    out, seen = [], set()
    for a in g.legal_parts():
        a = tuple(int(x) for x in a)
        if a in seen:
            continue
        seen.add(a)
        if conforms(masks_row, 0, a):
            out.append(a)
    return out


def run_trunk(
    runner: PolicyRunner,
    head,
    games: list[int],
    base_seed: int,
    acts: dict[int, list] | None,
    cfg: Config,
    max_episode_steps: int = 3000,
) -> tuple[dict[int, list], dict[int, dict]]:
    """Play ``games`` (env indices of base ``base_seed``) from the start. ``acts[g]`` = recorded
    actions (replayed exactly); ``None`` = the deterministic policy plays. Returns
    ``(decisions per game, trunk info per game: curve, outcome, steps)``."""
    dev = runner.dev
    n_env = max(games) + 1
    env = BalatroVecEnv(n_env, seed=base_seed, max_episode_steps=max_episode_steps)
    obs, masks = env.reset()
    mem = runner.model.mem_dim
    hidden = torch.zeros(n_env, mem, device=dev)
    active = np.zeros(n_env, bool)
    active[games] = True
    step = np.zeros(n_env, np.int64)
    decisions: dict[int, list] = {g: [] for g in games}
    info = {g: {"curve": [], "outcome": None, "steps": 0} for g in games}
    done_set: set[int] = set()
    while active.any():
        idx = np.flatnonzero(active)
        o = {k: v[idx] for k, v in obs.items()}
        m = {k: v[idx] for k, v in masks.items()}
        idx_t = torch.from_numpy(idx).to(dev)
        h_in = hidden[idx_t]
        prepared = prepare(o, m, dev)
        act, h2, feats, value = runner.run(o, m, h_in, True, prepared=prepared)
        hidden[idx_t] = h2
        pw = head(feats).cpu().numpy()
        val = value.float().cpu().numpy()
        ph, an = obs_phase(o), obs_ante(o)
        model_act = act.cpu().numpy()
        full = np.zeros((n_env, 4), np.int64)
        for r, e in enumerate(idx):
            full[e] = acts[e][step[e]] if acts is not None else model_act[r]
            info[e]["curve"].append((int(step[e]), int(ph[r]), int(an[r]), float(pw[r]), float(val[r])))
        # decisions to branch
        sel = [r for r in range(len(idx)) if int(ph[r]) in cfg.phases]
        if sel:
            states = [env.export_state(int(idx[r])) for r in sel]
            cand_lists, rows_of = [], []
            for t, r in enumerate(sel):
                row = {k: v[r : r + 1] for k, v in m.items()}
                c = enumerate_candidates(states[t], row)
                played = tuple(int(x) for x in full[idx[r]])
                if played not in c:
                    c.append(played)
                cand_lists.append(c)
                rows_of.extend([r] * len(c))
            ridx = torch.tensor(rows_of, device=dev)
            sub = _index_prepared(prepared, ridx)
            flat = torch.tensor([a for c in cand_lists for a in c], device=dev, dtype=torch.long)
            lp, _ok = runner.logp_of(sub, h_in[ridx], flat)
            lp = lp.float().cpu().numpy()
            pos = 0
            h2_np = h2.float().cpu().numpy()
            for t, r in enumerate(sel):
                c = cand_lists[t]
                p = np.exp(lp[pos : pos + len(c)])
                pos += len(c)
                played = tuple(int(x) for x in full[idx[r]])
                pi = c.index(played)
                order = [i for i in np.argsort(-p) if i != pi and p[i] >= cfg.min_alt_prob]
                keep = [pi] + order[: cfg.k - 1]
                if len(keep) < 2:
                    continue
                e = int(idx[r])
                decisions[e].append(
                    Decision(
                        game=e,
                        step=int(step[e]),
                        phase=int(ph[r]),
                        ante=int(an[r]),
                        state=states[t],
                        hidden_out=h2_np[r],
                        pwin=float(pw[r]),
                        cands=[(c[i], float(p[i])) for i in keep],
                        meta={"n_legal": len(c), "mass_covered": float(p.sum())},
                    )
                )
        _, _, _, done, _ = env.step(full)
        step[idx] += 1
        for rec in env.drain_finished():
            e = rec[0]
            if e in info and e not in done_set:
                done_set.add(e)
                active[e] = False
                info[e]["outcome"] = int(rec[2])
                info[e]["steps"] = int(step[e])
                info[e]["ante"] = int(rec[1])
    return decisions, info


# --------------------------------------------------------------------------- analysis driver


def _jobs(decisions: list[Decision], M: int, stage: int, cfg: Config) -> list:
    jobs = []
    for d in decisions:
        for j in range(M):
            seed = mix_seed(cfg.seed, d.game, d.step, j, stage)  # shared by all alternatives
            for a, _p in d.cands:
                jobs.append((d.state, d.hidden_out, np.array(a, np.int64), seed))
    return jobs


def _stats(q: np.ndarray) -> dict:
    """``q`` (n_alt, M) leaf values, row 0 = played. Paired (common random numbers) differences
    of every alternative to the played action."""
    M = q.shape[1]
    out = {"q": q.mean(1).tolist(), "m": M}
    diff = q[1:] - q[0:1]
    out["d"] = diff.mean(1).tolist()
    se = diff.std(1, ddof=1) / np.sqrt(M) if M > 1 else np.full(len(diff), np.nan)
    out["se"] = se.tolist()
    return out


def screen_and_confirm(
    pool: RolloutPool, decisions: dict[int, list], cfg: Config, log=print
) -> dict:
    """Runs both stages on the decisions of a chunk of games; fills ``Decision.screen`` /
    ``confirm`` and returns timing / counters."""
    allds = [d for ds in decisions.values() for d in ds]
    t0 = time.perf_counter()
    stats = {"decisions": len(allds)}
    if not allds:
        return stats
    jobs = _jobs(allds, cfg.m_screen, STAGE_SCREEN, cfg)
    vals, inf = pool.run(jobs)
    stats["screen"] = inf | {"seconds": time.perf_counter() - t0}
    pos = 0
    for d in allds:  # job order: decision, future j, alternative a
        a = len(d.cands)
        d.screen = vals[pos : pos + cfg.m_screen * a].reshape(cfg.m_screen, a).T
        pos += cfg.m_screen * a
    # choose the suspects of each game: best lower bound of the screened advantage
    conf = []
    for ds in decisions.values():
        scored = []
        for d in ds:
            s = _stats(d.screen)
            lb = np.array(s["d"]) - np.array(s["se"])
            b = int(np.argmax(lb))
            d.meta["screen_best"] = b + 1
            d.meta["screen_lb"] = float(lb[b])
            if s["d"][b] > 0:
                scored.append((float(lb[b]), d))
        scored.sort(key=lambda x: -x[0])
        conf.extend(d for _, d in scored[: cfg.n_confirm])
    t1 = time.perf_counter()
    if conf:
        jobs = _jobs(conf, cfg.m_confirm, STAGE_CONFIRM, cfg)
        vals, inf = pool.run(jobs)
        stats["confirm"] = inf | {"seconds": time.perf_counter() - t1, "decisions": len(conf)}
        pos = 0
        for d in conf:
            a = len(d.cands)
            d.confirm = vals[pos : pos + cfg.m_confirm * a].reshape(cfg.m_confirm, a).T
            pos += cfg.m_confirm * a
    return stats


def finalize_game(
    g: int, ds: list[Decision], tinfo: dict, cfg: Config, seed_info: dict | None = None
) -> dict:
    """Per-game JSON-able result: eval bar, analysed decisions, flagged mistakes."""
    out_decisions, mistakes = [], []
    for d in ds:
        s = json.loads(d.state)
        s1 = _stats(d.screen)
        s2 = _stats(d.confirm) if d.confirm is not None else None
        rec = {
            "step": d.step,
            "ante": d.ante,
            "phase": ["BlindSelect", "Play", "Shop", "Pack"][d.phase],
            "context": describe_state(s),
            "pwin": d.pwin,
            "n_legal": d.meta["n_legal"],
            "actions": [],
        }
        for i, (a, p) in enumerate(d.cands):
            e = {
                "action": list(a),
                "desc": describe_action(s, a),
                "kind": KIND_NAMES[a[0]],
                "prob": p,
                "q1": s1["q"][i],
            }
            if i:
                e.update(d1=s1["d"][i - 1], se1=s1["se"][i - 1])
                if s2:
                    e.update(q2=s2["q"][i], d2=s2["d"][i - 1], se2=s2["se"][i - 1])
            elif s2:
                e["q2"] = s2["q"][0]
            rec["actions"].append(e)
        if s2:
            d2, se2 = np.array(s2["d"]), np.array(s2["se"])
            b = int(np.argmax(d2))
            rec["confirmed_best"] = b + 1
            if d2[b] > cfg.threshold and d2[b] > cfg.z_min * max(se2[b], 1e-3):
                mistakes.append(
                    {
                        "step": d.step,
                        "ante": d.ante,
                        "phase": rec["phase"],
                        "context": rec["context"],
                        "played": rec["actions"][0]["desc"],
                        "played_kind": rec["actions"][0]["kind"],
                        "better": rec["actions"][b + 1]["desc"],
                        "better_kind": rec["actions"][b + 1]["kind"],
                        "played_prob": rec["actions"][0]["prob"],
                        "better_prob": rec["actions"][b + 1]["prob"],
                        "q_played": s2["q"][0],
                        "q_better": s2["q"][b + 1],
                        "delta": float(d2[b]),
                        "se": float(se2[b]),
                        "pwin_before": d.pwin,
                        "screen_delta": rec["actions"][b + 1]["d1"],
                    }
                )
        out_decisions.append(rec)
    return {
        "game": g,
        "outcome": "Win" if tinfo["outcome"] == 2 else "Loss",
        "final_ante": tinfo.get("ante"),
        "steps": tinfo["steps"],
        "curve": [
            {"step": s, "phase": p, "ante": a, "pwin": w, "value": v}
            for s, p, a, w, v in tinfo["curve"]
        ],
        "decisions": out_decisions,
        "mistakes": sorted(mistakes, key=lambda m: -m["delta"]),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
    }


def record_games(
    runner: PolicyRunner, base_seed: int, n_games: int, max_episode_steps: int = 3000
) -> tuple[list[list[tuple]], list[int]]:
    """Play the first episode of ``n_games`` envs with the deterministic policy; returns the
    recorded actions per game and the done codes (2 = win, 1 = loss, 3 = truncated)."""
    dev = runner.dev
    env = BalatroVecEnv(n_games, seed=base_seed, max_episode_steps=max_episode_steps)
    obs, masks = env.reset()
    hidden = runner.init_hidden(n_games)
    active = np.ones(n_games, bool)
    acts: list[list[tuple]] = [[] for _ in range(n_games)]
    code = [0] * n_games
    while active.any():
        idx = np.flatnonzero(active)
        o = {k: v[idx] for k, v in obs.items()}
        m = {k: v[idx] for k, v in masks.items()}
        idx_t = torch.from_numpy(idx).to(dev)
        a, h2, _, _ = runner.run(o, m, hidden[idx_t], True)
        hidden[idx_t] = h2
        a = a.cpu().numpy()
        full = np.zeros((n_games, 4), np.int64)
        full[idx] = a
        for r, e in enumerate(idx):
            acts[e].append(tuple(int(x) for x in a[r]))
        obs, masks, _, _, _ = env.step(full)
        for rec in env.drain_finished():
            if active[rec[0]]:
                active[rec[0]] = False
                code[rec[0]] = int(rec[2])
    return acts, code
