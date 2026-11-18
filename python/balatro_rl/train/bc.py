"""Behaviour cloning of the heuristic teacher (end-to-end smoke test of the whole model).

* ``collect``: roll out the teacher (``teacher_actions``, fair information) in ``num_envs`` envs
  for ``steps`` steps and store the *states* (observation + masks, compact dtypes) and the
  teacher's structured action per step as time-major ``.npy`` memmaps ``(T, N, ...)`` under
  ``runs/``. The reward and the episode-start flag are stored too (value targets, memory reset).
* ``load_gpu``: the whole dataset lives on the GPU (float16 floats, uint8 categoricals).
* ``train``: stateful truncated BPTT: groups of envs are swept in time order in chunks of
  ``chunk`` steps with the GRU state carried (detached) between chunks, exactly as at play time;
  chunks of different groups are interleaved. Loss = -(log p(kind) + log p(arg0) + log p(arg1) +
  log p(cards, canonical order)) of the teacher action, optionally + a value regression onto the
  discounted teacher return (pre-trains the critic).
* ``evaluate_heads``: teacher-forced accuracy per head on held-out envs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from balatro_rl import LAYOUT, BalatroVecEnv
from balatro_rl.model.migrate import LEGACY_DATA

OBS_NAMES = LAYOUT["obs_names"]
MASK_NAMES = LAYOUT["mask_names"]
_SPEC = {n: (np.dtype(d), tuple(s)) for n, d, s in LAYOUT["buffers"]}


def _store_dtype(name: str) -> np.dtype:
    dtype = _SPEC[name][0]
    if name.endswith("_valid") or name in ("kind_mask", "arg0_mask", "arg1_mask"):
        return np.dtype(bool)
    if name.endswith("_f"):
        return np.dtype(np.float16)
    if name.endswith("_i"):
        return np.dtype(np.uint8)  # every vocabulary is < 256 (checked in collect)
    return dtype  # sel_* keep their dtype (u32 / u8)


FIELDS = list(OBS_NAMES) + list(MASK_NAMES)
_LEGACY_UPGRADE = {name: fn for name, (_, fn) in LEGACY_DATA.items()}


# --------------------------------------------------------------------------- data collection


def collect(
    out_dir: Path,
    num_envs: int = 1024,
    steps: int = 1000,
    seed: int = 1,
    noise: float = 0.0,
    threads: int = 0,
    log=print,
) -> dict:
    """Teacher rollouts -> ``out_dir/{field}.npy`` (T, N, ...) + ``meta.json``. With ``noise > 0``
    each env executes a uniformly random (mask-conforming) action with that probability instead
    of the teacher's (the label stays the teacher's: visits off-teacher states)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assert max(max(v) for v in LAYOUT["vocab"].values()) < 256
    env = BalatroVecEnv(num_envs, seed=seed, num_threads=threads, max_episode_steps=10_000)
    obs, masks = env.reset()
    mm = {}
    for name in FIELDS:
        shape = _SPEC[name][1]
        mm[name] = np.lib.format.open_memmap(
            out_dir / f"{name}.npy",
            mode="w+",
            dtype=_store_dtype(name),
            shape=(steps, num_envs, *shape),
        )
    mm["actions"] = np.lib.format.open_memmap(
        out_dir / "actions.npy", mode="w+", dtype=np.uint32, shape=(steps, num_envs, 4)
    )
    mm["reward"] = np.lib.format.open_memmap(
        out_dir / "reward.npy", mode="w+", dtype=np.float32, shape=(steps, num_envs)
    )
    mm["first"] = np.lib.format.open_memmap(
        out_dir / "first.npy", mode="w+", dtype=bool, shape=(steps, num_envs)
    )
    rng = np.random.default_rng(seed + 1)
    first = np.ones(num_envs, dtype=bool)
    t0 = time.perf_counter()
    for t in range(steps):
        for name in OBS_NAMES:
            mm[name][t] = obs[name]
        for name in MASK_NAMES:
            mm[name][t] = masks[name]
        mm["first"][t] = first
        teacher = env.teacher_actions()
        mm["actions"][t] = teacher.astype(np.uint32)
        act = teacher
        if noise > 0:
            act = teacher.copy()
            pick = rng.random(num_envs) < noise
            if pick.any():
                act[pick] = env.random_actions()[pick]
        obs, masks, reward, done, _ = env.step(act)
        mm["reward"][t] = reward
        first = done != 0
        if (t + 1) % 100 == 0:
            rate = (t + 1) * num_envs / (time.perf_counter() - t0)
            log(f"  collect {t + 1}/{steps} steps ({rate:.0f}/s)")
    for a in mm.values():
        a.flush()
    fin = env.drain_finished()
    ante = np.array([r[1] for r in fin], dtype=float)
    meta = {
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "noise": noise,
        "samples": num_envs * steps,
        "collect_seconds": time.perf_counter() - t0,
        "teacher_episodes": len(fin),
        "teacher_mean_ante": float(ante.mean()) if len(fin) else None,
        "teacher_win_rate": float(np.mean([r[2] == 2 for r in fin])) if len(fin) else None,
        "bytes": int(sum(a.nbytes for a in mm.values())),
        "interface_version": LAYOUT["version"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_gpu(data_dir: Path, device, max_envs: int | None = None) -> dict[str, torch.Tensor]:
    """The dataset as GPU tensors ``(T, N, ...)`` (uint32 fields viewed as int32)."""
    out = {}
    names = FIELDS + ["actions", "reward", "first"]
    for name in names:
        arr = np.load(data_dir / f"{name}.npy", mmap_mode="r")
        if max_envs is not None:
            arr = arr[:, :max_envs]
        if arr.dtype == np.uint32:
            arr = arr.view(np.int32)
        # A dataset collected under an older interface has narrower columns: append the newer
        # ones while copying (``model.migrate.LEGACY_DATA``).
        legacy = name in _LEGACY_UPGRADE and arr.shape[-1] != _SPEC[name][1][-1]
        if legacy:
            assert arr.shape[-1] in LEGACY_DATA[name][0], f"{name}: unknown column count"
        shape = (*arr.shape[:-1], _SPEC[name][1][-1]) if legacy else arr.shape
        t = torch.empty(shape, dtype=torch.from_numpy(np.zeros(0, arr.dtype)).dtype, device=device)
        step = max(1, 64_000_000 // max(1, arr[0].nbytes))
        for t0 in range(0, arr.shape[0], step):
            part = np.array(arr[t0 : t0 + step])
            if legacy:
                part = _LEGACY_UPGRADE[name](part)
            t[t0 : t0 + step] = torch.from_numpy(part).to(device)
        out[name] = t
    return out


def discounted_returns(data: dict[str, torch.Tensor], gamma: float) -> torch.Tensor:
    """``G_t = r_t + gamma * (1 - done_t) * G_{t+1}`` per env stream; ``done_t = first[t+1]``;
    the stream end bootstraps with 0."""
    r, first = data["reward"], data["first"]
    steps = r.shape[0]
    g = torch.zeros_like(r)
    nxt = torch.zeros_like(r[0])
    for t in range(steps - 1, -1, -1):
        cont = ~first[t + 1] if t + 1 < steps else torch.zeros_like(first[0])
        nxt = r[t] + gamma * cont * nxt
        g[t] = nxt
    return g


# --------------------------------------------------------------------------- training


@dataclass
class BCConfig:
    epochs: int = 12
    chunk: int = 32  # BPTT length
    group: int = 128  # envs per update (batch = chunk * group rows)
    lr: float = 7e-4
    min_lr_frac: float = 0.05
    weight_decay: float = 1e-4
    warmup: int = 200
    clip: float = 1.0
    val_envs: int = 128
    value_coef: float = 0.1
    gamma: float = 0.99
    seed: int = 0
    max_minutes: float = 40.0


def _slice(data, t0, t1, idx):
    """Chunk ``[t0, t1)`` of env group ``idx`` as flat time-major tensors + first/actions."""
    n = idx.numel()
    length = t1 - t0
    obs = {k: data[k][t0:t1][:, idx].reshape(length * n, *data[k].shape[2:]) for k in OBS_NAMES}
    masks = {k: data[k][t0:t1][:, idx].reshape(length * n, *data[k].shape[2:]) for k in MASK_NAMES}
    actions = data["actions"][t0:t1][:, idx].reshape(length * n, 4).long()
    actions[:, 3] &= 0xFFFFFFFF
    first = data["first"][t0:t1][:, idx]
    return obs, masks, actions, first


def _bc_loss(model, data, returns, t0, t1, idx, hidden, cfg):
    obs, masks, actions, first = _slice(data, t0, t1, idx)
    r = model.evaluate_full(obs, masks, hidden, actions, first)
    ok = r["ok"]
    nll = -(r["lp_kind"] + r["lp_arg0"] + r["lp_arg1"] + r["lp_cards"])
    loss = (nll * ok).sum() / ok.sum().clamp(min=1)
    out = {"nll": loss.detach(), "ok": ok.float().mean().detach()}
    if cfg.value_coef > 0:
        target = returns[t0:t1][:, idx].reshape(-1)
        vloss = torch.nn.functional.smooth_l1_loss(r["value"], target)
        loss = loss + cfg.value_coef * vloss
        out["vloss"] = vloss.detach()
    return loss, out, r


@torch.no_grad()
def evaluate_heads(model, data, env_idx, chunk=32, group=128) -> dict:
    """Teacher-forced accuracy per head on the given envs (memory carried through time)."""
    model.eval()
    steps = data["first"].shape[0]
    acc = {}

    def add(key, hits, count):
        a = acc.setdefault(key, [0.0, 0.0])
        a[0] += float(hits)
        a[1] += float(count)

    for g0 in range(0, env_idx.numel(), group):
        idx = env_idx[g0 : g0 + group]
        hidden = model.init_hidden(idx.numel())
        for t0 in range(0, steps, chunk):
            t1 = min(steps, t0 + chunk)
            obs, masks, actions, first = _slice(data, t0, t1, idx)
            r = model.evaluate_full(obs, masks, hidden, actions, first)
            hidden = r["hidden"]
            ok = r["ok"]
            nll = -(r["lp_kind"] + r["lp_arg0"] + r["lp_arg1"] + r["lp_cards"])
            add("nll", (nll * ok).sum(), ok.sum())
            add("expressible", ok.sum(), ok.numel())
            phase = r["phase"]
            hit_all = torch.ones_like(ok)
            for comp, has in (
                ("kind", torch.ones_like(ok)),
                ("arg0", r["has_arg0"]),
                ("arg1", r["has_arg1"]),
                ("cards", r["has_cards"]),
            ):
                hit = r[f"hit_{comp}"]
                sel = ok & has
                add(f"{comp}_acc", (hit & sel).sum(), sel.sum())
                hit_all = hit_all & (hit | ~has)
                for p, name in enumerate(("blind", "play", "shop", "pack")):
                    sp = sel & (phase == p)
                    add(f"{comp}_acc/{name}", (hit & sp).sum(), sp.sum())
            add("full_acc", (hit_all & ok).sum(), ok.sum())
            for p, name in enumerate(("blind", "play", "shop", "pack")):
                sp = ok & (phase == p)
                add(f"full_acc/{name}", (hit_all & sp).sum(), sp.sum())
                add(f"rows/{name}", sp.sum(), ok.numel())
    model.train()
    out = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}
    return out


def train(model, data, cfg: BCConfig, out_dir: Path, log=print) -> dict:
    """Run BC; returns the history (per-epoch metrics). Saves ``model.pt`` and ``history.json``."""
    from balatro_rl.model import save_checkpoint

    dev = model.device
    steps, n_envs = data["first"].shape
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_envs, generator=g)
    val_idx = perm[: cfg.val_envs].to(dev)
    train_ids = perm[cfg.val_envs :]
    returns = discounted_returns(data, cfg.gamma) if cfg.value_coef > 0 else None
    n_groups = len(train_ids) // cfg.group
    updates_per_epoch = n_groups * math.ceil(steps / cfg.chunk)
    total_updates = updates_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(u):
        if u < cfg.warmup:
            return cfg.lr * (u + 1) / cfg.warmup
        p = (u - cfg.warmup) / max(1, total_updates - cfg.warmup)
        return cfg.lr * (
            cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
        )

    log(
        f"train: {len(train_ids)} train envs / {cfg.val_envs} val envs x {steps} steps, "
        f"{n_groups} groups x chunk {cfg.chunk}, {updates_per_epoch} updates/epoch"
    )
    history = []
    u = 0
    t_start = time.perf_counter()
    model.train()
    for epoch in range(cfg.epochs):
        ep_perm = train_ids[torch.randperm(len(train_ids), generator=g)]
        groups = [ep_perm[i * cfg.group : (i + 1) * cfg.group].to(dev) for i in range(n_groups)]
        hiddens = [model.init_hidden(cfg.group) for _ in groups]
        offset = int(torch.randint(0, cfg.chunk, (1,), generator=g))  # vary chunk boundaries
        bounds = [(0, offset)] if offset else []
        bounds += [(t, min(steps, t + cfg.chunk)) for t in range(offset, steps, cfg.chunk)]
        sums = {"nll": 0.0, "vloss": 0.0, "ok": 0.0}
        cnt = 0
        t_ep = time.perf_counter()
        for t0, t1 in bounds:
            for gi in torch.randperm(n_groups, generator=g).tolist():
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(u)
                loss, out, r = _bc_loss(model, data, returns, t0, t1, groups[gi], hiddens[gi], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                opt.step()
                hiddens[gi] = r["hidden"].detach()
                for k in sums:
                    if k in out:
                        sums[k] += float(out[k])
                cnt += 1
                u += 1
        elapsed = time.perf_counter() - t_start
        val = evaluate_heads(model, data, val_idx, cfg.chunk, cfg.group)
        rec = {
            "epoch": epoch + 1,
            "updates": u,
            "seconds": elapsed,
            "epoch_seconds": time.perf_counter() - t_ep,
            "train_nll": sums["nll"] / cnt,
            "train_vloss": sums["vloss"] / cnt,
            "lr": lr_at(u),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(rec)
        log(
            f"epoch {epoch + 1:>2} {rec['epoch_seconds']:.0f}s  train_nll={rec['train_nll']:.3f} "
            f"val_nll={val['nll']:.3f} kind={val['kind_acc']:.3f} arg0={val['arg0_acc']:.3f} "
            f"cards={val['cards_acc']:.3f} full={val['full_acc']:.3f}"
        )
        save_checkpoint(out_dir / "model.pt", model, bc_config=asdict(cfg), history=history)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if elapsed > cfg.max_minutes * 60:
            log("time budget reached, stopping")
            break
    return {"history": history, "final": history[-1]}
