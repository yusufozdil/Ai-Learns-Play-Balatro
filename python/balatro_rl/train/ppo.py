"""Recurrent PPO for the team model (CleanRL style; design notes: docs/design/training.md).

Data flow of one update (``Trainer.train``):

1. ``collect``: ``rollout_len`` (T) steps in ``num_envs`` (N) envs. Per step: ``prepare`` the env
   buffers (H2D), store them in compact GPU buffers ``(T, N, ...)``, ``model.act`` (sample), then
   the *route* of the current curriculum stage decides per env whether the model's action or the
   teacher's (``env.teacher_actions(select)``, only for the envs that need it) is executed.
   The GRU state is stored at every ``chunk``-th step (``h0``).
2. ``compute_gae`` with the shared critic (``first`` flags cut the trace at episode starts).
3. ``update``: ``epochs`` passes over recurrent minibatches = (time chunk of ``chunk`` steps) x
   (group of ``group`` envs). The chunk's GRU state is the stored ``h0`` of the rollout (detached),
   so in the first epoch before any step ``evaluate`` reproduces the rollout log-probabilities
   exactly (ratio = 1). Only *model-controlled* rows (``ctrl``) enter the policy loss, the entropy
   bonus and the KL penalty; the value loss uses every row.
4. Optional KL penalty to a frozen reference policy (the BC checkpoint), coefficient decaying
   linearly to 0: estimator ``k3 = exp(d) - 1 - d`` with ``d = log pi_ref(a) - log pi(a)`` on the
   sampled actions (reference log-probs are computed once per rollout, no-grad).

Reward scaling is a constant (``reward_scale``). Truncated episodes (``done == 3``, only after
``max_episode_steps``) are bootstrapped with the value of the observation the last action was
taken in (the env does not keep the last observation of the old episode, docs/design/rl_interface.md
section 8; a truncated policy is almost always looping, so ``V(s_T) ~ V(s_{T-1})``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from balatro_rl import LAYOUT, BalatroVecEnv
from balatro_rl.model import HEAD_NAMES, ModelConfig, TeamModel, load_checkpoint, save_checkpoint
from balatro_rl.model.inputs import MASK_NAMES, OBS_NAMES, prepare
from balatro_rl.train import evaluate as ev
from balatro_rl.train.curriculum import PHASES, Curriculum

KINDS = LAYOUT["caps"]["kinds"]
KIND_NAMES = [
    "SelectBlind", "SkipBlind", "Play", "Discard", "SellJoker", "EndShop", "UseConsumable",
    "SellConsumable", "MoveJoker", "Buy", "BuyAndUse", "OpenPack", "BuyVoucher", "Reroll",
    "PickPackItem", "SkipPack", "RerollBoss",
]  # fmt: skip
PARTS = ("kind", "arg0", "arg1", "cards")
_SPEC = {n: (np.dtype(d), tuple(s)) for n, d, s in LAYOUT["buffers"]}


@dataclass
class PPOConfig:
    # --- run
    name: str = "ppo"
    out_dir: str = "runs/ppo"
    seed: int = 0
    total_steps: float = 20e6  # env steps (all rows, teacher steps included)
    # --- env
    num_envs: int = 1024
    rollout_len: int = 128
    max_episode_steps: int = 3000
    threads: int = 0
    reward_scale: float = 0.1
    reward: dict = field(
        default_factory=dict
    )  # env reward terms (blind/ante/win/loss/shaping/invalid/reorder/margin/margin_cap/zero_play)
    # Training run configuration: comma lists of deck / stake names; every episode of every env
    # draws its (deck, stake) uniformly from the product (the observation tells the policy which).
    decks: str = "red"
    stakes: str = "white"
    endless: bool = False  # TRAINING envs only: continue after the ante-8 win (eval stays normal)
    endless_max_ante: int = 12  # endless: the episode ends after this ante's boss is beaten
    # --- PPO
    gamma: float = 0.997
    lam: float = 0.95
    epochs: int = 2
    chunk: int = 32  # BPTT length (rollout_len must be a multiple)
    group: int = 128  # envs per minibatch (rows = chunk * group)
    clip: float = 0.2
    vf_coef: float = 0.5
    clip_vloss: bool = False
    ent_coef: float = 0.003
    max_grad_norm: float = 0.5
    lr: float = 1e-4
    lr_final_frac: float = 0.1
    adam_eps: float = 1e-5
    target_kl: float | None = None  # stop the epochs early when approx_kl exceeds this
    norm_adv: bool = True
    bootstrap_truncation: bool = True  # GAE bootstraps through step-limit truncation (not terminal)
    # --- speed options (all opt-in; the defaults are the original fp32 / eager training)
    update_dtype: str = "fp32"  # "bf16": autocast the update (and KL reference) passes
    recompute_old_logp: bool = False  # old log-probs = a no-grad pass in the update dtype
    compile: bool = False  # torch.compile the encoder's Transformer stack (token-bucketed)
    compile_bucket: int = 8  # tokens padded to a multiple of this (compile only)
    profile: bool = False  # per-section timings (cuda-synchronised) in the log: ``prof`` dict
    # --- init from BC / KL to the BC policy
    init_checkpoint: str | None = None
    kl_coef: float = 0.0  # start coefficient of the KL penalty to the init checkpoint's policy
    kl_anneal_updates: int = 100  # linear decay to 0 over this many updates
    critic_warmup: int = 0  # first updates: only the critic MLP is trained (policy frozen)
    # --- curriculum
    curriculum: str = "all"
    # --- evaluation / logging / checkpoints
    eval_every: int = 10  # updates
    eval_games: int = 512
    eval_sampled: bool = True
    eval_move_limit: int | None = None  # validation repetition guard (evaluate.ModelActor)
    eval_deck: str = "red"  # validation configuration (default Red / White: comparable to history)
    eval_stake: str = "white"
    eval_set: str = "dev"  # evaluate.EVAL_SETS: dev = prefix of the test set (history), val = opt-in
    save_every: int = 5
    ckpt_keep_best: bool = True
    # --- model (used when not initialised from a checkpoint)
    model: dict = field(default_factory=lambda: dataclasses.asdict(ModelConfig()))

    def __post_init__(self):
        assert self.rollout_len % self.chunk == 0, "rollout_len must be a multiple of chunk"
        assert self.num_envs % self.group == 0, "num_envs must be a multiple of group"
        assert self.update_dtype in ("fp32", "bf16"), "update_dtype: fp32 | bf16"


# --------------------------------------------------------------------------- pure helpers


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    first: torch.Tensor,
    last_value: torch.Tensor,
    last_first: torch.Tensor,
    gamma: float,
    lam: float,
    trunc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(gamma, lambda) for a ``(T, N)`` rollout. ``first[t]`` = observation ``t`` starts a new
    episode (the previous step finished one), so the transition ``t - 1 -> t`` is terminal when
    ``first[t]``; ``last_value`` / ``last_first`` describe the observation after the last step.
    ``rewards[t]`` belongs to the transition ``t -> t + 1``. ``trunc[t]`` (optional): the
    transition ``t`` was cut by the step limit, not by the game: the trace is cut like at a
    terminal, but the target bootstraps with ``values[t]`` (the observation after the cut is
    unknown, see the module docs) instead of 0. Returns ``(advantages, returns)``."""
    steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(steps)):
        if t == steps - 1:
            nonterm = (~last_first).to(rewards.dtype)
            next_value = last_value
        else:
            nonterm = (~first[t + 1]).to(rewards.dtype)
            next_value = values[t + 1]
        boot = gamma * next_value * nonterm
        if trunc is not None:
            boot = boot + gamma * values[t] * trunc[t].to(rewards.dtype)
        delta = rewards[t] + boot - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def _rows(x: torch.Tensor, t0: int, t1: int, idx: torch.Tensor) -> torch.Tensor:
    """``x[t0:t1][:, idx]`` of a ``(T, N)`` tensor, flattened time-major (the minibatch order)."""
    return x[t0:t1][:, idx].reshape(-1)


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    if var < 1e-12:
        return float("nan")
    return float(1.0 - (target - pred).var() / var)


# --------------------------------------------------------------------------- profiling


class SectionTimer:
    """Accumulating wall-clock sections, cuda-synchronised at both ends (``--profile`` only; the
    synchronisation itself costs a little). Nested sections are all reported (so they overlap)."""

    def __init__(self, dev: torch.device):
        self.dev = dev
        self.prefix = ""
        self.t: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def __call__(self, name: str):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            self.t[self.prefix + name] += time.perf_counter() - t0

    def take(self) -> dict[str, float]:
        out = {k: round(v, 4) for k, v in self.t.items()}
        self.t.clear()
        return out


# --------------------------------------------------------------------------- logging


class RunLog:
    """JSONL log (one record per line) + a concise console line."""

    def __init__(self, out_dir: Path, echo=print):
        self.path = out_dir / "log.jsonl"
        self.echo = echo

    def write(self, rec: dict, line: str | None = None) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if line:
            self.echo(line)


# --------------------------------------------------------------------------- trainer


class Trainer:
    def __init__(self, cfg: PPOConfig, echo=print, device=None, resume: bool = False):
        self.cfg = cfg
        self.echo = echo
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.out, echo)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.curriculum = Curriculum(cfg.curriculum)

        self.update_idx = 0
        self.global_step = 0
        self.resumes = 0
        self.best = {"mean_ante": -1.0, "win_rate": -1.0, "update": 0}
        ck = None
        if resume and (self.out / "latest.pt").exists():
            self.model, ck = load_checkpoint(self.out / "latest.pt", self.dev)
            self.update_idx = ck["update"]
            self.global_step = ck["global_step"]
            self.resumes = ck.get("resumes", 0) + 1
            self.best = ck.get("best", self.best)
            self.echo(f"resumed {self.out / 'latest.pt'} at update {self.update_idx}")
        elif cfg.init_checkpoint:
            self.model, _ = load_checkpoint(cfg.init_checkpoint, self.dev)
            self.echo(f"initialised from {cfg.init_checkpoint}")
        else:
            self.model = TeamModel(ModelConfig(**cfg.model)).to(self.dev)
        self.ref = None
        if cfg.kl_coef > 0:
            assert cfg.init_checkpoint, "kl_coef > 0 needs init_checkpoint (the reference policy)"
            self.ref, _ = load_checkpoint(cfg.init_checkpoint, self.dev)
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        if ck is not None and "optimizer" in ck:
            self.opt.load_state_dict(ck["optimizer"])
        self.critic_params = {id(p) for p in self.model.critic.parameters()}
        self.timer = SectionTimer(self.dev) if cfg.profile else None
        self._p = self.timer if self.timer is not None else (lambda name: contextlib.nullcontext())
        if self.timer is not None:
            self.model._prof = self.timer
        self.compile_seconds = 0.0
        if cfg.compile:
            t0 = time.perf_counter()
            ok = self.model.encoder.enable_compile(cfg.compile_bucket)
            self.compile_seconds = time.perf_counter() - t0
            if ok:
                self.echo(f"torch.compile: encoder blocks compiled ({self.compile_seconds:.0f}s)")
            else:
                self.echo(
                    f"torch.compile FAILED, eager fallback: {self.model.encoder.compile_error}"
                )

        self.env = BalatroVecEnv(
            cfg.num_envs,
            seed=ev.train_seed(cfg.seed, self.resumes),
            num_threads=cfg.threads,
            max_episode_steps=cfg.max_episode_steps,
            reward=cfg.reward or None,
            endless=cfg.endless,
            endless_max_ante=cfg.endless_max_ante,
            **self.env_configs(cfg),
        )
        self.obs, self.masks = self.env.reset()
        n = cfg.num_envs
        self.first = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.hidden = self.model.init_hidden(n)
        self.ref_hidden = self.ref.init_hidden(n) if self.ref is not None else None
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed + 1000 * self.resumes)
        self.stage_idx, self.stage = self.curriculum.stage_at(self.global_step)
        self.route = self.stage.route_table()
        self._alloc()

    @staticmethod
    def env_configs(cfg: PPOConfig) -> dict:
        """``decks`` / ``stakes`` kwargs of the training env: none for the default Red / White (the
        fixed-configuration path), else the name lists to sample from."""
        decks = [x.strip() for x in cfg.decks.split(",") if x.strip()]
        stakes = [x.strip() for x in cfg.stakes.split(",") if x.strip()]
        assert decks and stakes, "decks / stakes must not be empty"
        if decks == ["red"] and stakes == ["white"]:
            return {}
        return {"decks": decks, "stakes": stakes}

    # ------------------------------------------------------------------ buffers

    def _alloc(self) -> None:
        cfg, dev = self.cfg, self.dev
        t, n = cfg.rollout_len, cfg.num_envs
        self.buf: dict[str, torch.Tensor] = {}
        for name in OBS_NAMES + MASK_NAMES:
            dtype, shape = _SPEC[name]
            if name.endswith("_valid") or name in ("kind_mask", "arg0_mask", "arg1_mask"):
                td = torch.bool
            elif name.endswith("_f"):
                td = torch.float32
            elif name.endswith("_i"):
                td = torch.uint8  # every vocabulary is < 256
            elif name in ("sel_candidates", "sel_required"):
                td = torch.int64
            else:
                td = torch.uint8  # sel_min / sel_max
            self.buf[name] = torch.zeros((t, n, *shape), dtype=td, device=dev)
        self.buf["actions"] = torch.zeros((t, n, 4), dtype=torch.int64, device=dev)
        for name in ("logp", "value", "reward"):
            self.buf[name] = torch.zeros((t, n), dtype=torch.float32, device=dev)
        self.buf["first"] = torch.zeros((t, n), dtype=torch.bool, device=dev)
        self.buf["trunc"] = torch.zeros((t, n), dtype=torch.bool, device=dev)
        self.buf["ctrl"] = torch.zeros((t, n), dtype=torch.bool, device=dev)
        self.buf["phase"] = torch.zeros((t, n), dtype=torch.long, device=dev)
        self.h0 = torch.zeros((t // cfg.chunk, n, self.model.mem_dim), device=dev)

    def _slice(self, t0: int, t1: int, idx: torch.Tensor):
        """Time-major flat minibatch ``(t1 - t0) * len(idx)`` rows of the rollout buffers."""
        n, length = idx.numel(), t1 - t0
        b = self.buf
        obs = {k: b[k][t0:t1][:, idx].reshape(length * n, *b[k].shape[2:]) for k in OBS_NAMES}
        masks = {k: b[k][t0:t1][:, idx].reshape(length * n, *b[k].shape[2:]) for k in MASK_NAMES}
        actions = b["actions"][t0:t1][:, idx].reshape(length * n, 4)
        return obs, masks, actions, b["first"][t0:t1][:, idx]

    # ------------------------------------------------------------------ rollout

    def collect(self) -> dict:
        cfg, dev, b = self.cfg, self.dev, self.buf
        model = self.model.eval()
        t_start = time.perf_counter()
        teacher_s, teacher_n, model_s = 0.0, 0, 0.0
        invalid = 0
        stage_all = bool(self.route[: len(PHASES)].all())
        p = self._p
        if self.timer is not None:
            self.timer.prefix = "rollout/"
        for t in range(cfg.rollout_len):
            if t % cfg.chunk == 0:
                self.h0[t // cfg.chunk] = self.hidden
            tm = time.perf_counter()
            with p("prepare_store"):  # H2D of the env buffers + copies into the rollout buffers
                o, m = prepare(self.obs, self.masks, dev)
                for k in OBS_NAMES:
                    b[k][t] = o[k]
                for k in MASK_NAMES:
                    b[k][t] = m[k]
                b["first"][t] = self.first
            with p("act"):
                actions, logp, _, value, self.hidden = model.act(
                    o, m, self.hidden, first=self.first, generator=self.gen
                )
            with p("d2h"):
                a_np = actions.cpu().numpy()
            model_s += time.perf_counter() - tm
            phase = self.obs["global_i"][:, 0]
            ctrl = self.route[phase]
            if not stage_all:
                use_teacher = ~ctrl
                if use_teacher.any():
                    tt = time.perf_counter()
                    ta = self.env.teacher_actions(use_teacher)
                    teacher_s += time.perf_counter() - tt
                    teacher_n += int(use_teacher.sum())
                    a_np[use_teacher] = ta[use_teacher]
                    actions = torch.from_numpy(a_np).to(dev)
            with p("store_misc"):  # small H2D copies of the step results
                b["actions"][t] = actions
                b["logp"][t] = logp
                b["value"][t] = value
                b["ctrl"][t] = torch.from_numpy(ctrl).to(dev)
                b["phase"][t] = torch.from_numpy(phase.astype(np.int64)).to(dev)
            with p("env_step"):
                _, _, reward, done, inv = self.env.step(a_np)
            invalid += int(inv.sum())
            with p("store_misc"):
                b["reward"][t] = torch.from_numpy(reward).to(dev)
                self.first = torch.from_numpy(done != 0).to(dev)
                b["trunc"][t] = torch.from_numpy(done == 3).to(dev)
        o, m = prepare(self.obs, self.masks, dev)
        self.last_value, _ = model.value(o, m, self.hidden, first=self.first)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        fin = self.env.drain_finished()
        return {
            "rollout_seconds": time.perf_counter() - t_start,
            "model_seconds": model_s,
            "teacher_seconds": teacher_s,
            "teacher_actions": teacher_n,
            "invalid_actions": invalid,
            "finished": fin,
        }

    def _amp(self):
        """Autocast context of the update / reference passes (``update_dtype``)."""
        if self.cfg.update_dtype == "bf16":
            return torch.autocast(self.dev.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _recompute_old_logp(self, env_perm: torch.Tensor, combos: list) -> torch.Tensor:
        """``--recompute-old-logp``: log-probs of the rollout actions under the *current* weights in
        the update dtype, over exactly the minibatches the first epoch will use (so the first
        minibatch has ratio 1 up to non-determinism, also with bf16)."""
        cfg = self.cfg
        out = torch.zeros_like(self.buf["logp"])
        self.model.train()
        for c, g in combos:
            idx = env_perm[g * cfg.group : (g + 1) * cfg.group]
            t0, t1 = c * cfg.chunk, (c + 1) * cfg.chunk
            obs, masks, actions, first = self._slice(t0, t1, idx)
            with self._amp():
                r = self.model.evaluate_full(obs, masks, self.h0[c][idx], actions, first)
            out[t0:t1, idx] = r["logp"].view(cfg.chunk, -1)
        return out

    @torch.no_grad()
    def _ref_logprobs(self) -> torch.Tensor:
        cfg = self.cfg
        t_len, n = cfg.rollout_len, cfg.num_envs
        out = torch.zeros(t_len, n, device=self.dev)
        for g0 in range(0, n, cfg.group):
            idx = torch.arange(g0, g0 + cfg.group, device=self.dev)
            h = self.ref_hidden[idx]
            for t0 in range(0, t_len, cfg.chunk):
                obs, masks, actions, first = self._slice(t0, t0 + cfg.chunk, idx)
                with self._amp():
                    r = self.ref.evaluate_full(obs, masks, h, actions, first)
                out[t0 : t0 + cfg.chunk, idx] = r["logp"].view(cfg.chunk, -1)
                h = r["hidden"].float()  # bf16 autocast: keep the carried state float32
            self.ref_hidden[idx] = h
        return out

    # ------------------------------------------------------------------ update

    def _lr(self) -> float:
        total = max(1, int(self.cfg.total_steps // (self.cfg.rollout_len * self.cfg.num_envs)))
        frac = min(1.0, self.update_idx / total)
        return self.cfg.lr * (1.0 - (1.0 - self.cfg.lr_final_frac) * frac)

    def _kl_coef(self) -> float:
        c = self.cfg
        if c.kl_coef <= 0:
            return 0.0
        return c.kl_coef * max(0.0, 1.0 - self.update_idx / max(1, c.kl_anneal_updates))

    def update(self) -> dict:
        cfg, dev, b = self.cfg, self.dev, self.buf
        t_start = time.perf_counter()
        model = self.model
        model.train()
        p = self._p
        if self.timer is not None:
            self.timer.prefix = "update/"
        with p("gae"):
            reward = b["reward"] * cfg.reward_scale
            adv, ret = compute_gae(
                reward,
                b["value"],
                b["first"],
                self.last_value,
                self.first,
                cfg.gamma,
                cfg.lam,
                b["trunc"] if cfg.bootstrap_truncation else None,
            )
            ctrl = b["ctrl"]
            if cfg.norm_adv and ctrl.any():
                a = adv[ctrl]
                adv = (adv - a.mean()) / (a.std() + 1e-8)
        beta = self._kl_coef()
        with p("kl_reference_pass"):
            ref_logp = self._ref_logprobs() if self.ref is not None and beta > 0 else None
        warmup = self.update_idx < cfg.critic_warmup
        lr = self._lr()
        for g in self.opt.param_groups:
            g["lr"] = lr

        n_chunks, n_groups = cfg.rollout_len // cfg.chunk, cfg.num_envs // cfg.group
        acc = {
            k: torch.zeros((), device=dev)
            for k in (
                "pg",
                "vloss",
                "ent",
                "kl",
                "approx_kl",
                "clipfrac",
                "ref_kl",
                "gnorm",
                "rows",
            )
        }
        ent_part = torch.zeros(len(HEAD_NAMES), len(PARTS), device=dev)
        ent_cnt = torch.zeros(len(HEAD_NAMES), device=dev)
        n_mb = 0
        stop = False
        old_buf = b["logp"]
        pending = None  # (env_perm, order) of epoch 0, drawn early for --recompute-old-logp
        if cfg.recompute_old_logp:
            env_perm = torch.randperm(cfg.num_envs, device=dev)
            combos = [(c, g) for c in range(n_chunks) for g in range(n_groups)]
            order = torch.randperm(len(combos)).tolist()
            pending = (env_perm, order)
            with p("recompute_old_logp"):
                old_buf = self._recompute_old_logp(env_perm, combos)
        for epoch in range(cfg.epochs):  # noqa: B007
            if pending is not None:
                (env_perm, order), pending = pending, None
            else:
                env_perm = torch.randperm(cfg.num_envs, device=dev)
                order = None
            combos = [(c, g) for c in range(n_chunks) for g in range(n_groups)]
            if order is None:
                order = torch.randperm(len(combos)).tolist()
            ep_kl = torch.zeros((), device=dev)
            for ci in order:
                c, g = combos[ci]
                idx = env_perm[g * cfg.group : (g + 1) * cfg.group]
                t0, t1 = c * cfg.chunk, (c + 1) * cfg.chunk
                with p("slice"):
                    obs, masks, actions, first = self._slice(t0, t1, idx)
                with p("forward"), self._amp():
                    r = model.evaluate_full(obs, masks, self.h0[c][idx], actions, first)
                new_logp, value = r["logp"], r["value"]
                old_logp, mb_adv, mb_ret, mb_val = (
                    _rows(x, t0, t1, idx) for x in (old_buf, adv, ret, b["value"])
                )
                with p("loss"):
                    w = (_rows(ctrl, t0, t1, idx) & r["ok"]).float()
                    wsum = w.sum().clamp(min=1.0)
                    on = w > 0  # rows the policy loss sees (garbage-proof: where, not multiply)
                    new_logp = torch.where(on, new_logp, old_logp)
                    log_ratio = new_logp - old_logp
                    ratio = log_ratio.clamp(-20.0, 20.0).exp()
                    pg = torch.max(
                        -mb_adv * ratio, -mb_adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                    )
                    pg_loss = (pg * w).sum() / wsum
                    if cfg.clip_vloss:
                        v_clip = mb_val + (value - mb_val).clamp(-cfg.clip, cfg.clip)
                        v_loss = (
                            0.5 * torch.max((value - mb_ret) ** 2, (v_clip - mb_ret) ** 2).mean()
                        )
                    else:
                        v_loss = 0.5 * ((value - mb_ret) ** 2).mean()
                    ent = (torch.where(on, r["entropy"], 0.0)).sum() / wsum
                    loss = cfg.vf_coef * v_loss
                    kl = torch.zeros((), device=dev)
                    if not warmup:
                        loss = loss + pg_loss - cfg.ent_coef * ent
                        if ref_logp is not None and beta > 0:
                            d = torch.where(on, _rows(ref_logp, t0, t1, idx) - new_logp, 0.0)
                            kl = ((d.exp() - 1 - d) * w).sum() / wsum
                            loss = loss + beta * kl
                self.opt.zero_grad(set_to_none=True)
                with p("backward"):
                    loss.backward()
                with p("optimizer"):
                    if warmup:  # critic-only: the backbone / heads receive no update
                        for prm in model.parameters():
                            if id(prm) not in self.critic_params:
                                prm.grad = None
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [q for q in model.parameters() if q.grad is not None], cfg.max_grad_norm
                    )
                    self.opt.step()
                with p("stats"), torch.no_grad():
                    approx_kl = (((ratio - 1) - log_ratio) * w).sum() / wsum
                    clipfrac = (((ratio - 1).abs() > cfg.clip).float() * w).sum() / wsum
                    acc["pg"] += pg_loss.detach()
                    acc["vloss"] += v_loss.detach()
                    acc["ent"] += ent.detach()
                    acc["kl"] += kl.detach()
                    acc["approx_kl"] += approx_kl
                    acc["clipfrac"] += clipfrac
                    acc["gnorm"] += gnorm
                    acc["rows"] += w.sum()
                    ep_kl += approx_kl
                    head = model.phase_head[r["phase"]]
                    for h in range(len(HEAD_NAMES)):
                        wh = w * (head == h)
                        ent_cnt[h] += wh.sum()
                        for pi, part in enumerate(PARTS):
                            ent_part[h, pi] += torch.where(wh > 0, r[f"ent_{part}"], 0.0).sum()
                    acc["ref_kl"] += kl.detach()
                n_mb += 1
            if cfg.target_kl is not None and float(ep_kl) / len(order) > cfg.target_kl:
                stop = True
                break
        stats = {k: float(v) / n_mb for k, v in acc.items() if k != "rows"}
        stats["policy_rows_per_mb"] = float(acc["rows"]) / n_mb
        stats["minibatches"] = n_mb
        stats["epochs_run"] = epoch + 1
        stats["early_stop"] = stop
        stats["lr"] = lr
        stats["kl_coef"] = beta
        stats["critic_warmup"] = warmup
        stats["explained_var"] = explained_variance(b["value"].flatten(), ret.flatten())
        stats["adv_abs"] = float(adv[ctrl].abs().mean()) if ctrl.any() else 0.0
        ent_cnt_c = ent_cnt.clamp(min=1.0)
        for h, hn in enumerate(HEAD_NAMES):
            stats[f"rows/{hn}"] = float(ent_cnt[h]) / n_mb
            for pi, part in enumerate(PARTS):
                stats[f"ent/{hn}/{part}"] = float(ent_part[h, pi] / ent_cnt_c[h])
        stats["update_seconds"] = time.perf_counter() - t_start
        if self.timer is not None:
            stats["prof"] = self.timer.take()  # rollout/* (from collect) and update/* sections
        if self.dev.type == "cuda":
            stats["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        return stats

    # ------------------------------------------------------------------ rollout statistics

    def action_stats(self) -> dict:
        """Kind distribution per head over the model-controlled rows of the last rollout."""
        b = self.buf
        sel = b["ctrl"]
        kinds = b["actions"][..., 0][sel]
        heads = self.model.phase_head[b["phase"][sel]]
        out = {}
        for h, hn in enumerate(HEAD_NAMES):
            k = kinds[heads == h]
            if k.numel() == 0:
                continue
            frac = torch.bincount(k, minlength=KINDS).float() / k.numel()
            out[hn] = {KIND_NAMES[i]: round(float(frac[i]), 4) for i in range(KINDS) if frac[i] > 0}
        return out

    @staticmethod
    def episode_stats(fin: list) -> dict:
        if not fin:
            return {"episodes": 0}
        ante = np.array([r[1] for r in fin], dtype=float)
        return {
            "episodes": len(fin),
            "ep_ante": float(ante.mean()),
            "ep_win": float(np.mean([r[2] == 2 for r in fin])),
            "ep_trunc": float(np.mean([r[2] == 3 for r in fin])),
            "ep_steps": float(np.mean([r[3] for r in fin])),
            "ep_return": float(np.mean([r[4] for r in fin])),
            "ep_blinds": float(np.mean([r[5] for r in fin])),
            "ep_invalid": float(np.mean([r[6] for r in fin])),
        }

    # ------------------------------------------------------------------ checkpoints / eval

    def save(self, name: str = "latest.pt", **extra) -> None:
        save_checkpoint(
            self.out / name,
            self.model,
            ppo_config=dataclasses.asdict(self.cfg),
            optimizer=self.opt.state_dict(),
            update=self.update_idx,
            global_step=self.global_step,
            resumes=self.resumes,
            best=self.best,
            **extra,
        )

    def evaluate(self) -> dict:
        """Periodic validation (a prefix of the fixed test set); keeps ``best.pt`` (deterministic
        mean ante, ties by win rate) and ``latest`` untouched."""
        cfg = self.cfg
        route = None if self.route[: len(PHASES)].all() else self.stage.label
        t0 = time.perf_counter()
        det = ev.evaluate_model(
            self.model,
            cfg.eval_games,
            True,
            route,
            max_episode_steps=cfg.max_episode_steps,
            move_limit=cfg.eval_move_limit,
            seed=ev.EVAL_SETS[cfg.eval_set][0],
            deck=cfg.eval_deck,
            stake=cfg.eval_stake,
        )
        rec = {"det_" + k: v for k, v in det.summary().items() if k not in ("name", "ante_hist")}
        rec["det_ante_hist"] = det.ante_hist
        if cfg.eval_sampled:
            sam = ev.evaluate_model(
                self.model,
                cfg.eval_games,
                False,
                route,
                max_episode_steps=cfg.max_episode_steps,
                move_limit=cfg.eval_move_limit,
                seed=ev.EVAL_SETS[cfg.eval_set][0],
                deck=cfg.eval_deck,
                stake=cfg.eval_stake,
            )
            rec.update(
                {"sam_" + k: v for k, v in sam.summary().items() if k not in ("name", "ante_hist")}
            )
        rec["eval_seconds"] = time.perf_counter() - t0
        score = (det.mean_ante, det.win_rate)
        if cfg.ckpt_keep_best and score > (self.best["mean_ante"], self.best["win_rate"]):
            self.best = {
                "mean_ante": det.mean_ante,
                "win_rate": det.win_rate,
                "update": self.update_idx,
            }
            self.save("best.pt")
            rec["new_best"] = True
        return rec

    # ------------------------------------------------------------------ main loop

    def train(self) -> None:
        cfg = self.cfg
        self.echo(
            f"PPO {cfg.name}: {self.model.num_parameters()['total']:,} params, "
            f"{cfg.num_envs} envs x {cfg.rollout_len} steps = "
            f"{cfg.num_envs * cfg.rollout_len:,} rows/update, curriculum {self.curriculum}, "
            f"device {self.dev}"
        )
        (self.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        rows = cfg.num_envs * cfg.rollout_len
        total_updates = int(cfg.total_steps // rows)
        t_run = time.perf_counter()
        steps_at_start = self.global_step
        while self.update_idx < total_updates:
            idx, stage = self.curriculum.stage_at(self.global_step)
            if idx != self.stage_idx or self.update_idx == 0:
                self.stage_idx, self.stage = idx, stage
                self.route = stage.route_table()
                self.echo(f"-- curriculum stage {idx}: model plays {stage.label}")
            col = self.collect()
            self.global_step += rows
            st = self.update()
            self.update_idx += 1
            fin = col.pop("finished")
            roll_s, upd_s = col["rollout_seconds"], st["update_seconds"]
            rec = {
                "type": "update",
                "update": self.update_idx,
                "global_step": self.global_step,
                "stage": self.stage_idx,
                "stage_label": self.stage.label,
                "wall_seconds": time.perf_counter() - t_run,
                "steps_per_sec": rows / (roll_s + upd_s),
                "rollout_steps_per_sec": rows / roll_s,
                "mean_reward_step": float(self.buf["reward"].mean()),
                **col,
                **st,
                **self.episode_stats(fin),
                "actions": self.action_stats(),
            }
            line = (
                f"u{self.update_idx:>4} step={self.global_step / 1e6:6.2f}M "
                f"{rows / (roll_s + upd_s):6.0f} st/s (roll {roll_s:.1f}s upd {upd_s:.1f}s"
                + (f" teach {col['teacher_seconds']:.1f}s" if col["teacher_actions"] else "")
                + f") ep={rec.get('episodes', 0)} ante={rec.get('ep_ante', float('nan')):.2f} "
                f"win={100 * rec.get('ep_win', float('nan')):.1f}% "
                f"blinds={rec.get('ep_blinds', float('nan')):.1f} "
                f"kl={st['approx_kl']:.4f} clip={st['clipfrac']:.3f} ev={st['explained_var']:.2f} "
                f"ent={st['ent']:.2f} vl={st['vloss']:.4f}"
                + (f" klref={st['ref_kl']:.3f}" if self.ref is not None else "")
                + (" [critic warmup]" if st["critic_warmup"] else "")
            )
            self.log.write(rec, line)
            if cfg.eval_every and self.update_idx % cfg.eval_every == 0:
                er = self.evaluate()
                er.update(type="eval", update=self.update_idx, global_step=self.global_step)
                self.log.write(
                    er,
                    f"   eval@{self.update_idx}: det ante={er['det_mean_ante']:.2f} "
                    f"win={100 * er['det_win_rate']:.1f}% blinds={er['det_mean_blinds']:.1f}"
                    + (
                        f" | sampled ante={er['sam_mean_ante']:.2f} "
                        f"win={100 * er['sam_win_rate']:.1f}%"
                        if "sam_mean_ante" in er
                        else ""
                    )
                    + (" *best*" if er.get("new_best") else "")
                    + f" ({er['eval_seconds']:.0f}s)",
                )
            if self.update_idx % cfg.save_every == 0:
                self.save()
        self.save()
        self.echo(
            f"done: {self.global_step - steps_at_start:,} env steps in "
            f"{time.perf_counter() - t_run:.0f}s"
        )
