"""Phase heads: ``kind`` -> ``arg0`` (pointer) -> ``arg1`` (pointer, MoveJoker) -> cards
(autoregressive pointer in canonical ascending order + STOP).

One ``Head`` instance per phase (Blind / Play / Shop / Pack); each is a complete head over all 17
kinds, because the masks (not the head) decide which kinds exist in a phase (consumables and
joker moves are legal in every phase). The same code path samples (``actions=None``) and scores
teacher-forced ``actions`` (PPO ``evaluate``, behaviour cloning).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from balatro_rl.env import LAYOUT
from balatro_rl.model.cards import canonical_step, pack, unpack
from balatro_rl.select import (
    KIND_DISCARD,
    KIND_MOVE_JOKER,
    KIND_PICK_PACK_ITEM,
    KIND_PLAY,
    KIND_USE_CONSUMABLE,
)

CAPS = LAYOUT["caps"]
KINDS = CAPS["kinds"]
MAX_ARG = CAPS["max_arg"]
CONS_CAP = CAPS["cons"]
JOKER_CAP = CAPS["joker"]
OFFER_CAP = CAPS["offer"]
PACK_ITEM_CAP = CAPS["pack_items"]
MAX_CARD_STEPS = CAPS["max_play_cards"]
PTR_TOKENS = JOKER_CAP + CONS_CAP + OFFER_CAP  # joker | consumable | offer
_ARRAY_OFFSET = {"joker": 0, "cons": JOKER_CAP, "offer": JOKER_CAP + CONS_CAP}


def _pointer_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """``index (17, 20)``: pointer-token row of ``(kind, arg0)`` in the joker|cons|offer concat;
    ``has (17,)``: the kind's ``arg0`` is a pointer at all."""
    index = torch.zeros(KINDS, MAX_ARG, dtype=torch.long)
    has = torch.zeros(KINDS, dtype=torch.bool)
    for kind, spec in enumerate(LAYOUT["arg0_pointer"]):
        if spec is None:
            continue
        array, row0, count = spec
        has[kind] = True
        index[kind, :count] = _ARRAY_OFFSET[array] + row0 + torch.arange(count)
    return index, has


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def choose(
    logits: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor | None = None,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
):
    """Masked categorical step. Returns ``(idx, logp, entropy, allowed, hit)``.

    ``allowed``: the chosen/target index is inside the mask; ``hit``: argmax equals the index.
    A row whose mask is empty is given index 0 (callers pass only rows with a choice)."""
    mask = mask.clone()
    mask[:, 0] |= ~mask.any(-1)
    logits = logits.float().masked_fill(~mask, float("-inf"))
    logp_all = torch.log_softmax(logits, -1)
    greedy = logits.argmax(-1)
    if target is not None:
        idx = target
    elif deterministic:
        idx = greedy
    else:
        u = torch.rand(logits.shape, device=logits.device, generator=generator)
        u = u.clamp(1e-12, 1 - 1e-7)
        idx = (logits - torch.log(-torch.log(u))).argmax(-1)
    allowed = mask.gather(-1, idx[:, None]).squeeze(-1)
    logp = torch.where(
        allowed, logp_all.gather(-1, idx[:, None]).squeeze(-1), torch.zeros_like(allowed).float()
    )
    ent = -(logp_all.exp() * logp_all.masked_fill(~mask, 0.0)).sum(-1)
    return idx, logp, ent, allowed, greedy == idx


class Head(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.scale = 1.0 / math.sqrt(d)
        self.kind_mlp = _mlp(d, d, KINDS)
        self.kind_emb = nn.Embedding(KINDS, d)
        self.q0 = _mlp(d, d, d)
        self.k0 = nn.Linear(d, d)
        self.arg0_in = nn.Linear(d, d, bias=False)
        self.q1 = _mlp(d, d, d)
        self.k1 = nn.Linear(d, d)
        self.step_emb = nn.Embedding(MAX_CARD_STEPS + 1, d)
        self.sel_in = nn.Linear(d, d, bias=False)
        self.card_state = _mlp(d, d, d)
        self.card_q = nn.Linear(d, d)
        self.card_k = nn.Linear(d, d)
        self.stop = nn.Linear(d, 1)
        ptr_index, ptr_has = _pointer_tables()
        self.register_buffer("ptr_index", ptr_index, persistent=False)
        self.register_buffer("ptr_has", ptr_has, persistent=False)

    def forward(
        self,
        ctx: torch.Tensor,
        ptr: torch.Tensor,
        hand: torch.Tensor,
        m: dict[str, torch.Tensor],
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """``ctx (n, d)``, ``ptr (n, 49, d)``, ``hand (n, nh, d)``, masks ``m`` (prepared). Returns
        a dict with ``actions (n, 4)``, ``logp``, ``entropy`` (chain sums), and the per-part
        ``lp_*`` / ``ent_*`` / ``hit_*`` / ``ok`` used by behaviour cloning and PPO."""
        n, nh, _ = hand.shape
        dev = ctx.device
        ar = torch.arange(n, device=dev)
        tgt = (lambda i: actions[:, i]) if actions is not None else (lambda i: None)

        # --- kind
        kind_logits = self.kind_mlp(ctx)
        kind, lp_kind, ent_kind, ok_kind, hit_kind = choose(
            kind_logits, m["kind_mask"], tgt(0), deterministic, generator
        )
        z0 = ctx + self.kind_emb(kind)

        # --- arg0 (pointer; kinds without a pointer have the single option 0)
        keys0 = self.k0(ptr)
        s_all = torch.einsum("nd,nld->nl", self.q0(z0), keys0) * self.scale
        table = self.ptr_index[kind]  # (n, 20) -> token row
        has_ptr = self.ptr_has[kind]
        s0 = s_all.gather(1, table)
        s0 = torch.where(has_ptr[:, None], s0, torch.zeros_like(s0))
        mask0 = m["arg0_mask"][ar, kind]
        arg0, lp_a0, ent_a0, ok_a0, hit_a0 = choose(s0, mask0, tgt(1), deterministic, generator)
        tok = ptr[ar, table.gather(1, arg0[:, None]).squeeze(1)]
        tok = tok * has_ptr[:, None]
        z1 = z0 + self.arg0_in(tok)
        n_opt0 = mask0.sum(-1)

        # --- arg1 (MoveJoker destination): any joker slot except arg0
        keys1 = self.k1(ptr[:, :JOKER_CAP])
        s1 = torch.einsum("nd,nld->nl", self.q1(z1), keys1) * self.scale
        is_move = kind == KIND_MOVE_JOKER
        mask1 = (
            m["arg1_mask"] & ~nn.functional.one_hot(arg0.clamp(max=JOKER_CAP - 1), JOKER_CAP).bool()
        )
        only0 = torch.zeros_like(mask1)
        only0[:, 0] = True
        mask1 = torch.where(is_move[:, None], mask1, only0)
        arg1, lp_a1, ent_a1, ok_a1, hit_a1 = choose(s1, mask1, tgt(2), deterministic, generator)
        n_opt1 = mask1.sum(-1)

        # --- cards
        row = torch.full_like(kind, -1)
        row = torch.where((kind == KIND_PLAY), 0, row)
        row = torch.where((kind == KIND_DISCARD), 1, row)
        row = torch.where((kind == KIND_USE_CONSUMABLE) & (arg0 < CONS_CAP), 2 + arg0, row)
        row = torch.where(
            (kind == KIND_PICK_PACK_ITEM) & (arg0 < PACK_ITEM_CAP), 2 + CONS_CAP + arg0, row
        )
        rc = row.clamp(min=0)
        mx = m["sel_max"][ar, rc]
        takes = (row >= 0) & (mx > 0)
        cand = unpack(m["sel_candidates"][ar, rc], nh)
        req = unpack(m["sel_required"][ar, rc], nh)
        mn = m["sel_min"][ar, rc]
        cards, lp_c, ent_c, ok_c, hit_c = self._cards(
            z1, hand, takes, cand, req, mn, mx, actions, deterministic, generator
        )
        if actions is not None:
            ok_c = ok_c & (takes | (actions[:, 3] == 0))

        out_actions = torch.stack([kind, arg0, torch.where(is_move, arg1, 0), cards], dim=1)
        ok = ok_kind & ok_a0 & (ok_a1 | ~is_move) & ok_c
        return {
            "actions": out_actions,
            "logp": lp_kind + lp_a0 + lp_a1 + lp_c,
            "entropy": ent_kind + ent_a0 + ent_a1 + ent_c,
            "lp_kind": lp_kind,
            "lp_arg0": lp_a0,
            "lp_arg1": lp_a1,
            "lp_cards": lp_c,
            "ent_kind": ent_kind,
            "ent_arg0": ent_a0,
            "ent_arg1": ent_a1,
            "ent_cards": ent_c,
            "hit_kind": hit_kind,
            "hit_arg0": hit_a0,
            "hit_arg1": hit_a1,
            "hit_cards": hit_c,
            "has_arg0": n_opt0 > 1,
            "has_arg1": is_move & (n_opt1 > 1),
            "has_cards": takes,
            "ok": ok,
        }

    def _cards(self, z, hand, takes, cand, req, mn, mx, actions, deterministic, generator):
        n, nh, _ = hand.shape
        dev = z.device
        keys = self.card_k(hand)
        partial = torch.zeros(n, nh, dtype=torch.bool, device=dev)
        last = torch.full((n,), -1, dtype=torch.long, device=dev)
        done = ~takes
        lp = torch.zeros(n, device=dev)
        ent = torch.zeros(n, device=dev)
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        hit = torch.ones(n, dtype=torch.bool, device=dev)
        if actions is not None:
            tbits = unpack(actions[:, 3], nh) & takes[:, None]
            rank = tbits.long().cumsum(1) - 1
            ok = ~((actions[:, 3] >> nh) != 0)  # bits beyond the hand slice are impossible
        steps = int(mx[takes].max()) if bool(takes.any()) else 0
        for t in range(steps + 1):
            nxt, stop = canonical_step(cand, req, mn, mx, partial, last)
            sel = torch.einsum("nh,nhd->nd", partial.to(hand.dtype), hand)
            s = self.card_state(z + self.step_emb.weight[t] + self.sel_in(sel))
            sc = torch.einsum("nd,nhd->nh", self.card_q(s), keys) * self.scale
            logits = torch.cat([sc, self.stop(s)], dim=1)
            mask = torch.cat([nxt, stop[:, None]], dim=1)
            active = ~done
            stop_only = torch.zeros_like(mask)
            stop_only[:, nh] = True
            mask = torch.where(active[:, None], mask, stop_only)
            if actions is None:
                tgt = None
            else:
                m_t = tbits & (rank == t)
                tgt = torch.where(m_t.any(1), m_t.float().argmax(1), torch.full_like(last, nh))
            idx, lp_t, ent_t, ok_t, hit_t = choose(logits, mask, tgt, deterministic, generator)
            a = active.float()
            lp = lp + lp_t * a
            ent = ent + ent_t * a
            ok = ok & (ok_t | ~active)
            hit = hit & (hit_t | ~active)
            is_stop = idx == nh
            newbit = (
                torch.nn.functional.one_hot(idx.clamp(max=nh - 1), nh).bool()
                & (active & ~is_stop)[:, None]
            )
            partial = partial | newbit
            last = torch.where(newbit.any(1), idx, last)
            done = done | (active & is_stop)
        if actions is not None:
            # Anything left unfinished after the loop would mean the target has too many cards.
            ok = ok & done
        return pack(partial), lp, ent, ok, hit
