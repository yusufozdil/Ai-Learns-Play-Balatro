"""Token encoder: per-type feature projection + categorical embeddings + type/position
embeddings, then a small pre-norm Transformer with padding masks.

Token order: ``[global (1), hand (nh), joker (nj), consumable (nc), offer (no)]``; every group is
sliced to the largest *valid* row of the batch (``nh`` = highest valid hand position + 1 ...).
The group outputs are returned padded back to the full capacities (jokers 20, consumables 8,
offers 21) so the pointer tables of the heads have fixed indices; the hand output keeps its
sliced width ``nh`` (cards are addressed by bit position, ``nh <= 32``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from balatro_rl.env import LAYOUT

CAPS = LAYOUT["caps"]
HAND_CAP, JOKER_CAP, CONS_CAP, OFFER_CAP = CAPS["hand"], CAPS["joker"], CAPS["cons"], CAPS["offer"]
VOCAB = LAYOUT["vocab"]

# Which embedding table each categorical column uses (columns of ``*_i`` in order). Columns with
# the same vocabulary meaning share a table: ranks/suits/enhancements/seals/editions of hand and
# offer cards, the item vocabulary of jokers/consumables/offers, hand types (last played hand,
# boss hand, tag hand, To Do List's hand).
_COLUMN_TABLES: dict[str, tuple[str, ...]] = {
    "global_i": (
        "phase",
        "blind",
        "boss",
        "handtype",
        "handtype",
        "open_pack",
        "tag",
        "handtype",
        "deck",
        "stake",
    ),
    "hand_i": ("rank", "suit", "enh", "edition", "seal"),
    "joker_i": ("item", "edition", "aux_suit", "aux_rank", "handtype"),
    "cons_i": ("item",),
    "offer_i": ("source", "item", "edition", "rank", "suit", "enh", "seal"),
}


# Tables added by interface version 2 (deck / stake of the run). They are laid out *after* every
# older table, so a v1 checkpoint migrates by appending zero rows (``model.migrate``).
NEW_TABLES_V2 = ("deck", "stake")


def _table_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for group, tables in _COLUMN_TABLES.items():
        assert len(tables) == len(VOCAB[group]), group
        for table, size in zip(tables, VOCAB[group], strict=True):
            assert sizes.setdefault(table, size) == size, f"table {table}: vocab mismatch"
    old = {t: n for t, n in sizes.items() if t not in NEW_TABLES_V2}
    return {**old, **{t: sizes[t] for t in NEW_TABLES_V2}}


def table_offsets(sizes: dict[str, int] | None = None) -> dict[str, int]:
    """First row of each table inside the shared embedding table (in ``sizes`` order)."""
    offsets, total = {}, 0
    for name, size in (sizes or TABLE_SIZES).items():
        offsets[name] = total
        total += size
    return offsets


TABLE_SIZES = _table_sizes()

# Interface v1 had 203 ``global_f`` and 8 ``global_i`` columns; v2 appended to both. The global
# token evaluates the old and the new columns as *separate* matmuls / sums that are then added, so a
# migrated v1 checkpoint (zero weights on the new columns) reproduces the v1 float32 result bit
# for bit: one wider GEMM would round differently (different blocking) and flip a few near-tied
# decisions. Mathematically it is the same single Linear / embedding sum.
V1_GLOBAL_F = 203
V1_GLOBAL_I = 8
# Interface v3 appended 6 ``global_f`` columns after v2's 214 and 4 ``joker_f`` columns after the
# 13 of v1/v2; they get their own matmul as well (zero weights after a v2 -> v3 migration add an
# exact 0 to the v2 result).
V2_GLOBAL_F = 214
V2_JOKER_F = 13
TOKEN_TYPES = ("global", "hand", "joker", "cons", "offer")


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, length, 3, self.heads, d // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, length, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class Encoder(nn.Module):
    def __init__(
        self, d: int = 128, layers: int = 4, heads: int = 4, ff_mult: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.d = d
        offsets = table_offsets()
        total = sum(TABLE_SIZES.values())
        self.table = nn.Embedding(total, d)  # one table; each column adds its offset
        for group, tables in _COLUMN_TABLES.items():
            offs = torch.tensor([offsets[t] for t in tables], dtype=torch.long)
            self.register_buffer(f"off_{group}", offs, persistent=False)
        names = LAYOUT["feature_names"]
        self.global_proj = nn.Sequential(
            nn.Linear(len(names["global_f"]), d), nn.GELU(), nn.Linear(d, d)
        )
        self.hand_proj = nn.Linear(len(names["hand_f"]), d)
        self.joker_proj = nn.Linear(len(names["joker_f"]), d)
        self.cons_proj = nn.Linear(len(names["cons_f"]), d)
        self.offer_proj = nn.Linear(len(names["offer_f"]), d)
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), d)
        self.pos_hand = nn.Embedding(HAND_CAP, d)
        self.pos_joker = nn.Embedding(JOKER_CAP, d)
        self.pos_cons = nn.Embedding(CONS_CAP, d)
        self.pos_offer = nn.Embedding(OFFER_CAP, d)
        self.in_ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d, heads, ff_mult, dropout) for _ in range(layers)])
        self.out_ln = nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)
        # Optional ``torch.compile`` of the Transformer stack (``enable_compile``); off by default.
        self._blocks_fn = None
        self._bucket = 0

    def _run_blocks(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask)
        return x

    def enable_compile(self, bucket: int = 8, mode: str | None = None, dynamic: bool | None = True):
        """Compile the Transformer stack with ``torch.compile``. The token count varies per batch
        (largest valid group sizes); it is padded up to a multiple of ``bucket`` with masked tokens
        (padding keys are masked, so valid tokens are unaffected up to float rounding) to bound the
        number of distinct shapes. Returns ``True`` when a forward + backward warm-up compiled;
        on any failure the eager path is kept and ``False`` is returned."""
        try:
            import torch._dynamo as dynamo

            dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)
            fn = torch.compile(self._run_blocks, mode=mode, dynamic=dynamic)
            dev = self.table.weight.device
            n = 64
            length = max(bucket, 1) * 4
            x = torch.randn(n, length, self.d, device=dev, requires_grad=True)
            mask = torch.ones(n, 1, 1, length, dtype=torch.bool, device=dev)
            fn(x, mask).sum().backward()  # compiles forward and backward
            with torch.no_grad():
                fn(x, mask)
            self.zero_grad(set_to_none=True)
        except Exception as e:  # noqa: BLE001 - any compiler / driver failure: stay eager
            self._blocks_fn, self._bucket = None, 0
            self.compile_error = repr(e)
            return False
        self._blocks_fn, self._bucket = fn, bucket
        return True

    def _cat(self, group: str, x: torch.Tensor, split: int | None = None) -> torch.Tensor:
        """Sum of the categorical embeddings of the columns of ``x (..., ncols)``; with ``split``
        the first ``split`` columns and the rest are summed separately, then added."""
        e = self.table(x + getattr(self, f"off_{group}"))
        if split is None:
            return e.sum(-2)
        return e[..., :split, :].sum(-2) + e[..., split:, :].sum(-2)

    def _global_proj(self, x: torch.Tensor) -> torch.Tensor:
        first, act, last = self.global_proj
        w = first.weight
        lead = F.linear(x[:, :V1_GLOBAL_F], w[:, :V1_GLOBAL_F].contiguous(), first.bias)
        h = lead + x[:, V1_GLOBAL_F:V2_GLOBAL_F] @ w[:, V1_GLOBAL_F:V2_GLOBAL_F].t()
        h = h + x[:, V2_GLOBAL_F:] @ w[:, V2_GLOBAL_F:].t()
        return last(act(h))

    def _joker_proj(self, x: torch.Tensor) -> torch.Tensor:
        """``joker_proj(x)`` evaluated as the v2 columns plus the v3 contribution columns."""
        lin = self.joker_proj
        w = lin.weight
        lead = F.linear(x[..., :V2_JOKER_F].contiguous(), w[:, :V2_JOKER_F].contiguous(), lin.bias)
        return lead + x[..., V2_JOKER_F:] @ w[:, V2_JOKER_F:].t()

    @staticmethod
    def _lengths(obs: dict[str, torch.Tensor]) -> list[int]:
        """Largest valid row + 1 of each group over the batch (one device sync)."""
        out = []
        for name, cap in (
            ("hand_valid", HAND_CAP),
            ("joker_valid", JOKER_CAP),
            ("cons_valid", CONS_CAP),
            ("offer_valid", OFFER_CAP),
        ):
            v = obs[name]
            ar = torch.arange(1, cap + 1, device=v.device)
            out.append((v * ar).max())
        return torch.stack(out).tolist()

    def forward(self, obs: dict[str, torch.Tensor]):
        """Returns ``(summary (n, d), hand (n, nh, d), ptr (n, 49, d))`` where ``ptr`` is the
        concatenation joker (20) | consumable (8) | offer (21) padded to capacity."""
        nh, nj, nc, no = self._lengths(obs)
        nh = max(nh, 1)
        n = obs["global_f"].shape[0]
        dev = obs["global_f"].device
        te = self.type_emb.weight

        g = self._global_proj(obs["global_f"]) + self._cat("global_i", obs["global_i"], V1_GLOBAL_I)
        g = g.unsqueeze(1) + te[0]
        h = (
            self.hand_proj(obs["hand_f"][:, :nh])
            + self._cat("hand_i", obs["hand_i"][:, :nh])
            + self.pos_hand.weight[:nh]
            + te[1]
        )
        j = (
            self._joker_proj(obs["joker_f"][:, :nj])
            + self._cat("joker_i", obs["joker_i"][:, :nj])
            + self.pos_joker.weight[:nj]
            + te[2]
        )
        c = (
            self.cons_proj(obs["cons_f"][:, :nc])
            + self._cat("cons_i", obs["cons_i"][:, :nc])
            + self.pos_cons.weight[:nc]
            + te[3]
        )
        o = (
            self.offer_proj(obs["offer_f"][:, :no])
            + self._cat("offer_i", obs["offer_i"][:, :no])
            + self.pos_offer.weight[:no]
            + te[4]
        )
        x = self.in_ln(torch.cat([g, h, j, c, o], dim=1))
        valid = torch.cat(
            [
                torch.ones(n, 1, dtype=torch.bool, device=dev),
                obs["hand_valid"][:, :nh],
                obs["joker_valid"][:, :nj],
                obs["cons_valid"][:, :nc],
                obs["offer_valid"][:, :no],
            ],
            dim=1,
        )
        if self._blocks_fn is None:
            attn_mask = valid[:, None, None, :]
            for block in self.blocks:
                x = block(x, attn_mask)
        else:
            length = x.shape[1]
            pad = -length % self._bucket if self._bucket > 1 else 0
            if pad:
                x = F.pad(x, (0, 0, 0, pad))
                valid = F.pad(valid, (0, pad))
            x = self._blocks_fn(x, valid[:, None, None, :])
            if pad:
                x = x[:, :length]
        x = self.out_ln(x)
        summary = x[:, 0]
        a = 1
        hand = x[:, a : a + nh]
        a += nh
        joker = F.pad(x[:, a : a + nj], (0, 0, 0, JOKER_CAP - nj))
        a += nj
        cons = F.pad(x[:, a : a + nc], (0, 0, 0, CONS_CAP - nc))
        a += nc
        offer = F.pad(x[:, a : a + no], (0, 0, 0, OFFER_CAP - no))
        return summary, hand, torch.cat([joker, cons, offer], dim=1)
