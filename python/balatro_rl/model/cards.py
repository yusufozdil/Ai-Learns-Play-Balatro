"""Canonical-order card selection (torch, vectorised) used by the model's card head.

The core rule (``docs/design/rl_interface.md`` section 6) lets the model add cards in *any* order,
so one card set has many action chains and a policy's probability of a set would be spread over
all of them. The model instead builds sets in **ascending position order**: every set has exactly
one chain, which makes ``logprob(set)`` well defined (and the BC target unique).

State: ``partial`` (bool ``(n, H)``) and ``last`` = highest selected position (``-1`` if none).
Position ``q`` may be picked next iff ``q`` is a candidate above ``last`` and, after picking it,
a legal completion still exists:

* ``q <= min(missing required)``  (a required card is never skipped),
* ``k + 1 <= max``,
* ``|missing after| <= max - k - 1``,
* ``min - k - 1 <= #candidates above q``  (the minimum count stays reachable; the core rule needs
  no such term because it is order-free).

STOP is allowed iff the set is finishable (all required in, ``min <= k <= max``). By induction a
feasible state either can STOP or has a feasible next pick (take the smallest card of any
completion), and every legal set is reached by exactly one chain (its own ascending order);
``tests_py/test_model.py`` checks both by brute force against ``balatro_rl.select``.
"""

from __future__ import annotations

import torch


def unpack(mask: torch.Tensor, nbits: int) -> torch.Tensor:
    """Bit-set int64 ``(...)`` -> bool ``(..., nbits)`` (bit ``i`` -> index ``i``)."""
    bits = torch.arange(nbits, device=mask.device, dtype=torch.int64)
    return ((mask.unsqueeze(-1) >> bits) & 1).bool()


def pack(bits: torch.Tensor) -> torch.Tensor:
    """bool ``(..., nbits)`` -> int64 bit set."""
    n = bits.shape[-1]
    weights = torch.tensor([1 << i for i in range(n)], device=bits.device, dtype=torch.int64)
    return (bits.long() * weights).sum(-1)


def canonical_step(
    cand: torch.Tensor,
    req: torch.Tensor,
    mn: torch.Tensor,
    mx: torch.Tensor,
    partial: torch.Tensor,
    last: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allowed next positions ``(n, H)`` bool and whether STOP is allowed ``(n,)`` bool.

    ``cand`` / ``req`` / ``partial`` are bool ``(n, H)``; ``mn`` / ``mx`` / ``last`` int64 ``(n,)``.
    """
    n, h = cand.shape
    pos = torch.arange(h, device=cand.device)
    cand = cand | req
    k = partial.sum(1)
    missing = req & ~partial
    n_missing = missing.sum(1)
    min_missing = torch.where(missing, pos, h).min(1).values
    avail = cand & ~partial & (pos > last[:, None])
    avail_i = avail.long()
    above = avail_i.sum(1, keepdim=True) - avail_i.cumsum(1)
    missing_after = n_missing[:, None] - missing.long()
    nxt = (
        avail
        & (pos <= min_missing[:, None])
        & ((k + 1) <= mx)[:, None]
        & (missing_after <= (mx - k - 1)[:, None])
        & ((mn - k - 1)[:, None] <= above)
    )
    stop = (n_missing == 0) & (k >= mn) & (k <= mx)
    return nxt, stop
