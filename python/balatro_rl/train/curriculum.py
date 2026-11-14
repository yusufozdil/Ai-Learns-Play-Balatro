"""Curriculum routing: which phases the model plays and which the heuristic teacher plays.

Each env step has a phase (BlindSelect 0, Play 1, Shop 2, PackOpen 3). A *route* says, per phase,
whether the action is chosen by the model ("model") or by the teacher (``env.teacher_actions``,
the fair-information heuristic team agent). Only model-chosen steps enter the policy loss; the
critic learns from every step.

Stage spec string (``--curriculum``), comma separated, each ``phases[@env_steps]``; the last stage
has no limit, an earlier stage ends once the *global* env-step counter reaches its limit:

    play@5e6,play+pack+blind@10e6,all        # PLAY only, then + PACK + BLIND, then everything

Phases are joined with ``+``: ``blind play shop pack``; ``all`` = all four. Aliases (the PRD's
stage numbers): ``s1`` = play, ``s2`` = play+pack+blind, ``s3`` = ``s4`` = all (stage 3 adds SHOP,
stage 4 is "end to end" = the same routing, usually with the teacher KL and the critic warm-up
switched off).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES = ("blind", "play", "shop", "pack")
PHASE_INDEX = {"blind": 0, "play": 1, "shop": 2, "pack": 3}
ALIASES = {
    "s1": "play",
    "s2": "play+pack+blind",
    "s3": "blind+play+shop+pack",
    "s4": "blind+play+shop+pack",
    "all": "blind+play+shop+pack",
    "none": "",
}


@dataclass(frozen=True)
class Stage:
    phases: frozenset[str]
    until: float | None  # global env step at which the stage ends (None = never)

    @property
    def label(self) -> str:
        return "+".join(p for p in PHASES if p in self.phases) or "none"

    def route_table(self) -> np.ndarray:
        """bool ``(6,)`` indexed by the ``phase`` id (the two terminal phases are never
        observed in a running env): True = the model plays this phase."""
        t = np.zeros(6, dtype=bool)
        for p in self.phases:
            t[PHASE_INDEX[p]] = True
        return t


def parse_phases(spec: str) -> frozenset[str]:
    spec = ALIASES.get(spec.strip().lower(), spec.strip().lower())
    names = [s for s in spec.replace(" ", "").split("+") if s]
    for n in names:
        if n not in PHASE_INDEX:
            raise ValueError(f"unknown phase {n!r} (expected one of {PHASES})")
    return frozenset(names)


class Curriculum:
    def __init__(self, spec: str = "all"):
        self.spec = spec
        self.stages: list[Stage] = []
        parts = [p for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("empty curriculum")
        for i, part in enumerate(parts):
            phases, _, until = part.partition("@")
            if until and i == len(parts) - 1:
                raise ValueError("the last stage must not have an @limit")
            if not until and i < len(parts) - 1:
                raise ValueError(f"stage {part!r} needs an @env_steps limit")
            self.stages.append(Stage(parse_phases(phases), float(until) if until else None))

    def stage_at(self, global_step: int) -> tuple[int, Stage]:
        for i, s in enumerate(self.stages):
            if s.until is None or global_step < s.until:
                return i, s
        return len(self.stages) - 1, self.stages[-1]

    def __repr__(self) -> str:
        return f"Curriculum({self.spec!r})"
