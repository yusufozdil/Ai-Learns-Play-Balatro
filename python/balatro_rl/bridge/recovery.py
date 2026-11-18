"""Unattended-run helpers: restart the game process and resume a log.

``restart_game`` runs ``bridge/restart_game.sh`` (kills Balatro + BalatroBot, starts them again,
waits for ``health``) up to ``attempts`` times and checks with the client afterwards.
``finished_seeds`` reads a JSONL log and returns the seeds that already have a ``game_end``
record, which is what ``--resume`` skips.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .client import Client

DEFAULT_SCRIPT = Path(__file__).resolve().parents[3] / "bridge" / "restart_game.sh"


def restart_game(
    client: Client,
    *,
    script: str | Path = DEFAULT_SCRIPT,
    attempts: int = 3,
    health_wait: float = 60.0,
    run: Callable[..., object] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Restart the game; returns ``{"ok", "attempts", "seconds", "detail"}`` (never raises)."""
    runner = run or subprocess.run
    t0 = time.time()
    detail = ""
    for n in range(1, attempts + 1):
        try:
            proc = runner(["bash", str(script)], capture_output=True, text=True, timeout=300)
            detail = (getattr(proc, "stdout", "") or "").strip()[-200:]
            err = (getattr(proc, "stderr", "") or "").strip()[-200:]
            if err:
                detail += " | " + err
        except Exception as e:  # noqa: BLE001 - a failed restart is retried
            detail = f"{type(e).__name__}: {e}"
        if client.wait_healthy(health_wait):
            return {"ok": True, "attempts": n, "seconds": round(time.time() - t0, 1),
                    "detail": detail}  # fmt: skip
        sleep(5.0)
    return {"ok": False, "attempts": attempts, "seconds": round(time.time() - t0, 1),
            "detail": detail}  # fmt: skip


def finished_seeds(log_path: str | Path) -> set[str]:
    """Seeds of the games of an existing log that have a ``game_end`` record."""
    p = Path(log_path)
    done: set[str] = set()
    if not p.exists():
        return done
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:  # a half-written last line of a killed run
            continue
        if rec.get("ev") == "game_end":
            done.add(str(rec["seed"]))
    return done
