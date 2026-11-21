#!/usr/bin/env python3
"""Serial GPU job queue for BalatroRL (stdlib only).

Layout under <root>/runs/queue/: pending/ running/ done/ failed/ logs/ status.json
Job file: pending/<NNN>_<name>.json
  {"name", "cmd", "cwd", "branch", "commit", "kind": "train|eval|bench",
   "run_dir": optional, "notes": optional, "expected_minutes": optional}

  gpu_queue.py run                 # daemon (single instance via flock)
  gpu_queue.py add --name N --cmd C --cwd D [--branch B] [--kind train]
  gpu_queue.py ls
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATES = ("pending", "running", "done", "failed")
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git(cwd, *args) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=20
        )
        return r.stdout.strip() or None
    except Exception:
        return None


class Queue:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.qdir = self.runs / "queue"
        self.gpu_lock = self.runs / ".gpu.lock"
        for s in (*STATES, "logs"):
            (self.qdir / s).mkdir(parents=True, exist_ok=True)

    def path(self, state: str, fname: str) -> Path:
        return self.qdir / state / fname

    def list_state(self, state: str) -> list[dict]:
        out = []
        for p in sorted((self.qdir / state).glob("*.json")):
            try:
                d = json.loads(p.read_text())
                if not isinstance(d, dict):
                    raise ValueError
            except (OSError, ValueError):
                d = {"name": p.stem, "bad_json": True}
            d["id"] = p.stem
            out.append(d)
        return out

    def next_number(self) -> int:
        best = 0
        for s in STATES:
            for p in (self.qdir / s).glob("*.json"):
                m = re.match(r"^(\d+)_", p.name)
                if m:
                    best = max(best, int(m.group(1)))
        return best + 1

    def add(
        self,
        name,
        cmd,
        cwd,
        branch=None,
        commit=None,
        kind="train",
        run_dir=None,
        notes=None,
        expected_minutes=None,
    ) -> Path:
        if not NAME_RE.match(name):
            raise ValueError(f"bad job name {name!r} (allowed: A-Za-z0-9_.+-)")
        if commit is None:
            commit = git(cwd, "rev-parse", "--short", "HEAD")
        if branch is None:
            branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
        job = {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "branch": branch,
            "commit": commit,
            "kind": kind,
            "run_dir": run_dir,
            "notes": notes,
            "expected_minutes": expected_minutes,
            "added": now_iso(),
        }
        n = self.next_number()
        p = self.path("pending", f"{n:03d}_{name}.json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2))
        os.replace(tmp, p)  # atomic: the daemon only globs *.json
        return p

    def write_status(self, current=None, runner_pid=None):
        fin = self.list_state("done") + self.list_state("failed")
        fin.sort(key=lambda d: d.get("ended") or "")
        st = {
            "updated": now_iso(),
            "runner_pid": runner_pid,
            "current": current,
            "pending": self.list_state("pending"),
            "finished": fin[-20:],
        }
        tmp = self.qdir / "status.json.tmp"
        tmp.write_text(json.dumps(st, indent=2))
        os.replace(tmp, self.qdir / "status.json")


class Runner:
    def __init__(self, q: Queue, poll: float = 2.0):
        self.q, self.poll = q, poll
        self.stop = False
        self.child: subprocess.Popen | None = None

    def _on_term(self, signum, frame):
        self.stop = True
        self._kill_child()

    def _kill_child(self):
        c = self.child
        if c and c.poll() is None:
            try:
                os.killpg(c.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def finish(self, fname, job, code, killed=False, note=None):
        job["ended"] = now_iso()
        job["exit_code"] = code
        job["duration_s"] = round(time.time() - job.pop("_t0", time.time()), 1)
        if killed:
            job["error"] = "killed"
        if note:
            job["error"] = note
        state = "done" if code == 0 and not killed and not note else "failed"
        job["state"] = state
        dst = self.q.path(state, fname)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2))
        os.replace(tmp, dst)
        try:
            self.q.path("running", fname).unlink()
        except FileNotFoundError:
            pass

    def run_job(self, pend: Path):
        fname = pend.name
        run_p = self.q.path("running", fname)
        try:
            job = json.loads(pend.read_text())
            cmd, cwd = job["cmd"], job["cwd"]
        except Exception as e:
            job = {"name": pend.stem, "_t0": time.time(), "started": now_iso()}
            os.replace(pend, run_p)
            self.finish(fname, job, 2, note=f"bad job file: {e}")
            return
        os.replace(pend, run_p)
        job["started"] = now_iso()
        job["_t0"] = time.time()
        job["id"] = pend.stem
        log = self.q.qdir / "logs" / f"{pend.stem}.log"
        job["log"] = str(log)
        cur = {k: v for k, v in job.items() if k != "_t0"}
        run_p.write_text(json.dumps(cur, indent=2))
        self.q.write_status(current=cur, runner_pid=os.getpid())
        code, killed = 1, False
        try:
            with open(log, "ab") as lf:
                lf.write(f"[queue] start {job['started']} cwd={cwd}\n[queue] cmd: {cmd}\n".encode())
                lf.flush()
                self.child = subprocess.Popen(
                    ["flock", str(self.q.gpu_lock), "bash", "-lc", cmd],
                    cwd=cwd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                last = time.time()
                while self.child.poll() is None:
                    time.sleep(0.5)
                    if time.time() - last > 15:
                        self.q.write_status(current=cur, runner_pid=os.getpid())
                        last = time.time()
                code = self.child.returncode
                killed = self.stop or code in (-signal.SIGTERM, -signal.SIGKILL)
                lf.write(f"[queue] exit {code}\n".encode())
        except Exception as e:  # e.g. missing cwd
            self.child = None
            self.finish(fname, job, 127, note=f"launch error: {e}")
            return
        self.child = None
        self.finish(fname, job, code, killed=killed)

    def recover_orphans(self):
        for p in sorted((self.q.qdir / "running").glob("*.json")):
            try:
                job = json.loads(p.read_text())
            except Exception:
                job = {"name": p.stem}
            job["_t0"] = time.time()
            self.finish(p.name, job, 1, note="orphaned (runner died)")

    def loop(self):
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)
        self.recover_orphans()
        print(f"[queue] runner pid {os.getpid()} watching {self.q.qdir}", flush=True)
        while not self.stop:
            pend = sorted((self.q.qdir / "pending").glob("*.json"))
            if pend:
                self.run_job(pend[0])
                self.q.write_status(current=None, runner_pid=os.getpid())
                continue
            self.q.write_status(current=None, runner_pid=os.getpid())
            for _ in range(int(self.poll * 4)):
                if self.stop:
                    break
                time.sleep(0.25)
        self.q.write_status(current=None, runner_pid=None)


def fmt_dur(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(REPO_ROOT))
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("run")
    a = sub.add_parser("add")
    a.add_argument("--name", required=True)
    a.add_argument("--cmd", required=True)
    a.add_argument("--cwd", required=True)
    a.add_argument("--branch")
    a.add_argument("--commit")
    a.add_argument("--kind", default="train", choices=["train", "eval", "bench"])
    a.add_argument("--run-dir")
    a.add_argument("--notes")
    a.add_argument("--expected-minutes", type=float)
    sub.add_parser("ls")
    args = ap.parse_args(argv)
    q = Queue(Path(args.root))
    if args.mode == "add":
        print(
            q.add(
                args.name,
                args.cmd,
                args.cwd,
                args.branch,
                args.commit,
                args.kind,
                args.run_dir,
                args.notes,
                args.expected_minutes,
            )
        )
    elif args.mode == "ls":
        for s in ("running", "pending", "done", "failed"):
            for j in q.list_state(s):
                print(
                    f"{s:8s} {j['id']:32s} {j.get('kind', '-'):5s} {j.get('commit', '-')!s:9s} "
                    f"exit={j.get('exit_code', '-')} dur={fmt_dur(j.get('duration_s'))}"
                )
    else:
        lock = open(q.qdir / ".runner.lock", "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another runner already holds the lock", file=sys.stderr)
            return 1
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        Runner(q).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
