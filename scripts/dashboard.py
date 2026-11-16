#!/usr/bin/env python3
"""BalatroRL training dashboard (stdlib only).

  uv run --no-sync python scripts/dashboard.py --port 8765 [--host 127.0.0.1] [--root REPO]

Serves one self-contained HTML page plus a small JSON API, all read-only, all
confined to <root>/runs:
  /api/state    now-running job, GPU/CPU/RAM, roadmap, queue lists
  /api/runs     summary of every runs/<name>/ with log.jsonl or config.json
  /api/run?name=X[&max=800]   eval + (downsampled) update series of one run
  /api/results  summaries of runs/eval/*.json
  /api/log?job=<id>|run=<name>[&n=200]   log tail (plain text)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
ACTIVE_SECS = 180  # a log touched within this window counts as "running"

UPDATE_KEYS = ("update", "global_step", "wall_seconds", "steps_per_sec", "ep_win", "ep_ante",
               "ep_return", "vloss", "ent", "approx_kl", "explained_var", "rollout_seconds",
               "update_seconds", "stage_label")
EVAL_KEYS = ("update", "global_step", "det_win_rate", "det_mean_ante", "det_games", "det_truncated",
             "sam_win_rate", "sam_mean_ante", "sam_games", "new_best", "eval_seconds")


# --------------------------------------------------------------------------- helpers
def read_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def iso_to_ts(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def tail_lines(path: Path, n: int = 200, max_bytes: int = 400_000) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    return "\n".join(data.decode("utf-8", "replace").splitlines()[-n:])


class LogCache:
    """Incrementally parsed log.jsonl files, keyed by path (offset based)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.files: dict[str, dict] = {}

    def get(self, path: Path) -> dict:
        key = str(path)
        with self.lock:
            st = self.files.setdefault(key, {"offset": 0, "updates": [], "evals": [], "mtime": 0.0})
            try:
                size = path.stat().st_size
                st["mtime"] = path.stat().st_mtime
            except OSError:
                return st
            if size < st["offset"]:  # truncated / rewritten
                st.update(offset=0, updates=[], evals=[])
            if size > st["offset"]:
                try:
                    with open(path, "rb") as f:
                        f.seek(st["offset"])
                        chunk = f.read(size - st["offset"])
                except OSError:
                    return st
                end = chunk.rfind(b"\n")
                if end >= 0:  # only consume complete lines
                    st["offset"] += end + 1
                    for line in chunk[:end].splitlines():
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(d, dict):
                            continue
                        if d.get("type") == "update":
                            st["updates"].append({k: num(d.get(k)) if k != "stage_label" else d.get(k)
                                                  for k in UPDATE_KEYS if k in d})
                        elif d.get("type") == "eval":
                            row = {k: (d.get(k) if k == "new_best" else num(d.get(k)))
                                   for k in EVAL_KEYS if k in d}
                            st["evals"].append(row)
            return st


def downsample(rows: list, maxn: int) -> list:
    if len(rows) <= maxn:
        return rows
    step = len(rows) / maxn
    out = [rows[int(i * step)] for i in range(maxn)]
    out[-1] = rows[-1]
    return out


def eta_info(updates: list, total) -> dict:
    """Progress + ETA from the last ~15 updates (step delta over wall delta)."""
    if not updates:
        return {}
    last = updates[-1]
    gs = last.get("global_step")
    info = {"global_step": gs, "total_steps": total, "wall_seconds": last.get("wall_seconds")}
    tail = [u for u in updates[-15:] if u.get("global_step") is not None and u.get("wall_seconds") is not None]
    sps = None
    if len(tail) >= 2:
        dw = tail[-1]["wall_seconds"] - tail[0]["wall_seconds"]
        ds = tail[-1]["global_step"] - tail[0]["global_step"]
        if dw > 0 and ds > 0:
            sps = ds / dw
    if sps is None:
        sps = last.get("steps_per_sec")
    info["steps_per_sec"] = sps
    if gs is not None and total and sps:
        info["progress"] = min(1.0, gs / total)
        info["eta_seconds"] = max(0.0, (total - gs) / sps)
    elif gs is not None and total:
        info["progress"] = min(1.0, gs / total)
    return info


# --------------------------------------------------------------------------- app
class App:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs"
        self.runs_real = os.path.realpath(self.runs)
        self.queue = self.runs / "queue"
        self.cache = LogCache()
        self.res_cache: dict[str, tuple] = {}
        self.sys_cache = (0.0, {})
        self.lock = threading.Lock()
        self.ensure_roadmap()

    # -- safety
    def safe_under_runs(self, p: Path) -> bool:
        rp = os.path.realpath(p)
        return rp == self.runs_real or rp.startswith(self.runs_real + os.sep)

    def run_dir(self, name: str):
        if not name or not NAME_RE.match(name) or name in (".", ".."):
            return None
        d = self.runs / name
        return d if d.is_dir() and self.safe_under_runs(d) else None

    # -- roadmap
    def ensure_roadmap(self):
        p = self.runs / "dashboard" / "roadmap.json"
        if p.exists():
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            sample = {"updated": datetime.now().astimezone().isoformat(timespec="seconds"),
                      "phase": "Örnek faz (orchestrator bu dosyayı günceller)",
                      "items": [
                          {"id": "E1", "title": "Örnek: deney 1", "status": "running", "branch": None, "result": None},
                          {"id": "E2", "title": "Örnek: deney 2", "status": "todo", "branch": None, "result": None},
                          {"id": "E0", "title": "Örnek: bitmiş iş", "status": "done", "branch": None, "result": "%50 win"}],
                      "notes": "Bu bir örnek dosyadır."}
            p.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
        except OSError:
            pass

    # -- system
    def system(self) -> dict:
        t, cached = self.sys_cache
        if time.time() - t < 4:
            return cached
        out: dict = {}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
            v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            out["gpu"] = {"util": float(v[0]), "mem_used": float(v[1]), "mem_total": float(v[2]), "temp": float(v[3])}
        except Exception:
            out["gpu"] = None
        try:
            out["load"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
            out["cpus"] = os.cpu_count()
        except OSError:
            out["load"] = None
        try:
            mi = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                mi[k] = int(v.split()[0])
            out["ram"] = {"total_mb": mi["MemTotal"] / 1024, "avail_mb": mi["MemAvailable"] / 1024}
        except (OSError, KeyError, ValueError):
            out["ram"] = None
        self.sys_cache = (time.time(), out)
        return out

    # -- runs
    def run_names(self) -> list[str]:
        names = []
        try:
            for d in self.runs.iterdir():
                if d.is_dir() and NAME_RE.match(d.name) and ((d / "log.jsonl").exists() or (d / "config.json").exists()):
                    names.append(d.name)
        except OSError:
            pass
        return names

    def run_summary(self, name: str) -> dict:
        d = self.runs / name
        cfg = read_json(d / "config.json", {}) or {}
        log = d / "log.jsonl"
        st = self.cache.get(log) if log.exists() else {"updates": [], "evals": [], "mtime": 0}
        total = num(cfg.get("total_steps"))
        eta = eta_info(st["updates"], total)
        last_ev = st["evals"][-1] if st["evals"] else {}
        best_ev = max((e for e in st["evals"] if e.get("det_win_rate") is not None),
                      key=lambda e: e["det_win_rate"], default={})
        mt = st["mtime"] or (d / "config.json").stat().st_mtime
        init = cfg.get("init_checkpoint")
        return {
            "name": name, "mtime": mt, "active": (time.time() - mt) < ACTIVE_SECS and bool(st["updates"]),
            "init": init, "total_steps": total, "global_step": eta.get("global_step"),
            "progress": eta.get("progress"), "eta_seconds": eta.get("eta_seconds"),
            "steps_per_sec": eta.get("steps_per_sec"),
            "decks": cfg.get("decks"), "stakes": cfg.get("stakes"),
            "reward": cfg.get("reward"), "eval_set": f"{cfg.get('eval_deck', '-')}/{cfg.get('eval_stake', '-')}",
            "seed": cfg.get("seed"), "num_envs": cfg.get("num_envs"), "lr": cfg.get("lr"),
            "last_val_win": last_ev.get("det_win_rate"), "last_val_ante": last_ev.get("det_mean_ante"),
            "best_val_win": best_ev.get("det_win_rate"), "n_evals": len(st["evals"]),
        }

    def run_series(self, name: str, maxn: int) -> dict | None:
        d = self.run_dir(name)
        if d is None:
            return None
        cfg = read_json(d / "config.json", {}) or {}
        st = self.cache.get(d / "log.jsonl")
        summ = self.run_summary(name)
        ref = None
        init = cfg.get("init_checkpoint")
        if isinstance(init, str):
            m = re.search(r"runs/([A-Za-z0-9_.+-]+)/", init)
            if m and m.group(1) != name and self.run_dir(m.group(1)):
                ev = self.cache.get(self.runs / m.group(1) / "log.jsonl")["evals"]
                if ev:
                    last = ev[-1]
                    ref = {"label": f"init: {m.group(1)}", "det_win_rate": last.get("det_win_rate"),
                           "det_mean_ante": last.get("det_mean_ante")}
        return {"name": name, "summary": summ, "config": {k: cfg.get(k) for k in (
                    "total_steps", "decks", "stakes", "init_checkpoint", "eval_deck", "eval_stake",
                    "eval_games", "eval_every", "reward", "lr", "seed", "num_envs")},
                "ref": ref, "evals": st["evals"], "updates": downsample(st["updates"], maxn)}

    # -- results
    def results(self) -> list[dict]:
        rows = []
        edir = self.runs / "eval"
        try:
            files = sorted(edir.glob("*.json"))
        except OSError:
            files = []
        for f in files:
            try:
                stt = f.stat()
            except OSError:
                continue
            key = str(f)
            sig = (stt.st_mtime, stt.st_size)
            c = self.res_cache.get(key)
            if not c or c[0] != sig:
                c = (sig, self.summarize_eval(f.stem, read_json(f)))
                self.res_cache[key] = c
            for r in c[1]:
                rows.append({**r, "mtime": stt.st_mtime})
        return rows

    @staticmethod
    def summarize_eval(stem: str, data) -> list[dict]:
        if not isinstance(data, dict):
            return []
        entries = [(None, data)] if "games" in data else [(k, v) for k, v in data.items()
                                                          if isinstance(v, dict) and "games" in v]
        out = []
        for k, v in entries:
            n, p = num(v.get("games")), num(v.get("win_rate"))
            se = math.sqrt(p * (1 - p) / n) if (n and p is not None and n > 0) else None
            tag = k if k is not None else None
            sset = "/".join(str(data[x]) for x in ("deck", "stake") if x in data) or None
            out.append({"file": stem, "entry": tag, "name": v.get("name") if isinstance(v.get("name"), str) else tag,
                        "set": sset, "checkpoint": data.get("checkpoint") if isinstance(data.get("checkpoint"), str) else None,
                        "global_step": num(data.get("global_step")), "games": n,
                        "mean_ante": num(v.get("mean_ante")), "win_rate": p,
                        "win_pct": p * 100 if p is not None else None, "se_pct": se * 100 if se is not None else None,
                        "truncated": num(v.get("truncated"))})
        return out

    # -- queue
    def queue_state(self) -> dict:
        def lst(state, limit=None):
            out = []
            try:
                files = sorted((self.queue / state).glob("*.json"))
            except OSError:
                files = []
            if limit:
                files = files[-limit:]
            for p in files:
                d = read_json(p)
                if not isinstance(d, dict):
                    d = {"name": p.stem, "bad_json": True}
                d["id"] = p.stem
                out.append(d)
            return out
        st = read_json(self.queue / "status.json", {}) or {}
        pid = st.get("runner_pid")
        alive = False
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        return {"running": lst("running"), "pending": lst("pending"), "done": lst("done", 30)[::-1],
                "failed": lst("failed", 30)[::-1], "runner_alive": alive}

    def job_run(self, job: dict) -> str | None:
        cands = []
        rd = job.get("run_dir")
        if isinstance(rd, str) and rd:
            cands.append(Path(rd).name)
        m = re.search(r"--name[ =]([A-Za-z0-9_.+-]+)", job.get("cmd") or "")
        if m:
            cands.append(m.group(1))
        if job.get("name"):
            cands.append(job["name"])
        for c in cands:
            if self.run_dir(c):
                return c
        return None

    def state(self) -> dict:
        q = self.queue_state()
        now = None
        cur_run = None
        if q["running"]:
            j = q["running"][0]
            started = iso_to_ts(j.get("started"))
            now = {"source": "queue", "job": j, "elapsed": time.time() - started if started else None}
            cur_run = self.job_run(j)
        else:
            best = None
            for n in self.run_names():
                s = self.run_summary(n)
                if s["active"] and (best is None or s["mtime"] > best["mtime"]):
                    best = s
            if best:
                cur_run = best["name"]
                now = {"source": "untracked", "job": {"name": best["name"], "kind": "train"}, "elapsed": None}
        run = self.run_summary(cur_run) if cur_run else None
        if now is not None:
            now["run"] = run
            now["run_name"] = cur_run
        names = self.run_names()
        default_run = cur_run
        if default_run is None and names:
            default_run = max(names, key=lambda n: (self.runs / n / "log.jsonl").stat().st_mtime
                              if (self.runs / n / "log.jsonl").exists() else 0)
        return {"time": time.time(), "now": now, "system": self.system(),
                "roadmap": read_json(self.runs / "dashboard" / "roadmap.json"),
                "queue": q, "default_run": default_run, "root": str(self.root)}


# --------------------------------------------------------------------------- http
def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "BalatroDash/1"

        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def jsend(self, obj, code=200):
            self.send(code, json.dumps(obj, allow_nan=False, default=str).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    return self.jsend(app.state())
                if u.path == "/api/runs":
                    rs = [app.run_summary(n) for n in app.run_names()]
                    rs.sort(key=lambda r: -r["mtime"])
                    return self.jsend(rs)
                if u.path == "/api/run":
                    try:
                        maxn = max(50, min(3000, int(q.get("max", 800))))
                    except ValueError:
                        maxn = 800
                    r = app.run_series(q.get("name", ""), maxn)
                    return self.jsend(r) if r else self.jsend({"error": "unknown run"}, 404)
                if u.path == "/api/results":
                    return self.jsend(app.results())
                if u.path == "/api/log":
                    try:
                        n = max(1, min(2000, int(q.get("n", 200))))
                    except ValueError:
                        n = 200
                    if "job" in q and NAME_RE.match(q["job"]):
                        p = app.queue / "logs" / f"{q['job']}.log"
                    elif "run" in q and NAME_RE.match(q["run"]):
                        p = app.runs / f"{q['run']}.log"
                    else:
                        return self.send(400, b"bad request", "text/plain")
                    if not app.safe_under_runs(p) or not p.is_file():
                        return self.send(404, b"no such log", "text/plain")
                    return self.send(200, tail_lines(p, n).encode(), "text/plain; charset=utf-8")
                return self.send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # never kill the server on odd data
                try:
                    self.jsend({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:
                    pass

    return H


PAGE = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BalatroRL Panel</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#1d2129;--mut:#6b7380;--line:#e1e4ea;--grid:#e8ebf0;--acc:#2f6fdb;--ok:#2a9d57;--bad:#d64545;--warn:#d98e0b}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--fg:#e4e7ec;--mut:#8b94a3;--line:#2c323c;--grid:#2a2f38;--acc:#5b9bff;--ok:#4cc27d;--bad:#ef6a6a;--warn:#e5a93b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:16px;margin:0}.mut{color:var(--mut)}main{padding:12px 16px;display:grid;gap:12px;max-width:1500px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
.card h2{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
.kv{display:flex;gap:16px;flex-wrap:wrap}.kv div{min-width:90px}.kv b{display:block;font-size:18px}
.bar{height:10px;background:var(--grid);border-radius:5px;overflow:hidden;margin:6px 0}.bar i{display:block;height:100%;background:var(--acc)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:3px 8px 3px 0;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none}td.n,th.n{text-align:right}
.tw{overflow-x:auto;max-height:360px;overflow-y:auto}tr.sel td{background:rgba(91,155,255,.12)}tr.click{cursor:pointer}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;border:1px solid var(--line)}
.s-running{color:var(--acc);border-color:var(--acc)}.s-done{color:var(--ok);border-color:var(--ok)}.s-blocked,.s-failed{color:var(--bad);border-color:var(--bad)}.s-todo{color:var(--mut)}.s-dropped{color:var(--mut);text-decoration:line-through}
.road{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px}.road .it{border:1px solid var(--line);border-radius:6px;padding:5px 8px}
.road .it small{display:block;color:var(--mut)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px}
.ch h3{font-size:12px;margin:0 0 2px;color:var(--mut)}.leg{font-size:12px;display:flex;gap:10px;flex-wrap:wrap}.leg i{display:inline-block;width:10px;height:3px;margin-right:4px;vertical-align:middle}
.chart{position:relative}.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:3px 6px;font-size:12px;display:none;white-space:nowrap;z-index:5}
pre{background:var(--bg);padding:8px;border-radius:6px;overflow:auto;max-height:340px;font-size:12px;margin:6px 0 0}
select,input,button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:3px 6px;font:inherit}
a{color:var(--acc);cursor:pointer}.empty{color:var(--mut);padding:30px 0;text-align:center}
.ctl{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
</style></head><body>
<header><h1>BalatroRL Panel</h1><span class="mut" id="upd"></span><span class="mut" id="err"></span></header>
<main>
<section class="card" id="now"><h2>Şu an koşan</h2><div id="nowbody" class="mut">yükleniyor…</div></section>
<section class="card"><h2>Yol haritası <span class="mut" id="roadmeta"></span></h2><div id="road"></div></section>
<section class="card"><h2>Canlı koşu</h2>
 <div class="ctl"><label>Koşu <select id="runsel"></select></label>
  <label>Karşılaştır (en fazla 2) <select id="cmpsel" multiple size="3"></select></label>
  <label>Yumuşatma <input id="smooth" type="number" min="1" max="200" value="10" style="width:60px"></label>
  <span class="mut" id="runinfo"></span></div>
 <div class="charts" id="charts"></div></section>
<section class="card"><h2>Kuyruk <span class="mut" id="qmeta"></span></h2><div id="queue"></div><div id="logbox"></div></section>
<section class="card"><h2>Sonuçlar (runs/eval)</h2>
 <div class="ctl"><input id="rfilter" placeholder="filtre…"><span class="mut" id="rcount"></span></div><div class="tw" id="results"></div></section>
<section class="card"><h2>Koşular</h2><div class="tw" id="runs"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const COLORS=['#4c8df0','#e8884c','#49b36f'];
const store={get(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v)}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const fd=s=>{if(s==null||!isFinite(s))return '-';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}s ${m}dk`:m?`${m}dk ${s%60}sn`:`${s}sn`};
const pct=(v,d=1)=>v==null?'-':(v*100).toFixed(d)+'%';
const f2=(v,d=2)=>v==null?'-':Number(v).toFixed(d);
const mil=v=>v==null?'-':(v/1e6).toFixed(2)+'M';
const when=t=>t?new Date(t*1000).toLocaleString('tr-TR',{dateStyle:'short',timeStyle:'short'}):'-';
const J=async u=>{const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(u+' '+r.status);return r.json()};
let S={},RUNS=[],RES=[],sel=store.get('sel',null),cmp=store.get('cmp',[]),openLog=null,sortR=store.get('sortR',{k:'mtime',d:-1}),sortRuns={k:'mtime',d:-1};

function nowCard(){
  const n=S.now,sy=S.system||{};let h='';
  if(!n){h='<div class="mut">Şu an koşan iş yok.</div>'+(S.queue.runner_alive?'':' <span class="pill s-failed">kuyruk daemon kapalı</span>');}
  else{const j=n.job,r=n.run;
    h=`<div class="kv"><div><span class="mut">${n.source==='queue'?'Kuyruk işi':'Kuyruk dışı koşu'}</span><b>${esc(j.name)}</b></div>
    <div><span class="mut">tür</span><b>${esc(j.kind||'-')}</b></div>
    <div><span class="mut">branch / commit</span><b>${esc(j.branch||'-')} ${esc(j.commit||'')}</b></div>
    <div><span class="mut">geçen süre</span><b>${fd(n.elapsed)}</b></div>`;
    if(r&&r.progress!=null)h+=`<div><span class="mut">ilerleme</span><b>${(r.progress*100).toFixed(1)}%</b></div><div><span class="mut">ETA</span><b>${fd(r.eta_seconds)}</b></div><div><span class="mut">hız</span><b>${f2(r.steps_per_sec,0)} adım/sn</b></div><div><span class="mut">son val kazanma</span><b>${pct(r.last_val_win)}</b></div>`;
    else if(j.expected_minutes&&n.elapsed!=null)h+=`<div><span class="mut">tahmini</span><b>${f2(j.expected_minutes,0)} dk</b></div>`;
    h+='</div>';
    if(r&&r.progress!=null)h+=`<div class="bar"><i style="width:${r.progress*100}%"></i></div><div class="mut">${mil(r.global_step)} / ${mil(r.total_steps)} adım${j.notes?' · '+esc(j.notes):''}</div>`;
    else if(j.notes)h+=`<div class="mut">${esc(j.notes)}</div>`;}
  h+='<hr style="border:0;border-top:1px solid var(--line);margin:8px 0"><div class="kv">';
  if(sy.gpu)h+=`<div><span class="mut">GPU</span><b>${sy.gpu.util}%</b>${sy.gpu.mem_used.toFixed(0)}/${sy.gpu.mem_total.toFixed(0)} MB · ${sy.gpu.temp}°C</div>`;
  if(sy.load)h+=`<div><span class="mut">CPU yük</span><b>${sy.load[0].toFixed(1)}</b>/ ${sy.cpus} çekirdek (5dk ${sy.load[1].toFixed(1)})</div>`;
  if(sy.ram)h+=`<div><span class="mut">RAM</span><b>${((sy.ram.total_mb-sy.ram.avail_mb)/1024).toFixed(1)} GB</b>/ ${(sy.ram.total_mb/1024).toFixed(1)} GB</div>`;
  h+=`<div><span class="mut">Kuyruk daemon</span><b>${S.queue.runner_alive?'açık':'kapalı'}</b>${S.queue.pending.length} bekliyor</div></div>`;
  $('nowbody').innerHTML=h;
}
function roadCard(){
  const r=S.roadmap;if(!r){$('road').innerHTML='<div class="mut">roadmap.json yok</div>';return}
  $('roadmeta').textContent=(r.phase?'· '+r.phase+' ':'')+(r.updated?'· '+r.updated:'');
  let h='<div class="road">';
  (r.items||[]).forEach(i=>{h+=`<div class="it"><span class="pill s-${esc(i.status)}">${esc(i.status)}</span> <b>${esc(i.id)}</b> ${esc(i.title)}<small>${esc(i.branch||'')}${i.result?' → '+esc(i.result):''}</small></div>`});
  h+='</div>';if(r.notes)h+=`<div class="mut" style="margin-top:6px">${esc(r.notes)}</div>`;$('road').innerHTML=h;
}
function jobRow(j,state){
  const d=j.duration_s!=null?fd(j.duration_s):(j.started?'…':'');
  return `<tr><td>${esc(j.id)}</td><td>${esc(j.kind||'')}</td><td>${esc(j.branch||'')} ${esc(j.commit||'')}</td><td class="n">${d}</td><td>${j.exit_code!=null?'exit '+j.exit_code:''}${j.error?' '+esc(j.error):''}</td><td>${esc(j.notes||'')}</td><td>${state==='pending'?'':`<a data-log="${esc(j.id)}">log</a>`}</td></tr>`;
}
function queueCard(){
  const q=S.queue;$('qmeta').textContent=q.runner_alive?'· daemon açık':'· daemon kapalı';
  const sec=(t,a,st)=>`<h3 style="font-size:13px;margin:8px 0 2px">${t} (${a.length})</h3>`+(a.length?`<div class="tw"><table><tr><th>iş</th><th>tür</th><th>branch/commit</th><th class="n">süre</th><th>sonuç</th><th>not</th><th></th></tr>${a.map(j=>jobRow(j,st)).join('')}</table></div>`:'<div class="mut">-</div>');
  $('queue').innerHTML=sec('Koşuyor',q.running,'running')+sec('Bekleyen',q.pending,'pending')+sec('Biten',q.done,'done')+sec('Başarısız',q.failed,'failed');
  document.querySelectorAll('[data-log]').forEach(a=>a.onclick=()=>{openLog=a.dataset.log;showLog()});
}
async function showLog(){
  if(!openLog){$('logbox').innerHTML='';return}
  try{const t=await (await fetch('/api/log?n=200&job='+encodeURIComponent(openLog),{cache:'no-store'})).text();
  $('logbox').innerHTML=`<div style="margin-top:8px"><b>${esc(openLog)}</b> — son 200 satır <a id="closelog">kapat</a></div><pre>${esc(t)}</pre>`;
  $('closelog').onclick=()=>{openLog=null;showLog()};const p=$('logbox').querySelector('pre');p.scrollTop=p.scrollHeight}catch(e){}
}
function sortRows(a,s){return a.slice().sort((x,y)=>{const u=x[s.k],v=y[s.k];if(u==null&&v==null)return 0;if(u==null)return 1;if(v==null)return -1;return (u<v?-1:u>v?1:0)*s.d})}
function mkTable(cols,rows,sortState,onSort,rowAttr){
  let h='<table><tr>'+cols.map(c=>`<th class="${c.n?'n':''}" data-k="${c.k}">${c.t}${sortState.k===c.k?(sortState.d>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
  rows.forEach(r=>{h+=`<tr ${rowAttr?rowAttr(r):''}>`+cols.map(c=>`<td class="${c.n?'n':''}">${c.f?c.f(r):esc(r[c.k]??'-')}</td>`).join('')+'</tr>'});
  return h+'</table>';
}
function resultsCard(){
  const f=$('rfilter').value.toLowerCase();
  const rows=sortRows(RES.filter(r=>!f||JSON.stringify([r.file,r.entry,r.name,r.set,r.checkpoint]).toLowerCase().includes(f)),sortR);
  $('rcount').textContent=`${rows.length}/${RES.length}`;
  const cols=[{k:'file',t:'dosya'},{k:'entry',t:'giriş',f:r=>esc(r.entry||'')},{k:'set',t:'set',f:r=>esc(r.set||'')},{k:'games',t:'oyun',n:1},{k:'mean_ante',t:'ort. ante',n:1,f:r=>f2(r.mean_ante)},
   {k:'win_pct',t:'kazanma % ± SE',n:1,f:r=>r.win_pct==null?'-':`${r.win_pct.toFixed(1)} ± ${f2(r.se_pct,1)}`},{k:'truncated',t:'kesilen',n:1,f:r=>pct(r.truncated)},{k:'global_step',t:'adım',n:1,f:r=>mil(r.global_step)},{k:'mtime',t:'değişiklik',f:r=>when(r.mtime)}];
  $('results').innerHTML=mkTable(cols,rows,sortR,null);
  $('results').querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortR={k,d:sortR.k===k?-sortR.d:-1};store.set('sortR',sortR);resultsCard()});
}
function runsCard(){
  const rows=sortRows(RUNS.map(r=>({...r,cfg:`${r.decks||'-'} / ${r.stakes||'-'}`,initn:r.init||'-'})),sortRuns);
  const cols=[{k:'name',t:'koşu',f:r=>(r.active?'<span class="pill s-running">aktif</span> ':'')+esc(r.name)},{k:'initn',t:'init',f:r=>esc(r.initn)},{k:'global_step',t:'adım',n:1,f:r=>`${mil(r.global_step)} / ${mil(r.total_steps)}`},
   {k:'cfg',t:'deck / stake'},{k:'eval_set',t:'val seti'},{k:'last_val_win',t:'son val kazanma',n:1,f:r=>pct(r.last_val_win)},{k:'best_val_win',t:'en iyi',n:1,f:r=>pct(r.best_val_win)},{k:'mtime',t:'değişiklik',f:r=>when(r.mtime)}];
  $('runs').innerHTML=mkTable(cols,rows,sortRuns,null,r=>`class="click ${r.name===sel?'sel':''}" data-run="${esc(r.name)}"`);
  $('runs').querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortRuns={k,d:sortRuns.k===k?-sortRuns.d:-1};runsCard()});
  $('runs').querySelectorAll('tr[data-run]').forEach(tr=>tr.onclick=()=>{setSel(tr.dataset.run);window.scrollTo({top:$('runsel').getBoundingClientRect().top+scrollY-60,behavior:'smooth'})});
}
/* ---------- charts ---------- */
function niceTicks(lo,hi,n){if(hi===lo){hi=lo+1}const raw=(hi-lo)/n,p=Math.pow(10,Math.floor(Math.log10(raw))),m=raw/p,st=(m<1.5?1:m<3?2:m<7?5:10)*p;const t=[];for(let v=Math.ceil(lo/st)*st;v<=hi+st*1e-6;v+=st)t.push(+v.toPrecision(12));return t}
function chart(el,series,o){
  o=o||{};const fmt=o.fmt||(v=>f2(v,2));
  const W=el.clientWidth||520,H=190,m={l:46,r:8,t:6,b:20};
  let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{if(p[1]!=null&&isFinite(p[1])){xs.push(p[0]);ys.push(p[1])}}));
  (o.hlines||[]).forEach(h=>{if(h.y!=null)ys.push(h.y)});
  if(!xs.length){el.innerHTML='<div class="empty">veri yok</div>';return}
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=o.ymin!=null?o.ymin:Math.min(...ys),y1=o.ymax!=null?o.ymax:Math.max(...ys);
  if(x1===x0)x1=x0+1;if(y1===y0){y1=y0+1}const pad=(y1-y0)*.06;if(o.ymin==null)y0-=pad;if(o.ymax==null)y1+=pad;if(o.floor0&&y0<0)y0=0;
  const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  let g='';niceTicks(y0,y1,4).forEach(t=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${m.l-4}" y="${Y(t)+4}" text-anchor="end" font-size="11" fill="var(--mut)">${esc(fmt(t))}</text>`});
  niceTicks(x0,x1,5).forEach(t=>{g+=`<text x="${X(t)}" y="${H-5}" text-anchor="middle" font-size="11" fill="var(--mut)">${(t/1e6).toFixed(t>=1e7?0:1)}M</text>`});
  (o.hlines||[]).forEach(h=>{if(h.y!=null)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="var(--mut)" stroke-dasharray="5 4"/><text x="${W-m.r-2}" y="${Y(h.y)-3}" text-anchor="end" font-size="11" fill="var(--mut)">${esc(h.label)} ${esc(fmt(h.y))}</text>`});
  series.forEach(s=>{const pts=s.pts.filter(p=>p[1]!=null&&isFinite(p[1]));if(!pts.length)return;
    g+=`<polyline fill="none" stroke="${s.color}" stroke-width="${s.w||1.8}" ${s.dash?'stroke-dasharray="4 3"':''} points="${pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}"/>`;
    if(pts.length<=40)pts.forEach(p=>g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="${s.color}"/>`)});
  g+=`<line id="cx" y1="${m.t}" y2="${H-m.b}" stroke="var(--mut)" style="display:none"/>`;
  el.innerHTML=`<div class="leg">${series.map(s=>`<span><i style="background:${s.color};${s.dash?'opacity:.6':''}"></i>${esc(s.name)}</span>`).join('')}</div><div class="chart"><svg width="${W}" height="${H}">${g}</svg><div class="tip"></div></div>`;
  const svg=el.querySelector('svg'),tip=el.querySelector('.tip'),cx=svg.querySelector('#cx');
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect(),xv=x0+(e.clientX-r.left-m.l)/(W-m.l-m.r)*(x1-x0);let lines=[],bx=null;
    series.forEach(s=>{let b=null;s.pts.forEach(p=>{if(p[1]!=null&&(b==null||Math.abs(p[0]-xv)<Math.abs(b[0]-xv)))b=p});if(b){lines.push(`<span style="color:${s.color}">●</span> ${esc(s.name)}: ${esc(fmt(b[1]))}`);if(bx==null||Math.abs(b[0]-xv)<Math.abs(bx-xv))bx=b[0]}});
    if(bx==null)return;cx.setAttribute('x1',X(bx));cx.setAttribute('x2',X(bx));cx.style.display='';
    tip.innerHTML=`<b>${mil(bx)}</b><br>`+lines.join('<br>');tip.style.display='block';const left=X(bx)+10>W-150?X(bx)-tip.offsetWidth-10:X(bx)+10;tip.style.left=left+'px';tip.style.top='6px'};
  svg.onmouseleave=()=>{tip.style.display='none';cx.style.display='none'};
}
function ma(pts,w){if(w<=1)return pts;const out=[];const q=[];let s=0;pts.forEach(p=>{if(p[1]==null){out.push([p[0],null]);return}q.push(p[1]);s+=p[1];if(q.length>w)s-=q.shift();out.push([p[0],s/q.length])});return out}
const serU=(d,k)=>d.updates.map(u=>[u.global_step,u[k]]);
const serE=(d,k)=>d.evals.map(u=>[u.global_step,u[k]]);
function metaText(d){const c=d.config||{};return `${d.name}: val ${c.eval_deck||'-'}/${c.eval_stake||'-'}, ${c.eval_games||'?'} oyun/${c.eval_every||'?'} güncelleme`}
async function liveCard(){
  const names=[sel,...cmp.filter(c=>c!==sel)].filter(Boolean);if(!names.length){$('charts').innerHTML='<div class="empty">koşu yok</div>';return}
  const ds=(await Promise.all(names.map(n=>J('/api/run?name='+encodeURIComponent(n)).catch(()=>null)))).filter(Boolean);if(!ds.length)return;
  const w=Math.max(1,+$('smooth').value||1);const multi=ds.length>1;
  const per=(fn)=>ds.map((d,i)=>fn(d,COLORS[i%3],i)).flat();
  const d0=ds[0];$('runinfo').textContent=metaText(d0.summary?{...d0,config:d0.config}:d0);
  const hl=[];if(d0.ref&&d0.ref.det_win_rate!=null)hl.push({y:d0.ref.det_win_rate,label:d0.ref.label});
  const hlA=[];if(d0.ref&&d0.ref.det_mean_ante!=null)hlA.push({y:d0.ref.det_mean_ante,label:d0.ref.label});
  const specs=[
   {t:'Val kazanma oranı (det düz, sampled kesik)',s:per((d,c)=>[{name:d.name+' det',color:c,pts:serE(d,'det_win_rate')},{name:d.name+' sam',color:c,dash:1,w:1.2,pts:serE(d,'sam_win_rate')}]),o:{fmt:v=>(v*100).toFixed(1)+'%',hlines:hl}},
   {t:'Val ortalama ante',s:per((d,c)=>[{name:d.name+' det',color:c,pts:serE(d,'det_mean_ante')},{name:d.name+' sam',color:c,dash:1,w:1.2,pts:serE(d,'sam_mean_ante')}]),o:{hlines:hlA}},
   {t:'Eğitim ep_win (yumuşatılmış)',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'ep_win'),w)}]),o:{fmt:v=>(v*100).toFixed(1)+'%'}},
   {t:'Eğitim ep_ante (yumuşatılmış)',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'ep_ante'),w)}])},
   {t:'Adım/sn',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'steps_per_sec'),w)}]),o:{fmt:v=>v.toFixed(0),floor0:1}},
   {t:'Explained variance',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'explained_var'),w)}])},
   {t:'Entropi',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'ent'),w)}])},
   {t:'approx_kl',s:per((d,c)=>[{name:d.name,color:c,pts:ma(serU(d,'approx_kl'),w)}]),o:{fmt:v=>v.toExponential(1)}}];
  const root=$('charts');
  if(root.children.length!==specs.length){root.innerHTML=specs.map(()=>'<div class="ch"><h3></h3><div class="cb"></div></div>').join('')}
  specs.forEach((sp,i)=>{const c=root.children[i];c.querySelector('h3').textContent=sp.t;chart(c.querySelector('.cb'),sp.s,sp.o)});
}
function setSel(n){sel=n;store.set('sel',n);fillSel();liveCard();runsCard()}
function fillSel(){
  const names=RUNS.map(r=>r.name);const rs=$('runsel');
  rs.innerHTML=names.map(n=>`<option ${n===sel?'selected':''}>${esc(n)}</option>`).join('');
  const cs=$('cmpsel');cs.innerHTML=names.filter(n=>n!==sel).map(n=>`<option ${cmp.includes(n)?'selected':''}>${esc(n)}</option>`).join('');
}
$('runsel').onchange=e=>setSel(e.target.value);
$('cmpsel').onchange=e=>{let v=[...e.target.selectedOptions].map(o=>o.value).slice(0,2);cmp=v;store.set('cmp',cmp);fillSel();liveCard()};
$('smooth').onchange=liveCard;$('rfilter').oninput=resultsCard;
let first=true;
async function tick(){
  try{
    [S,RUNS,RES]=await Promise.all([J('/api/state'),J('/api/runs'),J('/api/results')]);
    $('err').textContent='';$('upd').textContent='güncellendi '+new Date().toLocaleTimeString('tr-TR');
    if(!RUNS.some(r=>r.name===sel)){sel=(S.default_run&&RUNS.some(r=>r.name===S.default_run))?S.default_run:(RUNS[0]&&RUNS[0].name)||null}
    if(first){first=false}
    nowCard();roadCard();queueCard();resultsCard();runsCard();
    if(document.activeElement!==$('cmpsel')&&document.activeElement!==$('runsel'))fillSel();
    await liveCard();showLog();
  }catch(e){$('err').textContent='bağlantı hatası: '+e.message}
}
tick();setInterval(tick,10000);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = App(Path(args.root))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    srv.daemon_threads = True
    print(f"dashboard on http://{args.host}:{args.port}  (root {app.root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
