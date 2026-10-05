"""
obj_010 — metrics_path.py (exp_031b patch5): reproducibly put `metrics_v3f` (the arbiter axes that panel.py
imports) on sys.path, so the P1/stats scripts do NOT need a hand-set `PYTHONPATH=.../ship_patch2/src`. Call
ensure_metrics_v3f_on_path() BEFORE importing panel.

Resolution order: already-importable -> env (SCORE_PREDS_METRICS_SRC / SHIP_PATCH2_SRC / OBJ093_DIR) -> an
upward search from this file for a directory containing metrics_v3f.py (…/ship_patch2/src, …/src, …/clone/src).
metrics_v3f.py then self-resolves metrics_v3 from its own ../clone (its internal sys.path insert).
"""
from __future__ import annotations
import os, sys, importlib.util
from pathlib import Path


def ensure_metrics_v3f_on_path(explicit=None, verbose=True):
    if importlib.util.find_spec("metrics_v3f") is not None:
        return "already-importable"
    cands = []
    if explicit:
        cands.append(Path(explicit))
    for ev in ("SCORE_PREDS_METRICS_SRC", "SHIP_PATCH2_SRC"):
        v = os.environ.get(ev)
        if v:
            cands.append(Path(v))
    obj093 = os.environ.get("OBJ093_DIR")
    if obj093:
        cands += [Path(obj093) / "src", Path(obj093) / "clone"]
    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents)[:8]:
        cands += [up / "ship_patch2" / "src", up / "src", up / "ship" / "src", up / "clone" / "src"]
    seen = set()
    for c in cands:
        c = Path(c)
        if str(c) in seen:
            continue
        seen.add(str(c))
        if (c / "metrics_v3f.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            if verbose:
                print(f"[metrics_path] metrics_v3f resolved at {c}", flush=True)
            return str(c)
    if verbose:
        print("[metrics_path] metrics_v3f NOT found by search; relying on panel.py's own OBJ093_DIR fallback",
              flush=True)
    return None
