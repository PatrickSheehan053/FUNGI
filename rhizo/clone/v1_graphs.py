"""
obj_009 — graphs.py : build the graph-swap ARMS as (src, tgt, out-normalized weight) on the File A panel.

Base arms from parquet (exp_024 File A prunes of the PSGRN dense parent):
  fungi_bio, top_weight, knn, mst, fungi_synth_obj008.
Derived arms (built in-memory from fungi_bio, per spec — the structure-blind / directionality controls):
  shuffle (degree-preserving), empty (no edges), reverse (flip direction), labelperm (permute node labels).

Direction-preserving OUT-normalization ONLY (never symmetric): Ã[t,s] = A[t,s] / (Σ_t A[t,s] + ε) — each edge
divided by its SOURCE's out-strength (spec §Architecture). Uses the cloned obj_008 graph_io.load_edges.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graph_io  # cloned obj_008

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP24 = os.path.join(ROOT, "DATA", "EXPERIMENTS", "exp_024_multisubstrate_pruning_and_obj008_synthetic")
PRUNED = os.path.join(EXP24, "intermediate", "pruned_graphs")
FUNGI_BIO = os.path.join(EXP24, "intermediate", "fungi", "champions_full", "file_a_oldpanel.parquet")
EPS = 1e-8

BASE_PATHS = {
    "fungi_bio": FUNGI_BIO,
    "top_weight": os.path.join(PRUNED, "psgrn__top_weight.parquet"),
    "knn": os.path.join(PRUNED, "psgrn__knn.parquet"),
    "mst": os.path.join(PRUNED, "psgrn__mst.parquet"),
    "fungi_synth_obj008": os.path.join(PRUNED, "psgrn__fungi_synth_obj008.parquet"),
}
ARMS = ["fungi_bio", "top_weight", "shuffle", "knn", "mst", "empty", "reverse", "labelperm", "fungi_synth_obj008"]


def _load_base(path, g2i, N):
    src, tgt, w = graph_io.load_edges(path)
    si = pd.Series(src).map(g2i).to_numpy(); ti = pd.Series(tgt).map(g2i).to_numpy()
    keep = ~(pd.isna(si) | pd.isna(ti))
    si = si[keep].astype(np.int64); ti = ti[keep].astype(np.int64); w = w[keep].astype(np.float64)
    k2 = si != ti
    return si[k2], ti[k2], w[k2]


def _out_normalize(src, tgt, w, N):
    if len(w) == 0:
        return src.astype(np.int64), tgt.astype(np.int64), w.astype(np.float32)
    outstr = np.bincount(src, weights=w, minlength=N)
    wn = w / (outstr[src] + EPS)
    return src.astype(np.int64), tgt.astype(np.int64), wn.astype(np.float32)


def _deg_preserving_shuffle(src, tgt, w, N, seed=42):
    """Permute target endpoints globally: preserves per-source out-degree AND per-target in-degree exactly."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tgt)); t2 = tgt[perm].copy()
    for _ in range(20):
        sl = np.where(src == t2)[0]
        if len(sl) == 0:
            break
        sw = rng.permutation(len(t2))[:len(sl)]
        t2[sl], t2[sw] = t2[sw], t2[sl].copy()
    keep = src != t2
    return src[keep], t2[keep], w[keep]


def _labelperm(src, tgt, w, N, seed=42):
    rng = np.random.default_rng(seed); perm = rng.permutation(N)
    return perm[src], perm[tgt], w


def build_arm(arm, g2i, N):
    """Return (src, tgt, w_outnorm) int64/int64/float32 for the named arm."""
    if arm == "empty":
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    if arm in BASE_PATHS:
        s, t, w = _load_base(BASE_PATHS[arm], g2i, N)
        return _out_normalize(s, t, w, N)
    # derived from fungi_bio
    s, t, w = _load_base(FUNGI_BIO, g2i, N)
    if arm == "shuffle":
        s, t, w = _deg_preserving_shuffle(s, t, w, N)
    elif arm == "reverse":
        s, t = t, s
    elif arm == "labelperm":
        s, t, w = _labelperm(s, t, w, N)
    else:
        raise ValueError(f"unknown arm {arm}")
    return _out_normalize(s, t, w, N)


def arm_stats(arm, g2i, N):
    s, t, w = build_arm(arm, g2i, N)
    return dict(arm=arm, n_edges=int(len(s)), n_sources=int(len(np.unique(s))) if len(s) else 0)
