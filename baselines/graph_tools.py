"""
exp_033 — graph_tools.py : reusable arm-graph builders (CLONE-ONLY; pure numpy/pandas/scipy).

Produces RHIZO arm .npz files in the harness contract format (src / tgt / w_outnorm / w_raw), the format
build_arm() in graphs_v3f.py consumes. Node ids are integer indices into the fixed 5000-gene RPE1 panel.

Pruners (matched to an EXACT edge count E so every (FUNGI, Topweight, KNN) triplet is edge-identical):
  topweight_prune(s,t,w,E)   keep top-E edges by weight (global)
  knn_exact(s,t,w,E)         per-source top-k out-edges, fractional fill to EXACTLY E edges
Fusion (Super):
  borda_fuse(A,B,N,cap)      rank-consensus (mean of per-parent rank-normalized scores), top-cap
Arm E mechanism controls:
  scramble_ms(s,t,w,seed)    degree-preserving double-edge swap (Maslov-Sneppen; keeps out- AND in-degree)
  reverse(s,t,w)             flip every edge direction
  wstrip(s,t,w)              binarize weights (w_raw:=1)
IO:
  save_arm(path,s,t,w_raw)   w_outnorm = per-source sum-norm(w_raw); write 4-key npz
  coverage_report(s,t,meta)  n_src / held_out_coverage / train_coverage / out-edge coverage

Run `python graph_tools.py --selftest` to unit-test every function on tiny synthetic inputs.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ IO / norm
def outnorm_persource(src, tgt, w):
    """Per-source sum-normalization (row-stochastic) over the KEPT edge set — the locked arm convention."""
    w = np.asarray(w, np.float64); out = np.empty_like(w)
    if len(w) == 0:
        return out
    order = np.argsort(src, kind="stable"); ss = src[order]; ws = w[order].copy()
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1, [len(ss)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = ws[a:b]; tot = seg.sum()
        ws[a:b] = seg / tot if tot > 0 else 1.0 / max(b - a, 1)
    out[order] = ws
    return out


def save_arm(path, src, tgt, w_raw):
    """Write RHIZO arm .npz (src/tgt/w_outnorm/w_raw) — the exp_029/030 harness format."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64); w_raw = np.asarray(w_raw, np.float64)
    wn = outnorm_persource(src, tgt, w_raw) if len(src) else w_raw.astype(np.float32)
    np.savez(path, src=src, tgt=tgt, w_outnorm=wn.astype(np.float32), w_raw=w_raw.astype(np.float64))
    return dict(n_edges=int(len(src)), n_src=int(len(np.unique(src))) if len(src) else 0)


def load_dense(parquet_path, g2i, reg="Regulator", tgt="Target", w="Importance"):
    """Load a dense parent parquet -> (src_idx, tgt_idx, raw_w) mapped to panel index; drop self-loops/unmapped."""
    df = pd.read_parquet(parquet_path)
    cl = {c.lower(): c for c in df.columns}
    rc = cl.get(reg.lower()); tc = cl.get(tgt.lower()); wc = cl.get(w.lower(), cl.get("weight"))
    s = df[rc].map(g2i).to_numpy(); t = df[tc].map(g2i).to_numpy()
    ww = df[wc].to_numpy(dtype=np.float64)
    m = ~(pd.isna(s) | pd.isna(t)); s = s[m].astype(np.int64); t = t[m].astype(np.int64); ww = ww[m]
    keep = s != t
    return s[keep], t[keep], ww[keep]


# ------------------------------------------------------------------ pruners
def topweight_prune(src, tgt, w, E):
    """Keep top-E edges by weight (global). Ties broken by np.argpartition (deterministic given input order)."""
    if E is None or E >= len(src):
        return src.copy(), tgt.copy(), w.copy()
    keep = np.argpartition(w, -E)[-E:]
    return src[keep], tgt[keep], w[keep]


def knn_exact(src, tgt, w, E):
    """Per-source top-k out-edges with fractional fill so the result has EXACTLY E edges.
    Each source keeps its strongest edges up to a uniform integer cap k0; the residual budget is filled from
    the (k0)-th ranked edges globally by weight. Faithful 'KNN with a fractional per-node cap'."""
    n = len(src)
    if E is None or E >= n:
        return src.copy(), tgt.copy(), w.copy()
    order = np.lexsort((-w, src)); s = src[order]; t = tgt[order]; ww = w[order]
    change = np.r_[True, s[1:] != s[:-1]]; grp = np.where(change)[0]
    seglen = np.diff(np.r_[grp, len(s)])
    within = (np.arange(len(s)) - np.repeat(grp, seglen)).astype(np.int64)   # 0-based rank within source
    # count(k) = # edges with within < k  (== sum_i min(k, outdeg_i)); monotone increasing
    hist = np.bincount(within)
    cum = np.concatenate(([0], np.cumsum(hist)))     # cum[k] = count(k)
    k0 = int(np.searchsorted(cum, E, side="right") - 1)  # largest k with cum[k] <= E
    k0 = max(k0, 0)
    keep = within < k0
    base = int(keep.sum()); rem = E - base
    if rem > 0:
        tie = np.where(within == k0)[0]
        sel = tie[np.argsort(-ww[tie], kind="stable")[:rem]]
        keep[sel] = True
    return s[keep], t[keep], ww[keep]


# ------------------------------------------------------------------ fusion (Super)
def _edge_key(s, t, N): return s.astype(np.int64) * N + t.astype(np.int64)


def rank_normalize(w):
    if len(w) == 0: return w
    return rankdata(w, method="average") / len(w)


def borda_fuse(edgesA, edgesB, N, cap):
    """Rank-consensus fusion over the UNION of A,B edges (each parent rank-normalized; absent edge -> 0),
    then keep top-`cap` by fused score. Returns (src,tgt,fused_score)."""
    sA, tA, wA = edgesA; sB, tB, wB = edgesB
    kA = _edge_key(np.asarray(sA), np.asarray(tA), N); kB = _edge_key(np.asarray(sB), np.asarray(tB), N)
    rA = rank_normalize(np.asarray(wA, np.float64)); rB = rank_normalize(np.asarray(wB, np.float64))
    uk = np.union1d(kA, kB)
    scoreA = np.zeros(len(uk)); scoreB = np.zeros(len(uk))
    oA = np.argsort(kA); kAs = kA[oA]; rAs = rA[oA]
    pA = np.clip(np.searchsorted(kAs, uk), 0, len(kAs) - 1); hA = kAs[pA] == uk; scoreA[hA] = rAs[pA[hA]]
    oB = np.argsort(kB); kBs = kB[oB]; rBs = rB[oB]
    pB = np.clip(np.searchsorted(kBs, uk), 0, len(kBs) - 1); hB = kBs[pB] == uk; scoreB[hB] = rBs[pB[hB]]
    fused = 0.5 * (scoreA + scoreB)
    if cap is not None and cap < len(uk):
        top = np.argpartition(fused, -cap)[-cap:]
    else:
        top = np.arange(len(uk))
    us = (uk[top] // N).astype(np.int64); ut = (uk[top] % N).astype(np.int64)
    return us, ut, fused[top]


# ------------------------------------------------------------------ Arm E controls
def reverse(src, tgt, w):
    """Null / mechanism: reverse every edge direction (keeps degree sequences swapped)."""
    return tgt.copy(), src.copy(), w.copy()


def wstrip(src, tgt, w):
    """Mechanism: binarize edge weights (does RHIZO use fine weights?). w_raw := 1 for every kept edge."""
    return src.copy(), tgt.copy(), np.ones(len(src), np.float64)


def scramble_ms(src, tgt, w, seed=0, n_swaps_mult=10, batch=20000, max_passes=400):
    """Degree-preserving regulator-scramble via directed double-edge swaps (Maslov-Sneppen 2002).
    Swap (a->b),(c->d) -> (a->d),(c->b): preserves BOTH out-degree (a,c) and in-degree (b,d). Rejects
    self-loops and multi-edges. Weights travel with their ORIGINAL source slot (out-degree + per-source weight
    multiset preserved; only the target wiring is randomized). ~n_swaps_mult*E accepted swaps target."""
    rng = np.random.default_rng(seed)
    s = src.astype(np.int64).copy(); t = tgt.astype(np.int64).copy(); w = np.asarray(w, np.float64).copy()
    E = len(s)
    if E < 2:
        return s, t, w
    N = int(max(s.max(), t.max())) + 1
    eset = set((int(a) * N + int(b)) for a, b in zip(s, t))   # fast duplicate lookup on (src,tgt)
    target_swaps = n_swaps_mult * E
    done = 0; passes = 0
    while done < target_swaps and passes < max_passes:
        passes += 1
        i = rng.integers(0, E, size=batch); j = rng.integers(0, E, size=batch)
        for a, b in zip(i, j):
            if a == b:
                continue
            sa, ta = s[a], t[a]; sc, tc = s[b], t[b]
            if sa == tc or sc == ta:      # would create self-loop
                continue
            k1 = sa * N + tc; k2 = sc * N + ta
            if k1 in eset or k2 in eset:  # would create duplicate
                continue
            # perform swap: a keeps source sa, gets target tc; b keeps source sc, gets target ta
            eset.discard(sa * N + ta); eset.discard(sc * N + tc)
            t[a] = tc; t[b] = ta
            eset.add(k1); eset.add(k2)
            done += 1
            if done >= target_swaps:
                break
    return s, t, w, dict(accepted_swaps=int(done), passes=int(passes), target=int(target_swaps))


# ------------------------------------------------------------------ coverage
def coverage_report(src, tgt, train_pidx, held_pidx):
    srcs = set(np.unique(src).tolist()) if len(src) else set()
    tr = set(np.asarray(train_pidx).tolist()); ho = set(np.asarray(held_pidx).tolist())
    return dict(
        n_edges=int(len(src)), n_src=int(len(srcs)),
        held_out_coverage=round(len(srcs & ho) / max(len(ho), 1), 4),
        train_coverage=round(len(srcs & tr) / max(len(tr), 1), 4),
        n_src_in_train=int(len(srcs & tr)), n_src_in_heldout=int(len(srcs & ho)))


# ------------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(0)
    N = 50
    # random directed multigraph-free edge set
    pairs = set()
    while len(pairs) < 600:
        a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
        if a != b: pairs.add((a, b))
    s = np.array([p[0] for p in pairs]); t = np.array([p[1] for p in pairs])
    w = rng.random(len(s)) + 0.01
    ok = True

    # topweight exact count + really the top ones
    for E in (100, 250, len(s)):
        ts, tt, tw = topweight_prune(s, t, w, E)
        assert len(ts) == min(E, len(s)), f"topweight count {len(ts)} != {E}"
        if E < len(s):
            assert tw.min() >= np.sort(w)[-E] - 1e-12, "topweight not the largest"
    print("  topweight_prune: exact count + top-weight OK")

    # knn_exact hits EXACT E and respects a per-source cap
    for E in (60, 137, 300, len(s)):
        ks, kt, kw = knn_exact(s, t, w, E)
        assert len(ks) == min(E, len(s)), f"knn count {len(ks)} != {E}"
        # each source's kept edges are a top-prefix of its sorted-by-weight out-edges
        if E < len(s):
            for src_i in np.unique(ks):
                all_w = np.sort(w[s == src_i])[::-1]
                kept_w = np.sort(kw[ks == src_i])[::-1]
                assert np.all(kept_w <= all_w[:len(kept_w)] + 1e-9), "knn not a per-source top prefix"
    print("  knn_exact: exact E + per-source top-prefix OK")

    # borda fuse cap + union semantics
    s2, t2, w2 = knn_exact(s, t, w, 200)
    us, ut, uf = borda_fuse((s, t, w), (s2, t2, w2), N, cap=300)
    assert len(us) == 300 and len(np.unique(_edge_key(us, ut, N))) == 300, "borda cap/uniqueness"
    print("  borda_fuse: cap + unique edges OK")

    # scramble preserves out-degree and in-degree, changes wiring, no self-loop/dup
    ss, st, sw, info = scramble_ms(s, t, w, seed=1, n_swaps_mult=5)
    od0 = np.bincount(s, minlength=N); od1 = np.bincount(ss, minlength=N)
    idg0 = np.bincount(t, minlength=N); idg1 = np.bincount(st, minlength=N)
    assert np.array_equal(od0, od1), "scramble broke out-degree"
    assert np.array_equal(idg0, idg1), "scramble broke in-degree"
    assert (ss == st).sum() == 0, "scramble made self-loops"
    keys = _edge_key(ss, st, N); assert len(np.unique(keys)) == len(keys), "scramble made duplicates"
    changed = np.mean(t != st)
    print(f"  scramble_ms: out/in-degree preserved, no self-loop/dup, {changed:.0%} targets rewired "
          f"({info['accepted_swaps']} swaps) OK")

    # reverse / wstrip
    rs, rt, rw = reverse(s, t, w); assert np.array_equal(rs, t) and np.array_equal(rt, s)
    _, _, bw = wstrip(s, t, w); assert np.all(bw == 1)
    print("  reverse / wstrip OK")

    # save_arm round-trip + outnorm row-stochastic
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "_gt_selftest_arm.npz")
    info = save_arm(p, s, t, w)
    with np.load(p) as d:
        assert set(d.keys()) == {"src", "tgt", "w_outnorm", "w_raw"}, "arm keys"
        dsrc = d["src"].copy(); dwn = d["w_outnorm"].copy()
    for src_i in np.unique(dsrc):
        tot = dwn[dsrc == src_i].sum()
        assert abs(tot - 1.0) < 1e-5, f"outnorm not row-stochastic for src {src_i}: {tot}"
    try:
        os.remove(p)
    except OSError:
        pass
    print("  save_arm: 4-key npz + row-stochastic w_outnorm OK")

    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
