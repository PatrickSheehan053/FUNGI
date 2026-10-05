"""
obj_009.3 (RHIZO-final) — compute_topo_features.py

THE load-bearing new code: real, PURELY-GRAPH-DERIVED directed higher-order topology per-edge features, computed
IDENTICALLY from each arm's OWN directed adjacency (the graph-swap-fairness invariant — see the spec's INVARIANT
LINE). These are the anti-kNN weapon: directed regulatory-motif signals that an undirected co-expression graph
(kNN) structurally cannot represent, WITHOUT any external/identity prior (no cisTarget/TF-motif NES, no GO, no
gene embeddings, no FUNGI stored pruning scores — all forbidden).

Per-edge columns built here (cached to intermediate/graph_caches/topofeat__<arm>.npz):
  ffl_fwd     # directed 2-paths s->x->t : the edge s->t as the DIRECT arm of a feed-forward loop (FFL). The
              canonical directed GRN motif; a symmetric co-expression edge cannot carry this asymmetry.
  ffl_fanout  # common OUT-targets |{z: s->z AND t->z}| : co-regulation / bi-fan participation.
  cyc3        # directed 2-paths t->x->s : the edge s->t closing a directed 3-CYCLE (feedback). Feed-forward vs
              feedback is exactly the direction signal kNN's symmetry blurs.
  part_src    # community participation coefficient of the SOURCE node (bridge role; continuous cross_community).
  part_tgt    # community participation coefficient of the TARGET node.
  dash_recomp # a DASH-kernel-style component RECOMPUTED per-arm (effective-resistance x weight x bridge): the
              out-normalized weight scaled by endpoint peripherality (1 - mean inverse-eff-res weight). Uses ONLY
              this arm's own graph (reuses the per-arm oversquash_w cache = 1/(1+r/median r), r=diag(Lpinv)) — no
              FUNGI stored scores. Flagged in the spec as a likely-REDUNDANT candidate (RHIZO already reads
              effective resistance via the over-squash readout + weight + reciprocity + community); include, but
              expect it to add little beyond FFL/directed-motif. Credited only if it passes V7.

Counts are log1p-then-population-z-scored (like the degree features) so the edge gate sees well-scaled inputs;
part_* are already in [0,1]. Every column is a deterministic function of THIS arm's graph, applied uniformly to
ALL arms. CPU-only, cache-phase (respects the no-CPU||GPU rule). Idempotent/resumable (skips existing caches).

  python src/compute_topo_features.py                       # all study arms
  python src/compute_topo_features.py --arms fungi_bio,top_weight
  python src/compute_topo_features.py --arms fungi_bio,knn,top_weight,mst --skip-dash   # topology cols only
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import scipy.sparse as sp

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3 as G          # portable build_arm / load_data / load_oversquash_w / cached comm
import topo_common as TC

CACHE = G.CACHE
# arms the final run trains: FUNGI + the gauntlet opponents (kNN first-class) + the three nulls + empty sanity
STUDY_ARMS = ["fungi_bio", "top_weight", "knn", "mst", "shuffle", "reverse", "labelperm", "empty"]

TOPO_COLS = ["ffl_fwd", "ffl_fanout", "cyc3", "part_src", "part_tgt"]   # always built (no pinv needed)
DASH_COL = "dash_recomp"                                                 # optional (reuses oversquash_w cache)
ALL_COLS = TOPO_COLS + [DASH_COL]


def log(m): print(f"[topo {time.strftime('%H:%M:%S')}] {m}", flush=True)


def cache_path(arm):
    return os.path.join(CACHE, f"topofeat__{arm}.npz")


# ---------------------------------------------------------------- community participation coefficient
def participation_coefficient(src, tgt, N, comm):
    """Per-node participation coefficient P[v] = 1 - sum_c (k_v(c)/k_v)^2 over the node's total-degree edges
    (in+out, undirected multiplicity), using the arm's OWN Louvain communities `comm`. High P = a node whose
    edges span many communities (a bridge). 0 for isolated nodes. Extends the binary cross_community column."""
    P = np.zeros(N, np.float64)
    if src.size == 0:
        return P
    # community of the OTHER endpoint, once per incident edge (out edges: neighbor=tgt; in edges: neighbor=src)
    inc_node = np.concatenate([src, tgt])
    inc_comm = np.concatenate([comm[tgt], comm[src]]).astype(np.int64)
    ncomm = int(comm.max()) + 1 if comm.size else 1
    order = np.lexsort((inc_comm, inc_node))
    node_s = inc_node[order]; comm_s = inc_comm[order]
    # segment over (node) : total degree k_v ; and over (node,comm) : k_v(c)
    ktot = np.bincount(node_s, minlength=N).astype(np.float64)
    key = node_s.astype(np.int64) * np.int64(ncomm) + comm_s
    uk, cnt = np.unique(key, return_counts=True)
    nodes_of_key = (uk // ncomm).astype(np.int64)
    frac2 = (cnt.astype(np.float64) ** 2)
    sum_frac2 = np.zeros(N, np.float64)
    np.add.at(sum_frac2, nodes_of_key, frac2)
    nz = ktot > 0
    P[nz] = 1.0 - sum_frac2[nz] / (ktot[nz] ** 2)
    return P


# ---------------------------------------------------------------- directed higher-order motif counts
def _query_sparse(M_csr, rows, cols):
    """Vectorized M[rows[i], cols[i]] for a CSR sparse matrix -> dense float array (0 where absent)."""
    if len(rows) == 0:
        return np.zeros(0, np.float64)
    v = np.asarray(M_csr[rows, cols]).ravel()
    return np.asarray(v, np.float64)


def directed_motif_counts(src, tgt, N):
    """Return (ffl_fwd, ffl_fanout, cyc3) per edge from the arm's OWN binary directed adjacency.
      ffl_fwd[e]    = #{x : s->x AND x->t}  = A2[s,t]         (edge is the FFL shortcut arm)
      ffl_fanout[e] = #{z : s->z AND t->z}  = (A @ A^T)[s,t]   (co-regulation / bi-fan)
      cyc3[e]       = #{x : t->x AND x->s}  = A2[t,s]          (directed 3-cycle / feedback)
    A is binary (dedup'd), so these count DISTINCT intermediaries. Memory-bounded: the products are sparse."""
    E = src.size
    if E == 0:
        z = np.zeros(0, np.float64)
        return z, z.copy(), z.copy()
    A = sp.coo_matrix((np.ones(E, np.float64), (src, tgt)), shape=(N, N)).tocsr()
    A.data[:] = 1.0                                  # binary (collapse any accidental multiplicity)
    A.sum_duplicates()
    A2 = (A @ A).tocsr()                             # A2[i,j] = # directed 2-paths i->.->j
    AAt = (A @ A.T).tocsr()                          # AAt[i,j] = # common out-targets of i and j
    ffl_fwd = _query_sparse(A2, src, tgt)
    cyc3 = _query_sparse(A2, tgt, src)
    ffl_fanout = _query_sparse(AAt, src, tgt)
    return ffl_fwd, ffl_fanout, cyc3


# ---------------------------------------------------------------- DASH-style recomputed component
def dash_recomputed(arm, src, tgt, w_outnorm, N):
    """A DASH-kernel-style per-edge score RECOMPUTED on THIS arm's graph: effective-resistance x weight x bridge.
    Uses ONLY the arm's own out-normalized weight and its per-node inverse-eff-resistance weight hop_w in (0,1]
    (from the oversquash_w cache; hop_w low = peripheral = high eff-resistance = bridge-like). Score high for
    high-weight edges between peripheral (bridge) endpoints. Purely graph-derived; NOT FUNGI's stored scores."""
    if src.size == 0:
        return np.zeros(0, np.float64)
    hop_w = G.load_oversquash_w(arm, N).astype(np.float64)     # per-node 1/(1+r/median r), in (0,1]
    bridge = 1.0 - 0.5 * (hop_w[src] + hop_w[tgt])             # high when both endpoints peripheral (bridge)
    return np.asarray(w_outnorm, np.float64) * np.clip(bridge, 0.0, None)


# ---------------------------------------------------------------- build one arm
def build_arm_topo(arm, D, with_dash=True):
    N = D["N"]
    s, t, wn, wr = G.build_arm(arm, D["g2i"], N)
    out = {}
    cols = ALL_COLS if with_dash else TOPO_COLS
    if s.size == 0:
        for c in cols:
            out[c] = np.zeros(0, np.float32)
        out["_raw"] = {c: np.zeros(0, np.float32) for c in cols}
        return out, dict(arm=arm, n_edges=0, ffl_fwd_mean=0.0, ffl_fanout_mean=0.0, cyc3_mean=0.0,
                         part_src_mean=0.0, dash_built=bool(with_dash))
    ffl_fwd, ffl_fanout, cyc3 = directed_motif_counts(s, t, N)
    # community participation coefficient of each endpoint (arm's own Louvain, reuse the cached partition)
    comm = None
    try:
        comm = G._cached_comm(arm)
    except Exception:
        comm = TC.louvain_labels(s, t, wr, N, seed=0)
    P = participation_coefficient(s, t, N, comm)
    raw = dict(ffl_fwd=ffl_fwd, ffl_fanout=ffl_fanout, cyc3=cyc3,
               part_src=P[s].astype(np.float64), part_tgt=P[t].astype(np.float64))
    # scale: counts -> log1p then population z-score; participation already in [0,1] (kept raw)
    out = {}
    out["ffl_fwd"] = TC._zscore(np.log1p(ffl_fwd))
    out["ffl_fanout"] = TC._zscore(np.log1p(ffl_fanout))
    out["cyc3"] = TC._zscore(np.log1p(cyc3))
    out["part_src"] = raw["part_src"].astype(np.float32)
    out["part_tgt"] = raw["part_tgt"].astype(np.float32)
    if with_dash:
        dr = dash_recomputed(arm, s, t, wn, N)
        raw[DASH_COL] = dr
        out[DASH_COL] = TC._zscore(dr)
    meta = dict(arm=arm, n_edges=int(s.size),
                ffl_fwd_mean=float(ffl_fwd.mean()), ffl_fanout_mean=float(ffl_fanout.mean()),
                cyc3_mean=float(cyc3.mean()), part_src_mean=float(raw["part_src"].mean()),
                dash_built=bool(with_dash))
    out["_raw"] = raw
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(STUDY_ARMS))
    ap.add_argument("--skip-dash", action="store_true", help="build topology cols only (no dash_recomp)")
    ap.add_argument("--force", action="store_true", help="rebuild even if a cache exists")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    D = G.load_data()
    summary = []
    for arm in args.arms.split(","):
        p = cache_path(arm)
        if os.path.exists(p) and not args.force:
            log(f"{arm:11s} cache exists -> skip")
            continue
        t0 = time.time()
        out, meta = build_arm_topo(arm, D, with_dash=not args.skip_dash)
        raw = out.pop("_raw")
        save = {c: out[c].astype(np.float32) for c in out}
        save.update({f"raw_{c}": raw[c].astype(np.float32) for c in raw})
        save["cols"] = np.array(ALL_COLS if not args.skip_dash else TOPO_COLS, dtype=object)
        np.savez_compressed(p, **save)
        meta["secs"] = round(time.time() - t0, 1)
        summary.append(meta)
        log(f"{arm:11s} E={meta['n_edges']:>7d} ffl_fwd_mu={meta.get('ffl_fwd_mean',0):.2f} "
            f"cyc3_mu={meta.get('cyc3_mean',0):.2f} dash={meta['dash_built']} ({meta['secs']}s)")
    json.dump(summary, open(os.path.join(CACHE, "topofeat_summary.json"), "w"), indent=2, default=float)
    log(f"DONE -> {os.path.join(CACHE, 'topofeat_summary.json')}")


if __name__ == "__main__":
    main()
