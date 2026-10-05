"""
obj_009.1 — graphs_v2.py : clone of v1 graphs.py (SAME 8 arms, SAME out-normalization) extended to also
return the RAW edge weight (needed by the V-EDGE strength features) and to expose the directed-BFS-distance +
undirected-Laplacian primitives the P2 topology metrics consume. CPU-only.

build_arm returns (src, tgt, w_outnorm, w_raw) — the first three are byte-identical to v1's build_arm output
(so P1 is a true 1-to-1 reproduction); w_raw is the pre-normalization weight for out/in-strength features.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "clone"))

# HPC_sbatch_strengthen (SELF-CONTAINED): all data is BUNDLED under ../data (no thesis-root exp_024 tree needed).
# obj_009_data.npz = the v1 data cache (verbatim copy). arm_graphs/<arm>.npz = the per-arm (src,tgt,w_outnorm,
# w_raw) precomputed from exp_024 on the build box, so build_arm below reads them offline (no parquet/graph_io
# dependency at run time). The identities are IDENTICAL to obj_009.1/obj_009.2 build_arm output (verified).
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
ARM_GRAPHS = os.path.join(DATA, "arm_graphs")
V1_DATA = os.path.join(DATA, "obj_009_data.npz")
EPS = 1e-8

ARMS = ["fungi_bio", "top_weight", "shuffle", "knn", "mst", "empty", "reverse", "labelperm"]


def load_data():
    """Reuse the v1 data cache VERBATIM (do not rebuild). Mirrors ablation.load_data."""
    d = np.load(V1_DATA, allow_pickle=True)
    panel = [str(x) for x in d["panel_genes"]]
    return dict(panel=panel, N=len(panel), mu_ctrl=d["mu_ctrl"], signal_mask=d["signal_mask"].astype(bool),
                pidx=d["pidx"].astype(np.int64), s_full=d["s_full"], s_A=d["s_A"], s_B=d["s_B"],
                var_real=d["var_real"], g2i={g: i for i, g in enumerate(panel)})


def build_arm(arm, g2i, N):
    """Return (src, tgt, w_outnorm, w_raw) from the BUNDLED per-arm graph (data/arm_graphs/<arm>.npz).
    Byte-identical to obj_009.1/obj_009.2 build_arm output — precomputed offline on the build box so the
    HPC package needs no exp_024 parquet tree / graph_io / pyarrow."""
    p = os.path.join(ARM_GRAPHS, f"{arm}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"bundled arm graph not found: {p} (arm '{arm}'). Rebuild via build_bundle on the build box.")
    d = np.load(p)
    return (d["src"].astype(np.int64), d["tgt"].astype(np.int64),
            d["w_outnorm"].astype(np.float32), d["w_raw"].astype(np.float64))


def a_csc(src, tgt, N):
    """CSC of the raw adjacency A[t,s]; column s -> out-neighbors (targets) of source s (for DNSA)."""
    if len(src) == 0:
        return sp.csc_matrix((N, N))
    return sp.coo_matrix((np.ones(len(src)), (tgt, src)), shape=(N, N)).tocsc()


# ----------------------------------------------------------------- P2 structural primitives
def bfs_directed_dist(src, tgt, N, sources, cap=6):
    """Directed shortest-path HOP distance from each gene index in `sources` to all N nodes, capped at `cap`.
    Returns (int16[len(sources), N] with UNREACHABLE=cap+1, sources array). Edge s->t means from s reach t."""
    from scipy.sparse.csgraph import shortest_path
    sources = np.asarray(sources, np.int64)
    if len(src) == 0:
        D = np.full((len(sources), N), cap + 1, np.int16)
        for i, srow in enumerate(sources):
            D[i, srow] = 0
        return D, sources
    A = sp.coo_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N)).tocsr()  # A[s,t]=1 for edge s->t
    dist = shortest_path(A, method="D", unweighted=True, indices=sources)     # (len(sources), N), inf if unreach
    dist[~np.isfinite(dist)] = cap + 1
    dist = np.clip(dist, 0, cap + 1).astype(np.int16)
    return dist, sources


def undirected_laplacian_pinv(src, tgt, w_raw, N):
    """Dense pseudo-inverse of the undirected weighted Laplacian (conductance = summed reciprocal weight) +
    connected-component labels. Used for the effective-resistance diagnostic. O(N^3) dense pinv, CPU cache
    phase only. Returns (Lpinv float64[N,N], comp int[N])."""
    from scipy.sparse.csgraph import connected_components
    if len(src) == 0:
        return np.zeros((N, N), np.float64), np.arange(N)
    # symmetric conductance matrix
    W = sp.coo_matrix((np.asarray(w_raw, np.float64), (src, tgt)), shape=(N, N)).tocsr()
    W = W + W.T
    W.setdiag(0); W.eliminate_zeros()
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(deg) - W
    n_comp, comp = connected_components(W, directed=False)
    Lpinv = np.linalg.pinv(L.toarray().astype(np.float64), hermitian=True)
    return Lpinv, comp


def eff_resistance(Lpinv, comp, a, b):
    """Effective resistance R(a,b) = L+[a,a] + L+[b,b] - 2 L+[a,b]; NaN if a,b in different components."""
    if comp[a] != comp[b]:
        return np.nan
    return float(Lpinv[a, a] + Lpinv[b, b] - 2.0 * Lpinv[a, b])


def arm_stats(arm, g2i, N):
    s, t, wn, wr = build_arm(arm, g2i, N)
    return dict(arm=arm, n_edges=int(len(s)), n_sources=int(len(np.unique(s))) if len(s) else 0)


if __name__ == "__main__":
    D = load_data()
    print(f"panel N={D['N']} n_signal={int(D['signal_mask'].sum())} n_source={(D['pidx']>=0).sum()}")
    for a in ARMS:
        print(arm_stats(a, D["g2i"], D["N"]))
