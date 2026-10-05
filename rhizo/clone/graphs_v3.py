"""
obj_009.2 — graphs_v3.py : re-exports obj_009.1 graphs_v2 (build_arm, load_data, a_csc, BFS-distance,
Laplacian-pinv, ARMS) unchanged, and adds the CLEAN / degree-only / topo-only φ_e cache loaders (Phase-0 +
v_edge_clean). The clean φ_e reuses the ALREADY-CACHED Louvain communities from the full nodectx cache, so no
Louvain re-run is needed (CPU-light — safe to build while the GPU is busy).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
from graphs_v2 import (  # noqa: F401  re-export unchanged
    load_data, build_arm, a_csc, bfs_directed_dist, undirected_laplacian_pinv, eff_resistance, arm_stats,
    ARMS,
)
import graph_caches as GC   # clone (full φ_e / nodectx / bfs / effres loaders)
import edge_features_v3 as EF3

CACHE = GC.CACHE   # obj_009.2/intermediate/graph_caches (full caches copied here; clean caches written here too)


def _cached_comm(arm):
    """Reuse the Louvain community labels already stored in the full nodectx cache (no re-run)."""
    d = np.load(GC.NC.cache_path(CACHE, arm), allow_pickle=True)
    return d["comm"].astype(np.int64)


def load_edge_feat_profile(arm, profile, D=None):
    """Load/build the per-arm φ_e for a given profile. 'full' uses the copied obj_009.1 cache; 'clean'/
    'degree_only'/'topo_only' are built here (reusing cached comm) and cached as edgefeat_<profile>__<arm>.npz."""
    if profile == "full":
        return GC.load_edge_feat(arm, CACHE)
    p = os.path.join(CACHE, f"edgefeat_{profile}__{arm}.npz")
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)["phi"].astype(np.float32)
    if D is None:
        D = load_data()
    s, t, wn, wr = build_arm(arm, D["g2i"], D["N"])
    comm = _cached_comm(arm) if os.path.exists(GC.NC.cache_path(CACHE, arm)) else None
    if profile == "clean":
        phi, _ = EF3.compute_clean(s, t, wn, wr, D["N"], comm_labels=comm)
    elif profile == "dashmotif":
        # clean(6) + DASH sub-score + motif NES. The two extra columns come from per-arm caches
        # (dashmotif_dash__<arm>.npz / dashmotif_motif__<arm>.npz, built by compute_dash_motif.py). If a
        # cache is ABSENT, the column is ZERO-FILLED -> the arm degrades gracefully to v_edge_clean+2-zero-cols
        # (the edge gate learns to ignore constant-zero columns), so it NEVER blocks the run.
        dash = _load_extra_col(arm, "dash", len(s))
        motif = _load_extra_col(arm, "motif", len(s))
        phi, _ = EF3.compute_clean(s, t, wn, wr, D["N"], comm_labels=comm, dash=dash, motif=motif)
    elif profile == "degree_only":
        phi = EF3.compute_degree_only(s, t, wn, wr, D["N"], comm_labels=comm)
    elif profile == "topo_only":
        phi = EF3.compute_topo_only(s, t, wn, wr, D["N"], comm_labels=comm)
    else:
        raise ValueError(profile)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(p, phi=phi.astype(np.float32))
    return phi.astype(np.float32)


def _load_extra_col(arm, which, E):
    """Per-edge DASH sub-score / motif NES column for `arm`, from its cache, else a zero column of length E."""
    p = os.path.join(CACHE, f"dashmotif_{which}__{arm}.npz")
    if os.path.exists(p):
        v = np.load(p, allow_pickle=True)["v"].astype(np.float32)
        if v.shape[0] == E:
            return v
    return np.zeros(E, np.float32)


def dashmotif_status(arm):
    """(has_dash, has_motif) for `arm` — whether the real per-edge caches exist (else the columns are zero)."""
    d = os.path.exists(os.path.join(CACHE, f"dashmotif_dash__{arm}.npz"))
    m = os.path.exists(os.path.join(CACHE, f"dashmotif_motif__{arm}.npz"))
    return d, m


def load_oversquash_w(arm, N):
    """Per-node over-squash readout weight in (0,1] (inverse effective resistance), from the bundled cache
    (oversquash_w__<arm>.npz, built by build_caches.py). Missing -> all-ones (no reweighting; ovsq becomes a
    no-op that still passes ψ(0)=0)."""
    p = os.path.join(CACHE, f"oversquash_w__{arm}.npz")
    if os.path.exists(p):
        w = np.load(p, allow_pickle=True)["w"].astype(np.float32)
        if w.shape[0] == N:
            return w
    return np.ones(N, np.float32)


# convenience passthroughs
def load_node_ctx(arm):
    return GC.load_node_ctx(arm, CACHE)


def load_bfs(arm):
    return GC.load_bfs(arm, CACHE)


def load_effres(arm):
    return GC.load_effres(arm, CACHE)
