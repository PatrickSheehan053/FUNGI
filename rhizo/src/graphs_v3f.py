"""
obj_009.3 (RHIZO-final) — graphs_v3f.py : re-exports the portable obj_009.2 graph layer (build_arm, load_data,
BFS/Laplacian primitives, clean/degree/topo φ_e loaders, node-context + over-squash loaders) UNCHANGED, and adds
the `directed_topo` (a.k.a. `regulatory`) φ_e loader that stacks the KEPT clean columns with the real
directed higher-order topology columns from compute_topo_features.py.

kNN is a FIRST-CLASS arm here (it is in G.ARMS and has a bundled arm-graph + all caches) — the honest boss RHIZO
must beat on the causal axes (direction + residualized RSC + transfer), not on aggregate (kNN's turf).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3 as G          # clone (portable): build_arm/load_data/ARMS/BFS/Laplacian/clean+oversquash loaders
import edge_features_v3 as EF3
import edge_features_v3f as EF3F

# re-export the portable base API unchanged
from graphs_v3 import (  # noqa: F401
    load_data, build_arm, a_csc, bfs_directed_dist, undirected_laplacian_pinv, eff_resistance, arm_stats,
    load_node_ctx, load_bfs, load_effres, load_oversquash_w, dashmotif_status, ARMS,
)
CACHE = G.CACHE
GAUNTLET_ARMS = ["fungi_bio", "knn", "top_weight", "mst"]          # FUNGI vs the graph-type opponents
NULL_ARMS = ["shuffle", "reverse", "labelperm", "empty"]           # partial + clean nulls + ψ(0)=0 sanity


def _cached_comm(arm):
    return G._cached_comm(arm)


def _topofeat(arm, D, with_dash=True):
    """Load (or build on demand) the per-arm directed-topology cache from compute_topo_features.py."""
    p = os.path.join(CACHE, f"topofeat__{arm}.npz")
    if not os.path.exists(p):
        import compute_topo_features as CTF
        out, _ = CTF.build_arm_topo(arm, D, with_dash=with_dash)
        raw = out.pop("_raw")
        save = {c: out[c].astype(np.float32) for c in out}
        save.update({f"raw_{c}": raw[c].astype(np.float32) for c in raw})
        save["cols"] = np.array(CTF.ALL_COLS if with_dash else CTF.TOPO_COLS, dtype=object)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(p, **save)
    return np.load(p, allow_pickle=True)


def load_edge_feat_profile(arm, profile, D=None, with_dash=True):
    """Per-arm φ_e for a profile. 'directed_topo'/'regulatory' -> KEPT clean cols + directed-topology cols
    (cached as edgefeat_directed_topo[_nodash]__<arm>.npz). All other profiles delegate to the clone loader."""
    if profile in ("directed_topo", "regulatory"):
        tag = "directed_topo" if with_dash else "directed_topo_nodash"
        cp = os.path.join(CACHE, f"edgefeat_{tag}__{arm}.npz")
        if os.path.exists(cp):
            return np.load(cp, allow_pickle=True)["phi"].astype(np.float32)
        if D is None:
            D = load_data()
        s, t, wn, wr = build_arm(arm, D["g2i"], D["N"])
        comm = _cached_comm(arm) if os.path.exists(G.GC.NC.cache_path(CACHE, arm)) else None
        clean6, _ = EF3.compute_clean(s, t, wn, wr, D["N"], comm_labels=comm)
        if clean6.shape[0] == 0:
            dim = EF3F.profile_dim(profile, with_dash)
            phi = np.zeros((0, dim), np.float32)
        else:
            tc = _topofeat(arm, D, with_dash=with_dash)
            phi = EF3F.assemble_directed_topo(clean6, tc, with_dash=with_dash)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cp, phi=phi.astype(np.float32))
        return phi.astype(np.float32)
    return G.load_edge_feat_profile(arm, profile, D)


if __name__ == "__main__":
    D = load_data()
    for a in GAUNTLET_ARMS + NULL_ARMS:
        phi = load_edge_feat_profile(a, "directed_topo", D)
        print(f"{a:11s} directed_topo phi shape={phi.shape}")
