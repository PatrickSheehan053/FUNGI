"""
obj_009.1 — node_context.py : the graph-derived per-node context vector c_g ∈ R^{N×6} for V-FILM.

c_g conditions the SHARED readout multiplicatively (gamma = 1 + MLP(c_g), NO additive beta), restoring
per-gene adaptivity WITHOUT any identity parameter (invariant #2). Computed identically for every arm from the
swapped graph alone. The 6 features (config film.context_dim=6):
    0  z(log1p indeg)         in-degree (edge count)
    1  z(log1p outdeg)        out-degree
    2  z(pagerank)            directed, raw-weighted PageRank
    3  clustering             undirected local clustering coefficient in [0,1]
    4  z(log1p comm_size)     size of the node's undirected-Louvain community
    5  reciprocity_frac       |out_nbrs ∩ in_nbrs| / |out_nbrs ∪ in_nbrs| in [0,1]

*Why this differentiates FUNGI from greedy:* FUNGI's context vectors are structured (real degree/community/
reciprocity distribution) whereas greedy's are degenerate, so the same gamma-MLP extracts more per-gene signal
on FUNGI. Verified by the V8 context-shuffle control (permuting c_g across genes must COLLAPSE the FiLM gain).

zero-preservation is unaffected: FiLM multiplies f_g, which is 0 for any gene no walk reaches -> gamma⊙0 = 0.
CPU-only; cached per arm. Louvain partition is shared with edge_features.py (same helper+seed).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import topo_common as TC

CONTEXT_DIM = 6


def compute_node_context(src, tgt, w_raw, N, louvain_seed=0, comm_labels=None):
    """Return (c ∈ float32[N,6], comm_labels int[N], stats dict). Guaranteed NaN-free."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    w_raw = np.asarray(w_raw, np.float64)
    ds = TC.degree_strength(src, tgt, w_raw, N)
    pr = TC.pagerank_directed(src, tgt, w_raw, N)
    clu = TC.clustering_undirected(src, tgt, w_raw, N)
    recip = TC.reciprocity_per_node(src, tgt, N)
    if comm_labels is None:
        comm_labels = TC.louvain_labels(src, tgt, w_raw, N, seed=louvain_seed)
    _, inv, counts = np.unique(comm_labels, return_inverse=True, return_counts=True)
    comm_size = counts[inv].astype(np.float64)

    c0 = TC._zscore(np.log1p(ds["indeg"]))
    c1 = TC._zscore(np.log1p(ds["outdeg"]))
    c2 = TC._zscore(pr)
    c3 = clu.astype(np.float32)
    c4 = TC._zscore(np.log1p(comm_size))
    c5 = recip.astype(np.float32)
    c = np.stack([c0, c1, c2, c3, c4, c5], axis=1).astype(np.float32)
    assert np.isfinite(c).all(), "node context has non-finite values"
    stats = dict(n_nodes=int(N), n_isolated=int(((ds["indeg"] == 0) & (ds["outdeg"] == 0)).sum()),
                 n_communities=int(len(counts)), mean_clustering=float(clu.mean()),
                 mean_reciprocity=float(recip.mean()))
    return c, comm_labels, stats


def cache_path(cache_dir, arm):
    return os.path.join(cache_dir, f"nodectx__{arm}.npz")


def load_or_compute(cache_dir, arm, src, tgt, w_raw, N, louvain_seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, arm)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return d["c"].astype(np.float32), d["comm"].astype(np.int64), dict(d["stats"].item())
    c, comm, stats = compute_node_context(src, tgt, w_raw, N, louvain_seed)
    np.savez_compressed(p, c=c, comm=comm, stats=np.array(stats, dtype=object))
    return c, comm, stats


if __name__ == "__main__":
    import graphs_v2 as G
    D = G.load_data()
    for arm in ["fungi_bio", "top_weight", "shuffle", "empty"]:
        s, t, wn, wr = G.build_arm(arm, D["g2i"], D["N"])
        c, comm, st = compute_node_context(s, t, wr, D["N"])
        print(f"{arm:12s} n_iso={st['n_isolated']:>5d} n_comm={st['n_communities']:>5d} "
              f"mean_clu={st['mean_clustering']:.4f} mean_recip={st['mean_reciprocity']:.4f} "
              f"c_range=[{c.min():+.2f},{c.max():+.2f}]")
