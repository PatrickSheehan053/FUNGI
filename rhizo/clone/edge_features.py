"""
obj_009.1 — edge_features.py : the graph-derived per-edge feature vector phi_e ∈ R^{E×8} for V-EDGE.

phi_e replaces the scalar out-normalized weight in the DNPN message. It is computed identically for EVERY arm
by the SAME extractor from the swapped graph alone (graph-agnostic; no gene identity). The 8 features (config
edge_feat.dim=8):
    0  w_outnorm            the out-normalized edge weight (what v1 used as the scalar w)
    1  is_reciprocated      1 if the reverse edge (t->s) exists, else 0
    2  w_reciprocal         w_outnorm of the reverse edge (t->s), else 0
    3  z(log1p outdeg_s)    source out-degree (edge count), population z-scored over this arm's edges
    4  z(log1p indeg_t)     target in-degree
    5  z(log1p out_str_s)   source out-STRENGTH (raw weight sum) — carries scale lost by out-normalization
    6  z(log1p in_str_t)    target in-strength
    7  cross_community      1 if s,t are in DIFFERENT undirected-Louvain communities (an SCBER-bridge analog)
The hub-to-hub signal is the (feature3 × feature4) interaction the bias-free edge-MLP learns (kept dim=8).

*Why this differentiates FUNGI from greedy top_weight:* greedy retains only the strongest DIRECT out-edges, so
its phi_e is degenerate — near-zero reciprocity, no cross-community bridges — whereas FUNGI's pruning enforces
reciprocity/hub/bridge structure, so the SAME extractor yields a richer phi_e on FUNGI (verified: V7 shuffle
control + the arm-discrimination unit check in __main__).

CPU-only; cached per arm to intermediate/edge_feat_cache/. Louvain communities are shared with node_context.py
(same topo_common helper, same seed) so the cross-community bit and the community-size context agree.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import topo_common as TC

EDGE_FEAT_DIM = 8


def compute_edge_features(src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None):
    """Return (phi ∈ float32[E,8], stats dict). comm_labels (int[N]) may be passed to REUSE node_context's
    Louvain partition; if None it is computed here with the same helper+seed (identical result)."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    w_outnorm = np.asarray(w_outnorm, np.float64); w_raw = np.asarray(w_raw, np.float64)
    E = src.size
    if E == 0:
        return np.zeros((0, EDGE_FEAT_DIM), np.float32), dict(n_edges=0, frac_recip=0.0, frac_cross=0.0)

    ds = TC.degree_strength(src, tgt, w_raw, N)
    is_recip, w_recip = TC.reciprocity_per_edge(src, tgt, w_outnorm, N)
    if comm_labels is None:
        comm_labels = TC.louvain_labels(src, tgt, w_raw, N, seed=louvain_seed)
    cross = (comm_labels[src] != comm_labels[tgt]).astype(np.float32)

    f0 = w_outnorm.astype(np.float32)
    f3 = TC._zscore(np.log1p(ds["outdeg"][src]))
    f4 = TC._zscore(np.log1p(ds["indeg"][tgt]))
    f5 = TC._zscore(np.log1p(np.maximum(ds["out_str"][src], 0.0)))
    f6 = TC._zscore(np.log1p(np.maximum(ds["in_str"][tgt], 0.0)))

    phi = np.stack([f0, is_recip, w_recip, f3, f4, f5, f6, cross], axis=1).astype(np.float32)
    stats = dict(n_edges=int(E), frac_recip=float(is_recip.mean()), frac_cross=float(cross.mean()),
                 n_communities=int(len(np.unique(comm_labels))))
    return phi, stats


def cache_path(cache_dir, arm):
    return os.path.join(cache_dir, f"edgefeat__{arm}.npz")


def load_or_compute(cache_dir, arm, src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None):
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, arm)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return d["phi"].astype(np.float32), dict(d["stats"].item())
    phi, stats = compute_edge_features(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels)
    np.savez_compressed(p, phi=phi, stats=np.array(stats, dtype=object))
    return phi, stats


if __name__ == "__main__":
    # arm-discrimination sanity: greedy top_weight's reciprocity/cross-community columns must be NEAR-DEGENERATE
    # relative to FUNGI's (that is the whole premise of V-EDGE). CPU-only.
    import graphs_v2 as G
    D = G.load_data()
    for arm in ["fungi_bio", "top_weight", "shuffle", "knn", "mst"]:
        s, t, wn, wr = G.build_arm(arm, D["g2i"], D["N"])
        phi, st = compute_edge_features(s, t, wn, wr, D["N"])
        print(f"{arm:12s} E={st['n_edges']:>7d} frac_recip={st['frac_recip']:.4f} "
              f"frac_cross_comm={st['frac_cross']:.4f} n_comm={st.get('n_communities',0)}")
