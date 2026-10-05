"""
obj_009.2 — edge_features_v3.py : two graph-derived per-edge feature profiles.

  profile="full"   the obj_009.1 8-dim vector (w_outnorm, is_recip, w_recip, z(log1p outdeg_s), z(log1p indeg_t),
                   z(log1p out_str_s), z(log1p in_str_t), cross_community). Includes raw DEGREE/STRENGTH.
  profile="clean"  the confirmation-hardened profile: EXCLUDES raw degree/strength (the shuffle-preserved,
                   capacity-like signals) and keeps ONLY signals a degree-preserving shuffle DESTROYS and that
                   greedy top_weight structurally lacks:
                     [is_reciprocated, w_reciprocal, cross_community(Louvain bridge), hub_to_hub(bit),
                      reciprocity_frac_src, reciprocity_frac_tgt]  (+ optional dash_subscore, motif_nes)
                   The point (Phase-0 test c / the shuffle-blowup): if v_edge's win survives on `clean` φ_e AND
                   the shuffle arm collapses, the win is REAL topology, not degree/capacity.

hub_to_hub is a BINARY structural indicator (both endpoints are hubs), not the continuous degree z-score the
shuffle preserves — so it stays in `clean`. DASH sub-scores + cisTarget motif NES are optional add-on dims
(default OFF / zero-filled; wire via the FUNGI DASH kernel + obj_003.1 motif tooling when available — the gate
learns to ignore zero columns, so the clean result is valid without them).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import topo_common as TC
import edge_features as EF_full   # obj_009.1 (the `full` profile)

FULL_DIM = 8
CLEAN_DIM = 6   # is_recip, w_recip, cross_comm, hub_to_hub, recip_frac_src, recip_frac_tgt
                # (+2 optional: dash_subscore, motif_nes -> dim 8 when enabled)


def compute_clean(src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None,
                  hub_pctile=0.90, dash=None, motif=None):
    """Return (phi_clean float32[E, CLEAN_DIM(+opt)], stats). NO raw degree/strength continuous columns."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64)
    w_outnorm = np.asarray(w_outnorm, np.float64); w_raw = np.asarray(w_raw, np.float64)
    E = src.size
    extra = int(dash is not None) + int(motif is not None)
    if E == 0:
        return np.zeros((0, CLEAN_DIM + extra), np.float32), dict(n_edges=0, frac_recip=0.0, frac_cross=0.0)
    ds = TC.degree_strength(src, tgt, w_raw, N)
    is_recip, w_recip = TC.reciprocity_per_edge(src, tgt, w_outnorm, N)
    if comm_labels is None:
        comm_labels = TC.louvain_labels(src, tgt, w_raw, N, seed=louvain_seed)
    cross = (comm_labels[src] != comm_labels[tgt]).astype(np.float32)
    # hub-to-hub as a BINARY bit (degree structure, not the continuous z-score the shuffle preserves)
    out_hi = ds["outdeg"] >= np.quantile(ds["outdeg"][ds["outdeg"] > 0], hub_pctile) if (ds["outdeg"] > 0).any() else np.zeros(N, bool)
    in_hi = ds["indeg"] >= np.quantile(ds["indeg"][ds["indeg"] > 0], hub_pctile) if (ds["indeg"] > 0).any() else np.zeros(N, bool)
    hub2hub = (out_hi[src] & in_hi[tgt]).astype(np.float32)
    rfrac = TC.reciprocity_per_node(src, tgt, N)
    cols = [is_recip, w_recip, cross, hub2hub, rfrac[src].astype(np.float32), rfrac[tgt].astype(np.float32)]
    if dash is not None:
        cols.append(np.asarray(dash, np.float32))
    if motif is not None:
        cols.append(np.asarray(motif, np.float32))
    phi = np.stack(cols, axis=1).astype(np.float32)
    stats = dict(n_edges=int(E), frac_recip=float(is_recip.mean()), frac_cross=float(cross.mean()),
                 frac_hub2hub=float(hub2hub.mean()), n_communities=int(len(np.unique(comm_labels))),
                 dim=phi.shape[1])
    return phi, stats


def compute_degree_only(src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None):
    """Phase-0 test (c): φ_e = degree/strength ONLY (the shuffle-PRESERVED signals). If v_edge's gain is
    reproduced by this, the win is a degree/capacity artifact. Columns = the full profile's degree/strength z-cols."""
    phi_full, _ = EF_full.compute_edge_features(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels)
    if phi_full.shape[0] == 0:
        return np.zeros((0, 5), np.float32)
    # full cols: [0 w_outnorm, 1 is_recip, 2 w_recip, 3 z(outdeg_s), 4 z(indeg_t), 5 z(out_str), 6 z(in_str), 7 cross]
    return phi_full[:, [0, 3, 4, 5, 6]].astype(np.float32)   # weight + degree/strength only (no recip/community)


def compute_topo_only(src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None):
    """Phase-0 test (c): φ_e = topology ONLY (reciprocity/community — shuffle-DESTROYED). If the gain REQUIRES
    this set, the win is real FUNGI signal. == compute_clean without the hub_to_hub degree bit."""
    phi, _ = compute_clean(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels)
    if phi.shape[0] == 0:
        return np.zeros((0, 5), np.float32)
    return phi[:, [0, 1, 2, 4, 5]].astype(np.float32)   # is_recip, w_recip, cross, recip_frac_src/tgt (drop hub2hub)


def compute(profile, src, tgt, w_outnorm, w_raw, N, louvain_seed=0, comm_labels=None, **kw):
    if profile == "full":
        return EF_full.compute_edge_features(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels)
    if profile == "clean":
        return compute_clean(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels, **kw)
    if profile == "degree_only":
        return compute_degree_only(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels), {}
    if profile == "topo_only":
        return compute_topo_only(src, tgt, w_outnorm, w_raw, N, louvain_seed, comm_labels), {}
    raise ValueError(f"unknown edge-feature profile {profile!r}")


def profile_dim(profile, dash=False, motif=False):
    if profile == "full":
        return FULL_DIM
    if profile == "clean":
        return CLEAN_DIM + int(dash) + int(motif)
    if profile == "dashmotif":
        return CLEAN_DIM + 2   # clean(6) + DASH sub-score + motif NES (zero-filled if caches absent) = 8
    if profile in ("degree_only", "topo_only"):
        return 5
    raise ValueError(profile)


if __name__ == "__main__":
    import graphs_v2 as G
    D = G.load_data()
    for arm in ["fungi_bio", "top_weight", "shuffle"]:
        s, t, wn, wr = G.build_arm(arm, D["g2i"], D["N"])
        pc, st = compute_clean(s, t, wn, wr, D["N"])
        print(f"{arm:11s} clean dim={pc.shape[1]} frac_recip={st['frac_recip']:.4f} "
              f"frac_cross={st['frac_cross']:.4f} frac_hub2hub={st['frac_hub2hub']:.4f}")
