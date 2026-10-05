"""
obj_009.3 (RHIZO-final) — edge_features_v3f.py : the `directed_topo` (a.k.a. `regulatory`) per-edge feature
profile = the confirmed CLEAN topology columns + the real directed higher-order topology columns
(compute_topo_features.py), with the attribution-confirmed NOISE columns DROPPED.

Profiles:
  clean         (6-dim, inherited from obj_009.2 edge_features_v3): [is_recip, w_recip, cross_community,
                hub2hub, recip_frac_src, recip_frac_tgt]. The A-baseline of the HPO A/B.
  directed_topo (9-dim): [is_recip, w_recip, cross_community]                              (KEEP — the confirmed
                        + [ffl_fwd, ffl_fanout, cyc3, part_src, part_tgt, dash_recomp]      #1/#2 drivers + the
                                                                                            new anti-kNN weapon)
                DROPS hub2hub / recip_frac_src / recip_frac_tgt (5090 attribution: drop≈0 / negative -> noise).
  `regulatory`  = ALIAS of directed_topo (the name used in configs/hpo_v3f.yaml's edge_profile A/B).

Every directed_topo column is a deterministic function of the arm's OWN directed graph, computed identically for
all arms (the graph-swap-fairness invariant). NO external/identity priors, NO FUNGI stored pruning scores. The
new columns come from the per-arm topofeat cache; each is credited only if it passes V7 (verify_v3f).

`with_dash=False` -> the 8-dim topology-only variant (drops dash_recomp, the flagged-redundant column). The dim
is FIXED across all arms in a run (set once by the profile), so the model's edge_dim is consistent.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import edge_features_v3 as EF3   # clone: compute_clean / degree_only / topo_only (+ EF_full)

# the 3 CLEAN columns we KEEP (attribution: is_recip #1, cross_community #2, w_recip load-bearing)
KEEP_CLEAN = ["is_recip", "w_recip", "cross_community"]
# clean φ_e column order (from edge_features_v3.compute_clean)
CLEAN_ORDER = ["is_recip", "w_recip", "cross_community", "hub2hub", "recip_frac_src", "recip_frac_tgt"]
KEEP_CLEAN_IDX = [CLEAN_ORDER.index(c) for c in KEEP_CLEAN]
# the NEW directed-topology columns (from compute_topo_features.py), in fixed order
TOPO_NEW = ["ffl_fwd", "ffl_fanout", "cyc3", "part_src", "part_tgt"]
DASH_NEW = ["dash_recomp"]

DIRECTED_TOPO_COLS = KEEP_CLEAN + TOPO_NEW + DASH_NEW           # 9-dim (with dash)
DIRECTED_TOPO_COLS_NODASH = KEEP_CLEAN + TOPO_NEW              # 8-dim
# the columns that are NEW in v3f (V7-tested per feature; the clean 3 were already confirmed on the 5090)
NEW_FEATURE_COLS = TOPO_NEW + DASH_NEW


def profile_dim(profile, with_dash=True):
    if profile in ("directed_topo", "regulatory"):
        return len(DIRECTED_TOPO_COLS) if with_dash else len(DIRECTED_TOPO_COLS_NODASH)
    return EF3.profile_dim(profile)


def cols_for(profile, with_dash=True):
    if profile in ("directed_topo", "regulatory"):
        return list(DIRECTED_TOPO_COLS if with_dash else DIRECTED_TOPO_COLS_NODASH)
    if profile == "clean":
        return list(CLEAN_ORDER)
    return None


def assemble_directed_topo(clean6, topo_cache, with_dash=True):
    """Stack the 3 KEPT clean columns + the new directed-topology columns from the per-arm topofeat cache.
       clean6 : float32[E,6] from edge_features_v3.compute_clean (order = CLEAN_ORDER).
       topo_cache : dict-like (np.load of topofeat__<arm>.npz) with the TOPO_NEW (+DASH_NEW) columns [E]."""
    E = clean6.shape[0]
    cols = [clean6[:, i].astype(np.float32) for i in KEEP_CLEAN_IDX]
    names = list(TOPO_NEW) + (list(DASH_NEW) if with_dash else [])
    for nm in names:
        v = np.asarray(topo_cache[nm], np.float32) if nm in getattr(topo_cache, "files", topo_cache) else None
        if v is None or v.shape[0] != E:
            raise ValueError(f"directed_topo column {nm!r} missing/mismatched (E={E}, got "
                             f"{None if v is None else v.shape}); run compute_topo_features.py for this arm")
        cols.append(v)
    phi = np.stack(cols, axis=1).astype(np.float32)
    return phi


if __name__ == "__main__":
    import graphs_v3 as G
    D = G.load_data()
    for arm in ["fungi_bio", "top_weight", "knn"]:
        s, t, wn, wr = G.build_arm(arm, D["g2i"], D["N"])
        comm = None
        try:
            comm = G._cached_comm(arm)
        except Exception:
            pass
        clean6, _ = EF3.compute_clean(s, t, wn, wr, D["N"], comm_labels=comm)
        tc = np.load(os.path.join(G.CACHE, f"topofeat__{arm}.npz"), allow_pickle=True)
        phi = assemble_directed_topo(clean6, tc, with_dash=True)
        print(f"{arm:11s} directed_topo dim={phi.shape[1]} E={phi.shape[0]} cols={DIRECTED_TOPO_COLS}")
