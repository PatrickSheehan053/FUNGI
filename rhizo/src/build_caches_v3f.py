"""
obj_009.3 (RHIZO-final) — build_caches_v3f.py : the ONE cache-prep entry point the HPC operator runs on the
(CPU) build/login node BEFORE the GPU sbatch. Idempotent/resumable — every builder skips a cache that exists,
so re-running fills only the gaps (the bundled base arms already have most caches).

Builds, per arm (default = the gauntlet + null arms):
  nodectx__<arm>.npz        node context c_g (FiLM) + shared Louvain communities            (clone/node_context)
  edgefeat__<arm>.npz       the obj_009.1 full 8-dim φ_e (needed by clean/topo derivations)  (clone/edge_features)
  bfsdist__<arm>.npz        directed BFS-hop distance (dsRSC/dnsa_ge2hop P2 metrics)          (clone/graphs_v2)
  effres__<arm>.npz         effective-resistance diagnostic (only the close-comparison arms)  (clone/metrics_v2)
  oversquash_w__<arm>.npz   per-node inverse-eff-resistance over-squash weight (the ovsq readout)  (clone/build_caches)
  topofeat__<arm>.npz       the REAL directed higher-order topology columns                  (src/compute_topo_features)
  edgefeat_clean__<arm>.npz + edgefeat_directed_topo__<arm>.npz  the assembled φ_e profiles  (src/graphs_v3f)

The O(N^3) dense-Laplacian pinv (oversquash + effres) is the one-time "CPU spike" — done HERE, never on the GPU
node (respects the no-CPU||GPU rule). All caches are GRAPH-DERIVED + train-safe (no perturbation responses).

  python src/build_caches_v3f.py                 # all gauntlet + null arms
  python src/build_caches_v3f.py --arms fungi_bio,top_weight        # a subset (e.g. the local smoke)
  python src/build_caches_v3f.py --skip-dash     # topology φ_e without the flagged-redundant dash_recomp col
"""
from __future__ import annotations
import os, sys, json, time, argparse

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v2 as G2
import edge_features as EF
import node_context as NC
import graph_caches as GC
import build_caches as BC            # clone: oversquash_w + clean/dashmotif φ_e
import compute_topo_features as CTF
import graphs_v3f as G3F

CACHE = GC.CACHE
# exp_029 RPE1 cohort (labelperm dropped — not in the exp_025 arm set).
ALL_ARMS = ["fungi_bio", "top_weight", "knn", "mst", "hyphae_fungi", "ptf_borda", "borda", "shroom",
            "shuffle", "reverse", "empty"]
# exp_029 wave-2 hook: `export RHIZO_EXTRA_ARMS=biologic_mid,synth_mid,...` to cache extra arms too.
ALL_ARMS = ALL_ARMS + [a.strip() for a in os.environ.get("RHIZO_EXTRA_ARMS", "").split(",")
                       if a.strip() and a.strip() not in ALL_ARMS]
EFFRES_ARMS = {"fungi_bio", "top_weight", "ptf_borda", "shuffle", "empty"}   # close-comparison arms (effres is O(N^3))


def log(m): print(f"[caches_v3f {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--skip-dash", action="store_true")
    ap.add_argument("--no-effres", action="store_true")
    ap.add_argument("--clean-only", action="store_true",
                    help="exp_030: build ONLY the clean-profile caches the light screen consumes "
                         "(nodectx + base φ_e + bfs + oversquash_w + clean φ_e); skip the directed_topo/topofeat "
                         "build (unused at edge_profile=clean). Big speedup.")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    D = G2.load_data(); N = D["N"]
    src_ids, _ = GC.sources_and_rowmap(D)
    arms = args.arms.split(",")
    summary = []
    for arm in arms:
        t0 = time.time()
        s, t, wn, wr = G2.build_arm(arm, D["g2i"], N)
        # 1) node context + shared Louvain, then full φ_e (idempotent)
        c, comm, _ = NC.load_or_compute(CACHE, arm, s, t, wr, N, 0)
        EF.load_or_compute(CACHE, arm, s, t, wn, wr, N, 0, comm_labels=comm)
        # 2) BFS distance (P2 metrics) for every gauntlet/null arm
        GC.build_bfs(CACHE, arm, s, t, N, src_ids)
        # 3) effective-resistance diagnostic (only the close-comparison arms unless overridden)
        if not args.no_effres and arm in EFFRES_ARMS:
            GC.build_effres(CACHE, arm, s, t, wr, N, D)
        # 4) over-squash per-node weight (the ovsq readout) — needs the dense pinv
        BC.build_oversquash_w(arm, D)
        # 5) real directed-topology columns + 6) assembled φ_e profiles
        CTF.build_arm_topo  # (ensure import)
        G3F.load_edge_feat_profile(arm, "clean", D)                   # the clean φ_e the light screen consumes
        if not args.clean_only:
            G3F.load_edge_feat_profile(arm, "directed_topo", D, with_dash=not args.skip_dash)
        secs = round(time.time() - t0, 1)
        summary.append(dict(arm=arm, n_edges=int(len(s)), effres=(arm in EFFRES_ARMS and not args.no_effres),
                            secs=secs))
        log(f"{arm:11s} E={len(s):>7d} caches ready (effres={arm in EFFRES_ARMS}) ({secs}s)")
    json.dump(summary, open(os.path.join(CACHE, "build_caches_v3f_summary.json"), "w"), indent=2, default=float)
    log(f"DONE all caches -> {CACHE}")


if __name__ == "__main__":
    main()
