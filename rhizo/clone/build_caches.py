"""
obj_009.2 HPC_sbatch_strengthen — build_caches.py

Build (on the build box; then BUNDLE into intermediate/graph_caches/) the caches the strengthen study needs
beyond the ones inherited from obj_009.2:

  1. oversquash_w__<arm>.npz  — the per-node OVER-SQUASH readout weight w[n] in (0,1] (ARM 3). w[n] is a
     graph-intrinsic INVERSE-EFFECTIVE-RESISTANCE weight: r[n] = diag of the undirected Laplacian
     pseudo-inverse (a node's effective resistance to the graph; high = peripheral = over-squashed), and
     w[n] = 1/(1 + r[n]/median(r)). Peripheral/over-squashed nodes -> low weight. LEAKAGE-SAFE: derived from
     the graph ONLY (no perturbation responses), identical for train/val. O(N^3) dense pinv = the one-time
     "CPU spike"; done here, not on the GPU node.

  2. edgefeat_clean__<arm>.npz + edgefeat_dashmotif__<arm>.npz — precompute the clean / dashmotif φ_e for every
     training arm so the GPU node never rebuilds them (dashmotif columns come from compute_dash_motif.py; zero
     if absent).

Run:  python src/build_caches.py            # all arms
      python src/build_caches.py --arms fungi_bio,top_weight
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3 as G
from graphs_v2 import undirected_laplacian_pinv

CACHE = G.CACHE
# arms the study trains (fungi + honest opponent + the three nulls run every arm + empty sanity)
STUDY_ARMS = ["fungi_bio", "top_weight", "shuffle", "reverse", "labelperm", "empty"]


def log(m): print(f"[build_caches {time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_oversquash_w(arm, D):
    """Per-node inverse-effective-resistance weight (ARM 3). Graph-only -> leakage-safe."""
    p = os.path.join(CACHE, f"oversquash_w__{arm}.npz")
    N = D["N"]
    s, t, wn, wr = G.build_arm(arm, D["g2i"], N)
    if len(s) == 0:
        w = np.ones(N, np.float32)                       # empty graph -> neutral (arm is the ψ(0)=0 sanity)
        np.savez_compressed(p, w=w, note="empty graph -> all-ones (no reweighting)")
        return w, dict(arm=arm, n_edges=0, w_mean=1.0)
    t0 = time.time()
    Lpinv, comp = undirected_laplacian_pinv(s, t, wr, N)
    r = np.clip(np.diag(Lpinv).astype(np.float64), 0.0, None)   # effective-resistance-to-graph per node
    reached = r > 0
    scale = np.median(r[reached]) if reached.any() else 1.0
    scale = scale if scale > 1e-12 else 1.0
    w = (1.0 / (1.0 + r / scale)).astype(np.float32)     # inverse eff-res; high r (peripheral) -> low w
    w[~reached] = 1.0                                     # isolated nodes -> neutral (never reached anyway)
    np.savez_compressed(p, w=w, r=r.astype(np.float32), scale=float(scale),
                        note="w[n]=1/(1+r[n]/median(r)); r=diag(Lpinv undirected)")
    return w, dict(arm=arm, n_edges=int(len(s)), w_mean=float(w.mean()), w_min=float(w.min()),
                   w_max=float(w.max()), median_r=float(scale), secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(STUDY_ARMS))
    ap.add_argument("--skip-oversquash", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    D = G.load_data()
    arms = args.arms.split(",")
    summary = {"oversquash": [], "edgefeat": []}

    for a in arms:
        if not args.skip_oversquash:
            _, meta = build_oversquash_w(a, D)
            summary["oversquash"].append(meta)
            log(f"oversquash_w {a:11s} mean={meta.get('w_mean'):.4f} ({meta.get('secs','-')}s)")
        # precompute clean + dashmotif φ_e (dashmotif columns zero if compute_dash_motif hasn't run)
        for prof in ("clean", "dashmotif"):
            phi = G.load_edge_feat_profile(a, prof, D)
            hd, hm = G.dashmotif_status(a) if prof == "dashmotif" else (False, False)
            summary["edgefeat"].append(dict(arm=a, profile=prof, dim=int(phi.shape[1]),
                                            has_dash=bool(hd), has_motif=bool(hm)))
        log(f"edgefeat {a:11s} clean+dashmotif built (dash={hd}, motif={hm})")

    json.dump(summary, open(os.path.join(CACHE, "strengthen_caches_summary.json"), "w"), indent=2)
    log(f"DONE -> {os.path.join(CACHE, 'strengthen_caches_summary.json')}")


if __name__ == "__main__":
    main()
