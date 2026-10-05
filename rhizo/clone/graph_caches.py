"""
obj_009.1 — graph_caches.py : build ALL per-arm CPU-side caches BEFORE the GPU grid (respects the no-CPU∥GPU
freeze rule — heavy Louvain/PageRank/pinv/BFS run while the GPU is idle). Per arm:
  edgefeat__<arm>.npz  phi_e (E,8)     [V-EDGE]        — all arms a variant may train on
  nodectx__<arm>.npz   c_g (N,6), comm [V-FILM]        — all arms (Louvain shared with edge features)
  bfsdist__<arm>.npz   dist (n_src,N) int16, src_ids   [P2 dsRSC/reach/DNSA_ge2hop] — P2 arms only
  effres__<arm>.npz    eff_R (n_pert,), reach_frac     [P2 eff_resistance] — P2 arms only (dense Laplacian pinv)

Idempotent/resumable: skips any cache file already present. CLI: python graph_caches.py [--arms ...] [--p2 ...].
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import graphs_v2 as G
import edge_features as EF
import node_context as NC
import metrics_v2 as M2

CACHE = os.path.join(HERE, "..", "intermediate", "graph_caches")
BFS_CAP = 6


def log(m): print(f"[caches {time.strftime('%H:%M:%S')}] {m}", flush=True)


def sources_and_rowmap(D):
    src_ids = np.unique(D["pidx"][D["pidx"] >= 0]).astype(np.int64)
    src_row = {int(g): i for i, g in enumerate(src_ids)}
    return src_ids, src_row


def build_bfs(cache_dir, arm, src, tgt, N, src_ids):
    p = os.path.join(cache_dir, f"bfsdist__{arm}.npz")
    if os.path.exists(p):
        return
    dist, _ = G.bfs_directed_dist(src, tgt, N, src_ids, cap=BFS_CAP)
    np.savez_compressed(p, dist=dist, src_ids=src_ids)


def build_effres(cache_dir, arm, src, tgt, w_raw, N, D):
    p = os.path.join(cache_dir, f"effres__{arm}.npz")
    if os.path.exists(p):
        return
    Lpinv, comp = G.undirected_laplacian_pinv(src, tgt, w_raw, N)
    effR, frac = M2.eff_resistance_per_pert(D["s_full"], D["signal_mask"], D["pidx"], Lpinv, comp, k_deg=20)
    del Lpinv
    np.savez_compressed(p, eff_R=effR, reach_frac=frac)


def build_all(arms_edge_node, arms_p2, D, cache_dir=CACHE, louvain_seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    src_ids, _ = sources_and_rowmap(D)
    N = D["N"]
    stats = {}
    for arm in arms_edge_node:
        t0 = time.time()
        s, t, wn, wr = G.build_arm(arm, D["g2i"], N)
        # node context first (gives the shared Louvain partition), then edge features reuse it
        c, comm, cst = NC.load_or_compute(cache_dir, arm, s, t, wr, N, louvain_seed)
        phi, est = EF.load_or_compute(cache_dir, arm, s, t, wn, wr, N, louvain_seed, comm_labels=comm)
        if arm in arms_p2:
            build_bfs(cache_dir, arm, s, t, N, src_ids)
            build_effres(cache_dir, arm, s, t, wr, N, D)
        stats[arm] = dict(n_edges=int(len(s)), **{k: est.get(k) for k in ("frac_recip", "frac_cross")},
                          n_iso=cst.get("n_isolated"), n_comm=cst.get("n_communities"),
                          in_p2=(arm in arms_p2), secs=round(time.time() - t0, 1))
        log(f"{arm:12s} E={len(s):>7d} recip={est.get('frac_recip',0):.4f} p2={arm in arms_p2} ({stats[arm]['secs']}s)")
    json.dump(dict(src_ids_n=int(len(src_ids)), cap=BFS_CAP, stats=stats),
              open(os.path.join(cache_dir, "caches_summary.json"), "w"), indent=2, default=float)
    return stats


# ---------------------------------------------------------------- loaders (used by ablation_v2)
def load_edge_feat(arm, cache_dir=CACHE):
    d = np.load(EF.cache_path(cache_dir, arm), allow_pickle=True)
    return d["phi"].astype(np.float32)


def load_node_ctx(arm, cache_dir=CACHE):
    d = np.load(NC.cache_path(cache_dir, arm), allow_pickle=True)
    return d["c"].astype(np.float32)


def load_bfs(arm, cache_dir=CACHE):
    d = np.load(os.path.join(cache_dir, f"bfsdist__{arm}.npz"), allow_pickle=True)
    dist = d["dist"]; src_ids = d["src_ids"].astype(np.int64)
    src_row = {int(g): i for i, g in enumerate(src_ids)}
    return dist, src_row


def load_effres(arm, cache_dir=CACHE):
    d = np.load(os.path.join(cache_dir, f"effres__{arm}.npz"), allow_pickle=True)
    return d["eff_R"], d["reach_frac"]


if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fungi_bio,top_weight,shuffle,knn,mst,empty,reverse,labelperm")
    ap.add_argument("--p2", default="fungi_bio,top_weight,shuffle,empty")
    args = ap.parse_args()
    D = G.load_data()
    build_all(args.arms.split(","), set(args.p2.split(",")), D)
    log("DONE building graph caches")
