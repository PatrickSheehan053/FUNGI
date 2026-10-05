"""
obj_009 — ablation.py : the in-distribution graph-swap ablation on TWO difficulty rungs (Rung 3 zero-shot is
HYPHAE-gated, not run here).

  Rung 1 CELL-HELD-OUT (easiest): all perts in train+test; train on cell-half A's residual, eval vs cell-half
    B's (and symmetric), average per pert. Cell-sampling robustness. VALID (DNPN seed-only / no node features /
    no per-pert params / ψ(0)=0 ⇒ empty graph still RSC=0, graph is the only channel). NOT unseen-pert/gene.
  Rung 2 PERTURBATION-HELD-OUT CV (harder, leakage-clean): 2-fold over FUNGI-source train perts; model never
    sees the held-out perturbation's response. Unseen-perturbation-response generalization.

Same FIXED model, graph-swapped across arms: fungi_bio, top_weight, shuffle, knn, mst, empty, reverse,
labelperm (+ fungi_synth_obj008). RSC/DNSA/wR² at the K sweep. K-gap gauge: RSC(FUNGI)−RSC(top_weight) and
DNSA-vs-own-shuffle at each K. Report whether FUNGI's edge GROWS Rung 1 → Rung 2.

Model is FROZEN (one shared config); NO per-graph / per-rung / per-K hyperparameter tuning.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs as G
import harness as H
import metrics as M

DATA = os.path.join(HERE, "..", "intermediate", "obj_009_data.npz")
RES = os.path.join(HERE, "..", "results")
# FROZEN config selected by src/config_probe.py (shared, graph-agnostic; NO per-graph/per-K tuning). The
# config probe showed the model was UNDER-TRAINED at lr1e-3/120ep (RSC 0.012); lr2e-3 + 180 epochs lifts it
# to 0.031 (matches the depth lever K=4=0.031). d=64 gave no gain and spilled; neg_one is a minor +. K is the
# SWEPT variable (not tuned). ~0.55x DWLP absolute (the shared-readout / zero-shot-capable tradeoff).
FROZEN = dict(d=32, d_hidden=64, oversmooth="jk", dropout=0.1, lr=2e-3, weight_decay=1e-4,
              batch_perts=16, epochs=130, patience=12, min_delta=5e-4, grad_clip=1.0,
              corr_weight=1.0, mse_weight=0.1, delta_mode="neg_mu_ctrl")
# epochs 180->130, patience 25->12, min_delta 1e-4->5e-4: bounds each run + stops noise-level arms early
# (killed a 40-min outlier). Modest RSC cost, big speed win; still the same FROZEN config for every arm (fair).
ALL_ARMS = ["fungi_bio", "top_weight", "shuffle", "knn", "mst", "empty", "reverse", "labelperm"]


def log(m): print(f"[ablation {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_data():
    d = np.load(DATA, allow_pickle=True)
    return dict(panel=[str(x) for x in d["panel_genes"]], N=len(d["panel_genes"]),
                mu_ctrl=d["mu_ctrl"], signal_mask=d["signal_mask"].astype(bool),
                pidx=d["pidx"].astype(np.int64), s_full=d["s_full"], s_A=d["s_A"], s_B=d["s_B"],
                var_real=d["var_real"], g2i={g: i for i, g in enumerate([str(x) for x in d["panel_genes"]])})


def a_csc(src, tgt, N):
    A = sp.coo_matrix((np.ones(len(src)), (tgt, src)), shape=(N, N)).tocsc()  # col s -> out-neighbors of s
    return A


def _metrics(s_true, pred, D, A_csc, wr2w, eval_ix):
    rsc = M.rsc_per_pert(s_true[eval_ix], pred, D["signal_mask"], D["pidx"][eval_ix])
    dn = M.dnsa(s_true[eval_ix], pred, D["signal_mask"], D["pidx"][eval_ix], A_csc)
    wr = M.weighted_r2_per_pert(s_true[eval_ix], pred, D["signal_mask"], D["pidx"][eval_ix], weights=wr2w)
    names = np.array(D["panel"])  # not used; keyed by position
    return rsc, dn["per_pert"], wr, dn["dnsa_pair"]


def run_rung2(graph, D, K, seed, eval_ix, wr2w, A_csc):
    """2-fold perturbation-held-out CV. Returns per-pert rsc, dnsa, wr2 (arrays over eval_ix) + pair dnsa."""
    cfg = dict(FROZEN, K=K)
    # SPILL-FREE on the 2070 (Patrick's rule): activations scale with K, so deep-K uses a smaller batch. K=4
    # batch16 = 7.46GB (measured, no spill). K=5/6 auto-drop to keep it in VRAM. If ANY combo still spills it
    # is STOPPED and tagged 5090-only (see STATUS_HANDOFF.md); do NOT run a spilling combo locally.
    if K == 5:
        cfg["batch_perts"] = 10
    elif K >= 6:
        cfg["batch_perts"] = 8
    pidx = D["pidx"]; s = D["s_full"]; N = D["N"]
    rng = np.random.default_rng(seed); perm = rng.permutation(len(eval_ix)); half = len(perm) // 2
    folds = [perm[:half], perm[half:]]
    pred = np.zeros((len(eval_ix), N))
    for fa, fb in [(0, 1), (1, 0)]:
        fit = eval_ix[folds[fa]]; ev = eval_ix[folds[fb]]
        p, _ = H.train_one(graph, pidx[fit], s[fit], pidx[ev], s[ev], D["signal_mask"], D["mu_ctrl"],
                           cfg, seed=seed, delta_mode=cfg["delta_mode"])
        pred[folds[fb]] = p
    return _metrics(s, pred, D, A_csc, wr2w, eval_ix)


def run_rung1(graph, D, K, seed, eval_ix, wr2w, A_csc):
    """Cell-held-out: train on half A residual, eval vs half B (and symmetric); average per pert."""
    cfg = dict(FROZEN, K=K)
    # SPILL-FREE on the 2070 (Patrick's rule): activations scale with K, so deep-K uses a smaller batch. K=4
    # batch16 = 7.46GB (measured, no spill). K=5/6 auto-drop to keep it in VRAM. If ANY combo still spills it
    # is STOPPED and tagged 5090-only (see STATUS_HANDOFF.md); do NOT run a spilling combo locally.
    if K == 5:
        cfg["batch_perts"] = 10
    elif K >= 6:
        cfg["batch_perts"] = 8
    pidx = D["pidx"]; N = D["N"]
    outs = []
    for fit_s, eval_s in [(D["s_A"], D["s_B"]), (D["s_B"], D["s_A"])]:
        p, _ = H.train_one(graph, pidx[eval_ix], fit_s[eval_ix], pidx[eval_ix], eval_s[eval_ix],
                           D["signal_mask"], D["mu_ctrl"], cfg, seed=seed, delta_mode=cfg["delta_mode"])
        r, dn_pp, wr, dn_pair = _metrics(eval_s, p, D, A_csc, wr2w, eval_ix)
        outs.append((r, dn_pp, wr, dn_pair))
    rsc = np.nanmean([outs[0][0], outs[1][0]], axis=0)
    dn = np.nanmean([outs[0][1], outs[1][1]], axis=0)
    wr = np.nanmean([outs[0][2], outs[1][2]], axis=0)
    pair = float(np.nanmean([outs[0][3], outs[1][3]]))
    return rsc, dn, wr, pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="rung1,rung2")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--Ks", default="1,2,3,4,6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    D = load_data()
    arms = args.arms.split(","); Ks = [int(x) for x in args.Ks.split(",")]; seeds = [int(x) for x in args.seeds.split(",")]
    rungs = args.rungs.split(",")
    eval_ix = np.where(D["pidx"] >= 0)[0]
    wr2w = np.sqrt(np.maximum(D["var_real"], 0.0))
    log(f"eval universe={len(eval_ix)} arms={arms} Ks={Ks} seeds={seeds} rungs={rungs}")

    # precompute graphs + A_csc per arm
    graphs = {a: G.build_arm(a, D["g2i"], D["N"]) for a in arms}
    acsc = {a: a_csc(graphs[a][0], graphs[a][1], D["N"]) for a in arms}

    rows = []                # summary
    per_pert = {}            # (rung,arm,K) -> dict(rsc=array over seeds x perts stacked)
    for rung in rungs:
        runner = run_rung1 if rung == "rung1" else run_rung2
        for K in Ks:
            for a in arms:
                accR, accD, accW = [], [], []
                pair_acc = []
                for sd in seeds:
                    t0 = time.time()
                    rsc, dn, wr, pair = runner(graphs[a], D, K, sd, eval_ix, wr2w, acsc[a])
                    accR.append(rsc); accD.append(dn); accW.append(wr); pair_acc.append(pair)
                    log(f"{rung} {a:16s} K={K} seed={sd} RSC={np.nanmean(rsc):+.4f} DNSA={np.nanmean(dn):+.4f} ({time.time()-t0:.0f}s)")
                # per-pert mean over seeds
                rsc_pp = np.nanmean(np.stack(accR), axis=0)
                dn_pp = np.nanmean(np.stack(accD), axis=0)
                wr_pp = np.nanmean(np.stack(accW), axis=0)
                per_pert[(rung, a, K)] = dict(rsc=rsc_pp, dnsa=dn_pp, wr2=wr_pp)
                ob = M.one_sample_bootstrap(rsc_pp, n_boot=10000, seed=1)
                rows.append(dict(rung=rung, arm=a, K=K, n_edges=len(graphs[a][0]),
                                 rsc=ob["mean"], rsc_ci_lo=ob["ci_lo"], rsc_ci_hi=ob["ci_hi"],
                                 dnsa=float(np.nanmean(dn_pp)), dnsa_pair=float(np.nanmean(pair_acc)),
                                 wr2=float(np.nanmean(wr_pp)), n=int(np.isfinite(rsc_pp).sum())))
            # checkpoint after each (rung,K)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"obj_009_ablation_summary_{args.tag}.csv"), index=False)

    # significance + K-gap
    sig, kgap = [], []
    for rung in rungs:
        for K in Ks:
            def pp(a, m): return per_pert[(rung, a, K)][m] if (rung, a, K) in per_pert else None
            fr = pp("fungi_bio", "rsc"); tw = pp("top_weight", "rsc")
            fd = pp("fungi_bio", "dnsa"); fsh = pp("shuffle", "dnsa")
            if fr is not None and tw is not None:
                pb = M.paired_bootstrap(fr, tw, n_boot=10000, seed=2)
                kgap.append(dict(rung=rung, K=K, metric="rsc", fungi_minus_topweight=pb["mean_diff"],
                                 p=pb["p_one_sided"], fungi_rsc=float(np.nanmean(fr)), topweight_rsc=float(np.nanmean(tw))))
            # each arm vs own shuffle-ish (fungi vs shuffle) + arm vs top_weight
            for a in arms:
                r = pp(a, "rsc")
                if r is None: continue
                if tw is not None and a != "top_weight":
                    pbt = M.paired_bootstrap(r, tw, n_boot=10000, seed=3)
                    sig.append(dict(rung=rung, K=K, comparison=f"{a}_vs_top_weight", diff=pbt["mean_diff"], p=pbt["p_one_sided"]))
                if pp("shuffle", "rsc") is not None:
                    pbs = M.paired_bootstrap(r, pp("shuffle", "rsc"), n_boot=10000, seed=4)
                    sig.append(dict(rung=rung, K=K, comparison=f"{a}_vs_shuffle", diff=pbs["mean_diff"], p=pbs["p_one_sided"]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"obj_009_ablation_summary_{args.tag}.csv"), index=False)
    pd.DataFrame(sig).to_csv(os.path.join(RES, f"obj_009_significance_{args.tag}.csv"), index=False)
    pd.DataFrame(kgap).to_csv(os.path.join(RES, f"obj_009_kgap_{args.tag}.csv"), index=False)
    log("wrote ablation summary + significance + kgap")


if __name__ == "__main__":
    main()
