"""
obj_009.1 — ablation_v2.py : the variant x arm x K x seed x rung driver. ONE variant per invocation (the
orchestrator loops); resumable via the per-run CSV. Trains DNPNv2 (FROZEN HP, graph-swapped), scores P1
(RSC/DNSA/wr2 — byte-identical to v1) always, and P2 (dsRSC_ge2/ge3, reach_at_K, eff_resistance, DNSA_ge2hop)
whenever the (arm,K) is in the P2 cell. P2 REUSES the same trained predictions (no retraining).

Freeze-safe (v1): GPU-resident graph/targets/features, num_workers=0, no CPU<->GPU in the gradient loop.
Spill-free: batch drops by K and by variant cap (measured caps override the config in intermediate/spill_caps.json).

Outputs (long format, accumulated across variants):
  results/runs_<tag>.csv         one row per (variant,feat_shuffle,rung,K,arm,seed) scalar metrics (resumable)
  results/perpert_<tag>.npz      per-pert arrays (nanmean over seeds) keyed "<variant>|<fs>|<rung>|K<K>|<arm>|<metric>"
--summarize (run after all variants): writes P1_summary, P1_kgap, P2_topology, v1_vs_v2 CSVs.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTHONUTF8", "1")

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v2 as G
import harness_v2 as H
import metrics_v2 as M
import graph_caches as GC
from model_v2 import VARIANT_FEATURES
import yaml

RES = os.path.join(HERE, "..", "results")
INTER = os.path.join(HERE, "..", "intermediate")
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "obj_009_1.yaml")))
FROZEN = dict(CFG["frozen_hp"]); FROZEN["teleport_alpha"] = CFG["teleport"]["alpha"]
P2_ARMS = set(CFG["eval"]["P2_topology"]["arms"])
P2_KS = set(CFG["eval"]["P2_topology"]["Ks"])
BFS_CAP = 6


def log(m): print(f"[abl2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_data():
    return G.load_data()


def batch_for(variant, K):
    """SPILL-SAFE batch: honor the MEASURED per-(variant,K) caps from the spill probe (never above them)."""
    sp = CFG["spill_guard"]
    b = int(sp["base_batch"])
    b = min(b, int(sp["batch_by_K"].get(K, sp["batch_by_K"].get(str(K), b))))
    caps4, caps6 = {}, {}
    cp = os.path.join(INTER, "spill_caps.json")
    if os.path.exists(cp):
        d = json.load(open(cp)); caps4 = d.get("variant_batch_cap", {}); caps6 = d.get("variant_batch_cap_k6", {})
    meas = caps6.get(variant) if K >= 5 else caps4.get(variant)
    if meas is None:  # fall back to the provisional config cap if the probe hasn't run for this variant
        meas = CFG["spill_guard"]["variant_batch_cap"].get(variant)
    if meas is not None and int(meas) > 0:
        b = min(b, int(meas))
    return b


def _feats_for(variant, vfull_feats):
    if variant == "v_full" and vfull_feats is not None:
        return dict(vfull_feats)
    return dict(VARIANT_FEATURES[variant])


def _shuffle_feat(arr, kind, seed=1234):
    """V7/V8 anti-leak controls: permute edge features across edges / node context across genes."""
    if arr is None:
        return None
    rng = np.random.default_rng(seed)
    out = arr.copy()
    rng.shuffle(out, axis=0)   # permute rows (edges for phi_e, genes for c_g)
    return out


def run_arm(arm, D, K, seed, rung, variant, feats, eval_ix, wr2w, phi, ctx, acsc):
    cfg = dict(FROZEN, K=K, batch_perts=batch_for(variant, K))
    pidx = D["pidx"]; N = D["N"]
    graph = G.build_arm(arm, D["g2i"], N)[:3]  # (src,tgt,w_outnorm) — harness ignores w_raw
    kw = dict(variant=variant, feats=feats, phi_e=phi, c_g=ctx, delta_mode=cfg["delta_mode"])
    if rung == "rung2":
        rng = np.random.default_rng(seed); perm = rng.permutation(len(eval_ix)); half = len(perm) // 2
        folds = [perm[:half], perm[half:]]
        pred = np.zeros((len(eval_ix), N))
        for fa, fb in [(0, 1), (1, 0)]:
            fit = eval_ix[folds[fa]]; ev = eval_ix[folds[fb]]
            p, _ = H.train_one(graph, pidx[fit], D["s_full"][fit], pidx[ev], D["s_full"][ev],
                               D["signal_mask"], D["mu_ctrl"], cfg, seed=seed, **kw)
            pred[folds[fb]] = p
        s_eval = D["s_full"]
        return _score(D, s_eval, pred, eval_ix, wr2w, acsc, arm, K), cfg["batch_perts"]
    else:  # rung1 cell-held-out
        outs = []
        for fit_s, eval_s in [(D["s_A"], D["s_B"]), (D["s_B"], D["s_A"])]:
            p, _ = H.train_one(graph, pidx[eval_ix], fit_s[eval_ix], pidx[eval_ix], eval_s[eval_ix],
                               D["signal_mask"], D["mu_ctrl"], cfg, seed=seed, **kw)
            outs.append(_score(D, eval_s, p, eval_ix, wr2w, acsc, arm, K))
        # average the two symmetric directions per-pert
        avg = {}
        for k in outs[0]:
            a0, a1 = outs[0][k], outs[1][k]
            if isinstance(a0, np.ndarray):
                avg[k] = np.nanmean(np.stack([a0, a1]), axis=0)
            else:
                avg[k] = float(np.nanmean([a0, a1]))
        return avg, cfg["batch_perts"]


def _score(D, s_true, pred, eval_ix, wr2w, acsc, arm, K):
    """Return dict of per-pert arrays (P1 always; P2 when arm in P2_ARMS and K in P2_KS)."""
    st = s_true[eval_ix]
    mask = D["signal_mask"]; pe = D["pidx"][eval_ix]
    r = {}
    r["rsc"] = M.rsc_per_pert(st, pred, mask, pe)
    dn = M.dnsa(st, pred, mask, pe, acsc)
    r["dnsa"] = dn["per_pert"]; r["dnsa_pair"] = dn["dnsa_pair"]
    r["wr2"] = M.weighted_r2_per_pert(st, pred, mask, pe, weights=wr2w)
    if arm in P2_ARMS and K in P2_KS:
        dist, src_row = GC.load_bfs(arm)
        r["dsrsc_ge2"], _ = M.dsrsc_per_pert(st, pred, mask, pe, dist, src_row, 2, BFS_CAP)
        r["dsrsc_ge3"], _ = M.dsrsc_per_pert(st, pred, mask, pe, dist, src_row, 3, BFS_CAP)
        r["reach_at_K"] = M.reach_at_k_per_pert(st, mask, pe, dist, src_row, K, k_deg=20)
        effR, _ = GC.load_effres(arm)
        r["eff_res"] = effR[eval_ix]
        d2, _, _ = M.dnsa_ge2hop_per_pert(st, pred, mask, pe, dist, src_row, K, min_hops=2)
        r["dnsa_ge2hop"] = d2
    return r


SCALAR_KEYS = ["rsc", "dnsa", "wr2", "dsrsc_ge2", "dsrsc_ge3", "reach_at_K", "eff_res", "dnsa_ge2hop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--feat_shuffle", default="none", choices=["none", "edge", "context"])
    ap.add_argument("--vfull_feats", default="")   # json for v_full composition
    ap.add_argument("--arms", default="")          # override; else derived from config phases
    ap.add_argument("--Ks", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--rungs", default="rung1,rung2")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if args.summarize:
        return summarize(args.tag)
    if not args.variant:
        ap.error("--variant is required unless --summarize")
    # live skip-list (lets the already-running orchestrator drop variants without a kill/relaunch)
    skip_file = os.path.join(INTER, "skip_variants.json")
    if os.path.exists(skip_file) and args.variant in json.load(open(skip_file)):
        log(f"variant {args.variant} is in skip_variants.json -> SKIP"); return

    D = load_data()
    eval_ix = np.where(D["pidx"] >= 0)[0]
    wr2w = np.sqrt(np.maximum(D["var_real"], 0.0))
    vfull = json.loads(args.vfull_feats) if args.vfull_feats else None
    feats = _feats_for(args.variant, vfull)

    ep = CFG["eval"]["P1_reproduce"]
    if args.arms:
        arm_plan = {a: [int(x) for x in args.Ks.split(",")] for a in args.arms.split(",")}
    else:
        arm_plan = {}
        for a in ep["kgap_arms"] + ep["p2_extra_arms"]:
            arm_plan[a] = list(ep["Ks_full"])
        for a in ep["null_arms"]:
            arm_plan[a] = list(ep["Ks_null"])
    seeds = [int(x) for x in (args.seeds.split(",") if args.seeds else map(str, ep["seeds"]))]
    rungs = args.rungs.split(",")

    runs_csv = os.path.join(RES, f"runs_{args.tag}.csv")
    done = set()
    rows = []
    if os.path.exists(runs_csv):
        prev = pd.read_csv(runs_csv)
        rows = prev.to_dict("records")
        for _, rr in prev.iterrows():
            done.add((rr["variant"], rr["feat_shuffle"], rr["rung"], int(rr["K"]), rr["arm"], int(rr["seed"])))

    # precompute per-arm features + A_csc (once per arm)
    arms = list(arm_plan.keys())
    acsc = {a: G.a_csc(*G.build_arm(a, D["g2i"], D["N"])[:2], D["N"]) for a in arms}
    phi_cache, ctx_cache = {}, {}
    for a in arms:
        phi = GC.load_edge_feat(a) if feats["edge"] else None
        ctx = GC.load_node_ctx(a) if feats["film"] else None
        if args.feat_shuffle == "edge" and phi is not None:
            phi = _shuffle_feat(phi, "edge")
        if args.feat_shuffle == "context" and ctx is not None:
            ctx = _shuffle_feat(ctx, "context")
        phi_cache[a] = phi; ctx_cache[a] = ctx

    perpert = {}  # per-SEED key (idempotent on resume) -> per-pert array; summarize averages over seeds
    for rung in rungs:
        for a in arms:
            for K in arm_plan[a]:
                for sd in seeds:
                    key = (args.variant, args.feat_shuffle, rung, K, a, sd)
                    if key in done:
                        continue
                    t0 = time.time()
                    sc, bs = run_arm(a, D, K, sd, rung, args.variant, feats, eval_ix, wr2w,
                                     phi_cache[a], ctx_cache[a], acsc[a])
                    secs = time.time() - t0
                    row = dict(variant=args.variant, feat_shuffle=args.feat_shuffle, rung=rung, K=K, arm=a,
                               seed=sd, batch=bs, n_edges=int(len(acsc[a].data)), secs=round(secs, 1))
                    for mk in SCALAR_KEYS:
                        row[mk] = float(np.nanmean(sc[mk])) if mk in sc else np.nan
                    row["dnsa_pair"] = sc.get("dnsa_pair", np.nan)
                    rows.append(row)
                    for mk in SCALAR_KEYS + ["dnsa"]:
                        if mk in sc and isinstance(sc[mk], np.ndarray):
                            perpert[f"{args.variant}|{args.feat_shuffle}|{rung}|K{K}|{a}|{mk}|s{sd}"] = sc[mk]
                    pd.DataFrame(rows).to_csv(runs_csv, index=False)   # checkpoint every training
                    _merge_perpert(args.tag, perpert)                  # per-pert npz kept in sync (cheap, small)
                    log(f"{args.variant} {args.feat_shuffle} {rung} {a:12s} K={K} sd={sd} "
                        f"RSC={row['rsc']:+.4f} dsRSC2={row.get('dsrsc_ge2',np.nan):+.4f} b={bs} ({secs:.0f}s)")
    _merge_perpert(args.tag, perpert)
    log(f"variant {args.variant} ({args.feat_shuffle}) DONE")


def _merge_perpert(tag, perpert):
    p = os.path.join(RES, f"perpert_{tag}.npz")
    existing = {}
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        existing = {k: d[k] for k in d.files}
    existing.update(perpert)
    np.savez_compressed(p, **existing)


# ----------------------------------------------------------------- summarize
def _cell_pp(ppd, variant, fs, rung, K, arm, mk):
    """Per-pert array for a cell = nanmean over the per-seed keys '<...>|<mk>|s<sd>' (back-compat: seedless)."""
    pref = f"{variant}|{fs}|{rung}|K{K}|{arm}|{mk}|s"
    arrs = [ppd[k] for k in ppd if k.startswith(pref)]
    if arrs:
        return np.nanmean(np.stack(arrs), axis=0)
    return ppd.get(f"{variant}|{fs}|{rung}|K{K}|{arm}|{mk}")


def summarize(tag):
    runs = pd.read_csv(os.path.join(RES, f"runs_{tag}.csv"))
    pp = np.load(os.path.join(RES, f"perpert_{tag}.npz"), allow_pickle=True)
    ppd = {k: pp[k] for k in pp.files}

    # P1 summary (mean over seeds already; add bootstrap CI on the per-pert rsc)
    summ_rows = []
    for (variant, rung, arm, K), g in runs[runs.feat_shuffle == "none"].groupby(["variant", "rung", "arm", "K"]):
        rsc_pp = _cell_pp(ppd, variant, "none", rung, K, arm, "rsc")
        ci = M.one_sample_bootstrap(rsc_pp, n_boot=10000, seed=1) if rsc_pp is not None else dict(mean=g.rsc.mean(), ci_lo=np.nan, ci_hi=np.nan)
        row = dict(variant=variant, rung=rung, arm=arm, K=K, n_edges=int(g.n_edges.iloc[0]),
                   rsc=ci["mean"], rsc_ci_lo=ci["ci_lo"], rsc_ci_hi=ci["ci_hi"],
                   dnsa=g.dnsa.mean(), wr2=g.wr2.mean())
        for mk in ["dsrsc_ge2", "dsrsc_ge3", "reach_at_K", "eff_res", "dnsa_ge2hop"]:
            row[mk] = g[mk].mean()
        summ_rows.append(row)
    P1 = pd.DataFrame(summ_rows)
    P1.to_csv(os.path.join(RES, "obj_009_1_P1_summary.csv"), index=False)

    # kgap: RSC(fungi)-RSC(top_weight) + DNSA gap, paired bootstrap, per variant/K/rung
    kgap_rows = []
    for variant in P1.variant.unique():
        for rung in ["rung1", "rung2"]:
            for K in sorted(runs.K.unique()):
                fk = _cell_pp(ppd, variant, "none", rung, K, "fungi_bio", "rsc")
                tk = _cell_pp(ppd, variant, "none", rung, K, "top_weight", "rsc")
                if fk is not None and tk is not None:
                    pb = M.paired_bootstrap(fk, tk, n_boot=10000, seed=2)
                    fd = _cell_pp(ppd, variant, "none", rung, K, "fungi_bio", "dnsa")
                    td = _cell_pp(ppd, variant, "none", rung, K, "top_weight", "dnsa")
                    dgap = float(np.nanmean(fd) - np.nanmean(td)) if (fd is not None and td is not None) else np.nan
                    kgap_rows.append(dict(variant=variant, rung=rung, K=K, metric="rsc",
                                          fungi_minus_topweight=pb["mean_diff"], ci_lo=pb["ci_lo"], ci_hi=pb["ci_hi"],
                                          p=pb["p_one_sided"], fungi_rsc=float(np.nanmean(fk)),
                                          topweight_rsc=float(np.nanmean(tk)), dnsa_gap=dgap))
    KG = pd.DataFrame(kgap_rows)
    KG.to_csv(os.path.join(RES, "obj_009_1_P1_kgap.csv"), index=False)

    # P2 topology table + gaps (fungi vs top_weight) with paired p
    p2_rows = []
    for variant in P1.variant.unique():
        for rung in ["rung1", "rung2"]:
            for K in sorted(P2_KS):
                for metric in ["dsrsc_ge2", "dsrsc_ge3", "reach_at_K", "eff_res", "dnsa_ge2hop"]:
                    fk = _cell_pp(ppd, variant, "none", rung, K, "fungi_bio", metric)
                    tk = _cell_pp(ppd, variant, "none", rung, K, "top_weight", metric)
                    if fk is None or tk is None:
                        continue
                    pb = M.paired_bootstrap(fk, tk, n_boot=10000, seed=3)
                    row = dict(variant=variant, rung=rung, K=K, metric=metric,
                               fungi=float(np.nanmean(fk)), top_weight=float(np.nanmean(tk)),
                               gap=pb["mean_diff"], ci_lo=pb["ci_lo"], ci_hi=pb["ci_hi"], p=pb["p_one_sided"])
                    for other in ["shuffle", "empty"]:
                        ov = _cell_pp(ppd, variant, "none", rung, K, other, metric)
                        row[other] = float(np.nanmean(ov)) if ov is not None else np.nan
                    p2_rows.append(row)
    P2 = pd.DataFrame(p2_rows)
    P2.to_csv(os.path.join(RES, "obj_009_1_P2_topology.csv"), index=False)

    # v1_vs_v2: ΔGAP = kgap(variant) - kgap(v1_baseline) at matched (rung,K); + dsRSC/DNSA_ge2hop gap flips
    vv_rows = []
    base = KG[KG.variant == "v1_baseline"].set_index(["rung", "K"])
    for variant in [v for v in P1.variant.unique() if v != "v1_baseline"]:
        vk = KG[KG.variant == variant].set_index(["rung", "K"])
        for (rung, K), r in vk.iterrows():
            if (rung, K) in base.index:
                b = base.loc[(rung, K)]
                dgap = r["fungi_minus_topweight"] - b["fungi_minus_topweight"]
                # P2 flips
                p2v = P2[(P2.variant == variant) & (P2.rung == rung) & (P2.K == K)]
                ds2 = p2v[p2v.metric == "dsrsc_ge2"]
                dn2 = p2v[p2v.metric == "dnsa_ge2hop"]
                vv_rows.append(dict(variant=variant, rung=rung, K=K,
                                    gap_v2=r["fungi_minus_topweight"], gap_v1=b["fungi_minus_topweight"],
                                    delta_gap=dgap, gap_v2_ci_lo=r["ci_lo"], gap_v2_ci_hi=r["ci_hi"],
                                    dnsa_gap_v2=r["dnsa_gap"],
                                    dsRSC_ge2_gap=(float(ds2["gap"].iloc[0]) if len(ds2) else np.nan),
                                    dsRSC_ge2_p=(float(ds2["p"].iloc[0]) if len(ds2) else np.nan),
                                    dnsa_ge2hop_gap=(float(dn2["gap"].iloc[0]) if len(dn2) else np.nan),
                                    dnsa_ge2hop_p=(float(dn2["p"].iloc[0]) if len(dn2) else np.nan)))
    VV = pd.DataFrame(vv_rows)
    VV.to_csv(os.path.join(RES, "obj_009_1_v1_vs_v2.csv"), index=False)
    log(f"summarize DONE: P1({len(P1)}) kgap({len(KG)}) P2({len(P2)}) v1_vs_v2({len(VV)})")
    return dict(P1=len(P1), kgap=len(KG), P2=len(P2), vv=len(VV))


if __name__ == "__main__":
    main()
