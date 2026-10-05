"""
obj_009.3 (RHIZO-final) — run_trial_v3f.py : train ONE assembled-model HPO config and write a result json.

Trains the ASSEMBLED RHIZO model (edge_clean engine + FiLM + over-squash + swept fusion) graph-swapped on the
gauntlet arms (fungi_bio, top_weight, kNN, shuffle) and reports, all as GAPs (fungi − arm):
  gap                 = fungi − top_weight on aggregate RSC        (the HPO OBJECTIVE = gap_vs_topweight)
  gap_vs_knn          = fungi − kNN on aggregate RSC               (kNN is the honest boss on its own turf)
  gap_coexpr_resid    = fungi − {top_weight, knn} on the CO-EXPRESSION-RESIDUALIZED RSC   (FUNGI's expected win)
  dnsa_ge2hop_gap     = fungi − top_weight on directed long-range sign accuracy            (FUNGI's expected win)
  dsrsc_ge2_gap       = fungi − top_weight on distance≥2 RSC
  fungi_minus_shuffle = the degree-preserving null margin (a headline win must beat it)

φ_e / c_g / hop_w are wired via harness_v3f.arm_inputs (FiLM genuinely fires — unlike the strengthen study).
The φ_e profile is config['edge_profile'] (clean|directed_topo). Leakage-safe: the co-expression baseline is
built from the FIT split ONLY (per fold). Resumable (skips if --out exists). Launched N-up per GPU by hpo_v3f.
"""
from __future__ import annotations
import os, sys, json, time, argparse
_p = argparse.ArgumentParser(add_help=False); _p.add_argument("--gpu", default=os.environ.get("HPO_GPU", "0"))
_k, _ = _p.parse_known_args()
# DEFECT 2 (child side): NEVER setdefault a SLURM-set var. The parent (hpo_v3f) now sets CUDA_VISIBLE_DEVICES
# to this trial's single physical card; if that failed or we're launched standalone against a raw SLURM list
# ("0,1,.."), collapse it to --gpu EXPLICITLY. After masking, the visible card is device 0 -> code uses "cuda".
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if ("," in _cvd) or (_cvd == ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_k.gpu)
# thread vars: EXPLICIT (report follow-up) — sbatch exports 4 job-wide, but each STACKED trial wants 2 to avoid
# oversubscription. setdefault silently inherited the job-wide 4; set what the trial actually intends.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3f as G
import harness_v3f as H
import metrics_v3f as M
from model_v3f import make_feats
BFS_CAP = 6


def _cfg(c):
    return dict(d=c["d"], d_hidden=c["d_hidden"], K=c["K"], oversmooth="jk", dropout=c.get("dropout", 0.1),
                teleport_alpha=c.get("teleport_alpha", 0.15), gcnii_beta=c.get("gcnii_beta", 0.0),
                dropedge_p=c.get("dropedge_p", 0.0), attn_heads=c.get("attn_heads", 4),
                lr=c["lr"], weight_decay=c["weight_decay"], batch_perts=c["batch_perts"],
                epochs=c.get("epochs", 130), patience=c.get("patience", 12), min_delta=c.get("min_delta", 5e-4),
                grad_clip=1.0, corr_weight=1.0, mse_weight=0.1, delta_mode=c.get("delta_mode", "neg_mu_ctrl"))


def _score(D, s_true, pred, eval_ix, arm, K, Zfit):
    st = s_true[eval_ix]; mask = D["signal_mask"]; pe = D["pidx"][eval_ix]
    o = {"rsc": M.rsc_per_pert(st, pred, mask, pe),
         "rsc_coexpr_resid": M.rsc_coexpr_resid_per_pert(st, pred, mask, pe, Zfit)}
    try:
        dist, sr = G.load_bfs(arm)
        ds, _ = M.dsrsc_per_pert(st, pred, mask, pe, dist, sr, 2, BFS_CAP)
        d2, _, _ = M.dnsa_ge2hop_per_pert(st, pred, mask, pe, dist, sr, K, min_hops=2)
        o["dsrsc_ge2"] = ds; o["dnsa_ge2hop"] = d2
    except Exception:
        o["dsrsc_ge2"] = np.full(len(eval_ix), np.nan); o["dnsa_ge2hop"] = np.full(len(eval_ix), np.nan)
    return o


def _train_arm(arm, D, c, feats, phi, c_g, hop_w, eval_ix, seeds, rung, device="cuda"):
    cfg = _cfg(c); N = D["N"]; pidx = D["pidx"]; graph = G.build_arm(arm, D["g2i"], N)[:3]
    kw = dict(variant="assembled", feats=feats, phi_e=phi, c_g=c_g, hop_w=hop_w, delta_mode=cfg["delta_mode"],
              device=device)
    per = []
    for sd in seeds:
        if rung == "rung2":
            rng = np.random.default_rng(sd); perm = rng.permutation(len(eval_ix)); half = len(perm) // 2
            fo = [perm[:half], perm[half:]]; pred = np.zeros((len(eval_ix), N)); scfold = []
            for fa, fb in [(0, 1), (1, 0)]:
                fit = eval_ix[fo[fa]]; ev = eval_ix[fo[fb]]
                assert len(np.intersect1d(fit, ev)) == 0, "rung2 firewall: fit ∩ eval must be empty"
                p, _ = H.train_one(graph, pidx[fit], D["s_full"][fit], pidx[ev], D["s_full"][ev],
                                   D["signal_mask"], D["mu_ctrl"], cfg, seed=sd, **kw)
                pred[fo[fb]] = p
                Zfit = M.build_coexpr_Z(D["s_full"][fit])                          # leakage-safe: fit fold only
                scfold.append((fo[fb], _score(D, D["s_full"], p, ev, arm, c["K"], Zfit)))
            # stitch the two folds back to eval_ix order
            agg = {}
            for kmet in scfold[0][1]:
                v = np.full(len(eval_ix), np.nan)
                for idxs, sc in scfold:
                    v[idxs] = sc[kmet]
                agg[kmet] = v
            per.append(agg)
        else:   # rung1 = cell-held-out A/B (same perts, different cell halves)
            a = []
            for fs, es in [(D["s_A"], D["s_B"]), (D["s_B"], D["s_A"])]:
                p, _ = H.train_one(graph, pidx[eval_ix], fs[eval_ix], pidx[eval_ix], es[eval_ix],
                                   D["signal_mask"], D["mu_ctrl"], cfg, seed=sd, **kw)
                Zfit = M.build_coexpr_Z(fs[eval_ix])
                a.append(_score(D, es, p, eval_ix, arm, c["K"], Zfit))
            per.append({k: np.nanmean(np.stack([a[0][k], a[1][k]]), axis=0) for k in a[0]})
    # mean per-pert across seeds, and scalar means
    pp = {k: np.nanmean(np.stack([p[k] for p in per]), axis=0) for k in per[0]}
    out = {f"{k}_mean": float(np.nanmean(pp[k])) for k in pp}
    out["_pp"] = pp
    return out


def _gap(f, t, key, seed=1):
    pb = M.paired_bootstrap(f["_pp"][key], t["_pp"][key], n_boot=5000, seed=seed)
    return pb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0"); ap.add_argument("--device", default="cuda"); args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"[trial] {args.out} exists -> skip"); return
    c = json.load(open(args.config)); seeds = c.get("seeds", [0, 1]); rung = c.get("rung", "rung2")
    profile = c.get("edge_profile", "directed_topo")
    feats = make_feats("assembled", profile=profile, film=c.get("film", True), ovsq=c.get("ovsq", True))
    use_cuda = (args.device == "cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if use_cuda else "cpu"
    D = G.load_data(); eval_ix = np.where(D["pidx"] >= 0)[0]
    t0 = time.time()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    arms = c.get("arms", ["fungi_bio", "top_weight", "knn", "shuffle"])
    res = {}
    for arm in arms:
        phi, c_g, hop_w = H.arm_inputs(arm, feats, D, profile=profile)
        res[arm] = _train_arm(arm, D, c, feats, phi, c_g, hop_w, eval_ix, seeds, rung, device=device)
    peak = torch.cuda.max_memory_reserved() / 1e9 if use_cuda else 0.0
    out = dict(trial_id=c.get("trial_id"), config={k: v for k, v in c.items() if not k.startswith("_")},
               secs=round(time.time() - t0, 1), peak_vram_gb=round(peak, 2),
               edge_profile=profile,
               device=(torch.cuda.get_device_name(0) if use_cuda else "cpu"))
    f = res.get("fungi_bio")
    if f is not None and "top_weight" in res:
        t = res["top_weight"]
        pb = _gap(f, t, "rsc")
        out.update(fungi_rsc=f["rsc_mean"], topweight_rsc=t["rsc_mean"], gap=pb["mean_diff"], gap_p=pb["p_one_sided"],
                   dsrsc_ge2_gap=f["dsrsc_ge2_mean"] - t["dsrsc_ge2_mean"],
                   dnsa_ge2hop_gap=f["dnsa_ge2hop_mean"] - t["dnsa_ge2hop_mean"],
                   gap_coexpr_resid_vs_top=_gap(f, t, "rsc_coexpr_resid", seed=3)["mean_diff"])
    if f is not None and "knn" in res:
        k = res["knn"]
        out.update(knn_rsc=k["rsc_mean"], gap_vs_knn=_gap(f, k, "rsc")["mean_diff"],
                   gap_coexpr_resid_vs_knn=_gap(f, k, "rsc_coexpr_resid", seed=4)["mean_diff"],
                   dnsa_ge2hop_gap_vs_knn=f["dnsa_ge2hop_mean"] - k["dnsa_ge2hop_mean"])
    if f is not None and "shuffle" in res:
        out["shuffle_rsc"] = res["shuffle"]["rsc_mean"]; out["fungi_minus_shuffle"] = f["rsc_mean"] - res["shuffle"]["rsc_mean"]
    out["arms"] = {a: {k: v for k, v in res[a].items() if k != "_pp"} for a in res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"; json.dump(out, open(tmp, "w"), indent=2, default=float); os.replace(tmp, args.out)
    print(f"[trial] {c.get('trial_id')} prof={profile} gap={out.get('gap', float('nan')):+.4f} "
          f"gap_vs_knn={out.get('gap_vs_knn', float('nan')):+.4f} peak={peak:.1f}GB ({out['secs']}s)")


if __name__ == "__main__":
    main()
