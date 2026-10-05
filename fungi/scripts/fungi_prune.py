"""
exp_033 — fungi_prune.py : FUNGI size-setter prune of a cached substrate at a chosen lambda (density).

CLONE-ONLY. Adapted from exp_018/025 `fungi_run_full.py`. Reuses a pre-built Phase-0/2 substrate cache
(arrays.npz + meta.pkl) and runs FUNGI Phase-3 (Sobol) [+ optional Phase-5 refinement] + champion select,
writing the champion pruned graph parquet (Regulator/Target/Weight).

The ONLY substrate-independent knob we vary here is the density: the Sobol lambda (edges/gene) search
range. We set hp_cfg["lambda_density"] = [lam_lo/N, lam_hi/N] which get_static_bounds() uses DIRECTLY
(bypassing the static_hi=40 cap in compute_lambda_search_bounds), so we can reach the ~350k arm.
n_edges(champion) == round(lambda_selected * N) empirically (exp_025: lam 33.6276 -> 168,138 edges).

shatter_cfg.max_edge_count (275000 in the cached meta) is raised via --max-edge-count for the high-density arm
so the champion is not flagged is_shattered.

Usage:
  python fungi_prune.py --cache-dir <shroom_cache|hyphae_cache> --tag causal_lam44 \
     --lam-lo 40 --lam-hi 48 [--max-edge-count 400000] [--mode biologic] [--n-samples 8192] [--refine]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse, json, pickle, time, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

EXP = Path(__file__).resolve().parents[1]
CLONE_SRC = Path(os.environ.get("FUNGI_CLONE_SRC", str(EXP / "src")))
sys.path.insert(0, str(CLONE_SRC))
BASE_CFG = Path(os.environ.get("FUNGI_BASE_CFG", str(EXP / "configs" / "fungi_config_base.yaml")))

LIT_BOUNDS = {"alpha": [2.0, 2.8], "gini": [0.58, 0.75], "gini_in": [0.45, 0.60],
              "Q": [0.25, 0.40], "S_max": [0.09, 0.18], "C": [0.08, 0.20], "rho": [-0.35, -0.05],
              "reciprocity": [0.02, 0.12]}
LIVE_TARGETS = ["alpha", "gini_in", "S_max", "C", "rho", "reciprocity"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-base", default=str(EXP / "outputs" / "fungi"))
    ap.add_argument("--lam-lo", type=float, required=True, help="lambda lower bound (edges/gene)")
    ap.add_argument("--lam-hi", type=float, required=True, help="lambda upper bound (edges/gene)")
    ap.add_argument("--max-edge-count", type=int, default=None, help="override shatter_cfg.max_edge_count")
    ap.add_argument("--mode", default=None, help="biologic|synthetic (default: cache meta)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--refine", action="store_true", help="run Phase-5 refinement (default: phase3-only)")
    ap.add_argument("--from-archive", action="store_true")
    # --- exp_035 §11: select an ALTERNATE archive row as the champion -------------------------
    # The default (None) leaves select_diverse_cohort untouched => byte-identical to stock.
    # Needed because G_global_wc03's default champion (row 6660, 72,935 edges) sits at interior
    # margin 0.0038 and FAILS the experiment's own >=0.05 gate, while the equally-0-loss row 5044
    # (55,669 edges) is interior-healthy at 0.0505. Both are utopia_loss=0.0 exactly, so the
    # default pick is an arbitrary tie-break, not a quality ranking.
    ap.add_argument("--select-archive-index", type=int, default=None,
                    help="Rebuild the champion from this phase3_archive row index instead of the "
                         "select_diverse_cohort champion. Requires --from-archive, forbids --refine.")
    ap.add_argument("--champion-suffix", default="",
                    help="Suffix for the champion parquet/json stem so an alternate selection never "
                         "overwrites the default champion artifacts (no-deletion rule).")
    ap.add_argument("--bands-json", default=None,
                    help="override cache utopian_bounds from a JSON {metric:[lo,hi]} (NEG_random_bands)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the Sobol seed (exp_035 P3 seed-robustness). Default None = config value "
                         "= byte-identical to stock.")
    # kernel_firepower_v2 Stage-1 hub-promotion: recompute per_gene_kappa from the cache's G_work at these
    # values, OVERRIDING the baked (3.0/99.0) array. Identity (3.0/99.0) reproduces the baked array
    # byte-identical (regression-verified). No cache rebuild needed.
    ap.add_argument("--hub-multiplier", type=float, default=3.0,
                    help="pagerank_kappa hub_multiplier (identity 3.0). Range [3,20]. Raises S_max.")
    ap.add_argument("--hub-percentile", type=float, default=99.0,
                    help="pagerank_kappa hub_percentile (identity 99.0). Range [90,99]. Lower = more hubs.")
    args = ap.parse_args()
    CACHE = Path(args.cache_dir); OUT_BASE = Path(args.out_base); tag = args.tag
    out_root = OUT_BASE / tag
    out_root.mkdir(parents=True, exist_ok=True)

    arr = np.load(CACHE / "arrays.npz", allow_pickle=False)
    meta = pickle.load(open(CACHE / "meta.pkl", "rb"))
    N = meta["N_GENES"]; gene_names = meta["gene_names"]
    cfg = yaml.safe_load(open(BASE_CFG))

    W_arr = arr["W_arr"]; W_q = arr["W_source_quantile"]; D_arr = arr["D_arr"]
    sources_arr = arr["sources_arr"]; targets_arr = arr["targets_arr"]
    per_gene_kappa = arr["per_gene_kappa"]; source_pert_impact = arr["source_pert_impact"]
    # kernel_firepower_v2 Stage 1: override per_gene_kappa by recomputing PageRank hub multipliers on the
    # cache's OWN G_work at the requested (hub_multiplier, hub_percentile). At identity (3.0/99.0) this
    # reproduces arr["per_gene_kappa"] byte-identical (verified) => regression-safe; the DEFAULT does NOT
    # touch kappa unless a non-identity value is passed.
    if (abs(args.hub_multiplier - 3.0) > 1e-9) or (abs(args.hub_percentile - 99.0) > 1e-9):
        import scipy.sparse as _sp
        from engine import compute_pagerank_kappa_multipliers as _pk
        _Gwork = _sp.csr_matrix((arr["W_arr"].astype(np.float64),
                                 (arr["sources_arr"].astype(np.int64), arr["targets_arr"].astype(np.int64))),
                                shape=(N, N))
        per_gene_kappa = _pk(_Gwork, N, hub_percentile=float(args.hub_percentile),
                             hub_multiplier=float(args.hub_multiplier))
        _nh = int((per_gene_kappa > 1.0).sum())
        log(f"HUB-KAPPA OVERRIDE: hub_multiplier={args.hub_multiplier} hub_percentile={args.hub_percentile} "
            f"-> {_nh} hub genes at kappa_max={per_gene_kappa.max():.2f} (identity 3.0/99.0 = baked, untouched otherwise)")
    chi_prior = arr["chi_prior"]; chi_prior = chi_prior if chi_prior.size else None
    rho_prior = arr["rho_prior"]; rho_prior = rho_prior if rho_prior.size else None
    rdf_prior = arr["rdf_prior"]; er_normalized = arr["er_normalized"]
    inter_mask = arr["inter_mask"]; intra_mask = arr["intra_mask"]
    gene_community_labels = arr["gene_community_labels"]; gene_features = arr["gene_features"]
    W_spectra = arr["W_spectra"]; perturbed_nodes = arr["perturbed_nodes"]
    promiscuity_prior = arr["promiscuity_prior"] if "promiscuity_prior" in arr.files else None  # exp_035 tau_indeg

    utopian_bounds = {k: list(v) for k, v in meta["utopian_bounds"].items()}
    if args.bands_json:   # NEG_random_bands: override baked bands with a (seeded) random draw
        _ov = json.load(open(args.bands_json))
        for _k, _v in _ov.items():
            utopian_bounds[_k] = [float(_v[0]), float(_v[1])]
        log(f"BANDS OVERRIDE from {args.bands_json}: {utopian_bounds}")
    loss_weights = dict(meta["loss_weights"])
    shatter_cfg = dict(meta["shatter_cfg"]); kernel_flags = meta["kernel_flags"]
    er_eta = meta["er_eta"]; spectra_L = meta["spectra_L"]
    lam_eff, lam_q25, lam_q75 = meta["lam_eff"], meta["lam_q25"], meta["lam_q75"]
    fungi_mode = args.mode if args.mode else meta.get("FUNGI_MODE", "biologic")

    # --- density override: set lambda_density directly (edges/gene / N) so get_static_bounds bypasses cap ---
    if args.max_edge_count is not None:
        shatter_cfg["max_edge_count"] = int(args.max_edge_count)
    log(f"tag={tag} mode={fungi_mode} N={N} lam_range=[{args.lam_lo},{args.lam_hi}] edges/gene "
        f"(~{int(args.lam_lo*N):,}..{int(args.lam_hi*N):,} edges)  max_edge_count={shatter_cfg['max_edge_count']:,}")
    log(f"loss_weights={ {k: round(v,2) for k,v in loss_weights.items()} }")

    from search import generate_sobol_samples, SearchEvaluator
    from engine import init_gpu_context, release_gpu_context, build_graph_from_params
    hp_cfg = dict(cfg["hyperparameter_bounds"]); es_cfg = cfg["expansive_search"]
    hp_cfg["lambda_density"] = [args.lam_lo / N, args.lam_hi / N]   # THE density knob (edges/gene fraction)
    # exp_008 levers (exp_035 port): any lever present in hyperparameter_bounds becomes a searched
    # dim (canonical order after m_intra). sigma_scber/zeta_s externalize the baked SCBER/chi_s
    # exponents into searched coefficients. Absent -> no lever, byte-identical to stock.
    _LEVER_ORDER = ["sigma_scber", "m_inter", "eta_out", "zeta_s", "gamma_md", "tau_indeg", "theta_pa"]
    extra_hp_names = [nm for nm in _LEVER_ORDER if nm in hp_cfg]
    externalize_scber = "sigma_scber" in extra_hp_names
    externalize_chi_s = "zeta_s" in extra_hp_names
    if extra_hp_names:
        log(f"exp_008 levers ACTIVE: {extra_hp_names}  externalize_scber={externalize_scber} "
            f"externalize_chi_s={externalize_chi_s}")
    n_samples = args.n_samples if args.n_samples else es_cfg["n_samples"]
    # exp_035 P3: --seed overrides the Sobol seed for seed-robustness runs. Default None => config value
    # (byte-identical to stock). A different seed draws a different Sobol point set (same bounds).
    sobol_seed = args.seed if args.seed is not None else es_cfg["random_seed"]
    if args.seed is not None:
        log(f"SOBOL SEED OVERRIDE: {sobol_seed} (config default {es_cfg['random_seed']})")
    sobol_params, lb, ub = generate_sobol_samples(
        n_genes=N, n_samples=n_samples, hp_cfg=hp_cfg, seed=sobol_seed,
        lam_eff=lam_eff, lam_q25=lam_q25, lam_q75=lam_q75)
    gpu_cfg = cfg.get("gpu", {})
    evaluator = SearchEvaluator(
        W_arr=W_arr, W_q_arr=W_q, D_arr=D_arr, sources_arr=sources_arr, targets_arr=targets_arr,
        n_genes=N, perturbed_nodes=perturbed_nodes, utopian_bounds=utopian_bounds,
        loss_weights=loss_weights, shatter_cfg=shatter_cfg, per_gene_kappa=per_gene_kappa,
        source_pert_impact=source_pert_impact, n_workers=es_cfg["n_workers"], src_dir=str(CLONE_SRC),
        er_scores=er_normalized, er_eta=er_eta, inter_mask=inter_mask, intra_mask=intra_mask,
        chi_prior=chi_prior, rho_prior=rho_prior, chi_t_prior=None, rdf_prior=rdf_prior,
        mode=fungi_mode, deg_matrix_csr=None, gene_community_labels=gene_community_labels,
        kernel_flags=kernel_flags, gene_features=gene_features, spectra_L=spectra_L,
        exact_spectral=False, use_ray_pipeline=gpu_cfg.get("use_ray_pipeline", True))
    evaluator.extra_hp_names = extra_hp_names  # so refinement's secondary rebuild threads the levers

    # kernel_firepower_v2 Stage 3: parent (G_work) per-node OUT-degree, the prior the eta_out lever multiplies.
    # A NEGATIVE eta_out => boosts edges from out-hub sources => concentrates out-degree => LIFTS S_max
    # (the omega-limited blocker Stage 1 could not move). eta_out=0 => (1+outdeg)^0 = 1 => identity.
    parent_outdeg = np.bincount(np.asarray(sources_arr, dtype=np.int64), minlength=N).astype(np.float64)
    if "eta_out" in extra_hp_names:
        log(f"eta_out ACTIVE (out-hub omega boost/penalty): parent_outdeg max={int(parent_outdeg.max())} "
            f"median={int(np.median(parent_outdeg[parent_outdeg>0]))}")
    release_gpu_context()
    gpu_ctx = init_gpu_context(
        W=evaluator.Ws, W_q=evaluator.Wqs, sources=evaluator.srcs, targets=evaluator.tgts, n_genes=N,
        shatter_cfg=shatter_cfg, source_pert_impact=source_pert_impact, k_core_bounds=hp_cfg["k_core"],
        rdf_prior=rdf_prior, er_scores=evaluator.er_scores, er_eta=er_eta, inter_mask=evaluator.inter_mask,
        intra_mask=evaluator.intra_mask, chi_prior=chi_prior, chi_t_prior=None, rho_prior=rho_prior,
        kernel_flags=kernel_flags, n_ffl_buckets=gpu_cfg.get("n_ffl_buckets", 8),
        extra_hp_names=extra_hp_names, externalize_scber=externalize_scber,
        externalize_chi_s=externalize_chi_s, promiscuity_prior=promiscuity_prior,
        parent_outdeg=parent_outdeg)
    if not gpu_ctx.enabled:
        raise RuntimeError("GPU context did NOT initialize.")
    log("GPU ctx ready")

    archive_path = out_root / "phase3_archive.parquet"
    if args.from_archive and archive_path.exists():
        df = pd.read_parquet(archive_path)
        log(f"Phase 3 RELOADED from checkpoint: {len(df):,} graphs")
    else:
        shard_dir = str(out_root / "phase3_shards"); os.makedirs(shard_dir, exist_ok=True)
        t3 = time.time()
        df = evaluator.evaluate(list(sobol_params), shard_dir=shard_dir, desc=f"P3[{tag}]", show_progress=True)
        df.to_parquet(archive_path, index=False)
        log(f"Phase 3 done {(time.time()-t3)/60:.1f} min: {len(df):,} graphs")
    n_viable3 = int((df["is_shattered"] == 0).sum())
    best3 = float(df.loc[df["is_shattered"] == 0, "utopia_loss"].min()) if n_viable3 else float("nan")
    log(f"Phase 3 viable={n_viable3:,} best_loss={best3:.5f}")

    df_ref = None
    if args.refine:
        from refinement import run_ml_gmm_refinement
        t5 = time.time()
        df_ref, was_skipped = run_ml_gmm_refinement(
            df_phase3=df, lower=lb, upper=ub, evaluator=evaluator,
            refinement_cfg=cfg["refinement"], verbose=True, deg_matrix=None,
            mode=fungi_mode, gene_features=gene_features, kernel_flags=kernel_flags,
            spectra_L=spectra_L, secondary_cfg=cfg.get("secondary_objective"), secondary_windows=None)
        if df_ref is not None:
            df_ref.to_csv(out_root / "refinement_results.csv", index=False)
            log(f"Phase 5 done {(time.time()-t5)/60:.1f} min: {len(df_ref):,} evals")

    if args.select_archive_index is not None:
        if args.refine:
            raise SystemExit("--select-archive-index indexes the Phase-3 archive; refuse to combine "
                             "with --refine (refinement rows would shift the index).")
        if args.select_archive_index not in df.index:
            raise SystemExit(f"--select-archive-index {args.select_archive_index} not in archive "
                             f"(index range {df.index.min()}..{df.index.max()})")
        champ = df.loc[args.select_archive_index]
        if bool(champ["is_shattered"]):
            raise SystemExit(f"archive row {args.select_archive_index} is_shattered=True; refusing.")
        log(f"SELECT-OVERRIDE: champion = archive row {args.select_archive_index} "
            f"(loss={float(champ['utopia_loss']):.6f}, n_edges={int(champ['n_edges']):,}) "
            f"-- select_diverse_cohort BYPASSED")
    else:
        from refinement import select_diverse_cohort
        frames = [df] + ([df_ref] if df_ref is not None else [])
        df_all = pd.concat(frames, ignore_index=True)
        cohort = select_diverse_cohort(
            df_all, utopian_bounds, N, cohort_size=5, mode=fungi_mode, evaluator=evaluator,
            gene_features=gene_features, kernel_flags=kernel_flags, spectra_L=spectra_L,
            deg_matrix=None, secondary_cfg=cfg.get("secondary_objective"), secondary_windows=None)
        champ = cohort[cohort["is_champion"]].iloc[0] if "is_champion" in cohort.columns else \
            cohort.sort_values("cohort_rank").iloc[0]

    def g(col):
        return float(champ[col]) if col in champ and np.isfinite(champ[col]) else float("nan")
    metrics = {"alpha": g("alpha"), "gini": g("Gini"), "gini_in": g("gini_in"), "Q": g("Q"),
               "S_max": g("S_max"), "C": g("C"), "rho": g("rho"), "reciprocity": g("reciprocity")}
    champ_loss = float(champ["utopia_loss"])
    params = [float(champ[c]) for c in ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"]]
    # exp_008 levers: append the champion's searched lever values (stored as _lever{j} by the
    # evaluator) in canonical order so build_graph_from_params rebuilds the SCORED graph exactly.
    _lever_defaults = {"sigma_scber": 0.28, "m_inter": 1.0, "eta_out": 0.0, "zeta_s": 0.5, "gamma_md": 0.0,
                       "tau_indeg": 0.0, "theta_pa": 0.0}
    lever_params = []
    for j, nm in enumerate(extra_hp_names):
        col = f"_lever{j}"
        v = float(champ[col]) if (col in champ and np.isfinite(champ[col])) else _lever_defaults[nm]
        lever_params.append(v)
    params = params + lever_params
    if extra_hp_names:
        log(f"CHAMPION levers: {dict(zip(extra_hp_names, [round(x,4) for x in lever_params]))}")
    log(f"CHAMPION loss={champ_loss:.5f} lambda={params[4]:.3f} n_edges={int(champ['n_edges']):,} "
        f"alpha={metrics['alpha']:.3f} gini_in={metrics['gini_in']:.3f} S_max={metrics['S_max']:.3f} "
        f"C={metrics['C']:.3f} rho={metrics['rho']:.4f} recip={metrics['reciprocity']:.4f}")

    surv_s, surv_t, surv_W = build_graph_from_params(
        params, evaluator.Ws, evaluator.Wqs, evaluator.Ds, evaluator.srcs, evaluator.tgts, N,
        perturbed_nodes, shatter_cfg, per_gene_kappa, source_pert_impact, md_gate=None,
        er_scores=evaluator.er_scores, er_eta=er_eta, inter_mask=evaluator.inter_mask,
        chi_prior=chi_prior, rho_prior=rho_prior, chi_t_prior=None, rdf_prior=rdf_prior,
        kernel_flags=kernel_flags, intra_mask=evaluator.intra_mask,
        extra_hp_names=extra_hp_names, promiscuity_prior=promiscuity_prior,
        parent_outdeg=parent_outdeg)
    codes_raw = sources_arr.astype(np.int64) * N + targets_arr.astype(np.int64)
    order_c = np.argsort(codes_raw, kind="stable")
    codes_s, wspec_s = codes_raw[order_c], W_spectra[order_c]
    q = surv_s.astype(np.int64) * N + surv_t.astype(np.int64)
    pos = np.clip(np.searchsorted(codes_s, q), 0, len(codes_s) - 1)
    hit = codes_s[pos] == q
    weight_out = np.zeros(len(surv_s), dtype=np.float64); weight_out[hit] = wspec_s[pos[hit]]
    out_df = pd.DataFrame({"Regulator": [gene_names[i] for i in surv_s],
                           "Target": [gene_names[i] for i in surv_t],
                           "Weight": weight_out.astype(np.float32)})
    pruned_dir = OUT_BASE / "champions_full"; pruned_dir.mkdir(parents=True, exist_ok=True)
    champ_path = pruned_dir / f"{tag}{args.champion_suffix}.parquet"
    out_df.to_parquet(champ_path, index=False)
    log(f"Champion parquet -> {champ_path} ({len(out_df):,} rebuilt edges)")

    within_lit = {k: bool(LIT_BOUNDS[k][0] <= metrics[k] <= LIT_BOUNDS[k][1])
                  for k in LIT_BOUNDS if np.isfinite(metrics.get(k, float("nan")))}
    rec = {"tag": tag, "mode": fungi_mode, "run": "full_refinement" if args.refine else "phase3_only",
           "lam_lo": args.lam_lo, "lam_hi": args.lam_hi, "n_samples_phase3": n_samples,
           "utopia_loss": champ_loss, "phase3_best_loss": best3, "n_edges": int(len(surv_s)),
           "champion_lambda": params[4], "n_viable_phase3": n_viable3, "best_params": params,
           "metrics": metrics, "within_lit": within_lit,
           "within_lit_count": int(sum(within_lit.values())),
           "max_edge_count": int(shatter_cfg["max_edge_count"]),
           "extra_hp_names": list(extra_hp_names),
           "hp_names": ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"] + list(extra_hp_names),
           "hp_lower": [float(x) for x in lb], "hp_upper": [float(x) for x in ub],
           "champion_parquet": str(champ_path), "cache_dir": str(CACHE),
           "select_archive_index": args.select_archive_index}
    json.dump(rec, open(out_root / f"{tag}{args.champion_suffix}_champion.json", "w"), indent=2, default=str)
    log(f"within-lit {rec['within_lit_count']}/8  utopia_loss={champ_loss:.5f}  n_edges={rec['n_edges']:,}")

    release_gpu_context()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    log(f"{tag} DONE.")


if __name__ == "__main__":
    main()
