"""
exp_024 Part A — FUNGI Phase-0/2 cache builder for the ALTERNATIVE dense substrates (coexpr/ggm/random_sym).

Same as exp_018's fungi_build_substrate.py (CLONE-ONLY; FUNGI/src never imported except via clone), but:
  * accepts --matrix (a 4627x4627 .npy dense substrate) instead of a 21M-row parquet;
  * MONITORS FUNGI's Phase-0 diagnostic probes on these non-PSGRN substrates: logs the probe-derived
    utopian_bounds side-by-side with the RPE1 5k literature bands (exp_004d/exp_010) and flags divergence,
    writing intermediate/fungi_partA/probe_bounds_<tag>.json. Per the exp_024 design's Diagnostics-safety
    note + Patrick: if the probes propose unusual/poor bounds, fall back to the literature override (rebuild
    with FUNGI_BASE_CFG=clone/fungi_config_litbounds.yaml, which forces the literature bands via bounds_override).

The config's bounds_override / organic_targets set the objective (exp_010 lock: gini_out+Q disabled,
reciprocity enabled). sc_data_path (File A hybrid h5ad) is read from the config. CPU-heavy (Phase 0) — run
serial, never overlapping the GPU Phase-3 (freeze rule).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse, sys, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import scipy.sparse as sp
from scipy.stats import rankdata

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
FUNGI_DIR = REPO / "FUNGI"
CLONE_SRC = Path(os.environ.get("FUNGI_CLONE_SRC", str(EXP / "src")))
sys.path.insert(0, str(CLONE_SRC))
BASE_CFG = Path(os.environ.get("FUNGI_BASE_CFG", str(EXP / "configs" / "fungi_config_base.yaml")))

# RPE1 5k literature bands (exp_004d / exp_010 / fungi_run_full LIT_BOUNDS)
LIT = {"alpha": [2.0, 2.8], "gini_in": [0.45, 0.60], "S_max": [0.09, 0.18],
       "C": [0.08, 0.20], "rho": [-0.35, -0.05], "reciprocity": [0.02, 0.12]}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", help="npy 4627x4627 dense substrate (mat[i,j]=weight i->j)")
    ap.add_argument("--graph", help="dense parquet (alternative to --matrix)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    CACHE = Path(args.cache_dir); CACHE.mkdir(parents=True, exist_ok=True)
    MON = EXP / "outputs" / "fungi_partA"; MON.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = yaml.safe_load(open(BASE_CFG))
    FUNGI_MODE = cfg.get("fungi_mode", "biologic").lower()
    log(f"tag={args.tag} fungi_mode={FUNGI_MODE} base_cfg={BASE_CFG.name}")

    import scanpy as sc
    sc_path = Path(cfg["input"]["sc_data_path"])
    if not sc_path.is_absolute():
        sc_path = (FUNGI_DIR / sc_path).resolve()
    adata = sc.read_h5ad(str(sc_path))
    N_GENES = adata.n_vars
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    log(f"adata: {adata.n_obs:,} cells x {N_GENES:,} genes")

    if args.matrix:
        mat = np.load(args.matrix).astype(np.float64)
        assert mat.shape == (N_GENES, N_GENES), f"matrix {mat.shape} != ({N_GENES},{N_GENES})"
        np.fill_diagonal(mat, 0.0)
        raw_sparse_mat = sp.csr_matrix(mat)   # raw[i,j] = weight i->j (source=row, target=col)
        raw_sparse_mat.eliminate_zeros()
    else:
        gdf = pd.read_parquet(args.graph)
        _c = {c.lower(): c for c in gdf.columns}
        sc_col = _c.get("regulator", _c.get("source", gdf.columns[0]))
        tc = _c.get("target", gdf.columns[1]); wc = _c.get("importance", _c.get("weight", gdf.columns[2]))
        rows = gdf[sc_col].map(gene_to_idx).to_numpy(); cols = gdf[tc].map(gene_to_idx).to_numpy()
        vals = gdf[wc].to_numpy(dtype=np.float64); mask = ~(pd.isna(rows) | pd.isna(cols))
        raw_sparse_mat = sp.csr_matrix((vals[mask], (rows[mask].astype(int), cols[mask].astype(int))),
                                       shape=(N_GENES, N_GENES)); raw_sparse_mat.eliminate_zeros()
    log(f"Parent GRN: {raw_sparse_mat.nnz:,} edges, density {raw_sparse_mat.nnz/N_GENES**2:.4%}")

    from utils import build_gene_features, build_kernel_flags
    gene_features = build_gene_features(adata, n_components=50, seed=42)
    kernel_flags = build_kernel_flags(cfg)
    spectra_L = int(cfg.get("synthetic_targets", {}).get("spectra_depth_L", 3))

    from diagnostics import run_diagnostics, build_shatter_config
    log("Phase 0 diagnostics ...")
    utopian_bounds, loss_weights, diagnostic_report = run_diagnostics(
        adata=adata, n_genes=N_GENES, cfg_diagnostics=cfg["diagnostics"], cfg_input=cfg["input"],
        raw_sparse_mat=raw_sparse_mat, lambda_user_cfg=cfg.get("lambda_user"),
        bounds_override_cfg=cfg.get("bounds_override"), organic_targets_cfg=cfg.get("organic_targets"))
    lam_eff = diagnostic_report["lam_eff"]; lam_q25 = diagnostic_report["lam_q25"]; lam_q75 = diagnostic_report["lam_q75"]

    # ---- MONITOR: probe bounds vs literature ----
    mon = {"tag": args.tag, "base_cfg": BASE_CFG.name, "probe_bounds": {},
           "literature_bounds": LIT, "divergence_flags": {}}
    log("=== PROBE-DERIVED BOUNDS vs RPE1 LITERATURE (monitoring) ===")
    for k in ["alpha", "gini_in", "S_max", "C", "rho", "reciprocity"]:
        pb = [round(float(utopian_bounds[k][0]), 4), round(float(utopian_bounds[k][1]), 4)] if k in utopian_bounds else None
        mon["probe_bounds"][k] = pb
        lit = LIT[k]
        # flag if probe center is outside the literature band (rough divergence heuristic)
        diverge = False
        if pb is not None:
            pc = 0.5 * (pb[0] + pb[1]); lc = 0.5 * (lit[0] + lit[1]); lw = lit[1] - lit[0]
            diverge = abs(pc - lc) > lw   # center off by more than one literature-band width
        mon["divergence_flags"][k] = bool(diverge)
        flag = "  <-- DIVERGES" if diverge else ""
        log(f"  {k:12s} probe={pb}  lit={lit}{flag}")
    mon["any_divergence"] = bool(any(mon["divergence_flags"].values()))
    mon["lam_eff"] = float(lam_eff)
    json.dump(mon, open(MON / f"probe_bounds_{args.tag}.json", "w"), indent=2)
    log(f"loss_weights={ {k:round(v,2) for k,v in loss_weights.items()} }")
    log(f"ANY DIVERGENCE: {mon['any_divergence']} (see probe_bounds_{args.tag}.json)")

    # ---- Phase 2 (same as exp_018 builder) ----
    from filtering import adaptive_threshold_filter
    G_work = adaptive_threshold_filter(raw_sparse_mat, target_density=cfg["prefilter"]["target_density"])
    G_work_coo = G_work.tocoo()
    sources_raw = G_work_coo.row.copy(); targets_raw = G_work_coo.col.copy(); weights_raw = G_work_coo.data.copy()
    log(f"G_work: {G_work.nnz:,} edges (density {G_work.nnz/N_GENES**2:.4%})")

    W_ranked = rankdata(weights_raw, method="average").astype(np.float64) / len(weights_raw)
    if W_ranked.max() - W_ranked.min() < 1e-6:
        W_log = np.log1p(weights_raw.astype(np.float64))
        W_arr = (np.clip(W_log / max(W_log.max(), 1e-10), 0.001, 1.0) if W_log.max() > 1e-12
                 else np.random.default_rng(42).uniform(0.01, 1.0, len(weights_raw)))
    else:
        W_arr = W_ranked
    _w_raw_max = float(weights_raw.max())
    W_spectra = (np.log1p(weights_raw.astype(np.float64)) / np.log1p(_w_raw_max) if _w_raw_max > 0 else W_arr.copy())
    sources_arr = sources_raw.astype(np.int32); targets_arr = targets_raw.astype(np.int32)

    from engine import (compute_source_quantile_weights, compute_pagerank_kappa_multipliers,
                        compute_source_pert_impact, compute_chi_prior, compute_deg_row_prior, compute_rdf_prior)
    W_source_quantile = compute_source_quantile_weights(sources_arr, W_arr, N_GENES)
    D_arr = np.zeros(len(W_arr), dtype=np.float64)
    _pkc = cfg["pagerank_kappa"]
    per_gene_kappa = compute_pagerank_kappa_multipliers(
        G_work, N_GENES, alpha=_pkc.get("alpha", 0.85), n_iter=_pkc.get("n_iter", 60),
        hub_percentile=_pkc.get("hub_percentile", 99.0), hub_multiplier=_pkc.get("hub_multiplier", 3.0))
    source_pert_impact = compute_source_pert_impact(
        np.array(diagnostic_report["_impact_array"], dtype=np.float64),
        diagnostic_report["_perturbation_labels"], diagnostic_report["_name_to_idx"], N_GENES)

    chi_cfg = cfg.get("chi_prior", {}); zeta = float(chi_cfg.get("zeta", 1.0))
    deg_col_sums = np.array(diagnostic_report.get("_deg_col_sums", []), dtype=np.float64)
    deg_row_sums = np.array(diagnostic_report.get("_deg_row_sums", []), dtype=np.float64)
    deg_out_parent = np.asarray(raw_sparse_mat.sum(axis=1)).ravel().astype(np.float64)
    chi_prior = (compute_chi_prior(deg_col_sums, N_GENES, zeta=zeta)
                 if chi_cfg.get("enabled", True) and len(deg_col_sums) == N_GENES else None)
    rho_prior = (compute_deg_row_prior(deg_row_sums=deg_row_sums, deg_out_parent=deg_out_parent,
                 perturbation_labels=np.array(diagnostic_report["_perturbation_labels"]),
                 name_to_idx=diagnostic_report["_name_to_idx"], n_genes=N_GENES)
                 if len(deg_row_sums) == N_GENES else None)

    from effective_resistance import compute_scber_scores
    er_cfg = cfg.get("effective_resistance", {})
    er_normalized, inter_mask, er_raw, scber_diag = compute_scber_scores(
        G_work, sources_arr, targets_arr, cfg=er_cfg, gene_features=gene_features)
    er_eta = float(er_cfg.get("eta_inter", 0.20))
    gene_community_labels = scber_diag.get("membership") if scber_diag else None
    intra_mask = (~inter_mask) if inter_mask is not None else None
    log(f"SCBER: inter_frac={inter_mask.mean():.3f} n_comm={int(np.max(gene_community_labels))+1}")

    rdf_prior = compute_rdf_prior(sources_arr, targets_arr, W_arr, gene_features, N_GENES, top_n=50)

    from search import get_static_bounds
    lower_bounds, upper_bounds, hp_names = get_static_bounds(
        N_GENES, cfg["hyperparameter_bounds"], lam_eff=lam_eff, lam_q25=lam_q25, lam_q75=lam_q75)
    shatter_cfg = build_shatter_config(cfg["shatter"], N_GENES, utopian_bounds,
                                       [lower_bounds[4], upper_bounds[4]], mode=FUNGI_MODE)
    shatter_cfg["min_reach_frac"] = cfg.get("synthetic_targets", {}).get("min_reach_frac", 0.5)

    pert_col = cfg["input"]["perturbation_column"]; ctrl_label = cfg["input"]["control_label"]
    pert_genes = [g for g in adata.obs[pert_col].unique() if g != ctrl_label]
    perturbed_nodes = np.array([gene_names.index(g) for g in pert_genes
                                if g in gene_to_idx and gene_to_idx[g] < N_GENES], dtype=int)
    log(f"perturbed_nodes: {len(perturbed_nodes):,}")

    # exp_035 Task 4: per-target promiscuity prior = G_work in-degree (# strong edges pointing into each
    # target). Promiscuous housekeeping/cell-cycle hubs (targets of many co-expression edges) score high;
    # the tau_indeg lever down-weights edges into them to spread in-degree (attack gini_in).
    promiscuity_prior = np.bincount(targets_arr.astype(np.int64), minlength=N_GENES).astype(np.float64)
    log(f"promiscuity_prior (G_work in-degree): max={promiscuity_prior.max():.0f} "
        f"median={np.median(promiscuity_prior):.0f} n_hub(>2*med)={(promiscuity_prior>2*np.median(promiscuity_prior)).sum()}")
    np.savez_compressed(
        CACHE / "arrays.npz", W_arr=W_arr, W_source_quantile=W_source_quantile, D_arr=D_arr,
        sources_arr=sources_arr, targets_arr=targets_arr, per_gene_kappa=per_gene_kappa,
        source_pert_impact=source_pert_impact,
        chi_prior=(chi_prior if chi_prior is not None else np.array([])),
        rho_prior=(rho_prior if rho_prior is not None else np.array([])),
        rdf_prior=rdf_prior, er_normalized=er_normalized,
        inter_mask=inter_mask.astype(bool), intra_mask=intra_mask.astype(bool),
        gene_community_labels=np.asarray(gene_community_labels, dtype=np.int64),
        gene_features=gene_features, W_spectra=W_spectra, promiscuity_prior=promiscuity_prior,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds, perturbed_nodes=perturbed_nodes)
    meta = dict(N_GENES=int(N_GENES), gene_names=gene_names,
                utopian_bounds={k: [float(v[0]), float(v[1])] for k, v in utopian_bounds.items()},
                loss_weights={k: float(v) for k, v in loss_weights.items()},
                shatter_cfg=shatter_cfg, kernel_flags=kernel_flags, er_eta=er_eta,
                lam_eff=float(lam_eff), lam_q25=float(lam_q25), lam_q75=float(lam_q75),
                hp_names=hp_names, spectra_L=spectra_L, FUNGI_MODE=FUNGI_MODE, chi_zeta=zeta,
                has_chi=chi_prior is not None, has_rho=rho_prior is not None,
                n_comm=int(np.max(gene_community_labels)) + 1,
                graph_path=str(args.matrix or args.graph), sc_data_path=str(sc_path))
    pickle.dump(meta, open(CACHE / "meta.pkl", "wb"))
    json.dump({k: v for k, v in meta.items() if k != "gene_names"},
              open(CACHE / "meta.json", "w"), indent=2, default=str)
    log(f"substrate cache -> {CACHE} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
