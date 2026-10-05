"""
FUNGI v15.3 — Phase 0: Data-Driven Diagnostic Calibration

Changes from v15.2 (v15.1):
  ─────────────────────────────────────────────────────────────────────
  Q PROBE: Parent-graph slice replaces DEG matrix substrate (v15.3)
  ─────────────────────────────────────────────────────────────────────
  v15.2 used the DEG matrix as the primary Q substrate. The DEG matrix
  on VCC_4c has 73,479 directed edges, which symmetrises to ~40-50k
  undirected edges — far sparser than DASH output (~190k directed →
  ~110-130k undirected). At that lower density, Leiden modularity gives
  Q ≈ 0.44-0.52. Combined with delta_min=0.10 in bound enforcement, this
  produced exactly [0.4500, 0.5500] on every dataset.

  DASH achieves Q ≈ 0.17-0.18 at lam_eff density. The v15.2 target was
  structurally unreachable and dominated the loss at weight 21.4.

  The fix: use the parent-graph top-lam_eff slice (Step 1, already
  extracted as rows_sel/cols_sel/vals_sel) as the sole substrate, max-
  weight symmetrised. This sits at the same density as DASH output and
  yields Q ≈ 0.15-0.22 — a target DASH can satisfy.

  SEED FIX (v15.3): community_leiden now receives random_state=seed so
  the 3-seed loop produces genuine stochastic variation (it was a no-op
  previously, giving IQR=0 → conf=0.90 artificially).

Changes from v15.1 (v15.2):
  ─────────────────────────────────────────────────────────────────────
  Q PROBE: Winner-selection IQR replaces cross-resolution IQR (v15.2)
  ─────────────────────────────────────────────────────────────────────
  v15.0 collected IQR across all 9 runs (3 resolutions × 3 seeds).
  On a ~100k-edge parent-graph substrate (mean degree ~40), different
  resolutions produce structurally different Q values by design — lower
  resolution → fewer communities → higher Q.  The cross-resolution span
  consistently exceeded 0.20 Q units, making conf < 0.40 always true
  and unconditionally triggering the fixed prior (0.45, 0.55, 0.55).

  The fix: adopt SCBER's winner-selection strategy exactly.
    1. Collect Q values per resolution (3 seeds each).
    2. Pick the resolution whose median Q is closest to TARGET_Q=0.5.
    3. Report center = median of the winning resolution's Q values.
    4. Compute IQR within the winning resolution only (across 3 seeds).

  Within-resolution IQR measures genuine stochastic noise in Leiden's
  refinement phase, which is near zero for well-structured parent graphs
  (expected IQR ≈ 0.00–0.05).  This gives conf ≈ 0.75–0.90 on real
  data, matching the SCBER reference of Q=0.558 on the VCC_2 Classic
  parent graph.

Changes from v14.2 (v15.0 substrate fix still present):
  ─────────────────────────────────────────────────────────────────────
  Q PROBE: Parent-graph Leiden-Modularity replaces LFC-causal CPM
  ─────────────────────────────────────────────────────────────────────
  v11–v14.2 all shared the same substrate mistake: the LFC-causal graph
  is a bipartite projection (108 TF sources × 5,024 gene targets).
  After symmetrisation it produces a structurally bipartite-like
  adjacency with ~108 high-degree hub genes and ~4,916 low-degree
  targets.  No community-detection algorithm — Louvain, CPM, or any
  Leiden objective — can reliably partition this into the 6–20 macro-
  modules that correspond to biological reality, because:

    1. The bipartite structure violates the unipartite assortative-
       mixing assumption underlying all modularity-family metrics.
    2. IDF weights span 3 orders of magnitude (p25=0.06, p99=0.62),
       creating ~558× density contrast between local triangles and
       macro-community backbones. CPM's absolute density threshold
       cascades into micro-island fragmentation at every γ value.
    3. The SCBER module already solves this: it runs Leiden-Modularity
       on the symmetrised top-λ parent graph and consistently obtains
       Q ≈ 0.44–0.56, 7–14 communities on VCC_2 (Q=0.558, 7 comms).

  The fix: make _probe_Q do exactly what SCBER does.

  NEW SUBSTRATE — top-λ_eff parent graph (raw_sparse_mat):
    • Gene-gene LightGBM scores; not bipartite.
    • Weight distribution compressed by LightGBM to ~[0,1] with no
      3-order IDF spread → Leiden-Modularity's degree-preserving null
      model handles scale-free hubs correctly.
    • Same object DASH optimises; measuring Q here is measuring the
      same quantity as the DASH loss target.

  NEW ALGORITHM — igraph community_leiden(objective='modularity'):
    • Standard Newman modularity null model: penalises edges relative
      to degree-sequence expectation.
    • Leiden refinement prevents Louvain's disconnected-community
      pathology while keeping well-connected modules.
    • Resolution sweep [0.3, 0.5, 0.8] × 3 seeds = 9 runs, identical
      to SCBER's _detect_communities().  Winner-selection: pick the
      run whose Q is closest to 0.5; IQR across all 9 runs calibrates
      confidence.
    • No leidenalg required — uses igraph's built-in C++ Leiden
      (igraph ≥ 0.10 / 1.0.0+).  Falls back to igraph multilevel
      Louvain only when community_leiden is unavailable; if Louvain
      IQR ≥ 0.20, activates fixed prior (0.45, 0.55, 0.55).

  CAPABILITY DETECTION updated to test ModularityVertexPartition /
  community_leiden(objective='modularity') instead of CPM — since
  leidenalg CPM is no longer used.

  GENERALISES AUTOMATICALLY:
    • lam_eff already scales with dataset via computelameff; the
      substrate edge count (lam_eff × n_genes) therefore adapts to
      RPE1 and K562 without any manual tuning.
    • Different parent graphs (Classic vs Experimental) produce
      different Q targets, preserving dataset-specificity.

  Expected output for VCC_2 (parent graph, mean degree ~40 at λ_eff):
    conf ≈ 0.65–0.85, bounds ≈ [0.46, 0.56], center ≈ 0.50–0.52.
    Matches SCBER diagnostic Q=0.558 on the same substrate.

  ─────────────────────────────────────────────────────────────────────
  UNCHANGED from v14.2
  ─────────────────────────────────────────────────────────────────────
  All other probes: _probe_alpha, _probe_gini, _probe_smax, _probe_C,
  _probe_rho. _enforce_bound_constraints, build_impact_array,
  build_shatter_config. Public API signature unchanged.
  The degenerate-partition filter (n_comm > n_genes // 10) is retained
  as a secondary safety net but is no longer the primary defence.

Public API:
  run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                  raw_sparse_mat=None, lambda_user_cfg=None)
  build_impact_array(...)
  build_shatter_config(...)
"""

import gc
import warnings
# v3.0: silence scanpy's "DataFrame is highly fragmented" PerformanceWarning,
# which floods the log during rank_genes_groups (cosmetic only).
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from scipy.stats import spearmanr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)


PHYSICAL = {
    "alpha": (1.2, 3.5),
    "gini":  (0.30, 0.95),
    "gini_in": (0.20, 0.95),
    "S_max": (0.005, 0.30),
    "C":     (0.001, 0.30),
    "rho":   (-0.50, 0.15),
    # Q (modularity, restored session 5): matches bound_constraints.Q's
    # hard_floor/hard_ceiling in fungi_config.yaml.
    "Q":     (-0.50, 1.00),
}

FALLBACK_BOUNDS = {
    "alpha": [1.50, 1.90],
    "gini":  [0.62, 0.85],
    "gini_in": [0.45, 0.70],
    "S_max": [0.04, 0.12],
    "C":     [0.02, 0.08],
    "rho":   [-0.20, -0.04],
    # Q fallback (session 5): centered on the actual resolution=1.3 Q_igraph
    # measured on the real SHROOM_2 G_work substrate in session 4 (~0.026,
    # 304-307 communities) -- NOT the historical [0.10, 0.25] (parked-era) or
    # [0.30, 0.70] (pre-session-4 SCBER-coupled) constants, both calibrated
    # for sparser graphs or a broken community detector. Used only when the
    # probe's own Leiden run on THIS dataset is degenerate.
    "Q":     [0.015, 0.040],
    # reciprocity (exp_010): directed link reciprocity literature band
    # (Garlaschelli & Loffredo 2004; directed TRNs ~2.4% bidirectional). A pure
    # PRIOR -- no dense-graph probe; the band IS the target. Active only when
    # organic_targets.reciprocity.enabled (+ bounds_override.reciprocity).
    "reciprocity": [0.02, 0.12],
}
FALLBACK_CONFIDENCE = 0.25
FALLBACK_CONFIDENCE_Q = 0.40  # moderate, not the old parked conf=0.01

# exp_004d promotion (session 12): the established PSGRN Self-Train scale-free
# exponent, used as the biological anchor for gini_out's Pareto-consistent
# ceiling and S_max's natural-cutoff prior. A literature/biology anchor, NOT a
# substrate-measurement-derived fudge factor. = the exp_004c gate's --alpha-hat.
ALPHA_ANCHOR = 2.3022

MIN_SKELETON_EDGES = 500
LFC_CAUSAL_FLOOR = 0.01


def _safe_float(val, fallback=0.0):
    if val is None or not np.isfinite(val):
        return fallback
    return float(val)


def _clip_bound(lo, hi, param):
    floor, ceiling = PHYSICAL[param]
    lo = float(np.clip(lo, floor, ceiling))
    hi = float(np.clip(hi, floor, ceiling))
    if lo >= hi:
        center = (lo + hi) / 2.0
        eps = (ceiling - floor) * 0.02
        lo = max(center - eps, floor)
        hi = min(center + eps, ceiling)
    return lo, hi


def _gini(x):
    x = np.asarray(x, dtype=np.float64)
    if x.sum() == 0:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    n = len(x)
    cum_w = np.arange(1, n + 1) / n
    cum_v = np.cumsum(x) / x.sum()
    return float(1.0 - 2.0 * np.trapezoid(cum_v, cum_w))


def _weighted_percentile(values, weights, percentile):
    """Weighted percentile via sorted cumulative weights."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or weights.sum() == 0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_w = np.cumsum(weights) / weights.sum()
    idx = np.searchsorted(cum_w, percentile / 100.0)
    idx = int(np.clip(idx, 0, len(values) - 1))
    return float(values[idx])


# ---------------------------------------------------------------------------
# λ_eff: specificity_weighted_deg_count  [REPLACED from lfc_effective_rank]
# ---------------------------------------------------------------------------

def _compute_lam_eff(deg_matrix, lfc_matrix, valid_cq, name_to_idx,
                     n_active, n_tested, n_genes,
                     sample_weights=None):
    """
    Estimate expected edges/gene using a cascade-corrected, specificity-
    weighted DEG count. Returns (lam_center, lam_q25, lam_q75, erank_diagnostic).

    PRIMARY ESTIMATOR — specificity_weighted_deg_count
    -------------------------------------------------------
    For each active perturbation i, count how many genes it regulates, but
    discount genes that respond to many perturbations (likely indirect/cascade
    targets) using an IDF-style √-denominator:

        direct_reach_i = Σ_j DEG[i,j] / max(1, prevalence_j)^0.5

    where prevalence_j = number of perturbations that produce a DEG at gene j.
    The γ=0.5 exponent retains some signal from combinatorially-regulated genes
    while down-weighting pure cascade effectors.

    λ_center = clip(weighted_median(direct_reach_i), 4.0, 40.0)
    λ_q25, λ_q75 are saved for the probe substrate and search-range context.
    Clip is [4, 40]: an outer fence for degenerate inputs, not a design target.

    SECONDARY DIAGNOSTIC — lfc_effective_rank (Gavish-Donoho)
    -------------------------------------------------------
    Retained for telemetry only. Measures the effective number of independent
    regulatory programs (Roy & Vetterli 2007). Uses the Gavish-Donoho optimal
    hard threshold (2014) instead of the arbitrary top-50 truncation.
    NEVER used to set any probe substrate or optimisation search bound.
    """
    lam_q25 = 6.0
    lam_q75 = 20.0
    erank_diag = None

    # --- Primary estimator ---------------------------------------------------
    if (deg_matrix is not None and deg_matrix.nnz > 0 and n_active >= 5
            and len(valid_cq) >= 5):
        try:
            # prevalence_j: how many perturbations produce a DEG at each gene
            prevalence = np.asarray(deg_matrix.sum(axis=0)).ravel().astype(np.float64)

            direct_reach_vals = []
            direct_reach_weights = []

            for k, gene_name in enumerate(valid_cq):
                src_idx = name_to_idx.get(str(gene_name))
                if src_idx is None:
                    continue
                # Row of DEG matrix for this perturbation
                deg_row = np.asarray(deg_matrix[src_idx, :].toarray()).ravel()
                # IDF-style specificity weighting: 1 / sqrt(prevalence)
                denom = np.maximum(prevalence, 1.0) ** 0.5
                reach = float(np.sum(deg_row / denom))
                if reach > 0:
                    w = sample_weights[k] if (sample_weights is not None
                                              and k < len(sample_weights)) else 1.0
                    direct_reach_vals.append(reach)
                    direct_reach_weights.append(float(w))

            if len(direct_reach_vals) >= 5:
                dr = np.array(direct_reach_vals, dtype=np.float64)
                dw = np.array(direct_reach_weights, dtype=np.float64)

                lam_swdc_raw   = _weighted_percentile(dr, dw, 50.0)
                lam_q25        = _weighted_percentile(dr, dw, 25.0)
                lam_q75        = _weighted_percentile(dr, dw, 75.0)

                # --- Gavish-Donoho effective rank (v15.5: co-primary) ------
                erank_diag = _compute_erank_gd(lfc_matrix)

                # v15.5: MIN-OF-TWO estimator. The IDF-weighted reach
                # (specificity_weighted_deg_count) overshoots on targeted TF
                # panels because the 108 perturbation genes are selected
                # to be high-impact regulators whose median reach far exceeds
                # the average gene's. The SVD effective rank overshoots on
                # genome-scale panels (K562: erank=61, optimal density=12).
                # Taking the min automatically selects the conservative
                # (data-appropriate) estimator for each dataset type:
                #   VCC targeted: min(60+, 28) = 28  (erank wins)
                #   K562 genome:  min(13, 61)  = 13  (IDF wins)
                lam_erank = float(erank_diag) if erank_diag is not None else lam_swdc_raw
                lam_center_raw = min(lam_swdc_raw, lam_erank)

                lam_center = float(np.clip(lam_center_raw, 4.0, 50.0))
                lam_q25    = float(np.clip(lam_q25, 4.0, 40.0))
                lam_q75    = float(np.clip(lam_q75, 4.0, 60.0))

                # Ensure sensible ordering after clipping
                if lam_q25 > lam_center:
                    lam_q25 = lam_center
                if lam_q75 < lam_center:
                    lam_q75 = lam_center

                return lam_center, lam_q25, lam_q75, erank_diag
        except Exception:
            pass

    # --- Fallback: active-fraction heuristic (kept from v9 inner fallback) ---
    active_frac = float(n_active) / float(max(n_tested, 1))
    panel_coverage = float(n_tested) / float(max(n_genes, 1))
    if panel_coverage < 0.1:
        lam_fb = float(np.clip(8.0 + 8.0 * active_frac, 8.0, 18.0))
    else:
        lam_fb = float(np.clip(5.0 + 7.0 * active_frac, 5.0, 14.0))

    lam_q25 = max(lam_fb * 0.6, 4.0)
    lam_q75 = min(lam_fb * 1.4, 60.0)  # wider: Q75 informs search ceiling only
    erank_diag = _compute_erank_gd(lfc_matrix)
    return lam_fb, lam_q25, lam_q75, erank_diag


def _compute_erank_gd(lfc_matrix):
    """
    Secondary diagnostic only: SVD effective rank (Roy & Vetterli 2007)
    with Gavish-Donoho optimal hard threshold (2014) replacing arbitrary
    top-50 truncation.

    Returns the effective rank as a float, or None on failure.
    This value is logged but NEVER drives probe substrates or bounds.
    """
    if lfc_matrix is None or lfc_matrix.shape[0] < 10 or lfc_matrix.shape[1] < 10:
        return None
    try:
        L = lfc_matrix.astype(np.float64)
        m, n = L.shape
        # All singular values (no truncation yet)
        _, sv_all, _ = np.linalg.svd(L, full_matrices=False)
        sv_all = sv_all[sv_all > 1e-10]
        if len(sv_all) < 3:
            return None

        # Gavish-Donoho optimal hard threshold (unknown noise level form)
        # ω(β) polynomial approximation from Gavish & Donoho 2014
        beta = float(min(m, n)) / float(max(m, n))
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        y_med = float(np.median(sv_all))
        if y_med < 1e-12:
            return None
        threshold = omega * y_med

        sv_signal = sv_all[sv_all > threshold]

        # Fallback: 90% energy fraction if GD is too aggressive
        if len(sv_signal) < 3:
            sv_sq = sv_all ** 2
            cum_energy = np.cumsum(sv_sq) / sv_sq.sum()
            k90 = int(np.searchsorted(cum_energy, 0.90)) + 1
            sv_signal = sv_all[:max(k90, 3)]

        sv_norm = sv_signal / sv_signal.sum()
        entropy = -float(np.sum(sv_norm * np.log(sv_norm + 1e-12)))
        erank = float(np.exp(entropy))
        return round(erank, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Impact array + DEG matrix + LFC matrix builder  [UNCHANGED]
# ---------------------------------------------------------------------------

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25, n_jobs=6,
                       max_perts_for_de=500, min_cells_per_pert=5,
                       is_metacell=False, metacell_pooling_factor=None):
    print("Phase 0: Building perturbation impact array...")

    if is_metacell and metacell_pooling_factor and metacell_pooling_factor > 1:
        effective_lfc = lfc_threshold / np.sqrt(metacell_pooling_factor)
        print(f"  Metacell pooling (factor={metacell_pooling_factor}): "
              f"LFC cutoff {lfc_threshold:.3f} → {effective_lfc:.3f}")
    else:
        effective_lfc = lfc_threshold

    conditions_arr = adata.obs[perturbation_column].values
    ctrl_mask = conditions_arr == control_label
    unique_conds = [c for c in np.unique(conditions_arr) if c != control_label]
    n_total = len(unique_conds)
    print(f"  {n_total:,} perturbation groups detected.")

    X_csr = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
    ctrl_mean = np.asarray(X_csr[ctrl_mask].mean(axis=0)).ravel()
    log_ctrl = np.log1p(np.maximum(ctrl_mean, 0))

    proxy_scores = {}
    for cond in tqdm(unique_conds, desc="  LFC proxy", unit="pert", ncols=80):
        mask = conditions_arr == cond
        if mask.sum() < 2:
            proxy_scores[cond] = 0.0
            continue
        cm = np.asarray(X_csr[mask].mean(axis=0)).ravel()
        proxy_scores[cond] = float(np.mean(np.abs(np.log1p(np.maximum(cm, 0)) - log_ctrl)))

    selected_conds = unique_conds
    sample_weights_map = {c: 1.0 for c in unique_conds}

    if n_total > max_perts_for_de:
        sorted_conds = sorted(unique_conds, key=lambda c: proxy_scores[c], reverse=True)
        n_top = min(100, max_perts_for_de // 5)
        n_random = max_perts_for_de - n_top
        top_conds = sorted_conds[:n_top]
        tail_conds = sorted_conds[n_top:]
        rng = np.random.default_rng(42)
        n_draw = min(n_random, len(tail_conds))
        random_conds = list(rng.choice(tail_conds, size=n_draw, replace=False))
        selected_conds = top_conds + random_conds
        tail_w = float(len(tail_conds)) / max(n_draw, 1)
        for c in top_conds:
            sample_weights_map[c] = 1.0
        for c in random_conds:
            sample_weights_map[c] = tail_w
        print(f"  Selected {len(selected_conds)} perts "
              f"({n_top} top-proxy + {len(random_conds)} random tail).")
    else:
        print(f"  Running Wilcoxon on all {n_total} perturbations.")

    keep_mask = ctrl_mask.copy()
    for cond in selected_conds:
        keep_mask = keep_mask | (conditions_arr == cond)
    adata_sub = adata[keep_mask].copy()
    sc.tl.rank_genes_groups(adata_sub, groupby=perturbation_column,
                            reference=control_label, method=de_method,
                            use_raw=False, n_jobs=n_jobs)

    var_names = list(adata.var_names)
    n_genes = len(var_names)
    name_to_idx = {str(vn): i for i, vn in enumerate(var_names)}
    for cand in ["original_gene_id", "gene_name", "gene_names",
                 "gene_symbols", "gene_symbol", "feature_name",
                 "feature_id", "symbol", "Symbol"]:
        if cand in adata.var.columns:
            for i, sym in enumerate(adata.var[cand].astype(str).values):
                if sym != 'nan':
                    name_to_idx.setdefault(sym, i)

    impact_scores, valid_labels, valid_weights_list = [], [], []
    deg_rows, deg_cols = [], []
    n_tested = 0

    for cond in tqdm(selected_conds, desc="  DEG counts", unit="pert", ncols=80):
        n_tested += 1
        try:
            df = sc.get.rank_genes_groups_df(adata_sub, group=cond)
            sig = df[(df["pvals_adj"] < pval_threshold) &
                     (df["logfoldchanges"].abs() > effective_lfc)]
            impact_scores.append(len(sig))
            valid_labels.append(cond)
            valid_weights_list.append(sample_weights_map.get(cond, 1.0))
            pert_idx = name_to_idx.get(str(cond))
            if pert_idx is not None:
                for dname in sig["names"].values:
                    didx = name_to_idx.get(str(dname))
                    if didx is not None and didx != pert_idx:
                        deg_rows.append(pert_idx)
                        deg_cols.append(didx)
        except Exception:
            continue

    impact_array = np.array(impact_scores, dtype=np.float64)
    perturbation_labels = np.array(valid_labels)
    weights_arr = np.array(valid_weights_list, dtype=np.float64)
    nz = impact_array > 0
    n_active = int(nz.sum())
    impact_array = impact_array[nz]
    perturbation_labels = perturbation_labels[nz]
    weights_arr = weights_arr[nz]

    if len(deg_rows) > 0:
        deg_matrix = sp.coo_matrix(
            (np.ones(len(deg_rows)), (np.array(deg_rows), np.array(deg_cols))),
            shape=(n_genes, n_genes)).tocsr()
        deg_matrix.data = np.ones_like(deg_matrix.data)
    else:
        deg_matrix = sp.csr_matrix((n_genes, n_genes))

    X_full = (adata.X.toarray() if hasattr(adata.X, 'toarray')
              else np.array(adata.X)).astype(np.float32)
    ctrl_mean_full = X_full[ctrl_mask].mean(axis=0)
    log_ctrl_full = np.log1p(np.maximum(ctrl_mean_full, 0))

    lfc_vectors, valid_cq = [], []
    grp_vals = adata.obs[perturbation_column].values
    for cond in selected_conds:
        m = grp_vals == cond
        if m.sum() < 2:
            continue
        lfc = np.log1p(np.maximum(X_full[m].mean(axis=0), 0)) - log_ctrl_full
        if np.all(np.isfinite(lfc)) and np.any(lfc != 0):
            lfc_vectors.append(lfc)
            valid_cq.append(cond)
    del X_full
    gc.collect()

    lfc_matrix = (np.array(lfc_vectors, dtype=np.float32)
                  if lfc_vectors else np.zeros((0, n_genes), dtype=np.float32))

    print(f"  Active perts    : {n_active} / {n_tested} tested")
    print(f"  DEG matrix      : {deg_matrix.nnz:,} causal edges")
    print(f"  LFC matrix      : {lfc_matrix.shape[0]} perturbations × {n_genes} genes")

    return (impact_array, perturbation_labels, weights_arr,
            deg_matrix, lfc_matrix, valid_cq, name_to_idx, n_tested)


# ---------------------------------------------------------------------------
# Probe 1 — Alpha: topweight_powerlaw  [FIXES: confidence, xmin, LR gate]
# ---------------------------------------------------------------------------

def _probe_alpha(impact_array, lam_eff, raw_sparse_mat=None, n_genes=None):
    """
    Power-law MLE on the out-degree distribution of the topweight parent graph.

    R6 winner method retained. Three fixes applied in v11.0:
      1. Confidence inversion corrected: good fit (low D) → high conf.
      2. xmin: KS-minimising auto-selection; 50-node tail guard; fallback xmin=6.
      3. Lognormal LR gate: if lognormal fits significantly better, conf × 0.5.
    """
    if raw_sparse_mat is not None and n_genes is not None and raw_sparse_mat.nnz > 0:
        try:
            import powerlaw
            n_keep = max(int(lam_eff * n_genes), 1000)
            coo = raw_sparse_mat.tocoo()
            if coo.nnz > n_keep:
                top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
                src_top = coo.row[top_idx]
            else:
                src_top = coo.row

            od = np.bincount(src_top.astype(int), minlength=n_genes).astype(np.float64)
            # Use all nonzero degrees; let powerlaw choose xmin
            od_nz = od[od >= 1]

            if len(od_nz) >= 20:
                # Auto xmin selection (KS-minimising, Clauset et al. 2009)
                fit = powerlaw.Fit(od_nz, discrete=True, verbose=False)
                # Tail-size guard: if fewer than 50 nodes above xmin, refit at xmin=6
                xmin_auto = fit.power_law.xmin
                tail_size = int(np.sum(od_nz >= xmin_auto))
                if tail_size < 50:
                    fit = powerlaw.Fit(od_nz, xmin=6, discrete=True, verbose=False)

                a_raw = _safe_float(fit.power_law.alpha, 2.0)
                sigma = max(_safe_float(fit.power_law.sigma, 0.20), 0.10)
                dash_correction = 0.15
                center = float(np.clip(a_raw - dash_correction, 1.2, 3.5))
                hw = max(1.96 * sigma, 0.15)
                lo, hi = _clip_bound(center - hw, center + hw, "alpha")
                # v15.5: removed hi = PHYSICAL["alpha"][1] hardcode. The old
                # code forced the upper bound to 3.5 regardless of data, making
                # the alpha band unreasonably wide (documented in 06JUNE handoff).
                # _clip_bound already enforces physical limits [1.2, 3.5].

                # Corrected confidence: low KS distance D → high confidence (FIX A1)
                ks = _safe_float(fit.power_law.D, 0.5)
                conf = float(np.clip(-np.log10(max(ks, 1e-6)) / 3.0, 0.1, 1.0))

                # Lognormal LR gate (FIX A3)
                try:
                    R, p_lr = fit.distribution_compare(
                        'power_law', 'lognormal', normalized_ratio=True)
                    if R < 0 and p_lr < 0.10:
                        conf *= 0.5
                        conf = max(conf, 0.10)
                except Exception:
                    pass

                return lo, hi, conf
        except Exception:
            pass

    if len(impact_array) >= 10:
        try:
            import powerlaw
            fit = powerlaw.Fit(impact_array, discrete=True, verbose=False)
            xmin_auto = fit.power_law.xmin
            tail_size = int(np.sum(impact_array >= xmin_auto))
            if tail_size < 50:
                fit = powerlaw.Fit(impact_array, xmin=6, discrete=True, verbose=False)

            a_raw = _safe_float(fit.power_law.alpha, 2.3)
            sigma = max(_safe_float(fit.power_law.sigma, 0.20), 0.10)
            center = float(np.clip(a_raw, 1.2, 3.5))
            hw = max(1.96 * sigma, 0.15)
            lo, hi = _clip_bound(center - hw, center + hw, "alpha")

            # Corrected confidence (FIX A1)
            ks = _safe_float(fit.power_law.D, 0.5)
            conf = float(np.clip(-np.log10(max(ks, 1e-6)) / 3.0, 0.1, 1.0))

            # Lognormal LR gate (FIX A3)
            try:
                R, p_lr = fit.distribution_compare(
                    'power_law', 'lognormal', normalized_ratio=True)
                if R < 0 and p_lr < 0.10:
                    conf *= 0.5
                    conf = max(conf, 0.10)
            except Exception:
                pass

            return lo, hi, conf
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["alpha"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 2 — Gini: raw_deg_outdegree  [UNCHANGED — R6 winner, clean]
# ---------------------------------------------------------------------------

def _probe_gini(deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff,
                raw_sparse_mat=None, n_genes=None):
    """
    Out-degree Gini on the parent graph top-lam_eff slice.

    v15.5 substrate fix: the loss function measures Gini on the DASH output
    graph (all ~5000 genes). The old probe measured Gini on the DEG matrix
    (~108 nonzero rows = perturbation genes only). Measuring inequality
    across 108 hub TFs inflates Gini relative to the full 5000-gene
    distribution, creating a floor the optimizer cannot reach.

    Fix: use the same substrate as _probe_gini_in — the parent graph top
    (lam_eff × n_genes) edges by weight, computing out-degree (row sums)
    instead of in-degree (column sums). This is the closest available proxy
    for the DASH output degree distribution.

    Falls back to the DEG matrix measurement if the parent graph is missing.
    """
    # --- Primary: parent graph top-lam_eff slice (same substrate as loss) ---
    if (raw_sparse_mat is not None and raw_sparse_mat.nnz > 0
            and n_genes is not None):
        try:
            n_keep = max(int(lam_eff * n_genes), 1000)
            coo = raw_sparse_mat.tocoo()
            if coo.nnz > n_keep:
                top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
                src_top = coo.row[top_idx]
            else:
                src_top = coo.row

            od = np.bincount(src_top.astype(int), minlength=n_genes).astype(np.float64)
            od_nz = od[od > 0]

            if len(od_nz) >= 20:
                # exp_004d promotion (go_isodensity_global): out-degree Gini at
                # output density, ceiling clipped to the Pareto-consistent value
                # 1/(2*alpha-3) at ALPHA_ANCHOR. NO 0.30 floor (removed: it
                # laundered the shuffle/degenerate case and broke the
                # gini-derived alpha control -- exp_004c). NO n/(n-1) adj.
                center = float(_gini(od_nz))
                pl_ceiling = 1.0 / (2.0 * ALPHA_ANCHOR - 3.0)
                hi = min(center + 0.15, pl_ceiling)
                lo = center - 0.15
                lo, hi = _clip_bound(min(lo, hi), max(lo, hi), "gini")
                return lo, hi, 0.70
        except Exception:
            pass

    # --- Fallback: DEG matrix (legacy, 108-gene substrate) ----------------
    if deg_matrix is None or deg_matrix.nnz == 0:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        od = np.asarray(deg_matrix.sum(axis=1)).ravel().astype(np.float64)
        od_nz = od[od > 0]

        if len(od_nz) < 5:
            fb = FALLBACK_BOUNDS["gini"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        n_obs = len(od_nz)
        gini_raw = _gini(od_nz)
        gini_adj = gini_raw * n_obs / max(n_obs - 1, 1)
        center = float(np.clip(gini_adj, 0.30, 0.95))

        lo, hi = _clip_bound(center - 0.15, center + 0.15, "gini")
        conf = 0.55  # lower confidence for the 108-gene fallback
        return lo, hi, conf

    except Exception:
        pass

    fb = FALLBACK_BOUNDS["gini"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 2b — gini_in: in-degree Gini (REPLACES Q in the organic cohort)
# ---------------------------------------------------------------------------

def _probe_gini_in(raw_sparse_mat, n_genes, lam_eff, substrate_lam=None):
    """
    In-degree Gini on a top-lam parent-graph slice — regulatory-input
    concentration. How unevenly do regulatory edges converge onto target genes?

    Replaces the modularity (Q) probe in organic FUNGI. Two reasons it is a
    better-behaved target than Q:
      1. Different structural axis from out-degree Gini / S_max, so it adds
         genuinely new information (the entire in-degree side was unmeasured).
      2. No DASH term actively fights it. Q failed because SCBER boosts the
         inter-module bridges that Q penalises — the target was unreachable by
         construction. Nothing in the kernel suppresses in-degree concentration.

    Substrate matches what the loss measures (structural in-degree of surviving
    targets, np.bincount of surv_t), so there is no Q-style substrate mismatch:
    we take the top (lam_eff x n_genes) edges of the parent graph and Gini their
    column-sum (in-degree) distribution.

    Also catches technical artefacts: a single gene absorbing a large fraction
    of all regulatory edges (e.g. CD79B in-degree ~1% of all edges on VCC_4c)
    inflates in-degree Gini sharply.
    """
    if substrate_lam is None:
        substrate_lam = lam_eff
    if raw_sparse_mat is None or raw_sparse_mat.nnz == 0 or n_genes is None:
        fb = FALLBACK_BOUNDS["gini_in"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE
    try:
        # exp_004d promotion (gi_isodensity_shrunk): in-degree Gini at OUTPUT
        # density (lam_eff, not substrate_lam=lam_q25), window shrunk toward the
        # exponential anchor 0.5 (asymmetric: tighter on the high side).
        n_keep = max(int(lam_eff * n_genes), 100)
        coo = raw_sparse_mat.tocoo()
        if coo.nnz > n_keep:
            top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
            tgt_top = coo.col[top_idx]
        else:
            tgt_top = coo.col
        idd = np.bincount(tgt_top.astype(int), minlength=n_genes).astype(np.float64)
        idd_nz = idd[idd > 0]
        if len(idd_nz) < 5:
            fb = FALLBACK_BOUNDS["gini_in"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE
        center = float(_gini(idd_nz))
        lo, hi = _clip_bound(center - 0.10, center + 0.04, "gini_in")
        return lo, hi, 0.68
    except Exception:
        pass
    fb = FALLBACK_BOUNDS["gini_in"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE

def _probe_smax(raw_sparse_mat, deg_matrix, lfc_matrix, valid_cq,
                name_to_idx, n_genes, lam_eff, impact_array,
                substrate_lam=None):
    """
    Hub fraction on a topweight substrate anchored at the 25th percentile
    of the specificity-weighted DEG distribution (substrate_lam = λ_q25).

    v11.0 change: substrate is now substrate_lam (passed in from run_diagnostics),
    decoupled from λ_center. This eliminates the cascade:
      λ_eff cap → inflated substrate → inflated S_max target → optimizer failure.

    The bound measurement is still fully data-derived; only the substrate density
    has changed (from λ_eff−3 to λ_q25, which is more conservative and principled).

    Falls back to the impact-array cascade estimator if the parent graph is
    unavailable (unchanged from v10.0).
    """
    # Use substrate_lam if provided; fall back to a conservative default
    if substrate_lam is None:
        substrate_lam = max(lam_eff - 3.0, 1.0)  # safe fallback for legacy calls

    # exp_004d promotion (smax_rpe1_prior): S_max is a PRIOR-type target -- no
    # parent measurement reaches the literature band (the parent is extreme
    # hub-and-spoke; exp_004b/c/006). Theory bracket [structural cutoff
    # sqrt(lam_eff*N)/N, RPE1 master-regulator ceiling 0.18]; data-independent.
    # The 0.59 fudge factor is ELIMINATED.
    if n_genes is not None and n_genes > 0:
        try:
            struct = float(np.sqrt(lam_eff * n_genes) / n_genes)
            lo, hi = _clip_bound(struct, 0.18, "S_max")
            return lo, hi, 0.65
        except Exception:
            pass

    if impact_array is not None and len(impact_array) >= 5:
        try:
            nz = impact_array[impact_array > 0]
            max_impact = float(nz.max())
            mean_impact = float(nz.mean())
            cascade_factor = float(np.clip(max_impact / max(mean_impact, 1.0), 1.5, 8.0))
            direct_count = max_impact / cascade_factor
            center = float(np.clip(direct_count / n_genes, 0.01, 0.20))
            hw = max(center * 0.40, 0.015)
            lo, hi = _clip_bound(center - hw, center + hw, "S_max")
            return lo, hi, 0.35
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["S_max"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 4 — Q: leiden_modularity_fixed_resolution_1p3  [RESTORED session 5]
#
# Q was removed entirely in session 4 (was PARKED at [0.10, 0.25] conf=0.01
# since v15.4): SCBER boosts inter-community bridge edges, which structurally
# fights modularity, and no probe substrate could close that gap because the
# cause was the DASH/SCBER design itself having no counterbalancing force.
# Session 5 adds m_intra (engine.py, the 8th searched hyperparameter) as
# exactly that counterbalancing force -- a positive pull toward intra-
# community edges -- which makes Q achievable again, so it is restored here
# as a genuine target (not a low-confidence ghost).
#
# Redesigned, not revived as-is: the historical target windows ([0.10, 0.25]
# parked-era; [0.30, 0.70] pre-session-4, when SCBER's resolution selection
# was itself coupled to a target_q=0.5) were calibrated for sparser graphs or
# a broken community detector (session 4 found the old selection rule was
# mathematically guaranteed to pick the trivial single-community partition).
# Session 4's corrected, widened resolution sweep on the real SHROOM_2
# G_work substrate found genuine, non-trivial community structure first
# emerges at resolution=1.3 (Q_igraph ~= 0.026, ~305 communities, largest
# ~17% of nodes) -- this probe is calibrated around that finding, not the
# old constants.
#
# Substrate note: this probe runs at Phase 0, before Phase 2 builds G_work
# and SCBER's actual partition -- it cannot literally reuse SCBER's graph
# (chronological ordering), so like _probe_C/_probe_gini_in it approximates
# the same substrate using raw_sparse_mat's top-(substrate_lam x n_genes)
# slice, symmetrized and weighted. This is the same approximation every
# other density-dependent probe in this file already makes.
# ---------------------------------------------------------------------------

def _probe_Q(raw_sparse_mat, n_genes, lam_eff, substrate_lam=None,
            resolution=1.3, min_communities=10, max_dominant_fraction=0.5,
            n_seeds=3):
    """
    Modularity (Q) via igraph Leiden at a FIXED resolution=1.3 -- the
    resolution session 4 found to be the coarsest one producing genuine,
    non-trivial community structure on this substrate (see module-level
    comment above). Not swept: a fixed resolution makes this a genuine
    measurement of "how much real community structure exists here", not a
    search for whichever resolution looks best.

    Substrate: same top-(substrate_lam x n_genes) parent-graph slice
    _probe_C/_probe_gini_in use, symmetrized and edge-weighted (mirrors
    effective_resistance._detect_communities()'s own symmetrization).

    Multiple Leiden runs (n_seeds, default 3) at the same resolution give a
    genuine IQR for confidence -- igraph's community_leiden() has no exposed
    seed= argument, so consecutive calls draw from its own advancing global
    RNG (session 4 observed 304 vs 307 communities across two such runs --
    acceptable variance for this purpose).

    Falls back to a moderate-confidence biological prior (not the old
    parked conf=0.01) if the partition found is degenerate (too few
    communities, or one community dominates) -- this can legitimately
    happen on a different/smaller dataset than the one this was calibrated
    against.
    """
    if substrate_lam is None:
        substrate_lam = lam_eff
    if raw_sparse_mat is None or raw_sparse_mat.nnz == 0 or n_genes is None:
        fb = FALLBACK_BOUNDS["Q"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE_Q
    try:
        import igraph as ig
        n_keep = max(int(substrate_lam * n_genes), 100)
        coo = raw_sparse_mat.tocoo()
        if coo.nnz > n_keep:
            top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
            pruned = sp.coo_matrix(
                (coo.data[top_idx], (coo.row[top_idx], coo.col[top_idx])),
                shape=raw_sparse_mat.shape).tocsr()
        else:
            pruned = raw_sparse_mat.tocsr()

        A_sym = (pruned + pruned.T) / 2.0
        A_sym.data = np.abs(A_sym.data)
        A_sym.eliminate_zeros()
        coo_s = A_sym.tocoo()
        G = ig.Graph(n=n_genes,
                     edges=list(zip(coo_s.row.tolist(), coo_s.col.tolist())),
                     directed=False, edge_attrs={'weight': coo_s.data.tolist()})
        G.simplify(combine_edges='sum')

        qs, n_comms, largest_fracs = [], [], []
        for _ in range(max(int(n_seeds), 1)):
            part = G.community_leiden(weights='weight',
                                      objective_function='modularity',
                                      n_iterations=5,
                                      resolution=float(resolution))
            membership = np.array(part.membership)
            n_comm = len(set(membership.tolist()))
            largest_frac = float(np.bincount(membership).max()) / n_genes
            qs.append(float(part.modularity))
            n_comms.append(n_comm)
            largest_fracs.append(largest_frac)

        med_n_comm = float(np.median(n_comms))
        med_largest_frac = float(np.median(largest_fracs))

        if med_n_comm < min_communities or med_largest_frac > max_dominant_fraction:
            fb = FALLBACK_BOUNDS["Q"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE_Q

        q_arr = np.array(qs, dtype=np.float64)
        center = float(np.median(q_arr))
        iqr = float(np.percentile(q_arr, 75) - np.percentile(q_arr, 25))
        # Half-width floor: IQR across 3 seeds at a fixed resolution tends to
        # be near-zero (this is genuine stochastic noise in Leiden's
        # refinement phase, not measurement uncertainty about the target
        # itself) -- a relative floor keeps the window from collapsing to an
        # unreachable point target.
        hw = max(iqr * 0.75, abs(center) * 0.4, 0.008)
        lo, hi = _clip_bound(center - hw, center + hw, "Q")

        rel_iqr = iqr / max(abs(center), 1e-6)
        conf = float(np.clip(0.75 - rel_iqr, 0.35, 0.70))
        return lo, hi, conf
    except Exception:
        pass
    fb = FALLBACK_BOUNDS["Q"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE_Q

# ---------------------------------------------------------------------------
# Probe 5 — C: topweight_transitivity_decoupled_q25  [SUBSTRATE + DEDUP FIX]
# ---------------------------------------------------------------------------

def _probe_C(raw_sparse_mat, n_genes, lam_eff, deg_matrix=None,
             lfc_matrix=None, valid_cq=None, name_to_idx=None,
             substrate_lam=None):
    """
    Graph transitivity (clustering coefficient) on topweight parent.

    v11.0 changes:
      FIX A4: G.simplify(multiple=True, loops=True) added before
              transitivity_undirected() to prevent multigraph inflation from
              reciprocal directed edges. Correctness fix; modest but real bias.
      FIX C1: Substrate anchored at substrate_lam (= λ_q25 from run_diagnostics)
              instead of λ_eff × n_genes. Decoupled from λ_center errors.
    """
    # Use substrate_lam if provided; fall back to lam_eff for legacy calls
    if substrate_lam is None:
        substrate_lam = lam_eff

    try:
        if raw_sparse_mat is not None and raw_sparse_mat.nnz > 0:
            import igraph as ig
            # exp_004d promotion (c_isodensity_global): OUTPUT density (lam_eff,
            # not substrate_lam=lam_q25).
            n_keep = int(lam_eff * n_genes)
            n_keep = max(n_keep, 100)
            parent = raw_sparse_mat.tocsr()
            coo = parent.tocoo()
            if coo.nnz > n_keep:
                top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
                pruned = sp.coo_matrix(
                    (coo.data[top_idx], (coo.row[top_idx], coo.col[top_idx])),
                    shape=parent.shape).tocsr()
            else:
                pruned = parent
            coo_p = pruned.tocoo()
            G = ig.Graph(n=n_genes,
                         edges=list(zip(coo_p.row.tolist(), coo_p.col.tolist())),
                         directed=False)
            # FIX A4: remove multi-edges and self-loops from directed→undirected conversion
            G.simplify(multiple=True, loops=True)
            c_val = float(G.transitivity_undirected())
            if not np.isfinite(c_val):
                c_val = 0.06
            # exp_004d promotion: 1.25 scalar ELIMINATED -- raw undirected
            # transitivity is already in the literature band [0.08,0.20].
            center = float(np.clip(c_val, 0.001, 0.30))
            lo, hi = _clip_bound(center - 0.03, center + 0.03, "C")
            return lo, hi, 0.70
    except Exception:
        pass

    if (deg_matrix is not None and deg_matrix.nnz > MIN_SKELETON_EDGES
            and valid_cq is not None and name_to_idx is not None):
        try:
            import igraph as ig
            src_indices = []
            for gene_name in valid_cq:
                idx = name_to_idx.get(str(gene_name))
                if idx is not None:
                    src_indices.append(idx)

            if len(src_indices) >= 10:
                src_arr = np.array(src_indices, dtype=np.int64)
                deg_sub = deg_matrix[src_arr, :].toarray().astype(np.float64)
                n_perts = deg_sub.shape[0]
                edges = []
                for i in range(n_perts):
                    for j in range(i + 1, n_perts):
                        inter = np.sum((deg_sub[i] > 0) & (deg_sub[j] > 0))
                        union = np.sum((deg_sub[i] > 0) | (deg_sub[j] > 0))
                        if union > 0:
                            jac = inter / union
                            if jac > 0:
                                edges.append((i, j, jac))

                if len(edges) >= 10:
                    jac_scores = np.array([e[2] for e in edges])
                    thresh = np.percentile(jac_scores, 90)
                    filtered = [(i, j) for i, j, s in edges if s >= thresh]
                    if len(filtered) >= 5:
                        G_jac = ig.Graph(n=n_perts, edges=filtered,
                                         directed=False).simplify()
                        c_jac = float(G_jac.transitivity_undirected())
                        if np.isfinite(c_jac):
                            k_cal = 1.0 / (5.0 + 3.0 * (lam_eff / 15.0))
                            c_target = float(np.clip(c_jac * k_cal, 0.001, 0.15))
                            lo, hi = _clip_bound(c_target - 0.025, c_target + 0.025, "C")
                            return lo, hi, 0.35
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["C"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 6 — Rho: lfc_l2_bipartite_assortativity  [FIX: adaptive half-width]
# ---------------------------------------------------------------------------

def _probe_rho(lfc_matrix, valid_cq, name_to_idx):
    """
    Spearman of per-perturbation LFC L2 norms vs per-gene column L2 norms.
    R5/R6 winner. Center formula unchanged.

    v11.0 change (FIX D1): adaptive half-width based on Spearman p-value.
        hw = clip(0.08 + 0.12 × min(1, pval/0.05), 0.08, 0.20)
    A significant correlation (pval≤0.001) stays tight (~0.08).
    A non-significant one (pval≥0.05) widens to the 0.20 cap, acting as a
    soft sign prior rather than a sharply informative numerical target.
    This resolves the VCC near-miss (champion rho=−0.058 vs old bound −0.056).
    """
    if lfc_matrix is None or len(lfc_matrix) < 10 or not valid_cq:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        k_out_all = np.linalg.norm(lfc_matrix, axis=1)
        k_in_all = np.linalg.norm(lfc_matrix, axis=0)

        k_out_matched, k_in_matched = [], []
        for k, gene_name in enumerate(valid_cq):
            gene_idx = name_to_idx.get(str(gene_name))
            if gene_idx is not None and gene_idx < len(k_in_all):
                k_out_matched.append(k_out_all[k])
                k_in_matched.append(k_in_all[gene_idx])

        if len(k_out_matched) < 10:
            fb = FALLBACK_BOUNDS["rho"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        corr, pval = spearmanr(np.array(k_out_matched), np.array(k_in_matched))
        if not np.isfinite(corr):
            corr = 0.0

        center = float(np.clip(-0.15 * corr - 0.10, -0.45, 0.10))

        # FIX D1: adaptive half-width — widen when correlation is statistically weak
        pval_safe = float(pval) if np.isfinite(pval) else 1.0
        hw = float(np.clip(0.08 + 0.12 * min(1.0, pval_safe / 0.05), 0.08, 0.20))

        lo, hi = _clip_bound(center - hw, center + hw, "rho")
        hi = min(hi, 0.0)   # GRNs are disassortative or neutral; never assortative
        conf = float(np.clip(1.0 - pval_safe, 0.15, 0.90))
        return lo, hi, conf

    except Exception:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Bound enforcement  [UNCHANGED]
# ---------------------------------------------------------------------------

def _enforce_bound_constraints(bound_min, bound_max, center, constraints):
    delta_min = constraints["delta_min"]
    delta_max = constraints["delta_max"]
    hard_floor = constraints["hard_floor"]
    hard_ceiling = constraints["hard_ceiling"]

    bound_min = _safe_float(bound_min, hard_floor)
    bound_max = _safe_float(bound_max, hard_ceiling)
    center = _safe_float(center, (hard_floor + hard_ceiling) / 2)
    if bound_min > bound_max:
        bound_min, bound_max = bound_max, bound_min
    width = bound_max - bound_min
    if width < delta_min:
        exp = (delta_min - width) / 2.0
        bound_min -= exp
        bound_max += exp
    if (bound_max - bound_min) > delta_max:
        half = delta_max / 2.0
        bound_min = center - half
        bound_max = center + half
    if bound_min < hard_floor:
        deficit = hard_floor - bound_min
        bound_min = hard_floor
        bound_max = min(bound_max + deficit, hard_ceiling)
    if bound_max > hard_ceiling:
        surplus = bound_max - hard_ceiling
        bound_max = hard_ceiling
        bound_min = max(bound_min - surplus, hard_floor)
    return float(bound_min), float(bound_max)


# ---------------------------------------------------------------------------
# Weight normalisation  [FIX A6: iterative renorm after clip]
# ---------------------------------------------------------------------------

def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
    """
    Normalise probe confidences to weights summing to target_sum.

    v11.0 fix (A6): iterative clip-then-renormalise (≤5 passes, tol 1e-6)
    ensures weights sum to target_sum after clipping. Without this, the
    post-clip sum drifts when floor or ceiling binds, making loss values
    incomparable across runs with different probe confidence profiles.
    """
    names = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = FALLBACK_CONFIDENCE
    values = np.clip(values, 0.0, 1.0)
    total = values.sum()
    if total < 1e-10:
        values = np.ones(len(values)) / len(values) * target_sum
    else:
        values = values / total * target_sum

    # Iterative clip-then-renormalise
    for _ in range(5):
        values = np.clip(values, floor, ceiling)
        post_sum = values.sum()
        if abs(post_sum - target_sum) < 1e-6:
            break
        if post_sum < 1e-10:
            break
        values = values / post_sum * target_sum

    return {n: float(v) for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# Summary printer  [UPDATED: version, lambda method, new diagnostics]
# ---------------------------------------------------------------------------

_PARAM_DESCRIPTIONS = {
    "alpha":   "Scale-free degree exponent",
    "gini":    "Out-degree inequality (source hub dominance)",
    "gini_in": "In-degree inequality (regulatory-input concentration)",
    "S_max":   "Largest hub's target fraction",
    "C":       "Feed-forward loop density",
    "rho":     "Hub-to-effector disassortativity",
    "Q":       "Community modularity (Leiden, resolution=1.3 fixed)",
    "reciprocity": "Directed link reciprocity (low for causal GRNs; prior)",
}


def _print_summary(utopian_bounds, loss_weights, raw_confidences, lam_eff,
                   lam_q25, lam_q75, erank_diag, probes_used,
                   n_active, n_tested, n_genes, substrate_lam,
                   overridden_params=None, disabled_params=None):
    overridden_params = overridden_params or set()
    disabled_params = disabled_params or set()
    print(f"\n{'=' * 70}")
    print("FUNGI v16.0 — Phase 0 Diagnostic Summary")
    print(f"{'=' * 70}")
    print(f"  Dataset: {n_genes:,} HVGs | {n_active}/{n_tested} active perturbations")
    print(f"  λ_target = {lam_eff:.2f} edges/gene  [user-controlled]")
    print(f"  λ_search = [{lam_q25:.2f}, {lam_q75:.2f}]  "
          f"(probe substrate: {substrate_lam:.2f} edges/gene)")
    if erank_diag is not None:
        print(f"  λ_erank  = {erank_diag:.2f}  [diagnostic only: Gavish-Donoho effective rank]")
    print()

    conf_emoji = {True: "✓", False: "~"}
    for param in ["alpha", "gini", "gini_in", "Q", "S_max", "C", "rho", "reciprocity"]:
        lo, hi = utopian_bounds[param]
        cf = raw_confidences[param]
        w = loss_weights[param]
        desc = _PARAM_DESCRIPTIONS[param]
        if param in overridden_params:
            print(f"  ⊕ {param:>5s} [{lo:.4f}, {hi:.4f}]  "
                  f"conf={cf:.2f}  wt={w:.1f}  — {desc} [OVERRIDE]")
            continue
        if param in disabled_params:
            print(f"  ⊘ {param:>5s} [{lo:.4f}, {hi:.4f}]  "
                  f"conf={cf:.2f}  wt={w:.1f}  — {desc} "
                  f"[DISABLED -- probe data shown, excluded from loss]")
            continue
        high_conf = cf >= 0.5
        print(f"  {conf_emoji[high_conf]} {param:>5s} [{lo:.4f}, {hi:.4f}]  "
              f"conf={cf:.2f}  wt={w:.1f}  — {desc}")

    warnings_raised = []
    if utopian_bounds["S_max"][1] > 0.20:
        warnings_raised.append("S_max upper bound >0.20: unusually large hub expected")
    if utopian_bounds["C"][0] < 0.005:
        warnings_raised.append("C lower bound <0.005: very low clustering expected")
    if n_active < 30:
        warnings_raised.append(f"Only {n_active} active perturbations: low statistical power")

    low_conf_count = sum(1 for v in raw_confidences.values() if v < 0.35)
    if low_conf_count >= 3:
        warnings_raised.append(f"{low_conf_count} probes have low confidence (<0.35)")

    if lam_eff < 6.0:
        warnings_raised.append(
            f"λ_eff={lam_eff:.1f} is low — consider checking LFC matrix rank "
            f"or lowering de_lfc_threshold if this is a genome-scale screen")

    if (erank_diag is not None
            and abs(erank_diag - lam_eff) / max(lam_eff, 1.0) > 0.50):
        warnings_raised.append(
            f"λ_target ({lam_eff:.1f}) and erank ({erank_diag:.1f}) diverge "
            f">50%: consider adjusting lambda_user.target")

    if warnings_raised:
        print(f"\n  ⚠ Warnings:")
        for w in warnings_raised:
            print(f"    - {w}")

    proceed = n_active >= 20 and low_conf_count < 4
    if proceed:
        print(f"\n  Decision: PROCEED")
    else:
        print(f"\n  Decision: CAUTION — review diagnostics before proceeding")

    print(f"{'=' * 70}\n")
    return proceed


# ---------------------------------------------------------------------------
# Master runner  [UPDATED: passes substrate_lam to S_max and C probes]
# ---------------------------------------------------------------------------

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                    raw_sparse_mat=None, lambda_user_cfg=None,
                    bounds_override_cfg=None, organic_targets_cfg=None):
    """
    Full Phase 0 pipeline (v15.1).

    Changes from v15.0:
      - Q: IQR now computed within winning resolution only (3 seeds),
        not across all 9 runs. See _probe_Q docstring.

    bounds_override_cfg (v16.1): optional dict, e.g. cfg.get("bounds_override")
    from fungi_config.yaml -- {param: [lo, hi]} or {param: [lo, hi, conf]} for
    any subset of alpha/gini/gini_in/S_max/C/rho. Bypasses the probe result
    for listed params only; params not listed stay fully probe-driven. Applied
    after all probe/enforcement logic so the probe functions themselves are
    untouched. Confidence defaults to 1.0 (user-declared, fully trusted) when
    not given, which also feeds _normalize_weights so an override gets full
    weight in the loss, not just a cosmetic bound swap.

    organic_targets_cfg (session 5): optional dict, e.g. cfg.get("organic_
    targets") from fungi_config.yaml -- {param: {"enabled": bool}} for any
    subset of alpha/gini/gini_in/Q/S_max/C/rho. enabled=False forces that
    param's loss_weight to exactly 0.0 AFTER _normalize_weights runs --
    calculate_utopia_loss's existing smooth-penalty term already multiplies
    by this weight, so a 0.0 weight removes the target from the loss sum
    with no separate gating logic needed in engine.py. The probe itself
    still runs unmodified: its bound/confidence/probe_used stay populated
    and are still printed/reported -- disabling a target hides it from the
    OPTIMIZER, not from the diagnostics. Mirrors dash_kernel's existing
    enabled/disabled pattern, but for loss targets instead of kernel factors.

    Parameters and return values are otherwise unchanged.
    """
    pert_col = cfg_input["perturbation_column"]
    ctrl_label = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool = cfg_input.get("metacell_pooling_factor", None)

    bc = cfg_diagnostics["bound_constraints"]
    w_floor = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    n_jobs = cfg_diagnostics.get("n_jobs", 6)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    (impact_array, pert_labels, sample_weights,
     deg_matrix, lfc_matrix, valid_cq, name_to_idx,
     n_tested) = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method=cfg_diagnostics["de_method"],
        pval_threshold=cfg_diagnostics["de_pval_threshold"],
        lfc_threshold=cfg_diagnostics["de_lfc_threshold"],
        n_jobs=n_jobs,
        max_perts_for_de=max_perts,
        is_metacell=is_metacell,
        metacell_pooling_factor=mc_pool,
    )

    n_active = len(impact_array)

    # v16.0: Lambda is user-controlled. The data-driven estimators were
    # unreliable (IDF raced to ceiling, erank measures program count not
    # density). User sets lambda_target + lambda_halfwidth in the config.
    _lam_cfg = lambda_user_cfg or {}
    lam_eff = float(_lam_cfg.get("target", 40.0))
    lam_hw = float(_lam_cfg.get("halfwidth", 10.0))
    lam_q25 = max(lam_eff - lam_hw, 4.0)
    lam_q75 = lam_eff + lam_hw

    # Erank as diagnostic only
    erank_diag = _compute_erank_gd(lfc_matrix) if lfc_matrix is not None else None

    # substrate_lam drives the S_max and C probe substrates
    substrate_lam = float(max(lam_q25, 6.0))
    smax_substrate_lam = float(max(lam_eff - 3.0, 4.0))

    print(f"\n  λ_target = {lam_eff:.2f} edges/gene  [user-controlled]")
    print(f"  λ_search = [{lam_q25:.2f}, {lam_q75:.2f}]  substrate={substrate_lam:.2f}", end="")
    if erank_diag is not None:
        print(f"  |  erank(GD)={erank_diag:.2f} [diagnostic]")
    else:
        print()

    print("\n  Running probes...")
    alpha_lo, alpha_hi, alpha_conf = _probe_alpha(
        impact_array, lam_eff,
        raw_sparse_mat=raw_sparse_mat,
        n_genes=n_genes)

    gini_lo, gini_hi, gini_conf = _probe_gini(
        deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff,
        raw_sparse_mat=raw_sparse_mat, n_genes=n_genes)

    gini_in_lo, gini_in_hi, gini_in_conf = _probe_gini_in(
        raw_sparse_mat, n_genes, lam_eff, substrate_lam=substrate_lam)

    # v16.0: Gini-derived alpha RESTORED. The 250-bootstrap alpha investigation
    # (alpha_investigation.py) tested 7 independent probes at 9 substrate densities.
    # P5_Gini (α = 1/(2G)+1.5) was the ONLY probe that:
    #   - stayed within [2.0, 3.0] at all densities (2.22-2.29)
    #   - had the tightest bootstrap CI (std 0.003-0.004)
    #   - showed no density drift
    # All MLE-based probes drifted above 2.5 at higher densities.
    # Half-width 0.15 (wider than v15.4's 0.10) to account for the systematic
    # parent-graph → DASH-output substrate offset (~0.14 in Gini).
    try:
        if 0.01 < gini_lo < gini_hi < 0.99:
            _a_lo = 1.0 / (2.0 * gini_hi) + 1.5
            _a_hi = 1.0 / (2.0 * gini_lo) + 1.5
            _a_lo, _a_hi = _clip_bound(_a_lo, _a_hi, "alpha")
            alpha_lo, alpha_hi, alpha_conf = _a_lo, _a_hi, gini_conf * 0.90
    except Exception:
        pass

    smax_lo, smax_hi, smax_conf = _probe_smax(
        raw_sparse_mat, deg_matrix, lfc_matrix, valid_cq,
        name_to_idx, n_genes, lam_eff, impact_array,
        substrate_lam=smax_substrate_lam)

    C_lo, C_hi, C_conf = _probe_C(
        raw_sparse_mat, n_genes, lam_eff,
        deg_matrix, lfc_matrix, valid_cq, name_to_idx,
        substrate_lam=substrate_lam)

    rho_lo, rho_hi, rho_conf = _probe_rho(lfc_matrix, valid_cq, name_to_idx)

    Q_lo, Q_hi, Q_conf = _probe_Q(
        raw_sparse_mat, n_genes, lam_eff, substrate_lam=substrate_lam)

    alpha_lo, alpha_hi = _enforce_bound_constraints(
        alpha_lo, alpha_hi, (alpha_lo + alpha_hi) / 2, bc["alpha"])
    gini_lo, gini_hi = _enforce_bound_constraints(
        gini_lo, gini_hi, (gini_lo + gini_hi) / 2, bc["gini"])
    gini_in_lo, gini_in_hi = _enforce_bound_constraints(
        gini_in_lo, gini_in_hi, (gini_in_lo + gini_in_hi) / 2,
        bc.get("gini_in", bc["gini"]))
    smax_lo, smax_hi = _enforce_bound_constraints(
        smax_lo, smax_hi, (smax_lo + smax_hi) / 2, bc["S_max"])
    C_lo, C_hi = _enforce_bound_constraints(
        C_lo, C_hi, (C_lo + C_hi) / 2, bc["C"])
    rho_lo, rho_hi = _enforce_bound_constraints(
        rho_lo, rho_hi, (rho_lo + rho_hi) / 2, bc["rho"])
    Q_lo, Q_hi = _enforce_bound_constraints(
        Q_lo, Q_hi, (Q_lo + Q_hi) / 2, bc["Q"])

    utopian_bounds = {
        "alpha":   [alpha_lo, alpha_hi],
        "gini":    [gini_lo, gini_hi],
        "gini_in": [gini_in_lo, gini_in_hi],
        "S_max":   [smax_lo, smax_hi],
        "C":       [C_lo, C_hi],
        "rho":     [rho_lo, rho_hi],
        "Q":       [Q_lo, Q_hi],
        # reciprocity: literature PRIOR (no probe). Default = FALLBACK band;
        # bounds_override.reciprocity supersedes (same value in the exp_010 lock).
        "reciprocity": list(FALLBACK_BOUNDS["reciprocity"]),
    }

    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]

    raw_confidences = {
        "alpha": alpha_conf, "gini": gini_conf, "gini_in": gini_in_conf,
        "S_max": smax_conf, "C": C_conf, "rho": rho_conf, "Q": Q_conf,
        # reciprocity prior: FALLBACK_CONFIDENCE unless bounds_override gives a 3rd
        # element; gives a real loss weight when enabled (disabled -> forced 0 below).
        "reciprocity": FALLBACK_CONFIDENCE,
    }

    # Bounds override (v16.1): for any param listed in bounds_override_cfg,
    # replace the probe's bound/confidence outright. Purely additive
    # post-processing -- the probes above already ran unmodified.
    overridden_params = set()
    for param_key, ov in (bounds_override_cfg or {}).items():
        if param_key not in utopian_bounds or ov is None:
            continue
        utopian_bounds[param_key] = [float(ov[0]), float(ov[1])]
        raw_confidences[param_key] = float(ov[2]) if len(ov) > 2 else 1.0
        overridden_params.add(param_key)

    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # Organic target enable/disable (session 5): applied AFTER normalisation
    # so the live targets' relative weighting is unaffected, then a disabled
    # target's weight is force-set to exactly 0.0 -- doing this before
    # _normalize_weights would not work, since that function's weight_floor
    # clip would lift a 0 raw confidence straight back up to w_floor (1.0).
    # bound/confidence/probes_used are NOT touched -- the probe's own output
    # stays fully populated for display, only its loss contribution is cut.
    disabled_params = set()
    for param_key, tcfg in (organic_targets_cfg or {}).items():
        if param_key not in loss_weights:
            continue
        if not (tcfg or {}).get("enabled", True):
            loss_weights[param_key] = 0.0
            disabled_params.add(param_key)

    probes_used = {
        # exp_004d promotion (session 12): labels reflect the promoted gate-winner
        # probes (alpha gini-derived from the FLOORLESS gini; no 0.59/1.25 scalars).
        "alpha":   "gini_derived_lorenz_v16_floorless_v004d",
        "gini":    "go_isodensity_global_v004d",
        "gini_in": "gi_isodensity_shrunk_v004d",
        "S_max":   "smax_rpe1_prior_v004d",
        "C":       "c_isodensity_global_v004d",
        "rho":     "lfc_l2_bipartite_assortativity",
        "Q":       "leiden_modularity_fixed_resolution_1p3",
        "reciprocity": "literature_prior_garlaschelli2004",
    }
    for param_key in overridden_params:
        probes_used[param_key] = "user_override"

    proceed = _print_summary(
        utopian_bounds, loss_weights, raw_confidences, lam_eff,
        lam_q25, lam_q75, erank_diag, probes_used,
        n_active, n_tested, n_genes, substrate_lam,
        overridden_params=overridden_params, disabled_params=disabled_params)

    diagnostic_report = {
        "version": "16.0",
        "lam_eff": lam_eff,
        "lam_q25": lam_q25,
        "lam_q75": lam_q75,
        "lam_eff_erank_diagnostic": erank_diag,
        "substrate_lam": substrate_lam,
        "lam_method": "user_controlled",
        "disabled_organic_targets": sorted(disabled_params),
        "n_active": n_active,
        "n_tested": n_tested,
        "impact_range": ([float(impact_array.min()), float(impact_array.max())]
                         if len(impact_array) > 0 else [0, 0]),
        "deg_matrix_nnz": int(deg_matrix.nnz),
        "_deg_col_sums": np.asarray(deg_matrix.sum(axis=0)).ravel().tolist(),
        "_deg_row_sums": np.asarray(deg_matrix.sum(axis=1)).ravel().tolist(),
        "lfc_matrix_shape": list(lfc_matrix.shape),
        "is_metacell": is_metacell,
        "probes_used": probes_used,
        "bounds_override_applied": sorted(overridden_params),
        "raw_confidences": {k: _safe_float(v) for k, v in raw_confidences.items()},
        "utopian_bounds": utopian_bounds,
        "loss_weights": loss_weights,
        "proceed": proceed,
        "_impact_array": impact_array.tolist() if len(impact_array) > 0 else [],
        "_perturbation_labels": pert_labels.tolist() if len(pert_labels) > 0 else [],
        "_name_to_idx": name_to_idx,
    }

    return utopian_bounds, loss_weights, diagnostic_report


# ---------------------------------------------------------------------------
# Shatter config builder  [UNCHANGED]
# ---------------------------------------------------------------------------

def build_shatter_config(cfg_shatter, n_genes, utopian_bounds,
                         lambda_search_bounds, mode="biologic"):
    """
    Build the viability gate used by check_shatter.

    mode='biologic'   : full biological gate, including a clustering floor.
    mode='synthetic' : clustering floor relaxed (FAGCN exploits heterophily;
                       a synthetic-optimal graph can legitimately have near-
                       zero transitivity). Structural gates (orphans, GWCC,
                       edge count, target coverage, hub saturation) are kept
                       because a disconnected or hub-collapsed graph is bad
                       for any message-passing GNN regardless of mode.
    """
    lambda_min = cfg_shatter.get("lambda_min", 2.0)
    lambda_max = cfg_shatter.get("lambda_max", 35.0)
    s_max_mult = cfg_shatter.get("s_max_ceiling_multiplier", 2.0)
    s_max_cap = cfg_shatter.get("s_max_ceiling_hard_cap", 0.30)
    clust_gamma = cfg_shatter.get("min_clustering_gamma", 1.5)

    # Defensive: synthetic utopian_bounds may not carry an S_max key.
    s_max_bound = utopian_bounds.get("S_max", [0.05, 0.15])
    s_max_ceiling = min(
        round(s_max_bound[1] * s_max_mult, 4), s_max_cap)

    if lambda_search_bounds is None or lambda_search_bounds[0] is None:
        lo_d = lambda_min / n_genes
        hi_d = lambda_max / n_genes
    else:
        lo_d = lambda_search_bounds[0]
        hi_d = lambda_search_bounds[1]

    # lambda_search_bounds may arrive as density fraction (lo_d < 1.0, e.g. 0.008)
    # or as edges/gene (lo_d >= 1.0, e.g. 40.0).  Both yield the same min_clust.
    # NB: the historical bug multiplied an edges/gene value by n_genes a second
    # time, producing a clustering floor of ~60 against a metric capped at 1.0,
    # which shattered every graph as "clustering_collapse".
    if lo_d >= 1.0:
        mean_lam_per_gene = (lo_d + hi_d) / 2.0           # already edges/gene
    else:
        mean_lam_per_gene = (lo_d + hi_d) / 2.0 * n_genes  # density → edges/gene
    min_clust = round(clust_gamma * mean_lam_per_gene / n_genes, 6)

    cfg_out = {
        "max_orphan_fraction": cfg_shatter.get("max_orphan_fraction", 0.15),
        "min_gwcc_fraction":   cfg_shatter.get("min_gwcc_fraction", 0.50),
        "min_target_coverage": cfg_shatter.get("min_target_coverage", 0.40),
        "max_hub_saturation":  s_max_ceiling,
        "min_edge_count":      int(lambda_min * n_genes),
        "max_edge_count":      int(lambda_max * n_genes),
        "min_clustering":      min_clust,
    }

    if mode == "synthetic":
        # Disable the clustering floor; heterophilic graphs are valid targets.
        cfg_out["min_clustering"] = cfg_shatter.get("synthetic_min_clustering",
                                                     None)

    return cfg_out