"""
FUNGI v8.0 — Phase 0: Data-Driven Diagnostic Calibration

Major changes from v7.5:
  - λ_eff replaced with specificity-weighted median regulatory reach
  - Fixed critical bug: n_tested was always == n_active (always gave λ=16)
  - S_max probe uses DEG-matrix directness proxy + optional participation ratio
  - Gini probe uses LFC-magnitude for small panels, modified IDF for large
  - Two-pass architecture: pass1 (adata-only), pass2 (parent graph refines S_max, C)
  - Biologist-readable summary output

Public API:
  run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                  raw_sparse_mat=None)
  build_impact_array(...)
  build_shatter_config(...)
"""

import gc
import warnings

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


# ---------------------------------------------------------------------------
# Physical parameter bounds (hard clips)
# ---------------------------------------------------------------------------
PHYSICAL = {
    "alpha": (1.2, 3.5),
    "gini":  (0.30, 0.95),
    "S_max": (0.005, 0.30),
    "Q":     (0.05, 0.80),
    "C":     (0.001, 0.30),
    "rho":   (-0.50, 0.15),
}

FALLBACK_BOUNDS = {
    "alpha": [1.50, 1.90],
    "gini":  [0.62, 0.85],
    "S_max": [0.04, 0.12],
    "Q":     [0.40, 0.58],
    "C":     [0.02, 0.08],
    "rho":   [-0.20, -0.04],
}
FALLBACK_CONFIDENCE = 0.25

MIN_SKELETON_EDGES = 500
LFC_CAUSAL_FLOOR = 0.01


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

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
    return float(1.0 - 2.0 * np.trapz(cum_v, cum_w))


# ---------------------------------------------------------------------------
# λ_eff: specificity-weighted median regulatory reach
# ---------------------------------------------------------------------------

def _compute_lam_eff(deg_matrix, lfc_matrix, valid_cq, name_to_idx,
                     n_active, n_tested, n_genes):
    """
    Estimate expected edges/gene in the pruned graph from perturbation data.

    Uses ChatGPT deep-research formulation: for each perturbed gene i,
    compute a specificity-weighted regulatory reach score:
        u_i = sum_j D[i,j] * a[i,j] * s_j
    where:
        a[i,j] = min(1, |LFC[i,j]| / q90_j)  — bounded LFC amplitude
        s_j = (1 - prevalence_j)^eta           — specificity weight
        eta = 0.5 for small panels, 1.0 for genome-wide

    λ_center = clip(median(u_i), 4, 20)

    Falls back to a simple heuristic if DEG/LFC data is insufficient.
    """
    if n_tested <= 0:
        return 10.0

    # Try the specificity-weighted approach
    if (deg_matrix is not None and deg_matrix.nnz > MIN_SKELETON_EDGES
            and lfc_matrix is not None and len(lfc_matrix) >= 10
            and valid_cq is not None and len(valid_cq) >= 10):
        try:
            # Align perturbation rows to gene indices
            src_indices, lfc_row_idxs = [], []
            for k, gene_name in enumerate(valid_cq):
                idx = name_to_idx.get(str(gene_name))
                if idx is not None:
                    src_indices.append(idx)
                    lfc_row_idxs.append(k)

            if len(src_indices) >= 10:
                src_arr = np.array(src_indices, dtype=np.int64)
                deg_rows = np.asarray(
                    deg_matrix[src_arr, :].toarray(), dtype=np.float64)
                abs_lfc = np.abs(lfc_matrix[lfc_row_idxs, :]).astype(np.float64)

                # q90 per target gene
                q90 = np.percentile(abs_lfc, 90, axis=0)
                q90 = np.where(q90 < 1e-6, 1e-6, q90)
                amp = np.minimum(1.0, abs_lfc / q90[np.newaxis, :])

                # Prevalence and specificity
                prevalence = np.asarray(deg_matrix.sum(axis=0)).ravel()
                m_active = float(max((deg_matrix.sum(axis=1) > 0).sum(), 1))
                prevalence = prevalence / m_active

                # Panel-adaptive eta: small curated panels get less IDF
                eta = 0.5 if n_tested < 200 else 1.0
                spec = np.power(np.clip(1.0 - prevalence, 0.0, 1.0), eta)

                # Per-source regulatory reach
                u_vals = np.sum(deg_rows * amp * spec[np.newaxis, :], axis=1)
                u_vals = u_vals[u_vals > 0]

                if len(u_vals) >= 5:
                    lam_center = float(np.clip(np.median(u_vals), 4.0, 20.0))
                    return lam_center
        except Exception:
            pass

    # Fallback: simple heuristic based on active fraction and panel size
    active_frac = float(n_active) / float(max(n_tested, 1))
    panel_coverage = float(n_tested) / float(max(n_genes, 1))

    # Small curated panels (enriched for regulators) → denser
    if panel_coverage < 0.1:
        lam = float(np.clip(8.0 + 8.0 * active_frac, 8.0, 18.0))
    else:
        # Genome-wide → sparser
        lam = float(np.clip(5.0 + 7.0 * active_frac, 5.0, 14.0))
    return lam


# ---------------------------------------------------------------------------
# Impact array + DEG matrix + LFC matrix builder
# ---------------------------------------------------------------------------

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25, n_jobs=15,
                       max_perts_for_de=500, min_cells_per_pert=5,
                       is_metacell=False, metacell_pooling_factor=None):
    """
    Runs Wilcoxon DE and returns shared data structures for all probes.

    Returns
    -------
    impact_array, perturbation_labels, sample_weights,
    deg_matrix, lfc_matrix, valid_cq, name_to_idx, n_tested
    """
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

    # LFC proxy for stratified ranking
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

    # Stratified selection
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

    # Wilcoxon DE
    keep_mask = ctrl_mask.copy()
    for cond in selected_conds:
        keep_mask = keep_mask | (conditions_arr == cond)
    adata_sub = adata[keep_mask].copy()
    sc.tl.rank_genes_groups(adata_sub, groupby=perturbation_column,
                            reference=control_label, method=de_method,
                            use_raw=False, n_jobs=n_jobs)

    # Gene name → index map
    var_names = list(adata.var_names)
    n_genes = len(var_names)
    name_to_idx = {str(vn): i for i, vn in enumerate(var_names)}
    for cand in ["gene_name", "gene_symbols", "gene_symbol",
                 "feature_name", "symbol", "Symbol"]:
        if cand in adata.var.columns:
            for i, sym in enumerate(adata.var[cand].astype(str).values):
                name_to_idx.setdefault(sym, i)
            break

    # Build impact_array + deg_matrix
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

    # Build LFC matrix
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
# Probe 1 — Alpha: impact_powerlaw_diffusion_shift
# ---------------------------------------------------------------------------

def _probe_alpha(impact_array, lam_eff):
    """Power-law MLE on per-perturbation DEG counts + diffusion shift."""
    if len(impact_array) < 10:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        import powerlaw
        fit = powerlaw.Fit(impact_array, xmin=2, discrete=True, verbose=False)
        a_raw = _safe_float(fit.power_law.alpha, 2.3)
        sigma = max(_safe_float(fit.power_law.sigma, 0.20), 0.10)
        shift = 0.30 * (lam_eff / 10.0)
        center = float(np.clip(a_raw + shift, 1.2, 3.5))
        hw = max(1.96 * sigma, 0.15)
        lo, hi = _clip_bound(center - hw, center + hw, "alpha")
        ks = _safe_float(fit.power_law.D, 0.5)
        conf = float(np.clip(-np.log10(max(1.0 - ks, 1e-6)) / 3.0, 0.1, 1.0))
        return lo, hi, conf
    except Exception:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 2 — Gini: panel-aware LFC-magnitude / modified-IDF
# ---------------------------------------------------------------------------

def _probe_gini(deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff):
    """
    Panel-aware Gini estimator.

    Small panels (< 200 active perts): LFC-magnitude Gini. IDF is disabled
    because high-prevalence targets in curated panels are genuine regulatory
    circuit members, not housekeeping noise. Uses |LFC| sum per source as
    the regulatory output metric — more right-skewed than binary DEG counts,
    correctly capturing the OCT4 >> effector hierarchy.

    Large panels (>= 200): Coherence-aware modified IDF. High-prevalence
    targets are suppressed only if their response is weak and sign-incoherent
    (true noise). Strong, coherent high-prevalence targets are preserved.
    """
    m_active = 0
    if deg_matrix is not None and deg_matrix.nnz > 0:
        m_active = int((np.asarray(deg_matrix.sum(axis=1)).ravel() > 0).sum())

    # --- SMALL PANEL: LFC-magnitude Gini ---
    if m_active < 200:
        if (lfc_matrix is None or len(lfc_matrix) == 0
                or deg_matrix is None or deg_matrix.nnz < 100):
            return _gini_fallback(deg_matrix)

        try:
            src_indices, lfc_row_idxs = [], []
            for k, gene_name in enumerate(valid_cq):
                idx = name_to_idx.get(str(gene_name))
                if idx is not None:
                    src_indices.append(idx)
                    lfc_row_idxs.append(k)

            if len(src_indices) < 10:
                return _gini_fallback(deg_matrix)

            src_arr = np.array(src_indices, dtype=np.int64)
            abs_lfc = np.abs(lfc_matrix[lfc_row_idxs, :]).astype(np.float64)
            deg_rows = np.asarray(
                deg_matrix[src_arr, :].toarray(), dtype=np.float64)

            # LFC-magnitude weighted regulatory output per source gene
            g_lfc = np.sum(deg_rows * abs_lfc, axis=1)
            g_lfc = g_lfc[g_lfc > 0]

            if len(g_lfc) < 10:
                return _gini_fallback(deg_matrix)

            # Small-sample correction (Deltas 2003): G_adj = G_raw * n/(n-1)
            n_obs = len(g_lfc)
            gini_raw = _gini(g_lfc)
            gini_adj = gini_raw * n_obs / max(n_obs - 1, 1)
            center = float(np.clip(gini_adj, 0.30, 0.95))

            lo, hi = _clip_bound(center - 0.10, center + 0.10, "gini")
            conf = 0.65
            return lo, hi, conf

        except Exception:
            return _gini_fallback(deg_matrix)

    # --- LARGE PANEL: coherence-aware IDF ---
    try:
        n_genes = deg_matrix.shape[0]
        prevalence = np.asarray(deg_matrix.sum(axis=0)).ravel().astype(np.float64)

        if m_active < 5:
            return _gini_fallback(deg_matrix)

        # Coherence-aware nuisance scoring
        # High-prevalence targets are penalized only if response is weak/incoherent
        if lfc_matrix is not None and len(lfc_matrix) >= 10:
            src_indices, lfc_row_idxs = [], []
            for k, gene_name in enumerate(valid_cq):
                idx = name_to_idx.get(str(gene_name))
                if idx is not None:
                    src_indices.append(idx)
                    lfc_row_idxs.append(k)

            if len(src_indices) >= 10:
                src_arr = np.array(src_indices, dtype=np.int64)
                abs_lfc = np.abs(lfc_matrix[lfc_row_idxs, :]).astype(np.float64)
                deg_rows = np.asarray(
                    deg_matrix[src_arr, :].toarray(), dtype=np.float64)

                # q90 normalization
                q90 = np.percentile(abs_lfc, 90, axis=0)
                q90 = np.where(q90 < 1e-6, 1e-6, q90)
                amp = np.minimum(1.0, abs_lfc / q90[np.newaxis, :])

                # Sign coherence per target gene
                signed_lfc = lfc_matrix[lfc_row_idxs, :].astype(np.float64)
                sign_sum = np.abs(np.sum(np.sign(signed_lfc) * deg_rows, axis=0))
                deg_sum = np.sum(deg_rows, axis=0) + 1e-6
                coherence = sign_sum / deg_sum  # 1.0 = perfectly coherent

                # Median amplitude per target
                median_amp = np.median(amp * deg_rows, axis=0)

                # Nuisance penalty: high-prevalence AND low-coherence AND low-amplitude
                pi = prevalence / max(m_active, 1)
                alpha_pen = 0.85  # strong penalty for genome-wide
                nuisance = alpha_pen * pi * (1.0 - coherence) * (1.0 - median_amp)
                weight = np.clip(1.0 - nuisance, 0.05, 1.0)

                # Per-source regulatory reach
                g_vals = np.sum(deg_rows * amp * weight[np.newaxis, :], axis=1)
                g_vals = g_vals[g_vals > 0]

                if len(g_vals) >= 10:
                    center = float(np.clip(_gini(g_vals), 0.30, 0.95))
                    lo, hi = _clip_bound(center - 0.08, center + 0.08, "gini")
                    conf = 0.70
                    return lo, hi, conf

        # Simpler IDF fallback for large panels without LFC
        idf = np.log((m_active + 1.0) / (1.0 + prevalence))
        od = np.asarray(deg_matrix.sum(axis=1)).ravel().astype(np.float64)
        od_weighted = od * 1.0  # no IDF on source side
        od_weighted = od_weighted[od_weighted > 0]
        if len(od_weighted) >= 10:
            center = float(np.clip(_gini(od_weighted), 0.30, 0.95))
            lo, hi = _clip_bound(center - 0.10, center + 0.10, "gini")
            return lo, hi, 0.50

    except Exception:
        pass

    return _gini_fallback(deg_matrix)


def _gini_fallback(deg_matrix):
    try:
        if deg_matrix is not None and deg_matrix.nnz > 0:
            od = np.asarray(deg_matrix.sum(axis=1)).ravel().astype(np.float64)
            od = od[od > 0]
            if len(od) >= 5:
                n_obs = len(od)
                gini_raw = _gini(od)
                gini_adj = gini_raw * n_obs / max(n_obs - 1, 1)
                center = float(np.clip(gini_adj, 0.30, 0.95))
                lo, hi = _clip_bound(center - 0.12, center + 0.12, "gini")
                return lo, hi, FALLBACK_CONFIDENCE
    except Exception:
        pass
    fb = FALLBACK_BOUNDS["gini"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 3 — S_max: hybrid DEG-directness + participation ratio
# ---------------------------------------------------------------------------

def _probe_smax(raw_sparse_mat, deg_matrix, lfc_matrix, valid_cq,
                name_to_idx, n_genes, lam_eff, impact_array):
    """
    Hybrid S_max estimator (ChatGPT deep-research design).

    DEG-based directness proxy (always available):
        d_deg_i = sum_j D[i,j] * a[i,j] * (1-pi_j)^eta
    where a[i,j] = min(1, |LFC|/q90) and pi_j is target prevalence.

    Graph-based effective support (when parent graph available):
        d_graph_i = participation ratio of top-L outgoing weights per source row
        = 1 / sum(p_ij^2) where p_ij = w_ij / sum(w)

    Combined: d_i = sqrt(d_deg * d_graph) when both available, else d_deg only.
    S_max = median(top-3 d_i / n_genes).
    """
    # --- Path A: DEG-based directness proxy ---
    d_deg = None
    if (deg_matrix is not None and deg_matrix.nnz > 100
            and lfc_matrix is not None and len(lfc_matrix) >= 5
            and valid_cq is not None):
        try:
            src_indices, lfc_row_idxs = [], []
            for k, gene_name in enumerate(valid_cq):
                idx = name_to_idx.get(str(gene_name))
                if idx is not None:
                    src_indices.append(idx)
                    lfc_row_idxs.append(k)

            if len(src_indices) >= 5:
                src_arr = np.array(src_indices, dtype=np.int64)
                A = deg_matrix[src_arr, :].astype(np.float64).toarray()
                L = np.abs(lfc_matrix[lfc_row_idxs, :]).astype(np.float64)

                q90 = np.percentile(L, 90, axis=0)
                q90[q90 < 1e-6] = 1e-6
                amp = np.minimum(1.0, L / q90[np.newaxis, :])

                prevalence = A.sum(axis=0) / max(A.shape[0], 1)
                eta = 0.5 if A.shape[0] < 200 else 1.0
                spec = np.power(np.clip(1.0 - prevalence, 0.0, 1.0), eta)

                d_deg = (A * amp * spec[np.newaxis, :]).sum(axis=1)
        except Exception:
            d_deg = None

    # --- Path B: graph-based participation ratio ---
    d_graph = None
    topL = 256
    if (raw_sparse_mat is not None and raw_sparse_mat.nnz > 0
            and d_deg is not None and len(src_indices) >= 5):
        try:
            G = raw_sparse_mat.tocsr()
            d_graph = np.zeros(len(src_indices), dtype=np.float64)

            for k, s in enumerate(src_indices):
                start, end = G.indptr[s], G.indptr[s + 1]
                row = G.data[start:end].copy()
                if row.size == 0:
                    d_graph[k] = max(d_deg[k], 1.0)
                    continue

                if row.size > topL:
                    keep = np.argpartition(row, -topL)[-topL:]
                    row = row[keep]

                row = np.maximum(row, 0.0)
                total = row.sum()
                if total <= 0:
                    d_graph[k] = max(d_deg[k], 1.0)
                    continue

                p = row / total
                d_graph[k] = 1.0 / np.sum(p ** 2)
        except Exception:
            d_graph = None

    # --- Combine estimates ---
    if d_deg is not None and len(d_deg) >= 5:
        if d_graph is not None:
            d_hat = np.sqrt(np.maximum(d_deg, 1e-6) * np.maximum(d_graph, 1e-6))
            conf = 0.70
        else:
            d_hat = d_deg
            conf = 0.55

        s_vals = np.sort(d_hat / float(n_genes))
        top = s_vals[-min(3, len(s_vals)):]
        center = float(np.median(top))
        spread = float(top.max() - top.min()) if len(top) > 1 else 0.0
        half_width = max(0.015, 0.25 * spread, 0.20 * center)

        lo = float(np.clip(center - half_width, 0.01, 0.20))
        hi = float(np.clip(center + half_width, 0.02, 0.25))
        lo, hi = _clip_bound(lo, hi, "S_max")
        return lo, hi, conf

    # --- Path C: simple impact-array fallback ---
    if impact_array is not None and len(impact_array) >= 5:
        try:
            nz = impact_array[impact_array > 0]
            max_impact = float(nz.max())
            mean_impact = float(nz.mean())
            # Cascade correction: ratio of max to mean
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
# Probe 4 — Q: lfc_multiresolution_modularity (unchanged from v7.5)
# ---------------------------------------------------------------------------

def _probe_Q(lfc_matrix, valid_cq, name_to_idx, deg_matrix, n_genes, lam_eff):
    """Multi-resolution Louvain modularity on the LFC-causal graph."""
    if (lfc_matrix is None or len(lfc_matrix) == 0
            or deg_matrix is None or deg_matrix.nnz < MIN_SKELETON_EDGES):
        fb = FALLBACK_BOUNDS["Q"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        import igraph as ig

        rows, cols, vals = [], [], []
        for k, gene_name in enumerate(valid_cq):
            src_idx = name_to_idx.get(str(gene_name))
            if src_idx is None:
                continue
            lfc_row = np.abs(lfc_matrix[k])
            sig_mask = lfc_row > LFC_CAUSAL_FLOOR
            if deg_matrix.nnz > 0:
                deg_row = np.asarray(deg_matrix[src_idx, :].toarray()).ravel()
                sig_mask = sig_mask & (deg_row > 0)
            for j in np.where(sig_mask)[0]:
                if int(j) != src_idx:
                    rows.append(src_idx)
                    cols.append(int(j))
                    vals.append(float(lfc_row[j]))

        if len(rows) < MIN_SKELETON_EDGES:
            rows, cols, vals = [], [], []
            for k, gene_name in enumerate(valid_cq):
                src_idx = name_to_idx.get(str(gene_name))
                if src_idx is None:
                    continue
                lfc_row = np.abs(lfc_matrix[k])
                sig_mask = lfc_row > LFC_CAUSAL_FLOOR
                for j in np.where(sig_mask)[0]:
                    if int(j) != src_idx:
                        rows.append(src_idx)
                        cols.append(int(j))
                        vals.append(float(lfc_row[j]))

        if len(rows) < 100:
            fb = FALLBACK_BOUNDS["Q"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        coo_u = sp.coo_matrix((vals, (rows, cols)), shape=(n_genes, n_genes))
        G = ig.Graph(n=n_genes,
                     edges=list(zip(coo_u.row.tolist(), coo_u.col.tolist())),
                     directed=False)

        q_vals = []
        for resolution in [0.4, 0.5, 0.6]:
            for _ in range(5):
                try:
                    part = G.community_multilevel(resolution=resolution)
                    q_vals.append(float(part.modularity))
                except Exception:
                    q_vals.append(0.45)

        center = float(np.median(q_vals))
        iqr = float(np.subtract(*np.percentile(q_vals, [75, 25])))
        hw = max(iqr, 0.04)
        lo, hi = _clip_bound(center - hw, center + hw, "Q")
        conf = float(np.clip(1.0 - iqr / 0.3, 0.2, 0.9))
        return lo, hi, conf

    except Exception:
        fb = FALLBACK_BOUNDS["Q"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 5 — C: topweight_transitivity (with improved fallback)
# ---------------------------------------------------------------------------

def _probe_C(raw_sparse_mat, n_genes, lam_eff, deg_matrix=None,
             lfc_matrix=None, valid_cq=None, name_to_idx=None):
    """
    Global transitivity on the topweight graph at λ_eff edges/gene.

    Fallback: DEG co-occurrence Jaccard graph transitivity (Gemini design).
    This is more principled than LFC cosine similarity because it measures
    target-set overlap, which directly proxies feed-forward loop density.
    """
    # Primary: topweight transitivity
    try:
        if raw_sparse_mat is not None and raw_sparse_mat.nnz > 0:
            import igraph as ig
            n_keep = int(lam_eff * n_genes)
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
            c_val = float(G.transitivity_undirected())
            if not np.isfinite(c_val):
                c_val = 0.06
            lo, hi = _clip_bound(c_val - 0.03, c_val + 0.03, "C")
            return lo, hi, 0.70
    except Exception:
        pass

    # Fallback: DEG co-occurrence Jaccard transitivity
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

                # Jaccard similarity between perturbation target sets
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
                    # Threshold at P90 of Jaccard scores
                    jac_scores = np.array([e[2] for e in edges])
                    thresh = np.percentile(jac_scores, 90)
                    filtered = [(i, j) for i, j, s in edges if s >= thresh]

                    if len(filtered) >= 5:
                        G_jac = ig.Graph(n=n_perts, edges=filtered,
                                         directed=False).simplify()
                        c_jac = float(G_jac.transitivity_undirected())
                        if np.isfinite(c_jac):
                            # Scale: Jaccard transitivity on perturbation graph
                            # maps to actual C via empirical calibration
                            k_cal = 1.0 / (5.0 + 3.0 * (lam_eff / 15.0))
                            c_target = float(np.clip(c_jac * k_cal, 0.001, 0.15))
                            lo, hi = _clip_bound(c_target - 0.025, c_target + 0.025, "C")
                            return lo, hi, 0.35
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["C"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 6 — Rho: lfc_l2_bipartite_assortativity (unchanged from v7.5)
# ---------------------------------------------------------------------------

def _probe_rho(lfc_matrix, valid_cq, name_to_idx):
    """Spearman of per-perturbation LFC L2 norms vs per-gene column L2 norms."""
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
        lo, hi = _clip_bound(center - 0.08, center + 0.08, "rho")
        conf = float(np.clip(1.0 - float(pval), 0.15, 0.90)) if np.isfinite(pval) else 0.50
        return lo, hi, conf

    except Exception:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Bound enforcement + weight normalisation
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


def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
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
    values = np.clip(values, floor, ceiling)
    return {n: float(v) for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# Biologist-readable summary
# ---------------------------------------------------------------------------

_PARAM_DESCRIPTIONS = {
    "alpha": "Scale-free degree exponent",
    "gini":  "Regulatory inequality (hub dominance)",
    "S_max": "Largest hub's target fraction",
    "Q":     "Functional module separation",
    "C":     "Feed-forward loop density",
    "rho":   "Hub-to-effector disassortativity",
}


def _print_summary(utopian_bounds, loss_weights, raw_confidences, lam_eff,
                   probes_used, n_active, n_tested, n_genes):
    """Print a biologist-readable Phase 0 summary."""
    print(f"\n{'=' * 70}")
    print("FUNGI v8.0 — Phase 0 Diagnostic Summary")
    print(f"{'=' * 70}")
    print(f"  Dataset: {n_genes:,} HVGs | {n_active}/{n_tested} active perturbations")
    print(f"  Estimated graph density (λ_center): {lam_eff:.1f} edges/gene")
    print()

    conf_emoji = {True: "✓", False: "~"}
    for param in ["alpha", "gini", "S_max", "Q", "C", "rho"]:
        lo, hi = utopian_bounds[param]
        cf = raw_confidences[param]
        w = loss_weights[param]
        high_conf = cf >= 0.5
        desc = _PARAM_DESCRIPTIONS[param]
        print(f"  {conf_emoji[high_conf]} {param:>5s} [{lo:.4f}, {hi:.4f}]  "
              f"conf={cf:.2f}  wt={w:.1f}  — {desc}")

    # Sanity warnings
    warnings_raised = []
    if utopian_bounds["S_max"][1] > 0.20:
        warnings_raised.append("S_max upper bound >0.20: unusually large hub expected")
    if utopian_bounds["C"][0] < 0.005:
        warnings_raised.append("C lower bound <0.005: very low clustering expected")
    if n_active < 30:
        warnings_raised.append(f"Only {n_active} active perturbations: low statistical power")

    low_conf_count = sum(1 for v in raw_confidences.values() if v < 0.35)
    if low_conf_count >= 3:
        warnings_raised.append(f"{low_conf_count}/6 probes have low confidence (<0.35)")

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
# Master runner (public API)
# ---------------------------------------------------------------------------

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                    raw_sparse_mat=None):
    """
    Full Phase 0 pipeline (v8.0).

    Key changes from v7.5:
    - λ_eff uses specificity-weighted median regulatory reach (not active_frac)
    - Fixed: n_tested is now the actual number tested, not len(impact_array)
    - S_max uses DEG-directness + participation ratio hybrid
    - Gini uses LFC-magnitude for small panels
    - Biologist-readable output

    Parameters
    ----------
    adata           : AnnData
    n_genes         : int
    cfg_diagnostics : dict
    cfg_input       : dict
    raw_sparse_mat  : scipy.sparse or None — parent graph

    Returns
    -------
    utopian_bounds, loss_weights, diagnostic_report
    """
    pert_col = cfg_input["perturbation_column"]
    ctrl_label = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool = cfg_input.get("metacell_pooling_factor", None)

    bc = cfg_diagnostics["bound_constraints"]
    w_floor = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    n_jobs = cfg_diagnostics.get("n_jobs", 15)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    # Step 1: build shared data (now returns n_tested correctly)
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

    # Step 2: compute λ_eff using new formula
    lam_eff = _compute_lam_eff(
        deg_matrix, lfc_matrix, valid_cq, name_to_idx,
        n_active, n_tested, n_genes)
    print(f"\n  λ_center = {lam_eff:.2f} edges/gene")

    # Step 3: run probes
    print("\n  Running probes...")
    alpha_lo, alpha_hi, alpha_conf = _probe_alpha(impact_array, lam_eff)

    gini_lo, gini_hi, gini_conf = _probe_gini(
        deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff)

    smax_lo, smax_hi, smax_conf = _probe_smax(
        raw_sparse_mat, deg_matrix, lfc_matrix, valid_cq,
        name_to_idx, n_genes, lam_eff, impact_array)

    Q_lo, Q_hi, Q_conf = _probe_Q(
        lfc_matrix, valid_cq, name_to_idx, deg_matrix, n_genes, lam_eff)

    C_lo, C_hi, C_conf = _probe_C(
        raw_sparse_mat, n_genes, lam_eff,
        deg_matrix, lfc_matrix, valid_cq, name_to_idx)

    rho_lo, rho_hi, rho_conf = _probe_rho(lfc_matrix, valid_cq, name_to_idx)

    # Step 4: enforce bound constraints
    alpha_lo, alpha_hi = _enforce_bound_constraints(
        alpha_lo, alpha_hi, (alpha_lo + alpha_hi) / 2, bc["alpha"])
    gini_lo, gini_hi = _enforce_bound_constraints(
        gini_lo, gini_hi, (gini_lo + gini_hi) / 2, bc["gini"])
    smax_lo, smax_hi = _enforce_bound_constraints(
        smax_lo, smax_hi, (smax_lo + smax_hi) / 2, bc["S_max"])
    Q_lo, Q_hi = _enforce_bound_constraints(
        Q_lo, Q_hi, (Q_lo + Q_hi) / 2, bc["Q"])
    C_lo, C_hi = _enforce_bound_constraints(
        C_lo, C_hi, (C_lo + C_hi) / 2, bc["C"])
    rho_lo, rho_hi = _enforce_bound_constraints(
        rho_lo, rho_hi, (rho_lo + rho_hi) / 2, bc["rho"])

    utopian_bounds = {
        "alpha": [alpha_lo, alpha_hi],
        "gini":  [gini_lo, gini_hi],
        "S_max": [smax_lo, smax_hi],
        "Q":     [Q_lo, Q_hi],
        "C":     [C_lo, C_hi],
        "rho":   [rho_lo, rho_hi],
    }

    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]

    raw_confidences = {
        "alpha": alpha_conf, "gini": gini_conf, "S_max": smax_conf,
        "Q": Q_conf, "C": C_conf, "rho": rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    probes_used = {
        "alpha": "impact_powerlaw_diffusion_shift",
        "gini":  "panel_aware_lfc_magnitude" if n_active < 200 else "coherence_aware_idf",
        "S_max": "deg_directness_participation_ratio",
        "Q":     "lfc_multiresolution_modularity",
        "C":     "topweight_transitivity" if (raw_sparse_mat is not None) else "deg_jaccard_fallback",
        "rho":   "lfc_l2_bipartite_assortativity",
    }

    proceed = _print_summary(
        utopian_bounds, loss_weights, raw_confidences, lam_eff,
        probes_used, n_active, n_tested, n_genes)

    diagnostic_report = {
        "version": "9.0",
        "lam_eff": lam_eff,
        "n_active": n_active,
        "n_tested": n_tested,
        "impact_range": ([float(impact_array.min()), float(impact_array.max())]
                         if len(impact_array) > 0 else [0, 0]),
        "deg_matrix_nnz": int(deg_matrix.nnz),
        "lfc_matrix_shape": list(lfc_matrix.shape),
        "is_metacell": is_metacell,
        "probes_used": probes_used,
        "raw_confidences": {k: _safe_float(v) for k, v in raw_confidences.items()},
        "utopian_bounds": utopian_bounds,
        "loss_weights": loss_weights,
        "proceed": proceed,
        # Stored for downstream use in engine (ψ prior computation)
        "_impact_array": impact_array.tolist() if len(impact_array) > 0 else [],
        "_perturbation_labels": perturbation_labels.tolist() if len(perturbation_labels) > 0 else [],
        "_name_to_idx": name_to_idx,
    }

    return utopian_bounds, loss_weights, diagnostic_report


# ---------------------------------------------------------------------------
# Shatter config builder
# ---------------------------------------------------------------------------

def build_shatter_config(cfg_shatter, n_genes, utopian_bounds,
                         lambda_search_bounds):
    lambda_min = cfg_shatter.get("lambda_min", 2.0)
    lambda_max = cfg_shatter.get("lambda_max", 30.0)
    s_max_mult = cfg_shatter.get("s_max_ceiling_multiplier", 2.0)
    s_max_cap = cfg_shatter.get("s_max_ceiling_hard_cap", 0.30)
    clust_gamma = cfg_shatter.get("min_clustering_gamma", 1.5)

    s_max_ceiling = min(
        round(utopian_bounds["S_max"][1] * s_max_mult, 4), s_max_cap)

    if lambda_search_bounds is None or lambda_search_bounds[0] is None:
        lo_d = lambda_min / n_genes
        hi_d = lambda_max / n_genes
    else:
        lo_d = lambda_search_bounds[0]
        hi_d = lambda_search_bounds[1]

    lam_exp = (lo_d + hi_d) / 2 * n_genes
    min_clust = round(clust_gamma * (lam_exp / n_genes), 6)

    return {
        "max_orphan_fraction": cfg_shatter.get("max_orphan_fraction", 0.15),
        "min_gwcc_fraction": cfg_shatter.get("min_gwcc_fraction", 0.35),
        "max_hub_saturation": s_max_ceiling,
        "min_edge_count": int(lambda_min * n_genes),
        "max_edge_count": int(lambda_max * n_genes),
        "min_clustering": min_clust,
    }
