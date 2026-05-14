"""
FUNGI v7.5 — Phase 0: Data-Driven Diagnostic Calibration

Production probe roster (selected after R1–R5 tournament, May 2026):

  alpha  ← impact_powerlaw_diffusion_shift
           Power-law MLE on per-perturbation DEG impact counts, plus a
           λ_eff-scaled diffusion-attenuation shift. Uses Wilcoxon DEG
           data only — independent of LightGBM weights.

  gini   ← specificity_idf_lfc_gini
           IDF-weighted, |LFC|/q90-normalised regulatory reach. Fully
           independent of LightGBM. Converges with topweight-based Gini
           (R5 circular-logic validation: 2.6% normalised divergence).

  S_max  ← topweight_sparse_hub_fraction
           Max hub fraction on a sparse topweight substrate at (λ_eff−3).
           Best EM by a substantial margin (1.377 vs 1.589 next-best).
           Regulon-DB probe validates the estimate externally.

  Q      ← lfc_multiresolution_modularity
           Louvain at 3 resolutions × 5 seeds on the LFC-causal graph.
           Evidence-based; R5 confirmed substrate-independence (1.8%
           divergence from topweight probe).

  C      ← topweight_transitivity
           Global transitivity on topweight graph at exactly λ_eff.
           Circular-logic concern resolved: lfc_cosine_calibrated probe
           converges to within 4% normalised divergence (R5 finding).
           CHITIN correction: C is ~38-43% lower on CHITIN-preprocessed
           input; this probe detects it automatically via the parent graph.

  rho    ← lfc_l2_bipartite_assortativity
           Spearman of per-perturbation LFC L2 norms (regulatory-output
           proxy) vs per-gene LFC column L2 norms (responsiveness proxy),
           aligned by gene name. Fully evidence-based; converges with the
           topweight probe at 2.3% normalised divergence (R5 validation).

Public API (identical to v7.4 for drop-in compatibility):
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
# Physical parameter bounds (hard clips — nothing outside these is biology)
# ---------------------------------------------------------------------------
PHYSICAL = {
    "alpha": (1.2,  3.5),
    "gini":  (0.30, 0.95),
    "S_max": (0.005, 0.30),
    "Q":     (0.05, 0.80),
    "C":     (0.001, 0.30),
    "rho":   (-0.50, 0.15),
}

FALLBACK_BOUNDS = {
    "alpha": [2.1, 2.7],
    "gini":  [0.55, 0.75],
    "S_max": [0.04, 0.10],
    "Q":     [0.30, 0.55],
    "C":     [0.03, 0.09],
    "rho":   [-0.20, -0.04],
}
FALLBACK_CONFIDENCE = 0.3

MIN_SKELETON_EDGES = 500
LFC_CAUSAL_FLOOR   = 0.01


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
        eps    = (ceiling - floor) * 0.02
        lo     = max(center - eps, floor)
        hi     = min(center + eps, ceiling)
    return lo, hi


def _compute_lam_eff(impact_array, n_tested):
    """λ_eff: expected edges/gene in the pruned graph."""
    if n_tested <= 0 or len(impact_array) == 0:
        return 10.0
    active_frac = float(len(impact_array)) / float(n_tested)
    return float(np.clip(6.0 + 10.0 * min(max(active_frac, 0.0), 1.0), 6.0, 16.0))


# ---------------------------------------------------------------------------
# Impact array + DEG matrix + LFC matrix
# ---------------------------------------------------------------------------

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25, n_jobs=15,
                       max_perts_for_de=500, min_cells_per_pert=5,
                       is_metacell=False, metacell_pooling_factor=None):
    """
    Runs Wilcoxon DE on a stratified subset of perturbations and returns:

      impact_array       : float64 (n_active,) — DEG count per perturbation
      perturbation_labels: str array (n_active,) — gene names of those perts
      sample_weights     : float64 (n_active,) — Top-K / Random-Tail weights
      deg_matrix         : sparse (n_genes × n_genes) — M[i,j]=1 iff gene j
                           was a DEG when gene i was perturbed
      lfc_matrix         : float32 (n_valid_lfc × n_genes) — mean log-fold-
                           change vectors; rows indexed by valid_cq
      valid_cq           : list of str — gene labels for lfc_matrix rows
      name_to_idx        : dict str→int — gene name to var_names position

    Compared with v7.4, this function now also builds lfc_matrix and
    valid_cq so all downstream probes can share the computation.

    Metacell correction: effective LFC cutoff is scaled by
    1/sqrt(metacell_pooling_factor) when is_metacell=True.
    """
    print("Phase 0: Building perturbation impact array...")

    if is_metacell and metacell_pooling_factor and metacell_pooling_factor > 1:
        effective_lfc = lfc_threshold / np.sqrt(metacell_pooling_factor)
        print(f"  Metacell pooling (factor={metacell_pooling_factor}): "
              f"LFC cutoff {lfc_threshold:.3f} → {effective_lfc:.3f}")
    else:
        effective_lfc = lfc_threshold

    conditions_arr = adata.obs[perturbation_column].values
    ctrl_mask      = conditions_arr == control_label
    unique_conds   = [c for c in np.unique(conditions_arr) if c != control_label]
    n_total        = len(unique_conds)
    print(f"  {n_total:,} perturbation groups detected.")

    # LFC proxy for stratified ranking
    X_csr     = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
    ctrl_mean = np.asarray(X_csr[ctrl_mask].mean(axis=0)).ravel()
    log_ctrl  = np.log1p(np.maximum(ctrl_mean, 0))

    proxy_scores = {}
    for cond in tqdm(unique_conds, desc="  LFC proxy", unit="pert", ncols=80):
        mask = conditions_arr == cond
        if mask.sum() < 2:
            proxy_scores[cond] = 0.0
            continue
        cm  = np.asarray(X_csr[mask].mean(axis=0)).ravel()
        proxy_scores[cond] = float(np.mean(np.abs(np.log1p(np.maximum(cm, 0)) - log_ctrl)))

    # Stratified selection: Top-K + Random Tail
    selected_conds = unique_conds
    sample_weights_map = {c: 1.0 for c in unique_conds}

    if n_total > max_perts_for_de:
        sorted_conds = sorted(unique_conds, key=lambda c: proxy_scores[c], reverse=True)
        n_top        = min(100, max_perts_for_de // 5)
        n_random     = max_perts_for_de - n_top
        top_conds    = sorted_conds[:n_top]
        tail_conds   = sorted_conds[n_top:]
        rng          = np.random.default_rng(42)
        n_draw       = min(n_random, len(tail_conds))
        random_conds = list(rng.choice(tail_conds, size=n_draw, replace=False))
        selected_conds = top_conds + random_conds
        tail_w = float(len(tail_conds)) / max(n_draw, 1)
        for c in top_conds:    sample_weights_map[c] = 1.0
        for c in random_conds: sample_weights_map[c] = tail_w
        print(f"  Selected {len(selected_conds)} perts "
              f"({n_top} top-proxy + {len(random_conds)} random tail).")
    else:
        print(f"  Running Wilcoxon on all {n_total} perturbations.")

    # Wilcoxon DE on subset
    keep_mask = ctrl_mask.copy()
    for cond in selected_conds:
        keep_mask = keep_mask | (conditions_arr == cond)
    adata_sub = adata[keep_mask].copy()
    sc.tl.rank_genes_groups(adata_sub, groupby=perturbation_column,
                            reference=control_label, method=de_method,
                            use_raw=False, n_jobs=n_jobs)

    # Gene name → index map
    var_names   = list(adata.var_names)
    n_genes     = len(var_names)
    name_to_idx = {str(vn): i for i, vn in enumerate(var_names)}
    symbol_col  = None
    for cand in ["gene_name", "gene_symbols", "gene_symbol",
                 "feature_name", "symbol", "Symbol"]:
        if cand in adata.var.columns:
            symbol_col = cand
            break
    if symbol_col is not None:
        for i, sym in enumerate(adata.var[symbol_col].astype(str).values):
            name_to_idx.setdefault(sym, i)

    # Build impact_array + deg_matrix
    impact_scores, valid_labels, valid_weights_list = [], [], []
    deg_rows, deg_cols = [], []
    n_tested = 0

    for cond in tqdm(selected_conds, desc="  DEG counts", unit="pert", ncols=80):
        n_tested += 1
        try:
            df  = sc.get.rank_genes_groups_df(adata_sub, group=cond)
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

    impact_array        = np.array(impact_scores, dtype=np.float64)
    perturbation_labels = np.array(valid_labels)
    weights_arr         = np.array(valid_weights_list, dtype=np.float64)
    nz                  = impact_array > 0
    impact_array        = impact_array[nz]
    perturbation_labels = perturbation_labels[nz]
    weights_arr         = weights_arr[nz]

    if len(deg_rows) > 0:
        deg_matrix = sp.coo_matrix(
            (np.ones(len(deg_rows)), (np.array(deg_rows), np.array(deg_cols))),
            shape=(n_genes, n_genes)).tocsr()
        deg_matrix.data = np.ones_like(deg_matrix.data)
    else:
        deg_matrix = sp.csr_matrix((n_genes, n_genes))

    # Build LFC matrix (n_valid_lfc × n_genes)
    X_full    = (adata.X.toarray() if hasattr(adata.X, 'toarray')
                 else np.array(adata.X)).astype(np.float32)
    ctrl_mean_full = X_full[ctrl_mask].mean(axis=0)
    log_ctrl_full  = np.log1p(np.maximum(ctrl_mean_full, 0))

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

    print(f"  Active perts    : {len(impact_array)} / {n_tested} tested")
    print(f"  DEG matrix      : {deg_matrix.nnz:,} causal edges")
    print(f"  LFC matrix      : {lfc_matrix.shape[0]} perturbations × {n_genes} genes")

    return (impact_array, perturbation_labels, weights_arr,
            deg_matrix, lfc_matrix, valid_cq, name_to_idx)


# ---------------------------------------------------------------------------
# Probe 1 — Alpha: impact_powerlaw_diffusion_shift
# ---------------------------------------------------------------------------

def _probe_alpha(impact_array, lam_eff):
    """
    Fit a discrete power law to the per-perturbation DEG impact array.
    Applies a λ_eff-scaled diffusion-attenuation shift to convert from
    impact-space α to graph-degree-space α.

    The shift 0.30 × (λ_eff / 10) corrects for indirect downstream effects
    captured in the impact array but absent in the pruned graph's degree
    distribution. Denser expected graphs (higher λ_eff) need larger corrections.
    """
    if len(impact_array) < 10:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        import powerlaw
        fit    = powerlaw.Fit(impact_array, xmin=2, discrete=True, verbose=False)
        a_raw  = _safe_float(fit.power_law.alpha, 2.3)
        sigma  = max(_safe_float(fit.power_law.sigma, 0.20), 0.10)
        shift  = 0.30 * (lam_eff / 10.0)
        center = float(np.clip(a_raw + shift, 1.2, 3.5))
        hw     = max(1.96 * sigma, 0.15)
        lo, hi = _clip_bound(center - hw, center + hw, "alpha")
        # Confidence from goodness-of-fit
        ks  = _safe_float(fit.power_law.D, 0.5)
        conf = float(np.clip(-np.log10(max(1.0 - ks, 1e-6)) / 3.0, 0.1, 1.0))
        return lo, hi, conf
    except Exception:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 2 — Gini: specificity_idf_lfc_gini
# ---------------------------------------------------------------------------

def _probe_gini(deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff):
    """
    IDF-weighted, |LFC|/q90-normalised regulatory reach Gini.

    For each perturbed gene i, compute a specificity-weighted regulatory
    reach score:
        g_i = Σ_j  1[DEG_ij] · IDF_j · min(1, |LFC_ij| / q90_j)

    where:
        IDF_j = log((M+1) / (1+p_j))  — down-weights genes that respond
                to many perturbations (systematic variation or housekeeping)
        q90_j = 90th percentile of |LFC_j| across all perturbations
                — normalises by each gene's typical response scale
        p_j   = prevalence of gene j as a DEG across all perturbations

    The Gini of {g_i} measures out-degree inequality without using any
    LightGBM edge weights. R5 validation: 2.6% normalised divergence
    from topweight-based Gini → circular logic concern resolved.
    """
    if (deg_matrix is None or deg_matrix.nnz < MIN_SKELETON_EDGES
            or lfc_matrix is None or len(lfc_matrix) == 0):
        # Fallback: weighted Gini on raw impact counts
        return _gini_fallback(deg_matrix)

    try:
        n_genes = deg_matrix.shape[0]

        # Prevalence and IDF
        prevalence = np.asarray(deg_matrix.sum(axis=0)).ravel().astype(np.float64)
        M_active   = float((deg_matrix.sum(axis=1) > 0).sum())
        if M_active < 5:
            return _gini_fallback(deg_matrix)
        idf = np.log((M_active + 1.0) / (1.0 + prevalence))

        # q90 of |LFC| per gene (across perturbations)
        abs_lfc = np.abs(lfc_matrix)
        q90     = np.percentile(abs_lfc, 90, axis=0)
        q90     = np.where(q90 < 1e-6, 1e-6, q90)

        # Align lfc_matrix rows to deg_matrix rows by gene name
        src_indices, lfc_row_idxs = [], []
        for k, gene_name in enumerate(valid_cq):
            idx = name_to_idx.get(str(gene_name))
            if idx is not None:
                src_indices.append(idx)
                lfc_row_idxs.append(k)

        if len(src_indices) < 10:
            return _gini_fallback(deg_matrix)

        src_arr  = np.array(src_indices, dtype=np.int64)
        abs_lfc_aligned = abs_lfc[lfc_row_idxs, :]  # (n_valid × n_genes)
        deg_rows = np.asarray(
            deg_matrix[src_arr, :].toarray(), dtype=np.float64)

        # Specificity reach per gene
        lfc_weight = np.minimum(1.0, abs_lfc_aligned / q90[np.newaxis, :])
        g_vals = np.sum(deg_rows * idf[np.newaxis, :] * lfc_weight, axis=1)
        g_vals = g_vals[g_vals > 0]

        if len(g_vals) < 10:
            return _gini_fallback(deg_matrix)

        center = float(np.clip(_gini(g_vals), 0.30, 0.95))
        lo, hi = _clip_bound(center - 0.08, center + 0.08, "gini")
        conf   = 0.75
        return lo, hi, conf

    except Exception:
        return _gini_fallback(deg_matrix)


def _gini_fallback(deg_matrix):
    """Fallback: Gini of out-degrees from deg_matrix."""
    try:
        if deg_matrix is not None and deg_matrix.nnz > 0:
            od = np.asarray(deg_matrix.sum(axis=1)).ravel().astype(np.float64)
            od = od[od > 0]
            if len(od) >= 5:
                center = float(np.clip(_gini(od), 0.30, 0.95))
                lo, hi = _clip_bound(center - 0.10, center + 0.10, "gini")
                return lo, hi, FALLBACK_CONFIDENCE
    except Exception:
        pass
    fb = FALLBACK_BOUNDS["gini"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


def _gini(x):
    x = np.asarray(x, dtype=np.float64)
    if x.sum() == 0:
        return 0.0
    order = np.argsort(x)
    x     = x[order]
    n     = len(x)
    cum_w = np.arange(1, n + 1) / n
    cum_v = np.cumsum(x) / x.sum()
    return float(1.0 - 2.0 * np.trapz(cum_v, cum_w))


# ---------------------------------------------------------------------------
# Probe 3 — S_max: topweight_sparse_hub_fraction
# ---------------------------------------------------------------------------

def _probe_smax(raw_sparse_mat, impact_array, n_genes, lam_eff):
    """
    Hub fraction from topweight-sparse substrate.

    Prunes the parent graph to (λ_eff − 3) × n_genes edges (sparse
    substrate: only the strongest hub connections survive), then returns
    the (P95 + max) / 2 out-degree as the center estimate.

    The −3 offset means only the dominant hub's core connections survive;
    this gives a cleaner estimate of the master regulator's regulatory
    fraction than the full λ_eff graph.

    Fallback to impact-array-based estimate if parent graph is unavailable.
    """
    try:
        if raw_sparse_mat is not None and raw_sparse_mat.nnz > 0:
            lam_sparse = max(lam_eff - 3.0, 4.0)
            n_keep     = int(lam_sparse * n_genes)
            parent     = raw_sparse_mat.tocsr()
            coo        = parent.tocoo()
            if coo.nnz > n_keep:
                top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
                pruned  = sp.coo_matrix(
                    (coo.data[top_idx], (coo.row[top_idx], coo.col[top_idx])),
                    shape=parent.shape).tocsr()
            else:
                pruned = parent
            od  = np.asarray((pruned != 0).sum(axis=1)).ravel()
            mx  = float(od.max()) / n_genes
            p95 = float(np.percentile(od, 95)) / n_genes
            center = (p95 + mx) / 2.0
            hw     = max((mx - p95) / 2.0, 0.02)
            lo, hi = _clip_bound(center - hw, center + hw, "S_max")
            snr  = mx / max(p95, 1e-6)
            conf = float(np.clip(np.log10(max(snr, 1.01)) / 3.0, 0.1, 1.0))
            return lo, hi, conf

    except Exception:
        pass

    # Impact-array fallback
    if len(impact_array) >= 5:
        try:
            p95    = float(np.percentile(impact_array, 95))
            max_i  = float(impact_array.max())
            center = ((p95 + max_i) / 2.0) / n_genes
            center = float(np.clip(center, 0.005, 0.30))
            lo, hi = _clip_bound(center - 0.03, center + 0.05, "S_max")
            return lo, hi, FALLBACK_CONFIDENCE
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["S_max"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 4 — Q: lfc_multiresolution_modularity
# ---------------------------------------------------------------------------

def _probe_Q(lfc_matrix, valid_cq, name_to_idx, deg_matrix, n_genes, lam_eff):
    """
    Multi-resolution Louvain modularity on the LFC-causal graph.

    Builds a directed graph from the DEG matrix weighted by |LFC| magnitude
    (edges where the source gene was perturbed and the target showed a
    significant LFC response). Runs Louvain community detection at three
    resolutions (0.4, 0.5, 0.6) × 5 runs each = 15 estimates.

    Uses median as center and IQR as adaptive bound half-width.
    CHITIN-corrected graphs produce smaller IQR (tighter bounds) because
    their community structure is less ambiguous.

    R5 validation: 1.8% normalised divergence from topweight probe.
    Q ≈ 0.50 is confirmed substrate-independent across all 5 datasets.
    """
    if (lfc_matrix is None or len(lfc_matrix) == 0
            or deg_matrix is None or deg_matrix.nnz < MIN_SKELETON_EDGES):
        fb = FALLBACK_BOUNDS["Q"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        import igraph as ig

        # Build LFC-causal matrix: edges where source was perturbed and
        # target showed |LFC| > threshold, intersected with DEG calls.
        n_perts = len(lfc_matrix)
        rows, cols, vals = [], [], []

        for k, gene_name in enumerate(valid_cq):
            src_idx = name_to_idx.get(str(gene_name))
            if src_idx is None:
                continue
            lfc_row = np.abs(lfc_matrix[k])
            sig_mask = lfc_row > LFC_CAUSAL_FLOOR
            # Additionally require agreement with DEG calls
            if deg_matrix.nnz > 0:
                deg_row  = np.asarray(deg_matrix[src_idx, :].toarray()).ravel()
                sig_mask = sig_mask & (deg_row > 0)
            for j in np.where(sig_mask)[0]:
                if int(j) != src_idx:
                    rows.append(src_idx); cols.append(int(j))
                    vals.append(float(lfc_row[j]))

        if len(rows) < MIN_SKELETON_EDGES:
            # Relax: use pure LFC without DEG intersection
            rows, cols, vals = [], [], []
            for k, gene_name in enumerate(valid_cq):
                src_idx = name_to_idx.get(str(gene_name))
                if src_idx is None:
                    continue
                lfc_row  = np.abs(lfc_matrix[k])
                sig_mask = lfc_row > LFC_CAUSAL_FLOOR
                for j in np.where(sig_mask)[0]:
                    if int(j) != src_idx:
                        rows.append(src_idx); cols.append(int(j))
                        vals.append(float(lfc_row[j]))

        if len(rows) < 100:
            fb = FALLBACK_BOUNDS["Q"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        causal_mat = sp.coo_matrix((vals, (rows, cols)),
                                   shape=(n_genes, n_genes)).tocsr()

        # Build igraph — undirected projection
        coo_u = causal_mat.tocoo()
        G     = ig.Graph(n=n_genes,
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
        iqr    = float(np.subtract(*np.percentile(q_vals, [75, 25])))
        hw     = max(iqr, 0.04)
        lo, hi = _clip_bound(center - hw, center + hw, "Q")
        conf   = float(np.clip(1.0 - iqr / 0.3, 0.2, 0.9))
        return lo, hi, conf

    except Exception:
        fb = FALLBACK_BOUNDS["Q"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 5 — C: topweight_transitivity
# ---------------------------------------------------------------------------

def _probe_C(raw_sparse_mat, n_genes, lam_eff, lfc_matrix=None):
    """
    Global transitivity on the topweight graph at exactly λ_eff edges/gene.

    The "COMPROMISE" probe from R4/R5: no λ offset, exactly λ_eff × n_genes
    edges retained. Outperforms all other C probes by a substantial margin
    (EM 1.804 vs 2.054 next-best).

    CHITIN effect: CHITIN-corrected parent graphs produce C ≈ 0.04–0.05,
    while uncorrected graphs produce C ≈ 0.06–0.08. The probe detects this
    automatically because the CHITIN-processed parent graph has fewer
    co-expression triangles than the uncorrected graph.

    Fallback: LFC cosine calibration if parent graph is unavailable.
    """
    try:
        if raw_sparse_mat is not None and raw_sparse_mat.nnz > 0:
            import igraph as ig
            n_keep  = int(lam_eff * n_genes)
            parent  = raw_sparse_mat.tocsr()
            coo     = parent.tocoo()
            if coo.nnz > n_keep:
                top_idx = np.argpartition(coo.data, -n_keep)[-n_keep:]
                pruned  = sp.coo_matrix(
                    (coo.data[top_idx], (coo.row[top_idx], coo.col[top_idx])),
                    shape=parent.shape).tocsr()
            else:
                pruned = parent
            coo_p = pruned.tocoo()
            G     = ig.Graph(n=n_genes,
                             edges=list(zip(coo_p.row.tolist(), coo_p.col.tolist())),
                             directed=False)
            c_val = float(G.transitivity_undirected())
            if not np.isfinite(c_val):
                c_val = 0.06
            lo, hi = _clip_bound(c_val - 0.03, c_val + 0.03, "C")
            return lo, hi, 0.70

    except Exception:
        pass

    # LFC cosine calibration fallback (no parent graph)
    if lfc_matrix is not None and len(lfc_matrix) >= 10:
        try:
            norms   = np.linalg.norm(lfc_matrix, axis=1, keepdims=True)
            norms   = np.where(norms < 1e-10, 1e-10, norms)
            lfc_n   = lfc_matrix / norms
            sim     = lfc_n @ lfc_n.T
            np.fill_diagonal(sim, 0)
            sim     = np.clip(sim, 0, 1)
            thresh  = 0.30
            adj     = (sim > thresh).astype(np.int8)
            edges   = list(zip(*np.where(adj > 0)))
            if len(edges) >= 10:
                import igraph as ig
                n_p   = sim.shape[0]
                G_cos = ig.Graph(n=n_p, edges=edges, directed=False).simplify()
                c_cos = float(G_cos.transitivity_undirected())
                if np.isfinite(c_cos):
                    k_cal   = 1.0 / (6.0 + 4.0 * (lam_eff / 15.0))
                    c_target = float(np.clip(c_cos * k_cal, 0.001, 0.20))
                    lo, hi  = _clip_bound(c_target - 0.02, c_target + 0.02, "C")
                    return lo, hi, FALLBACK_CONFIDENCE
        except Exception:
            pass

    fb = FALLBACK_BOUNDS["C"]
    return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Probe 6 — Rho: lfc_l2_bipartite_assortativity
# ---------------------------------------------------------------------------

def _probe_rho(lfc_matrix, valid_cq, name_to_idx):
    """
    Spearman correlation of per-perturbation LFC L2 norms vs per-gene
    LFC column L2 norms, aligned by gene name.

    Out-degree proxy: row-wise L2 norm of the LFC matrix — how much total
    transcriptomic disruption does knocking down gene i cause? Genes that
    regulate many targets strongly have large row norms.

    In-degree proxy: column-wise L2 norm at the PERTURBED GENE'S index —
    how strongly does this gene respond when other genes are knocked down?
    This measures whether the gene is also a regulatory target itself.

    A negative Spearman correlation (regulatory powerhouses are NOT highly
    responsive to others' perturbations) maps to disassortative topology
    (hubs connect to non-hubs). Affine mapping:
        rho_center = −0.15 × r_s − 0.10

    R5 validation: 2.3% normalised divergence from topweight probe.
    Fully independent of LightGBM edge weights.

    Critical implementation note: alignment is by GENE NAME via name_to_idx,
    not by positional index. Using positional index was the original Gemini
    bug — it compared mismatched genes and produced garbage correlations.
    """
    if lfc_matrix is None or len(lfc_matrix) < 10 or not valid_cq:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        k_out_all = np.linalg.norm(lfc_matrix, axis=1)  # (n_perts,)
        k_in_all  = np.linalg.norm(lfc_matrix, axis=0)  # (n_genes,)

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

        # Affine transform: pos Spearman (hubs are also responsive → less
        # disassortative) → rho toward zero; neg Spearman → more negative rho
        center = float(np.clip(-0.15 * corr - 0.10, -0.45, 0.10))
        lo, hi = _clip_bound(center - 0.08, center + 0.08, "rho")

        conf = float(np.clip(1.0 - float(pval), 0.15, 0.90)) if np.isfinite(pval) else 0.50
        return lo, hi, conf

    except Exception:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Bound enforcement + weight normalisation (unchanged from v7.4 interface)
# ---------------------------------------------------------------------------

def _enforce_bound_constraints(bound_min, bound_max, center, constraints):
    delta_min    = constraints["delta_min"]
    delta_max    = constraints["delta_max"]
    hard_floor   = constraints["hard_floor"]
    hard_ceiling = constraints["hard_ceiling"]

    bound_min = _safe_float(bound_min, hard_floor)
    bound_max = _safe_float(bound_max, hard_ceiling)
    center    = _safe_float(center, (hard_floor + hard_ceiling) / 2)

    if bound_min > bound_max:
        bound_min, bound_max = bound_max, bound_min

    width = bound_max - bound_min
    if width < delta_min:
        exp = (delta_min - width) / 2.0
        bound_min -= exp
        bound_max += exp
    if (bound_max - bound_min) > delta_max:
        half      = delta_max / 2.0
        bound_min = center - half
        bound_max = center + half
    if bound_min < hard_floor:
        deficit   = hard_floor - bound_min
        bound_min = hard_floor
        bound_max = min(bound_max + deficit, hard_ceiling)
    if bound_max > hard_ceiling:
        surplus   = bound_max - hard_ceiling
        bound_max = hard_ceiling
        bound_min = max(bound_min - surplus, hard_floor)

    return float(bound_min), float(bound_max)


def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
    names  = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)
    bad    = ~np.isfinite(values)
    if bad.any():
        values[bad] = FALLBACK_CONFIDENCE
    values = np.clip(values, 0.0, 1.0)
    total  = values.sum()
    if total < 1e-10:
        values = np.ones(len(values)) / len(values) * target_sum
    else:
        values = values / total * target_sum
    values = np.clip(values, floor, ceiling)
    return {n: float(v) for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# Master runner (public API — identical signature to v7.4)
# ---------------------------------------------------------------------------

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                    raw_sparse_mat=None):
    """
    Full Phase 0 pipeline (v7.5). Drop-in replacement for v7.4.

    Parameters
    ----------
    adata           : AnnData — full single-cell / metacell dataset
    n_genes         : int — number of genes in the var_names space
    cfg_diagnostics : dict — diagnostic config (see FUNGI config docs)
    cfg_input       : dict — perturbation_column, control_label, is_metacell, ...
    raw_sparse_mat  : scipy.sparse CSR or None — LightGBM parent graph
                      (n_genes × n_genes), edge weights = importance scores.
                      Required for S_max and C probes; other probes
                      are evidence-based and work without it.

    Returns
    -------
    utopian_bounds     : dict — {param: [lo, hi]} for all 6 parameters
    loss_weights       : dict — {param: float} normalised Sobol weights
    diagnostic_report  : dict — full provenance record
    """
    print("=" * 72)
    print("FUNGI v7.5 — Phase 0 Diagnostic Calibration")
    print("=" * 72)

    pert_col    = cfg_input["perturbation_column"]
    ctrl_label  = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool     = cfg_input.get("metacell_pooling_factor", None)

    n_boot    = cfg_diagnostics["n_bootstrap"]
    bc        = cfg_diagnostics["bound_constraints"]
    w_floor   = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    n_jobs    = cfg_diagnostics.get("n_jobs", 15)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    # Step 1: shared data construction
    (impact_array, pert_labels, sample_weights,
     deg_matrix, lfc_matrix, valid_cq, name_to_idx) = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method=cfg_diagnostics["de_method"],
        pval_threshold=cfg_diagnostics["de_pval_threshold"],
        lfc_threshold=cfg_diagnostics["de_lfc_threshold"],
        n_jobs=n_jobs,
        max_perts_for_de=max_perts,
        is_metacell=is_metacell,
        metacell_pooling_factor=mc_pool,
    )

    lam_eff = _compute_lam_eff(impact_array, len(impact_array))
    print(f"\n  λ_eff = {lam_eff:.2f}")

    # Step 2: run each probe
    print("\n  Diagnosing alpha (impact_powerlaw_diffusion_shift)...")
    alpha_lo, alpha_hi, alpha_conf = _probe_alpha(impact_array, lam_eff)

    print("  Diagnosing gini (specificity_idf_lfc_gini)...")
    gini_lo, gini_hi, gini_conf = _probe_gini(
        deg_matrix, lfc_matrix, valid_cq, name_to_idx, lam_eff)

    print("  Diagnosing S_max (topweight_sparse_hub_fraction)...")
    smax_lo, smax_hi, smax_conf = _probe_smax(
        raw_sparse_mat, impact_array, n_genes, lam_eff)

    print("  Diagnosing Q (lfc_multiresolution_modularity)...")
    Q_lo, Q_hi, Q_conf = _probe_Q(
        lfc_matrix, valid_cq, name_to_idx, deg_matrix, n_genes, lam_eff)

    print("  Diagnosing C (topweight_transitivity)...")
    C_lo, C_hi, C_conf = _probe_C(raw_sparse_mat, n_genes, lam_eff, lfc_matrix)

    print("  Diagnosing rho (lfc_l2_bipartite_assortativity)...")
    rho_lo, rho_hi, rho_conf = _probe_rho(lfc_matrix, valid_cq, name_to_idx)

    # Step 3: enforce bound constraints
    alpha_lo, alpha_hi = _enforce_bound_constraints(
        alpha_lo, alpha_hi, (alpha_lo + alpha_hi) / 2, bc["alpha"])
    gini_lo, gini_hi   = _enforce_bound_constraints(
        gini_lo,  gini_hi,  (gini_lo + gini_hi) / 2,  bc["gini"])
    smax_lo, smax_hi   = _enforce_bound_constraints(
        smax_lo,  smax_hi,  (smax_lo + smax_hi) / 2,  bc["S_max"])
    Q_lo, Q_hi         = _enforce_bound_constraints(
        Q_lo,     Q_hi,     (Q_lo + Q_hi) / 2,         bc["Q"])
    C_lo, C_hi         = _enforce_bound_constraints(
        C_lo,     C_hi,     (C_lo + C_hi) / 2,         bc["C"])
    rho_lo, rho_hi     = _enforce_bound_constraints(
        rho_lo,   rho_hi,   (rho_lo + rho_hi) / 2,     bc["rho"])

    utopian_bounds = {
        "alpha": [alpha_lo, alpha_hi],
        "gini":  [gini_lo,  gini_hi],
        "S_max": [smax_lo,  smax_hi],
        "Q":     [Q_lo,     Q_hi],
        "C":     [C_lo,     C_hi],
        "rho":   [rho_lo,   rho_hi],
    }

    # NaN guard
    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]

    # Step 4: weights
    raw_confidences = {
        "alpha": alpha_conf, "gini": gini_conf, "S_max": smax_conf,
        "Q":     Q_conf,     "C":    C_conf,     "rho":   rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # Step 5: print summary
    print(f"\n{'=' * 72}")
    print("Phase 0 Results")
    print(f"  λ_eff = {lam_eff:.2f}")
    print(f"{'=' * 72}")
    probes_used = {
        "alpha": "impact_powerlaw_diffusion_shift",
        "gini":  "specificity_idf_lfc_gini",
        "S_max": "topweight_sparse_hub_fraction",
        "Q":     "lfc_multiresolution_modularity",
        "C":     "topweight_transitivity",
        "rho":   "lfc_l2_bipartite_assortativity",
    }
    for param in utopian_bounds:
        lo, hi = utopian_bounds[param]
        w      = loss_weights[param]
        cf     = raw_confidences[param]
        print(f"  {param:>6s} [{lo:.4f}, {hi:.4f}]  "
              f"wt={w:.2f}  conf={cf:.2f}  ← {probes_used[param]}")
    print(f"{'=' * 72}\n")

    diagnostic_report = {
        "version":             "7.5",
        "lam_eff":             lam_eff,
        "impact_array_size":   len(impact_array),
        "impact_range":        ([float(impact_array.min()), float(impact_array.max())]
                                if len(impact_array) > 0 else [0, 0]),
        "deg_matrix_nnz":      int(deg_matrix.nnz),
        "lfc_matrix_shape":    list(lfc_matrix.shape),
        "is_metacell":         is_metacell,
        "probes_used":         probes_used,
        "raw_confidences":     {k: _safe_float(v) for k, v in raw_confidences.items()},
        "utopian_bounds":      utopian_bounds,
        "loss_weights":        loss_weights,
    }

    return utopian_bounds, loss_weights, diagnostic_report


# ---------------------------------------------------------------------------
# Shatter config builder (unchanged interface from v7.4)
# ---------------------------------------------------------------------------

def build_shatter_config(cfg_shatter, n_genes, utopian_bounds,
                         lambda_search_bounds):
    lambda_min  = cfg_shatter.get("lambda_min", 2.0)
    lambda_max  = cfg_shatter.get("lambda_max", 30.0)
    s_max_mult  = cfg_shatter.get("s_max_ceiling_multiplier", 2.0)
    s_max_cap   = cfg_shatter.get("s_max_ceiling_hard_cap", 0.30)
    clust_gamma = cfg_shatter.get("min_clustering_gamma", 1.5)

    s_max_ceiling = min(
        round(utopian_bounds["S_max"][1] * s_max_mult, 4),
        s_max_cap)

    if lambda_search_bounds is None or lambda_search_bounds[0] is None:
        lo_d = lambda_min / n_genes
        hi_d = lambda_max / n_genes
    else:
        lo_d = lambda_search_bounds[0]
        hi_d = lambda_search_bounds[1]

    lam_exp   = (lo_d + hi_d) / 2 * n_genes
    min_clust = round(clust_gamma * (lam_exp / n_genes), 6)

    return {
        "max_orphan_fraction": cfg_shatter.get("max_orphan_fraction", 0.15),
        "min_gwcc_fraction":   cfg_shatter.get("min_gwcc_fraction", 0.35),
        "max_hub_saturation":  s_max_ceiling,
        "min_edge_count":      int(lambda_min * n_genes),
        "max_edge_count":      int(lambda_max * n_genes),
        "min_clustering":      min_clust,
    }
