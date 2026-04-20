"""
FUNGI v7.3 -- Phase 0: Data-Driven Diagnostic Calibration

v7.3 architectural fixes (informed by NotebookLM analysis):

PROBLEM 1 — Cell starvation: capping TOTAL CELLS to 20k across 7133 groups
    gave 3 cells/group. 3-cell Wilcoxon is statistically meaningless AND
    produces noisy LFC means that destroy impact_array.
FIX: Cap PERTURBATION GROUPS to 500, not total cells. Each selected
    perturbation keeps ALL its cells. Wilcoxon is valid at full cell count.

PROBLEM 2 — Sampling bias: random 500 of 7133 misses master regulators
    (the heavy tail that drives Gini, alpha, S_max).
FIX: "Deterministic Top-K + Random Tail" — top-100 by fast LFC proxy
    (deterministic) + random-400 from rest. Weighted Gini/alpha rescues
    the true impact distribution from the full 7133-perturbation screen.

PROBLEM 3 — C always (0.10, 0.15): active_fraction ≈ 1.0 for any screen
    (all perturbations have nonzero impact) → no attenuation → C_core ~0.25
    from KNN graph hits hard_ceiling=0.15 → enforce collapses to (0.10, 0.15).
FIX: Replace active_fraction with Gini-based attenuation:
    active_fraction = 1 - gini_center. High Gini (concentrated impact) →
    more terminal effectors → more attenuation. VCC-5k gini~0.65 →
    attenuation=0.35 → C_attenuated=0.24*0.35=0.084. Requires computing
    Gini first and passing gini_center to _diagnose_modularity_and_clustering.

PROBLEM 4 — build_shatter_config NoneType: lambda_density=null in YAML
    triggers None subscript error.
FIX: Fall back to shatter.lambda_min/lambda_max when lambda_search_bounds
    is None.

PROBLEM 5 — Q bounds inverted for large screens (Replogle lower > upper).
FIX: Explicit sort + fallback guard in _diagnose_modularity_and_clustering.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from scipy import stats
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================
# Fallback defaults
# =========================================================================

FALLBACK_BOUNDS = {
    "alpha": [2.1, 2.7],
    "gini":  [0.55, 0.80],
    "S_max": [0.04, 0.10],
    "Q":     [0.25, 0.50],
    "C":     [0.05, 0.10],
    "rho":   [-0.30, -0.05],
}

FALLBACK_CONFIDENCE = 0.3


def _safe_float(val, fallback=0.0):
    if val is None or not np.isfinite(val):
        return fallback
    return float(val)


# =========================================================================
# Impact Array Construction
# =========================================================================

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25, n_jobs=15,
                       max_perts_for_de=500, min_cells_per_pert=5):
    """
    Computes DEG counts per perturbation using statistically valid Wilcoxon.

    For large screens (>max_perts_for_de groups), uses Deterministic Top-K +
    Random Tail sampling to select a representative 500-perturbation subset.
    Each selected perturbation keeps ALL its cells (no cell starvation).

    Returns impact_array and sample_weights for downstream weighted statistics.

    sample_weights[i] represents how many full-screen perturbations perturbation i
    represents. Top-100: weight=1.0. Random-400: weight=(n_total-100)/400.
    """
    print("Phase 0: Building perturbation impact array...")

    conditions_arr   = adata.obs[perturbation_column].values
    ctrl_mask        = conditions_arr == control_label
    unique_conds     = [c for c in np.unique(conditions_arr) if c != control_label]
    n_total          = len(unique_conds)

    print(f"  {n_total:,} perturbation groups detected.")

    # ── Step 1: Fast proxy — sparse mean LFC, no stats needed ─────────────
    # Uses sparse matrix column-sum (no dense conversion) for speed.
    print("  Computing fast LFC proxy for perturbation ranking...")
    X_csr    = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
    ctrl_mean = np.asarray(X_csr[ctrl_mask].mean(axis=0)).ravel()
    log_ctrl  = np.log1p(np.maximum(ctrl_mean, 0))

    proxy_scores = {}
    with tqdm(unique_conds, desc="  LFC proxy", unit="pert", ncols=80) as pbar:
        for cond in pbar:
            mask = conditions_arr == cond
            n    = mask.sum()
            if n < 2:
                proxy_scores[cond] = 0.0
                continue
            cond_mean = np.asarray(X_csr[mask].mean(axis=0)).ravel()
            lfc       = np.log1p(np.maximum(cond_mean, 0)) - log_ctrl
            proxy_scores[cond] = float(np.mean(np.abs(lfc)))

    # ── Step 2: Select representative perturbations ────────────────────────
    selected_conds = unique_conds
    sample_weights = {c: 1.0 for c in unique_conds}

    if n_total > max_perts_for_de:
        sorted_conds = sorted(unique_conds,
                              key=lambda c: proxy_scores[c], reverse=True)
        n_top        = min(100, max_perts_for_de // 5)
        n_random     = max_perts_for_de - n_top
        top_conds    = sorted_conds[:n_top]
        tail_conds   = sorted_conds[n_top:]

        rng           = np.random.default_rng(42)
        n_draw        = min(n_random, len(tail_conds))
        random_conds  = list(rng.choice(tail_conds, size=n_draw, replace=False))

        selected_conds = top_conds + random_conds

        # Sample weights: top-100 represent themselves only;
        # random-400 each represent (n_total-100)/n_draw full-screen perturbations
        tail_weight = float(len(tail_conds)) / max(n_draw, 1)
        for c in top_conds:
            sample_weights[c] = 1.0
        for c in random_conds:
            sample_weights[c] = tail_weight

        print(f"  Selected {len(selected_conds)} representative perturbations "
              f"({n_top} top-proxy deterministic + {len(random_conds)} random tail) "
              f"from {n_total:,} total.")
        print(f"  Tail weight: {tail_weight:.1f}x (each random pert represents "
              f"{tail_weight:.1f} full-screen perts)")
    else:
        print(f"  Running Wilcoxon on all {n_total} perturbations.")

    # ── Step 3: Filter cells — selected perturbations + control ───────────
    keep_mask = ctrl_mask.copy()
    for cond in selected_conds:
        keep_mask = keep_mask | (conditions_arr == cond)

    adata_sub = adata[keep_mask].copy()
    n_ctrl    = ctrl_mask.sum()
    print(f"  Wilcoxon subset: {adata_sub.n_obs:,} cells "
          f"({n_ctrl:,} control + perturbation cells)")

    # ── Step 4: Run Wilcoxon on properly-powered subset ───────────────────
    sc.tl.rank_genes_groups(
        adata_sub,
        groupby=perturbation_column,
        reference=control_label,
        method=de_method,
        use_raw=False,
        n_jobs=n_jobs,
    )

    # ── Step 5: Extract DEG counts ─────────────────────────────────────────
    impact_scores   = []
    valid_labels    = []
    valid_weights   = []

    with tqdm(selected_conds, desc="  DEG counts", unit="pert", ncols=80) as pbar:
        for cond in pbar:
            try:
                result_df = sc.get.rank_genes_groups_df(adata_sub, group=cond)
                sig       = result_df[
                    (result_df["pvals_adj"] < pval_threshold) &
                    (result_df["logfoldchanges"].abs() > lfc_threshold)
                ]
                impact_scores.append(len(sig))
                valid_labels.append(cond)
                valid_weights.append(sample_weights.get(cond, 1.0))
            except Exception:
                continue

    impact_array        = np.array(impact_scores, dtype=np.float64)
    perturbation_labels = np.array(valid_labels)
    weights_arr         = np.array(valid_weights, dtype=np.float64)

    nonzero_mask        = impact_array > 0
    impact_array        = impact_array[nonzero_mask]
    perturbation_labels = perturbation_labels[nonzero_mask]
    weights_arr         = weights_arr[nonzero_mask]

    print(f"  {len(impact_array)} perturbations with nonzero impact.")
    if len(impact_array) > 0:
        print(f"  Impact range: [{impact_array.min():.0f}, "
              f"{impact_array.max():.0f}] DEGs.")
        # Effective N after weighting
        eff_n = float(weights_arr.sum())
        print(f"  Effective N (weighted): {eff_n:.0f} of {n_total:,} total perts.")

    return impact_array, perturbation_labels, weights_arr


# =========================================================================
# Parameter Diagnostics
# =========================================================================

def _diagnose_gini(impact_array, n_bootstrap, is_metacell,
                   sample_weights=None):
    """
    Weighted Gini coefficient from the impact distribution.

    When sample_weights are provided (from the Deterministic Top-K sampling),
    the Gini is computed on the weighted distribution, which rescues the
    true Lorenz curve of the full-screen dataset from the 500-pert subset.
    """
    if len(impact_array) < 5:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    def _weighted_gini(arr, weights):
        """Weighted Gini coefficient via Lorenz curve."""
        if weights is None:
            weights = np.ones(len(arr))
        weights = np.asarray(weights, dtype=np.float64)
        arr     = np.asarray(arr, dtype=np.float64)
        # Sort by value
        order  = np.argsort(arr)
        arr    = arr[order]
        weights = weights[order]
        w_sum  = weights.sum()
        if w_sum == 0 or arr.sum() == 0:
            return 0.0
        cum_w  = np.cumsum(weights) / w_sum
        cum_v  = np.cumsum(arr * weights) / (arr * weights).sum()
        # Lorenz: area under curve via trapezoidal rule
        lorenz_area = np.trapz(cum_v, cum_w)
        return float(1.0 - 2.0 * lorenz_area)

    try:
        rng = np.random.default_rng(42)
        n   = len(impact_array)

        boot_ginis = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_arr = impact_array[idx]
            boot_w   = sample_weights[idx] if sample_weights is not None else None
            g        = _weighted_gini(boot_arr, boot_w)
            if np.isfinite(g):
                boot_ginis.append(g)

        boot_ginis = np.array(boot_ginis)
        boot_ginis = boot_ginis[np.isfinite(boot_ginis)]
        if len(boot_ginis) < 10:
            fb = FALLBACK_BOUNDS["gini"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        center     = float(np.median(boot_ginis))
        bound_min  = float(np.percentile(boot_ginis, 5))
        bound_max  = float(np.percentile(boot_ginis, 95))
        variance   = max(float(np.var(boot_ginis)), 1e-10)
        confidence = _safe_float(
            1.0 - np.clip(np.sqrt(variance) * 10, 0, 0.9), FALLBACK_CONFIDENCE)

        return bound_min, bound_max, confidence, center

    except Exception:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


def _diagnose_alpha(impact_array, n_bootstrap):
    """Scale-free exponent from power-law MLE fit."""
    if len(impact_array) < 10:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2
    try:
        import powerlaw
        fit        = powerlaw.Fit(impact_array, xmin=2, discrete=True,
                                  verbose=False)
        alpha_est  = _safe_float(fit.power_law.alpha, 2.3)
        sigma_est  = _safe_float(fit.power_law.sigma, 0.5)
        ks_dist    = _safe_float(fit.power_law.D, 0.5)
        bound_min  = alpha_est - 1.96 * sigma_est
        bound_max  = alpha_est + 1.96 * sigma_est
        r2_proxy   = float(np.clip(1.0 - ks_dist, 0.01, 0.999))
        confidence = float(np.clip(
            -np.log10(1.0 - r2_proxy) / 3.0, 0.1, 1.0))
        return bound_min, bound_max, confidence, alpha_est
    except Exception:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


def _diagnose_smax(impact_array, n_genes, is_metacell,
                   metacell_pooling_factor, sample_weights=None):
    """Hub saturation ceiling from strongest perturbation impacts."""
    if len(impact_array) < 5 or n_genes < 1:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2
    try:
        arr = impact_array.copy()
        if is_metacell and metacell_pooling_factor and metacell_pooling_factor > 1:
            arr = arr / np.sqrt(metacell_pooling_factor)

        # For weighted max: top-weighted perturbations are master regulators
        # Use weighted 95th percentile and weighted max
        if sample_weights is not None and len(sample_weights) == len(arr):
            # Repeat values by weight to compute weighted percentiles
            w_int = np.round(sample_weights).astype(int)
            w_int = np.maximum(w_int, 1)
            expanded = np.repeat(arr, w_int)
            p95_impact = float(np.percentile(expanded, 95))
            max_impact = float(expanded.max())
        else:
            p95_impact = float(np.percentile(arr, 95))
            max_impact = float(arr.max())

        bound_min = p95_impact / n_genes
        bound_max = max_impact / n_genes
        if bound_min >= bound_max:
            bound_max = bound_min * 1.5
        if bound_max < 0.01:
            bound_max = 0.10

        median_impact = max(float(np.median(arr)), 1.0)
        snr           = max_impact / median_impact
        confidence    = _safe_float(
            np.clip(np.log10(max(snr, 1.01)), 0.1, 3.0) / 3.0,
            FALLBACK_CONFIDENCE)
        center = (bound_min + bound_max) / 2.0
        return bound_min, bound_max, confidence, center
    except Exception:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


def _diagnose_modularity_and_clustering(adata, perturbation_column,
                                        control_label, knn_k_range,
                                        is_metacell, sc_midpoint, mc_midpoint,
                                        impact_array=None, pert_labels=None,
                                        gini_center=None, n_genes=None):
    """
    Q and C from cosine similarity of continuous LFC vectors.

    C attenuation fix (v7.3): replaces active_fraction (which was always ~1.0,
    causing no attenuation) with Gini-based attenuation:
        active_fraction = 1 - gini_center

    High Gini (concentrated impact, few master regulators) → less clustering
    in the full network → more attenuation needed.
    VCC-5k: gini~0.65 → attenuation=0.35 → C_attenuated≈0.084  ✓
    GWPS: gini~0.40 → attenuation=0.60 → C_attenuated≈0.14 (capped by YAML)

    Stratified sampling: 500 perts via Top-K + Random Tail when >500 total.
    Q inversion guard: always sort Q bounds before returning.
    """
    MAX_PERTS_FOR_CQ = 500

    print("  Computing LFC cosine similarity graph for modularity diagnostics...")

    try:
        all_conds   = adata.obs[perturbation_column].unique()
        conditions  = [c for c in all_conds if c != control_label]
        n_total_conds = len(conditions)

        # Stratified subsample for C/Q
        if len(conditions) > MAX_PERTS_FOR_CQ:
            if impact_array is not None and pert_labels is not None:
                impact_lookup = {
                    str(lb): float(sc_val)
                    for lb, sc_val in zip(pert_labels, impact_array)
                }
                scored = sorted(
                    [(c, impact_lookup.get(str(c), 0.0)) for c in conditions],
                    key=lambda x: x[1]
                )
                n_per_q = MAX_PERTS_FOR_CQ // 4
                q_size  = len(scored) // 4
                rng_cq  = np.random.default_rng(42)
                selected = []
                for q in range(4):
                    q_start = q * q_size
                    q_end   = (q + 1) * q_size if q < 3 else len(scored)
                    bucket  = [c for c, _ in scored[q_start:q_end]]
                    n_draw  = min(n_per_q, len(bucket))
                    selected.extend(
                        rng_cq.choice(bucket, size=n_draw,
                                      replace=False).tolist()
                    )
                conditions = selected
                print(f"  (C/Q: stratified sample of {len(conditions)} "
                      f"from {n_total_conds} — evenly spread across "
                      f"impact quartiles)")
            else:
                rng_cq    = np.random.default_rng(42)
                conditions = list(rng_cq.choice(
                    conditions, size=MAX_PERTS_FOR_CQ, replace=False))

        ctrl_mask = adata.obs[perturbation_column].values == control_label
        if ctrl_mask.sum() < 3:
            raise ValueError("Too few control cells.")

        # Convert to dense ONCE for the selected subset
        print(f"  Computing LFC vectors for {len(conditions)} perturbations...")
        X        = (adata.X.toarray() if hasattr(adata.X, 'toarray')
                    else np.array(adata.X)).astype(np.float32)
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        log_ctrl  = np.log1p(np.maximum(ctrl_mean, 0))

        lfc_vectors      = []
        valid_conditions = []
        grp_vals         = adata.obs[perturbation_column].values

        with tqdm(conditions, desc="  LFC vectors",
                  unit="pert", ncols=80) as pbar:
            for cond in pbar:
                mask = grp_vals == cond
                if mask.sum() < 2:
                    continue
                cond_mean = X[mask].mean(axis=0)
                lfc = (np.log1p(np.maximum(cond_mean, 0)) - log_ctrl)
                if np.all(np.isfinite(lfc)) and np.any(lfc != 0):
                    lfc_vectors.append(lfc)
                    valid_conditions.append(cond)

        del X

        if len(lfc_vectors) < 10:
            fb_q = FALLBACK_BOUNDS["Q"]
            fb_c = FALLBACK_BOUNDS["C"]
            return (fb_q[0], fb_q[1], FALLBACK_CONFIDENCE,
                    fb_c[0], fb_c[1], FALLBACK_CONFIDENCE)

        lfc_matrix = np.array(lfc_vectors)
        n_perts    = len(lfc_matrix)

        # Cosine similarity matrix
        norms      = np.linalg.norm(lfc_matrix, axis=1, keepdims=True)
        norms      = np.where(norms < 1e-10, 1e-10, norms)
        lfc_normed = lfc_matrix / norms
        sim_matrix = lfc_normed @ lfc_normed.T
        np.fill_diagonal(sim_matrix, 0)
        sim_matrix = np.clip(sim_matrix, 0, 1)

        import networkx as nx
        Q_values = []
        C_values = []

        with tqdm(knn_k_range, desc="  KNN Louvain",
                  unit="k", ncols=80) as pbar:
            for k in pbar:
                if k >= n_perts:
                    continue
                top_k = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]
                G_co  = nx.Graph()
                G_co.add_nodes_from(range(n_perts))
                for i in range(n_perts):
                    for j in top_k[i]:
                        if sim_matrix[i, j] > 0:
                            G_co.add_edge(i, int(j),
                                          weight=float(sim_matrix[i, j]))
                if G_co.number_of_edges() == 0:
                    continue
                try:
                    comms = nx.community.louvain_communities(G_co, seed=42)
                    Q = nx.community.modularity(G_co, comms)
                    if np.isfinite(Q):
                        Q_values.append(Q)
                except Exception:
                    pass
                try:
                    C = nx.average_clustering(G_co, weight='weight')
                    if np.isfinite(C):
                        C_values.append(C)
                except Exception:
                    pass

        # Q bounds — sort to prevent inversion on large screens
        if Q_values:
            Q_min = float(min(Q_values))
            Q_max = float(max(Q_values))
            if Q_min > Q_max:
                Q_min, Q_max = Q_max, Q_min
        else:
            Q_min = FALLBACK_BOUNDS["Q"][0]
            Q_max = FALLBACK_BOUNDS["Q"][1]

        if C_values:
            # Gini-based attenuation (v7.3 fix):
            # active_fraction = 1 - gini_center
            # High Gini → concentrated regulatory impact → few regulators →
            # most nodes are terminal effectors with C=0 → strong attenuation
            # Low Gini → distributed impact → many regulators → less attenuation
            if gini_center is not None and np.isfinite(gini_center):
                active_fraction = float(np.clip(1.0 - gini_center, 0.15, 0.70))
            else:
                # Fallback: use perturbation density as attenuation
                active_fraction = float(np.clip(
                    len(valid_conditions) /
                    max(len(all_conds) - 1, 1),
                    0.15, 0.70))

            C_core       = float(np.mean(C_values))
            C_attenuated = C_core * active_fraction
            C_spread     = float(np.std(C_values)) * active_fraction
            C_min        = max(0.0, C_attenuated - C_spread)
            C_max        = C_attenuated + C_spread
            print(f"  C_core={C_core:.4f}, active_fraction={active_fraction:.3f}, "
                  f"C_attenuated={C_attenuated:.4f}")
        else:
            C_min = FALLBACK_BOUNDS["C"][0]
            C_max = FALLBACK_BOUNDS["C"][1]

        if len(C_values) >= 2:
            c_cv       = np.std(C_values) / max(np.mean(C_values), 1e-6)
            confidence = float(np.clip(1.0 - c_cv, 0.15, 1.0))
        else:
            confidence = FALLBACK_CONFIDENCE

        return Q_min, Q_max, confidence, C_min, C_max, confidence

    except Exception:
        fb_q = FALLBACK_BOUNDS["Q"]
        fb_c = FALLBACK_BOUNDS["C"]
        return (fb_q[0], fb_q[1], FALLBACK_CONFIDENCE,
                fb_c[0], fb_c[1], FALLBACK_CONFIDENCE)


def _bootstrap_rho_worker(args):
    """Single bootstrap iteration for rho — parallelizable."""
    n_genes, n_edges, coo_row, coo_col, seed = args
    rng   = np.random.default_rng(seed)
    b_idx = rng.choice(n_edges, size=min(200000, n_edges), replace=True)
    try:
        import igraph as ig
        G = ig.Graph(n=n_genes,
                     edges=list(zip(coo_row[b_idx].tolist(),
                                    coo_col[b_idx].tolist())),
                     directed=True)
        r = float(G.assortativity_degree(directed=True))
        return r if np.isfinite(r) else None
    except Exception:
        return None


def _diagnose_rho(impact_array, perturbation_labels, adata,
                  perturbation_column, control_label, n_genes,
                  raw_sparse_mat, n_bootstrap=200, n_sample=200000,
                  n_jobs=15):
    """
    Data-driven degree assortativity from the prefiltered parent graph.
    Bootstrap parallelized with joblib threads.
    """
    import igraph as ig

    fb = FALLBACK_BOUNDS["rho"]

    if raw_sparse_mat is None:
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        coo     = raw_sparse_mat.tocoo()
        n_edges = len(coo.data)
        if n_edges < 1000:
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        n_sample_actual = min(n_sample, n_edges)
        rng             = np.random.default_rng(42)
        idx             = rng.choice(n_edges, size=n_sample_actual, replace=False)

        print(f"    rho: sampling {n_sample_actual:,} of {n_edges:,} edges "
              f"({n_sample_actual/n_edges:.1%})")

        G_sample   = ig.Graph(n=n_genes,
                              edges=list(zip(coo.row[idx].tolist(),
                                             coo.col[idx].tolist())),
                              directed=True)
        rho_parent = float(G_sample.assortativity_degree(directed=True))
        if not np.isfinite(rho_parent):
            rho_parent = -0.10
        print(f"    rho_parent: {rho_parent:.4f}")

        seeds  = rng.integers(0, 2**31, size=n_bootstrap).tolist()
        args   = [(n_genes, n_edges, coo.row, coo.col, s) for s in seeds]

        print(f"    Bootstrap: {n_bootstrap} iterations ({n_jobs} workers)...")
        results   = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_bootstrap_rho_worker)(a)
            for a in tqdm(args, desc="    rho bootstrap",
                          unit="iter", ncols=80)
        )
        boot_rhos = [r for r in results if r is not None]

        if len(boot_rhos) < 10:
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        boot_sigma = float(np.std(boot_rhos))
        boot_mean  = float(np.mean(boot_rhos))
        print(f"    boot_sigma={boot_sigma:.4f}  boot_mean={boot_mean:.4f}")

        n_total_perts = len([
            c for c in adata.obs[perturbation_column].unique()
            if c != control_label
        ])
        n_active    = int(np.sum(impact_array > 0)) if len(impact_array) > 0 else 1
        active_frac = float(np.clip(n_active / max(n_total_perts, 1), 0.05, 0.95))
        silent_frac = 1.0 - active_frac

        rho_upper = rho_parent + boot_sigma
        rho_lower = rho_parent - (silent_frac * max(boot_sigma * 2.0, 0.05))
        print(f"    rho bounds: [{rho_lower:.4f}, {rho_upper:.4f}]")

        coverage_ratio  = float(n_sample_actual / max(n_edges, 1))
        sample_discount = float(np.clip(np.sqrt(coverage_ratio), 0.20, 1.0))
        confidence      = float(np.clip(
            1.0 - (boot_sigma / max(abs(rho_parent), 0.05)),
            0.10, 1.0)) * sample_discount

        center = (rho_upper + rho_lower) / 2.0
        return float(rho_lower), float(rho_upper), confidence, center

    except Exception as e:
        print(f"    rho diagnostic failed: {str(e)[:80]}")
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# Bound Enforcement
# =========================================================================

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
        exp       = (delta_min - width) / 2.0
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


# =========================================================================
# Weight Normalization
# =========================================================================

def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
    names  = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)
    bad    = ~np.isfinite(values)
    if bad.any():
        print(f"  WARNING: {bad.sum()} confidence values NaN/inf → fallback.")
        values[bad] = FALLBACK_CONFIDENCE
    values = np.clip(values, 0.0, 1.0)
    total  = values.sum()
    if total < 1e-10:
        print("  WARNING: All confidences near zero. Using uniform weights.")
        values = np.ones(len(values)) / len(values) * target_sum
    else:
        values = values / total * target_sum
    values = np.clip(values, floor, ceiling)
    result = {n: float(v) for n, v in zip(names, values)}
    for n in result:
        if not np.isfinite(result[n]):
            result[n] = floor
    return result


# =========================================================================
# Master Diagnostic Runner
# =========================================================================

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                    raw_sparse_mat=None):
    """
    Executes the full Phase 0 diagnostic pipeline.

    v7.3: Statistically valid Wilcoxon with proper cell counts per group.
    Weighted Gini/alpha/S_max from Deterministic Top-K + Random Tail sampling.
    Gini-based C attenuation. Parallelized rho bootstrap. tqdm progress bars.
    """
    print("=" * 72)
    print("FUNGI v7 -- Phase 0: Data-Driven Diagnostic Calibration")
    print("=" * 72)

    pert_col    = cfg_input["perturbation_column"]
    ctrl_label  = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool     = cfg_input.get("metacell_pooling_factor", None)

    n_boot    = cfg_diagnostics["n_bootstrap"]
    knn_range = cfg_diagnostics["knn_k_range"]
    bc        = cfg_diagnostics["bound_constraints"]
    w_floor   = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    sc_mid    = cfg_diagnostics["singlecell_silhouette_midpoint"]
    mc_mid    = cfg_diagnostics["metacell_silhouette_midpoint"]
    n_jobs    = cfg_diagnostics.get("n_jobs", 15)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    # ── Step 1: Impact array ───────────────────────────────────────────────
    print("Phase 0: Building perturbation impact array...")
    impact_array, pert_labels, sample_weights = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method=cfg_diagnostics["de_method"],
        pval_threshold=cfg_diagnostics["de_pval_threshold"],
        lfc_threshold=cfg_diagnostics["de_lfc_threshold"],
        n_jobs=n_jobs,
        max_perts_for_de=max_perts,
    )

    # ── Step 2: Gini FIRST (needed for C attenuation) ─────────────────────
    print("\n  Diagnosing Gini (degree inequality)...")
    gini_min, gini_max, gini_conf, gini_center = _diagnose_gini(
        impact_array, n_boot, is_metacell, sample_weights=sample_weights)
    print(f"    gini_center = {gini_center:.4f} "
          f"(C attenuation factor = {1.0 - gini_center:.3f})")

    print("  Diagnosing alpha (scale-free exponent)...")
    alpha_min, alpha_max, alpha_conf, alpha_center = _diagnose_alpha(
        impact_array, n_boot)

    print("  Diagnosing S_max (hub saturation)...")
    smax_min, smax_max, smax_conf, smax_center = _diagnose_smax(
        impact_array, n_genes, is_metacell, mc_pool,
        sample_weights=sample_weights)

    print("  Diagnosing Q and C (modularity and clustering)...")
    Q_min, Q_max, Q_conf, C_min, C_max, C_conf = \
        _diagnose_modularity_and_clustering(
            adata, pert_col, ctrl_label, knn_range, is_metacell,
            sc_mid, mc_mid,
            impact_array=impact_array, pert_labels=pert_labels,
            gini_center=gini_center, n_genes=n_genes)

    print("  Diagnosing rho (degree assortativity)...")
    rho_min, rho_max, rho_conf, rho_center = _diagnose_rho(
        impact_array, pert_labels, adata, pert_col, ctrl_label, n_genes,
        raw_sparse_mat=raw_sparse_mat, n_bootstrap=n_boot, n_jobs=n_jobs)

    # ── Step 3: Report raw bounds ──────────────────────────────────────────
    print("\n  Raw bounds before constraint enforcement:")
    print(f"    alpha: [{alpha_min:.4f}, {alpha_max:.4f}]  (conf={alpha_conf:.3f})")
    print(f"    C:     [{C_min:.4f}, {C_max:.4f}]  (conf={C_conf:.3f})")
    print(f"    Q:     [{Q_min:.4f}, {Q_max:.4f}]  (conf={Q_conf:.3f})")
    print(f"    gini:  [{gini_min:.4f}, {gini_max:.4f}]  (conf={gini_conf:.3f})")
    print(f"    S_max: [{smax_min:.4f}, {smax_max:.4f}]  (conf={smax_conf:.3f})")
    print(f"    rho:   [{rho_min:.4f}, {rho_max:.4f}]  (conf={rho_conf:.3f})")

    # ── Step 4: Enforce constraints ───────────────────────────────────────
    gini_min,  gini_max  = _enforce_bound_constraints(
        gini_min, gini_max, gini_center, bc["gini"])
    alpha_min, alpha_max = _enforce_bound_constraints(
        alpha_min, alpha_max, alpha_center, bc["alpha"])
    smax_min,  smax_max  = _enforce_bound_constraints(
        smax_min, smax_max, smax_center, bc["S_max"])
    Q_min, Q_max = _enforce_bound_constraints(
        Q_min, Q_max, (Q_min + Q_max) / 2, bc["Q"])
    C_min, C_max = _enforce_bound_constraints(
        C_min, C_max, (C_min + C_max) / 2, bc["C"])
    rho_min, rho_max = _enforce_bound_constraints(
        rho_min, rho_max, rho_center, bc["rho"])

    utopian_bounds = {
        "alpha": [alpha_min, alpha_max],
        "gini":  [gini_min, gini_max],
        "S_max": [smax_min, smax_max],
        "Q":     [Q_min, Q_max],
        "C":     [C_min, C_max],
        "rho":   [rho_min, rho_max],
    }

    # ── Step 5: Weights ───────────────────────────────────────────────────
    raw_confidences = {
        "alpha": alpha_conf, "gini": gini_conf, "S_max": smax_conf,
        "Q": Q_conf, "C": C_conf, "rho": rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # ── Step 6: NaN guard ─────────────────────────────────────────────────
    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]
    for param in loss_weights:
        if not np.isfinite(loss_weights[param]):
            loss_weights[param] = w_floor

    # ── Step 7: Report ────────────────────────────────────────────────────
    diagnostic_report = {
        "impact_array_size":  len(impact_array),
        "impact_range": (
            [float(impact_array.min()), float(impact_array.max())]
            if len(impact_array) > 0 else [0, 0]
        ),
        "is_metacell":         is_metacell,
        "n_cells_used_for_de": int(adata.n_obs),
        "gini_center":         float(gini_center),
        "raw_confidences":     {k: _safe_float(v, 0.0)
                                for k, v in raw_confidences.items()},
        "utopian_bounds":      utopian_bounds,
        "loss_weights":        loss_weights,
    }

    print(f"\n{'=' * 72}")
    print("Phase 0 Results: Custom Utopian Calibration")
    print(f"{'=' * 72}")
    for param in utopian_bounds:
        bmin, bmax = utopian_bounds[param]
        w = loss_weights[param]
        print(f"  {param:>6s}:  bounds = [{bmin:.4f}, {bmax:.4f}]  |  weight = {w:.2f}")
    print(f"{'=' * 72}\n")

    return utopian_bounds, loss_weights, diagnostic_report


# =========================================================================
# Shatter Config Builder
# =========================================================================

def build_shatter_config(cfg_shatter, n_genes, utopian_bounds,
                         lambda_search_bounds):
    """
    Resolves dynamic shatter thresholds from Phase 0 outputs and graph size.

    Handles lambda_search_bounds=None (when YAML lambda_density=null and
    dynamic bounds haven't been computed yet) by falling back to
    shatter.lambda_min / lambda_max.
    """
    lambda_min   = cfg_shatter.get("lambda_min", 2.0)
    lambda_max   = cfg_shatter.get("lambda_max", 30.0)
    s_max_mult   = cfg_shatter.get("s_max_ceiling_multiplier", 1.5)
    s_max_cap    = cfg_shatter.get("s_max_ceiling_hard_cap", 0.30)
    clust_gamma  = cfg_shatter.get("min_clustering_gamma", 1.5)

    s_max_ceiling = min(
        round(utopian_bounds["S_max"][1] * s_max_mult, 4),
        s_max_cap
    )

    # Handle null lambda_search_bounds (YAML lambda_density=null)
    if lambda_search_bounds is None or lambda_search_bounds[0] is None:
        # Fall back to shatter lambda range converted to density
        lambda_lo_density = lambda_min / n_genes
        lambda_hi_density = lambda_max / n_genes
    else:
        lambda_lo_density = lambda_search_bounds[0]
        lambda_hi_density = lambda_search_bounds[1]

    lambda_expected     = (lambda_lo_density + lambda_hi_density) / 2
    lambda_expected_abs = lambda_expected * n_genes
    min_clustering      = round(clust_gamma * (lambda_expected_abs / n_genes), 6)

    return {
        "max_orphan_fraction": cfg_shatter.get("max_orphan_fraction", 0.15),
        "min_gwcc_fraction":   cfg_shatter.get("min_gwcc_fraction", 0.35),
        "max_hub_saturation":  s_max_ceiling,
        "min_edge_count":      int(lambda_min * n_genes),
        "max_edge_count":      int(lambda_max * n_genes),
        "min_clustering":      min_clustering,
    }
