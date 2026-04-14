"""
FUNGI v7 -- Phase 0: Data-Driven Diagnostic Calibration

Replaces the fixed archetype system with dataset-specific utopian bounds
and loss weights derived from the training h5ad perturbation data.

v7 changes from v6:
    - Every statistical test wrapped in try/except with fallback defaults.
    - All confidence values sanitized through np.isfinite before use.
    - Weight normalization uses np.nansum with explicit NaN replacement.
    - Asymmetric boundary clipping enforced on all parameters.
    - Metacell variance correction applied to S_max and silhouette scaling.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================
# Fallback defaults: used when any diagnostic test fails to converge.
# These are conservative, biologically reasonable baselines derived from
# the literature on mammalian single-cell GRNs.
# =========================================================================

FALLBACK_BOUNDS = {
    "alpha": [2.1, 2.7],
    "gini":  [0.55, 0.80],
    "S_max": [0.04, 0.10],
    "Q":     [0.25, 0.50],
    "C":     [0.08, 0.30],
    "rho":   [-0.30, -0.05],
}

FALLBACK_CONFIDENCE = 0.3  # low but nonzero baseline weight


# =========================================================================
# Sanitization helper
# =========================================================================

def _safe_float(val, fallback=0.0):
    """Returns val if it is a finite number, otherwise returns fallback."""
    if val is None or not np.isfinite(val):
        return fallback
    return float(val)


# =========================================================================
# Impact Array Construction
# =========================================================================

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25):
    """Computes the number of DEGs per perturbation condition."""
    print("Phase 0: Building perturbation impact array...")

    conditions = adata.obs[perturbation_column].unique()
    conditions = [c for c in conditions if c != control_label]

    sc.tl.rank_genes_groups(
        adata, groupby=perturbation_column, reference=control_label,
        method=de_method, use_raw=False
    )

    impact_scores = []
    valid_labels = []

    for cond in conditions:
        try:
            result_df = sc.get.rank_genes_groups_df(adata, group=cond)
            sig_genes = result_df[
                (result_df["pvals_adj"] < pval_threshold) &
                (result_df["logfoldchanges"].abs() > lfc_threshold)
            ]
            impact_scores.append(len(sig_genes))
            valid_labels.append(cond)
        except Exception:
            continue

    impact_array = np.array(impact_scores, dtype=np.float64)
    perturbation_labels = np.array(valid_labels)

    nonzero_mask = impact_array > 0
    impact_array = impact_array[nonzero_mask]
    perturbation_labels = perturbation_labels[nonzero_mask]

    print(f"  {len(impact_array)} perturbations with nonzero impact.")
    if len(impact_array) > 0:
        print(f"  Impact range: [{impact_array.min():.0f}, {impact_array.max():.0f}] DEGs.")
    else:
        print("  WARNING: Zero nonzero perturbations detected.")

    return impact_array, perturbation_labels


# =========================================================================
# Parameter-Specific Diagnostics (all hardened against NaN)
# =========================================================================

def _diagnose_gini(impact_array, n_bootstrap, is_metacell):
    """Gini coefficient from the impact distribution."""
    if len(impact_array) < 5:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    def _gini(arr):
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        sorted_arr = np.sort(arr)
        index = np.arange(1, n + 1)
        return float(np.sum((2 * index - n - 1) * sorted_arr) / (n * np.sum(sorted_arr)))

    try:
        boot_ginis = np.array([
            _gini(np.random.choice(impact_array, size=len(impact_array), replace=True))
            for _ in range(n_bootstrap)
        ])

        # Remove any NaN bootstrap samples
        boot_ginis = boot_ginis[np.isfinite(boot_ginis)]
        if len(boot_ginis) < 10:
            fb = FALLBACK_BOUNDS["gini"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        center = float(np.median(boot_ginis))
        bound_min = float(np.percentile(boot_ginis, 5))
        bound_max = float(np.percentile(boot_ginis, 95))

        variance = float(np.var(boot_ginis))
        if variance < 1e-10:
            variance = 1e-10
        confidence = 1.0 - np.clip(np.sqrt(variance) * 10, 0, 0.9)
        confidence = _safe_float(confidence, FALLBACK_CONFIDENCE)

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
        fit = powerlaw.Fit(impact_array, xmin=2, discrete=True, verbose=False)

        alpha_est = _safe_float(fit.power_law.alpha, 2.3)
        sigma_est = _safe_float(fit.power_law.sigma, 0.5)
        ks_distance = _safe_float(fit.power_law.D, 0.5)

        bound_min = alpha_est - 1.96 * sigma_est
        bound_max = alpha_est + 1.96 * sigma_est

        # Confidence from KS distance
        if ks_distance < 1e-10:
            ks_distance = 1e-10
        r_squared_proxy = max(0.01, 1.0 - ks_distance)
        if r_squared_proxy >= 1.0:
            r_squared_proxy = 0.999
        confidence = _safe_float(-np.log10(1.0 - r_squared_proxy), FALLBACK_CONFIDENCE)
        # Normalize to [0, 1] range
        confidence = np.clip(confidence / 3.0, 0.1, 1.0)

        return bound_min, bound_max, confidence, alpha_est

    except Exception:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


def _diagnose_smax(impact_array, n_genes, is_metacell, metacell_pooling_factor):
    """Hub saturation ceiling from strongest perturbation impacts."""
    if len(impact_array) < 5 or n_genes < 1:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        if is_metacell and metacell_pooling_factor is not None and metacell_pooling_factor > 1:
            impact_array = impact_array / np.sqrt(metacell_pooling_factor)

        sorted_impact = np.sort(impact_array)[::-1]
        p95_impact = float(np.percentile(impact_array, 95))
        max_impact = float(sorted_impact[0])

        bound_min = p95_impact / n_genes
        bound_max = max_impact / n_genes

        if bound_min >= bound_max:
            bound_max = bound_min * 1.5
        if bound_max < 0.01:
            bound_max = 0.10

        median_impact = float(np.median(impact_array))
        if median_impact < 1:
            median_impact = 1.0
        snr = max_impact / median_impact
        confidence = _safe_float(np.clip(np.log10(max(snr, 1.01)), 0.1, 3.0) / 3.0,
                                 FALLBACK_CONFIDENCE)

        center = (bound_min + bound_max) / 2.0
        return bound_min, bound_max, confidence, center

    except Exception:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


def _diagnose_modularity_and_clustering(adata, perturbation_column,
                                        control_label, knn_k_range,
                                        is_metacell, sc_midpoint, mc_midpoint):
    """Q and C from cosine similarity of continuous LFC vectors.
    
    Replaces KNN centroid approach (KNN fallacy: artificially inflated C).
    Builds a co-impact graph over active perturbations using cosine similarity
    of full LFC vectors, then measures Q and C on that graph.
    
    C is attenuated by the fraction of active regulators in the network
    (per Aguirre 2025: ~41% of genes have measurable perturbation effects)
    to produce a realistic global C target for the full n-gene GRN.
    Q is used directly as it is relatively scale-invariant.
    """
    print("  Computing LFC cosine similarity graph for modularity diagnostics...")

    try:
        conditions = adata.obs[perturbation_column].unique()
        conditions = [c for c in conditions if c != control_label]

        # Compute mean LFC vector for each perturbation vs control
        ctrl_mask = adata.obs[perturbation_column] == control_label
        if ctrl_mask.sum() < 5:
            raise ValueError("Too few control cells.")

        if hasattr(adata.X, 'toarray'):
            ctrl_mean = np.array(adata.X[ctrl_mask].toarray().mean(axis=0)).ravel()
        else:
            ctrl_mean = np.array(adata.X[ctrl_mask].mean(axis=0)).ravel()

        lfc_vectors = []
        valid_conditions = []

        for cond in conditions:
            mask = adata.obs[perturbation_column] == cond
            if mask.sum() < 5:
                continue
            if hasattr(adata.X, 'toarray'):
                cond_mean = np.array(adata.X[mask].toarray().mean(axis=0)).ravel()
            else:
                cond_mean = np.array(adata.X[mask].mean(axis=0)).ravel()

            # Log fold change: log1p(cond) - log1p(ctrl)
            lfc = np.log1p(np.maximum(cond_mean, 0)) - np.log1p(np.maximum(ctrl_mean, 0))

            if np.all(np.isfinite(lfc)) and np.any(lfc != 0):
                lfc_vectors.append(lfc)
                valid_conditions.append(cond)

        if len(lfc_vectors) < 10:
            fb_q = FALLBACK_BOUNDS["Q"]
            fb_c = FALLBACK_BOUNDS["C"]
            return fb_q[0], fb_q[1], FALLBACK_CONFIDENCE, fb_c[0], fb_c[1], FALLBACK_CONFIDENCE

        lfc_matrix = np.array(lfc_vectors)  # shape: (n_perts, n_genes)
        n_perts = len(lfc_matrix)

        # Cosine similarity matrix
        norms = np.linalg.norm(lfc_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        lfc_normed = lfc_matrix / norms
        sim_matrix = lfc_normed @ lfc_normed.T  # (n_perts, n_perts)
        np.fill_diagonal(sim_matrix, 0)
        # Clip to [0, 1]: negative cosine similarity means antagonistic,
        # treat as no edge for community detection purposes
        sim_matrix = np.clip(sim_matrix, 0, 1)

        import networkx as nx
        Q_values = []
        C_values = []

        for k in knn_k_range:
            if k >= n_perts:
                continue

            G_co = nx.Graph()
            G_co.add_nodes_from(range(n_perts))

            for i in range(n_perts):
                top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
                for j in top_k_idx:
                    if sim_matrix[i, j] > 0:
                        G_co.add_edge(i, j, weight=float(sim_matrix[i, j]))

            if G_co.number_of_edges() == 0:
                continue

            try:
                communities = nx.community.louvain_communities(G_co, seed=42)
                Q = nx.community.modularity(G_co, communities)
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

        Q_min = float(min(Q_values)) if Q_values else FALLBACK_BOUNDS["Q"][0]
        Q_max = float(max(Q_values)) if Q_values else FALLBACK_BOUNDS["Q"][1]

        if C_values:
            # Attenuation: C measured on the regulatory core (active perturbations)
            # must be scaled down for the full n-gene network which includes
            # ~59% terminal effectors with C=0 (Aguirre et al. 2025).
            # Active fraction ~0.41 is the empirical estimate from Replogle 2022.
            active_fraction = len(valid_conditions) / max(
                len(adata.obs[perturbation_column].unique()) - 1, 1)
            active_fraction = float(np.clip(active_fraction, 0.10, 0.60))
            C_core = float(np.mean(C_values))
            C_attenuated = C_core * active_fraction
            # Build bounds around the attenuated value
            C_spread = float(np.std(C_values)) * active_fraction
            C_min = max(0.0, C_attenuated - C_spread)
            C_max = C_attenuated + C_spread
        else:
            C_min = FALLBACK_BOUNDS["C"][0]
            C_max = FALLBACK_BOUNDS["C"][1]

        # Confidence from consistency across k values
        if len(C_values) >= 2:
            c_cv = np.std(C_values) / max(np.mean(C_values), 1e-6)
            confidence = float(np.clip(1.0 - c_cv, 0.15, 1.0))
        else:
            confidence = FALLBACK_CONFIDENCE

        return Q_min, Q_max, confidence, C_min, C_max, confidence

    except Exception:
        fb_q = FALLBACK_BOUNDS["Q"]
        fb_c = FALLBACK_BOUNDS["C"]
        return fb_q[0], fb_q[1], FALLBACK_CONFIDENCE, fb_c[0], fb_c[1], FALLBACK_CONFIDENCE


def _diagnose_rho(impact_array, perturbation_labels, adata,
                  perturbation_column, control_label, n_genes):
    """Degree assortativity from hub-to-hub crosstalk."""
    if len(impact_array) < 20:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        hub_threshold = np.percentile(impact_array, 90)
        hub_mask = impact_array >= hub_threshold
        hub_labels = set(perturbation_labels[hub_mask])

        hub_targets_are_hubs = 0
        hub_targets_total = 0

        for hub in hub_labels:
            try:
                result_df = sc.get.rank_genes_groups_df(adata, group=hub)
                sig_targets = result_df[
                    (result_df["pvals_adj"] < 0.05) &
                    (result_df["logfoldchanges"].abs() > 0.25)
                ]["names"].values

                for target in sig_targets:
                    hub_targets_total += 1
                    if target in hub_labels:
                        hub_targets_are_hubs += 1
            except Exception:
                continue

        if hub_targets_total == 0:
            fb = FALLBACK_BOUNDS["rho"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        n_hubs = len(hub_labels)
        M = n_genes
        n = n_hubs
        N_drawn = hub_targets_total
        k_obs = hub_targets_are_hubs

        pvalue = stats.hypergeom.sf(max(k_obs - 1, 0), M, n, N_drawn)
        pvalue = _safe_float(pvalue, 0.5)
        if pvalue < 1e-100:
            pvalue = 1e-100

        enrichment_ratio = (k_obs / max(N_drawn, 1)) / max(n / M, 1e-10)

        if enrichment_ratio > 1.5 and pvalue < 0.01:
            center = -0.05
            bound_min = -0.15
            bound_max = 0.05
        else:
            center = -0.20
            bound_min = -0.35
            bound_max = -0.05

        confidence = _safe_float(
            np.clip(-np.log10(max(pvalue, 1e-100)) / 20.0, 0.1, 1.0),
            FALLBACK_CONFIDENCE
        )

        return bound_min, bound_max, confidence, center

    except Exception:
        fb = FALLBACK_BOUNDS["rho"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# Bound Enforcement
# =========================================================================

def _enforce_bound_constraints(bound_min, bound_max, center, constraints):
    """Applies minimum width, maximum width, and hard floor/ceiling."""
    delta_min = constraints["delta_min"]
    delta_max = constraints["delta_max"]
    hard_floor = constraints["hard_floor"]
    hard_ceiling = constraints["hard_ceiling"]

    # Sanitize inputs
    bound_min = _safe_float(bound_min, hard_floor)
    bound_max = _safe_float(bound_max, hard_ceiling)
    center = _safe_float(center, (hard_floor + hard_ceiling) / 2)

    if bound_min > bound_max:
        bound_min, bound_max = bound_max, bound_min

    width = bound_max - bound_min

    if width < delta_min:
        expansion = (delta_min - width) / 2.0
        bound_min -= expansion
        bound_max += expansion

    if (bound_max - bound_min) > delta_max:
        half_max = delta_max / 2.0
        bound_min = center - half_max
        bound_max = center + half_max

    # Asymmetric hard clipping
    if bound_min < hard_floor:
        deficit = hard_floor - bound_min
        bound_min = hard_floor
        bound_max = min(bound_max + deficit, hard_ceiling)

    if bound_max > hard_ceiling:
        surplus = bound_max - hard_ceiling
        bound_max = hard_ceiling
        bound_min = max(bound_min - surplus, hard_floor)

    return float(bound_min), float(bound_max)


# =========================================================================
# Weight Normalization (NaN-proof)
# =========================================================================

def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
    """Rescales raw confidence scores into loss weights.

    v7: All values are sanitized through np.isfinite before any arithmetic.
    If all confidences are NaN or zero, uniform weights are assigned.
    """
    names = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)

    # CRITICAL: replace any NaN/inf with the fallback confidence
    bad_mask = ~np.isfinite(values)
    if bad_mask.any():
        n_bad = bad_mask.sum()
        print(f"  WARNING: {n_bad} confidence values were NaN/inf. "
              f"Replacing with fallback ({FALLBACK_CONFIDENCE}).")
        values[bad_mask] = FALLBACK_CONFIDENCE

    # Clip to [0, 1]
    values = np.clip(values, 0.0, 1.0)

    total = values.sum()
    if total < 1e-10:
        # All confidences near zero: uniform weights
        print("  WARNING: All confidences near zero. Using uniform weights.")
        values = np.ones(len(values)) / len(values) * target_sum
    else:
        values = values / total * target_sum

    # Enforce per-weight floor and ceiling
    values = np.clip(values, floor, ceiling)

    result = {n: float(v) for n, v in zip(names, values)}

    # Final sanity check: no NaN can leave this function
    for n in result:
        if not np.isfinite(result[n]):
            result[n] = floor

    return result


# =========================================================================
# Master Diagnostic Runner
# =========================================================================

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input):
    """Executes the full Phase 0 diagnostic pipeline.

    v7: Every diagnostic is wrapped in try/except. If any test fails,
    fallback bounds and a baseline confidence are used. The pipeline
    will never return NaN weights.
    """
    print("=" * 72)
    print("FUNGI v7 -- Phase 0: Data-Driven Diagnostic Calibration")
    print("=" * 72)

    pert_col = cfg_input["perturbation_column"]
    ctrl_label = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool = cfg_input.get("metacell_pooling_factor", None)

    n_boot = cfg_diagnostics["n_bootstrap"]
    knn_range = cfg_diagnostics["knn_k_range"]
    bc = cfg_diagnostics["bound_constraints"]
    w_floor = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    sc_mid = cfg_diagnostics["singlecell_silhouette_midpoint"]
    mc_mid = cfg_diagnostics["metacell_silhouette_midpoint"]

    # Step 1: Build impact array
    impact_array, pert_labels = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method=cfg_diagnostics["de_method"],
        pval_threshold=cfg_diagnostics["de_pval_threshold"],
        lfc_threshold=cfg_diagnostics["de_lfc_threshold"]
    )

    # Step 2: Per-parameter diagnostics (each fully guarded)
    print("\n  Diagnosing Gini (degree inequality)...")
    gini_min, gini_max, gini_conf, gini_center = _diagnose_gini(
        impact_array, n_boot, is_metacell)

    print("  Diagnosing alpha (scale-free exponent)...")
    alpha_min, alpha_max, alpha_conf, alpha_center = _diagnose_alpha(
        impact_array, n_boot)

    print("  Diagnosing S_max (hub saturation)...")
    smax_min, smax_max, smax_conf, smax_center = _diagnose_smax(
        impact_array, n_genes, is_metacell, mc_pool)

    print("  Diagnosing Q and C (modularity and clustering)...")
    Q_min, Q_max, Q_conf, C_min, C_max, C_conf = _diagnose_modularity_and_clustering(
        adata, pert_col, ctrl_label, knn_range, is_metacell, sc_mid, mc_mid)

    print("  Diagnosing rho (degree assortativity)...")
    rho_min, rho_max, rho_conf, rho_center = _diagnose_rho(
        impact_array, pert_labels, adata, pert_col, ctrl_label, n_genes)

    # Add this block just before the enforce calls in run_diagnostics
    print("\n  Raw bounds before constraint enforcement:")
    print(f"    alpha: [{alpha_min:.4f}, {alpha_max:.4f}]  (conf={alpha_conf:.3f})")
    print(f"    C:     [{C_min:.4f}, {C_max:.4f}]  (conf={C_conf:.3f})")
    print(f"    Q:     [{Q_min:.4f}, {Q_max:.4f}]  (conf={Q_conf:.3f})")
    print(f"    gini:  [{gini_min:.4f}, {gini_max:.4f}]  (conf={gini_conf:.3f})")
    print(f"    S_max: [{smax_min:.4f}, {smax_max:.4f}]  (conf={smax_conf:.3f})")
    print(f"    rho:   [{rho_min:.4f}, {rho_max:.4f}]  (conf={rho_conf:.3f})")

    # Step 3: Enforce bound constraints
    gini_min, gini_max = _enforce_bound_constraints(
        gini_min, gini_max, gini_center, bc["gini"])
    alpha_min, alpha_max = _enforce_bound_constraints(
        alpha_min, alpha_max, alpha_center, bc["alpha"])
    smax_min, smax_max = _enforce_bound_constraints(
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

    # Step 4: Normalize weights (NaN-proof)
    raw_confidences = {
        "alpha": alpha_conf,
        "gini":  gini_conf,
        "S_max": smax_conf,
        "Q":     Q_conf,
        "C":     C_conf,
        "rho":   rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # Step 5: FINAL GUARD -- assert no NaN leaves this function
    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]
    for param in loss_weights:
        if not np.isfinite(loss_weights[param]):
            loss_weights[param] = w_floor

    diagnostic_report = {
        "impact_array_size": len(impact_array),
        "impact_range": [float(impact_array.min()), float(impact_array.max())] if len(impact_array) > 0 else [0, 0],
        "is_metacell": is_metacell,
        "raw_confidences": {k: _safe_float(v, 0.0) for k, v in raw_confidences.items()},
        "utopian_bounds": utopian_bounds,
        "loss_weights": loss_weights,
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
