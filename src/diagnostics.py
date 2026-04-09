"""
FUNGI v6 -- Phase 0: Data-Driven Diagnostic Calibration

Replaces the fixed archetype system (Generalist / Structuralist / Modularist /
Hierarchist) with a single, dataset-specific set of utopian bounds and loss
weights derived empirically from the training h5ad file.

The module performs six statistical evaluations on the perturbation response
data to reverse-engineer the topological properties that the pruned graph
should exhibit.  Each evaluation returns:
    - A utopian bound range [min, max] for one of the six output parameters.
    - A scalar weight for the utopian loss function.

References:
    Clauset, Shalizi & Newman (2009). Power-law distributions in empirical data.
    Blondel et al. (2008). Fast unfolding of communities in large networks.
    Moran (1950). Notes on continuous stochastic phenomena.
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
# Impact Array Construction
# =========================================================================

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25):
    """Computes the number of significantly altered genes (DEGs) per
    perturbation condition, producing a 1D impact array.

    Parameters
    ----------
    adata : AnnData
        Training h5ad with perturbation metadata.
    perturbation_column : str
        Column in adata.obs identifying perturbation conditions.
    control_label : str
        Label for unperturbed control cells.
    de_method : str
        Method for scanpy.tl.rank_genes_groups.
    pval_threshold : float
        Adjusted p-value cutoff for DEG significance.
    lfc_threshold : float
        Minimum absolute log-fold-change for DEG significance.

    Returns
    -------
    impact_array : np.ndarray
        1D array of DEG counts per perturbation (length = n_perturbations).
    perturbation_labels : np.ndarray
        Corresponding perturbation condition names.
    """
    print("Phase 0: Building perturbation impact array...")

    # Identify perturbation groups (excluding controls)
    conditions = adata.obs[perturbation_column].unique()
    conditions = [c for c in conditions if c != control_label]

    # Run DE analysis: each perturbation vs. control
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

    # Remove perturbations with zero impact (non-functional knockouts)
    nonzero_mask = impact_array > 0
    impact_array = impact_array[nonzero_mask]
    perturbation_labels = perturbation_labels[nonzero_mask]

    print(f"  {len(impact_array)} perturbations with nonzero impact.")
    print(f"  Impact range: [{impact_array.min():.0f}, {impact_array.max():.0f}] DEGs.")
    return impact_array, perturbation_labels


# =========================================================================
# Parameter-Specific Diagnostic Functions
# =========================================================================

def _diagnose_gini(impact_array, n_bootstrap, is_metacell):
    """Reverse-engineers the Gini coefficient from the impact distribution.

    The Gini coefficient of the perturbation impact array mirrors the degree
    inequality of the underlying causal graph: if a few perturbations cause
    massive transcriptomic shifts while most do little, the regulatory
    network must have high degree inequality.

    Returns (bound_min, bound_max, weight).
    """
    def _gini(arr):
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        sorted_arr = np.sort(arr)
        index = np.arange(1, n + 1)
        return float(np.sum((2 * index - n - 1) * sorted_arr) / (n * np.sum(sorted_arr)))

    # Bootstrap the Gini estimate
    boot_ginis = np.array([
        _gini(np.random.choice(impact_array, size=len(impact_array), replace=True))
        for _ in range(n_bootstrap)
    ])

    center = np.median(boot_ginis)
    bound_min = np.percentile(boot_ginis, 5)
    bound_max = np.percentile(boot_ginis, 95)

    # Weight: inversely proportional to bootstrap variance
    variance = np.var(boot_ginis)
    if variance < 1e-10:
        variance = 1e-10
    raw_weight = 1.0 / variance
    # We normalize weights later; just return the raw signal
    confidence = 1.0 - np.clip(np.sqrt(variance) * 10, 0, 0.9)

    return bound_min, bound_max, confidence, center


def _diagnose_alpha(impact_array, n_bootstrap):
    """Reverse-engineers the scale-free exponent from the impact distribution.

    Fits a power-law to the perturbation impact array using MLE. The exponent
    of this fit is the target alpha for the pruned graph.

    Returns (bound_min, bound_max, weight).
    """
    try:
        import powerlaw
    except ImportError:
        raise ImportError("The 'powerlaw' package is required. Install via: pip install powerlaw")

    # Primary fit with automatic xmin selection (Clauset et al. 2009)
    fit = powerlaw.Fit(impact_array, discrete=True, verbose=False)
    alpha_est = fit.power_law.alpha
    sigma_est = fit.power_law.sigma  # standard error from MLE

    # KS distance for weight calculation
    ks_distance = fit.power_law.D

    # Bounds from 95% confidence interval
    bound_min = alpha_est - 1.96 * sigma_est
    bound_max = alpha_est + 1.96 * sigma_est

    # Weight: inverse KS distance (better fit = higher weight)
    if ks_distance < 1e-10:
        ks_distance = 1e-10
    raw_weight = 1.0 / ks_distance

    # Confidence metric (normalized for downstream scaling)
    # Use R-squared proxy from comparing empirical vs fitted CDF
    r_squared_proxy = max(0.0, 1.0 - ks_distance)
    if r_squared_proxy > 0.999:
        r_squared_proxy = 0.999
    confidence = -np.log10(1.0 - r_squared_proxy)  # "nines" scale

    return bound_min, bound_max, confidence, alpha_est


def _diagnose_smax(impact_array, n_genes, is_metacell, metacell_pooling_factor):
    """Reverse-engineers the hub saturation ceiling from the strongest
    perturbation impacts.

    S_max = I_max / N, representing the maximum fraction of the genome
    regulated by a single master regulator.

    Returns (bound_min, bound_max, weight).
    """
    # Metacell correction: E-distance inflates when variance is compressed
    if is_metacell and metacell_pooling_factor is not None and metacell_pooling_factor > 1:
        impact_array = impact_array / np.sqrt(metacell_pooling_factor)

    sorted_impact = np.sort(impact_array)[::-1]

    # Use the 95th percentile to the max as the bound range
    p95_impact = np.percentile(impact_array, 95)
    max_impact = sorted_impact[0]

    bound_min = p95_impact / n_genes
    bound_max = max_impact / n_genes

    # Ensure bound_min < bound_max
    if bound_min >= bound_max:
        bound_max = bound_min * 1.5

    # Weight: signal-to-noise ratio of the strongest perturbation
    median_impact = np.median(impact_array)
    if median_impact < 1:
        median_impact = 1
    snr = max_impact / median_impact
    confidence = np.clip(np.log10(snr), 0.1, 3.0)

    center = (bound_min + bound_max) / 2.0
    return bound_min, bound_max, confidence, center


def _diagnose_modularity_and_clustering(adata, perturbation_column,
                                        control_label, knn_k_range,
                                        is_metacell, sc_midpoint, mc_midpoint):
    """Reverse-engineers modularity (Q) and clustering (C) from phenotypic
    clustering of perturbation centroids.

    If perturbations cluster into distinct phenotypic islands, the underlying
    causal graph must be modular (signal stays within pathway boundaries).

    Returns (Q_min, Q_max, Q_confidence, C_min, C_max, C_confidence).
    """
    print("  Computing perturbation centroids for modularity diagnostics...")

    conditions = adata.obs[perturbation_column].unique()
    conditions = [c for c in conditions if c != control_label]

    # Compute perturbation centroids
    centroids = []
    valid_conditions = []
    for cond in conditions:
        mask = adata.obs[perturbation_column] == cond
        if mask.sum() < 5:
            continue
        subset = adata[mask]
        if hasattr(subset.X, 'toarray'):
            centroid = np.array(subset.X.toarray().mean(axis=0)).ravel()
        else:
            centroid = np.array(subset.X.mean(axis=0)).ravel()
        centroids.append(centroid)
        valid_conditions.append(cond)

    if len(centroids) < 10:
        print("  WARNING: Too few valid perturbations for modularity diagnostics.")
        return 0.2, 0.6, 0.5, 0.05, 0.30, 0.5

    centroid_matrix = np.array(centroids)

    Q_values = []
    C_values = []
    silhouette_scores = []

    for k in knn_k_range:
        if k >= len(centroids):
            continue

        # Build KNN graph on perturbation centroids
        knn_adj = kneighbors_graph(centroid_matrix, n_neighbors=k,
                                   mode='connectivity', include_self=False)

        # Convert to networkx for Q and C calculation
        import networkx as nx
        G_knn = nx.from_scipy_sparse_array(knn_adj)

        # Modularity via Louvain
        try:
            communities = nx.community.louvain_communities(G_knn, seed=42)
            Q = nx.community.modularity(G_knn, communities)
            Q_values.append(Q)
        except Exception:
            Q_values.append(0.3)

        # Clustering coefficient
        C = nx.average_clustering(G_knn)
        C_values.append(C)

        # Silhouette score for weight calculation
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(len(communities), len(centroids) - 1)
            if n_clusters >= 2:
                labels = np.zeros(len(centroids), dtype=int)
                for ci, comm in enumerate(communities):
                    for node in comm:
                        labels[node] = ci
                sil = silhouette_score(centroid_matrix, labels)
                silhouette_scores.append(sil)
        except Exception:
            silhouette_scores.append(0.25)

    if not Q_values:
        return 0.2, 0.6, 0.5, 0.05, 0.30, 0.5

    Q_min = min(Q_values)
    Q_max = max(Q_values)
    C_min = min(C_values)
    C_max = max(C_values)

    # Weight via silhouette sigmoid
    mean_sil = np.mean(silhouette_scores) if silhouette_scores else 0.25
    midpoint = mc_midpoint if is_metacell else sc_midpoint
    # Logistic scaling: floor near W_min for poor clustering, ceiling for strong
    k_steepness = 8.0
    sigmoid_val = 1.0 / (1.0 + np.exp(-k_steepness * (mean_sil - midpoint)))
    Q_confidence = sigmoid_val
    C_confidence = sigmoid_val

    return Q_min, Q_max, Q_confidence, C_min, C_max, C_confidence


def _diagnose_rho(impact_array, perturbation_labels, adata,
                  perturbation_column, control_label, n_genes):
    """Reverse-engineers degree assortativity from hub-to-hub crosstalk.

    Identifies the top-impact perturbations (hub genes), then checks whether
    their downstream targets are enriched for other hub genes.  If hubs
    primarily affect non-hubs, the network is disassortative (negative rho).

    Returns (bound_min, bound_max, confidence, center).
    """
    if len(impact_array) < 20:
        return -0.30, -0.05, 0.5, -0.15

    # Identify hub perturbations (top 10% by impact)
    hub_threshold = np.percentile(impact_array, 90)
    hub_mask = impact_array >= hub_threshold
    hub_labels = set(perturbation_labels[hub_mask])

    # Get the DEG targets of hub perturbations
    hub_targets_are_hubs = 0
    hub_targets_total = 0

    try:
        for hub in hub_labels:
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
        return -0.30, -0.05, 0.5, -0.15

    if hub_targets_total == 0:
        return -0.30, -0.05, 0.5, -0.15

    # Hypergeometric test: are hub genes overrepresented among targets?
    n_hubs = len(hub_labels)
    M = n_genes                    # population size
    n = n_hubs                     # successes in population
    N_drawn = hub_targets_total    # sample size
    k_obs = hub_targets_are_hubs   # observed successes

    pvalue = stats.hypergeom.sf(k_obs - 1, M, n, N_drawn)

    # Interpret: low p-value with high enrichment = assortative (hub-hub)
    enrichment_ratio = (k_obs / max(N_drawn, 1)) / (n / M)

    if enrichment_ratio > 1.5 and pvalue < 0.01:
        # Hub-hub crosstalk detected: rho should be closer to 0
        center = -0.05
        bound_min = -0.15
        bound_max = 0.05
    else:
        # Disassortative: hubs regulate non-hubs
        center = -0.20
        bound_min = -0.35
        bound_max = -0.05

    # Confidence from p-value (negative log scale)
    if pvalue < 1e-100:
        pvalue = 1e-100
    confidence = np.clip(-np.log10(pvalue) / 20.0, 0.1, 1.0)

    return bound_min, bound_max, confidence, center


# =========================================================================
# Bound Enforcement (delta_min, delta_max, asymmetric clipping)
# =========================================================================

def _enforce_bound_constraints(bound_min, bound_max, center, constraints):
    """Applies minimum width, maximum width, and hard floor/ceiling clipping.

    Parameters
    ----------
    bound_min, bound_max : float
        Raw bounds from the diagnostic.
    center : float
        Empirical center estimate.
    constraints : dict
        Must contain: delta_min, delta_max, hard_floor, hard_ceiling.

    Returns
    -------
    bound_min, bound_max : float
        Constrained bounds.
    """
    delta_min = constraints["delta_min"]
    delta_max = constraints["delta_max"]
    hard_floor = constraints["hard_floor"]
    hard_ceiling = constraints["hard_ceiling"]

    width = bound_max - bound_min

    # Enforce minimum width
    if width < delta_min:
        expansion = (delta_min - width) / 2.0
        bound_min -= expansion
        bound_max += expansion

    # Enforce maximum width
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
# Weight Normalization
# =========================================================================

def _normalize_weights(raw_weights, floor, ceiling, target_sum=100.0):
    """Rescales raw confidence scores into loss weights that sum to a
    stable target, preventing any single parameter from dominating.

    Each weight is bounded by [floor, ceiling] after scaling.

    Parameters
    ----------
    raw_weights : dict
        {param_name: raw_confidence_score}.
    floor, ceiling : float
        Per-weight minimum and maximum.
    target_sum : float
        Desired sum of all weights.

    Returns
    -------
    dict : {param_name: final_weight}
    """
    names = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)

    # Clip to [0, 1] range before scaling
    values = np.clip(values, 0.0, 1.0)

    # Scale so that proportional structure is preserved but sum hits target
    total = values.sum()
    if total < 1e-10:
        # All confidences near zero: use uniform weights
        values = np.ones(len(values)) / len(values) * target_sum
    else:
        values = values / total * target_sum

    # Enforce per-weight floor and ceiling
    values = np.clip(values, floor, ceiling)

    return {n: float(v) for n, v in zip(names, values)}


# =========================================================================
# Master Diagnostic Runner
# =========================================================================

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input):
    """Executes the full Phase 0 diagnostic pipeline.

    Parameters
    ----------
    adata : AnnData
        Training h5ad (already loaded).
    n_genes : int
        Number of genes in the GRN.
    cfg_diagnostics : dict
        The 'diagnostics' section of the YAML config.
    cfg_input : dict
        The 'input' section of the YAML config.

    Returns
    -------
    utopian_bounds : dict
        {param_name: [min, max]} for the six output topology parameters.
    loss_weights : dict
        {param_name: weight} for the utopian loss function.
    diagnostic_report : dict
        Full diagnostic metadata for logging and reproducibility.
    """
    print("=" * 72)
    print("FUNGI v6 -- Phase 0: Data-Driven Diagnostic Calibration")
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

    # Step 2: Per-parameter diagnostics
    print("\n  Diagnosing Gini (degree inequality)...")
    gini_min, gini_max, gini_conf, gini_center = _diagnose_gini(
        impact_array, n_boot, is_metacell
    )

    print("  Diagnosing alpha (scale-free exponent)...")
    alpha_min, alpha_max, alpha_conf, alpha_center = _diagnose_alpha(
        impact_array, n_boot
    )

    print("  Diagnosing S_max (hub saturation)...")
    smax_min, smax_max, smax_conf, smax_center = _diagnose_smax(
        impact_array, n_genes, is_metacell, mc_pool
    )

    print("  Diagnosing Q and C (modularity and clustering)...")
    Q_min, Q_max, Q_conf, C_min, C_max, C_conf = _diagnose_modularity_and_clustering(
        adata, pert_col, ctrl_label, knn_range, is_metacell, sc_mid, mc_mid
    )

    print("  Diagnosing rho (degree assortativity)...")
    rho_min, rho_max, rho_conf, rho_center = _diagnose_rho(
        impact_array, pert_labels, adata, pert_col, ctrl_label, n_genes
    )

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

    # Step 4: Normalize weights
    raw_confidences = {
        "alpha": alpha_conf,
        "gini":  gini_conf,
        "S_max": smax_conf,
        "Q":     Q_conf,
        "C":     C_conf,
        "rho":   rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # Step 5: Build diagnostic report
    diagnostic_report = {
        "impact_array_size": len(impact_array),
        "impact_range": [float(impact_array.min()), float(impact_array.max())],
        "is_metacell": is_metacell,
        "raw_confidences": raw_confidences,
        "utopian_bounds": utopian_bounds,
        "loss_weights": loss_weights,
    }

    # Print summary
    print(f"\n{'=' * 72}")
    print("Phase 0 Results: Custom Utopian Calibration")
    print(f"{'=' * 72}")
    for param in utopian_bounds:
        bmin, bmax = utopian_bounds[param]
        w = loss_weights[param]
        print(f"  {param:>6s}:  bounds = [{bmin:.4f}, {bmax:.4f}]  |  weight = {w:.2f}")
    print(f"{'=' * 72}\n")

    return utopian_bounds, loss_weights, diagnostic_report
