"""
FUNGI v7.4 -- Phase 0: Data-Driven Diagnostic Calibration

Changes vs v7.3:
  alpha : impact-fit + universal diffusion shift, width-clipped to
          physically valid scale-free range [1.8, 3.5].
  Gini  : impact bootstrap center, universal +-0.10 width, clipped
          to physical scale-free range [0.40, 0.80].
  S_max : unchanged.
  Q     : unchanged.
  C     : empirical MST-based transfer function on LFC-cosine graph.
          Replaces the Gini-attenuation hack. Measures what fraction
          of cosine-clustering survives when functional cliques are
          collapsed to their minimum spanning tree, then scales C_core
          by that empirical ratio.
  rho   : new causal-skeleton approach. Builds a perturbation x gene
          DEG matrix, masks the parent graph with it (edge-wise AND),
          and measures rho on the intersection. No literature prior.

API changes:
  build_impact_array now returns 4 values instead of 3:
      impact_array, labels, weights, deg_matrix
  The deg_matrix is a sparse (n_genes, n_genes) matrix where
  M[i, j] = 1 iff gene j was a DEG when gene i was perturbed.
  Used by _diagnose_rho_causal.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import warnings
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Fallback defaults, used only when a diagnostic completely fails.
FALLBACK_BOUNDS = {
    "alpha": [2.1, 2.7],
    "gini":  [0.55, 0.75],
    "S_max": [0.04, 0.10],
    "Q":     [0.25, 0.50],
    "C":     [0.04, 0.12],
    "rho":   [-0.25, -0.05],
}
FALLBACK_CONFIDENCE = 0.3


def _safe_float(val, fallback=0.0):
    if val is None or not np.isfinite(val):
        return fallback
    return float(val)


# =========================================================================
# Impact Array + DEG matrix construction
# =========================================================================

def build_impact_array(adata, perturbation_column, control_label,
                       de_method="wilcoxon", pval_threshold=0.05,
                       lfc_threshold=0.25, n_jobs=15,
                       max_perts_for_de=500, min_cells_per_pert=5,
                       is_metacell=False, metacell_pooling_factor=None):
    """
    Runs Wilcoxon + BH on a stratified subset of perturbations and returns:
      impact_array       : DEG count per perturbation (nonzero only)
      perturbation_labels: gene symbols of those perturbations
      sample_weights     : Top-K + Random-Tail projection weights
      deg_matrix         : sparse (n_genes, n_genes) where M[i,j]=1 iff
                           gene j was a DEG when gene i was perturbed

    The deg_matrix feeds _diagnose_rho_causal. Rows are populated only
    for the perturbations actually tested; all other rows are zero.

    Metacell LFC correction (v7.3): effective cutoff is scaled by
    1/sqrt(metacell_pooling_factor) when is_metacell=True.
    """
    print("Phase 0: Building perturbation impact array...")

    # Metacell-aware LFC cutoff
    if is_metacell and metacell_pooling_factor and metacell_pooling_factor > 1:
        effective_lfc_cutoff = lfc_threshold / np.sqrt(metacell_pooling_factor)
        print(f"  Metacell pooling (factor={metacell_pooling_factor}): "
              f"LFC cutoff {lfc_threshold:.3f} -> "
              f"{effective_lfc_cutoff:.3f} effective.")
    else:
        effective_lfc_cutoff = lfc_threshold

    conditions_arr = adata.obs[perturbation_column].values
    ctrl_mask      = conditions_arr == control_label
    unique_conds   = [c for c in np.unique(conditions_arr) if c != control_label]
    n_total        = len(unique_conds)
    print(f"  {n_total:,} perturbation groups detected.")

    # Fast LFC proxy for ranking
    print("  Computing fast LFC proxy for perturbation ranking...")
    X_csr     = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
    ctrl_mean = np.asarray(X_csr[ctrl_mask].mean(axis=0)).ravel()
    log_ctrl  = np.log1p(np.maximum(ctrl_mean, 0))

    proxy_scores = {}
    for cond in tqdm(unique_conds, desc="  LFC proxy", unit="pert", ncols=80):
        mask = conditions_arr == cond
        if mask.sum() < 2:
            proxy_scores[cond] = 0.0
            continue
        cond_mean = np.asarray(X_csr[mask].mean(axis=0)).ravel()
        lfc       = np.log1p(np.maximum(cond_mean, 0)) - log_ctrl
        proxy_scores[cond] = float(np.mean(np.abs(lfc)))

    # Stratified selection: Top-K + Random Tail
    selected_conds = unique_conds
    sample_weights = {c: 1.0 for c in unique_conds}

    if n_total > max_perts_for_de:
        sorted_conds = sorted(unique_conds,
                              key=lambda c: proxy_scores[c], reverse=True)
        n_top        = min(100, max_perts_for_de // 5)
        n_random     = max_perts_for_de - n_top
        top_conds    = sorted_conds[:n_top]
        tail_conds   = sorted_conds[n_top:]

        rng          = np.random.default_rng(42)
        n_draw       = min(n_random, len(tail_conds))
        random_conds = list(rng.choice(tail_conds, size=n_draw, replace=False))

        selected_conds = top_conds + random_conds
        tail_weight    = float(len(tail_conds)) / max(n_draw, 1)
        for c in top_conds:    sample_weights[c] = 1.0
        for c in random_conds: sample_weights[c] = tail_weight

        print(f"  Selected {len(selected_conds)} perts "
              f"({n_top} top-proxy + {len(random_conds)} random tail) "
              f"from {n_total:,} total.")
    else:
        print(f"  Running Wilcoxon on all {n_total} perturbations.")

    # Cell filter and Wilcoxon
    keep_mask = ctrl_mask.copy()
    for cond in selected_conds:
        keep_mask = keep_mask | (conditions_arr == cond)
    adata_sub = adata[keep_mask].copy()
    print(f"  Wilcoxon subset: {adata_sub.n_obs:,} cells.")

    sc.tl.rank_genes_groups(
        adata_sub, groupby=perturbation_column,
        reference=control_label, method=de_method,
        use_raw=False, n_jobs=n_jobs,
    )

    # Build gene-name -> index map for the deg_matrix
    var_names = list(adata.var_names)
    n_genes   = len(var_names)

    symbol_col = None
    for candidate in ["gene_name", "gene_symbols", "gene_symbol",
                      "feature_name", "symbol", "Symbol"]:
        if candidate in adata.var.columns:
            symbol_col = candidate
            break

    name_to_idx = {str(vn): i for i, vn in enumerate(var_names)}
    if symbol_col is not None:
        symbols = adata.var[symbol_col].astype(str).values
        for i, sym in enumerate(symbols):
            name_to_idx.setdefault(sym, i)
        print(f"  Gene index: using adata.var['{symbol_col}'] + var_names "
              f"({len(name_to_idx):,} names mapped).")
    else:
        print(f"  Gene index: using var_names only "
              f"({len(name_to_idx):,} names mapped).")

    # Extract DEG counts AND DEG identities per perturbation
    impact_scores = []
    valid_labels  = []
    valid_weights = []
    deg_rows      = []  # perturbation index
    deg_cols      = []  # DEG gene index

    for cond in tqdm(selected_conds, desc="  DEG counts", unit="pert", ncols=80):
        try:
            result_df = sc.get.rank_genes_groups_df(adata_sub, group=cond)
            sig = result_df[
                (result_df["pvals_adj"] < pval_threshold) &
                (result_df["logfoldchanges"].abs() > effective_lfc_cutoff)
            ]
            impact_scores.append(len(sig))
            valid_labels.append(cond)
            valid_weights.append(sample_weights.get(cond, 1.0))

            # Record DEG gene indices for this perturbation, if the
            # perturbed gene is itself present in var_names (otherwise
            # we can't assign a row index in the square matrix).
            pert_idx = name_to_idx.get(str(cond), None)
            if pert_idx is not None:
                for deg_name in sig["names"].values:
                    deg_idx = name_to_idx.get(str(deg_name), None)
                    if deg_idx is not None and deg_idx != pert_idx:
                        deg_rows.append(pert_idx)
                        deg_cols.append(deg_idx)
        except Exception:
            continue

    impact_array        = np.array(impact_scores, dtype=np.float64)
    perturbation_labels = np.array(valid_labels)
    weights_arr         = np.array(valid_weights, dtype=np.float64)

    nonzero_mask        = impact_array > 0
    n_tested            = len(impact_array)
    impact_array        = impact_array[nonzero_mask]
    perturbation_labels = perturbation_labels[nonzero_mask]
    weights_arr         = weights_arr[nonzero_mask]

    # Build sparse DEG matrix: (n_genes, n_genes), row=perturbation gene,
    # col=DEG gene. Binary.
    if len(deg_rows) > 0:
        deg_matrix = sp.coo_matrix(
            (np.ones(len(deg_rows)),
             (np.array(deg_rows), np.array(deg_cols))),
            shape=(n_genes, n_genes)
        ).tocsr()
        # Deduplicate any repeated entries
        deg_matrix.data = np.ones_like(deg_matrix.data)
    else:
        deg_matrix = sp.csr_matrix((n_genes, n_genes))

    n_nonzero = len(impact_array)
    print(f"  Wilcoxon tested : {n_tested} of {n_total:,} perturbations.")
    print(f"  Active perts    : {n_nonzero} of {n_tested} tested "
          f"({100.0 * n_nonzero / max(n_tested, 1):.1f}% in-sample hit rate).")
    print(f"  DEG matrix      : {deg_matrix.nnz:,} causal edges "
          f"(perturbation -> DEG).")

    if n_nonzero > 0:
        print(f"  Impact range    : [{impact_array.min():.0f}, "
              f"{impact_array.max():.0f}] DEGs.")
        eff_n = float(weights_arr.sum())
        print(f"  Effective N     : {eff_n:.0f} of {n_total:,} "
              f"(~{100.0 * eff_n / max(n_total, 1):.1f}% screen-wide).")

    return impact_array, perturbation_labels, weights_arr, deg_matrix


# =========================================================================
# Alpha: impact-fit + universal diffusion shift, clipped to scale-free range
# =========================================================================

def _diagnose_alpha(impact_array, n_bootstrap):
    """
    Alpha from power-law MLE on the impact array, plus a universal +0.3
    diffusion shift. Width is the MLE sigma interval, then clipped to
    the physically valid scale-free range [1.8, 3.5]:
      alpha < 1.8 implies infinite variance (unphysical for finite graphs)
      alpha > 3.5 is effectively a Poisson graph (not scale-free)
    """
    DIFFUSION_SHIFT = 0.30   # universal: impact-alpha systematically
                             # under-shoots graph-degree-alpha by ~0.3
    PHYS_FLOOR      = 1.8
    PHYS_CEILING    = 3.5
    MIN_WIDTH       = 0.30

    if len(impact_array) < 10:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        import powerlaw
        fit       = powerlaw.Fit(impact_array, xmin=2, discrete=True,
                                 verbose=False)
        alpha_raw = _safe_float(fit.power_law.alpha, 2.3)
        sigma     = _safe_float(fit.power_law.sigma, 0.2)
        ks_dist   = _safe_float(fit.power_law.D, 0.5)

        alpha_center = alpha_raw + DIFFUSION_SHIFT

        half_width = max(1.96 * sigma, MIN_WIDTH / 2)
        bound_min  = alpha_center - half_width
        bound_max  = alpha_center + half_width

        bound_min = max(bound_min, PHYS_FLOOR)
        bound_max = min(bound_max, PHYS_CEILING)
        if bound_min >= bound_max:
            bound_min, bound_max = PHYS_FLOOR, PHYS_FLOOR + MIN_WIDTH

        r2_proxy = float(np.clip(1.0 - ks_dist, 0.01, 0.999))
        conf     = float(np.clip(-np.log10(1.0 - r2_proxy) / 3.0, 0.1, 1.0))

        return bound_min, bound_max, conf, alpha_center

    except Exception:
        fb = FALLBACK_BOUNDS["alpha"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# Gini: impact bootstrap center + universal width + scale-free clip
# =========================================================================

def _diagnose_gini(impact_array, n_bootstrap, is_metacell,
                   sample_weights=None):
    """
    Weighted Gini bootstrap for the center, then add universal +-0.10
    width, clipped to physical scale-free range [0.40, 0.80]:
      Gini < 0.40 -> too uniform for any scale-free structure
      Gini > 0.80 -> single oligarch; no meaningful network
    """
    UNIVERSAL_WIDTH = 0.10
    PHYS_FLOOR      = 0.40
    PHYS_CEILING    = 0.80

    if len(impact_array) < 5:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    def weighted_gini(arr, weights):
        if weights is None:
            weights = np.ones(len(arr))
        order   = np.argsort(arr)
        arr_s   = arr[order]
        w_s     = weights[order]
        w_sum   = w_s.sum()
        if w_sum == 0 or arr_s.sum() == 0:
            return 0.0
        cum_w  = np.cumsum(w_s) / w_sum
        cum_v  = np.cumsum(arr_s * w_s) / (arr_s * w_s).sum()
        lorenz = np.trapz(cum_v, cum_w)
        return float(1.0 - 2.0 * lorenz)

    try:
        rng = np.random.default_rng(42)
        n   = len(impact_array)
        boot_ginis = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            g = weighted_gini(
                impact_array[idx],
                sample_weights[idx] if sample_weights is not None else None
            )
            if np.isfinite(g):
                boot_ginis.append(g)

        boot_ginis = np.array(boot_ginis)
        if len(boot_ginis) < 10:
            fb = FALLBACK_BOUNDS["gini"]
            return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

        center = float(np.median(boot_ginis))

        bound_min = center - UNIVERSAL_WIDTH
        bound_max = center + UNIVERSAL_WIDTH
        bound_min = max(bound_min, PHYS_FLOOR)
        bound_max = min(bound_max, PHYS_CEILING)
        if bound_min >= bound_max:
            # Shift the window inside the physical range
            if center < (PHYS_FLOOR + PHYS_CEILING) / 2:
                bound_min = PHYS_FLOOR
                bound_max = PHYS_FLOOR + 2 * UNIVERSAL_WIDTH
            else:
                bound_max = PHYS_CEILING
                bound_min = PHYS_CEILING - 2 * UNIVERSAL_WIDTH

        variance = max(float(np.var(boot_ginis)), 1e-10)
        conf     = _safe_float(
            1.0 - np.clip(np.sqrt(variance) * 10, 0, 0.9),
            FALLBACK_CONFIDENCE)

        return bound_min, bound_max, conf, center

    except Exception:
        fb = FALLBACK_BOUNDS["gini"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# S_max: unchanged from v7.3
# =========================================================================

def _diagnose_smax(impact_array, n_genes, is_metacell,
                   metacell_pooling_factor, sample_weights=None):
    """
    Hub saturation from percentile / max of impact array, normalised by
    n_genes. Metacell correction lives in build_impact_array now.
    """
    if len(impact_array) < 5 or n_genes < 1:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        arr = impact_array.copy()

        if sample_weights is not None and len(sample_weights) == len(arr):
            w_int    = np.maximum(np.round(sample_weights).astype(int), 1)
            expanded = np.repeat(arr, w_int)
            p95      = float(np.percentile(expanded, 95))
            max_imp  = float(expanded.max())
        else:
            p95     = float(np.percentile(arr, 95))
            max_imp = float(arr.max())

        bound_min = p95 / n_genes
        bound_max = max_imp / n_genes
        if bound_min >= bound_max:
            bound_max = bound_min * 1.5
        if bound_max < 0.01:
            bound_max = 0.10

        median_imp = max(float(np.median(arr)), 1.0)
        snr        = max_imp / median_imp
        conf       = float(np.clip(
            np.log10(max(snr, 1.01)) / 3.0, 0.1, 1.0))
        center     = (bound_min + bound_max) / 2.0
        return bound_min, bound_max, conf, center

    except Exception:
        fb = FALLBACK_BOUNDS["S_max"]
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# Q + C: MST-based empirical transfer function for C
# =========================================================================

def _diagnose_modularity_and_clustering(adata, perturbation_column,
                                        control_label, knn_k_range,
                                        impact_array=None, pert_labels=None,
                                        n_genes=None):
    """
    Q from Louvain on the LFC-cosine KNN graph (unchanged).
    C from an empirical MST-based transfer function.

    C transfer function:
      For each Leiden community, compare the clustering *inside* that
      community to the clustering that would remain if we kept only the
      minimum spanning tree of the community. The ratio of
      (MST-based-C) / (community-C) tells us what fraction of
      functional-clique clustering is backed by a structural regulatory
      skeleton. Multiply C_core by that empirical ratio to project from
      functional-equivalence clustering to physical-regulatory clustering.

    Stratified sampling on impact quartiles remains.
    """
    MAX_PERTS_FOR_CQ = 500
    MIN_C_FLOOR      = 0.02    # universal: GRNs always have some FFL motifs
    MAX_C_CEILING    = 0.25    # universal: fully-connected graphs are not GRNs

    print("  Computing LFC cosine similarity graph for Q and C...")

    try:
        all_conds     = adata.obs[perturbation_column].unique()
        conditions    = [c for c in all_conds if c != control_label]
        n_total_conds = len(conditions)

        # Stratified quartile sampling
        if len(conditions) > MAX_PERTS_FOR_CQ:
            if impact_array is not None and pert_labels is not None:
                impact_lookup = {str(lb): float(v)
                                 for lb, v in zip(pert_labels, impact_array)}
                scored = sorted(
                    [(c, impact_lookup.get(str(c), 0.0)) for c in conditions],
                    key=lambda x: x[1]
                )
                n_per_q = MAX_PERTS_FOR_CQ // 4
                q_size  = len(scored) // 4
                rng     = np.random.default_rng(42)
                selected = []
                for q in range(4):
                    start = q * q_size
                    end   = (q + 1) * q_size if q < 3 else len(scored)
                    bucket = [c for c, _ in scored[start:end]]
                    n_draw = min(n_per_q, len(bucket))
                    selected.extend(rng.choice(bucket, size=n_draw,
                                               replace=False).tolist())
                conditions = selected
                print(f"  C/Q: quartile-stratified sample of "
                      f"{len(conditions)} from {n_total_conds}.")
            else:
                rng = np.random.default_rng(42)
                conditions = list(rng.choice(
                    conditions, size=MAX_PERTS_FOR_CQ, replace=False))

        ctrl_mask = adata.obs[perturbation_column].values == control_label
        if ctrl_mask.sum() < 3:
            raise ValueError("Too few control cells.")

        print(f"  Computing LFC vectors for {len(conditions)} perturbations...")
        X = (adata.X.toarray() if hasattr(adata.X, 'toarray')
             else np.array(adata.X)).astype(np.float32)
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        log_ctrl  = np.log1p(np.maximum(ctrl_mean, 0))

        lfc_vectors      = []
        valid_conditions = []
        grp_vals         = adata.obs[perturbation_column].values

        for cond in tqdm(conditions, desc="  LFC vectors",
                         unit="pert", ncols=80):
            mask = grp_vals == cond
            if mask.sum() < 2:
                continue
            cond_mean = X[mask].mean(axis=0)
            lfc = np.log1p(np.maximum(cond_mean, 0)) - log_ctrl
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

        norms      = np.linalg.norm(lfc_matrix, axis=1, keepdims=True)
        norms      = np.where(norms < 1e-10, 1e-10, norms)
        lfc_normed = lfc_matrix / norms
        sim_matrix = lfc_normed @ lfc_normed.T
        np.fill_diagonal(sim_matrix, 0)
        sim_matrix = np.clip(sim_matrix, 0, 1)

        import networkx as nx
        Q_values          = []
        C_values          = []
        transfer_ratios   = []

        for k in tqdm(knn_k_range, desc="  KNN Louvain+MST",
                      unit="k", ncols=80):
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
                Q     = nx.community.modularity(G_co, comms)
                if np.isfinite(Q):
                    Q_values.append(Q)
            except Exception:
                comms = None

            try:
                C_core = nx.average_clustering(G_co, weight='weight')
                if np.isfinite(C_core):
                    C_values.append(C_core)
            except Exception:
                C_core = None

            # Empirical MST transfer ratio: per-community, what fraction
            # of clustering survives collapse to minimum spanning tree?
            if comms is not None and C_core is not None:
                try:
                    community_Cs = []
                    mst_Cs       = []
                    for comm in comms:
                        if len(comm) < 4:
                            continue
                        sub = G_co.subgraph(comm).copy()
                        if sub.number_of_edges() < 3:
                            continue
                        community_Cs.append(nx.average_clustering(sub,
                                                                  weight='weight'))
                        # MST on inverse similarity (treat high sim as short edge)
                        sub_inv = sub.copy()
                        for u, v, d in sub_inv.edges(data=True):
                            d['weight'] = 1.0 - d['weight'] + 1e-6
                        mst = nx.minimum_spanning_tree(sub_inv, weight='weight')
                        # Restore similarity weights on MST edges
                        mst_sim = nx.Graph()
                        mst_sim.add_nodes_from(mst.nodes())
                        for u, v in mst.edges():
                            mst_sim.add_edge(u, v,
                                             weight=G_co[u][v]['weight'])
                        mst_Cs.append(nx.average_clustering(mst_sim,
                                                            weight='weight'))
                    if len(community_Cs) > 0:
                        # Ratio of MST-preserved to raw-clique clustering
                        numer = np.mean(mst_Cs) if len(mst_Cs) > 0 else 0.0
                        denom = np.mean(community_Cs)
                        if denom > 1e-6:
                            ratio = numer / denom
                            # Add a universal floor: real GRNs have some
                            # beyond-MST structure (FFL motifs) that pure
                            # MST measurement misses. The floor 0.30 is a
                            # universal property of directed graphs with
                            # triangular motifs, not a cell-type parameter.
                            ratio = max(ratio, 0.30)
                            transfer_ratios.append(float(ratio))
                except Exception:
                    pass

        # Q bounds
        if Q_values:
            Q_min = float(min(Q_values))
            Q_max = float(max(Q_values))
            if Q_min > Q_max:
                Q_min, Q_max = Q_max, Q_min
        else:
            Q_min, Q_max = FALLBACK_BOUNDS["Q"]

        # C bounds via empirical transfer function
        if C_values and transfer_ratios:
            C_core_mean     = float(np.mean(C_values))
            C_core_std      = float(np.std(C_values))
            transfer_factor = float(np.mean(transfer_ratios))
            C_projected     = C_core_mean * transfer_factor
            C_spread        = C_core_std * transfer_factor

            C_min = max(0.0, C_projected - C_spread)
            C_max = C_projected + C_spread

            # Universal physical clamp
            C_min = max(C_min, MIN_C_FLOOR)
            C_max = min(C_max, MAX_C_CEILING)
            if C_min >= C_max:
                C_max = C_min + 0.05

            print(f"  C_core={C_core_mean:.4f}  transfer_factor={transfer_factor:.3f}  "
                  f"C_projected={C_projected:.4f}")
        else:
            C_min, C_max = FALLBACK_BOUNDS["C"]

        if len(C_values) >= 2:
            c_cv = np.std(C_values) / max(np.mean(C_values), 1e-6)
            conf = float(np.clip(1.0 - c_cv, 0.15, 1.0))
        else:
            conf = FALLBACK_CONFIDENCE

        return Q_min, Q_max, conf, C_min, C_max, conf

    except Exception as e:
        print(f"  Q/C diagnostic failed: {str(e)[:80]}")
        fb_q = FALLBACK_BOUNDS["Q"]
        fb_c = FALLBACK_BOUNDS["C"]
        return (fb_q[0], fb_q[1], FALLBACK_CONFIDENCE,
                fb_c[0], fb_c[1], FALLBACK_CONFIDENCE)


# =========================================================================
# rho: causal skeleton mask
# =========================================================================

def _diagnose_rho_causal(deg_matrix, raw_sparse_mat, n_genes,
                         n_bootstrap=100, n_jobs=15):
    """
    Rho measured on the causal skeleton: parent_graph ∩ DEG_matrix.

    The parent graph contains all LightGBM-gain candidate edges (poisoned
    by co-expression cliques, which push rho positive).
    The DEG matrix contains all intervention-supported perturbation ->
    DEG pairs (poisoned by downstream avalanches, which inflate in-degree).
    Their elementwise AND keeps only edges supported by BOTH predictive
    gain AND causal evidence, stripping most noise on both sides.

    Rho on this causal subgraph is a direct, dataset-specific, dimensionally-
    consistent estimate of the regulatory graph's assortativity.
    """
    fb = FALLBACK_BOUNDS["rho"]

    if raw_sparse_mat is None or deg_matrix is None:
        print("    rho causal: missing parent graph or DEG matrix -> fallback.")
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2

    try:
        import igraph as ig

        parent = raw_sparse_mat.tocsr()
        deg    = deg_matrix.tocsr()

        # Parent graph binary pattern
        parent_bin = (parent != 0).astype(np.int8)
        # Mask: keep parent edges that are also in deg_matrix
        causal = parent_bin.multiply(deg)
        causal = causal.tocoo()
        causal.eliminate_zeros()

        n_edges = causal.nnz
        print(f"    rho causal: parent {parent.nnz:,} edges, "
              f"DEG {deg.nnz:,} edges, intersection {n_edges:,} edges.")

        if n_edges < 1000:
            print(f"    rho causal: intersection too small ({n_edges}), "
                  "falling back to parent-graph rho.")
            # Fall back: rho on parent graph alone (v7.3 behaviour)
            pcoo    = parent.tocoo()
            n_samp  = min(200000, pcoo.nnz)
            rng     = np.random.default_rng(42)
            idx     = rng.choice(pcoo.nnz, size=n_samp, replace=False)
            G       = ig.Graph(n=n_genes,
                               edges=list(zip(pcoo.row[idx].tolist(),
                                              pcoo.col[idx].tolist())),
                               directed=True)
            rho_val = float(G.assortativity_degree(directed=True))
            if not np.isfinite(rho_val):
                rho_val = -0.10
            return (rho_val - 0.05, rho_val + 0.05,
                    FALLBACK_CONFIDENCE * 0.5, rho_val)

        # Measure rho on the full causal intersection
        G_full = ig.Graph(n=n_genes,
                          edges=list(zip(causal.row.tolist(),
                                         causal.col.tolist())),
                          directed=True)
        rho_center = float(G_full.assortativity_degree(directed=True))
        if not np.isfinite(rho_center):
            rho_center = -0.10
        print(f"    rho_causal_center: {rho_center:.4f}")

        # Bootstrap confidence via edge resampling
        seeds = np.random.default_rng(42).integers(
            0, 2**31, size=n_bootstrap).tolist()

        def boot_worker(seed):
            rng_b = np.random.default_rng(seed)
            idx   = rng_b.choice(n_edges, size=n_edges, replace=True)
            try:
                Gb = ig.Graph(n=n_genes,
                              edges=list(zip(causal.row[idx].tolist(),
                                             causal.col[idx].tolist())),
                              directed=True)
                r = float(Gb.assortativity_degree(directed=True))
                return r if np.isfinite(r) else None
            except Exception:
                return None

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(boot_worker)(s) for s in
            tqdm(seeds, desc="    rho bootstrap", unit="iter", ncols=80)
        )
        boot_rhos = [r for r in results if r is not None]

        if len(boot_rhos) < 10:
            return (rho_center - 0.05, rho_center + 0.05,
                    FALLBACK_CONFIDENCE, rho_center)

        boot_sigma = float(np.std(boot_rhos))
        print(f"    boot_sigma: {boot_sigma:.4f}")

        # Universal physical clamp: real GRNs are at least weakly
        # disassortative. An observed rho > 0 from the causal skeleton
        # likely still has residual co-expression noise, so we shift the
        # upper bound into the physically plausible range.
        rho_lower = rho_center - max(boot_sigma * 2.0, 0.05)
        rho_upper = rho_center + max(boot_sigma * 1.5, 0.03)

        # Confidence from sigma relative to magnitude
        conf = float(np.clip(
            1.0 - (boot_sigma / max(abs(rho_center), 0.05)), 0.10, 1.0))

        return rho_lower, rho_upper, conf, rho_center

    except Exception as e:
        print(f"    rho causal failed: {str(e)[:80]}")
        return fb[0], fb[1], FALLBACK_CONFIDENCE, (fb[0] + fb[1]) / 2


# =========================================================================
# Bound enforcement + weight normalization
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
        print(f"  WARNING: {bad.sum()} confidence values NaN -> fallback.")
        values[bad] = FALLBACK_CONFIDENCE
    values = np.clip(values, 0.0, 1.0)
    total  = values.sum()
    if total < 1e-10:
        print("  WARNING: all confidences ~0; using uniform weights.")
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
# Master runner
# =========================================================================

def run_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                    raw_sparse_mat=None):
    """
    Full Phase 0 pipeline (v7.4).
    """
    print("=" * 72)
    print("FUNGI v7.4 -- Phase 0 Diagnostic Calibration")
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
    n_jobs    = cfg_diagnostics.get("n_jobs", 15)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    # Step 1: impact array + DEG matrix
    impact_array, pert_labels, sample_weights, deg_matrix = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method=cfg_diagnostics["de_method"],
        pval_threshold=cfg_diagnostics["de_pval_threshold"],
        lfc_threshold=cfg_diagnostics["de_lfc_threshold"],
        n_jobs=n_jobs,
        max_perts_for_de=max_perts,
        is_metacell=is_metacell,
        metacell_pooling_factor=mc_pool,
    )

    # Step 2: Gini (shift + universal width)
    print("\n  Diagnosing Gini...")
    gini_min, gini_max, gini_conf, gini_center = _diagnose_gini(
        impact_array, n_boot, is_metacell, sample_weights=sample_weights)

    # Step 3: alpha (diffusion-shifted + sigma-width, clipped)
    print("  Diagnosing alpha...")
    alpha_min, alpha_max, alpha_conf, alpha_center = _diagnose_alpha(
        impact_array, n_boot)

    # Step 4: S_max (unchanged)
    print("  Diagnosing S_max...")
    smax_min, smax_max, smax_conf, smax_center = _diagnose_smax(
        impact_array, n_genes, is_metacell, mc_pool,
        sample_weights=sample_weights)

    # Step 5: Q and C (Q unchanged, C via MST transfer function)
    print("  Diagnosing Q and C...")
    Q_min, Q_max, Q_conf, C_min, C_max, C_conf = \
        _diagnose_modularity_and_clustering(
            adata, pert_col, ctrl_label, knn_range,
            impact_array=impact_array, pert_labels=pert_labels,
            n_genes=n_genes)

    # Step 6: rho (causal skeleton)
    print("  Diagnosing rho (causal skeleton)...")
    rho_min, rho_max, rho_conf, rho_center = _diagnose_rho_causal(
        deg_matrix, raw_sparse_mat, n_genes,
        n_bootstrap=n_boot, n_jobs=n_jobs)

    # Step 7: raw report
    print("\n  Raw bounds before constraint enforcement:")
    for name, (lo, hi, cf) in [
        ("alpha", (alpha_min, alpha_max, alpha_conf)),
        ("C",     (C_min,     C_max,     C_conf)),
        ("Q",     (Q_min,     Q_max,     Q_conf)),
        ("gini",  (gini_min,  gini_max,  gini_conf)),
        ("S_max", (smax_min,  smax_max,  smax_conf)),
        ("rho",   (rho_min,   rho_max,   rho_conf)),
    ]:
        print(f"    {name:>6s}: [{lo:.4f}, {hi:.4f}]  (conf={cf:.3f})")

    # Step 8: enforce constraints
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

    # Step 9: weights
    raw_confidences = {
        "alpha": alpha_conf, "gini": gini_conf, "S_max": smax_conf,
        "Q": Q_conf, "C": C_conf, "rho": rho_conf,
    }
    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)

    # Step 10: NaN guard
    for param in utopian_bounds:
        for i in range(2):
            if not np.isfinite(utopian_bounds[param][i]):
                utopian_bounds[param][i] = FALLBACK_BOUNDS[param][i]
    for param in loss_weights:
        if not np.isfinite(loss_weights[param]):
            loss_weights[param] = w_floor

    # Step 11: report
    diagnostic_report = {
        "impact_array_size":  len(impact_array),
        "impact_range": (
            [float(impact_array.min()), float(impact_array.max())]
            if len(impact_array) > 0 else [0, 0]
        ),
        "deg_matrix_nnz":      int(deg_matrix.nnz),
        "is_metacell":         is_metacell,
        "n_cells_used_for_de": int(adata.n_obs),
        "gini_center":         float(gini_center),
        "alpha_center":        float(alpha_center),
        "rho_center":          float(rho_center),
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
# Shatter config builder (unchanged interface)
# =========================================================================

def build_shatter_config(cfg_shatter, n_genes, utopian_bounds,
                         lambda_search_bounds):
    """
    Builds dynamic shatter thresholds. Interface preserved for downstream
    compatibility even though the engine's shatter logic is being phased out.
    """
    lambda_min   = cfg_shatter.get("lambda_min", 2.0)
    lambda_max   = cfg_shatter.get("lambda_max", 30.0)
    s_max_mult   = cfg_shatter.get("s_max_ceiling_multiplier", 2.0)
    s_max_cap    = cfg_shatter.get("s_max_ceiling_hard_cap", 0.30)
    clust_gamma  = cfg_shatter.get("min_clustering_gamma", 1.5)

    s_max_ceiling = min(
        round(utopian_bounds["S_max"][1] * s_max_mult, 4),
        s_max_cap
    )

    if lambda_search_bounds is None or lambda_search_bounds[0] is None:
        lambda_lo_density = lambda_min / n_genes
        lambda_hi_density = lambda_max / n_genes
    else:
        lambda_lo_density = lambda_search_bounds[0]
        lambda_hi_density = lambda_search_bounds[1]

    lambda_expected_abs = (lambda_lo_density + lambda_hi_density) / 2 * n_genes
    min_clustering      = round(clust_gamma * (lambda_expected_abs / n_genes), 6)

    return {
        "max_orphan_fraction": cfg_shatter.get("max_orphan_fraction", 0.15),
        "min_gwcc_fraction":   cfg_shatter.get("min_gwcc_fraction", 0.35),
        "max_hub_saturation":  s_max_ceiling,
        "min_edge_count":      int(lambda_min * n_genes),
        "max_edge_count":      int(lambda_max * n_genes),
        "min_clustering":      min_clustering,
    }
