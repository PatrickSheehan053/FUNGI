"""
FUNGI v6 -- DASH Kernel Engine

Core computation module containing:
    - Dynamic topology (per-edge FFL motif participation)
    - Dual-pass pruning (local guardrails + global fill)
    - Expanded shatter criteria (8 conditions)
    - Data-driven utopia loss function (custom bounds and weights)
    - Single-graph evaluation entry point (designed for Ray remote calls)

Changes from v5:
    - Utopia loss now accepts dynamic bounds and weights from Phase 0.
    - Archetype weight system removed entirely.
    - Shatter criteria expanded: spectral dominance, Moran's I, alpha bounds,
      clustering floor, density ceiling, GWCC fraction.
    - k_core bounds corrected to [1.0, 15.0] (linear scaling).
    - All print statements removed from hot-path functions for Ray compat.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import powerlaw
import igraph as ig
import warnings
import graphblas as gb

from graph_utils import compute_spectral_dominance_ratio, compute_morans_i

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================================
# Dynamic Topology (Per-Edge T_local from Feed-Forward Loops)
# =========================================================================

def compute_dynamic_topology(W_sorted, sources_sorted, targets_sorted, k_core, n_genes):
    """Calculates T_local using 100% GraphBLAS and 1D NumPy Binary Searches."""
    target_edges = int(n_genes * k_core)
    target_edges = min(target_edges, len(W_sorted))
    
    T_local = np.zeros(len(W_sorted), dtype=np.float64)
    if target_edges < 1:
        return T_local

    core_rows = sources_sorted[:target_edges]
    core_cols = targets_sorted[:target_edges]

    # 1. Build GraphBLAS Matrix
    A_core_gb = gb.Matrix.from_coo(
        core_rows, core_cols, np.ones(target_edges, dtype=np.float64), 
        nrows=n_genes, ncols=n_genes
    )

    # 2. Bare-Metal Masked FFL Calculation
    T_matrix_gb = gb.semiring.plus_times(A_core_gb @ A_core_gb).new(mask=A_core_gb.S)

    # 3. Native GraphBLAS Reductions for Degrees (NO SCIPY!)
    # Let the C-library sum the rows and columns instantly
    d_out = np.zeros(n_genes, dtype=np.float64)
    out_idx, out_vals = A_core_gb.reduce_rowwise(gb.monoid.plus).new().to_coo()
    d_out[out_idx] = out_vals

    d_in = np.zeros(n_genes, dtype=np.float64)
    in_idx, in_vals = A_core_gb.reduce_columnwise(gb.monoid.plus).new().to_coo()
    d_in[in_idx] = in_vals

    # 4. Lightning-Fast 1D Flat Index Mapping (NO SCIPY!)
    t_rows, t_cols, z_values = T_matrix_gb.to_coo()
    
    if len(z_values) > 0:
        # Compress 2D coordinates into 1D flat indices for instant lookups
        core_flat = core_rows.astype(np.int64) * n_genes + core_cols
        t_flat = t_rows.astype(np.int64) * n_genes + t_cols
        
        # GraphBLAS outputs in sorted order, so a binary search takes microseconds
        sort_idx = np.searchsorted(t_flat, core_flat)
        
        # Protect against out-of-bounds indices
        valid_idx = np.clip(sort_idx, 0, len(t_flat) - 1)
        valid_mask = (t_flat[valid_idx] == core_flat)
        
        mapped_z = np.zeros(target_edges, dtype=np.float64)
        mapped_z[valid_mask] = z_values[valid_idx[valid_mask]]
    else:
        mapped_z = np.zeros(target_edges, dtype=np.float64)

    # 5. Calculate Local Motif Participation
    local_cap = np.minimum(d_out[core_rows], d_in[core_cols])
    valid = (mapped_z > 0) & (local_cap > 0)
    
    T_local[:target_edges][valid] = mapped_z[valid] / local_cap[valid]
    np.clip(T_local, 0.0, 1.0, out=T_local)
    
    return T_local

# =========================================================================
# Dual-Pass Pruning
# =========================================================================

def dual_pass_pruning(omega_scores, W, sources, targets, perturbed_nodes,
                      n_genes, lam, K=3):
    """Two-pass edge selection:
    Pass 1: Local guardrails protect top-K edges from each perturbed node.
    Pass 2: Global scale-free fill uses remaining budget on highest omega.
    """
    total_budget = int(np.round(n_genes * lam))

    # Pass 1: local guardrails
    protected_indices = []
    for p_node in perturbed_nodes:
        p_edges = np.where(sources == p_node)[0]
        if len(p_edges) > 0:
            k_actual = min(K, len(p_edges))
            top_k_idx = p_edges[np.argsort(omega_scores[p_edges])[-k_actual:]]
            protected_indices.extend(top_k_idx)

    protected_indices = np.unique(protected_indices)

    # Pass 2: global fill
    remaining_budget = total_budget - len(protected_indices)

    if remaining_budget > 0:
        mask = np.ones(len(omega_scores), dtype=bool)
        mask[protected_indices] = False
        available_indices = np.where(mask)[0]

        if remaining_budget < len(available_indices):
            top_remaining_idx_rel = np.argpartition(
                omega_scores[available_indices], -remaining_budget
            )[-remaining_budget:]
            top_remaining_idx = available_indices[top_remaining_idx_rel]
        else:
            top_remaining_idx = available_indices

        final_indices = np.concatenate([protected_indices, top_remaining_idx])
    else:
        final_indices = protected_indices[:total_budget]

    return sources[final_indices], targets[final_indices], W[final_indices]


# =========================================================================
# Expanded Shatter Criteria
# =========================================================================

def check_shatter(surviving_sources, surviving_targets, surviving_W,
                  out_degrees, n_genes, active_nodes, shatter_cfg):
    """Evaluates all shatter conditions in fail-fast order (cheapest first).

    Returns (is_shattered: bool, reason: str or None).
    """
    n_edges = len(surviving_sources)

    # 1. Density collapse (the hairball)
    max_edges = shatter_cfg.get("max_edge_count", 500000)
    if n_edges > max_edges:
        return True, "density_collapse"

    # 2. Orphan collapse (too many disconnected genes)
    max_orphan = shatter_cfg.get("max_orphan_fraction", 0.70)
    orphan_fraction = (n_genes - active_nodes) / n_genes if n_genes > 0 else 1.0
    if orphan_fraction > max_orphan:
        return True, "orphan_collapse"

    # 3. Dictator hub (S_max)
    max_hub_sat = shatter_cfg.get("max_hub_saturation", 0.15)
    s_max = (np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0
    if s_max > max_hub_sat:
        return True, "dictator_hub"

    # 4. GWCC percolation
    min_gwcc = shatter_cfg.get("min_gwcc_fraction", 0.30)
    if n_edges > 0 and active_nodes > 0:
        sparse_G = sp.coo_matrix(
            (np.ones(n_edges), (surviving_sources, surviving_targets)),
            shape=(n_genes, n_genes)
        )
        _, labels = connected_components(csgraph=sparse_G, directed=False,
                                         return_labels=True)
        gwcc_size = np.bincount(labels).max()
        gwcc_fraction = gwcc_size / n_genes
        if gwcc_fraction < min_gwcc:
            return True, "gwcc_percolation"
    else:
        gwcc_fraction = 0.0
        if min_gwcc > 0:
            return True, "gwcc_percolation"

    # 5. Clustering collapse
    min_clustering = shatter_cfg.get("min_clustering", 0.01)
    if n_edges > 50:
        try:
            edges = list(zip(surviving_sources.tolist(), surviving_targets.tolist()))
            ig_g = ig.Graph(n=n_genes, edges=edges, directed=True)
            ig_u = ig_g.as_undirected(mode="collapse")
            C_obs = ig_u.transitivity_undirected()
            if not np.isnan(C_obs) and C_obs < min_clustering:
                return True, "clustering_collapse"
        except Exception:
            pass

    # 6. Spectral masking (compute only if graph is large enough)
    max_spectral = shatter_cfg.get("max_spectral_dominance_ratio", 50.0)
    if n_edges > 100:
        ratio = compute_spectral_dominance_ratio(
            surviving_sources, surviving_targets, surviving_W, n_genes
        )
        if ratio > max_spectral:
            return True, "spectral_masking"

    # 7. Moran's I spatial coherence
    min_morans = shatter_cfg.get("min_morans_i", 0.02)
    if n_edges > 50 and active_nodes > 10:
        morans = compute_morans_i(
            surviving_sources, surviving_targets, out_degrees, n_genes
        )
        if morans < min_morans:
            return True, "texture_collapse"

    return False, None


# =========================================================================
# Data-Driven Utopia Loss Function
# =========================================================================

def calculate_utopia_loss(surviving_sources, surviving_targets, surviving_W,
                          n_genes, out_degrees, active_nodes, kappa,
                          utopian_bounds, loss_weights):
    """Calculates the Euclidean distance from the utopian target zone.

    Unlike v5, this version uses dynamically computed bounds and weights
    from Phase 0 diagnostics rather than fixed archetype weights.

    Parameters
    ----------
    utopian_bounds : dict
        {param: [min, max]} from Phase 0.
    loss_weights : dict
        {param: weight} from Phase 0.
    """
    n_edges = len(surviving_sources)

    # ---- Alpha (Scale-Free Shape) ----
    alpha_obs = 1.0
    if len(out_degrees) > 10 and np.max(out_degrees) > 1:
        try:
            fit = powerlaw.Fit(out_degrees, xmin=1, discrete=True, verbose=False)
            alpha_obs = fit.power_law.alpha
        except Exception:
            pass

    b = utopian_bounds["alpha"]
    w = loss_weights["alpha"]
    term_alpha = 0.0
    if alpha_obs < b[0]:
        term_alpha = w * ((b[0] - alpha_obs) / max(b[0], 1e-6)) ** 2
    elif alpha_obs > b[1]:
        term_alpha = w * ((alpha_obs - b[1]) / max(b[1], 1e-6)) ** 2

    # ---- Gini (Hierarchical Flow) ----
    G_obs = 1.0
    if active_nodes > 1 and np.sum(out_degrees) > 0:
        sorted_deg = np.sort(out_degrees)
        n = len(out_degrees)
        G_obs = (2.0 * np.sum(np.arange(1, n + 1) * sorted_deg)) / \
                (n * np.sum(sorted_deg)) - (n + 1) / n

    b = utopian_bounds["gini"]
    w = loss_weights["gini"]
    term_G = 0.0
    if G_obs < b[0]:
        term_G = w * ((b[0] - G_obs) / max(b[0], 1e-6)) ** 2
    elif G_obs > b[1]:
        term_G = w * ((G_obs - b[1]) / max(b[1], 1e-6)) ** 2

    # ---- S_max (Biophysical Constraint) ----
    max_degree = np.max(out_degrees) if len(out_degrees) > 0 else 0
    s_max = max_degree / n_genes
    relu_val = max(0.0, (s_max - kappa) / max(kappa, 1e-6))
    b = utopian_bounds["S_max"]
    w = loss_weights["S_max"]
    term_sat = w * relu_val ** 2

    # Default penalties (overwritten if graph is large enough for igraph)
    C_obs, Q_obs, rho_obs = 0.0, 0.0, 1.0
    term_C = loss_weights["C"] * 1.0
    term_Q = loss_weights["Q"] * 1.0
    term_rho = loss_weights["rho"] * 1.0

    if n_edges > 100:
        try:
            edges = list(zip(surviving_sources.tolist(),
                             surviving_targets.tolist()))
            ig_graph = ig.Graph(n=n_genes, edges=edges, directed=True,
                                edge_attrs={'weight': surviving_W.tolist()})

            # Assortativity
            rho_calc = ig_graph.assortativity_degree(directed=True)
            if not np.isnan(rho_calc):
                rho_obs = rho_calc
                b = utopian_bounds["rho"]
                w = loss_weights["rho"]
                term_rho = 0.0
                if rho_obs < b[0]:
                    term_rho = w * ((b[0] - rho_obs) / max(abs(b[0]), 1e-6)) ** 2
                elif rho_obs > b[1]:
                    term_rho = w * ((rho_obs - b[1]) / max(abs(b[1]), 1e-6)) ** 2

            # Clustering and Modularity (undirected projection)
            ig_undirected = ig_graph.as_undirected(
                mode="collapse", combine_edges=dict(weight="sum"))

            C_calc = ig_undirected.transitivity_undirected()
            if not np.isnan(C_calc):
                C_obs = C_calc
                b = utopian_bounds["C"]
                w = loss_weights["C"]
                term_C = 0.0
                if C_obs < b[0]:
                    term_C = w * ((b[0] - C_obs) / max(b[0], 1e-6)) ** 2
                elif C_obs > b[1]:
                    term_C = w * ((C_obs - b[1]) / max(b[1], 1e-6)) ** 2

            partition = ig_undirected.community_multilevel()
            Q_obs = partition.modularity
            b = utopian_bounds["Q"]
            w = loss_weights["Q"]
            term_Q = 0.0
            if Q_obs < b[0]:
                term_Q = w * ((b[0] - Q_obs) / max(b[0], 1e-6)) ** 2
            elif Q_obs > b[1]:
                term_Q = w * ((Q_obs - b[1]) / max(b[1], 1e-6)) ** 2

        except Exception:
            pass

    L_utopia = np.sqrt(
        term_alpha + term_C + term_Q + term_G + term_rho + term_sat
    )

    topo_metrics = {
        'alpha': alpha_obs,
        'C': C_obs,
        'Q': Q_obs,
        'Gini': G_obs,
        'rho': rho_obs,
        'S_max': s_max,
    }

    return L_utopia, topo_metrics


# =========================================================================
# DASH Kernel (Single Parameter Set)
# =========================================================================

def run_dash_and_score(params, W, D, sources, targets, G_baseline_coo,
                       n_genes, perturbed_nodes, utopian_bounds,
                       loss_weights, shatter_cfg):
    """Full pipeline for one parameter set: compute omega, prune, triage,
    and score against the data-driven utopian loss.

    This function is designed to be called inside a Ray remote worker.
    It contains no print statements in the hot path.
    """
    beta, gamma, delta, kappa, k_core, lam = params

    # Bypass the COO matrix entirely and pass the pre-sorted arrays!
    T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
    epsilon = 1e-6

    # Dynamic baseline anchoring for hub penalty
    target_baseline_density = 0.0055
    lambda_baseline = n_genes * target_baseline_density
    gamma_dynamic = gamma * (lam / lambda_baseline)

    # DASH edge scoring
    numerator = (W ** beta) * (1 + delta * T_local)
    denominator = (np.log1p(D) + epsilon) ** gamma_dynamic
    omega_scores = numerator / denominator

    # Dual-pass pruning
    surviving_sources, surviving_targets, surviving_W = dual_pass_pruning(
        omega_scores, W, sources, targets, perturbed_nodes, n_genes, lam, K=3
    )

    # Compute basic degree statistics
    out_degrees = np.bincount(surviving_sources, minlength=n_genes)
    in_degrees = np.bincount(surviving_targets, minlength=n_genes)
    all_degrees = out_degrees + in_degrees
    active_nodes = np.count_nonzero(all_degrees > 0)

    # Fail-fast shatter check
    is_shattered, shatter_reason = check_shatter(
        surviving_sources, surviving_targets, surviving_W,
        out_degrees, n_genes, active_nodes, shatter_cfg
    )

    if is_shattered:
        s_max = (np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0
        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': 999.0,
            'is_shattered': 1, 'shatter_reason': shatter_reason,
            'n_edges': len(surviving_sources),
            'active_nodes': active_nodes,
            'alpha': 1.0, 'Gini': 1.0, 'rho': 1.0,
            'C': 0.0, 'Q': 0.0, 'S_max': s_max,
        }

    # Full utopia evaluation
    utopia_loss, topo_metrics = calculate_utopia_loss(
        surviving_sources, surviving_targets, surviving_W,
        n_genes, out_degrees, active_nodes, kappa,
        utopian_bounds, loss_weights
    )

    # GWCC percolation penalty (additive)
    sparse_G = sp.coo_matrix(
        (np.ones(len(surviving_W)), (surviving_sources, surviving_targets)),
        shape=(n_genes, n_genes)
    )
    _, labels = connected_components(csgraph=sparse_G, directed=False,
                                     return_labels=True)
    gwcc_size = np.bincount(labels).max()
    gwcc_fraction = gwcc_size / n_genes

    gwcc_target = 0.45
    gwcc_penalty = 0.0
    if gwcc_fraction < gwcc_target:
        gwcc_penalty = 8.0 * ((gwcc_target - gwcc_fraction) / gwcc_target) ** 2

    utopia_loss = np.sqrt(utopia_loss ** 2 + gwcc_penalty)

    return {
        'beta': beta, 'gamma': gamma, 'delta': delta,
        'kappa': kappa, 'k_core': k_core, 'lambda': lam,
        'utopia_loss': utopia_loss,
        'is_shattered': 0, 'shatter_reason': None,
        'n_edges': len(surviving_sources),
        'active_nodes': active_nodes,
        'gwcc_fraction': gwcc_fraction,
        'alpha': topo_metrics['alpha'], 'Gini': topo_metrics['Gini'],
        'rho': topo_metrics['rho'], 'C': topo_metrics['C'],
        'Q': topo_metrics['Q'], 'S_max': topo_metrics['S_max'],
    }
