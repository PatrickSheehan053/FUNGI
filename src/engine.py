"""
FUNGI v7 -- DASH Kernel Engine

v7 changes from v6:
    - EVERY topological metric wrapped in try/except with maximum penalty
      fallback. No function in this module can return NaN.
    - GraphBLAS (python-graphblas) used for SpGEMM triangle calculation,
      with scipy.sparse fallback if not installed.
    - Assortativity, Gini, alpha: explicit zero-variance guards.
    - Shatter check returns 999.0 penalty (not NaN) for broken graphs.
    - All intermediate float values sanitized via _safe_val.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import igraph as ig
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================
# GraphBLAS availability check
# =========================================================================

_GRAPHBLAS_AVAILABLE = False
try:
    import graphblas as gb
    from graphblas import Matrix, binary, semiring
    _GRAPHBLAS_AVAILABLE = True
except ImportError:
    pass


def _safe_val(x, fallback=0.0):
    """Returns x if finite, else fallback."""
    if x is None or not np.isfinite(x):
        return fallback
    return float(x)


# =========================================================================
# Dynamic Topology: GraphBLAS or scipy fallback
# =========================================================================

def compute_dynamic_topology(G_baseline_coo, k_core, n_genes):
    """Per-edge local motif participation ratio (T_local).

    v7: Uses GraphBLAS semiring for the SpGEMM if available.
    Falls back to scipy.sparse if python-graphblas is not installed.
    """
    target_edges = int(n_genes * k_core)
    target_edges = min(target_edges, len(G_baseline_coo.data))

    if target_edges < 1:
        return np.zeros(len(G_baseline_coo.data), dtype=np.float64)

    surviving_indices = np.argpartition(
        G_baseline_coo.data, -target_edges
    )[-target_edges:]

    core_rows = G_baseline_coo.row[surviving_indices]
    core_cols = G_baseline_coo.col[surviving_indices]

    if _GRAPHBLAS_AVAILABLE:
        T_local = _topology_graphblas(
            core_rows, core_cols, surviving_indices,
            G_baseline_coo, n_genes, target_edges
        )
    else:
        T_local = _topology_scipy(
            core_rows, core_cols, surviving_indices,
            G_baseline_coo, n_genes, target_edges
        )

    np.clip(T_local, 0.0, 1.0, out=T_local)
    return T_local


def _topology_graphblas(core_rows, core_cols, surviving_indices,
                        G_baseline_coo, n_genes, target_edges):
    """SpGEMM triangle calculation using GraphBLAS semirings."""
    # Build binary core matrix in GraphBLAS
    A_core = Matrix.from_coo(
        core_rows.astype(np.uint64),
        core_cols.astype(np.uint64),
        np.ones(target_edges, dtype=np.int64),
        nrows=n_genes, ncols=n_genes
    )

    # Masked SpGEMM: A_core @ A_core, keeping only entries where A_core exists
    # This is the key optimization: GraphBLAS skips zero-by-zero multiplies
    # and the mask prevents fill-in, all at C-level speed outside the GIL.
    T_matrix = A_core.mxm(A_core, semiring.plus_times).new(mask=A_core.S)

    # Extract triangle counts for core edges
    z_rows, z_cols, z_vals = T_matrix.to_coo()

    # Build lookup for surviving edges
    T_local = np.zeros(len(G_baseline_coo.data), dtype=np.float64)

    # Map z_values back to the baseline edge indices
    z_dict = {}
    for r, c, v in zip(z_rows, z_cols, z_vals):
        z_dict[(int(r), int(c))] = float(v)

    d_out = np.zeros(n_genes, dtype=np.float64)
    d_in = np.zeros(n_genes, dtype=np.float64)
    for r, c in zip(core_rows, core_cols):
        d_out[r] += 1
        d_in[c] += 1

    for idx, (r, c) in zip(surviving_indices, zip(core_rows, core_cols)):
        z = z_dict.get((int(r), int(c)), 0.0)
        if z > 0:
            local_cap = min(d_out[r], d_in[c])
            if local_cap > 0:
                T_local[idx] = z / local_cap

    return T_local


def _topology_scipy(core_rows, core_cols, surviving_indices,
                    G_baseline_coo, n_genes, target_edges):
    """SpGEMM triangle calculation using scipy.sparse (fallback)."""
    A_core = sp.coo_matrix(
        (np.ones(target_edges, dtype=np.int32), (core_rows, core_cols)),
        shape=(n_genes, n_genes)
    ).tocsr()

    T_matrix = A_core.dot(A_core).multiply(A_core)

    T_local = np.zeros(len(G_baseline_coo.data), dtype=np.float64)
    z_values = T_matrix[core_rows, core_cols].A1

    d_out = np.array(A_core.sum(axis=1)).ravel()
    d_in = np.array(A_core.sum(axis=0)).ravel()

    local_cap = np.minimum(d_out[core_rows], d_in[core_cols])
    valid = (z_values > 0) & (local_cap > 0)
    T_local[surviving_indices[valid]] = z_values[valid] / local_cap[valid]

    return T_local


# =========================================================================
# Dual-Pass Pruning (unchanged from v6)
# =========================================================================

def dual_pass_pruning(omega_scores, W, sources, targets, perturbed_nodes,
                      n_genes, lam, K=3):
    """Two-pass edge selection."""
    total_budget = int(np.round(n_genes * lam))

    protected_indices = []
    for p_node in perturbed_nodes:
        p_edges = np.where(sources == p_node)[0]
        if len(p_edges) > 0:
            k_actual = min(K, len(p_edges))
            top_k_idx = p_edges[np.argsort(omega_scores[p_edges])[-k_actual:]]
            protected_indices.extend(top_k_idx)

    protected_indices = np.unique(protected_indices)

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
    """Fail-fast shatter check. Returns (is_shattered, reason)."""
    n_edges = len(surviving_sources)

    # 1. Density collapse
    if n_edges > shatter_cfg.get("max_edge_count", 500000):
        return True, "density_collapse"

    # 2. Orphan collapse
    orphan_frac = (n_genes - active_nodes) / max(n_genes, 1)
    if orphan_frac > shatter_cfg.get("max_orphan_fraction", 0.70):
        return True, "orphan_collapse"

    # 3. Dictator hub
    s_max = (np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0
    if s_max > shatter_cfg.get("max_hub_saturation", 0.15):
        return True, "dictator_hub"

    # 4. GWCC percolation
    if n_edges > 0 and active_nodes > 0:
        try:
            sparse_G = sp.coo_matrix(
                (np.ones(n_edges), (surviving_sources, surviving_targets)),
                shape=(n_genes, n_genes)
            )
            _, labels = connected_components(csgraph=sparse_G, directed=False,
                                             return_labels=True)
            gwcc_size = np.bincount(labels).max()
            gwcc_fraction = gwcc_size / n_genes
            if gwcc_fraction < shatter_cfg.get("min_gwcc_fraction", 0.30):
                return True, "gwcc_percolation"
        except Exception:
            return True, "gwcc_percolation"

    # 5. Clustering collapse
    if n_edges > 50:
        try:
            edges = list(zip(surviving_sources.tolist(), surviving_targets.tolist()))
            ig_g = ig.Graph(n=n_genes, edges=edges, directed=True)
            ig_u = ig_g.as_undirected(mode="collapse")
            C_obs = ig_u.transitivity_undirected()
            if np.isfinite(C_obs) and C_obs < shatter_cfg.get("min_clustering", 0.01):
                return True, "clustering_collapse"
        except Exception:
            pass

    # 6. Spectral masking
    if n_edges > 100:
        try:
            from graph_utils import compute_spectral_dominance_ratio
            ratio = compute_spectral_dominance_ratio(
                surviving_sources, surviving_targets, surviving_W, n_genes
            )
            if np.isfinite(ratio) and ratio > shatter_cfg.get("max_spectral_dominance_ratio", 50.0):
                return True, "spectral_masking"
        except Exception:
            pass

    # 7. Moran's I
    if n_edges > 50 and active_nodes > 10:
        try:
            from graph_utils import compute_morans_i
            morans = compute_morans_i(
                surviving_sources, surviving_targets, out_degrees, n_genes
            )
            if np.isfinite(morans) and morans < shatter_cfg.get("min_morans_i", 0.02):
                return True, "texture_collapse"
        except Exception:
            pass

    return False, None


# =========================================================================
# NaN-Proof Utopia Loss Function
# =========================================================================

def calculate_utopia_loss(surviving_sources, surviving_targets, surviving_W,
                          n_genes, out_degrees, active_nodes, kappa,
                          utopian_bounds, loss_weights):
    """Calculates Euclidean distance from the utopian target zone.

    v7: Every single metric calculation is individually guarded.
    If a metric computation crashes (divide-by-zero, NaN from degenerate
    degree distributions), the function assigns a maximum penalty for
    that specific term rather than returning NaN for the whole loss.
    """
    n_edges = len(surviving_sources)

    def _penalty(param_name, observed):
        """Computes the penalty for one parameter against its bounds."""
        b = utopian_bounds[param_name]
        w = _safe_val(loss_weights[param_name], 1.0)
        obs = _safe_val(observed, 0.0)

        if obs < b[0]:
            denom = max(abs(b[0]), 1e-6)
            return w * ((b[0] - obs) / denom) ** 2
        elif obs > b[1]:
            denom = max(abs(b[1]), 1e-6)
            return w * ((obs - b[1]) / denom) ** 2
        return 0.0

    # ---- Alpha ----
    alpha_obs = 1.0
    try:
        if len(out_degrees) > 10 and np.max(out_degrees) > 1:
            unique_degrees = np.unique(out_degrees[out_degrees > 0])
            if len(unique_degrees) >= 3:
                import powerlaw
                fit = powerlaw.Fit(out_degrees[out_degrees > 0],
                                   xmin=1, discrete=True, verbose=False)
                val = fit.power_law.alpha
                alpha_obs = _safe_val(val, 1.0)
    except Exception:
        alpha_obs = 1.0
    term_alpha = _penalty("alpha", alpha_obs)

    # ---- Gini ----
    G_obs = 1.0
    try:
        if active_nodes > 1:
            deg_sum = np.sum(out_degrees)
            if deg_sum > 0:
                sorted_deg = np.sort(out_degrees)
                n = len(out_degrees)
                G_obs = (2.0 * np.sum(np.arange(1, n + 1) * sorted_deg)) / \
                        (n * deg_sum) - (n + 1) / n
                G_obs = _safe_val(G_obs, 1.0)
    except Exception:
        G_obs = 1.0
    term_G = _penalty("gini", G_obs)

    # ---- S_max ----
    s_max = 0.0
    try:
        max_degree = int(np.max(out_degrees)) if len(out_degrees) > 0 else 0
        s_max = max_degree / max(n_genes, 1)
    except Exception:
        pass
    relu_val = max(0.0, (s_max - kappa) / max(kappa, 1e-6))
    w_sat = _safe_val(loss_weights["S_max"], 1.0)
    term_sat = w_sat * relu_val ** 2

    # ---- Defaults for igraph metrics ----
    C_obs, Q_obs, rho_obs = 0.0, 0.0, 1.0
    # Default penalties if igraph fails
    term_C = _safe_val(loss_weights["C"], 1.0) * 1.0
    term_Q = _safe_val(loss_weights["Q"], 1.0) * 1.0
    term_rho = _safe_val(loss_weights["rho"], 1.0) * 1.0

    if n_edges > 100:
        try:
            edges = list(zip(surviving_sources.tolist(),
                             surviving_targets.tolist()))
            ig_graph = ig.Graph(n=n_genes, edges=edges, directed=True,
                                edge_attrs={'weight': surviving_W.tolist()})

            # Assortativity (zero-variance guard)
            try:
                rho_calc = ig_graph.assortativity_degree(directed=True)
                if np.isfinite(rho_calc):
                    rho_obs = rho_calc
                    term_rho = _penalty("rho", rho_obs)
            except Exception:
                pass

            # Clustering and Modularity
            try:
                ig_undirected = ig_graph.as_undirected(
                    mode="collapse", combine_edges=dict(weight="sum"))

                C_calc = ig_undirected.transitivity_undirected()
                if np.isfinite(C_calc):
                    C_obs = C_calc
                    term_C = _penalty("C", C_obs)

                partition = ig_undirected.community_multilevel()
                Q_calc = partition.modularity
                if np.isfinite(Q_calc):
                    Q_obs = Q_calc
                    term_Q = _penalty("Q", Q_obs)
            except Exception:
                pass

        except Exception:
            pass

    # Final Euclidean distance (guaranteed finite)
    raw_sum = (
        _safe_val(term_alpha) + _safe_val(term_C) + _safe_val(term_Q) +
        _safe_val(term_G) + _safe_val(term_rho) + _safe_val(term_sat)
    )
    L_utopia = np.sqrt(max(raw_sum, 0.0))

    topo_metrics = {
        'alpha': _safe_val(alpha_obs, 1.0),
        'C': _safe_val(C_obs),
        'Q': _safe_val(Q_obs),
        'Gini': _safe_val(G_obs, 1.0),
        'rho': _safe_val(rho_obs, 1.0),
        'S_max': _safe_val(s_max),
    }

    return L_utopia, topo_metrics


# =========================================================================
# DASH Kernel (Single Parameter Set)
# =========================================================================

def run_dash_and_score(params, W, D, sources, targets, G_baseline_coo,
                       n_genes, perturbed_nodes, utopian_bounds,
                       loss_weights, shatter_cfg):
    """Full pipeline for one parameter set.

    v7: This function is guaranteed to return a dict with a finite
    utopia_loss value. It never returns NaN under any circumstances.
    """
    try:
        beta, gamma, delta, kappa, k_core, lam = params

        T_local = compute_dynamic_topology(G_baseline_coo, k_core, n_genes)
        epsilon = 1e-6

        target_baseline_density = 0.0055
        lambda_baseline = n_genes * target_baseline_density
        gamma_dynamic = gamma * (lam / max(lambda_baseline, 1e-6))

        numerator = (W ** beta) * (1 + delta * T_local)
        denominator = (np.log1p(D) + epsilon) ** gamma_dynamic
        omega_scores = numerator / denominator

        surviving_sources, surviving_targets, surviving_W = dual_pass_pruning(
            omega_scores, W, sources, targets, perturbed_nodes, n_genes, lam, K=3
        )

        out_degrees = np.bincount(surviving_sources, minlength=n_genes)
        in_degrees = np.bincount(surviving_targets, minlength=n_genes)
        all_degrees = out_degrees + in_degrees
        active_nodes = int(np.count_nonzero(all_degrees > 0))

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
                'C': 0.0, 'Q': 0.0, 'S_max': _safe_val(s_max),
            }

        utopia_loss, topo_metrics = calculate_utopia_loss(
            surviving_sources, surviving_targets, surviving_W,
            n_genes, out_degrees, active_nodes, kappa,
            utopian_bounds, loss_weights
        )

        # GWCC penalty
        gwcc_fraction = 0.0
        try:
            sparse_G = sp.coo_matrix(
                (np.ones(len(surviving_W)), (surviving_sources, surviving_targets)),
                shape=(n_genes, n_genes)
            )
            _, labels = connected_components(csgraph=sparse_G, directed=False,
                                             return_labels=True)
            gwcc_size = np.bincount(labels).max()
            gwcc_fraction = gwcc_size / n_genes
        except Exception:
            pass

        gwcc_target = 0.45
        gwcc_penalty = 0.0
        if gwcc_fraction < gwcc_target:
            gwcc_penalty = 8.0 * ((gwcc_target - gwcc_fraction) / gwcc_target) ** 2

        utopia_loss = np.sqrt(max(utopia_loss ** 2 + gwcc_penalty, 0.0))

        # FINAL GUARD: if utopia_loss is somehow still NaN, assign max penalty
        utopia_loss = _safe_val(utopia_loss, 999.0)

        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': utopia_loss,
            'is_shattered': 0, 'shatter_reason': None,
            'n_edges': len(surviving_sources),
            'active_nodes': active_nodes,
            'gwcc_fraction': _safe_val(gwcc_fraction),
            'alpha': topo_metrics['alpha'], 'Gini': topo_metrics['Gini'],
            'rho': topo_metrics['rho'], 'C': topo_metrics['C'],
            'Q': topo_metrics['Q'], 'S_max': topo_metrics['S_max'],
        }

    except Exception as e:
        # Absolute last resort: if the entire function crashes,
        # return a valid penalty dict so the pipeline survives.
        beta, gamma, delta, kappa, k_core, lam = params
        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': 999.0,
            'is_shattered': 1, 'shatter_reason': f'exception:{str(e)[:80]}',
            'n_edges': 0, 'active_nodes': 0,
            'alpha': 1.0, 'Gini': 1.0, 'rho': 1.0,
            'C': 0.0, 'Q': 0.0, 'S_max': 0.0,
        }
