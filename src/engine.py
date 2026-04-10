"""
FUNGI v6.2 -- DASH Kernel Engine

v6.2 fixes:
    - GraphBLAS semiring call corrected: A.mxm(A, semiring).new(mask=A.S)
    - T_local[:target_edges][valid] copy-not-view bug fixed
    - NaN guards on all topology metrics (powerlaw, assortativity, Gini)
    - Top-level try/except on run_dash_and_score returns 999.0 on any crash
    - Spectral masking and Moran's I shatter checks import-guarded
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import powerlaw
import igraph as ig
import warnings
import graphblas as gb

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _safe(x, fallback=0.0):
    """Returns x if finite, else fallback."""
    if x is None or not np.isfinite(x):
        return fallback
    return float(x)


# =========================================================================
# Dynamic Topology (GraphBLAS)
# =========================================================================

def compute_dynamic_topology(W_sorted, sources_sorted, targets_sorted,
                             k_core, n_genes):
    """Per-edge local FFL participation using GraphBLAS masked SpGEMM.

    Expects edges pre-sorted descending by weight so the top-k core
    is a simple slice [:target_edges].
    """
    target_edges = int(n_genes * k_core)
    target_edges = min(target_edges, len(W_sorted))

    T_local = np.zeros(len(W_sorted), dtype=np.float64)
    if target_edges < 1:
        return T_local

    core_rows = sources_sorted[:target_edges]
    core_cols = targets_sorted[:target_edges]

    # 1. Build binary GraphBLAS matrix
    A = gb.Matrix.from_coo(
        core_rows.astype(np.uint64),
        core_cols.astype(np.uint64),
        np.ones(target_edges, dtype=np.float64),
        nrows=n_genes, ncols=n_genes
    )

    # 2. Masked SpGEMM: T = (A @ A) masked by A.S
    #    This is the CORRECT python-graphblas syntax.
    T_gb = A.mxm(A, gb.semiring.plus_times).new(mask=A.S)

    # 3. Degree reductions via GraphBLAS (no scipy)
    d_out = np.zeros(n_genes, dtype=np.float64)
    out_v = A.reduce_rowwise(gb.monoid.plus).new()
    oi, ov = out_v.to_coo()
    d_out[oi] = ov

    d_in = np.zeros(n_genes, dtype=np.float64)
    in_v = A.reduce_columnwise(gb.monoid.plus).new()
    ii, iv = in_v.to_coo()
    d_in[ii] = iv

    # 4. Extract triangle counts and map back to core edges
    t_rows, t_cols, z_vals = T_gb.to_coo()

    if len(z_vals) > 0:
        # Flat-index binary search for O(E log E) mapping
        core_flat = core_rows.astype(np.int64) * n_genes + core_cols.astype(np.int64)
        t_flat = t_rows.astype(np.int64) * n_genes + t_cols.astype(np.int64)

        # t_flat may not be sorted if GraphBLAS version does not guarantee it
        t_order = np.argsort(t_flat)
        t_flat_sorted = t_flat[t_order]
        z_vals_sorted = z_vals[t_order]

        sort_idx = np.searchsorted(t_flat_sorted, core_flat)
        valid_idx = np.clip(sort_idx, 0, len(t_flat_sorted) - 1)
        hit_mask = (t_flat_sorted[valid_idx] == core_flat)

        mapped_z = np.zeros(target_edges, dtype=np.float64)
        mapped_z[hit_mask] = z_vals_sorted[valid_idx[hit_mask]]
    else:
        mapped_z = np.zeros(target_edges, dtype=np.float64)

    # 5. Local motif participation ratio
    local_cap = np.minimum(d_out[core_rows], d_in[core_cols])
    valid = (mapped_z > 0) & (local_cap > 0)

    # FIX: direct index assignment (no chained slice copy bug)
    ratios = np.zeros(target_edges, dtype=np.float64)
    ratios[valid] = mapped_z[valid] / local_cap[valid]
    T_local[:target_edges] = ratios

    np.clip(T_local, 0.0, 1.0, out=T_local)
    return T_local


# =========================================================================
# Dual-Pass Pruning
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
            top_idx_rel = np.argpartition(
                omega_scores[available_indices], -remaining_budget
            )[-remaining_budget:]
            top_idx = available_indices[top_idx_rel]
        else:
            top_idx = available_indices

        final_indices = np.concatenate([protected_indices, top_idx])
    else:
        final_indices = protected_indices[:total_budget]

    return sources[final_indices], targets[final_indices], W[final_indices]


# =========================================================================
# Shatter Criteria
# =========================================================================

def check_shatter(surviving_sources, surviving_targets, surviving_W,
                  out_degrees, n_genes, active_nodes, shatter_cfg):
    """Fail-fast shatter check, cheapest conditions first."""
    n_edges = len(surviving_sources)

    if n_edges > shatter_cfg.get("max_edge_count", 500000):
        return True, "density_collapse"

    orphan_frac = (n_genes - active_nodes) / max(n_genes, 1)
    if orphan_frac > shatter_cfg.get("max_orphan_fraction", 0.70):
        return True, "orphan_collapse"

    s_max = (np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0
    if s_max > shatter_cfg.get("max_hub_saturation", 0.15):
        return True, "dictator_hub"

    if n_edges > 0 and active_nodes > 0:
        try:
            sparse_G = sp.coo_matrix(
                (np.ones(n_edges), (surviving_sources, surviving_targets)),
                shape=(n_genes, n_genes)
            )
            _, labels = connected_components(csgraph=sparse_G, directed=False,
                                             return_labels=True)
            gwcc_fraction = np.bincount(labels).max() / n_genes
            if gwcc_fraction < shatter_cfg.get("min_gwcc_fraction", 0.30):
                return True, "gwcc_percolation"
        except Exception:
            return True, "gwcc_percolation"

    if n_edges > 50:
        try:
            edges = list(zip(surviving_sources.tolist(), surviving_targets.tolist()))
            ig_u = ig.Graph(n=n_genes, edges=edges, directed=True).as_undirected(mode="collapse")
            C_obs = ig_u.transitivity_undirected()
            if np.isfinite(C_obs) and C_obs < shatter_cfg.get("min_clustering", 0.01):
                return True, "clustering_collapse"
        except Exception:
            pass

    if n_edges > 100:
        try:
            from graph_utils import compute_spectral_dominance_ratio
            ratio = compute_spectral_dominance_ratio(
                surviving_sources, surviving_targets, surviving_W, n_genes)
            if np.isfinite(ratio) and ratio > shatter_cfg.get("max_spectral_dominance_ratio", 50.0):
                return True, "spectral_masking"
        except Exception:
            pass

    if n_edges > 50 and active_nodes > 10:
        try:
            from graph_utils import compute_morans_i
            morans = compute_morans_i(
                surviving_sources, surviving_targets, out_degrees, n_genes)
            if np.isfinite(morans) and morans < shatter_cfg.get("min_morans_i", 0.02):
                return True, "texture_collapse"
        except Exception:
            pass

    return False, None


# =========================================================================
# Utopia Loss (NaN-proof)
# =========================================================================

def calculate_utopia_loss(surviving_sources, surviving_targets, surviving_W,
                          n_genes, out_degrees, active_nodes, kappa,
                          utopian_bounds, loss_weights):
    """Euclidean distance from the utopian target zone.

    Every metric individually guarded. Never returns NaN.
    """
    n_edges = len(surviving_sources)

    def _pen(param, obs):
        b = utopian_bounds[param]
        w = _safe(loss_weights[param], 1.0)
        o = _safe(obs, 0.0)
        if o < b[0]:
            return w * ((b[0] - o) / max(abs(b[0]), 1e-6)) ** 2
        elif o > b[1]:
            return w * ((o - b[1]) / max(abs(b[1]), 1e-6)) ** 2
        return 0.0

    # Alpha
    alpha_obs = 1.0
    try:
        if len(out_degrees) > 10 and np.max(out_degrees) > 1:
            unique_deg = np.unique(out_degrees[out_degrees > 0])
            if len(unique_deg) >= 3:
                fit = powerlaw.Fit(out_degrees[out_degrees > 0],
                                   xmin=1, discrete=True, verbose=False)
                alpha_obs = _safe(fit.power_law.alpha, 1.0)
    except Exception:
        pass
    term_alpha = _pen("alpha", alpha_obs)

    # Gini
    G_obs = 1.0
    try:
        if active_nodes > 1 and np.sum(out_degrees) > 0:
            sd = np.sort(out_degrees)
            n = len(sd)
            G_obs = _safe(
                (2.0 * np.sum(np.arange(1, n+1) * sd)) / (n * np.sum(sd)) - (n+1)/n,
                1.0
            )
    except Exception:
        pass
    term_G = _pen("gini", G_obs)

    # S_max
    s_max = _safe((np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0)
    relu_val = max(0.0, (s_max - kappa) / max(kappa, 1e-6))
    term_sat = _safe(loss_weights["S_max"], 1.0) * relu_val ** 2

    # Defaults for igraph metrics
    C_obs, Q_obs, rho_obs = 0.0, 0.0, 1.0
    term_C = _safe(loss_weights["C"], 1.0)
    term_Q = _safe(loss_weights["Q"], 1.0)
    term_rho = _safe(loss_weights["rho"], 1.0)

    if n_edges > 100:
        try:
            edges = list(zip(surviving_sources.tolist(), surviving_targets.tolist()))
            ig_g = ig.Graph(n=n_genes, edges=edges, directed=True,
                            edge_attrs={'weight': surviving_W.tolist()})

            try:
                rc = ig_g.assortativity_degree(directed=True)
                if np.isfinite(rc):
                    rho_obs = rc
                    term_rho = _pen("rho", rho_obs)
            except Exception:
                pass

            try:
                ig_u = ig_g.as_undirected(mode="collapse", combine_edges=dict(weight="sum"))
                cc = ig_u.transitivity_undirected()
                if np.isfinite(cc):
                    C_obs = cc
                    term_C = _pen("C", C_obs)
                part = ig_u.community_multilevel()
                qm = part.modularity
                if np.isfinite(qm):
                    Q_obs = qm
                    term_Q = _pen("Q", Q_obs)
            except Exception:
                pass

        except Exception:
            pass

    raw = (_safe(term_alpha) + _safe(term_C) + _safe(term_Q) +
           _safe(term_G) + _safe(term_rho) + _safe(term_sat))
    L = np.sqrt(max(raw, 0.0))

    topo = {
        'alpha': _safe(alpha_obs, 1.0), 'C': _safe(C_obs),
        'Q': _safe(Q_obs), 'Gini': _safe(G_obs, 1.0),
        'rho': _safe(rho_obs, 1.0), 'S_max': _safe(s_max),
    }
    return L, topo


# =========================================================================
# DASH Kernel Entry Point
# =========================================================================

def run_dash_and_score(params, W, D, sources, targets, G_baseline_coo,
                       n_genes, perturbed_nodes, utopian_bounds,
                       loss_weights, shatter_cfg):
    """Full pipeline for one parameter set. Guaranteed finite output."""
    try:
        beta, gamma, delta, kappa, k_core, lam = params

        T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
        epsilon = 1e-6

        lambda_baseline = n_genes * 0.0055
        gamma_dynamic = gamma * (lam / max(lambda_baseline, 1e-6))

        numerator = (W ** beta) * (1 + delta * T_local)
        denominator = (np.log1p(D) + epsilon) ** gamma_dynamic
        omega_scores = numerator / denominator

        surv_src, surv_tgt, surv_W = dual_pass_pruning(
            omega_scores, W, sources, targets, perturbed_nodes, n_genes, lam, K=3
        )

        out_deg = np.bincount(surv_src, minlength=n_genes)
        in_deg = np.bincount(surv_tgt, minlength=n_genes)
        active = int(np.count_nonzero(out_deg + in_deg > 0))

        is_shattered, reason = check_shatter(
            surv_src, surv_tgt, surv_W, out_deg, n_genes, active, shatter_cfg
        )

        if is_shattered:
            sm = _safe((np.max(out_deg) / n_genes) if len(out_deg) > 0 else 0)
            return {
                'beta': beta, 'gamma': gamma, 'delta': delta,
                'kappa': kappa, 'k_core': k_core, 'lambda': lam,
                'utopia_loss': 999.0,
                'is_shattered': 1, 'shatter_reason': reason,
                'n_edges': len(surv_src), 'active_nodes': active,
                'alpha': 1.0, 'Gini': 1.0, 'rho': 1.0,
                'C': 0.0, 'Q': 0.0, 'S_max': sm,
            }

        loss, topo = calculate_utopia_loss(
            surv_src, surv_tgt, surv_W, n_genes, out_deg, active, kappa,
            utopian_bounds, loss_weights
        )

        # GWCC penalty
        gwcc_frac = 0.0
        try:
            sp_G = sp.coo_matrix(
                (np.ones(len(surv_W)), (surv_src, surv_tgt)),
                shape=(n_genes, n_genes)
            )
            _, lb = connected_components(csgraph=sp_G, directed=False, return_labels=True)
            gwcc_frac = _safe(np.bincount(lb).max() / n_genes)
        except Exception:
            pass

        gwcc_pen = 0.0
        if gwcc_frac < 0.45:
            gwcc_pen = 8.0 * ((0.45 - gwcc_frac) / 0.45) ** 2
        loss = _safe(np.sqrt(max(loss**2 + gwcc_pen, 0.0)), 999.0)

        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': loss,
            'is_shattered': 0, 'shatter_reason': None,
            'n_edges': len(surv_src), 'active_nodes': active,
            'gwcc_fraction': gwcc_frac,
            'alpha': topo['alpha'], 'Gini': topo['Gini'],
            'rho': topo['rho'], 'C': topo['C'],
            'Q': topo['Q'], 'S_max': topo['S_max'],
        }

    except Exception as e:
        beta, gamma, delta, kappa, k_core, lam = params
        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': 999.0,
            'is_shattered': 1, 'shatter_reason': f'crash:{str(e)[:60]}',
            'n_edges': 0, 'active_nodes': 0,
            'alpha': 1.0, 'Gini': 1.0, 'rho': 1.0,
            'C': 0.0, 'Q': 0.0, 'S_max': 0.0,
        }
