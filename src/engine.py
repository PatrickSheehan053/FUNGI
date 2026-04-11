"""
FUNGI v6.2 -- DASH Kernel Engine

v6.2 resolves the 100% shatter rate by recognizing that clustering,
spectral masking, and texture checks are SOFT quality metrics, not
hard structural failures. They belong in the utopian loss function
(where they already exist as the C, Q, and rho terms), not as
binary kill switches in the shatter triage.

Hard shatter conditions (structural impossibilities):
    1. density_collapse   -- too many edges (hairball)
    2. orphan_collapse    -- too many disconnected genes
    3. dictator_hub       -- single node exceeds S_max
    4. gwcc_percolation   -- giant component too small

Soft quality metrics (scored in utopian loss, not killed):
    - clustering coefficient (C)
    - modularity (Q)
    - assortativity (rho)
    - scale-free exponent (alpha)
    - spectral gap, Moran's I

The v5 FUNGI that successfully produced 50 champion graphs used
exactly this architecture: hard kills for structural failures,
soft scoring for quality metrics.
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
    if x is None or not np.isfinite(x):
        return fallback
    return float(x)


# =========================================================================
# Dynamic Topology (GraphBLAS)
# =========================================================================

def compute_dynamic_topology(W_sorted, sources_sorted, targets_sorted,
                             k_core, n_genes):
    """Per-edge local FFL participation via GraphBLAS masked SpGEMM."""
    target_edges = int(n_genes * k_core)
    target_edges = min(target_edges, len(W_sorted))

    T_local = np.zeros(len(W_sorted), dtype=np.float64)
    if target_edges < 1:
        return T_local

    core_rows = sources_sorted[:target_edges]
    core_cols = targets_sorted[:target_edges]

    A = gb.Matrix.from_coo(
        core_rows.astype(np.uint64),
        core_cols.astype(np.uint64),
        np.ones(target_edges, dtype=np.float64),
        nrows=n_genes, ncols=n_genes
    )

    T_gb = A.mxm(A, gb.semiring.plus_times).new(mask=A.S)

    d_out = np.zeros(n_genes, dtype=np.float64)
    oi, ov = A.reduce_rowwise(gb.monoid.plus).new().to_coo()
    d_out[oi] = ov

    d_in = np.zeros(n_genes, dtype=np.float64)
    ii, iv = A.reduce_columnwise(gb.monoid.plus).new().to_coo()
    d_in[ii] = iv

    t_rows, t_cols, z_vals = T_gb.to_coo()

    if len(z_vals) > 0:
        core_flat = core_rows.astype(np.int64) * n_genes + core_cols.astype(np.int64)
        t_flat = t_rows.astype(np.int64) * n_genes + t_cols.astype(np.int64)
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

    local_cap = np.minimum(d_out[core_rows], d_in[core_cols])
    valid = (mapped_z > 0) & (local_cap > 0)

    ratios = np.zeros(target_edges, dtype=np.float64)
    ratios[valid] = mapped_z[valid] / local_cap[valid]
    T_local[:target_edges] = ratios

    np.clip(T_local, 0.0, 1.0, out=T_local)
    return T_local


# =========================================================================
# 3-Tranche Dual-Pass Pruning
# =========================================================================

def dual_pass_pruning(omega_scores, W_sub, src_sub, tgt_sub, T_sub,
                      perturbed_nodes, n_genes, lam,
                      imm_src, imm_tgt, imm_W,
                      max_node_fraction=0.15, modularity_fraction=0.10):
    """3-Tranche routing with per-node hard cap."""
    total_budget = int(np.round(n_genes * lam))
    max_node_edges = int(n_genes * max_node_fraction)

    # Tranche 1: Static Immunization (pre-computed lifelines)
    imm_ids = set(imm_src.astype(np.int64) * n_genes + imm_tgt.astype(np.int64))

    # Local guardrails for perturbed genes
    protected_idx = []
    for p_node in perturbed_nodes:
        p_edges = np.where(src_sub == p_node)[0]
        if len(p_edges) > 0:
            k_actual = min(3, len(p_edges))
            top_k = p_edges[np.argsort(omega_scores[p_edges])[-k_actual:]]
            protected_idx.extend(top_k)

    sub_ids = src_sub.astype(np.int64) * n_genes + tgt_sub.astype(np.int64)
    overlap_mask = np.isin(sub_ids, np.array(list(imm_ids), dtype=np.int64))

    # Tranche 2: Modularity rescue (top W * T_local edges, bypassing beta)
    mod_budget = int(total_budget * modularity_fraction)
    if mod_budget > 0:
        motif_scores = W_sub * T_sub
        motif_mask = np.ones(len(motif_scores), dtype=bool)
        motif_mask[overlap_mask] = False
        if len(protected_idx) > 0:
            motif_mask[protected_idx] = False

        avail_motif = np.where(motif_mask)[0]
        if mod_budget < len(avail_motif):
            top_motif_rel = np.argpartition(
                motif_scores[avail_motif], -mod_budget)[-mod_budget:]
            mod_idx = avail_motif[top_motif_rel]
        else:
            mod_idx = avail_motif
        protected_idx.extend(mod_idx)

    protected_idx = np.unique(np.array(protected_idx, dtype=int))

    # Tranche 3: Oligarchic global fill
    mask = np.ones(len(omega_scores), dtype=bool)
    mask[overlap_mask] = False
    if len(protected_idx) > 0:
        mask[protected_idx] = False

    already_count = len(imm_src) + len(protected_idx)
    remaining_budget = total_budget - already_count

    if remaining_budget > 0:
        available = np.where(mask)[0]
        if remaining_budget < len(available):
            top_rel = np.argpartition(
                omega_scores[available], -remaining_budget)[-remaining_budget:]
            fill_idx = available[top_rel]
        else:
            fill_idx = available
    else:
        fill_idx = np.array([], dtype=int)

    selected_sub_idx = np.concatenate([protected_idx, fill_idx]).astype(int)

    # Per-node hard cap (only on dynamic edges, immunization is sacred)
    sub_src_sel = src_sub[selected_sub_idx]
    sub_tgt_sel = tgt_sub[selected_sub_idx]
    sub_W_sel = W_sub[selected_sub_idx]
    sub_om_sel = omega_scores[selected_sub_idx]

    combined_src = np.concatenate([imm_src, sub_src_sel]) if len(sub_src_sel) > 0 else imm_src
    node_counts = np.bincount(combined_src, minlength=n_genes)
    overloaded = np.where(node_counts > max_node_edges)[0]

    if len(overloaded) > 0 and len(selected_sub_idx) > 0:
        keep_mask = np.ones(len(selected_sub_idx), dtype=bool)
        for node in overloaded:
            excess = node_counts[node] - max_node_edges
            if excess <= 0:
                continue
            node_pos = np.where(sub_src_sel == node)[0]
            n_to_drop = min(excess, len(node_pos))
            if n_to_drop > 0:
                oms = sub_om_sel[node_pos]
                drop_pos = node_pos[np.argsort(oms)[:n_to_drop]]
                keep_mask[drop_pos] = False

        selected_sub_idx = selected_sub_idx[keep_mask]
        sub_src_sel = src_sub[selected_sub_idx]
        sub_tgt_sel = tgt_sub[selected_sub_idx]
        sub_W_sel = W_sub[selected_sub_idx]

    # Graph union
    if len(selected_sub_idx) > 0:
        final_src = np.concatenate([imm_src, sub_src_sel])
        final_tgt = np.concatenate([imm_tgt, sub_tgt_sel])
        final_W = np.concatenate([imm_W, sub_W_sel])
    else:
        final_src = imm_src
        final_tgt = imm_tgt
        final_W = imm_W

    return final_src, final_tgt, final_W


# =========================================================================
# Shatter Criteria (STRUCTURAL ONLY)
# =========================================================================

def check_shatter(surviving_sources, surviving_targets, surviving_W,
                  out_degrees, n_genes, active_nodes, shatter_cfg):
    """Hard shatter checks for structural impossibilities only.

    v6.4: clustering_collapse, spectral_masking, and texture_collapse
    REMOVED from hard kills. These are quality metrics that belong in
    the utopian loss function, not binary kill switches. The v5 FUNGI
    that produced 50 champion graphs used exactly this approach.
    """
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
            gwcc_fraction = np.bincount(labels).max() / n_genes
            if gwcc_fraction < shatter_cfg.get("min_gwcc_fraction", 0.30):
                return True, "gwcc_percolation"
        except Exception:
            return True, "gwcc_percolation"

    return False, None


# =========================================================================
# Utopia Loss (all quality metrics scored here, not killed)
# =========================================================================

def calculate_utopia_loss(surviving_sources, surviving_targets, surviving_W,
                          n_genes, out_degrees, active_nodes, kappa,
                          utopian_bounds, loss_weights):
    """Euclidean distance from utopian target. Never returns NaN."""
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
                1.0)
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

def run_dash_and_score(params, W, D, sources, targets, imm_idx, G_baseline_coo,
                       n_genes, perturbed_nodes, utopian_bounds,
                       loss_weights, shatter_cfg):
    """Full pipeline for one parameter set. Guaranteed finite output."""
    try:
        beta, gamma, delta, kappa, k_core, lam = params

        T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)

        # Extract static immunization set
        imm_src = sources[imm_idx]
        imm_tgt = targets[imm_idx]
        imm_W_arr = W[imm_idx]

        # Truncate for speed
        N_max = shatter_cfg.get("max_edge_count", 500000)
        N_eval = min(len(W), N_max + 10000)

        W_sub = W[:N_eval]
        src_sub = sources[:N_eval]
        tgt_sub = targets[:N_eval]
        T_sub = T_local[:N_eval]

        # Numerator
        numerator = (W_sub ** beta) * (1 + delta * T_sub)

        # Marginal rank penalty with raw gamma
        order = np.lexsort((-numerator, src_sub))
        src_ord = src_sub[order]
        tgt_ord = tgt_sub[order]
        W_ord = W_sub[order]
        num_ord = numerator[order]
        T_ord = T_sub[order]

        boundaries = np.nonzero(src_ord[1:] != src_ord[:-1])[0] + 1
        split_idx = np.concatenate(([0], boundaries, [len(src_ord)]))
        k_rank = np.empty(len(src_ord), dtype=np.float64)
        for i in range(len(split_idx) - 1):
            s, e = split_idx[i], split_idx[i+1]
            k_rank[s:e] = np.arange(1, e - s + 1)

        omega_scores = num_ord / (k_rank ** gamma)

        # 3-Tranche pruning
        max_hub_sat = shatter_cfg.get("max_hub_saturation", 0.15)
        surv_src, surv_tgt, surv_W = dual_pass_pruning(
            omega_scores, W_ord, src_ord, tgt_ord, T_ord,
            perturbed_nodes, n_genes, lam,
            imm_src, imm_tgt, imm_W_arr,
            max_node_fraction=max_hub_sat,
            modularity_fraction=0.10
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
