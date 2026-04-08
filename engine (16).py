import numpy as np
import scipy.sparse as sp
import networkx as nx
from joblib import Parallel, delayed
import powerlaw
import igraph as ig
import scipy.sparse.csgraph as csgraph
from scipy.sparse.csgraph import connected_components
import warnings
import pandas as pd

# Suppress powerlaw convergence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Utopia Loss Function
# =============================================================================

def calculate_utopia_loss(surviving_sources, surviving_targets, surviving_W,
                          n_genes, out_degrees, active_nodes, kappa):
    """Calculates the Euclidean distance from the utopian target zone across
    six topological parameters: alpha, C, Q, Gini, rho, and S_max.

    v4.1: Added modularity ceiling (Q_max = 0.75) to prevent PPR signal
    from becoming trapped in sink modules (the Modularist trap)."""

    n_edges = len(surviving_sources)
    is_shattered = 0

    # 1. Scale-Free Shape (alpha)
    alpha_obs = 1.0
    if len(out_degrees) > 10 and np.max(out_degrees) > 1:
        try:
            fit = powerlaw.Fit(out_degrees, xmin=1, discrete=True, verbose=False)
            alpha_obs = fit.power_law.alpha
        except Exception:
            pass
    term_alpha = 10.0 * (alpha_obs - 2.3)**2

    # 2. Hierarchical Flow (Gini) — relative percentage scaling
    G_obs = 1.0
    if active_nodes > 1 and np.sum(out_degrees) > 0:
        sorted_deg = np.sort(out_degrees)
        n = len(out_degrees)
        G_obs = (2.0 * np.sum((np.arange(1, n + 1) * sorted_deg))) / (n * np.sum(sorted_deg)) - (n + 1) / n

    term_G = 0.0
    target_G_min, target_G_max = 0.65, 0.85
    if G_obs < target_G_min:
        term_G = 5.0 * ((target_G_min - G_obs) / target_G_min)**2
    elif G_obs > target_G_max:
        term_G = 5.0 * ((G_obs - target_G_max) / target_G_max)**2

    # 3. Biophysical Constraint (S_max ReLU using dynamic kappa)
    max_degree = np.max(out_degrees) if len(out_degrees) > 0 else 0
    s_max = max_degree / n_genes
    relu_val = max(0.0, (s_max - kappa) / kappa)
    term_sat = 15.0 * (relu_val)**2

    # Defaults for severe penalties (overwritten if graph is large enough)
    C_obs, Q_obs, rho_obs = 0.0, 0.0, 1.0

    # 4. Modular Texture (Clustering C) — default penalty
    term_C = 5.0 * ((0.0 - 0.35) / 0.35)**2

    # 5. Functional Communities (Modularity Q) — default penalty
    term_Q = 10.0 * ((0.40 - 0.0) / 0.40)**2

    # 6. Wiring Logic (Assortativity rho) — default penalty
    term_rho = 5.0 * (1.0 - (-0.3))**2

    if n_edges > 100:
        try:
            edges = list(zip(surviving_sources, surviving_targets))
            ig_graph = ig.Graph(n=n_genes, edges=edges, directed=True,
                                edge_attrs={'weight': surviving_W})

            rho_calc = ig_graph.assortativity_degree(directed=True)
            if not np.isnan(rho_calc):
                rho_obs = rho_calc
                term_rho = 5.0 * (rho_obs - (-0.3))**2

            ig_graph_undirected = ig_graph.as_undirected(
                mode="collapse", combine_edges=dict(weight="sum"))

            C_calc = ig_graph_undirected.transitivity_undirected()
            if not np.isnan(C_calc):
                C_obs = C_calc
                term_C = 5.0 * ((C_obs - 0.35) / 0.35)**2

            partition = ig_graph_undirected.community_multilevel()
            Q_obs = partition.modularity

            # v4.1 FIX 5: Modularity ceiling
            # Penalize both below Q_min and above Q_max.
            # Q too high creates strong intra-module / weak inter-module
            # structure that traps PPR teleportation inside sink modules.
            term_Q = 0.0
            target_Q_min = 0.40
            target_Q_max = 0.75
            if Q_obs < target_Q_min:
                term_Q = 10.0 * ((target_Q_min - Q_obs) / target_Q_min)**2
            elif Q_obs > target_Q_max:
                term_Q = 10.0 * ((Q_obs - target_Q_max) / target_Q_max)**2

        except Exception:
            pass

    # Final Euclidean Distance
    L_utopia = np.sqrt(term_alpha + term_C + term_Q + term_G + term_rho + term_sat)

    topo_metrics = {
        'alpha': alpha_obs,
        'C': C_obs,
        'Q': Q_obs,
        'Gini': G_obs,
        'rho': rho_obs,
        'S_max': s_max,
    }

    return L_utopia, is_shattered, topo_metrics


# =============================================================================
# Dynamic Topology (Per-Edge T_local from Feed-Forward Loops)
# =============================================================================

def compute_dynamic_topology(G_baseline_coo, k_core, n_genes):
    """Calculates per-edge local motif participation ratio (T_local).

    v4.1 changes:
    - FIX 1: Core edge budget scales as n * k_core (linear) instead of
      n^2 * k_core (quadratic). The k_core parameter now represents the
      average degree of the topological core, keeping triangle density
      comparable across datasets of different sizes.
    - FIX 2: Replaces global max-scaling (T_norm) with local motif
      participation ratio: T_local(u,v) = z_uv / min(d_out(u), d_in(v)).
      This measures what fraction of a node's local neighborhood participates
      in feed-forward loops through edge (u,v), making the signal informative
      regardless of absolute triangle count."""

    # 1. Determine exact edge budget — LINEAR scaling (v4.1 Fix 1)
    target_edges = int(n_genes * k_core)
    target_edges = min(target_edges, len(G_baseline_coo.data))

    if target_edges < 1:
        return np.zeros(len(G_baseline_coo.data), dtype=np.float64)

    # 2. Extract exact indices to prevent tie-breaker inflation
    surviving_indices = np.argpartition(G_baseline_coo.data, -target_edges)[-target_edges:]

    core_rows = G_baseline_coo.row[surviving_indices]
    core_cols = G_baseline_coo.col[surviving_indices]

    # 3. Build A_core sparse matrix (binary representation)
    A_core = sp.coo_matrix(
        (np.ones(target_edges, dtype=np.int32), (core_rows, core_cols)),
        shape=(n_genes, n_genes)
    ).tocsr()

    # 4. Hadamard mask isolates true FFLs, prevents memory fill-in
    T_matrix = A_core.dot(A_core).multiply(A_core)

    # 5. Compute raw triangle counts for surviving edges
    T_raw = np.zeros(len(G_baseline_coo.data), dtype=np.float64)
    z_values = T_matrix[core_rows, core_cols].A1
    T_raw[surviving_indices] = z_values

    # 6. Local motif participation ratio (v4.1 Fix 2)
    # T_local(u,v) = z_uv / min(d_out(u), d_in(v))
    # This normalizes by local connectivity, not global max.
    d_out = np.array(A_core.sum(axis=1)).ravel()   # out-degree per node in core
    d_in = np.array(A_core.sum(axis=0)).ravel()     # in-degree per node in core

    T_local = np.zeros(len(G_baseline_coo.data), dtype=np.float64)
    local_cap = np.minimum(d_out[core_rows], d_in[core_cols])
    valid = (z_values > 0) & (local_cap > 0)
    T_local[surviving_indices[valid]] = z_values[valid] / local_cap[valid]

    # Clip to [0, 1] for numerical safety
    np.clip(T_local, 0.0, 1.0, out=T_local)

    return T_local


# =============================================================================
# Dual-Pass Pruning
# =============================================================================

def dual_pass_pruning(omega_scores, W, sources, targets, perturbed_nodes,
                      n_genes, lam, K=3):
    """Two-pass edge selection:
    Pass 1 — Local guardrails protect top-K edges from each perturbed node.
    Pass 2 — Global scale-free fill uses the remaining budget on highest omega."""

    total_budget = int(np.round(n_genes * lam))

    # Pass 1: Local guardrails (CRISPR perturbation targets)
    protected_indices = []
    for p_node in perturbed_nodes:
        p_edges = np.where(sources == p_node)[0]
        if len(p_edges) > 0:
            k_actual = min(K, len(p_edges))
            top_k_idx = p_edges[np.argsort(omega_scores[p_edges])[-k_actual:]]
            protected_indices.extend(top_k_idx)

    protected_indices = np.unique(protected_indices)

    # Pass 2: Global fill (highest omega from remaining candidates)
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


# =============================================================================
# DASH Kernel (Single Parameter Set)
# =============================================================================

def run_dash_and_score(params, W, D, sources, targets, G_baseline_coo,
                       n_genes, perturbed_nodes):
    """Full pipeline for one parameter set: compute omega, prune, and score.

    v4.1: Added GWCC percolation penalty (Fix 3). If the Giant Weakly
    Connected Component fraction falls below the dataset's active regulator
    fraction, a penalty is added to the utopia loss to guide the optimizer
    toward connected topologies."""

    beta, gamma, delta, kappa, k_core, lam = params

    T_local = compute_dynamic_topology(G_baseline_coo, k_core, n_genes)
    epsilon = 1e-6

    # Dynamic Baseline Anchoring
    target_baseline_density = 0.0055
    lambda_baseline = n_genes * target_baseline_density
    gamma_dynamic = gamma * (lam / lambda_baseline)

    # DASH Edge Scoring Equation
    numerator = (W ** beta) * (1 + delta * T_local)
    denominator = (np.log1p(D) + epsilon) ** gamma_dynamic
    omega_scores = numerator / denominator

    # Dual-pass pruning
    surviving_sources, surviving_targets, surviving_W = dual_pass_pruning(
        omega_scores, W, sources, targets, perturbed_nodes, n_genes, lam, K=3
    )

    # Fail-fast topological triage
    out_degrees = np.bincount(surviving_sources, minlength=n_genes)
    in_degrees = np.bincount(surviving_targets, minlength=n_genes)
    all_degrees = out_degrees + in_degrees
    active_nodes = np.count_nonzero(all_degrees > 0)

    n_orphans = n_genes - active_nodes
    n_islands = 0
    is_shattered = 0

    if ((n_genes - active_nodes) / n_genes) > 0.70:
        is_shattered = 1

    s_max = (np.max(out_degrees) / n_genes) if len(out_degrees) > 0 else 0
    if s_max > 0.35:
        is_shattered = 1

    # v4.1 FIX 3: GWCC percolation penalty
    gwcc_fraction = 0.0
    if is_shattered == 0 and active_nodes > 0:
        surviving_G_sparse = sp.coo_matrix(
            (np.ones(len(surviving_W)), (surviving_sources, surviving_targets)),
            shape=(n_genes, n_genes)
        )
        n_components, labels = connected_components(
            csgraph=surviving_G_sparse, directed=False, return_labels=True
        )
        n_islands = n_components - n_orphans

        # Compute GWCC fraction: largest component / total genes
        component_sizes = np.bincount(labels)
        gwcc_size = component_sizes.max()
        gwcc_fraction = gwcc_size / n_genes

    if is_shattered == 1:
        relu_val = max(0.0, (s_max - kappa) / kappa)

        term_alpha_dummy = 16.9
        term_C_dummy = 5.0
        term_Q_dummy = 10.0
        term_G_dummy = 5.0 * ((1.0 - 0.85) / 0.85)**2
        term_rho_dummy = 8.45
        term_sat_dummy = 15.0 * (relu_val)**2

        L_utopia = np.sqrt(
            term_alpha_dummy + term_C_dummy + term_Q_dummy
            + term_G_dummy + term_rho_dummy + term_sat_dummy
        )
        topo_metrics = {
            'alpha': 1.0, 'C': 0.0, 'Q': 0.0,
            'Gini': 1.0, 'rho': 1.0, 'S_max': s_max,
        }

        return {
            'beta': beta, 'gamma': gamma, 'delta': delta,
            'kappa': kappa, 'k_core': k_core, 'lambda': lam,
            'utopia_loss': L_utopia, 'is_shattered': is_shattered,
            'n_orphans': n_orphans, 'n_islands': n_islands,
            'gwcc_fraction': gwcc_fraction,
            'alpha': topo_metrics['alpha'], 'Gini': topo_metrics['Gini'],
            'rho': topo_metrics['rho'], 'C': topo_metrics['C'],
            'Q': topo_metrics['Q'], 'S_max': topo_metrics['S_max'],
        }

    # Full utopia evaluation
    utopia_loss, is_shattered, topo_metrics = calculate_utopia_loss(
        surviving_sources, surviving_targets, surviving_W,
        n_genes, out_degrees, active_nodes, kappa
    )

    # v4.1 FIX 3 (continued): Apply GWCC percolation penalty
    # Target GWCC fraction derived from Aguirre et al. 2025:
    # ~41% of KOs produce downstream effects, so the effective regulatory
    # subnetwork should connect at least ~45% of genes.
    gwcc_target = 0.45
    gwcc_penalty = 0.0
    if gwcc_fraction < gwcc_target:
        gwcc_penalty = 8.0 * ((gwcc_target - gwcc_fraction) / gwcc_target)**2

    utopia_loss = np.sqrt(utopia_loss**2 + gwcc_penalty)

    return {
        'beta': beta, 'gamma': gamma, 'delta': delta,
        'kappa': kappa, 'k_core': k_core, 'lambda': lam,
        'utopia_loss': utopia_loss, 'is_shattered': is_shattered,
        'n_orphans': n_orphans, 'n_islands': n_islands,
        'gwcc_fraction': gwcc_fraction,
        'alpha': topo_metrics['alpha'], 'Gini': topo_metrics['Gini'],
        'rho': topo_metrics['rho'], 'C': topo_metrics['C'],
        'Q': topo_metrics['Q'], 'S_max': topo_metrics['S_max'],
    }


# =============================================================================
# Parallel Batch Execution
# =============================================================================

def execute_dash_batch(param_list, W_arr, D_arr, sources_arr, targets_arr,
                       n_genes, perturbed_nodes, n_jobs=-1):
    """Distributes the DASH kernel search across all available CPU cores.
    Pre-computes the baseline COO matrix once for all workers."""

    print("Pre-computing structural matrices for parallel execution...")

    # Build baseline sparse matrix in COO format for compute_dynamic_topology
    G_baseline_csr = sp.csr_matrix(
        (W_arr, (sources_arr, targets_arr)),
        shape=(n_genes, n_genes)
    )
    G_baseline_coo = G_baseline_csr.tocoo()

    print(f"Executing parallel search across {n_jobs} cores...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_dash_and_score)(
            params, W_arr, D_arr, sources_arr, targets_arr,
            G_baseline_coo, n_genes, perturbed_nodes
        )
        for params in param_list
    )

    return pd.DataFrame(results)