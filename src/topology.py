"""
FUNGI v6 -- Topology Module

Contains:
    - Core triangle calculation (FFLs) -- fast, used during search.
    - Persistent homology (Betti curves, Wasserstein distance) -- slow,
      used ONLY during the final audit phase on the cohort.
    - Early Precision Ratio (EPR) -- requires ground truth, audit only.
"""

import numpy as np
import scipy.sparse as sp


# =========================================================================
# Core Triangle Calculation (used during search)
# =========================================================================

def calculate_core_triangles(G_core):
    """Computes the number of closed triangles (Feed-Forward Loops) for
    edges in the diamond core.

    Parameters
    ----------
    G_core : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the high-confidence core subgraph.

    Returns
    -------
    T_matrix : scipy.sparse.csr_matrix
        Triangle count matrix where T[i,j] = number of intermediate
        nodes k such that i->k and k->j both exist in the core,
        AND i->j exists (Hadamard mask).
    """
    A_core = G_core.copy()
    A_core.data = np.ones_like(A_core.data, dtype=np.int32)

    T_matrix = A_core.dot(A_core)
    T_matrix = T_matrix.multiply(A_core)

    return T_matrix


# =========================================================================
# Persistent Homology (Audit Phase Only)
# =========================================================================

def compute_persistence_wasserstein(sources_candidate, targets_candidate,
                                    weights_candidate,
                                    sources_reference, targets_reference,
                                    weights_reference,
                                    n_genes, max_dimension=1):
    """Computes 0D and 1D Wasserstein distances between the persistence
    diagrams of a candidate graph and a reference (dense parent) graph.

    This function is computationally expensive and should ONLY be called
    on the final audit cohort (50 graphs), not during the 100k search.

    Parameters
    ----------
    sources/targets/weights_candidate : np.ndarray
        Edge data for the pruned candidate graph.
    sources/targets/weights_reference : np.ndarray
        Edge data for the dense parent graph.
    n_genes : int
    max_dimension : int
        Maximum homological dimension (0 = components, 1 = loops).

    Returns
    -------
    dict
        {'wasserstein_0d': float, 'wasserstein_1d': float}

    Raises
    ------
    ImportError
        If ripser or persim are not installed.
    """
    try:
        from ripser import ripser
        from persim import wasserstein
    except ImportError:
        raise ImportError(
            "Persistent homology requires 'ripser' and 'persim'. "
            "Install via: pip install ripser persim"
        )

    def _graph_to_distance_matrix(sources, targets, weights, n):
        """Converts a weighted directed graph to a symmetric distance matrix
        suitable for Rips filtration."""
        # Use 1/weight as distance (strong edges = close)
        safe_weights = np.clip(weights, 1e-10, None)
        distances = 1.0 / safe_weights

        D = sp.coo_matrix((distances, (sources, targets)), shape=(n, n))
        D = D + D.T  # symmetrize
        D = D.toarray()

        # Set diagonal to 0
        np.fill_diagonal(D, 0.0)

        # Replace zeros (no edge) with a large value (disconnected)
        D[D == 0] = D.max() * 2.0
        np.fill_diagonal(D, 0.0)

        return D

    # Build distance matrices
    D_candidate = _graph_to_distance_matrix(
        sources_candidate, targets_candidate, weights_candidate, n_genes
    )
    D_reference = _graph_to_distance_matrix(
        sources_reference, targets_reference, weights_reference, n_genes
    )

    # Subsample for computational feasibility (>3000 nodes is too slow)
    max_nodes = 2000
    if n_genes > max_nodes:
        idx = np.random.choice(n_genes, max_nodes, replace=False)
        idx.sort()
        D_candidate = D_candidate[np.ix_(idx, idx)]
        D_reference = D_reference[np.ix_(idx, idx)]

    # Compute persistence diagrams
    result_cand = ripser(D_candidate, maxdim=max_dimension,
                         distance_matrix=True)
    result_ref = ripser(D_reference, maxdim=max_dimension,
                        distance_matrix=True)

    results = {}

    # 0D Wasserstein (connected components)
    dgm_cand_0 = result_cand['dgms'][0]
    dgm_ref_0 = result_ref['dgms'][0]
    # Remove infinite death times
    dgm_cand_0 = dgm_cand_0[np.isfinite(dgm_cand_0[:, 1])]
    dgm_ref_0 = dgm_ref_0[np.isfinite(dgm_ref_0[:, 1])]
    results['wasserstein_0d'] = float(wasserstein(dgm_cand_0, dgm_ref_0))

    # 1D Wasserstein (loops/cycles)
    if max_dimension >= 1:
        dgm_cand_1 = result_cand['dgms'][1]
        dgm_ref_1 = result_ref['dgms'][1]
        if len(dgm_cand_1) > 0 and len(dgm_ref_1) > 0:
            results['wasserstein_1d'] = float(wasserstein(dgm_cand_1, dgm_ref_1))
        else:
            results['wasserstein_1d'] = float('inf')

    return results


# =========================================================================
# Early Precision Ratio (Audit Phase Only)
# =========================================================================

def compute_epr(predicted_edges, ground_truth_edges, n_genes, top_fraction=0.1):
    """Computes the Early Precision Ratio (EPR) for a candidate graph.

    EPR = precision(top predicted edges) / precision(random predictor).
    An EPR <= 1.0 means the pruning logic performs no better than chance.

    Parameters
    ----------
    predicted_edges : set of (int, int)
        Edges in the pruned candidate graph.
    ground_truth_edges : set of (int, int)
        Known true causal edges (from perturbation data).
    n_genes : int
    top_fraction : float
        Fraction of predicted edges to evaluate (top by confidence).

    Returns
    -------
    float
        EPR value. Higher is better; <= 1.0 indicates shatter-level failure.
    """
    n_predicted = len(predicted_edges)
    if n_predicted == 0 or len(ground_truth_edges) == 0:
        return 0.0

    n_top = max(int(n_predicted * top_fraction), 1)
    # Since edges are already the "surviving" set, take all as top
    top_edges = predicted_edges

    true_positives = len(top_edges & ground_truth_edges)
    precision_model = true_positives / n_top

    # Random predictor precision
    max_possible_edges = n_genes * (n_genes - 1)
    precision_random = len(ground_truth_edges) / max(max_possible_edges, 1)

    if precision_random < 1e-12:
        return float('inf') if true_positives > 0 else 0.0

    epr = precision_model / precision_random
    return float(epr)
