"""
FUNGI v6 -- Adaptive Thresholding Pre-filter.

Reduces the dense parent graph to a manageable candidate pool by retaining
only the top edges by weight.  Uses argpartition for O(N) edge selection.
"""

import numpy as np
import scipy.sparse as sp


def adaptive_threshold_filter(sparse_mat, target_density=0.20):
    """Reduces the graph to the exact target density by keeping only the
    top edges based on raw weights.

    Parameters
    ----------
    sparse_mat : scipy.sparse matrix
        Dense parent graph (any sparse format).
    target_density : float
        Fraction of the maximum possible edges to retain.

    Returns
    -------
    scipy.sparse.csr_matrix
        Filtered graph at the target density.
    """
    print(f"Filtering graph to {target_density * 100:.1f}% density...")

    num_nodes = sparse_mat.shape[0]
    target_edges = int(num_nodes * num_nodes * target_density)

    if sparse_mat.nnz <= target_edges:
        print("Graph is already at or below target density.")
        return sparse_mat.tocsr()

    coo_mat = sparse_mat.tocoo()

    surviving_indices = np.argpartition(coo_mat.data, -target_edges)[-target_edges:]

    filtered_matrix = sp.coo_matrix(
        (coo_mat.data[surviving_indices],
         (coo_mat.row[surviving_indices], coo_mat.col[surviving_indices])),
        shape=coo_mat.shape
    )

    filtered_csr = filtered_matrix.tocsr()
    actual_density = filtered_csr.nnz / (num_nodes * num_nodes)

    print(f"  Filtered to {filtered_csr.nnz:,} edges ({actual_density:.4%} density).")
    return filtered_csr
