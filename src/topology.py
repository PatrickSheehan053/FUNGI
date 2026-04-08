# src/topology.py
import numpy as np
import scipy.sparse as sp

def calculate_core_triangles(G_core: sp.csr_matrix) -> sp.csr_matrix:
    """
    Computes the number of closed triangles (Feed-Forward Loops)
    specifically for the edges that exist in the Diamond Core.
    """
    print("Calculating core topologies...")

    # 1. Create a binary adjacency matrix for pure structure
    A_core = G_core.copy()

    # OPTIMIZATION: Force int32 for clean, lightweight integer counting
    A_core.data = np.ones_like(A_core.data, dtype=np.int32)

    # 2. Sparse matrix multiplication (Computes all paths of length 2)
    # T_matrix[i, j] = number of intermediate nodes k where i -> k -> j
    T_matrix = A_core.dot(A_core)

    # 3. CRITICAL FIX: The Topological Mask
    # Element-wise multiplication keeps only paths of length 2 where a direct edge (i -> j) ALSO exists.
    # This isolates true Feed-Forward Loops (Triangles) and prevents RAM explosion via fill-in.
    T_matrix = T_matrix.multiply(A_core)

    print(f"✓ Triangle matrix computed. Non-zero entries: {T_matrix.nnz:,}")
    return T_matrix
