import numpy as np
import scipy.sparse as sp

def adaptive_threshold_filter(sparse_mat, target_density=0.20):
    """
    Reduces the graph to the exact target density by keeping only the top edges 
    based on raw weights. Uses optimized COO indexing to prevent memory overflow 
    and guarantees strict edge budgeting.
    """
    print(f"Filtering graph to {target_density*100}% density...")
    
    # CRITICAL: Must use shape[0] to extract the integer dimension, not the tuple
    num_nodes = sparse_mat.shape[0]
    
    # Assumes self-loops (N^2) are permitted in the raw adjacency matrix.
    # If self-loops are strictly forbidden downstream, change to: num_nodes * (num_nodes - 1)
    target_edges = int(num_nodes * num_nodes * target_density)
    
    if sparse_mat.nnz <= target_edges:
        print("Graph is already at or below target density.")
        return sparse_mat.tocsr()
    
    # Convert to COO format for safe coordinate indexing
    coo_mat = sparse_mat.tocoo()
    
    # THE FIX: Use argpartition to grab the exact indices of the top N edges.
    # This guarantees the graph is pruned to exactly the target density, 
    # bypassing the risk of keeping excess edges due to duplicate boundary probabilities.
    surviving_indices = np.argpartition(coo_mat.data, -target_edges)[-target_edges:]
    
    # Build the new filtered matrix directly from the surviving indices
    filtered_matrix = sp.coo_matrix(
        (coo_mat.data[surviving_indices], 
        (coo_mat.row[surviving_indices], coo_mat.col[surviving_indices])),
        shape=coo_mat.shape
    )
    
    # Convert back to CSR for fast downstream row-operations
    filtered_csr = filtered_matrix.tocsr()
    actual_density = filtered_csr.nnz / (num_nodes * num_nodes)
    
    print(f"✓ Filtered to {filtered_csr.nnz:,} edges ({actual_density:.4%} density).")
    return filtered_csr
