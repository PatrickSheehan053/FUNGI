"""
FUNGI v6 -- Graph loading, structural metrics, and topology utilities.

Expanded from v5 to include spectral gap computation and spatial coherence
(Moran's I) for the upgraded shatter criteria.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import networkx as nx
from pathlib import Path


# -------------------------------------------------------------------------
# Graph ingestion
# -------------------------------------------------------------------------

def _load_npz(filepath):
    """Handle all .npz variants and return (sparse_mat, genes_or_None).

    Priority order:
    1. Keys ``adj_matrix`` + ``genes`` -- FUNGI / GuanLab dense format.
    2. Valid SciPy sparse archive (contains ``format`` key).
    3. Generic NumPy archive -- use the first 2D array as the adjacency
       matrix and, if present, a ``genes`` array for node labels.
    """
    loaded = np.load(filepath, allow_pickle=True)
    keys = loaded.files

    if "adj_matrix" in keys and "genes" in keys:
        print("  Detected adj_matrix + genes format.")
        return sp.csr_matrix(loaded["adj_matrix"]), loaded["genes"]

    if "format" in keys:
        print("  Detected SciPy sparse format.")
        return sp.load_npz(filepath), None

    mat = None
    _gene_keys = ["genes", "gene_names", "node_names", "labels"]
    genes = None
    for gk in _gene_keys:
        if gk in keys:
            genes = loaded[gk]
            break

    for k in keys:
        arr = loaded[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            mat = arr
            print(f"  Using key '{k}' as adjacency matrix (shape {arr.shape}).")
            break

    if mat is None:
        available = {k: loaded[k].shape for k in keys if isinstance(loaded[k], np.ndarray)}
        raise ValueError(
            f"Could not locate a 2D adjacency matrix in {filepath.name}. "
            f"Available arrays: {available}"
        )

    return sp.csr_matrix(mat), genes


def load_graph(filepath):
    """Load a graph from disk and return (NetworkX DiGraph, scipy CSR matrix).

    Supported formats: .npz, .csv (edge list with source/target columns).
    """
    filepath = Path(filepath)
    print(f"Loading graph from {filepath.name} ...")

    if filepath.suffix == ".npz":
        sparse_mat, genes = _load_npz(filepath)
        G = nx.from_scipy_sparse_array(sparse_mat, create_using=nx.DiGraph)

        if genes is not None:
            mapping = {i: gene for i, gene in enumerate(genes)}
            G = nx.relabel_nodes(G, mapping)
            print(f"  Node labels mapped from 'genes' array ({len(genes):,} entries).")

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
        G = nx.from_pandas_edgelist(
            df, source="source", target="target",
            edge_attr=True, create_using=nx.DiGraph,
        )
        sparse_mat = nx.to_scipy_sparse_array(G)
    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats are .npz and .csv."
        )

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.4%}")
    return G, sparse_mat


# -------------------------------------------------------------------------
# Structural metrics
# -------------------------------------------------------------------------

def calculate_gini(degree_sequence):
    """Compute the Gini coefficient of a degree sequence."""
    arr = np.sort(np.asarray(degree_sequence, dtype=np.float64))
    if arr.sum() == 0:
        return 0.0
    n = arr.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))


# -------------------------------------------------------------------------
# Spectral gap computation (for shatter criteria)
# -------------------------------------------------------------------------

def compute_spectral_dominance_ratio(sources, targets, weights, n_genes,
                                     n_eigenvalues=3):
    """Computes the ratio lambda_1 / lambda_2 of the graph's adjacency matrix.

    A very large ratio indicates spectral masking: the dominant eigenvalue
    absorbs all local topological variation, making the graph useless for
    downstream GNN inference.

    Parameters
    ----------
    sources, targets : np.ndarray
        Edge source and target indices.
    weights : np.ndarray
        Edge weights.
    n_genes : int
        Total number of nodes.
    n_eigenvalues : int
        Number of leading eigenvalues to compute.

    Returns
    -------
    float
        Ratio lambda_1 / lambda_2.  Returns np.inf if lambda_2 is zero.
    """
    if len(sources) < n_eigenvalues + 1:
        return np.inf

    A = sp.csr_matrix((weights, (sources, targets)), shape=(n_genes, n_genes))

    try:
        eigenvalues = splinalg.eigsh(
            A.T + A,  # symmetrize for real eigenvalues
            k=min(n_eigenvalues, n_genes - 2),
            which='LM',
            return_eigenvectors=False
        )
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        if len(eigenvalues) < 2 or eigenvalues[1] < 1e-12:
            return np.inf

        return float(eigenvalues[0] / eigenvalues[1])

    except Exception:
        return np.inf


# -------------------------------------------------------------------------
# Spatial coherence: Moran's I (for shatter criteria)
# -------------------------------------------------------------------------

def compute_morans_i(sources, targets, node_values, n_genes):
    """Computes Moran's I autocorrelation statistic on the graph.

    Measures whether neighboring genes provide rank-consistent regulatory
    signals.  A value near zero or negative indicates a "house of cards"
    topology where node values are uncorrelated with their neighbors.

    Parameters
    ----------
    sources, targets : np.ndarray
        Edge source and target indices.
    node_values : np.ndarray
        Per-gene attribute (e.g., out-degree or expression variance).
    n_genes : int
        Total number of nodes.

    Returns
    -------
    float
        Moran's I statistic.  Positive = spatially clustered,
        near zero = random, negative = dispersed.
    """
    if len(sources) == 0 or n_genes < 3:
        return 0.0

    x = node_values.astype(np.float64)
    x_bar = x.mean()
    x_dev = x - x_bar

    ss = np.sum(x_dev ** 2)
    if ss < 1e-12:
        return 0.0

    W = len(sources)  # total weight (unweighted: number of edges)

    # Sum of cross-deviation products for connected pairs
    cross_sum = np.sum(x_dev[sources] * x_dev[targets])

    morans_i = (n_genes / W) * (cross_sum / ss)
    return float(morans_i)
