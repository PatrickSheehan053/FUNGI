"""
FUNGI v3 — Graph loading and structural utilities.

Functions in this module were originally defined inline in the pruning
pipeline notebook.  They are collected here for reuse and testability.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from pathlib import Path


# -------------------------------------------------------------------------
# Graph ingestion
# -------------------------------------------------------------------------

def _load_npz(filepath):
    """Handle all .npz variants and return (sparse_mat, genes_or_None).

    Priority order:
    1. Keys ``adj_matrix`` + ``genes`` — FUNGI v3 / GuanLab dense format.
    2. Valid SciPy sparse archive (contains ``format`` key).
    3. Generic NumPy archive — use the first 2D array as the adjacency
       matrix and, if present, a ``genes`` array for node labels.
    """
    loaded = np.load(filepath, allow_pickle=True)
    keys = loaded.files

    # --- Case 1: explicit adj_matrix + genes (FUNGI v3 / GuanLab) ----------
    if "adj_matrix" in keys and "genes" in keys:
        print("  Detected adj_matrix + genes format.")
        return sp.csr_matrix(loaded["adj_matrix"]), loaded["genes"]

    # --- Case 2: native SciPy sparse archive -------------------------------
    if "format" in keys:
        print("  Detected SciPy sparse format.")
        # Re-load through SciPy's own reader so indices are handled correctly
        return sp.load_npz(filepath), None

    # --- Case 3: generic NumPy archive (find the first 2D array) -----------
    mat = None

    # Look for gene/node labels under common key names
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

    Supported formats
    -----------------
    * ``.npz`` — NumPy archive containing a 2D adjacency matrix (dense or
      sparse).  Recognized variants:

      - keys ``adj_matrix`` + ``genes`` (FUNGI v3 / GuanLab dense format)
      - native SciPy sparse archive (saved with ``scipy.sparse.save_npz``)
      - generic archive with any 2D array + optional node-label key
        (checks ``genes``, ``gene_names``, ``node_names``, ``labels``)

    * ``.csv`` — edge-list table with columns ``source``, ``target``, and
      optional edge attributes.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the graph file on disk.

    Returns
    -------
    G : nx.DiGraph
        Directed graph with gene-name node labels (when available).
    sparse_mat : scipy.sparse.csr_matrix
        CSR adjacency matrix aligned with the node ordering of *G*.
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
    """Compute the Gini coefficient of a degree sequence.

    Parameters
    ----------
    degree_sequence : array-like
        Non-negative degree values.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    arr = np.sort(np.asarray(degree_sequence, dtype=np.float64))
    if arr.sum() == 0:
        return 0.0
    n = arr.shape[0]
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))