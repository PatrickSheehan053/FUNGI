"""
FUNGI v7.1 -- Graph loading, structural metrics, and topology utilities.

v7.1 fix: _load_edge_list now renames the detected weight column to 'weight'
before constructing the NetworkX graph. This ensures nx.to_scipy_sparse_array
picks up the actual continuous importance values instead of defaulting to 1.

Previous bug: nx.from_pandas_edgelist stored edge weights under the parquet's
column name (e.g. 'Importance'), but nx.to_scipy_sparse_array looks for an
attribute called 'weight' by default. When it didn't find 'weight', it
silently assigned 1 to every edge, destroying all LightGBM gain signal.
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
    """Handle all .npz variants and return (sparse_mat, genes_or_None)."""
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


# -------------------------------------------------------------------------
# Edge-list column auto-detection
# -------------------------------------------------------------------------

_SOURCE_ALIASES = [
    "source", "Source", "TF", "tf", "regulator", "Regulator",
    "gene1", "Gene1", "from", "src", "row",
]
_TARGET_ALIASES = [
    "target", "Target", "gene", "Gene", "gene2", "Gene2",
    "to", "tgt", "col", "dst",
]
_WEIGHT_ALIASES = [
    "weight", "Weight", "importance", "Importance", "score", "Score",
    "coef_mean", "coef", "edge_weight", "value",
]


def _detect_edge_columns(df):
    """Auto-detect source, target, and optional weight columns."""
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    def _find(aliases):
        for alias in aliases:
            if alias in cols:
                return alias
            if alias.lower() in cols_lower:
                return cols_lower[alias.lower()]
        return None

    src_col = _find(_SOURCE_ALIASES)
    tgt_col = _find(_TARGET_ALIASES)
    wt_col = _find(_WEIGHT_ALIASES)

    if src_col is None or tgt_col is None:
        if len(cols) == 2:
            src_col, tgt_col = cols[0], cols[1]
            wt_col = None
            print(f"  Positional fallback: source='{src_col}', target='{tgt_col}'")
        elif len(cols) >= 3:
            src_col, tgt_col, wt_col = cols[0], cols[1], cols[2]
            print(f"  Positional fallback: source='{src_col}', target='{tgt_col}', weight='{wt_col}'")
        else:
            raise ValueError(
                f"Cannot auto-detect source/target columns. "
                f"Available columns: {cols}. "
                f"Expected one of {_SOURCE_ALIASES[:5]} for source and "
                f"one of {_TARGET_ALIASES[:5]} for target."
            )

    return src_col, tgt_col, wt_col


def _load_edge_list(df, filepath_name):
    """Converts an edge-list DataFrame to (nx.DiGraph, scipy CSR matrix).

    v7.1 FIX: If the weight column is not already named 'weight', rename it
    before passing to nx.from_pandas_edgelist. This ensures the actual edge
    importance values are stored under the 'weight' attribute that
    nx.to_scipy_sparse_array expects by default.
    """
    src_col, tgt_col, wt_col = _detect_edge_columns(df)
    print(f"  Columns detected: source='{src_col}', target='{tgt_col}'"
          + (f", weight='{wt_col}'" if wt_col else ""))

    # Rename the weight column to 'weight' so NetworkX stores it correctly
    if wt_col is not None and wt_col != "weight":
        df = df.rename(columns={wt_col: "weight"})
        print(f"  Renamed '{wt_col}' -> 'weight' for NetworkX compatibility.")
        wt_col = "weight"

    edge_attr = "weight" if wt_col else None
    G = nx.from_pandas_edgelist(
        df, source=src_col, target=tgt_col,
        edge_attr=edge_attr, create_using=nx.DiGraph,
    )
    sparse_mat = nx.to_scipy_sparse_array(G)

    # Verify the weights were actually preserved
    if wt_col:
        w_data = sparse_mat.data
        n_unique = len(np.unique(w_data[:min(10000, len(w_data))]))
        if n_unique <= 1:
            print(f"  WARNING: Sparse matrix has only {n_unique} unique weight values!")
        else:
            print(f"  Weight range in sparse matrix: [{w_data.min():.4f}, {w_data.max():.4f}]")

    return G, sparse_mat


def load_graph(filepath):
    """Load a graph from disk and return (NetworkX DiGraph, scipy CSR matrix).

    Supported formats:
        .npz      -- NumPy/SciPy adjacency matrix archive
        .csv      -- Edge list (auto-detects column names)
        .parquet  -- Edge list (auto-detects column names, SPORE native)
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
        G, sparse_mat = _load_edge_list(df, filepath.name)

    elif filepath.suffix == ".parquet":
        print("  Detected Parquet format.")
        df = pd.read_parquet(filepath)
        G, sparse_mat = _load_edge_list(df, filepath.name)

    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats are .npz, .csv, and .parquet."
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
    """Computes the ratio lambda_1 / lambda_2 of the graph's adjacency matrix."""
    if len(sources) < n_eigenvalues + 1:
        return np.inf

    A = sp.csr_matrix((weights, (sources, targets)), shape=(n_genes, n_genes))

    try:
        eigenvalues = splinalg.eigsh(
            A.T + A,
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
    """Computes Moran's I autocorrelation statistic on the graph."""
    if len(sources) == 0 or n_genes < 3:
        return 0.0

    x = node_values.astype(np.float64)
    x_bar = x.mean()
    x_dev = x - x_bar

    ss = np.sum(x_dev ** 2)
    if ss < 1e-12:
        return 0.0

    W = len(sources)
    cross_sum = np.sum(x_dev[sources] * x_dev[targets])

    morans_i = (n_genes / W) * (cross_sum / ss)
    return float(morans_i)
