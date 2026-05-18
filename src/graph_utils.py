"""
FUNGI v10.0 -- Graph loading, structural metrics, and topology utilities.

v10.0: load_graph returns three values:
  G, sparse_mat, experimental_df

  sparse_mat      — Importance-weighted CSR. Always used for Phase 0 probes
                    and DASH W ranking. LightGBM Importance is the structural
                    backbone; experimental signal modifies within it, never
                    replaces it.
  experimental_df — DataFrame with extra experimental columns (combined_score,
                    md_score, stability, consensus_sign, sign_agreement,
                    pert_efficiency, method_votes) or None for standard GRNs.

v7.1 fix (retained): renames detected weight column to 'weight' so
nx.to_scipy_sparse_array picks up continuous importance values.
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


EXPERIMENTAL_COLS = [
    "combined_score", "md_score", "md_confidence", "stability",
    "consensus_sign", "sign_agreement", "pert_efficiency",
    "method_votes", "in_lgbm", "in_md",
]

def _load_edge_list(df, filepath_name):
    src_col, tgt_col, wt_col = _detect_edge_columns(df)
    print(f"  Columns detected: source='{src_col}', target='{tgt_col}'"
          + (f", weight='{wt_col}'" if wt_col else ""))

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

    if wt_col:
        w_data = sparse_mat.data
        n_unique = len(np.unique(w_data[:min(10000, len(w_data))]))
        if n_unique <= 1:
            print(f"  WARNING: Sparse matrix has only {n_unique} unique weight values!")
        else:
            print(f"  Weight range in sparse matrix: [{w_data.min():.4f}, {w_data.max():.4f}]")

    present_exp = [c for c in EXPERIMENTAL_COLS if c in df.columns]
    experimental_df = df[[src_col, tgt_col] + present_exp].copy() if present_exp else None
    if present_exp:
        print(f"  Experimental GRN columns detected: {present_exp}")

    return G, sparse_mat, experimental_df


def load_graph(filepath):
    filepath = Path(filepath)
    print(f"Loading graph from {filepath.name} ...")

    if filepath.suffix == ".npz":
        sparse_mat, genes = _load_npz(filepath)
        G = nx.from_scipy_sparse_array(sparse_mat, create_using=nx.DiGraph)
        if genes is not None:
            mapping = {i: gene for i, gene in enumerate(genes)}
            G = nx.relabel_nodes(G, mapping)
            print(f"  Node labels mapped from 'genes' array ({len(genes):,} entries).")
        experimental_df = None

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
        G, sparse_mat, experimental_df = _load_edge_list(df, filepath.name)

    elif filepath.suffix == ".parquet":
        print("  Detected Parquet format.")
        df = pd.read_parquet(filepath)
        G, sparse_mat, experimental_df = _load_edge_list(df, filepath.name)

    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats are .npz, .csv, and .parquet."
        )

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.4%}")
    return G, sparse_mat, experimental_df

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
