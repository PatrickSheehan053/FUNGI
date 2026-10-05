"""
FUNGI — shared housekeeping utilities.

Two helpers used by both organic and synthetic modes:

  build_gene_features(adata, ...)   -> (n_genes, D) unit-norm feature matrix
      Per-gene feature vectors for feature-based heterophily (synthetic loss)
      and feature-aware SCBER. L2-normalised rows so a row dot product equals
      cosine similarity.

  build_kernel_flags(cfg)           -> dict of 8 on/off flags
      Assembles the modular DASH kernel switches from the config 'dash_kernel'
      block. Every factor defaults to ON. Consumed by
      engine.run_dash_and_score(kernel_flags=...).
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Per-gene feature matrix (expression PCA, unit-normalised)
# ---------------------------------------------------------------------------

def build_gene_features(adata, n_components=50, seed=42, verbose=True):
    """
    Build a per-gene feature matrix from the expression data.

    Each gene's feature vector is its expression profile across metacells,
    reduced with truncated SVD to n_components and L2-normalised so that a row
    dot product is exactly the cosine similarity between two genes. Used by:
      - calculate_synthetic_loss (feature-based heterophily = mean cosine
        distance across edges), and
      - compute_scber_scores (feature-aware bridge re-weighting, GOKU-style).

    Note: the scGPT embeddings SPECTRA actually consumes would be a more
    faithful feature space. Expression PCA is the proxy available inside the
    FUNGI pipeline and captures the co-expression structure that cosine
    (dis)similarity between connected genes is meant to reflect.

    Parameters
    ----------
    adata        : AnnData — cells (or metacells) x genes
    n_components : int      — SVD dimensionality (default 50)
    seed         : int
    verbose      : bool

    Returns
    -------
    feats : np.ndarray float64 (n_genes, k) — unit-norm rows
    """
    from sklearn.decomposition import TruncatedSVD

    X = adata.X
    # genes as rows: (n_genes, n_cells)
    GxC = X.T
    if sp.issparse(GxC):
        GxC = GxC.tocsr().astype(np.float64)
    else:
        GxC = np.asarray(GxC, dtype=np.float64)

    n_genes = GxC.shape[0]
    n_cells = GxC.shape[1]
    k = int(min(n_components, max(n_cells - 1, 1), max(n_genes - 1, 1)))
    k = max(k, 2)

    svd = TruncatedSVD(n_components=k, random_state=seed)
    feats = svd.fit_transform(GxC)  # (n_genes, k)

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = (feats / norms).astype(np.float64)

    if verbose:
        ev = float(svd.explained_variance_ratio_.sum())
        print(f"  Gene features: {n_genes} genes x {k} dims "
              f"(expression SVD, {ev*100:.1f}% variance, unit-norm rows)")
    return feats


# ---------------------------------------------------------------------------
# Modular DASH kernel flags
# ---------------------------------------------------------------------------

_KERNEL_FACTORS = ['weight', 'ffl', 'pert_impact', 'rdf',
                   'scber', 'chi_s', 'chi_t', 'rho']

_KERNEL_FACTOR_DESC = {
    'weight':      'W_q^beta  (LightGBM backbone)',
    'ffl':         'exp(delta*T)  (feed-forward-loop motif)',
    'pert_impact': 'pi_s^psi  (perturbation impact prior)',
    'rdf':         'RDF_s^nu  (regulatory diversity factor)',
    'scber':       'R^eta_bridge  (source-conditioned bridge ER)',
    'chi_s':       'chi_s^zeta_chi  (source pleiotropy prior)',
    'chi_t':       'chi_t^zeta_chi  (target pleiotropy prior)',
    'rho':         'rho_s  (causal output-efficiency prior)',
}


def build_kernel_flags(cfg, verbose=True):
    """
    Assemble the DASH kernel on/off flags from the config 'dash_kernel' block.

    Each factor in the multiplicative DASH chain can be switched off; when off
    it contributes 1.0 (neutral). Every factor defaults to ON if unspecified,
    so omitting the block (or any entry) reproduces the full kernel.

    Accepts either form per factor:
        weight: true
        weight: {enabled: true}

    Returns
    -------
    flags : dict[str, bool] — one entry per kernel factor
    """
    dk = (cfg or {}).get('dash_kernel', {}) or {}
    flags = {}
    for f in _KERNEL_FACTORS:
        entry = dk.get(f, True)
        if isinstance(entry, dict):
            flags[f] = bool(entry.get('enabled', True))
        else:
            flags[f] = bool(entry)

    if verbose:
        on = [f for f in _KERNEL_FACTORS if flags[f]]
        off = [f for f in _KERNEL_FACTORS if not flags[f]]
        print(f"  DASH kernel factors ON ({len(on)}/8): {', '.join(on)}")
        if off:
            print(f"  DASH kernel factors OFF: {', '.join(off)}")
            for f in off:
                print(f"      - {f}: {_KERNEL_FACTOR_DESC[f]}  -> factor = 1.0")
    return flags


def kernel_flags_summary(flags):
    """One-line string of enabled factors, for logging/provenance."""
    return "+".join(f for f in _KERNEL_FACTORS if flags.get(f, True))