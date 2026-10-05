"""
obj_008 — dwlp.py  (Directed Weighted Linear Propagation)

The instrument. NO node features. The only path from "gene p was perturbed" to "gene t changed" is through
the graph edges.

  inject   h^(0) = e_p * delta_p           (localized shift on the perturbed node only)
  propagate h^(k) = A-tilde @ h^(k-1)       (K hops, direction-preserving normalized adjacency)
  readout  s_hat_p[g] = w_g . [h^(1)_g, ..., h^(K)_g]     (independent per-gene ridge, NO intercept)
  target   s_p = Delta_p - mu_bar           (train-only specific residual)

Load-bearing property: gene g's readout only ever sees g's OWN propagated features, which are 0 if no
directed walk of length <=K reaches g. Empty graph => A-tilde has only self-loops => zero specific signal
for every t != p, by construction.

Everything is vectorized (einsum over tiny K x K systems); CPU-only, seconds per arm.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def make_delta(pert_gene_idx: np.ndarray, mu_ctrl: np.ndarray, mode: str = "neg_mu_ctrl") -> np.ndarray:
    """delta_p for each perturbation. Invalid (perturbed gene not a node, idx<0) -> 0 (no injection)."""
    d = np.zeros(len(pert_gene_idx), dtype=np.float64)
    valid = pert_gene_idx >= 0
    idx = pert_gene_idx[valid]
    if mode == "neg_mu_ctrl":
        d[valid] = -mu_ctrl[idx]
    elif mode == "neg_half_mu_ctrl":
        d[valid] = -0.5 * mu_ctrl[idx]
    elif mode == "neg_one":
        d[valid] = -1.0
    else:
        raise ValueError(f"unknown delta_mode {mode!r}")
    return d


def propagate(At: sp.csr_matrix, pert_gene_idx: np.ndarray, delta: np.ndarray, K: int) -> np.ndarray:
    """Return feats of shape (K, N, P): feats[k, g, i] = h^(k+1)_g for perturbation i.
    H^(0) is a sparse N x P matrix with delta at (perturbed node, pert col)."""
    N = At.shape[0]
    P = len(pert_gene_idx)
    valid = pert_gene_idx >= 0
    cols = np.where(valid)[0]
    rows = pert_gene_idx[valid]
    H = np.zeros((N, P), dtype=np.float64)
    H[rows, cols] = delta[valid]
    feats = np.empty((K, N, P), dtype=np.float64)
    for k in range(K):
        H = At @ H
        feats[k] = H
    return feats


def _features_matrix(feats: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    """(K, N, P) -> (P, |gene_idx|, K) feature tensor for the requested genes."""
    sub = feats[:, gene_idx, :]                 # (K, G, P)
    return np.transpose(sub, (2, 1, 0))         # (P, G, K)


class DWLP:
    """Directed Weighted Linear Propagation with an independent per-gene ridge readout."""

    def __init__(self, At, mu_ctrl, signal_mask, K=3, alpha=1.0, delta_mode="neg_mu_ctrl",
                 fit_signal_only=True):
        self.At = At.tocsr()
        self.mu_ctrl = np.asarray(mu_ctrl, float)
        self.signal_mask = np.asarray(signal_mask, bool)
        self.K = int(K)
        self.alpha = float(alpha)
        self.delta_mode = delta_mode
        self.fit_signal_only = fit_signal_only
        self.N = self.At.shape[0]
        # genes we actually fit/predict a readout for (others predict 0; metrics only score S anyway)
        self.fit_idx = np.where(self.signal_mask)[0] if fit_signal_only else np.arange(self.N)
        self.W = None  # (|fit_idx|, K)

    def fit(self, pert_gene_idx_train, s_train):
        delta = make_delta(pert_gene_idx_train, self.mu_ctrl, self.delta_mode)
        feats = propagate(self.At, pert_gene_idx_train, delta, self.K)
        X = _features_matrix(feats, self.fit_idx)          # (P, G, K)
        y = s_train[:, self.fit_idx]                       # (P, G)
        # per-gene ridge, no intercept: W_g = (X_g^T X_g + alpha I)^-1 X_g^T y_g
        Gram = np.einsum("pgk,pgl->gkl", X, X)             # (G, K, K)
        Gram += self.alpha * np.eye(self.K)[None, :, :]
        Xty = np.einsum("pgk,pg->gk", X, y)                # (G, K)
        self.W = np.linalg.solve(Gram, Xty[:, :, None])[:, :, 0]  # (G, K)
        return self

    def predict(self, pert_gene_idx):
        assert self.W is not None, "call fit() first"
        delta = make_delta(pert_gene_idx, self.mu_ctrl, self.delta_mode)
        feats = propagate(self.At, pert_gene_idx, delta, self.K)
        X = _features_matrix(feats, self.fit_idx)          # (P, G, K)
        pred_s = np.einsum("pgk,gk->pg", X, self.W)        # (P, G)
        S_pred = np.zeros((len(pert_gene_idx), self.N), dtype=np.float64)
        S_pred[:, self.fit_idx] = pred_s
        return S_pred


def parameter_free_predict(At, pert_gene_idx, mu_ctrl, beta, delta_mode="neg_mu_ctrl"):
    """No learned head: s_hat_p[g] = sum_k beta_k * h^(k)_g. Used by the go/no-go probe."""
    beta = np.asarray(beta, float)
    K = len(beta)
    delta = make_delta(pert_gene_idx, mu_ctrl, delta_mode)
    feats = propagate(At, pert_gene_idx, delta, K)          # (K, N, P)
    S_pred = np.einsum("k,kgi->ig", beta, feats)            # (P, N)
    return S_pred
