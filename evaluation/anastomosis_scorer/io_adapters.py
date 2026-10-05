"""
obj_010 ANASTOMOSIS — io_adapters.py : one adapter per input serialization -> the canonical ScoredUnit.

Mode B (mean_lfc, RHIZO contract): npz with keys {panel,N,signal_mask,mu_ctrl,<split>_names,<split>_LFC,<split>_pidx}.
  means_* reconstructed as mu_ctrl + LFC_* (exact; see contract_notes.md). A single npz is the TRUE answer key; a
  prediction is a second npz in the same contract (pred_path). If no pred is given, means_pred := means_true and
  is_self flagged (used only for dataset-property / null scoring, never as a real prediction score).
Mode A (per_cell, proto layout): pred_{seed}.h5ad / true_{seed}.h5ad (+ control pool). Carries cells for the
  cell-only tiers. (Loader implemented; exercised in the GPU window.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class ScoredUnit:
    means_true: np.ndarray            # (n_pert, n_gene)
    means_pred: np.ndarray            # (n_pert, n_gene)
    mu_ctrl: np.ndarray               # (n_gene,)
    signal_mask: np.ndarray           # (n_gene,) bool
    pert_gene_idx: np.ndarray         # (n_pert,) panel idx, -1 if off-panel
    gene_names: np.ndarray
    split: str
    coverage_frac: float = 1.0
    input_kind: str = "mean_lfc"
    is_self: bool = False             # pred==true (dataset-property only)
    # cell-only (Mode A); None in Mode B
    X_cells_true: Optional[np.ndarray] = None
    X_cells_pred: Optional[np.ndarray] = None
    X_ctrl_pool: Optional[np.ndarray] = None
    # provenance passthrough
    meta: dict = field(default_factory=dict)

    @property
    def n_pert(self): return self.means_true.shape[0]
    @property
    def n_gene(self): return self.means_true.shape[1]
    @property
    def has_cells(self): return self.X_cells_true is not None and self.X_cells_pred is not None


def _splits_in(npz):
    return [k[:-4] for k in npz.files if k.endswith("_LFC")]


def load_mean_lfc(true_path, split="test", pred_path=None) -> ScoredUnit:
    """Mode B. means = mu_ctrl + LFC (exact reconstruction). If pred_path given, its LFC is means_pred."""
    z = np.load(true_path, allow_pickle=True)
    if split not in _splits_in(z):
        raise ValueError(f"split '{split}' not in {true_path} (have {_splits_in(z)})")
    panel = np.asarray(z["panel"]); mu = np.asarray(z["mu_ctrl"], np.float64)
    sig = np.asarray(z["signal_mask"]).astype(bool)
    lfc_t = np.asarray(z[f"{split}_LFC"], np.float64); pidx = np.asarray(z[f"{split}_pidx"], np.int64)
    means_t = mu[None, :] + lfc_t
    if pred_path is not None:
        zp = np.load(pred_path, allow_pickle=True)
        lfc_p = np.asarray(zp[f"{split}_LFC"], np.float64)
        assert lfc_p.shape == lfc_t.shape, f"pred/true LFC shape mismatch {lfc_p.shape} vs {lfc_t.shape}"
        means_p = mu[None, :] + lfc_p; is_self = False
    else:
        means_p = means_t.copy(); is_self = True
    N = int(z["N"]) if "N" in z.files else len(panel)
    cov = float(np.mean(pidx >= 0)) if len(pidx) else 0.0
    return ScoredUnit(means_true=means_t, means_pred=means_p, mu_ctrl=mu, signal_mask=sig,
                      pert_gene_idx=pidx, gene_names=panel, split=split, coverage_frac=cov,
                      input_kind="mean_lfc", is_self=is_self,
                      meta=dict(true_path=str(true_path), pred_path=str(pred_path) if pred_path else None, N=N))


def reconstruct_gate(true_path, split="test") -> float:
    """Exactness: means_true - mu_ctrl must equal the stored LFC to 0. Returns max|diff|."""
    z = np.load(true_path, allow_pickle=True)
    u = load_mean_lfc(true_path, split=split)
    return float(np.abs((u.means_true - u.mu_ctrl[None, :]) - np.asarray(z[f"{split}_LFC"], np.float64)).max())


def load_per_cell(directory, seed, pert_col="gene", control_label="non-targeting") -> ScoredUnit:
    """Mode A. pred_{seed}.h5ad / true_{seed}.h5ad -> per-pert means + retained cells (cell tiers)."""
    import scanpy as sc
    directory = Path(directory)
    tp = directory / f"true_{seed}.h5ad"; pp = directory / f"pred_{seed}.h5ad"
    at = sc.read_h5ad(tp); ap = sc.read_h5ad(pp)
    genes = list(at.var_names)
    def dense(X): return X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    tg = at.obs[pert_col].astype(str).to_numpy()
    perts = [p for p in np.unique(tg) if p != control_label]
    mu = dense(at[tg == control_label].X).mean(0) if (tg == control_label).any() else dense(at.X).mean(0)
    def pmeans(ad, col):
        g = ad.obs[pert_col].astype(str).to_numpy()
        return np.stack([dense(ad[g == p].X).mean(0) if (g == p).any() else np.full(len(genes), np.nan) for p in perts])
    mt, mp = pmeans(at, pert_col), pmeans(ap, pert_col)
    g2i = {g: i for i, g in enumerate(genes)}
    pidx = np.array([g2i.get(p, -1) for p in perts], np.int64)
    sig = (np.abs(mt - mu[None, :]) >= 0.25).any(0)
    return ScoredUnit(means_true=mt, means_pred=mp, mu_ctrl=mu, signal_mask=sig, pert_gene_idx=pidx,
                      gene_names=np.array(genes), split="test", input_kind="per_cell",
                      X_cells_true=dense(at.X), X_cells_pred=dense(ap.X),
                      X_ctrl_pool=dense(at[tg == control_label].X) if (tg == control_label).any() else None,
                      meta=dict(dir=str(directory), seed=int(seed)))


ADAPTERS = {"mean_lfc": load_mean_lfc, "per_cell": load_per_cell}
