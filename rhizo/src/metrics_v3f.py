"""
obj_009.3 (RHIZO-final) — metrics_v3f.py : the obj_009.2 scoreboard re-exported UNCHANGED (RSC, dsRSC_ge2,
dnsa_ge2hop, paired_bootstrap, ...) PLUS one new ADDED honest axis:

  rsc_coexpr_resid — the CO-EXPRESSION-RESIDUALIZED RSC. Aggregate RSC rewards co-expression (which is exactly
  why an undirected co-expression graph like kNN wins in-distribution — CausalBench / geneRNIB). This metric
  removes the component of the prediction↔response agreement that a Pearson CO-EXPRESSION baseline already
  explains, and reports what remains: the partial correlation  corr(pred, true | coexpr_baseline)  per
  perturbation. It isolates the CAUSAL / directed-regulatory component where FUNGI's structure should beat kNN.
  This is an ADDED axis, NOT a replacement for aggregate RSC.

LEAKAGE-SAFE: the co-expression baseline is built from the FIT-split response signatures ONLY (train-only), and
is graph-AGNOSTIC (identical for every arm) so the graph-swap stays fair. Mean-baseline (pred≡0) -> 0 (partialling
a constant-0 prediction leaves a constant-0 residual -> corr 0), so the metric respects the ≤0 mean-baseline gate.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "clone"))
from metrics_v3 import (  # noqa: F401  re-export the obj_009.2 scoreboard byte-identical
    rsc_per_pert, dnsa, weighted_r2_per_pert, topk_sign_acc, auprc_per_pert,
    paired_bootstrap, one_sample_bootstrap, wilcoxon_paired, fisher_mean, _pearson,
    dsrsc_per_pert, reach_at_k_per_pert, eff_resistance_per_pert, dnsa_ge2hop_per_pert, true_top_degs,
    UNREACH,
)


def build_coexpr_Z(s_fit, signal_mask=None):
    """Standardize each gene's response profile across the FIT perturbations -> Z (n_fit, N). The co-expression
    baseline for perturbing gene g at target j is coexpr(g,j) = (Z[:,g] . Z[:,j]) / n_fit. Built train-only."""
    Z = np.asarray(s_fit, np.float64).copy()
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Z - mu) / sd
    return Z


def _residualize(y, b):
    """Residual of y after removing the linear component explained by [1, b] (OLS). If b is ~constant, this is
    just mean-centering (=> partial corr reduces to plain corr)."""
    y = np.asarray(y, np.float64)
    b = np.asarray(b, np.float64)
    if b.std() < 1e-12:
        return y - y.mean()
    X = np.column_stack([np.ones_like(b), b])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def rsc_coexpr_resid_per_pert(S_true, S_pred, signal_mask, pert_gene_idx, Zfit, n_fit=None):
    """Per-pert partial correlation corr(pred, true | coexpr_baseline) over signal genes, excluding the
    perturbed gene. Zfit = build_coexpr_Z(fit split). Returns array[n_pert] (nan where not scorable)."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    if Zfit is None:
        # no baseline available -> fall back to plain RSC (still a valid, if un-residualized, axis)
        return rsc_per_pert(S_true, S_pred, signal_mask, pert_gene_idx)
    nf = float(Zfit.shape[0] if n_fit is None else n_fit)
    for i in range(n):
        g = pert_gene_idx[i]
        cols = sig[sig != g] if g >= 0 else sig
        if len(cols) < 3:
            continue
        yt = S_true[i, cols]; yp = S_pred[i, cols]
        if g >= 0:
            b = (Zfit[:, g] @ Zfit[:, cols]) / nf          # coexpr(g, cols): the co-expression baseline
        else:
            b = np.zeros(len(cols))
        rt = _residualize(yt, b); rp = _residualize(yp, b)
        out[i] = _pearson(rp, rt)
    return out
