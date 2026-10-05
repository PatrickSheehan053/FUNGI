"""
systema_metrics.py -- the core SYSTEMA edge-level decontamination metrics for
obj_004_systema_graph_eval.

The single conceptual move vs obj_003's stat_prec: obj_003 tests each top-K edge
R->T by R-knock test cells vs CONTROL test cells (Mann-Whitney on T) -- which credits
ANY significant shift, including the shared convergent response (cell-cycle arrest)
present under essentially every essential-gene knockdown. obj_004's spec_prec tests
R-knock vs the GLOBAL PERTURBED centroid cells (all perturbed test cells) -- so only
a REGULATOR-SPECIFIC residual counts. This is SYSTEMA's centroid subtraction applied
at the edge level (Viñas Torné et al. 2026).

  stat_prec@K  = frac(top-K edges with T significantly shifted in R-knock vs CONTROL)
  spec_prec@K  = frac(top-K edges with T significantly shifted in R-knock vs GLOBAL-PERT)
  sysvar_gap@K = stat_prec@K - spec_prec@K        (the contamination fraction)
  sysvar_cos@K = top-K mean of cosine(s_R, s_sys), s_R=mu_R-mu_ctrl, s_sys=mu_pert-mu_ctrl
                 (high => regulator's response aligns with the shared/systematic axis)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# obj_003.2 GPU kernel (shared, promoted to v2.2); lives in obj_003's src, imported by obj_004 from
# there. Imported only when device=='cuda'.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "obj_003_grn_bio_evaluator_v2" / "src"))
_EXACT_MIN_N = 8  # scipy uses the EXACT MWU null when min(n_ref,n_int)<=8 & tie-free -> route to scipy


def make_gpu_ctx(device, cooldown_ms=0):
    """obj_003.2 resident-GPU reference cache factory (shared design with obj_003)."""
    if device != "cuda":
        return None
    from gpu_context import GpuContext
    return GpuContext(device="cuda", cooldown_ms=cooldown_ms)


def _mwu(ref, intv, device):
    if device == "cuda" and min(ref.shape[0], intv.shape[0]) > _EXACT_MIN_N:
        from gpu_stat_kernels import mannwhitneyu_gpu
        return mannwhitneyu_gpu(ref, intv, device="cuda")
    _, p = stats.mannwhitneyu(ref, intv, axis=0)
    return np.asarray(p)


def _precision_vs_reference(edges_df, k, X_test, gene_to_col, pert_test, X_ref_test,
                            min_cells, p_threshold, device="cpu", gpu_ctx=None):
    """Generic top-K precision: fraction of top-K edges R->T whose target T is
    significantly shifted (Mann-Whitney p<thr) in R-knock test cells vs the reference
    test-cell set X_ref_test. X_ref_test=control -> stat_prec; =global-perturbed ->
    spec_prec. Exact structural clone of obj_003.stat_metrics.stat_precision_topk."""
    top = edges_df.head(k)
    n_not_in_panel = int((~top["Target"].isin(gene_to_col)).sum())
    valid = top[top["Target"].isin(gene_to_col)]
    n_tp = n_scored = n_skip_cov = 0
    for regulator, group in valid.groupby("Regulator", sort=False, observed=True):
        knock_mask = pert_test == regulator
        if int(knock_mask.sum()) < min_cells:
            n_skip_cov += len(group)
            continue
        tcols = np.asarray([gene_to_col[t] for t in group["Target"]])
        int_slice = X_test[knock_mask][:, tcols]
        n_knock = int(knock_mask.sum())
        # obj_003.2 resident-GPU path: the reference (X_ref_test = control OR ~28k global-perturbed) is
        # uploaded once and sliced on-device; only the small knock slice is uploaded. n_knock<=8->scipy.
        if gpu_ctx is not None and device == "cuda" and n_knock > _EXACT_MIN_N:
            pvals = gpu_ctx.mwu_cols(X_ref_test, tcols, int_slice)
        else:
            pvals = _mwu(X_ref_test[:, tcols], int_slice, device)
        n_tp += int((pvals < p_threshold).sum())
        n_scored += len(tcols)
    return dict(n_tp=n_tp, n_scored=n_scored, n_skip_cov=n_skip_cov,
                n_not_in_panel=n_not_in_panel, prec=(n_tp / k if k else 0.0),
                fraction_scored=round(n_scored / k, 6) if k else 0.0)


def spec_prec_topk(edges_df, k, X_test, gene_to_col, pert_test, X_pert_test,
                   min_cells, p_threshold, device="cpu", gpu_ctx=None):
    """spec_prec: R-knock vs the GLOBAL PERTURBED centroid cells (regulator-specific)."""
    return _precision_vs_reference(edges_df, k, X_test, gene_to_col, pert_test,
                                   X_pert_test, min_cells, p_threshold, device=device, gpu_ctx=gpu_ctx)


def stat_prec_topk(edges_df, k, X_test, gene_to_col, pert_test, X_ctrl_test,
                   min_cells, p_threshold, device="cpu", gpu_ctx=None):
    """stat_prec recomputed inline (R-knock vs CONTROL) for the contamination gap."""
    return _precision_vs_reference(edges_df, k, X_test, gene_to_col, pert_test,
                                   X_ctrl_test, min_cells, p_threshold, device=device, gpu_ctx=gpu_ctx)


def sysvar_cosine_topk(edges_df, k, reg_shift, sys_shift, cosine_on, gene_to_col):
    """top-K mean cosine of each edge regulator's shift s_R with the systematic axis
    s_sys. full_vector: cos over the whole panel shift (per-regulator, counted once per
    top-K edge from that regulator). target_component: project onto the edge's target
    gene only (sign-aware contribution along the systematic axis at that target)."""
    top = edges_df.head(k)
    sys_n = sys_shift / (np.linalg.norm(sys_shift) + 1e-12)
    if cosine_on == "full_vector":
        reg_cos = {r: float(np.dot(v, sys_n) / (np.linalg.norm(v) + 1e-12))
                   for r, v in reg_shift.items()}
        vals = [reg_cos[r] for r in top["Regulator"] if r in reg_cos]
    else:  # target_component
        vals = []
        for r, t in zip(top["Regulator"], top["Target"]):
            if r in reg_shift and t in gene_to_col:
                j = gene_to_col[t]
                v = reg_shift[r]
                # cosine of the (scalar) target components — sign of alignment at T
                denom = (abs(v[j]) * abs(sys_shift[j])) + 1e-12
                vals.append(float(v[j] * sys_shift[j] / denom)) if denom > 1e-11 else None
        vals = [x for x in vals if x is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)
