"""
stat_metrics.py -- held-out statistical/causal metrics for
obj_003_grn_bio_evaluator_v2: stat_prec, wass_test, and the new FOR@K.

stat_prec/wass_test are a direct adaptation of exp_003b's validated
stat_precision_evaluator.py::stat_precision_topk (matches exp_003b/
results_summary.csv to within numerical noise -- see Test 1 in the design
doc). The one change from that script: `groupby("Regulator", sort=False)`
gains `observed=True` explicitly here too (the categorical-dtype phantom-
empty-group bug that crashed exp_003b's first draft -- already fixed there,
carried forward correctly here, not reintroduced).

FOR@K (False Omission Rate) is new: it is the recall-side complement to
stat_prec. stat_prec asks "of the model's own top-K predictions, how many
are causally supported on held-out data" (a precision question). FOR@K asks
"of the gold-standard pairs that COULD have been tested (regulator has
held-out perturbation coverage), how many did the model's top-K miss" (a
recall/omission question) -- it uses the external gold standard, not the
model's own predictions, as the basis set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# obj_003.2 GPU kernels (shared, promoted to v2.2); live beside this file in obj_003's src, so both
# obj_003 and obj_004 import them from here. Only imported when --device cuda is actually used.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scipy uses the EXACT MWU null when min(n_ref,n_int) <= this AND the column is tie-free; below/at
# it we route the whole regulator to scipy (handles exact + asymptotic-with-ties correctly). Above
# it scipy ALWAYS uses asymptotic, which mannwhitneyu_gpu matches (validated: 0 stat_prec flips).
_EXACT_MIN_N = 8


def make_gpu_ctx(device, cooldown_ms=0):
    """Factory for the obj_003.2 resident-GPU reference cache (gpu_context). Returns a GpuContext when
    device=='cuda', else None. Created ONCE per evaluator so each reference matrix uploads a single
    time (reused across all k and all regulators)."""
    if device != "cuda":
        return None
    from gpu_context import GpuContext
    return GpuContext(device="cuda", cooldown_ms=cooldown_ms)


def _mwu(ref, intv, device):
    """MWU p-values [C] == scipy.stats.mannwhitneyu(ref, intv, axis=0). GPU when device=='cuda' and
    min(n_ref,n_int) > 8; else scipy (safe default + exact-regime fallback). This per-slice path is the
    fallback when no gpu_ctx is threaded in; the resident-reference path is preferred (see below)."""
    if device == "cuda" and min(ref.shape[0], intv.shape[0]) > _EXACT_MIN_N:
        from gpu_stat_kernels import mannwhitneyu_gpu
        return mannwhitneyu_gpu(ref, intv, device="cuda")
    _, p = stats.mannwhitneyu(ref, intv, axis=0)
    return np.asarray(p)


def _wass_cols(ref, intv, device):
    """1-D Wasserstein per column == scipy.stats.wasserstein_distance per target. Exact for any n, so
    GPU whenever device=='cuda'."""
    if device == "cuda":
        from gpu_stat_kernels import wasserstein1d_gpu
        return list(wasserstein1d_gpu(ref, intv, device="cuda"))
    return [stats.wasserstein_distance(ref[:, j], intv[:, j]) for j in range(ref.shape[1])]


def stat_precision_topk(edges_df: pd.DataFrame, k: int, X_test: np.ndarray, gene_to_col: dict,
                         pert_test: np.ndarray, X_ctrl_test: np.ndarray,
                         min_cells: int, p_threshold: float, device: str = "cpu",
                         gpu_ctx=None) -> dict:
    top = edges_df.head(k)
    n_skipped_target_not_in_panel = int((~top["Target"].isin(gene_to_col)).sum())
    valid = top[top["Target"].isin(gene_to_col)]

    n_true_positive = 0
    n_scored = 0
    n_skipped_no_coverage = 0
    wass_list = []

    for regulator, group in valid.groupby("Regulator", sort=False, observed=True):
        knock_mask = pert_test == regulator
        n_knock = int(knock_mask.sum())
        if n_knock < min_cells:
            n_skipped_no_coverage += len(group)
            continue
        X_knock_test = X_test[knock_mask]
        target_cols = np.asarray([gene_to_col[t] for t in group["Target"]])
        int_slice = X_knock_test[:, target_cols]
        # obj_003.2 resident-GPU path: reference (X_ctrl_test) already on-device -> slice cols on-device,
        # only the small knock slice is uploaded. n_knock<=8 still routes to scipy (exact-null regime).
        if gpu_ctx is not None and device == "cuda" and n_knock > _EXACT_MIN_N:
            pvals = gpu_ctx.mwu_cols(X_ctrl_test, target_cols, int_slice)
            wass_list.extend(gpu_ctx.wass_cols(X_ctrl_test, target_cols, int_slice))
        else:
            obs_slice = X_ctrl_test[:, target_cols]
            pvals = _mwu(obs_slice, int_slice, device)
            wass_list.extend(_wass_cols(obs_slice, int_slice, device))
        n_true_positive += int((pvals < p_threshold).sum())
        n_scored += len(target_cols)

    accounted = n_skipped_target_not_in_panel + n_skipped_no_coverage + n_scored
    assert accounted == len(top), (
        f"stat_precision bookkeeping mismatch: {n_skipped_target_not_in_panel} not-in-panel + "
        f"{n_skipped_no_coverage} no-coverage + {n_scored} scored = {accounted}, expected {len(top)}.")

    stat_prec = n_true_positive / k if k else 0.0
    wass_mean = float(np.mean(wass_list)) if wass_list else float("nan")
    fraction_scored = n_scored / k if k else 0.0
    return dict(stat_prec=stat_prec, wass_test=wass_mean, n_true_positive=n_true_positive,
                n_scored=n_scored, n_skipped_no_perturbation_coverage=n_skipped_no_coverage,
                n_skipped_target_not_in_panel=n_skipped_target_not_in_panel,
                fraction_scored=round(fraction_scored, 6))


def compute_for_k(gold_pairs_panel: set, edges_df: pd.DataFrame, k: int,
                   available_perturbations: set) -> dict:
    """FOR@K: of the gold-standard pairs whose regulator has held-out
    perturbation coverage (`available_perturbations`, i.e. evaluable in
    principle), what fraction does the candidate's own top-K NOT include.
    Denominator is |evaluable_gold|, not K -- unlike stat_prec, this is not
    comparable across candidates with a fixed-K denominator; it answers a
    different question (recall against the gold standard, not precision of
    one's own predictions)."""
    top = edges_df.head(k)
    top_k_set = set(zip(top["Regulator"], top["Target"]))

    evaluable_gold = {(a, b) for (a, b) in gold_pairs_panel if a in available_perturbations}
    n_evaluable = len(evaluable_gold)
    n_missed = len(evaluable_gold - top_k_set)
    for_k = n_missed / n_evaluable if n_evaluable else float("nan")
    return dict(for_k=for_k, for_n_evaluable=n_evaluable, for_n_missed=n_missed)


def compute_stat_metrics(edges_df: pd.DataFrame, k: int, X_test: np.ndarray, gene_to_col: dict,
                          pert_test: np.ndarray, X_ctrl_test: np.ndarray, min_cells: int,
                          p_threshold: float, gold_pairs_panel: set, control_label: str,
                          device: str = "cpu", gpu_ctx=None) -> dict:
    """Orchestrates stat_prec/wass_test + FOR@K for one (candidate, k) pair,
    matching the design doc's Step 5 point 3 signature."""
    stat = stat_precision_topk(edges_df, k, X_test, gene_to_col, pert_test, X_ctrl_test,
                                min_cells, p_threshold, device=device, gpu_ctx=gpu_ctx)
    available_perturbations = set(np.unique(pert_test)) - {control_label}
    for_result = compute_for_k(gold_pairs_panel, edges_df, k, available_perturbations)
    return {**stat, **for_result}
