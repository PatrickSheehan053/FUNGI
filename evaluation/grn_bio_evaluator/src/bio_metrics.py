"""
bio_metrics.py -- biological/database-overlap metrics for obj_003_grn_bio_evaluator_v2.

Adapts clone/metrics_obj001.py. `biological_topk`, `aupr`, `auroc`, `epr_topk`
are preserved exactly (same logic, same return schema) -- these are
backward-compatibility-critical (Test 2). Changes vs. obj_001:

1. `wasserstein_topk`'s `groupby("Regulator", sort=False)` gains
   `observed=True`. obj_001's version has the same call WITHOUT it, which
   silently materializes a phantom empty group for every category-dtype
   value not present in the current top-K slice (legacy pandas default).
   It happens not to crash there only because that function calls `len()`
   on the phantom groups and never builds an array from them; exp_003b's
   stat_precision_evaluator.py hit the exact same pattern and DID crash
   (IndexError on empty-float64-array fancy indexing) before `observed=True`
   was added. Fixed here at the source rather than carried forward.
2. New `biological_topk_perdb`: takes the `per_db` dict already produced by
   `DatabaseLoaderV2.build_pooled()` and returns per-database precision
   without any additional data loading (exp_003b's per_database_evaluator.py
   had to reload each database in isolation; not necessary anymore since
   build_pooled already returns per_db for free).
3. `bio_prec_pbs` (via `biological_topk` called against the PBS set) is
   computed by the orchestrator (grn_eval_v2.py) using this same function --
   no special-cased PBS variant needed, since `biological_topk` is already
   pooled-set-agnostic. It is labeled a diagnostic, not primary/secondary,
   in the output schema -- see grn_eval_v2.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score


def biological_topk(edges_df: pd.DataFrame, pooled: set, k: int) -> dict:
    top = edges_df.head(k)
    predicted = set(zip(top["Regulator"], top["Target"]))
    assert len(predicted) == len(top), (
        f"top-{k} slice has {len(top)} rows but only {len(predicted)} distinct (Regulator, "
        f"Target) pairs -- duplicate-pair check should have caught this upstream.")
    tp = len(predicted & pooled)
    fp = len(predicted) - tp
    n_gold_in_panel = len(pooled)
    fn = n_gold_in_panel - tp
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / n_gold_in_panel if n_gold_in_panel else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return dict(k=k, true_positives=tp, false_positives=fp, false_negatives=fn,
                n_gold_in_panel=n_gold_in_panel, precision=precision, recall=recall, f1=f1)


def biological_topk_perdb(edges_df: pd.DataFrame, per_db: dict, k: int) -> dict:
    """per_db: {db_name: set of (Regulator, Target) pairs}, as returned by
    DatabaseLoaderV2.build_pooled()['per_db']. Returns {db_name: precision}."""
    out = {}
    for name, pairs in per_db.items():
        if not pairs:
            out[name] = None
            continue
        out[name] = biological_topk(edges_df, pairs, k)["precision"]
    return out


def causal_coverage(edges_df: pd.DataFrame, causal_sources: set, k: int) -> dict:
    """obj_003.1: the causal panel's analogue of stat_metrics' `fraction_scored`
    (the honesty number). Fraction of the top-K predicted edges whose Regulator is a
    TF present as a SOURCE in the causal pool -- i.e. edges the causal panel can, in
    principle, score. Unlike stat_prec's coverage (only ~868 perturbed sources on
    RPE1), this counts every edge whose regulator is a known-TF in the directed causal
    gold set, so causal_coverage >= stat_prec_coverage by construction. NaN-safe:
    empty pool / k>n_edges are handled by the caller-visible n_covered/k ratio.

    `causal_sources`: {r for (r, t) in causal_pool}, precomputed by build_causal_pool."""
    top = edges_df.head(k)
    n = len(top)
    if n == 0 or not causal_sources:
        return dict(k=k, n_covered=0, n_topk=n, causal_coverage=float("nan"))
    n_covered = int(top["Regulator"].isin(causal_sources).sum())
    return dict(k=k, n_covered=n_covered, n_topk=n,
                causal_coverage=n_covered / k if k else float("nan"))


def wasserstein_topk(edges_df: pd.DataFrame, k: int, X: np.ndarray,
                      gene_to_col: dict, pert_values: np.ndarray,
                      control_label: str, min_cells: int) -> dict:
    top = edges_df.head(k)
    ctrl_mask = pert_values == control_label
    X_ctrl = X[ctrl_mask]

    n_skipped_not_in_panel = int((~top["Target"].isin(gene_to_col)).sum())
    valid = top[top["Target"].isin(gene_to_col)]

    distances = []
    n_skipped_no_coverage = 0
    for regulator, group in valid.groupby("Regulator", sort=False, observed=True):
        knock_mask = pert_values == regulator
        n_knock = int(knock_mask.sum())
        if n_knock < min_cells:
            n_skipped_no_coverage += len(group)
            continue
        X_knock = X[knock_mask]
        for target in group["Target"]:
            tgt_col = gene_to_col[target]
            distances.append(wasserstein_distance(X_ctrl[:, tgt_col], X_knock[:, tgt_col]))

    mean_w = float(np.mean(distances)) if distances else float("nan")
    accounted = n_skipped_not_in_panel + n_skipped_no_coverage + len(distances)
    assert accounted == len(top), (
        f"Wasserstein bookkeeping mismatch: {n_skipped_not_in_panel} not-in-panel + "
        f"{n_skipped_no_coverage} no-coverage + {len(distances)} scored = {accounted}, "
        f"expected {len(top)} (top-{k} slice size).")
    return dict(k=k, mean_wasserstein=mean_w, n_scored=len(distances),
                n_skipped_no_perturbation_coverage=n_skipped_no_coverage,
                n_skipped_target_not_in_panel=n_skipped_not_in_panel,
                fraction_scored=len(distances) / k if k else 0.0)


def aupr(edges_df: pd.DataFrame, pooled: set) -> float:
    labels = np.fromiter(
        ((1 if (r, t) in pooled else 0) for r, t in zip(edges_df["Regulator"], edges_df["Target"])),
        dtype=np.int8, count=len(edges_df))
    scores = edges_df["Importance"].to_numpy(dtype=np.float64)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def auroc(edges_df: pd.DataFrame, pooled: set) -> float:
    labels = np.fromiter(
        ((1 if (r, t) in pooled else 0) for r, t in zip(edges_df["Regulator"], edges_df["Target"])),
        dtype=np.int8, count=len(edges_df))
    scores = edges_df["Importance"].to_numpy(dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def epr_topk(edges_df: pd.DataFrame, pooled: set, n_panel_genes: int, k: int) -> dict:
    bio = biological_topk(edges_df, pooled, k)
    precision_at_k = bio["precision"]
    max_possible_pairs = n_panel_genes * (n_panel_genes - 1)
    random_baseline_precision = len(pooled) / max_possible_pairs if max_possible_pairs else 0.0
    epr = (precision_at_k / random_baseline_precision) if random_baseline_precision > 0 else float("nan")
    return dict(k=k, epr=epr, precision_at_k=precision_at_k,
                random_baseline_precision=random_baseline_precision)
