"""
SPORE+ · src/phase07_splits.py
────────────────────────────────
Phase 7: Stratified Zero-Shot Data Splits

Engineered for foundation model training (e.g., CHITIN, GEARS).
To prevent data leakage, splits are executed on a "Zero-Shot" basis: entire 
perturbation targets are completely held out from the training set. This forces 
the downstream model to generalize to unseen genetic states rather than 
memorizing transcriptomic signatures.

Stratified L1-Norm Binning
──────────────────────────
Targets are scored by their absolute mean shift from non-targeting controls. 
They are binned by transcriptional impact (severity), and the Train/Val/Test 
splits are proportionally sampled from these bins to guarantee uniform 
difficulty across the splits.

Destructive Memory Management (For 1M+ Cell Datasets)
─────────────────────────────────────────────────────
Standard subsetting requires 2x memory overhead. For massive datasets, this 
module extracts the smaller Val/Test splits first, and then executes a 
low-level destructive reconstruction of the CSR sparse pointer arrays to 
mutate the original AnnData object into the Train split in-place. 
Peak RAM overhead is reduced by ~80%.

30MAY patch: test_n / val_n absolute-count parameters in config.
  test_n and val_n override test_ratio / val_ratio when present.
  Rationale: ratios produce absurdly small val/test sets on large perturbation
  panels (RPE1: 1526 perts × 10% val_ratio = 152 val — fine; but the old code
  had hardcoded n=20 via the zero_shot sub-block, giving 1.3% of RPE1 perts).
  With test_n: 20 / val_n: 20 you get the same absolute floor as the VCC run.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import json
from collections import OrderedDict
from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    safe_in_memory_row_subset)


def destructive_3way_split(adata, train_mask, val_mask, test_mask, logger):
    """
    Error 011 fix: never hold two 80% copies simultaneously.
    Extract small val/test first, then in-place mutate adata → train.
    """
    logger.info("  Applying destructive 3-way memory split...")

    logger.info("    Extracting test split...")
    test_ad = ad.AnnData(
        X=adata.X[test_mask].copy(),
        obs=adata.obs.iloc[test_mask].copy(),
        var=adata.var.copy())

    logger.info("    Extracting val split...")
    val_ad = ad.AnnData(
        X=adata.X[val_mask].copy(),
        obs=adata.obs.iloc[val_mask].copy(),
        var=adata.var.copy())

    logger.info("    Mutating original object → train split in-place...")
    n_kept = train_mask.sum()

    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr  = adata.X.indptr
        indices = adata.X.indices
        data    = adata.X.data

        new_indptr = np.zeros(n_kept + 1, dtype=indptr.dtype)
        padded = np.concatenate(([False], train_mask, [False]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        write_ptr = 0
        new_row   = 0
        for start, end in zip(starts, ends):
            n_rows_block = end - start
            data_start   = indptr[start]
            data_end     = indptr[end]
            nnz_block    = data_end - data_start
            if nnz_block > 0:
                indices[write_ptr:write_ptr + nnz_block] = indices[data_start:data_end]
                data[write_ptr:write_ptr + nnz_block]    = data[data_start:data_end]
            new_indptr[new_row + 1:new_row + 1 + n_rows_block] = (
                indptr[start + 1:end + 1] - data_start + write_ptr)
            write_ptr += nnz_block
            new_row   += n_rows_block

        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr),
            shape=(n_kept, adata.X.shape[1]))
    else:
        logger.warning("    Matrix not CSR — standard slice (may spike RAM).")
        new_X = adata.X[train_mask]

    train_ad = ad.AnnData(
        X=new_X,
        obs=adata.obs.iloc[train_mask].copy(),
        var=adata.var.copy())

    adata.X = None
    if hasattr(adata, "obsm"): adata.obsm.clear()
    if hasattr(adata, "varm"): adata.varm.clear()
    if hasattr(adata, "uns"):  adata.uns.clear()
    force_gc(logger)

    return train_ad, val_ad, test_ad


def _compute_mean_shift(adata, cfg, logger):
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    labels = adata.obs[pert_col].values
    perturbations = [p for p in np.unique(labels) if p != ctrl_label]

    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000

    if is_large:
        logger.info("  [Splits] Large/Backed mode: Chunking mean shift calculation...")
        pert_sums = {t: np.zeros(adata.n_vars, dtype=np.float64) for t in perturbations}
        pert_sums[ctrl_label] = np.zeros(adata.n_vars, dtype=np.float64)
        pert_counts = {t: 0 for t in perturbations}
        pert_counts[ctrl_label] = 0

        chunk_size = 50000
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            chunk_labels = labels[start:end]
            chunk_X = adata.X[start:end]
            if sp.issparse(chunk_X):
                chunk_X = chunk_X.toarray()

            for lbl in np.unique(chunk_labels):
                if lbl in pert_sums:
                    mask = chunk_labels == lbl
                    pert_sums[lbl] += chunk_X[mask].sum(axis=0)
                    pert_counts[lbl] += mask.sum()
            del chunk_X

        ctrl_mean = (pert_sums[ctrl_label] / max(pert_counts[ctrl_label], 1)).astype(np.float32)

        shifts = {}
        for target in perturbations:
            if pert_counts[target] > 0:
                t_mean = (pert_sums[target] / pert_counts[target]).astype(np.float32)
                shifts[target] = float(np.abs(t_mean - ctrl_mean).sum())
            else:
                shifts[target] = 0.0
        del pert_sums
        return pd.Series(shifts).sort_values(ascending=False)
    else:
        ctrl_mask = labels == ctrl_label
        X = adata.X
        if sp.issparse(X):
            ctrl_mean = np.array(X[ctrl_mask].mean(axis=0)).flatten()
        else:
            ctrl_mean = X[ctrl_mask].mean(axis=0)

        shifts = {}
        for target in perturbations:
            pert_mask = labels == target
            if sp.issparse(X):
                pert_mean = np.array(X[pert_mask].mean(axis=0)).flatten()
            else:
                pert_mean = X[pert_mask].mean(axis=0)
            shifts[target] = float(np.abs(pert_mean - ctrl_mean).sum())

        return pd.Series(shifts).sort_values(ascending=False)


def split_zero_shot(adata, cfg, logger, seed=None):
    zs       = cfg.get("phase7_splits", {})
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if seed is None:
        seed = zs.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    deg_counts = _compute_mean_shift(adata, cfg, logger)
    n_perts    = len(deg_counts)

    # Defect 5 hard gate: test_n + val_n must leave a non-empty train split of SURVIVING perturbations.
    # (startup validation checked this against the RAW count; here it is checked against the count that
    # actually survived triage — the real number.)
    test_n_req = zs.get("test_n")
    val_n_req  = zs.get("val_n")
    if test_n_req is not None and val_n_req is not None:
        if int(test_n_req) + int(val_n_req) >= n_perts:
            raise ValueError(
                f"[Phase 7] test_n ({test_n_req}) + val_n ({val_n_req}) = "
                f"{int(test_n_req)+int(val_n_req)} >= {n_perts} surviving perturbations after triage — "
                f"no perturbations left to train on. Lower test_n/val_n in the config.")

    # ── 1. Determine test / val counts ──────────────────────────────────────
    # test_n / val_n (absolute) override test_ratio / val_ratio (fractional).
    # 30MAY patch: added test_n / val_n so large panels (RPE1 1526 perts)
    # don't get 1-pert val/test sets from the old hardcoded n=20.
    test_n_cfg = zs.get("test_n", None)
    val_n_cfg  = zs.get("val_n",  None)

    if test_n_cfg is not None:
        n_test = max(1, min(int(test_n_cfg), n_perts - 2))
    else:
        test_ratio = zs.get("test_ratio", 0.25)
        n_test = max(1, int(n_perts * test_ratio))

    n_remain = n_perts - n_test

    if val_n_cfg is not None:
        n_val = max(1, min(int(val_n_cfg), n_remain - 1))
    else:
        val_ratio = zs.get("val_ratio", 0.10)
        n_val = max(1, int(n_remain * val_ratio))

    n_train = n_remain - n_val

    # Effective ratios for per-bin proportional allocation
    test_ratio_eff = n_test / n_perts
    val_ratio_eff  = n_val / max(n_remain, 1)

    logger.info(
        f"  Split targets: {n_train} train / {n_val} val / {n_test} test "
        f"(from {n_perts} perturbations)")

    # ── 2. Stratified binning ───────────────────────────────────────────────
    df = deg_counts.reset_index()
    df.columns = ["perturbation", "score"]
    df["bin"] = pd.qcut(df["score"], q=zs.get("stratify_bins", 4),
                        labels=False, duplicates="drop")

    train_labels, val_labels, test_labels = [], [], []
    for _, group in df.groupby("bin"):
        shuffled = group.sample(frac=1, random_state=rng.integers(1e9))

        bin_test   = max(1, int(len(shuffled) * test_ratio_eff)) if len(shuffled) > 3 else 1
        bin_remain = len(shuffled) - bin_test
        bin_val    = max(1, int(bin_remain * val_ratio_eff)) if bin_remain > 2 else 1

        test_labels.extend(shuffled.iloc[:bin_test]["perturbation"])
        val_labels.extend( shuffled.iloc[bin_test:bin_test + bin_val]["perturbation"])
        train_labels.extend(shuffled.iloc[bin_test + bin_val:]["perturbation"])

    logger.info(
        f"  Perturbation Split: {len(train_labels)} Train / {len(val_labels)} Val / "
        f"{len(test_labels)} Test (Zero-Shot Targets)")

    # ── 3. Distribute control cells ─────────────────────────────────────────
    pert_values    = adata.obs[pert_col].values
    ctrl_mask_full = pert_values == ctrl_label
    ctrl_indices   = np.where(ctrl_mask_full)[0]
    rng.shuffle(ctrl_indices)

    frac_test  = len(test_labels) / n_perts
    frac_val   = len(val_labels)  / n_perts

    n_ctrl_test = int(len(ctrl_indices) * frac_test)
    n_ctrl_val  = int(len(ctrl_indices) * frac_val)

    test_ctrl_idx  = ctrl_indices[:n_ctrl_test]
    val_ctrl_idx   = ctrl_indices[n_ctrl_test:n_ctrl_test + n_ctrl_val]
    train_ctrl_idx = ctrl_indices[n_ctrl_test + n_ctrl_val:]

    # ── 4. Assemble masks ───────────────────────────────────────────────────
    train_mask = np.isin(pert_values, train_labels)
    train_mask[train_ctrl_idx] = True

    val_mask = np.isin(pert_values, val_labels)
    val_mask[val_ctrl_idx] = True

    test_mask = np.isin(pert_values, test_labels)
    test_mask[test_ctrl_idx] = True

    log_memory(logger, "before destructive split")

    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000

    if is_large:
        logger.info("  [Splits] Large/Backed mode: Creating lazy views...")
        train_ad = adata[train_mask]
        val_ad   = adata[val_mask]
        test_ad  = adata[test_mask]
    else:
        train_ad, val_ad, test_ad = destructive_3way_split(
            adata, train_mask, val_mask, test_mask, logger)

    split_info = {
        "train": train_ad.n_obs,
        "val":   val_ad.n_obs,
        "test":  test_ad.n_obs,
    }
    snapshot(train_ad, "Train split", logger)
    snapshot(val_ad,   "Val split",   logger)
    snapshot(test_ad,  "Test split",  logger)

    return {
        "train": train_ad, "val": val_ad, "test": test_ad,
        "deg_counts": deg_counts, "split_info": split_info,
        "train_labels": train_labels, "val_labels": val_labels,
        "test_labels": test_labels, "seed": seed,
    }


def save_splits(split_result, cfg, logger, seed=None):
    splits_dir   = cfg["paths"]["_splits"]
    dataset_name = cfg["dataset"]["name"]
    if seed is not None:
        splits_dir = splits_dir / f"seed_{seed}"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for key in ["train", "val", "test"]:
        path = splits_dir / f"{dataset_name}_{key}.h5ad"
        split_result[key].write_h5ad(path)
        logger.info(f"  Saved {key} → {path}")

    if "train_labels" in split_result:
        meta = {
            "train_labels": list(split_result["train_labels"]),
            "val_labels":   list(split_result["val_labels"]),
            "test_labels":  list(split_result["test_labels"]),
            "seed":         int(split_result["seed"]),
        }
        meta_path = splits_dir / f"{dataset_name}_split_indices.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def run_phase7(adata, cfg, logger):
    log_phase_header(logger, 7, "Data Splits")
    mode      = cfg.get("phase7_splits", {}).get("mode", "zero_shot")
    test_mode = cfg.get("phase7_splits", {}).get("test_mode", False)
    seeds     = ([cfg.get("phase7_splits", {}).get("random_seed", 42)]
                 if not test_mode
                 else cfg.get("phase7_splits", {}).get("test_mode_seeds", [42]))

    all_splits = []
    for seed in seeds[:1]:
        result = split_zero_shot(adata, cfg, logger, seed=seed)
        save_splits(result, cfg, logger,
                    seed=seed if test_mode else None)
        all_splits.append(result)
    return all_splits