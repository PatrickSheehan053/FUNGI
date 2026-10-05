"""
SPORE+ · src/phase08_hvg.py
──────────────────────────────
Phase 8: Dimensionality Reduction (Highly Variable Genes)

Large-dataset path (triggered when n_obs > large_dataset_threshold):
  Mirrors the Phase 10 subprocess architecture exactly.
  1. Caller frees the in-memory splits via force_heap_return().
  2. This module launches phase08_worker.py in a fresh subprocess with
     allocator-safe env vars (MALLOC_ARENA_MAX=2, MALLOC_MMAP_THRESHOLD_=128MB).
  3. Worker reads Phase 7 .h5ad files in backed mode (never loads full matrix).
  4. Worker streams column subset to p8 files and writes hvg_names.json.
  5. Parent reloads the small (5k gene) p8 files.

Small-dataset path: original in-process logic, unchanged.
"""

import gc
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import scanpy as sc

# The Phase 8 large-dataset path runs this module inside phase08_worker.py, which imports it as a
# TOP-LEVEL module (`from phase08_hvg import ...`) with no package context — so a package-relative
# `.utils` import fails there (it did: the K562 large path crashed at import). Try relative first
# (normal in-package use), fall back to absolute (worker subprocess with src/ on sys.path).
try:
    from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                        safe_in_memory_gene_subset, get_cell_cycle_genes, load_protected_genes)
except ImportError:
    from utils import (log_phase_header, snapshot, log_memory, force_gc,
                       safe_in_memory_gene_subset, get_cell_cycle_genes, load_protected_genes)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_heldout_target_ids(splits_dir, dataset_name, var_names, ctrl_label,
                            sep="+", is_combinatorial=False, logger=None):
    """
    exp_009.1 rebuild (Defect 2) — read the Phase 7 split JSON and return the set of held-out
    (val + test) perturbation-target gene IDENTITIES that exist in the panel var_names.

    The whole zero-shot design requires every held-out target to survive HVG as a graph node. Phase 7
    already wrote {name}_split_indices.json with val_labels/test_labels. SPORE knows these — so it
    force-carries the identities AUTOMATICALLY instead of depending on a hand-augmented protected list.

    FIREWALL: only gene IDENTITY crosses the split (zero-shot). HVG variance stats are computed on the
    TRAIN subsample only; val/test RESPONSE cells never enter HVG selection. Forcing a held-out target
    to be a node leaks nothing (GEARS/Roohani 2023).
    """
    path = Path(splits_dir) / f"{dataset_name}_split_indices.json"
    if not path.exists():
        if logger:
            logger.warning(f"  Held-out force-carry: split JSON not found at {path} — "
                           f"held-out targets NOT auto-carried (coverage may under-report)")
        return set()
    with open(path) as f:
        meta = json.load(f)
    labels = list(meta.get("val_labels", [])) + list(meta.get("test_labels", []))
    targets = set()
    for lbl in labels:
        if str(lbl) == str(ctrl_label):
            continue
        if is_combinatorial and sep and sep in str(lbl):
            for part in str(lbl).split(sep):
                if part.strip():
                    targets.add(part.strip())
        else:
            targets.add(str(lbl))
    var_set = set(var_names)
    present = targets & var_set
    if logger:
        logger.info(f"  Held-out (val+test) targets: {len(targets)} identities; "
                    f"{len(present)} present in the post-triage panel (to force-carry into HVG)")
    return present


def force_include_into_hvg(force_ids, hvg_names, hvg_stats, var_names, rank_col,
                           do_not_drop, flag_col, tag, logger=None):
    """
    Force-include `force_ids` into `hvg_names`, keeping panel size stable by SWAPPING out the
    lowest-variance droppable HVGs (or STACKING if not enough are droppable). Mirrors the existing
    target/protected rescue logic so all force-include operations behave identically.

    `do_not_drop` = genes that must never be swapped out (targets, ghosts, other protected/forced genes).
    Mutates hvg_names (a set) and hvg_stats in place; returns hvg_names.
    """
    if flag_col not in hvg_stats.columns:
        hvg_stats[flag_col] = False
    to_add = (set(map(str, force_ids)) & set(map(str, var_names))) - set(hvg_names)
    if not to_add:
        return hvg_names
    n_add        = len(to_add)
    safe_to_drop = list(set(hvg_names) - set(map(str, force_ids)) - set(do_not_drop))
    if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_add:
        safe_scores   = hvg_stats.loc[safe_to_drop, rank_col].sort_values()
        genes_to_drop = safe_scores.head(n_add).index.tolist()
        hvg_names.difference_update(genes_to_drop)
        hvg_names.update(to_add)
        hvg_stats.loc[genes_to_drop, "highly_variable"]   = False
        hvg_stats.loc[list(to_add), "highly_variable"]    = True
        hvg_stats.loc[list(to_add), flag_col]             = True
        if logger:
            logger.info(f"  {tag}: swapped {n_add} genes into HVG (panel size stable)")
    else:
        hvg_names.update(to_add)
        hvg_stats.loc[list(to_add), flag_col] = True
        if logger:
            logger.info(f"  {tag}: stacked {n_add} genes into HVG")
    return hvg_names


def _sanitize_uns(uns_dict):
    """Convert numpy scalars in uns dict to Python native types for h5py."""
    def _clean(v):
        if isinstance(v, dict):
            return {str(kk): _clean(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_clean(i) for i in v]
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return {str(k): _clean(v) for k, v in uns_dict.items()}


def _stream_to_p8(backed_adata, keep_mask, out_path,
                  extra_uns=None, chunk_rows=50_000, logger=None):
    """
    Read a backed split in sequential row-chunks, keep only keep_mask
    gene columns, accumulate, and write to out_path. Never densifies
    the full matrix.

    Peak subprocess RAM = chunk (chunk_rows × n_kept_genes, sparse) +
    accumulated vstack result (~10 GB for 1.45M × 5k train).
    """
    import anndata as ad

    n_obs    = backed_adata.n_obs
    gene_idx = np.where(keep_mask)[0]
    n_kept   = len(gene_idx)

    chunks = []
    for start in range(0, n_obs, chunk_rows):
        end   = min(start + chunk_rows, n_obs)
        chunk = backed_adata.X[start:end]
        if sp.issparse(chunk):
            chunk = chunk.tocsr()[:, gene_idx]
        else:
            chunk = sp.csr_matrix(chunk[:, gene_idx])
        chunks.append(chunk.astype(np.float32))
        del chunk

    X_new = sp.vstack(chunks, format="csr")
    del chunks
    gc.collect()

    var_new   = backed_adata.var.iloc[gene_idx].copy()
    obs_new   = backed_adata.obs.copy()
    uns_new   = _sanitize_uns(extra_uns or {})

    adata_out = ad.AnnData(X=X_new, obs=obs_new, var=var_new, uns=uns_new)
    adata_out.write_h5ad(out_path)
    del adata_out, X_new
    gc.collect()

    if logger:
        logger.info(f"    → {n_obs:,} × {n_kept:,} genes written to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER LOGIC  (called by phase08_worker.py CLI)
# ═══════════════════════════════════════════════════════════════════════════

def run_worker_logic(
    train_path, val_path, test_path, splits_dir, dataset,
    pert_col="gene", ctrl_label="non-targeting",
    n_top_genes=5000, method="seurat_v3",
    is_combinatorial=False, sep="+",
    protected_gene_list=None,
    logger=None,
):
    """
    Full Phase 8 computation. Reads splits from disk in backed mode,
    never loads the full matrix into RAM. Writes p8 files + hvg_names.json.

    Called by phase08_worker.py (subprocess CLI). Can also be called
    directly for testing.
    """
    import anndata as ad

    def log(msg):
        if logger:
            logger.info(msg)

    rss       = lambda: psutil.Process().memory_info().rss / 1e9
    splits_dir = Path(splits_dir)

    # ── 1. Open train in backed mode (0 GB cost) ──────────────────────────
    log(f"  Opening train in backed mode: {train_path}")
    train_backed = ad.read_h5ad(train_path, backed="r")
    n_obs  = train_backed.n_obs
    n_vars = train_backed.n_vars
    log(f"  Train: {n_obs:,} × {n_vars:,} genes  (RAM: {rss():.1f} GB)")

    # ── 2. Stratified subsample indices (obs metadata only, no X read) ────
    n_sample = min(100_000, n_obs)
    rng      = np.random.default_rng(42)

    if pert_col in train_backed.obs.columns:
        log(f"  Stratified sampling {n_sample:,}/{n_obs:,} cells...")
        frac     = n_sample / n_obs
        df       = pd.DataFrame({"pert": train_backed.obs[pert_col]})
        df["orig_idx"] = np.arange(n_obs)
        sampled  = []
        for _, group in df.groupby("pert", observed=True):
            if len(group) == 0:
                continue
            n_take = max(1, min(int(np.round(len(group) * frac)), len(group)))
            sampled.extend(
                rng.choice(group["orig_idx"].values, size=n_take, replace=False))
        idx_sorted = np.sort(np.array(sampled))
    else:
        log("  Random subsample (pert col not found)...")
        idx_sorted = np.sort(rng.choice(n_obs, size=n_sample, replace=False))

    # ── 3. Extract subsample via backed slicing ────────────────────────────
    # backed_adata[sorted_idx, :] uses h5py's sequential sparse read,
    # which only loads the relevant rows from disk. Never reads the full matrix.
    log(f"  Loading {len(idx_sorted):,}-cell subsample from disk "
        f"(backed slice, fresh heap)...")
    calc_adata = train_backed[idx_sorted, :].to_memory()
    if sp.issparse(calc_adata.X):
        calc_adata.X = calc_adata.X.tocsr()
        calc_adata.X.data = calc_adata.X.data.astype(np.float32, copy=False)
    log(f"  Subsample loaded  (RAM: {rss():.1f} GB)")

    # ── 4. HVG computation ─────────────────────────────────────────────────
    log(f"  Running HVG: method={method}, n_top={n_top_genes:,}")
    sc.pp.highly_variable_genes(
        calc_adata, flavor=method, n_top_genes=n_top_genes, subset=False)

    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means",
                          "dispersions", "dispersions_norm",
                          "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    hvg_names = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    rank_col  = ("variances_norm" if "variances_norm" in hvg_stats.columns
                 else "dispersions_norm")

    del calc_adata
    gc.collect()
    log(f"  HVG done  (RAM: {rss():.1f} GB)")

    # ── 5. Rescue operations ───────────────────────────────────────────────
    hvg_stats["rescued_target"] = False
    hvg_stats["rescued_ghost"]  = False

    cc_s, cc_g2m = get_cell_cycle_genes()
    cc_genes     = set(cc_s + cc_g2m)
    var_upper    = {v.upper(): v for v in train_backed.var_names}
    cc_in_matrix = [var_upper[g] for g in cc_genes if g in var_upper]

    var_set    = set(train_backed.var_names)
    raw_labels = set(train_backed.obs[pert_col].unique()) - {ctrl_label}
    targets    = set()
    for lbl in raw_labels:
        if is_combinatorial and sep and sep in str(lbl):
            for part in str(lbl).split(sep):
                targets.add(part.strip())
        else:
            targets.add(str(lbl))

    target_rescue = (targets & var_set) - hvg_names
    if target_rescue:
        n_rescue     = len(target_rescue)
        safe_to_drop = list(hvg_names - targets - set(cc_in_matrix))
        if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
            safe_scores   = hvg_stats.loc[safe_to_drop, rank_col].sort_values()
            genes_to_drop = safe_scores.head(n_rescue).index.tolist()
            hvg_names.difference_update(genes_to_drop)
            hvg_names.update(target_rescue)
            hvg_stats.loc[genes_to_drop, "highly_variable"]     = False
            hvg_stats.loc[list(target_rescue), "highly_variable"] = True
            hvg_stats.loc[list(target_rescue), "rescued_target"]  = True
            log(f"  Target rescue: swapped {n_rescue} targets "
                f"(core stable at {n_top_genes:,})")
        else:
            hvg_names.update(target_rescue)
            hvg_stats.loc[list(target_rescue), "rescued_target"] = True
            log(f"  Target rescue: stacked {len(target_rescue)} targets")

    ghosts = [g for g in cc_in_matrix if g not in hvg_names]
    if ghosts:
        log(f"  Ghost rescue: {len(ghosts)} cell-cycle genes stacked")
        hvg_names.update(ghosts)
        hvg_stats.loc[ghosts, "rescued_ghost"] = True

    # ── PROTECTED-GENE force-carry (large path — previously MISSING here; the config's protected list
    #    was never consumed on the K562/large path). Mirrors the small-dataset path. ────────────────────
    hvg_stats["rescued_protected"] = False
    if protected_gene_list:
        protected_ids, prot_report = load_protected_genes(protected_gene_list, train_backed, logger)
        force_include_into_hvg(
            protected_ids, hvg_names, hvg_stats, train_backed.var_names, rank_col,
            do_not_drop=(set(cc_in_matrix) | set(ghosts) |
                         set(hvg_stats.index[hvg_stats["rescued_target"]])),
            flag_col="rescued_protected", tag="Protected force-carry", logger=logger)
        log(f"  Protected genes: {prot_report.get('n_resolved', 0)}/"
            f"{prot_report.get('n_requested', 0)} resolved")

    # ── HELD-OUT TARGET force-carry (large path — Defect 2; auto val/test target identities). ──────────
    hvg_stats["rescued_heldout"] = False
    heldout_ids = load_heldout_target_ids(
        splits_dir, dataset, train_backed.var_names, ctrl_label,
        sep=sep, is_combinatorial=is_combinatorial, logger=logger)
    force_include_into_hvg(
        heldout_ids, hvg_names, hvg_stats, train_backed.var_names, rank_col,
        do_not_drop=(set(cc_in_matrix) | set(ghosts) | set(heldout_ids) |
                     set(hvg_stats.index[hvg_stats["rescued_target"]]) |
                     set(hvg_stats.index[hvg_stats["rescued_protected"]])),
        flag_col="rescued_heldout", tag="Held-out target force-carry", logger=logger)

    keep_mask = np.array([g in hvg_names for g in train_backed.var_names])
    n_kept    = int(keep_mask.sum())
    log(f"  Feature set: {n_vars:,} → {n_kept:,} genes")

    core_features = [g for g in train_backed.var_names
                     if g in hvg_names and g not in set(ghosts)]

    # ── 6. Write hvg_names.json for parent to reload ──────────────────────
    meta_path = splits_dir / f"{dataset}_hvg_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"hvg_names": list(hvg_names),
                   "core_features": core_features}, f)
    log(f"  HVG metadata → {meta_path}")

    # ── 7. Stream p8 files for all three splits ────────────────────────────
    # Close train backed before re-opening in the loop
    train_backed.file.close()

    for split_key, split_path in [
        ("train", train_path),
        ("val",   val_path),
        ("test",  test_path),
    ]:
        if not Path(split_path).exists():
            log(f"  Skipping {split_key}: not found at {split_path}")
            continue

        out_path = splits_dir / f"{dataset}_{split_key}_p8.h5ad"
        log(f"  Streaming {split_key} → {out_path}  (RAM: {rss():.1f} GB)")

        split_backed = ad.read_h5ad(split_path, backed="r")
        extra_uns    = ({
            "hvg_stats":           hvg_stats.to_dict("index"),
            "spore_core_features": core_features,
        } if split_key == "train" else {})

        _stream_to_p8(
            split_backed, keep_mask, out_path,
            extra_uns=extra_uns,
            chunk_rows=50_000,
            logger=logger,
        )
        split_backed.file.close()
        gc.collect()
        log(f"  {split_key} p8 done  (RAM: {rss():.1f} GB)")

    log("  Phase 8 worker complete. OS will reclaim heap on exit.")


# ═══════════════════════════════════════════════════════════════════════════
#  SUBPROCESS LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════

def run_phase8_subprocess(cfg, logger, timeout=7200):
    """
    Launch phase08_worker.py in a fresh subprocess with allocator-safe
    environment variables. Mirrors the Phase 10 subprocess architecture.
    """
    splits_dir   = cfg["paths"]["_splits"]
    dataset_name = cfg["dataset"]["name"]
    p8           = cfg.get("phase8_hvg", {})
    ds           = cfg.get("dataset", {})

    train_path = str(splits_dir / f"{dataset_name}_train.h5ad")
    val_path   = str(splits_dir / f"{dataset_name}_val.h5ad")
    test_path  = str(splits_dir / f"{dataset_name}_test.h5ad")

    for label, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Phase 7 disk file not found for {label}: {path}\n"
                f"Run Phase 7 (save_splits) before Phase 8.")

    worker = str(Path(__file__).parent / "phase08_worker.py")
    if not os.path.exists(worker):
        raise FileNotFoundError(
            f"phase08_worker.py not found at: {worker}")

    child_env = os.environ.copy()
    child_env.update({
        "MALLOC_MMAP_THRESHOLD_": "134217728",
        "MALLOC_ARENA_MAX":       "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "OPENBLAS_NUM_THREADS":   "1",
        "MKL_NUM_THREADS":        "1",
        "OMP_NUM_THREADS":        "1",
    })

    cmd = [
        sys.executable, worker,
        "--train",       train_path,
        "--val",         val_path,
        "--test",        test_path,
        "--splits-dir",  str(splits_dir),
        "--dataset",     dataset_name,
        "--pert-col",    ds.get("perturbation_col", "gene"),
        "--ctrl-label",  ds.get("control_label", "non-targeting"),
        "--n-top-genes", str(p8.get("n_top_genes", 5000)),
        "--method",      p8.get("method", "seurat_v3"),
        "--sep",         ds.get("perturbation_separator", "+"),
    ]
    # Defect 2/4: forward the protected list so the large/K562 path force-carries protected genes
    # (it silently didn't before — the arg existed in the worker CLI but was never passed here, and
    # run_worker_logic didn't even accept it → TypeError on any large run).
    protected_path = p8.get("protected_gene_list")
    if protected_path:
        cmd += ["--protected-gene-list", str(protected_path)]
    if ds.get("perturbation_structure") == "combinatorial":
        cmd.append("--is-combinatorial")

    parent_rss = psutil.Process().memory_info().rss / 1e9
    logger.info(f"  Launching Phase 8 subprocess (parent RSS: {parent_rss:.1f} GB)")
    logger.info(f"    env: MALLOC_ARENA_MAX=2, MALLOC_MMAP_THRESHOLD_=134217728")
    logger.info(f"    Train  → {train_path}")
    logger.info(f"    Output → {splits_dir}/{dataset_name}_*_p8.h5ad")

    result = subprocess.run(
        cmd, env=child_env, check=False, text=True, capture_output=False,
        timeout=timeout)

    after_rss = psutil.Process().memory_info().rss / 1e9
    logger.info(f"  Subprocess exited (code={result.returncode}). "
                f"Parent RSS: {after_rss:.1f} GB")

    if result.returncode != 0:
        raise RuntimeError(
            f"Phase 8 subprocess failed (exit code {result.returncode}). "
            f"See worker stdout/stderr above.")

    return str(splits_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  SMALL-DATASET IN-PROCESS PATH  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def select_hvgs(adata, cfg: dict, logger, split_label: str = "train"):
    """In-process HVG selection for small datasets only."""
    import anndata as ad

    if "hvg_stats" in adata.uns:
        logger.warning(f"  Phase 8 already run on {split_label}. Skipping.")
        return adata, list(adata.var_names)

    log_phase_header(logger, 8, f"Dimensionality Reduction – HVG ({split_label})")
    p8     = cfg.get("phase8_hvg", {})
    n_top  = p8.get("n_top_genes", 5000)
    method = p8.get("method", "seurat_v3")

    if not p8.get("enabled", True):
        logger.info("  Phase 8: HVG selection DISABLED")
        return adata, []

    log_memory(logger, "Phase 8 start")

    if getattr(adata, "isbacked", False):
        logger.info("  Pulling backed matrix into RAM...")
        adata = adata.to_memory()
        force_gc(logger)

    cc_s, cc_g2m = get_cell_cycle_genes()
    cc_genes     = set(cc_s + cc_g2m)
    var_upper    = {v.upper(): v for v in adata.var_names}
    cc_in_matrix = [var_upper[g] for g in cc_genes if g in var_upper]

    n_sample = min(100_000, adata.n_obs)
    pert_col = cfg.get("dataset", {}).get("perturbation_col", "perturbation")

    if pert_col in adata.obs.columns:
        logger.info(f"  Stratified subsample {n_sample:,}/{adata.n_obs:,} cells...")
        frac = n_sample / adata.n_obs
        df   = pd.DataFrame({"pert": adata.obs[pert_col]})
        df["orig_idx"] = np.arange(adata.n_obs)
        rng  = np.random.default_rng(42)
        sampled = []
        for _, group in df.groupby("pert", observed=True):
            if len(group) == 0:
                continue
            n_take = max(1, min(int(np.round(len(group) * frac)), len(group)))
            sampled.extend(
                rng.choice(group["orig_idx"].values, size=n_take, replace=False))
        idx_sorted = np.sort(np.array(sampled))
    else:
        rng = np.random.default_rng(42)
        idx_sorted = np.sort(rng.choice(adata.n_obs, size=n_sample, replace=False))

    calc_X     = adata.X[idx_sorted, :]
    calc_adata = ad.AnnData(X=calc_X, var=adata.var.copy())
    if sp.issparse(calc_adata.X):
        calc_adata.X.data = calc_adata.X.data.astype(np.float32, copy=False)
    del calc_X

    logger.info(f"  Running HVG: method={method}, n_top={n_top:,}")
    sc.pp.highly_variable_genes(
        calc_adata, flavor=method, n_top_genes=n_top, subset=False)

    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means",
                          "dispersions", "dispersions_norm",
                          "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    hvg_names = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    rank_col  = ("variances_norm" if "variances_norm" in hvg_stats.columns
                 else "dispersions_norm")
    del calc_adata
    force_gc(logger)

    hvg_stats["rescued_target"] = False
    hvg_stats["rescued_ghost"]  = False

    if p8.get("rescue_perturbation_targets", True):
        ctrl_label = cfg["dataset"]["control_label"]
        is_comb    = (cfg.get("dataset", {}).get("perturbation_structure")
                      == "combinatorial")
        sep        = cfg.get("dataset", {}).get("perturbation_separator", "+")
        var_set    = set(adata.var_names)
        raw_labels = set(adata.obs[pert_col].unique()) - {ctrl_label}
        targets    = set()
        for lbl in raw_labels:
            if is_comb and sep and sep in str(lbl):
                for part in str(lbl).split(sep):
                    targets.add(part.strip())
            else:
                targets.add(str(lbl))

        target_rescue = (targets & var_set) - hvg_names
        if target_rescue:
            n_rescue     = len(target_rescue)
            safe_to_drop = list(hvg_names - targets - set(cc_in_matrix))
            if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
                safe_scores   = hvg_stats.loc[safe_to_drop, rank_col].sort_values()
                genes_to_drop = safe_scores.head(n_rescue).index.tolist()
                hvg_names.difference_update(genes_to_drop)
                hvg_names.update(target_rescue)
                hvg_stats.loc[genes_to_drop, "highly_variable"]     = False
                hvg_stats.loc[list(target_rescue), "highly_variable"] = True
                hvg_stats.loc[list(target_rescue), "rescued_target"]  = True
                logger.info(f"  Target rescue: swapped {n_rescue} targets")
            else:
                hvg_names.update(target_rescue)
                hvg_stats.loc[list(target_rescue), "rescued_target"] = True

    ghosts = [g for g in cc_in_matrix if g not in hvg_names]
    if ghosts:
        logger.info(f"  Ghost rescue: {len(ghosts)} cell-cycle genes stacked")
        hvg_names.update(ghosts)
        hvg_stats.loc[ghosts, "rescued_ghost"] = True

    # ── PROTECTED-GENE rescue (exp_022: wire the config's protected_gene_list, previously DEAD) ──────────
    # Force-include the curated protected genes + the recovered held-out perturbation TARGETS regardless of
    # variance rank (previously the protected_gene_list was passed but never consumed -> only 59/109 protected
    # genes reached the panel). FIREWALL: only gene IDENTITIES cross; HVG variance stats above were computed on
    # the TRAIN subsample only, and val/test RESPONSE cells never enter here. Mirrors target_rescue's swap.
    hvg_stats["rescued_protected"] = False
    protected_path = p8.get("protected_gene_list")
    if protected_path:
        from .utils import load_protected_genes
        protected_ids, prot_report = load_protected_genes(protected_path, adata, logger)
        protected_set = set(protected_ids)
        protected_rescue = (protected_set & set(adata.var_names)) - hvg_names
        logger.info(f"  Protected genes: {prot_report.get('n_resolved', 0)}/"
                    f"{prot_report.get('n_requested', 0)} resolved; {len(protected_rescue)} to force-include")
        if protected_rescue:
            n_rescue = len(protected_rescue)
            already_rescued = set(hvg_stats.index[hvg_stats["rescued_target"]]) | set(ghosts)
            safe_to_drop = list(hvg_names - protected_set - set(cc_in_matrix) - already_rescued)
            if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
                safe_scores   = hvg_stats.loc[safe_to_drop, rank_col].sort_values()
                genes_to_drop = safe_scores.head(n_rescue).index.tolist()
                hvg_names.difference_update(genes_to_drop)
                hvg_names.update(protected_rescue)
                hvg_stats.loc[genes_to_drop, "highly_variable"]        = False
                hvg_stats.loc[list(protected_rescue), "highly_variable"] = True
                hvg_stats.loc[list(protected_rescue), "rescued_protected"] = True
                logger.info(f"  Protected rescue: swapped {n_rescue} protected genes (budget stable)")
            else:
                hvg_names.update(protected_rescue)
                hvg_stats.loc[list(protected_rescue), "rescued_protected"] = True
                logger.info(f"  Protected rescue: stacked {n_rescue} protected genes")

    # ── HELD-OUT TARGET force-carry (Defect 2: auto, no hand-built complete list) ────────────────────────
    # Force every val/test perturbation-target IDENTITY that survived triage to be a graph node, so
    # zero-shot coverage does not depend on a protected list the user must remember to augment. The
    # existing target_rescue above only covers TRAIN perturbations (they are the only labels in this
    # train-split obs); val/test target identities come from the Phase 7 split JSON. FIREWALL: identities
    # only — HVG stats are train-subsample-derived; val/test response cells never enter here.
    hvg_stats["rescued_heldout"] = False
    ctrl_label_h = cfg["dataset"]["control_label"]
    sep_h        = cfg.get("dataset", {}).get("perturbation_separator", "+")
    is_comb_h    = cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial"
    heldout_ids  = load_heldout_target_ids(
        cfg["paths"]["_splits"], cfg["dataset"]["name"], adata.var_names,
        ctrl_label_h, sep=sep_h, is_combinatorial=is_comb_h, logger=logger)
    _already_forced = set(ghosts)
    if "rescued_target" in hvg_stats.columns:
        _already_forced |= set(hvg_stats.index[hvg_stats["rescued_target"]])
    if "rescued_protected" in hvg_stats.columns:
        _already_forced |= set(hvg_stats.index[hvg_stats["rescued_protected"]])
    force_include_into_hvg(
        heldout_ids, hvg_names, hvg_stats, adata.var_names, rank_col,
        do_not_drop=(set(cc_in_matrix) | _already_forced | set(heldout_ids)),
        flag_col="rescued_heldout", tag="Held-out target force-carry", logger=logger)

    keep_mask     = np.array([g in hvg_names for g in adata.var_names])
    n_kept        = keep_mask.sum()
    core_features = [g for g in adata.var_names
                     if g in hvg_names and g not in set(ghosts)]

    logger.info(f"  Subsetting {adata.n_vars:,} → {n_kept:,} features")
    adata_new = safe_in_memory_gene_subset(adata, keep_mask=keep_mask, logger=logger)
    del adata
    force_gc(logger)

    adata_new.uns["hvg_stats"]          = hvg_stats.to_dict("index")
    adata_new.uns["spore_core_features"] = core_features

    snapshot(adata_new, "Post HVG selection", logger)
    log_memory(logger, "Phase 8 end")
    return adata_new, list(hvg_names)


def apply_hvg_to_other_splits(splits: dict, hvg_names: list, cfg: dict, logger):
    """Apply HVG mask to val/test — small-dataset path only."""
    logger.info(f"  Applying HVG mask to val/test ({len(hvg_names):,} features)...")
    hvg_set = set(hvg_names)
    for key in ["val", "test"]:
        if key not in splits:
            continue
        split_adata = splits[key]
        if getattr(split_adata, "isbacked", False):
            split_adata = split_adata.to_memory()
            force_gc(logger)
        keep_mask     = np.array([g in hvg_set for g in split_adata.var_names])
        splits[key]   = safe_in_memory_gene_subset(split_adata, keep_mask=keep_mask, logger=logger)
        snapshot(splits[key], f"HVG applied to {key}", logger)
        force_gc(logger)
    return splits


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_phase8(splits, cfg, logger):
    """
    Routes to subprocess path for large datasets, in-process for small.

    Large-dataset contract:
      - Caller must have already called force_heap_return() and passed
        splits=None (or an empty dict) so the large matrix is freed.
      - This function reads from the Phase 7 disk files directly.

    Small-dataset contract:
      - splits is the live dict with in-memory AnnData objects.
    """
    import anndata as ad

    large_threshold = cfg.get("runtime", {}).get("large_dataset_threshold", 1_000_000)
    dataset_name    = cfg["dataset"]["name"]
    splits_dir      = cfg["paths"]["_splits"]

    # Determine whether to use the subprocess path
    train_adata = (splits or {}).get("train") if splits else None
    is_large    = (
        train_adata is None                          # caller freed splits
        or getattr(train_adata, "isbacked", False)
        or train_adata.n_obs > large_threshold
    )

    if is_large:
        log_phase_header(logger, 8, "Dimensionality Reduction – HVG (subprocess)")
        log_memory(logger, "Phase 8 start")

        logger.info("  Step 1: launching Phase 8 in fresh subprocess...")
        run_phase8_subprocess(cfg, logger)

        logger.info("  Step 2: reloading p8 splits into parent (small, 5k genes)...")
        splits_out = {}
        for key in ["train", "val", "test"]:
            p8_path = splits_dir / f"{dataset_name}_{key}_p8.h5ad"
            if p8_path.exists():
                splits_out[key] = ad.read_h5ad(p8_path)
                snap = splits_out[key]
                logger.info(f"    Loaded {key} p8: {snap.n_obs:,} × {snap.n_vars:,}")
            else:
                logger.warning(f"    {key} p8 file not found: {p8_path}")

        meta_path = splits_dir / f"{dataset_name}_hvg_meta.json"
        with open(meta_path) as f:
            hvg_meta = json.load(f)
        hvg_names = hvg_meta["hvg_names"]

        log_memory(logger, "Phase 8 end")
        logger.info(f"  Phase 8 complete: {len(hvg_names):,} HVGs selected")
        return splits_out, hvg_names

    # Small-dataset in-process path
    train_new, hvg_names = select_hvgs(
        splits["train"], cfg, logger, split_label="train")
    splits["train"] = train_new
    splits          = apply_hvg_to_other_splits(splits, hvg_names, cfg, logger)
    return splits, hvg_names
