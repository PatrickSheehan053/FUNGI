"""
SPORE+ · src/phase03_ambient.py
────────────────────────────────
Milestone 2 · Ambient RNA Detection

Designed to scale efficiently to 2M+ cell datasets (e.g., Replogle K562).
For CRISPR-screen datasets, two situations arise:

  A. Raw Matrix Mode: Unfiltered 10x matrix is available. Runs a SoupX-style 
     ambient correction that estimates the background profile from empty droplets.
     
  B. Global Profile Mode: Only the filtered h5ad is available. Computes a 
     data-driven "ambient signature" as the mean expression profile across 
     the entire dataset.

Scoring & Safety Mechanism
──────────────────────────
For each cell, we compute the cosine similarity between its expression profile 
and the ambient signature. 

CRITICAL: Cells are only flagged if they have an ambient cosine > threshold 
AND their total UMI count is in the bottom percentile (e.g., bottom 10%). 
(High ambient signal + Low UMI = dying cell / empty droplet).

This phase is non-destructive by default. It adds `ambient_score` and 
`ambient_flagged` to .obs. Removal requires `remove_flagged: true` in the config.
"""

from __future__ import annotations

import gc
import warnings
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3(adata, cfg: dict, logger) -> "AnnData":
    """
    Phase 3 · Ambient RNA Detection.

    Config keys used (all under `phase3_ambient:`):
        enabled          : bool   (default False — M1 datasets skip this)
        mode             : str    "control_profile" | "raw_matrix"
                                  "control_profile" uses non-targeting controls
                                  "raw_matrix" uses empty droplets (requires raw_h5_path)
        raw_h5_path      : str    path to unfiltered Cell Ranger h5 (mode=raw_matrix only)
        flag_threshold   : float  ambient score above which cells are flagged (default 0.8)
        remove_flagged   : bool   remove flagged cells from dataset (default False)
        min_umi_pct      : float  additionally require cell total_counts < this percentile
                                  to be flagged (default 0.10 → bottom 10%)

    Returns adata with new obs columns:
        ambient_score    : float  [0,1] cosine similarity to ambient signature
        ambient_flagged  : bool   True if score > threshold AND low UMI (if enabled)
    """
    log = logger
    phase_cfg = cfg.get("phase3_ambient", {})

    if not phase_cfg.get("enabled", False):
        log.info("  Phase 3: DISABLED (Milestone 2 feature)")
        log.info("  Set phase3_ambient.enabled: true to activate")
        return adata

    log.info("═" * 65)
    log.info("  PHASE 3 · Ambient RNA Detection")
    log.info("═" * 65)

    mode          = phase_cfg.get("mode", "control_profile")
    threshold     = float(phase_cfg.get("flag_threshold", 0.80))
    remove        = bool(phase_cfg.get("remove_flagged", False))
    min_umi_pct   = float(phase_cfg.get("min_umi_pct", 0.10))

    n_start = adata.n_obs

    if mode == "raw_matrix":
        raw_path = phase_cfg.get("raw_h5_path")
        if not raw_path:
            log.warning("  Phase 3: mode=raw_matrix but no raw_h5_path set. "
                        "Falling back to control_profile mode.")
            mode = "control_profile"
        else:
            adata = _ambient_from_raw_matrix(adata, raw_path, threshold,
                                              min_umi_pct, log)
            # If raw_matrix failed internally, it won't write 'ambient_score'. Force fallback.
            if "ambient_score" not in adata.obs:
                mode = "control_profile"

    if mode in ["control_profile", "global_profile"]:
        adata = _ambient_from_global_profile(
            adata, threshold, min_umi_pct, log)

    n_flagged = int(adata.obs.get("ambient_flagged", False).sum())
    log.info(f"  Ambient flagged: {n_flagged:,} / {n_start:,} cells "
             f"({n_flagged/max(n_start,1)*100:.2f}%)")

    if remove and n_flagged > 0:
        mask = ~adata.obs["ambient_flagged"].astype(bool)
        adata = adata[mask].copy()
        log.info(f"  Removed {n_flagged:,} ambient-flagged cells → {adata.n_obs:,} remaining")
    elif remove and n_flagged == 0:
        log.info("  No cells flagged for removal.")

    gc.collect()
    return adata


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy A: Global-profile ambient scoring (Replaces control_profile)
# ─────────────────────────────────────────────────────────────────────────────

def _ambient_from_global_profile(adata, threshold: float, min_umi_pct: float, logger) -> "AnnData":
    """
    Estimate ambient signature from the global dataset average.
    Score each cell by cosine similarity to the global ambient signature.
    Flag cells above threshold that also have low total UMI.
    """
    logger.info(f"  Computing ambient signature from global dataset average ({adata.n_obs:,} cells)...")

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # ── OOM FIREWALL: Chunked Mean Calculation ──
    # Threshold set to 1,000,000 cells
    is_large = getattr(adata, 'isbacked', False) or n_cells > 1000000
    
    if is_large:
        logger.info("  Large dataset detected. Chunking global profile calculation to prevent OOM...")
        global_sum = np.zeros(n_genes, dtype=np.float64)
        chunk_size = 50000
        
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = adata.X[start:end]
            
            if sp.issparse(chunk):
                global_sum += np.asarray(chunk.sum(axis=0)).flatten()
            else:
                global_sum += np.asarray(chunk.sum(axis=0)).flatten()
                
            del chunk # Immediately free chunk RAM
        
        ambient_profile = (global_sum / n_cells).astype(np.float32)
        del global_sum
        
    else:
        # Fast path for small, fully-in-memory datasets
        X = adata.X
        if sp.issparse(X):
            ambient_profile = np.asarray(X.mean(axis=0)).ravel()
        else:
            ambient_profile = X.mean(axis=0).ravel()

    ambient_profile = ambient_profile.astype(np.float32)
    amb_norm = np.linalg.norm(ambient_profile)
    
    if amb_norm < 1e-10:
        logger.warning("  Phase 3: ambient profile is near-zero — scoring skipped")
        adata.obs["ambient_score"]   = 0.0
        adata.obs["ambient_flagged"] = False
        return adata

    amb_unit = ambient_profile / amb_norm

    # Score in batches to avoid memory blowouts on massive datasets
    batch_size = 50_000
    scores     = np.zeros(n_cells, dtype=np.float32)
    
    # ── THE FIX: Define X here so the scoring loop can safely access it ──
    X = adata.X

    for start in range(0, n_cells, batch_size):
        end   = min(start + batch_size, n_cells)
        batch = X[start:end]
        if sp.issparse(batch):
            batch = batch.toarray()
        batch    = batch.astype(np.float32)
        norms    = np.linalg.norm(batch, axis=1, keepdims=True)
        norms    = np.where(norms < 1e-10, 1e-10, norms)
        batch_u  = batch / norms
        scores[start:end] = batch_u @ amb_unit

    adata.obs["ambient_score"] = scores.tolist()

    # UMI percentile threshold for co-flagging
    if "total_counts" in adata.obs.columns:
        umi_vals     = adata.obs["total_counts"].values.astype(float)
        umi_thresh   = float(np.percentile(umi_vals, min_umi_pct * 100))
        low_umi_mask = umi_vals <= umi_thresh
    else:
        low_umi_mask = np.ones(n_cells, dtype=bool)

    adata.obs["ambient_flagged"] = (
        (scores > threshold) & low_umi_mask
    ).tolist()

    logger.info(f"  Ambient score: mean={scores.mean():.3f}, "
                f"max={scores.max():.3f}, "
                f">threshold: {(scores > threshold).sum():,}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy B: raw matrix (empty droplet) ambient scoring
# ─────────────────────────────────────────────────────────────────────────────

def _ambient_from_raw_matrix(adata, raw_h5_path: str, threshold: float,
                              min_umi_pct: float, logger) -> "AnnData":
    """
    Estimate ambient from empty droplets in the raw (unfiltered) Cell Ranger
    matrix. Uses the SoupX approach: empty droplets (cells not in filtered set)
    define the ambient RNA profile.

    Requires scanpy and the raw .h5 file output by Cell Ranger.
    Falls back to control_profile mode if the file can't be loaded.
    """
    import os
    if not os.path.exists(raw_h5_path):
        logger.warning(f"  Phase 3: raw_h5_path not found: {raw_h5_path}. "
                       "Falling back to control_profile mode.")
        return adata

    try:
        import scanpy as sc
        logger.info(f"  Loading raw matrix from {raw_h5_path}...")
        raw = sc.read_10x_h5(raw_h5_path, backed=True)

        # Get barcodes that are NOT in the filtered set = empty droplets
        filtered_bcs = set(adata.obs_names.tolist())
        all_bcs      = set(raw.obs_names.tolist())
        empty_bcs    = list(all_bcs - filtered_bcs)

        if len(empty_bcs) < 100:
            logger.warning(f"  Phase 3: only {len(empty_bcs)} empty droplets found. "
                           "Falling back to control_profile mode.")
            return adata

        # Sample up to 10k empty droplets for ambient profile
        rng      = np.random.default_rng(42)
        n_sample = min(10_000, len(empty_bcs))
        sample   = rng.choice(empty_bcs, size=n_sample, replace=False).tolist()

        raw_sub  = raw[sample].copy()
        X_empty  = raw_sub.X
        if sp.issparse(X_empty):
            X_empty = X_empty.toarray()

        # Align genes with filtered set
        raw_genes      = set(raw.var_names.tolist())
        filtered_genes = adata.var_names.tolist()
        common         = [g for g in filtered_genes if g in raw_genes]

        if len(common) < 100:
            logger.warning("  Phase 3: < 100 common genes between raw and filtered. "
                           "Falling back to control_profile mode.")
            return adata

        raw_gene_idx  = [raw.var_names.get_loc(g) for g in common]
        filt_gene_idx = [filtered_genes.index(g)  for g in common]

        ambient_profile = X_empty[:, raw_gene_idx].mean(axis=0).astype(np.float32)
        amb_norm        = np.linalg.norm(ambient_profile)
        if amb_norm < 1e-10:
            return adata
        amb_unit = ambient_profile / amb_norm

        logger.info(f"  Ambient profile estimated from {n_sample:,} empty droplets")

        # Score filtered cells
        batch_size = 50_000
        n_cells    = adata.n_obs
        scores     = np.zeros(n_cells, dtype=np.float32)
        X          = adata.X

        for start in range(0, n_cells, batch_size):
            end   = min(start + batch_size, n_cells)
            batch = X[start:end]
            if sp.issparse(batch):
                batch = batch.toarray()
            batch    = batch[:, filt_gene_idx].astype(np.float32)
            norms    = np.linalg.norm(batch, axis=1, keepdims=True)
            norms    = np.where(norms < 1e-10, 1e-10, norms)
            scores[start:end] = (batch / norms) @ amb_unit

        adata.obs["ambient_score"]   = scores.tolist()
        
        if "total_counts" in adata.obs.columns:
            umi_vals     = adata.obs["total_counts"].values.astype(float)
            umi_thresh   = float(np.percentile(umi_vals, min_umi_pct * 100))
            low_umi_mask = umi_vals <= umi_thresh
        else:
            low_umi_mask = np.ones(n_cells, dtype=bool)

        adata.obs["ambient_flagged"] = ((scores > threshold) & low_umi_mask).tolist()
        
        del raw, raw_sub, X_empty
        gc.collect()
        return adata

    except Exception as e:
        logger.warning(f"  Phase 3: raw matrix ambient scoring failed ({e}). "
                       "Falling back to control_profile mode.")
        return adata
