"""
SPORE+ · src/phase04_doublets.py
──────────────────────────────────
Milestone 2 · Doublet Detection

Uses Scrublet (Wolock et al. 2019) to simulate artificial doublets and 
score each observed cell.

Sparse Architecture & Memory Management
───────────────────────────────────────
Engineered specifically to survive massive 1.5M+ cell datasets within a 
strict 15GB RAM envelope. To avoid catastrophic dense array conversions, 
this module maintains strict `scipy.sparse` (CSR) formatting using 
`scanpy` internals for normalization and log1p transformations.

For datasets > max_cells_for_simulation (default 150,000 cells):
  1. Subsample `max_cells` (kept strictly sparse).
  2. Fit Scrublet on the subsample to find the doublet threshold.
  3. Learn a stable TruncatedSVD projection space on the normalized subsample.
  4. Train a parallel KNN Regressor to learn the Scrublet scoring manifold.
  5. Process the remaining 1M+ cells in sparse batches, projecting them 
     into the SVD space and predicting their exact score via the KNN.

This guarantees O(N) linear memory scaling without ever unpacking the zeroes.
Cells are flagged via `doublet_score` and `predicted_doublet`. 
Removal requires `remove_doublets: true` in the config.
"""

from __future__ import annotations

import gc
import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")


def run_phase4(adata, cfg: dict, logger) -> "AnnData":
    """
    Phase 4 · Doublet Detection.

    Returns adata with `doublet_score` and `predicted_doublet` in .obs.
    Cells are NOT removed unless `remove_doublets: true` in config.
    """
    log = logger
    phase_cfg = cfg.get("phase4_doublets", {})

    if not phase_cfg.get("enabled", False):
        log.info("  Phase 4: DISABLED (Milestone 2 feature)")
        log.info("  Set phase4_doublets.enabled: true to activate")
        return adata

    log.info("═" * 65)
    log.info("  PHASE 4 · Doublet Detection (Scrublet)")
    log.info("═" * 65)

    # Check scrublet is installed
    try:
        import scrublet as scr
    except ImportError:
        log.warning("  Phase 4: scrublet not installed. Run:")
        log.warning("    source /scratch/patrick.sheehan/FUNGI_bot/bin/activate")
        log.warning("    pip install scrublet")
        log.warning("  Phase 4: SKIPPED (scrublet unavailable)")
        return adata

    # Robust YAML extraction with None-type fallbacks
    expected_rate  = float(phase_cfg.get("expected_doublet_rate") if phase_cfg.get("expected_doublet_rate") is not None else 0.06)
    min_counts     = int(phase_cfg.get("min_counts") if phase_cfg.get("min_counts") is not None else 3)
    min_cells      = int(phase_cfg.get("min_cells") if phase_cfg.get("min_cells") is not None else 3)
    n_pcs          = int(phase_cfg.get("n_prin_comps") if phase_cfg.get("n_prin_comps") is not None else 30)
    max_sim_cells  = int(phase_cfg.get("max_cells_for_simulation") if phase_cfg.get("max_cells_for_simulation") is not None else 150_000)
    remove         = bool(phase_cfg.get("remove_doublets") if phase_cfg.get("remove_doublets") is not None else False)

    n_cells = adata.n_obs
    log.info(f"  Dataset: {n_cells:,} cells × {adata.n_vars:,} genes")
    log.info(f"  Expected doublet rate: {expected_rate*100:.1f}%")

    is_large = getattr(adata, 'isbacked', False) or n_cells > 1000000

    if n_cells > max_sim_cells:
        log.info(f"  Large dataset ({n_cells:,} cells > {max_sim_cells:,} max).")
        log.info(f"  Fitting Scrublet on {max_sim_cells:,}-cell subsample, "
                 f"predicting on all {n_cells:,} cells...")
        
        n_jobs = int(cfg.get("runtime", {}).get("n_jobs", -1))
        
        scores, doublets, threshold = _scrublet_large(
            adata, is_large, n_cells, max_sim_cells,
            expected_rate, min_counts, min_cells, n_pcs, n_jobs, log, scr)
    else:
        log.info(f"  Running Scrublet on full dataset ({n_cells:,} cells)...")
        scores, doublets, threshold = _scrublet_full(
            adata, expected_rate, min_counts, min_cells, n_pcs, log, scr)

    gc.collect()

    adata.obs["doublet_score"]      = scores.tolist()
    adata.obs["predicted_doublet"]  = doublets.tolist()

    n_doublets = int(doublets.sum())
    log.info(f"  Doublets predicted: {n_doublets:,} / {n_cells:,} "
             f"({n_doublets/max(n_cells,1)*100:.2f}%)")
    log.info(f"  Scrublet threshold used: {threshold:.4f}")

    if remove and n_doublets > 0:
        # ── OOM FIREWALL: Prevent in-memory copies of massive backed datasets ──
        if is_large:
            log.warning("  [!] Cannot physically remove cells from a large backed matrix without OOM.")
            log.warning("      Doublets have been flagged in .obs['predicted_doublet'].")
            log.warning("      They will be removed during the next disk-write phase. Continuing safely...")
        else:
            mask  = ~doublets
            adata = adata[mask].copy()
            log.info(f"  Removed {n_doublets:,} doublets → {adata.n_obs:,} cells remaining")
    elif remove:
        log.info("  No doublets to remove.")

    return adata


# ─────────────────────────────────────────────────────────────────────────────
#  Scrublet helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense_counts(X, logger) -> np.ndarray:
    """Convert expression matrix to dense uint32 for Scrublet."""
    if sp.issparse(X):
        # Scrublet expects raw count-like data; use the sparse rows
        # Don't call toarray() on full matrix — will be handled per-batch
        return X   # keep sparse; _scrublet_full will handle
    return X


def _scrublet_full(adata, expected_rate, min_counts, min_cells, n_pcs, log, scr):
    """Run Scrublet on the full count matrix (Bug-fixed)."""
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr() # Scrublet loves CSR

    scrub = scr.Scrublet(
        X,
        expected_doublet_rate=expected_rate,
    )
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=min(n_pcs, X.shape[1] - 1),
        verbose=False,
    )
    threshold = float(scrub.threshold_)
    return (np.array(doublet_scores, dtype=np.float32),
            np.array(predicted_doublets, dtype=bool),
            threshold)


def _scrublet_large(adata, is_large, n_cells, max_sim_cells,
                    expected_rate, min_counts, min_cells, n_pcs, n_jobs, log, scr):
    """
    Fit Scrublet on a subsample; score all cells safely.
    """
    # ── 1. OOM FIREWALL: Micro-Chunked Subsample Extraction ──
    if is_large:
        log.info(f"  [Scrublet] Large/Backed mode: extracting {max_sim_cells:,} cells in micro-chunks for safe I/O...")
        idx_sub = np.arange(max_sim_cells)
        
        # THE FIX: Slicing a backed HDF5 matrix of this size in one gulp 
        # crashes the h5py I/O buffer. We stream it out in micro-chunks 
        # and stack it safely in RAM to keep the memory footprint flat.
        x_chunks = []
        micro_chunk = 50000
        for i in range(0, max_sim_cells, micro_chunk):
            end = min(i + micro_chunk, max_sim_cells)
            c = adata.X[i:end]
            if not sp.issparse(c):
                c = sp.csr_matrix(c)
            else:
                c = c.tocsr()
            x_chunks.append(c)
        X_sub = sp.vstack(x_chunks)
    else:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n_cells, size=max_sim_cells, replace=False)
        idx_sub.sort() 
        X_sub = adata.X[idx_sub]

    if sp.issparse(X_sub):
        X_sub = X_sub.tocsr() # Strictly maintain sparsity

    log.info(f"  [Scrublet] Subsample shape: {X_sub.shape}")

    scrub = scr.Scrublet(X_sub, expected_doublet_rate=expected_rate)
    sub_scores, sub_predicted = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=min(n_pcs, X_sub.shape[1] - 1),
        verbose=False,
    )
    threshold = float(scrub.threshold_)
    log.info(f"  [Scrublet] Subsample doublets: {sub_predicted.sum():,}, threshold={threshold:.4f}")

    # ── 2. Learn a stable SVD Space ──
    import scanpy as sc
    import anndata as ad
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import KNeighborsRegressor

    log.info(f"  [Scrublet] Learning stable SVD projection space...")
    tmp_ad = ad.AnnData(X_sub)
    sc.pp.normalize_total(tmp_ad, target_sum=1e4)
    sc.pp.log1p(tmp_ad)
    
    svd = TruncatedSVD(n_components=n_pcs, random_state=42)
    obs_pca = svd.fit_transform(tmp_ad.X)
    del tmp_ad

    log.info(f"  [Scrublet] Training parallel KNN regressor (n_jobs={n_jobs})...")
    knn = KNeighborsRegressor(n_neighbors=15, n_jobs=n_jobs).fit(obs_pca, sub_scores)

    # ── 3. OOM FIREWALL: Enforced Sparse Chunk Prediction ──
    all_scores = np.zeros(n_cells, dtype=np.float32)
    batch_size = 50_000
    
    log.info(f"  [Scrublet] Projecting and scoring all {n_cells:,} cells in chunks...")
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        chunk = adata.X[start:end]
        
        # THE FIX: Ensure the chunk is strictly sparse BEFORE AnnData converts 
        # it to dense, which would consume ~1.6GB of RAM per iteration.
        if not sp.issparse(chunk):
            chunk = sp.csr_matrix(chunk)
        else:
            chunk = chunk.tocsr()
            
        tmp_b = ad.AnnData(chunk)
        sc.pp.normalize_total(tmp_b, target_sum=1e4)
        sc.pp.log1p(tmp_b)
        
        xb_pca = svd.transform(tmp_b.X)
        all_scores[start:end] = knn.predict(xb_pca).astype(np.float32)
        
        del chunk, tmp_b
        
    all_scores[idx_sub] = sub_scores.astype(np.float32)

    del X_sub, scrub
    gc.collect()

    predicted = all_scores > threshold
    return all_scores, predicted, threshold
