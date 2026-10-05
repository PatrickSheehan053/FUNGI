"""
SPORE+ · src/phase10_confounders.py
──────────────────────────────────────
Phase 10: Confounder Mitigation & Batch Correction

Executes dimensionality reduction and batch correction in an isolated memory 
environment. 

Architectural Highlights
────────────────────────
1. Subsample PCA: Fits TruncatedSVD on a representative 100k-cell subset, 
   then projects the full 1M+ cell sparse matrix into the stable subspace.
2. Harmony Integration (Korsunsky et al., 2019): Iterative clustering and 
   correction applied directly to the PCA embeddings to mitigate batch effects 
   (e.g., separate sequencing lanes or cell lines).
3. Ghost Excision: Automatically strips out temporary Cell Cycle "Ghost" genes 
   after the embeddings are generated, returning the dataset to its strict 
   5,000-feature core envelope for downstream foundation model training.
"""

import gc
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp


# ═══════════════════════════════════════════════════════════════════════════
#  AGGRESSIVE PARENT CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

def force_heap_return(splits_dict=None, var_names=None, logger=None):
    """
    Best-effort OS-level memory reclamation for a live Jupyter kernel.

    Three layers (synthesised from Gemini + Perplexity + ChatGPT reports):
      1. .X = None before del: numpy buffers freed in the 1st gc pass (Perplexity)
      2. IPython output-cache clearing: Out[...], _, __ hold hidden refs (ChatGPT)
      3. malloc_trim(0) + mallopt(M_ARENA_MAX=2): release heap pages (all three)

    Cannot guarantee reaching 15 GB — subprocess isolation handles the rest.
    Returns the RSS floor (GB) after cleanup.
    """
    def rss():
        return psutil.Process().memory_info().rss / 1e9

    before = rss()
    if logger:
        logger.info(f"  force_heap_return starting at {before:.1f} GB")

    # ── Layer 1: nullify .X before del ────────────────────────────────────
    if splits_dict is not None:
        for key in list(splits_dict.keys()):
            obj = splits_dict.pop(key)
            try:
                if hasattr(obj, "X") and obj.X is not None:
                    obj.X = None
                if hasattr(obj, "obsm"):
                    obj.obsm.clear()
                if hasattr(obj, "uns"):
                    obj.uns.clear()
            except Exception:
                pass
            del obj

    # ── Layer 2: IPython output cache ─────────────────────────────────────
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            if var_names:
                for name in var_names:
                    try:
                        ip.del_var(name, by_name=True)
                    except Exception:
                        ip.user_ns.pop(name, None)
            for attr in ("_", "__", "___"):
                ip.user_ns.pop(attr, None)
                try:
                    setattr(ip.displayhook, attr, None)
                except Exception:
                    pass
            try:
                if "Out" in ip.user_ns and hasattr(ip.user_ns["Out"], "clear"):
                    ip.user_ns["Out"].clear()
            except Exception:
                pass
            try:
                ip.history_manager.output_hist.clear()
            except Exception:
                pass
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
    except Exception as e:
        if logger:
            logger.warning(f"  IPython cache clear failed (non-fatal): {e}")

    # ── Layer 3: GC + malloc_trim + mallopt ───────────────────────────────
    for _ in range(3):
        gc.collect()
    after_gc = rss()
    if logger:
        logger.info(
            f"  After GC: {after_gc:.1f} GB (freed {before - after_gc:.1f} GB)")

    try:
        import ctypes
        from ctypes.util import find_library
        libc = ctypes.CDLL(find_library("c") or "libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype  = ctypes.c_int
        r = libc.malloc_trim(0)
        if logger:
            logger.info(f"  malloc_trim(0) returned {r} (1=pages returned)")
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.mallopt.restype  = ctypes.c_int
        libc.mallopt(-8, 2)   # M_ARENA_MAX = 2
        if logger:
            logger.info("  mallopt: MALLOC_ARENA_MAX set to 2")
    except Exception as e:
        if logger:
            logger.warning(f"  malloc_trim/mallopt failed (non-fatal): {e}")

    after = rss()
    if logger:
        logger.info(
            f"  Heap floor: {after:.1f} GB "
            f"(freed {before - after:.1f} GB total)")
        if after > 70:
            logger.warning(
                f"  Heap still at {after:.1f} GB — subprocess will handle "
                f"Phase 10 in a completely fresh heap regardless.")
    return after


# ═══════════════════════════════════════════════════════════════════════════
#  COMPUTATIONAL LOGIC  (used by both worker CLI and any direct call)
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_csr_float32(adata):
    """Ensure adata.X is CSR float32 — cheap no-op if already correct."""
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X.astype(np.float32))
    else:
        if adata.X.format != "csr":
            adata.X = adata.X.tocsr()
        if adata.X.data.dtype != np.float32:
            adata.X.data = adata.X.data.astype(np.float32, copy=False)
    return adata


def safe_scale_inplace(adata, max_value=10.0, chunk_rows=50_000, logger=None):
    """
    Chunked per-gene std scaling. Replaces sc.pp.scale (Error 038 fix).
    Processes 50k rows at a time: ~1 GB peak per chunk vs ~43 GB full copy.
    """
    adata = _ensure_csr_float32(adata)
    X       = adata.X
    n_obs   = X.shape[0]
    n_genes = X.shape[1]

    if logger:
        logger.info(
            f"  Scaling in-place: {n_obs:,} x {n_genes:,}, "
            f"chunk={chunk_rows:,} rows")

    gene_sum    = np.zeros(n_genes, dtype=np.float64)
    gene_sum_sq = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_obs, chunk_rows):
        end  = min(start + chunk_rows, n_obs)
        ptr0 = int(X.indptr[start])
        ptr1 = int(X.indptr[end])
        if ptr0 == ptr1:
            continue
        cidx  = X.indices[ptr0:ptr1]
        cdata = X.data[ptr0:ptr1].astype(np.float64)
        gene_sum    += np.bincount(cidx, weights=cdata,    minlength=n_genes)
        cdata       **= 2
        gene_sum_sq += np.bincount(cidx, weights=cdata, minlength=n_genes)
        del cdata

    std = np.sqrt(np.maximum(
        gene_sum_sq / n_obs - (gene_sum / n_obs) ** 2, 0.0))
    std[std < 1e-10] = 1.0
    del gene_sum, gene_sum_sq
    gc.collect()
    std_f32 = std.astype(np.float32)
    del std

    for start in range(0, n_obs, chunk_rows):
        end  = min(start + chunk_rows, n_obs)
        ptr0 = int(X.indptr[start])
        ptr1 = int(X.indptr[end])
        if ptr0 == ptr1:
            continue
        cidx = X.indices[ptr0:ptr1]
        X.data[ptr0:ptr1] /= std_f32[cidx]
        if max_value is not None:
            np.clip(X.data[ptr0:ptr1], 0.0, max_value,
                    out=X.data[ptr0:ptr1])

    del std_f32
    if logger:
        logger.info(f"  Scaling complete (zero_center=False, max={max_value})")
    return adata


def subsample_pca(adata, n_comps=50, n_sample=100_000,
                  random_state=42, logger=None):
    """
    Memory-safe PCA: fit TruncatedSVD on n_sample cells (~2.7 GB),
    project all cells via sparse @ dense (~300 MB output, no densification).
    """
    from sklearn.decomposition import TruncatedSVD

    n_obs    = adata.n_obs
    n_sample = min(n_sample, n_obs)
    rng      = np.random.default_rng(random_state)
    idx      = np.sort(rng.choice(n_obs, size=n_sample, replace=False))

    if logger:
        rss = psutil.Process().memory_info().rss / 1e9
        logger.info(
            f"  Subsample PCA: fit on {n_sample:,}/{n_obs:,} cells "
            f"(RAM: {rss:.1f} GB)")

    X_sub = adata.X[idx]
    if not sp.issparse(X_sub):
        X_sub = sp.csr_matrix(X_sub)

    svd = TruncatedSVD(
        n_components=n_comps, algorithm="randomized",
        n_iter=4, random_state=random_state)
    svd.fit(X_sub)
    del X_sub
    gc.collect()

    components_f32 = svd.components_.astype(np.float32)
    X_pca = adata.X.dot(components_f32.T)
    adata.obsm["X_pca"] = np.asarray(X_pca, dtype=np.float32)
    adata.uns["pca"] = {
        "components":               components_f32,
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "params":                   {"n_comps": n_comps, "n_sample": n_sample},
    }
    var_exp = svd.explained_variance_ratio_.sum() * 100
    if logger:
        rss = psutil.Process().memory_info().rss / 1e9
        logger.info(
            f"  PCA complete: {n_comps} PCs explain ~{var_exp:.1f}% "
            f"(RAM: {rss:.1f} GB)")
    return adata


def run_harmony(adata, batch_key="gem_group",
                max_iter_harmony=20, logger=None):
    """
    Direct harmonypy call. Handles shape-transpose bug (Error 031),
    str/category cast (Error 030), and log suppression (Error 032).
    """
    logging.getLogger("harmonypy").setLevel(logging.ERROR)

    if batch_key not in adata.obs.columns:
        if logger:
            logger.warning(
                f"  Harmony: '{batch_key}' not in .obs — using uncorrected PCA")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()
        return adata

    # Error 030: enforce str/category
    adata.obs[batch_key] = (adata.obs[batch_key]
                            .astype(str).astype("category"))
    n_batches = adata.obs[batch_key].nunique()
    if logger:
        logger.info(f"  Harmony: '{batch_key}', {n_batches} batches...")

    try:
        import harmonypy
        import logging as _logging
        # Suppress harmonypy's verbose lambda/theta arrays (one entry per batch)
        # Must be set AFTER import, as harmonypy creates its logger at import time
        _logging.getLogger("harmonypy").setLevel(_logging.ERROR)
        Z = harmonypy.run_harmony(
            adata.obsm["X_pca"], adata.obs,
            batch_key, max_iter_harmony=max_iter_harmony,
            verbose=False).Z_corr

        # Error 031: dynamic shape check — don't trust library orientation
        if Z.shape[0] == adata.n_obs:
            adata.obsm["X_pca_harmony"] = Z.astype(np.float32)
        elif Z.shape[1] == adata.n_obs:
            adata.obsm["X_pca_harmony"] = Z.T.astype(np.float32)
        else:
            raise ValueError(
                f"Harmony output {Z.shape} doesn't match n_obs={adata.n_obs}")

        if logger:
            rss = psutil.Process().memory_info().rss / 1e9
            logger.info(
                f"  Harmony complete: "
                f"X_pca_harmony={adata.obsm['X_pca_harmony'].shape} "
                f"(RAM: {rss:.1f} GB)")

    except ImportError:
        if logger:
            logger.warning(
                "  harmonypy not installed — using uncorrected PCA. "
                "pip install harmonypy --break-system-packages")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()
    except Exception as e:
        if logger:
            logger.error(f"  Harmony failed: {e}. Using uncorrected PCA.")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()

    return adata


def run_worker_logic(input_path, output_path, embed_path, obs_path,
                     batch_key="gem_group", n_comps=50, n_sample=100_000,
                     chunk_rows=50_000, max_harmony_iter=20,
                     cc_skip_threshold=500_000, logger=None):
    """
    Full Phase 10 computation. Called by phase10_worker.py (CLI) and
    can also be called directly for testing.

    Saves:
      output_path — full train_p10.h5ad (for Phase 12 metacell)
      embed_path  — lightweight .npz: X_pca_harmony, X_pca, pca_components
      obs_path    — obs DataFrame as .parquet (for Phase 11)
    """
    import anndata as ad

    def log(msg):
        if logger:
            logger.info(msg)

    rss = lambda: psutil.Process().memory_info().rss / 1e9

    log(f"  Reading {input_path} ...")
    adata = ad.read_h5ad(input_path)
    adata = _ensure_csr_float32(adata)
    log(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes "
        f"(RAM: {rss():.1f} GB)")

    # Cell cycle: dummy values for large datasets (n_obs > 500k)
    if adata.n_obs > cc_skip_threshold:
        if logger:
            logger.warning(
                f"  CC scoring skipped (n_obs={adata.n_obs:,} > "
                f"{cc_skip_threshold:,}): score_genes control matrix = ~8 GB spike")
        adata.obs["S_score"]   = np.float32(0.0)
        adata.obs["G2M_score"] = np.float32(0.0)
        adata.obs["phase"]     = "unknown"

    adata = safe_scale_inplace(
        adata, max_value=10.0, chunk_rows=chunk_rows, logger=logger)
    for _ in range(3):
        gc.collect()

    adata = subsample_pca(
        adata, n_comps=n_comps, n_sample=n_sample, logger=logger)
    for _ in range(3):
        gc.collect()

    # ── FREE adata.X before Harmony (critical RAM fix) ───────────────────────
    # After PCA, the 27.5 GB sparse matrix is no longer needed.
    # Harmony only operates on adata.obsm["X_pca"] (300 MB).
    # Deleting X drops subprocess from ~42.7 GB to ~15 GB before Harmony
    # allocates its R matrix (1.5M × 100 × 8 = 1.2 GB dense).
    #
    # Without this: parent(88) + subprocess(42.7) + Harmony(1.2) = 132 GB → OOM
    # With this:    parent(88) + subprocess(15)   + Harmony(1.2) = 104 GB ✓
    #
    # Phase 12 reloads train_p9.h5ad for metacell averaging —
    # that uses the NORMALIZED expression matrix (not the PCA-scaled version),
    # which is correct: metacell means should be computed on normalized counts.
    log(f"  Freeing adata.X before Harmony (42.7 GB → ~15 GB subprocess)...")
    rss_before = rss()
    adata.X = None
    for _ in range(3):
        gc.collect()
    try:
        import ctypes
        from ctypes.util import find_library
        libc = ctypes.CDLL(find_library("c") or "libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype  = ctypes.c_int
        libc.malloc_trim(0)
    except Exception:
        pass
    log(f"  After freeing X: {rss():.1f} GB (was {rss_before:.1f} GB)")

    adata = run_harmony(
        adata, batch_key=batch_key,
        max_iter_harmony=max_harmony_iter, logger=logger)
    for _ in range(3):
        gc.collect()

    # Do NOT save train_p10.h5ad — adata.X was deleted above.
    # Phase 12 reloads train_p9.h5ad from disk and adds embeddings from .npz.
    # (output_path argument kept for API compatibility but not used here)

    # Save lightweight outputs (parent reloads these, not the full 41.5 GB)
    log(f"  Writing embeddings → {embed_path}")
    np.savez_compressed(
        embed_path,
        X_pca_harmony  = adata.obsm["X_pca_harmony"].astype(np.float32),
        X_pca          = adata.obsm["X_pca"].astype(np.float32),
        obs_names      = np.array(adata.obs_names.tolist()),
        pca_components = adata.uns["pca"]["components"].astype(np.float32),
        pca_var_ratio  = np.array(
            adata.uns["pca"]["explained_variance_ratio"], dtype=np.float32),
    )

    log(f"  Writing obs metadata → {obs_path}")
    adata.obs.to_parquet(obs_path)

    log(f"  Worker complete (RAM: {rss():.1f} GB). Exiting — OS will reclaim all.")


# ═══════════════════════════════════════════════════════════════════════════
#  SUBPROCESS LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════

def run_phase10_subprocess(train_p9_path, output_dir, cfg, logger,
                           timeout=7200):
    """
    Launch phase10_worker.py (a thin CLI shim over run_worker_logic) in a
    fresh subprocess with allocator-safe environment variables set.
    """
    output_dir = Path(output_dir)
    splits_dir = cfg["paths"]["_splits"]
    name       = cfg.get("dataset", {}).get("name", "dataset")

    train_p10  = str(splits_dir / f"{name}_train_p10.h5ad")
    embed_path = str(output_dir / f"{name}_train_p10_embed.npz")
    obs_path   = str(output_dir / f"{name}_train_p10_obs.parquet")

    # exp_009.1 rebuild (Defect 5): read the batch key from the ACTUAL config location the user sets —
    # phase10_confounders.batch_correction.batch_key — with dataset.batch_col as a back-compat fallback.
    # The prior code read ONLY dataset.batch_col (default 'gem_group'), so the config's batch_key was DEAD:
    # a config saying batch_key: gemgroup silently ran on 'gem_group' (or fell through to uncorrected PCA).
    # startup validation (spore_light.validate_config_against_data) already hard-failed if this key is
    # enabled-but-absent from obs, so by here it is present (or batch correction is disabled).
    bc = cfg.get("phase10_confounders", {}).get("batch_correction", {})
    if not bc.get("enabled", True):
        # sentinel is guaranteed absent from obs -> run_harmony falls back to uncorrected PCA (= disabled)
        batch_key = "__NO_BATCH_CORRECTION__"
        logger.info("  Phase 10: batch_correction disabled in config — using uncorrected PCA")
    else:
        batch_key = (bc.get("batch_key")
                     or cfg.get("dataset", {}).get("batch_col")
                     or "gem_group")
        logger.info(f"  Phase 10: batch_key = '{batch_key}' (from batch_correction.batch_key)")

    worker = str(Path(__file__).parent / "phase10_worker.py")
    if not os.path.exists(worker):
        raise FileNotFoundError(
            f"phase10_worker.py not found at: {worker}\n"
            f"Ensure it is in the same src/ directory as phase10_confounders.py")

    child_env = os.environ.copy()
    child_env.update({
        "MALLOC_MMAP_THRESHOLD_": "134217728",   
        "MALLOC_ARENA_MAX":       "2",            
        "MALLOC_TRIM_THRESHOLD_": "131072",       
        "OPENBLAS_NUM_THREADS":   "1",            
        "MKL_NUM_THREADS":        "1",
        "OMP_NUM_THREADS":        "1",
        "NUMEXPR_NUM_THREADS":    "1",
    })

    cmd = [
        sys.executable, worker,
        "--input",            str(train_p9_path),
        "--output",           train_p10,
        "--embed",            embed_path,
        "--obs",              obs_path,
        "--batch-key",        batch_key,
        "--n-comps",          "50",
        "--n-sample",         "100000",
        "--chunk-rows",       "50000",
        "--max-harmony-iter", "20",
    ]

    parent_rss = psutil.Process().memory_info().rss / 1e9
    logger.info(
        f"  Launching Phase 10 subprocess (parent RSS: {parent_rss:.1f} GB)")
    logger.info(
        f"    env: MALLOC_MMAP_THRESHOLD_=134217728, MALLOC_ARENA_MAX=2")
    logger.info(f"    Input  → {train_p9_path}")
    logger.info(f"    Output → {train_p10}")

    result = subprocess.run(
        cmd, env=child_env, check=False, text=True, capture_output=False)

    after_rss = psutil.Process().memory_info().rss / 1e9
    logger.info(
        f"  Subprocess exited (code={result.returncode}). "
        f"Parent RSS: {after_rss:.1f} GB (parent never loaded the matrix)")

    if result.returncode != 0:
        raise RuntimeError(
            f"Phase 10 subprocess failed (exit code {result.returncode}). "
            f"See worker stdout/stderr above for details.")

    return {
        "train_p10_path": train_p10,
        "embed_path":     embed_path,
        "obs_path":       obs_path,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD LIGHTWEIGHT RESULTS INTO PARENT
# ═══════════════════════════════════════════════════════════════════════════

def load_phase10_results(output_paths, logger):
    """
    Load only the 300 MB embeddings back into the parent process.

    Returns a minimal AnnData with:
      .obsm['X_pca_harmony']  — for Phase 11 cell line detection
      .obsm['X_pca']          — backup embedding
      .obs                    — full obs DataFrame (perturbation labels etc.)
      .uns['pca']             — PCA metadata

    No expression matrix in memory. Phase 12 reloads train_p10.h5ad from disk.
    """
    import pandas as pd
    import anndata as ad

    logger.info(f"  Loading embeddings (~300 MB): {output_paths['embed_path']}")
    data   = np.load(output_paths["embed_path"])
    obs_df = pd.read_parquet(output_paths["obs_path"])
    n_obs  = len(obs_df)

    # Placeholder X (1 column of zeros) — Phase 11 never reads it
    adata_p10 = ad.AnnData(
        X=sp.csr_matrix((n_obs, 1), dtype=np.float32),
        obs=obs_df)
    adata_p10.obsm["X_pca_harmony"] = data["X_pca_harmony"].astype(np.float32)
    adata_p10.obsm["X_pca"]         = data["X_pca"].astype(np.float32)
    adata_p10.uns["pca"] = {
        "components":               data["pca_components"].astype(np.float32),
        "explained_variance_ratio": data["pca_var_ratio"].tolist(),
    }

    parent_rss = psutil.Process().memory_info().rss / 1e9
    # f-string fix: single quotes inside double-quoted f-string (Python 3.9)
    harmony_shape = adata_p10.obsm['X_pca_harmony'].shape
    obs_shape     = adata_p10.obs.shape
    logger.info(
        f"  Loaded: X_pca_harmony={harmony_shape}, "
        f"obs={obs_shape}. Parent RSS={parent_rss:.1f} GB")
    return adata_p10

def strip_ghost_genes(splits_dir, dataset_name, logger):
    """
    Loads the P9 matrices from disk, reads the 'spore_core_features' from .uns,
    slices out the temporary Ghost genes, and saves the strict 5,000-feature matrices as P10.
    """
    import anndata as ad
    import numpy as np
    import gc
    from pathlib import Path
    
    logger.info("  Step 4: Executing Ghost Excision (returning to Core Feature envelope)...")
    
    # ── Load Train First to extract the Core Feature list ──
    train_p9_path = Path(splits_dir) / f"{dataset_name}_train_p9.h5ad"
    if not train_p9_path.exists():
        logger.warning("  WARNING: train_p9.h5ad not found. Ghost excision skipped.")
        return
        
    logger.info("    Slicing train split...")
    tmp_ad = ad.read_h5ad(train_p9_path)
    
    if "spore_core_features" not in tmp_ad.uns:
        logger.warning("  WARNING: 'spore_core_features' not found in train_p9.uns. Ghost excision skipped.")
        del tmp_ad
        return
        
    core_set = set(tmp_ad.uns["spore_core_features"])
    keep_mask = np.array([g in core_set for g in tmp_ad.var_names])
    n_dropped = len(keep_mask) - keep_mask.sum()
    
    if n_dropped > 0:
        tmp_ad = tmp_ad[:, keep_mask].copy()
        logger.info(f"      Dropped {n_dropped} Ghost genes. Final shape: {tmp_ad.shape}")
    else:
        logger.info(f"      No Ghost genes found. Shape remains: {tmp_ad.shape}")
        
    tmp_ad.write_h5ad(Path(splits_dir) / f"{dataset_name}_train_p10.h5ad")
    del tmp_ad
    for _ in range(3): gc.collect()
    
    # ── Now process Val and Test splits ──
    for key in ["val", "test"]:
        p9_path = Path(splits_dir) / f"{dataset_name}_{key}_p9.h5ad"
        p10_path = Path(splits_dir) / f"{dataset_name}_{key}_p10.h5ad"
        
        if not p9_path.exists():
            continue
            
        logger.info(f"    Slicing {key} split...")
        tmp_ad = ad.read_h5ad(p9_path)
        keep_mask = np.array([g in core_set for g in tmp_ad.var_names])
        n_dropped = len(keep_mask) - keep_mask.sum()
        
        if n_dropped > 0:
            tmp_ad = tmp_ad[:, keep_mask].copy()
            logger.info(f"      Dropped {n_dropped} Ghost genes. Final shape: {tmp_ad.shape}")
        else:
            logger.info(f"      No Ghost genes found. Shape remains: {tmp_ad.shape}")
            
        tmp_ad.write_h5ad(p10_path)
        del tmp_ad
        for _ in range(3): gc.collect()
        
    logger.info("  Ghost Excision complete. Dataset strict envelope restored.")

def ghost_excision_streaming(cfg: dict, logger, chunk_rows: int = 50_000):
    """
    Remove stacked ghost (cell-cycle) genes from p9 splits by streaming.
    Never loads a full split into the parent's RAM.
    Peak RAM = parent overhead + one 50k-row chunk (~0.5 GB).
    """
    import anndata as ad
    import ctypes
    from ctypes.util import find_library

    dataset_name = cfg["dataset"]["name"]
    splits_dir   = cfg["paths"]["_splits"]

    def malloc_trim_now():
        try:
            libc = ctypes.CDLL(find_library("c") or "libc.so.6")
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype  = ctypes.c_int
            libc.malloc_trim(0)
        except Exception:
            pass

    # Read the core feature list from the p8 train file (backed = 0 GB cost)
    p8_path = splits_dir / f"{dataset_name}_train_p8.h5ad"
    if not p8_path.exists():
        logger.warning("  Ghost Excision: train_p8.h5ad not found — copying p9 → p10")
        import shutil
        for key in ["train", "val", "test"]:
            src = splits_dir / f"{dataset_name}_{key}_p9.h5ad"
            dst = splits_dir / f"{dataset_name}_{key}_p10.h5ad"
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"  [{key}] p9 → p10 (copy, no ghost removal)")
        return

    p8_backed     = ad.read_h5ad(p8_path, backed="r")
    all_var_names = list(p8_backed.var_names)
    core_features = p8_backed.uns.get("spore_core_features", None)
    p8_backed.file.close()

    if not core_features:
        logger.warning("  Ghost Excision: spore_core_features missing — skipping")
        return

    core_set  = set(core_features)
    keep_mask = np.array([g in core_set for g in all_var_names])
    gene_idx  = np.where(keep_mask)[0]
    n_ghost   = int((~keep_mask).sum())
    n_kept    = int(keep_mask.sum())
    logger.info(f"  Ghost Excision: removing {n_ghost} ghost gene(s), "
                f"keeping {n_kept} core HVGs")

    for key in ["train", "val", "test"]:
        p9_path  = splits_dir / f"{dataset_name}_{key}_p9.h5ad"
        p10_path = splits_dir / f"{dataset_name}_{key}_p10.h5ad"

        if not p9_path.exists():
            logger.warning(f"  [{key}] p9 file not found, skipping")
            continue

        rss = psutil.Process().memory_info().rss / 1e9
        logger.info(f"  [{key}] Streaming ghost excision  (RAM before: {rss:.1f} GB)")

        backed = ad.read_h5ad(p9_path, backed="r")
        n_obs  = backed.n_obs

        chunks = []
        for start in range(0, n_obs, chunk_rows):
            end   = min(start + chunk_rows, n_obs)
            chunk = backed.X[start:end]
            if sp.issparse(chunk):
                chunk = chunk.tocsr()[:, gene_idx]
            else:
                chunk = sp.csr_matrix(chunk[:, gene_idx])
            chunks.append(chunk.astype(np.float32))
            del chunk

        X_new     = sp.vstack(chunks, format="csr")
        del chunks
        gc.collect()

        var_new   = backed.var.iloc[gene_idx].copy()
        obs_new   = backed.obs.copy()
        adata_out = ad.AnnData(X=X_new, obs=obs_new, var=var_new)
        backed.file.close()

        adata_out.write_h5ad(p10_path)
        del adata_out, X_new
        gc.collect()
        malloc_trim_now()

        rss = psutil.Process().memory_info().rss / 1e9
        logger.info(f"  [{key}] {n_obs:,} × {n_kept:,} saved → {p10_path}  "
                    f"(RAM after: {rss:.1f} GB)")

# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC NOTEBOOK ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_phase10(splits, cfg, logger):
    """
    Called from the notebook. Three steps:
      1. force_heap_return() — best-effort parent cleanup
      2. run_phase10_subprocess() — worker does computation in fresh process
      3. load_phase10_results() — parent reloads only 300 MB embeddings

    Returns {'train': lightweight_adata, '_output_paths': {...}}
    """
    from .utils import log_phase_header, log_memory
    log_phase_header(logger, 10, "Confounder Mitigation")
    log_memory(logger, "Phase 10 start")

    name          = cfg.get("dataset", {}).get("name", "dataset")
    splits_dir    = cfg["paths"]["_splits"]
    train_p9_path = str(splits_dir / f"{name}_train_p9.h5ad")
    output_dir    = str(cfg["paths"]["_processed"])

    if not os.path.exists(train_p9_path):
        raise FileNotFoundError(
            f"train_p9.h5ad not found: {train_p9_path}\n"
            f"Run Phase 9 (save-all) before Phase 10.")

    logger.info("  Step 1: aggressive parent cleanup (IPython + malloc_trim)...")
    force_heap_return(
        splits_dict=splits,
        var_names=["splits", "adata", "train", "val", "test"],
        logger=logger)

    logger.info("  Step 2: Phase 10 in fresh subprocess...")
    output_paths = run_phase10_subprocess(
        train_p9_path, output_dir, cfg, logger)

    logger.info("  Step 3: loading lightweight results into parent...")
    adata_p10 = load_phase10_results(output_paths, logger)
    
    # Ghost Excision: strip temporary cell-cycle genes via streaming disk reads.
    # ghost_excision_streaming reads each p9 split in 50k-row chunks so the
    # parent never loads a full 1.45M-cell matrix into RAM.
    ghost_excision_streaming(cfg, logger)
    
    log_memory(logger, "Phase 10 complete")

    return {"train": adata_p10, "_output_paths": output_paths}
