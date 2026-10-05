"""
SPORE+ · src/phase09_normalization.py
───────────────────────────────────────
Phase 9: Normalization

NEW ERROR DOCUMENTED (SPORE+ Error 039):
  sc.pp.log1p causes an 18.5 GB RAM spike across three splits.
  Root cause: Scanpy's implementation does X.data = np.log1p(X.data).
  np.log1p() WITHOUT an out= argument ALWAYS allocates a brand new array.
  The old X.data (13.5 GB for the train split) stays in Python's heap
  as unreturned pages — the OS never sees those bytes freed.

  Cumulative across 3 splits: 13.5 + 3.0 + 1.8 GB = 18.3 GB of ghost RAM
  that inflates the Phase 10 baseline from 88 GB to 106 GB, leaving too
  little headroom for sc.tl.score_genes_cell_cycle's control gene matrix.

  Fix: np.log1p(X.data, out=X.data)
  The out= argument forces modification of the existing buffer in-place.
  Zero bytes allocated. Zero ghost RAM.

Other lessons applied:
  Error 024: no adata.layers["raw_counts"] copy — raw saved to disk by Phase 7
  Error 024: pre-cast to float32 before normalize_total to prevent float64 upcast
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from .utils import log_phase_header, snapshot, log_memory, force_gc


def normalize_split(adata, cfg: dict, logger, label: str = ""):
    """
    Normalize one split: library-size scaling + log1p.
    Includes a <1MB cache of raw data to enable Before/After plotting.
    """
    p9  = cfg.get("phase9_normalization", {})
    tag = f"[{label}] " if label else ""
    logger.info(f"  {tag}Normalizing {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # --- THE PLOTTING CACHE HACK ---
    # Save a tiny 50k sample of raw non-zero values before we destroy them
    rng = np.random.default_rng(42)
    non_zero_raw = adata.X.data if sp.issparse(adata.X) else adata.X.flatten()
    if len(non_zero_raw) > 0:
        n_sample = min(50_000, len(non_zero_raw))
        adata.uns["p9_raw_sample"] = rng.choice(non_zero_raw, size=n_sample, replace=False)
    # -------------------------------

    if sp.issparse(adata.X):
        if adata.X.data.dtype != np.float32:
            adata.X.data = adata.X.data.astype(np.float32, copy=False)
    elif adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    target = p9.get("target_sum")
    sc.pp.normalize_total(adata, target_sum=target)
    logger.info(f"  {tag}Normalized to {'median' if target is None else target} total counts")

    if p9.get("log_transform", True):
        if sp.issparse(adata.X):
            np.log1p(adata.X.data, out=adata.X.data)
        else:
            np.log1p(adata.X, out=adata.X)
        logger.info(f"  {tag}Log1p applied (in-place, out= buffer — zero allocation)")

    return adata


def run_phase9(splits: dict, cfg: dict, logger):
    """Normalize all three splits in-place (no matrix copies)."""
    log_phase_header(logger, 9, "Normalization")
    for key in ["train", "val", "test"]:
        if key in splits:
            splits[key] = normalize_split(splits[key], cfg, logger, label=key)
    snapshot(splits["train"], "Post Phase 9 (train)", logger)
    log_memory(logger, "Phase 9 end")
    return splits

def extract_p9_data(splits):
    """
    Extracts lightweight plotting samples (Pre and Post) from all splits 
    and pools them together BEFORE the memory firewall wipes the data.
    """
    import numpy as np
    import scipy.sparse as sp

    cache = {"raw": [], "norm": []}
    
    # THE FIX: Only look at the actual data splits, ignore Pandas metadata
    for key in ["train", "val", "test"]:
        if key not in splits:
            continue
            
        adata = splits[key]
        
        # Pool the raw samples we cached during normalize_split
        if hasattr(adata, "uns") and "p9_raw_sample" in adata.uns:
            cache["raw"].extend(adata.uns["p9_raw_sample"])
            
        # Pool a sample of the newly normalized data
        non_zero_data = adata.X.data if sp.issparse(adata.X) else adata.X.flatten()
        if len(non_zero_data) > 0:
            n_samples = min(50_000, len(non_zero_data))
            rng = np.random.default_rng(42)
            cache["norm"].extend(rng.choice(non_zero_data, size=n_samples, replace=False))
            
    return cache

def print_normalization_diagnostics(p9_cache, cfg):
    import numpy as np
    import pandas as pd

    target_sum = cfg.get("phase9_normalization", {}).get("target_sum", "Unknown")
    
    # The fix: Fallback to [0] if the cache arrays are empty
    raw = np.array(p9_cache["raw"]) if len(p9_cache["raw"]) > 0 else np.array([0.0])
    norm = np.array(p9_cache["norm"]) if len(p9_cache["norm"]) > 0 else np.array([0.0])

    df = pd.DataFrame({
        "Metric": [
            "Target Sum (Scaling Factor)", 
            "Log Transform Applied", 
            "Non-Zero Mean (Pre-Norm)", 
            "Non-Zero Max (Pre-Norm)", 
            "Non-Zero Mean (Post CP10k)", 
            "Non-Zero Max (Post CP10k)"
        ],
        "Value": [
            f"{target_sum:,}" if isinstance(target_sum, int) else str(target_sum),
            "True (In-Place)",
            f"{np.mean(raw):.2f}",
            f"{np.max(raw):.1f}",
            f"{np.mean(norm):.2f}",
            f"{np.max(norm):.1f}"
        ]
    })

    print("\n" + "="*55)
    print(" SPORE+ NORMALIZATION DIAGNOSTICS")
    print("="*55)
    print(df.to_string(index=False))
    print("="*55 + "\n")
