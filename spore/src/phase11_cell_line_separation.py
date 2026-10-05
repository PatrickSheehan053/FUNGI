"""
SPORE+ · src/phase11_cell_line_separation.py
─────────────────────────────────────────────
Phase 11: Cell Line Detection & Separation

This phase identifies distinct cell line populations and adds a
'sporeplus_cell_line' column to .obs. It uses a 3-tier approach:

  Tier 1 (labeled):   .obs already has a cell line column (detected in Phase 1)
  Tier 2 (hinted):    expected_n_cell_lines known → k-means on Harmony PCA
  Tier 3 (automatic): HDBSCAN + GMM BIC sweep + bootstrap stability validation

After labeling, a V-direction validation checks whether detected populations
have genuinely divergent regulatory contexts (different regulatory programs)
or just differ in state/intensity (same program, don't split).

MEMORY SAFETY:
  All clustering operates on adata.obsm['X_pca_harmony'] (or 'X_pca'),
  which is a compact float32 matrix. Never loads adata.X for clustering.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from .utils import log_phase_header, snapshot, log_memory, force_gc


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 1: USE EXISTING LABELS
# ═══════════════════════════════════════════════════════════════════════════

def _tier1_use_existing_labels(adata, cell_line_col: str, logger) -> pd.Series:
    """Use existing .obs cell line labels directly."""
    labels = adata.obs[cell_line_col].astype(str)
    unique = sorted(labels.unique().tolist())
    logger.info(
        f"  Tier 1: using existing '{cell_line_col}' column → "
        f"{len(unique)} cell lines: {unique[:8]}"
        f"{'...' if len(unique) > 8 else ''}")
    return labels


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: GET PCA EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def _get_embedding(adata, logger):
    """
    Retrieve the best available PCA embedding for clustering.
    Preference: X_pca_harmony > X_pca.
    """
    if "X_pca_harmony" in adata.obsm:
        logger.info("  Using Harmony-corrected PCA for cell line clustering")
        return adata.obsm["X_pca_harmony"].astype(np.float32)
    elif "X_pca_chitin" in adata.obsm:
        logger.info("  Using CHITIN PCA for cell line clustering")
        return adata.obsm["X_pca_chitin"].astype(np.float32)
    elif "X_pca" in adata.obsm:
        logger.info("  Using standard PCA for cell line clustering")
        return adata.obsm["X_pca"].astype(np.float32)
    else:
        logger.warning(
            "  No PCA embedding found. "
            "Run Phase 10 (confounder mitigation) before Phase 11.")
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  THE LEIDEN OPTIMIZER (Replaces K-Means and GMM)
# ═══════════════════════════════════════════════════════════════════════════

def _run_leiden_with_target_k(adata, target_k: int, logger, use_rep="X_pca_harmony"):
    """
    Builds a KNN graph and uses a binary search to find the exact Leiden 
    resolution that yields the target_k number of clusters.
    """
    import scanpy as sc
    import pandas as pd

    logger.info(f"    Building KNN graph on {use_rep}...")
    # Build the graph once
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)

    logger.info(f"    Running binary search for Leiden resolution to hit K={target_k}...")
    res_low, res_high = 0.01, 5.0
    res = 1.0
    best_labels = None
    best_k_diff = float('inf')

    # Binary search loop (max 20 iterations)
    for i in range(20):
        sc.tl.leiden(adata, resolution=res, key_added="temp_leiden", random_state=42)
        n_clusters = adata.obs["temp_leiden"].nunique()

        # Track the closest match just in case it oscillates and misses exact K
        if abs(n_clusters - target_k) < best_k_diff:
            best_k_diff = abs(n_clusters - target_k)
            best_labels = adata.obs["temp_leiden"].copy()

        if n_clusters == target_k:
            logger.info(f"    ✓ Target K={target_k} reached at resolution={res:.3f}")
            break
        elif n_clusters > target_k:
            res_high = res
            res = (res + res_low) / 2.0
        else:
            res_low = res
            res = (res + res_high) / 2.0

    if adata.obs["temp_leiden"].nunique() != target_k:
        logger.warning(f"    ⚠ Binary search settled on K={best_labels.nunique()} instead of exact target K={target_k}.")
        
    labels_int = best_labels.astype(int)
    labels = pd.Series([f"cell_line_{i+1}" for i in labels_int], index=adata.obs_names, dtype="category")
    
    # Cleanup temp column
    del adata.obs["temp_leiden"]
    return labels

# ═══════════════════════════════════════════════════════════════════════════
#  TIER 2: HINT-GUIDED LEIDEN
# ═══════════════════════════════════════════════════════════════════════════

def _tier2_hint_guided_kmeans(adata, n_cell_lines: int, logger):
    """
    Note: Function kept original name for pipeline compatibility, 
    but now completely powered by Graph-based Leiden clustering.
    """
    emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"

    logger.info(f"  Tier 2: Hint-guided Leiden clustering (target K={n_cell_lines}) on {adata.n_obs:,} cells...")

    labels = _run_leiden_with_target_k(adata, target_k=n_cell_lines, logger=logger, use_rep=emb_key)

    for lbl in sorted(labels.unique()):
        n = (labels == lbl).sum()
        logger.info(f"    {lbl}: {n:,} cells")
    return labels

# ═══════════════════════════════════════════════════════════════════════════
#  TIER 3: AUTOMATIC DETECTION (HDBSCAN + LEIDEN)
# ═══════════════════════════════════════════════════════════════════════════

def _tier3_automatic_detection(adata, p11: dict, logger):
    """
    Fully automatic cell line detection:
    1. HDBSCAN (subsampled) to get the true biological K.
    2. Leiden binary search to gracefully assign all cells to K clusters.
    """
    import numpy as np
    import pandas as pd
    
    auto_cfg = p11.get("auto_detect", {})
    metadata = {}
    
    emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    emb = adata.obsm[emb_key]
    n_pcs = min(30, emb.shape[1])
    emb_sub = emb[:, :n_pcs]

    # Step 1: HDBSCAN for K estimate (Subsampled for speed)
    MAX_HDBSCAN_CELLS = auto_cfg.get("max_hdbscan_cells", 50_000)
    try:
        import hdbscan
        n_hdbscan = min(MAX_HDBSCAN_CELLS, adata.n_obs)
        
        if n_hdbscan < adata.n_obs:
            rng_hdb = np.random.default_rng(42)
            hdb_idx = rng_hdb.choice(adata.n_obs, size=n_hdbscan, replace=False)
            emb_hdbscan = emb_sub[hdb_idx]
            logger.info(f"  HDBSCAN: subsampled to {n_hdbscan:,}/{adata.n_obs:,} cells")
        else:
            emb_hdbscan = emb_sub

        min_cluster_size = max(50, int(n_hdbscan * 0.01))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, metric="euclidean", core_dist_n_jobs=1)
        clusterer.fit(emb_hdbscan)
        
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        hdbscan_k = max(1, n_clusters)
        noise_frac = (clusterer.labels_ == -1).mean()
        
        logger.info(f"  HDBSCAN found K={hdbscan_k} biological clusters ({noise_frac*100:.1f}% noise points in sample)")
        metadata["hdbscan_k"] = hdbscan_k
        
    except ImportError:
        logger.warning("  hdbscan not installed. Defaulting K=1.")
        hdbscan_k = 1

    # Step 2: Fit full dataset using Leiden targeting the HDBSCAN K
    if hdbscan_k == 1:
        labels = pd.Series(["cell_line_1"] * adata.n_obs, index=adata.obs_names, dtype="category")
        logger.info("  Tier 3: K=1 detected — single cell line dataset")
        confidence = 1.0
    else:
        logger.info(f"  Mapping full dataset using Leiden targeting K={hdbscan_k}...")
        labels = _run_leiden_with_target_k(adata, target_k=hdbscan_k, logger=logger, use_rep=emb_key)
        confidence = 0.9  # Baseline high confidence since we backed it with HDBSCAN
        
        for lbl in sorted(labels.unique()):
            n = (labels == lbl).sum()
            logger.info(f"    {lbl}: {n:,} cells ({n/adata.n_obs*100:.1f}%)")

    return labels, hdbscan_k, confidence, metadata

# ═══════════════════════════════════════════════════════════════════════════
#  V-DIRECTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def _v_direction_validation(adata, cell_line_labels, cfg, logger):
    """
    For each detected cell line, compute V = O_pert - C_ctrl in gene space.
    Compare V vectors across cell lines.

    If cosine similarity between any two cell lines' V vectors > threshold:
    → Same regulatory program, likely metabolic state or batch artifact.
    → Flag for potential merge.

    If cosine similarity < threshold:
    → Divergent regulatory programs → genuine cell lines → split warranted.

    Returns: validation_results dict with per-cell-line V vectors and
    pairwise cosine similarities.
    """
    p11         = cfg.get("phase11_cell_line", {})
    cos_thresh  = p11.get("v_direction_angle_threshold", 0.5)
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label  = cfg.get("dataset", {}).get("control_label", "non-targeting")

    import scipy.sparse as sp
    unique_lines = sorted(cell_line_labels.unique().tolist())

    V_vectors = {}
    for line in unique_lines:
        line_mask  = cell_line_labels == line
        line_adata = adata[line_mask]

        pert_values = line_adata.obs[pert_col].values
        ctrl_mask_l = pert_values == ctrl_label
        pert_mask_l = ~ctrl_mask_l

        if ctrl_mask_l.sum() < 10 or pert_mask_l.sum() < 10:
            logger.warning(
                f"  V-direction: '{line}' has too few ctrl ({ctrl_mask_l.sum()}) "
                f"or pert ({pert_mask_l.sum()}) cells — skipping")
            continue

        X = line_adata.X
        if sp.issparse(X):
            C_ctrl = np.asarray(X[ctrl_mask_l].mean(axis=0)).flatten()
            O_pert = np.asarray(X[pert_mask_l].mean(axis=0)).flatten()
        else:
            C_ctrl = X[ctrl_mask_l].mean(axis=0)
            O_pert = X[pert_mask_l].mean(axis=0)

        V = O_pert - C_ctrl
        V_norm = np.linalg.norm(V)
        if V_norm > 1e-10:
            V_vectors[line] = V / V_norm
        else:
            logger.warning(f"  V-direction: '{line}' V is near-zero")

    # Compute pairwise cosines between all V vectors
    line_names = list(V_vectors.keys())
    n = len(line_names)
    cosines = {}
    merge_flags = []

    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = line_names[i], line_names[j]
            cos = float(np.dot(V_vectors[name_i], V_vectors[name_j]))
            cosines[(name_i, name_j)] = cos
            if cos > cos_thresh:
                merge_flags.append((name_i, name_j, cos))
                logger.warning(
                    f"  V-direction: '{name_i}' ↔ '{name_j}' "
                    f"cos={cos:.3f} > {cos_thresh} → PARALLEL programs. "
                    f"May be same cell line in different states.")
            else:
                logger.info(
                    f"  V-direction: '{name_i}' ↔ '{name_j}' "
                    f"cos={cos:.3f} → DIVERGENT programs. Split warranted.")

    validation = {
        "V_vectors":    V_vectors,
        "pairwise_cos": cosines,
        "merge_flags":  merge_flags,
        "n_merge_flagged": len(merge_flags),
    }
    return validation


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 11
# ═══════════════════════════════════════════════════════════════════════════

def run_phase11(adata, cfg: dict, logger):
    """
    Run cell line detection and add 'sporeplus_cell_line' to .obs.

    Returns: (adata_with_cell_line_label, detection_meta dict)
    """
    log_phase_header(logger, 11, "Cell Line Detection & Separation")
    p11 = cfg.get("phase11_cell_line", {})

    enabled = p11.get("enabled", "auto")
    if enabled is False:
        logger.info("  Phase 11: DISABLED in config")
        adata.obs["sporeplus_cell_line"] = "single_line"
        return adata, {"skipped": True, "n_cell_lines": 1}

    log_memory(logger, "Phase 11 start")
    detection_meta = {}

    # ── Determine which tier to use ────────────────────────────────────────
    cell_line_col  = p11.get("cell_line_col")   # populated by Phase 1 Tier 1
    expected_n     = p11.get("expected_n_cell_lines")

    if cell_line_col and cell_line_col in adata.obs.columns:
        # ── TIER 1: existing labels ────────────────────────────────────────
        logger.info(f"  → Tier 1: using existing column '{cell_line_col}'")
        labels = _tier1_use_existing_labels(adata, cell_line_col, logger)
        detection_meta["tier_used"] = 1
        detection_meta["source_col"] = cell_line_col

    elif expected_n and int(expected_n) > 0:
        # ── TIER 2: hint-guided k-means ────────────────────────────────────
        logger.info(f"  → Tier 2: hint-guided k-means (expected K={expected_n})")
        labels = _tier2_hint_guided_kmeans(adata, int(expected_n), logger)
        if labels is None:
            logger.warning("  Tier 2 failed (no PCA embedding). Defaulting to single line.")
            adata.obs["sporeplus_cell_line"] = "single_line"
            return adata, {"error": "no_embedding", "n_cell_lines": 1}
        detection_meta["tier_used"] = 2
        detection_meta["expected_n"] = expected_n

    elif p11.get("auto_detect", {}).get("enabled", True):
        # ── TIER 3: fully automatic ────────────────────────────────────────
        logger.info("  → Tier 3: fully automatic detection")
        labels, n_detected, confidence, auto_meta = _tier3_automatic_detection(
            adata, p11, logger)
        if labels is None:
            logger.warning("  Tier 3 failed. Defaulting to single line.")
            adata.obs["sporeplus_cell_line"] = "single_line"
            return adata, {"error": "tier3_failed", "n_cell_lines": 1}
        detection_meta.update({
            "tier_used": 3,
            "n_detected": n_detected,
            "confidence": confidence,
            **auto_meta,
        })
    else:
        logger.info("  No detection method available. Treating as single cell line.")
        adata.obs["sporeplus_cell_line"] = "single_line"
        return adata, {"n_cell_lines": 1}

    # ── Apply labels to .obs ───────────────────────────────────────────────
    adata.obs["sporeplus_cell_line"] = labels.values
    unique_lines = sorted(labels.unique().tolist())
    n_lines = len(unique_lines)
    detection_meta["cell_lines"] = unique_lines
    detection_meta["n_cell_lines"] = n_lines

    logger.info(f"  {n_lines} cell line(s) detected: {unique_lines[:8]}")

    # ── Filter cell lines with too few cells ───────────────────────────────
    min_cells = p11.get("min_cells_per_cell_line", 500)
    line_counts = adata.obs["sporeplus_cell_line"].value_counts()
    small_lines = line_counts[line_counts < min_cells].index.tolist()
    if small_lines:
        logger.warning(
            f"  ⚠ {len(small_lines)} cell line(s) have < {min_cells} cells "
            f"and will be EXCLUDED from Phase 12 output: {small_lines}")
        detection_meta["excluded_small_lines"] = small_lines

    # ── V-direction validation ─────────────────────────────────────────────
    if p11.get("v_direction_validation", True) and n_lines > 1:
        logger.info("  Running V-direction validation...")
        try:
            v_valid = _v_direction_validation(adata, labels, cfg, logger)
            detection_meta["v_validation"] = {
                "pairwise_cos": {
                    f"{k[0]}_{k[1]}": v
                    for k, v in v_valid["pairwise_cos"].items()},
                "n_merge_flagged": v_valid["n_merge_flagged"],
            }
            if v_valid["n_merge_flagged"] > 0:
                logger.warning(
                    f"  ⚠ {v_valid['n_merge_flagged']} cell line pair(s) have parallel "
                    f"V vectors. These may be the same cell line in different states "
                    f"rather than genuinely distinct cell lines.")
        except Exception as e:
            logger.warning(f"  V-direction validation failed: {e}")

    # ── auto mode: if single cell line detected, effectively a no-op ───────
    if enabled == "auto" and n_lines == 1:
        logger.info(
            "  Single cell line detected — Phase 12 will produce "
            "standard (non-split) output.")

    snapshot(adata, "Post Phase 11 Cell Line Detection", logger)
    log_memory(logger, "Phase 11 end")
    return adata, detection_meta

def generate_pca_diagnostic_table(adata_p10, cfg, top_n=5):
    """
    Extracts the top genes driving PC1, PC2, and PC3 via PCA loadings,
    then calculates their mean normalized expression per cell line
    by reading lazily from the backed Phase 9 disk file.
    """
    import anndata as ad
    import pandas as pd
    import numpy as np
    
    if "pca" not in adata_p10.uns or "components" not in adata_p10.uns["pca"]:
        print("PCA components not found. Cannot generate diagnostic table.")
        return None

    components = adata_p10.uns["pca"]["components"]  # Shape: (n_pcs, n_genes)
    
    # Ensure we actually have 3 PCs to evaluate
    if components.shape[0] < 3:
        print("Less than 3 Principal Components available in the model.")
        return None

    # Setup path to the full normalized data on disk
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    splits_dir   = cfg.get("paths", {}).get("_splits", cfg.get("paths", {}).get("splits_dir"))
    p9_path      = f"{splits_dir}/{dataset_name}_train_p9.h5ad"
    
    try:
        # Open in backed mode (0 GB RAM used)
        adata_p9 = ad.read_h5ad(p9_path, backed='r')
        gene_names = np.array(adata_p9.var_names)
    except FileNotFoundError:
        print(f"Could not find {p9_path} on disk. Make sure Phase 9 was saved.")
        return None

    # Get the indices of the highest absolute weights for PC1, PC2, and PC3
    top_pc1_idx = np.argsort(np.abs(components[0]))[-top_n:][::-1]
    top_pc2_idx = np.argsort(np.abs(components[1]))[-top_n:][::-1]
    top_pc3_idx = np.argsort(np.abs(components[2]))[-top_n:][::-1]
    
    top_genes_pc1 = gene_names[top_pc1_idx]
    top_genes_pc2 = gene_names[top_pc2_idx]
    top_genes_pc3 = gene_names[top_pc3_idx]
    
    # Combine into a unique set so we only pull distinct genes from the disk once
    target_genes = list(set(top_genes_pc1) | set(top_genes_pc2) | set(top_genes_pc3))

    print(f"Extracting expression for {len(target_genes)} driver genes from disk...")
    
    # Load ONLY the target genes into memory
    expr_chunk = adata_p9[:, target_genes].to_memory()
    expr_chunk.obs["cell_line"] = adata_p10.obs["sporeplus_cell_line"].values
    
    # Compute mean expression per cell line
    df_expr = pd.DataFrame(
        expr_chunk.X.toarray() if hasattr(expr_chunk.X, "toarray") else expr_chunk.X, 
        columns=target_genes
    )
    df_expr["Cell Line"] = expr_chunk.obs["cell_line"].values
    mean_expr = df_expr.groupby("Cell Line").mean().T

    # Format the final diagnostic table
    records = []
    
    # Iterate through all three PCs
    pc_data = [
        ("PC1", top_genes_pc1, components[0][top_pc1_idx]), 
        ("PC2", top_genes_pc2, components[1][top_pc2_idx]),
        ("PC3", top_genes_pc3, components[2][top_pc3_idx])
    ]
    
    for pc, top_genes, weights in pc_data:
        for gene, weight in zip(top_genes, weights):
            row = {"Principal Component": pc, "Driver Gene": gene, "Loading Weight": round(weight, 3)}
            # Add the mean expression values
            for cl in mean_expr.columns:
                row[f"{cl} (Mean CP10k)"] = round(mean_expr.loc[gene, cl], 2)
            records.append(row)

    diag_df = pd.DataFrame(records)
    return diag_df
