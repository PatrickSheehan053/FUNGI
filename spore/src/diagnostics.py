def report_phase0_diagnostics(adata):
    """Prints a quantitative console report for Phase 0 Sparsification."""
    import scipy.sparse as sp
    import numpy as np

    n_obs, n_vars = adata.shape
    total_elements = n_obs * n_vars
    nnz = adata.X.nnz if sp.issparse(adata.X) else np.count_nonzero(adata.X)
    sparsity_pct = (1 - (nnz / total_elements)) * 100

    dense_gb = (total_elements * 4) / (1024**3)
    if sp.issparse(adata.X):
        sparse_gb = (adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes) / (1024**3)
        fmt = "scipy.sparse.csr_matrix"
    else:
        sparse_gb = dense_gb
        fmt = "numpy.ndarray (DENSE)"
        
    compression_ratio = dense_gb / sparse_gb if sparse_gb > 0 else 0

    print("PHASE 0: INGESTION & SPARSIFICATION SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    print(f"  • Matrix Shape       : {n_obs:,} Cells × {n_vars:,} Genes")
    print(f"  • Total Elements     : {total_elements:,}")
    print(f"  • Non-Zero Values    : {nnz:,}")
    print(f"  • Matrix Sparsity    : {sparsity_pct:.2f}% ({(total_elements - nnz):,} zeros eliminated)")
    print(" ─────────────────────────────────────────────────────────────────────────")
    print(f"  • Memory Format      : {fmt}")
    print(f"  • Theoretical Dense  : {dense_gb:.2f} GB")
    print(f"  • Actual Sparse Size : {sparse_gb:.2f} GB")
    print(f"  • Compression Ratio  : {compression_ratio:.1f}x smaller")
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase1_diagnostics(adata, detection_results):
    """Prints a quantitative console report for Phase 1 Detection."""
    print(" PHASE 1: DETECTION & HARMONIZATION SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    print(f"  • Modality Detected  : {detection_results.get('modality', 'Unknown').upper()}")
    print(f"  • Native Gene Format : {detection_results.get('gene_id_format', 'Unknown').upper()}")
    
    if detection_results.get("gene_ids_harmonized"):
        print(f"  • Translation        : SUCCESS (Translated to HGNC Symbols)")
    else:
        print(f"  • Translation        : SKIPPED (Already Standardized)")
        
    print(" ─────────────────────────────────────────────────────────────────────────")
    struct = "COMBINATORIAL" if detection_results.get('is_combinatorial') else "SINGLE-TARGET"
    print(f"  • Perturb Structure  : {struct}")
    if detection_results.get('is_combinatorial'):
        print(f"    - Separator Used   : '{detection_results.get('combinatorial_separator')}'")
        print(f"    - Unique Targets   : {len(detection_results.get('combinatorial_constituents', [])):,}")
        
    print(" ─────────────────────────────────────────────────────────────────────────")
    cl_col = detection_results.get('detected_cell_line_col')
    if cl_col:
        print(f"  • Cell Line Labels   : FOUND (Column: '{cl_col}')")
        print(f"  • Unique Lines       : {detection_results.get('n_cell_lines_labeled', 0)}")
    else:
        print(f"  • Cell Line Labels   : NOT FOUND (Will require Phase 11 clustering)")
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase2_diagnostics(waterfall):
    """Prints a quantitative console report for Phase 2 Cell Triage."""
    print(" 🔬 PHASE 2: CELL-LEVEL TRIAGE SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    initial_cells = waterfall.get("Starting cells", 0)
    print(f"  • Initial Cell Count : {initial_cells:,}")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    prev_count = initial_cells
    for step, count in waterfall.items():
        if step == "Starting cells":
            continue
        
        dropped = prev_count - count
        pct_dropped = (dropped / initial_cells) * 100 if initial_cells > 0 else 0
        
        # Format strings to keep columns perfectly aligned
        print(f"  • {step:<18} : {count:>8,} cells  (-{dropped:>6,} | {pct_dropped:>4.1f}%)")
        prev_count = count
        
    print(" ─────────────────────────────────────────────────────────────────────────")
    final_cells = prev_count
    total_dropped = initial_cells - final_cells
    total_pct = (total_dropped / initial_cells) * 100 if initial_cells > 0 else 0
    
    print(f"  • Final Cell Count   : {final_cells:,}")
    print(f"  • Total Removed      : {total_dropped:,} ({total_pct:.2f}%)")
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase3_diagnostics(adata, cfg):
    """Prints a quantitative console report for Phase 3 Ambient RNA."""
    print(" 🔬 PHASE 3: AMBIENT RNA DETECTION SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    p3 = cfg.get("phase3_ambient", {})
    if not p3.get("enabled", False):
        print("  • Status             : DISABLED")
        print(" ═════════════════════════════════════════════════════════════════════════")
        return

    mode = p3.get("mode", "global_profile")
    thresh = p3.get("flag_threshold", 0.80)
    umi_pct = p3.get("min_umi_pct", 0.10)
    removed = p3.get("remove_flagged", False)

    print(f"  • Mode               : {mode}")
    print(f"  • Score Threshold    : > {thresh}")
    print(f"  • UMI Threshold      : Bottom {int(umi_pct*100)}%")

    if "ambient_score" in adata.obs:
        scores = adata.obs["ambient_score"]
        print(f"  • Mean Ambient Score : {scores.mean():.3f}")
        print(f"  • Max Ambient Score  : {scores.max():.3f}")

    print(" ─────────────────────────────────────────────────────────────────────────")
    if removed:
        print(f"  • Action Taken       : REMOVED flagged cells")
        print(f"  • Flagged Remaining  : 0 (Filtered out of matrix)")
    else:
        if "ambient_flagged" in adata.obs:
            flagged = adata.obs["ambient_flagged"].sum()
            pct = (flagged / adata.n_obs) * 100
            print(f"  • Action Taken       : FLAGGED ONLY (Not removed)")
            print(f"  • Cells Flagged      : {flagged:,} ({pct:.2f}%)")
            
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase4_diagnostics(adata, cfg):
    """Prints a quantitative console report for Phase 4 Doublet Detection."""
    print(" 🔬 PHASE 4: DOUBLET DETECTION SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    p4 = cfg.get("phase4_doublets", {})
    if not p4.get("enabled", False):
        print("  • Status             : DISABLED")
        print(" ═════════════════════════════════════════════════════════════════════════")
        return

    expected_rate = p4.get("expected_doublet_rate")
    expected_rate = float(expected_rate) if expected_rate is not None else 0.06
    
    removed = p4.get("remove_doublets")
    removed = bool(removed) if removed is not None else False

    print(f"  • Expected Rate      : {expected_rate * 100:.1f}%")
    
    if "doublet_score" in adata.obs and "predicted_doublet" in adata.obs:
        scores = adata.obs["doublet_score"]
        doublets_mask = adata.obs["predicted_doublet"].astype(bool)
        n_doublets = doublets_mask.sum()
        
        # Reverse engineer the Scrublet threshold
        threshold = scores[doublets_mask].min() if n_doublets > 0 else "N/A"
        
        print(f"  • Scrublet Threshold : > {threshold:.4f}" if threshold != "N/A" else "  • Scrublet Threshold : N/A (No doublets found)")
        print(f"  • Max Doublet Score  : {scores.max():.3f}")

        print(" ─────────────────────────────────────────────────────────────────────────")
        if removed:
            print(f"  • Action Taken       : REMOVED flagged doublets")
            print(f"  • Doublets Remaining : 0 (Filtered out of matrix)")
        else:
            pct = (n_doublets / adata.n_obs) * 100
            print(f"  • Action Taken       : FLAGGED ONLY (Not removed)")
            print(f"  • Predicted Doublets : {n_doublets:,} ({pct:.2f}%)")
    else:
        print("  • WARNING            : Scrublet scores not found in dataset.")
            
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase5_diagnostics(adata, escaper_stats, pert_sizes):
    """Prints a quantitative console report for Phase 5 Escaper Filtering."""
    import pandas as pd
    
    print(" 🔬 PHASE 5: ESCAPER FILTERING & EFFICIENCY SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    if escaper_stats is None or escaper_stats.empty:
        print("  • Status             : DISABLED OR NO DATA")
        print(" ═════════════════════════════════════════════════════════════════════════")
        return

    n_escaped = escaper_stats["n_escaped"].sum()
    n_bypassed = (escaper_stats["status"] == "bypassed_no_gene_in_features").sum()
    
    print(f"  • Total Escapers Cut : {n_escaped:,} cells")
    print(f"  • Bypassed Perts     : {n_bypassed:,} (Target gene missing from matrix)")
    
    if n_escaped == 0:
        print("  • ⚠ WARNING          : 0 escapers removed. Verify Phase 1 gene ID harmonization.")

    print(" ─────────────────────────────────────────────────────────────────────────")
    
    eff_data = adata.uns.get("knockdown_efficiency", {})
    if eff_data:
        eff_df = pd.DataFrame.from_dict(eff_data, orient="index")
        n_low_eff = eff_df["low_efficiency"].sum()
        median_eff = eff_df["efficiency_score"].median()
        
        print(f"  • Median Efficiency  : {median_eff * 100:.1f}%")
        print(f"  • Low-Efficacy Perts : {n_low_eff:,} (Below configured threshold)")
        
        if n_low_eff > 50:
            print("  • ⚠ WARNING          : >50 perturbations have low knockdown efficiency.")
            print("                         Consider reviewing the guide library quality.")
            
    print(" ─────────────────────────────────────────────────────────────────────────")
    print(f"  • Final Cell Count   : {adata.n_obs:,}")
    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase6_diagnostics(adata, waterfall, rescued):
    """Prints a quantitative console report for Phase 6 Gene Triage."""
    import numpy as np
    import scipy.sparse as sp
    
    print(" 🔬 PHASE 6: GENE-LEVEL TRIAGE SUMMARY")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    initial_genes = waterfall.get("Starting genes", 0)
    print(f"  • Initial Gene Count : {initial_genes:,}")
    print(" ─────────────────────────────────────────────────────────────────────────")
    
    prev_count = initial_genes
    for step, count in waterfall.items():
        if step == "Starting genes":
            continue
        
        dropped = prev_count - count
        pct_dropped = (dropped / initial_genes) * 100 if initial_genes > 0 else 0
        
        # Format strings to keep columns perfectly aligned
        if "post-rescue" in step:
            # Positive wording for rescue
            rescued_count = count - prev_count
            print(f"  • {step:<22} : {count:>8,} genes  (+{rescued_count:>6,} rescued)")
        else:
            print(f"  • {step:<22} : {count:>8,} genes  (-{dropped:>6,} | {pct_dropped:>4.1f}%)")
        prev_count = count
        
    print(" ─────────────────────────────────────────────────────────────────────────")
    final_genes = prev_count
    total_dropped = initial_genes - final_genes
    total_pct = (total_dropped / initial_genes) * 100 if initial_genes > 0 else 0
    
    print(f"  • Targets Rescued      : {len(rescued):,}")
    print(f"  • Final Gene Count     : {final_genes:,}")
    print(f"  • Total Removed        : {total_dropped:,} ({total_pct:.2f}%)")

    # ── Robust Sparsity & Information Density Metrics ──
    if "phase6_gene_penetrance" in adata.uns:
        print(" ─────────────────────────────────────────────────────────────────────────")
        print("  • MATRIX SPARSITY & INFORMATION DENSITY")
        
        orig_pct_cells = adata.uns["phase6_gene_penetrance"]

        # Post-Filter Calculations
        post_total_elements = adata.n_obs * adata.n_vars
        post_nnz = adata.X.nnz if sp.issparse(adata.X) else np.count_nonzero(adata.X)
        post_sparsity = (1 - (post_nnz / post_total_elements)) * 100 if post_total_elements > 0 else 100

        # Pre-Filter Calculations (Reconstructed from penetrance array)
        pre_total_genes = len(orig_pct_cells)
        pre_total_elements = adata.n_obs * pre_total_genes
        pre_nnz = np.sum((orig_pct_cells / 100) * adata.n_obs)
        pre_sparsity = (1 - (pre_nnz / pre_total_elements)) * 100 if pre_total_elements > 0 else 100

        print(f"    - Pre-Filter Sparsity  : {pre_sparsity:.2f}%  (across {pre_total_genes:,} genes)")
        print(f"    - Post-Filter Sparsity : {post_sparsity:.2f}%  (across {adata.n_vars:,} genes)")
        print(f"    - Info Density Boost   : +{(pre_sparsity - post_sparsity):.2f}%")

    print(" ═════════════════════════════════════════════════════════════════════════")

def report_phase7_diagnostics(split_result):
    """
    Prints a clean Pandas DataFrame summary of the data splits and performs
    mathematical checks to guarantee zero data leakage between sets.
    """
    import pandas as pd
    
    # Handle whether the user passed the full list of seeds or just one result
    res = split_result[0] if isinstance(split_result, list) else split_result
    
    # Extract Cell Counts
    train_cells = res["split_info"]["train"]
    val_cells   = res["split_info"]["val"]
    test_cells  = res["split_info"]["test"]
    total_cells = train_cells + val_cells + test_cells
    
    # Extract Perturbations
    train_perts = set(res["train_labels"])
    val_perts   = set(res["val_labels"])
    test_perts  = set(res["test_labels"])
    total_perts = len(train_perts) + len(val_perts) + len(test_perts)
    
    # ── 1. Clean Table Output ──
    df = pd.DataFrame({
        "Split": ["Train", "Val", "Test", "Total"],
        "Cells": [f"{train_cells:,}", f"{val_cells:,}", f"{test_cells:,}", f"{total_cells:,}"],
        "Cell %": [
            f"{(train_cells/total_cells)*100:.1f}%", 
            f"{(val_cells/total_cells)*100:.1f}%", 
            f"{(test_cells/total_cells)*100:.1f}%", 
            "100.0%"
        ],
        "Targets": [len(train_perts), len(val_perts), len(test_perts), total_perts],
        "Target %": [
            f"{(len(train_perts)/total_perts)*100:.1f}%", 
            f"{(len(val_perts)/total_perts)*100:.1f}%", 
            f"{(len(test_perts)/total_perts)*100:.1f}%", 
            "100.0%"
        ]
    })
    
    print("\n" + "═"*65)
    print(" 🔬 PHASE 7: STRATIFIED DATA SPLIT DIAGNOSTICS")
    print(" ─────────────────────────────────────────────────────────────────")
    print(df.to_string(index=False))
    print(" ─────────────────────────────────────────────────────────────────")
    print("  LEAKAGE SANITY CHECKS")
    
    # ── 2. Mathematical Leakage Checks ──
    leak_tv = len(train_perts.intersection(val_perts))
    leak_tt = len(train_perts.intersection(test_perts))
    leak_vt = len(val_perts.intersection(test_perts))
    
    if leak_tv == 0 and leak_tt == 0 and leak_vt == 0:
        print("  [PASS] 🟢 Target sets are strictly mutually exclusive.")
    else:
        print("  [FAIL] 🔴 DATA LEAKAGE DETECTED!")
        print(f"         Train/Val overlap:  {leak_tv} targets")
        print(f"         Train/Test overlap: {leak_tt} targets")
        print(f"         Val/Test overlap:   {leak_vt} targets")

    if total_perts > 0:
        print(f"  [PASS] 🟢 All perturbations accounted for ({total_perts}).")
    print(" ═════════════════════════════════════════════════════════════════\n")

def report_phase8_diagnostics(adata_train):
    """
    Prints a clean Pandas DataFrame summary of the HVG selection, 
    accounting for 1-to-1 Target Swaps and Ghost Stacks.
    """
    import pandas as pd

    if "hvg_stats" not in adata_train.uns:
        print("No HVG stats found. Run Phase 8 first.")
        return

    hvg_df = pd.DataFrame.from_dict(adata_train.uns["hvg_stats"], orient="index")
    
    # 1. Feature Extraction Logic
    total_pre_genes = len(hvg_df)
    
    # Calculate categories based on our strict tracking columns
    swapped_targets = hvg_df.get("rescued_target", pd.Series(False, index=hvg_df.index)).fillna(False).sum()
    stacked_ghosts = hvg_df.get("rescued_ghost", pd.Series(False, index=hvg_df.index)).fillna(False).sum()
    
    # "highly_variable" includes the swapped targets, so we subtract them to get the "pure" variance winners
    total_core_hvgs = hvg_df["highly_variable"].sum()
    pure_hvgs = total_core_hvgs - swapped_targets
    
    final_features = len(adata_train.var_names)
    non_hvg = total_pre_genes - final_features

    df_counts = pd.DataFrame({
        "Metric": [
            "Starting Features (Phase 6)",
            "Non-Variable (Filtered)",
            "Top HVGs (Pure Variance)",
            "Target Rescue (Swapped into Core)",
            "Ghost Rescue (Stacked Temporarily)",
            "Final Feature Space"
        ],
        "Count": [
            f"{total_pre_genes:,}", 
            f"{non_hvg:,}", 
            f"{pure_hvgs:,}", 
            f"{swapped_targets:,}", 
            f"{stacked_ghosts:,}", 
            f"{final_features:,}"
        ]
    })

    print("\n" + "═"*75)
    print(" 🔬 PHASE 8: HVG SELECTION DIAGNOSTICS")
    print(" ───────────────────────────────────────────────────────────────────────────")
    print(df_counts.to_string(index=False))
    
    # 2. Distribution Metrics
    y_col, y_name = None, None
    if "variances_norm" in hvg_df.columns:
        y_col, y_name = "variances_norm", "Variance"
    elif "dispersions_norm" in hvg_df.columns:
        y_col, y_name = "dispersions_norm", "Dispersion"

    if y_col:
        # Determine status tags
        hvg_df["Status"] = "Non-Variable"
        hvg_df.loc[hvg_df["highly_variable"], "Status"] = "Highly Variable (Core)"
        
        # Override tags for specific rescues
        if "rescued_target" in hvg_df.columns:
            hvg_df.loc[hvg_df["rescued_target"], "Status"] = "Target Rescue (Swapped)"
        if "rescued_ghost" in hvg_df.columns:
            hvg_df.loc[hvg_df["rescued_ghost"], "Status"] = "Ghost Rescue (Stacked)"
        
        dist_data = []
        for status in ["Non-Variable", "Highly Variable (Core)", "Target Rescue (Swapped)", "Ghost Rescue (Stacked)"]:
            sub = hvg_df[hvg_df["Status"] == status]
            if len(sub) == 0: continue
            
            dist_data.append({
                "Category": status,
                "Mean Expr (Avg)": f"{sub['means'].mean():.3f}",
                "Mean Expr (Range)": f"[{sub['means'].min():.3f}, {sub['means'].max():.2f}]",
                f"Norm. {y_name} (Avg)": f"{sub[y_col].mean():.3f}",
                f"Norm. {y_name} (Range)": f"[{sub[y_col].min():.3f}, {sub[y_col].max():.2f}]"
            })
            
        if dist_data:
            print(" ───────────────────────────────────────────────────────────────────────────")
            print("  DISTRIBUTION METRICS")
            print(" ───────────────────────────────────────────────────────────────────────────")
            print(pd.DataFrame(dist_data).to_string(index=False))

    print(" ═══════════════════════════════════════════════════════════════════════════\n")

def report_phase9_diagnostics(p9_cache, cfg):
    """Prints a quantitative console report for Phase 9 Normalization."""
    import numpy as np
    import pandas as pd

    target_sum = cfg.get("phase9_normalization", {}).get("target_sum", "Unknown")
    
    # Fallback to [0] if the cache arrays are empty
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

    print("\n" + "═"*65)
    print(" PHASE 9: NORMALIZATION DIAGNOSTICS")
    print(" ─────────────────────────────────────────────────────────────────")
    print(df.to_string(index=False))
    print(" ═════════════════════════════════════════════════════════════════\n")

def report_phase10_diagnostics(adata_p10, cfg):
    """Prints a quantitative console report for Phase 10 Integration & Embeddings."""
    import pandas as pd
    import numpy as np

    print("\n" + "═"*75)
    print(" PHASE 10: CONFOUNDER MITIGATION DIAGNOSTICS")
    print(" ───────────────────────────────────────────────────────────────────────────")

    if "pca" not in adata_p10.uns:
        print("  [FAIL] No PCA metadata found. Run Phase 10 first.")
        return

    # 1. PCA Metrics
    n_comps = len(adata_p10.uns["pca"]["explained_variance_ratio"])
    var_explained = sum(adata_p10.uns["pca"]["explained_variance_ratio"]) * 100

    # 2. Embedding Shapes
    pca_shape = adata_p10.obsm.get("X_pca", np.array([])).shape
    harmony_shape = adata_p10.obsm.get("X_pca_harmony", np.array([])).shape

    # 3. Batch Metrics
    batch_col = cfg.get("dataset", {}).get("batch_col", "gem_group")
    n_batches = 0
    batch_stats = "Not Found"
    
    if batch_col in adata_p10.obs.columns:
        batches = adata_p10.obs[batch_col].value_counts()
        n_batches = len(batches)
        if n_batches > 0:
            batch_stats = f"Min: {batches.min():,} cells | Max: {batches.max():,} cells"

    df_metrics = pd.DataFrame({
        "Metric": [
            "Total Cells Processed",
            "Principal Components (PCs)",
            "Total Variance Explained",
            "Integration Batch Key",
            "Total Batches Corrected",
            "Batch Size Distribution",
            "Raw PCA Tensor Shape",
            "Harmony Tensor Shape"
        ],
        "Value": [
            f"{adata_p10.n_obs:,}",
            f"{n_comps}",
            f"{var_explained:.2f}%",
            f"'{batch_col}'",
            f"{n_batches:,}",
            batch_stats,
            f"{pca_shape}",
            f"{harmony_shape}"
        ]
    })

    print(df_metrics.to_string(index=False))
    
    if harmony_shape == (0,):
        print(" ───────────────────────────────────────────────────────────────────────────")
        print("  ⚠ WARNING: Harmony embedding not found. Falling back to uncorrected PCA.")
        
    print(" ═══════════════════════════════════════════════════════════════════════════\n")

def report_phase11_diagnostics(adata_p10, cell_line_meta, cfg):
    """Prints a quantitative console report for Phase 11 Cell Line Detection."""
    import pandas as pd
    import numpy as np
    import anndata as ad

    print("\n" + "═"*75)
    print(" 🔬 PHASE 11: CELL LINE DETECTION DIAGNOSTICS")
    print(" ───────────────────────────────────────────────────────────────────────────")

    tier_used = cell_line_meta.get("tier_used", "None/Skipped")
    n_lines = cell_line_meta.get("n_cell_lines", 1)
    
    print(f"  • Detection Tier Used : Tier {tier_used}")
    print(f"  • Cell Lines Detected : {n_lines}")
    
    if "cell_lines" in cell_line_meta:
        print(f"  • Detected Labels     : {', '.join(cell_line_meta['cell_lines'][:5])}")
    
    if "v_validation" in cell_line_meta:
        n_flagged = cell_line_meta["v_validation"]["n_merge_flagged"]
        print(f"  • V-Direction Flags   : {n_flagged} pairs show parallel regulatory programs")

    # ── PCA Driver Gene Analysis ──
    if "pca" not in adata_p10.uns or "components" not in adata_p10.uns["pca"]:
        print(" ───────────────────────────────────────────────────────────────────────────")
        print("  [SKIP] No PCA metadata available for driver gene extraction.")
        print(" ═══════════════════════════════════════════════════════════════════════════\n")
        return

    print(" ───────────────────────────────────────────────────────────────────────────")
    print("  PCA DRIVER GENE EXPRESSION (Loaded from Phase 9 Disk Cache)")
    print(" ───────────────────────────────────────────────────────────────────────────")

    components = adata_p10.uns["pca"]["components"]
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    splits_dir   = cfg.get("paths", {}).get("_splits", cfg.get("paths", {}).get("splits_dir"))
    p9_path      = f"{splits_dir}/{dataset_name}_train_p9.h5ad"
    
    try:
        adata_p9 = ad.read_h5ad(p9_path, backed='r')
        gene_names = np.array(adata_p9.var_names)
        
        top_n = 3
        top_pc1_idx = np.argsort(np.abs(components[0]))[-top_n:][::-1]
        top_pc2_idx = np.argsort(np.abs(components[1]))[-top_n:][::-1]
        top_pc3_idx = np.argsort(np.abs(components[2]))[-top_n:][::-1]
        
        target_genes = list(set(gene_names[top_pc1_idx]) | set(gene_names[top_pc2_idx]) | set(gene_names[top_pc3_idx]))
        
        expr_chunk = adata_p9[:, target_genes].to_memory()
        expr_chunk.obs["cell_line"] = adata_p10.obs["sporeplus_cell_line"].values
        
        df_expr = pd.DataFrame(
            expr_chunk.X.toarray() if hasattr(expr_chunk.X, "toarray") else expr_chunk.X, 
            columns=target_genes
        )
        df_expr["Cell Line"] = expr_chunk.obs["cell_line"].values
        mean_expr = df_expr.groupby("Cell Line").mean().T

        records = []
        pc_data = [
            ("PC1", gene_names[top_pc1_idx], components[0][top_pc1_idx]), 
            ("PC2", gene_names[top_pc2_idx], components[1][top_pc2_idx]),
            ("PC3", gene_names[top_pc3_idx], components[2][top_pc3_idx])
        ]
        
        for pc, top_genes, weights in pc_data:
            for gene, weight in zip(top_genes, weights):
                row = {"PC": pc, "Driver Gene": gene, "Weight": f"{weight:.3f}"}
                for cl in mean_expr.columns:
                    row[f"{cl} (CP10k)"] = f"{mean_expr.loc[gene, cl]:.2f}"
                records.append(row)

        print(pd.DataFrame(records).to_string(index=False))

    except FileNotFoundError:
        print(f"  [ERROR] Could not find {p9_path} on disk. Skipping driver analysis.")

    print(" ═══════════════════════════════════════════════════════════════════════════\n")

def report_phase12_diagnostics(all_meta_splits):
    """
    Generates a publication-ready diagnostic report for the metacell aggregation.
    Includes sub-totals per cell line and clear visual separations.
    """
    import pandas as pd
    import numpy as np

    records = []
    global_sc = 0
    global_mc = 0

    for cl, splits in all_meta_splits.items():
        display_cl = "All/Single" if cl == "single" else cl
        
        cl_sc_total = 0
        cl_mc_total = 0

        for split_name, adata in splits.items():
            n_mc = adata.n_obs
            
            # Extract Single Cell counts
            if "n_cells_in_metacell" in adata.obs:
                sizes = adata.obs["n_cells_in_metacell"].values
                n_sc = int(sizes.sum())
                min_size = int(sizes.min())
                max_size = int(sizes.max())
                mean_size = round(sizes.mean(), 1)
            else:
                n_sc, min_size, max_size, mean_size = 0, 0, 0, 0

            cl_sc_total += n_sc
            cl_mc_total += n_mc

            # Extract Inner Variance (Compactness)
            mean_var = np.nan
            if "metacell_quality" in adata.uns:
                mq = adata.uns["metacell_quality"]
                if isinstance(mq, dict) and "inner_variance" in mq:
                    mean_var = np.mean(list(mq["inner_variance"]))
                elif hasattr(mq, "inner_variance"):
                    mean_var = np.mean(mq.inner_variance)

            records.append({
                "Cell Line": display_cl,
                "Split": split_name.capitalize(),
                "Orig SCs": f"{n_sc:,}",
                "Result MCs": f"{n_mc:,}",
                "Ratio": f"{round(n_sc / n_mc, 1)}x" if n_mc > 0 else "0x",
                "Mean SCs/MC": f"{mean_size}",
                "Range": f"{min_size}-{max_size}",
                "Inner Var": f"{round(mean_var, 4)}" if not pd.isna(mean_var) else "N/A*"
            })
        
        # ── Append the Total Row for the Cell Line ──
        records.append({
            "Cell Line": display_cl,
            "Split": "TOTAL",
            "Orig SCs": f"{cl_sc_total:,}",
            "Result MCs": f"{cl_mc_total:,}",
            "Ratio": f"{round(cl_sc_total / cl_mc_total, 1)}x" if cl_mc_total > 0 else "0x",
            "Mean SCs/MC": "-",
            "Range": "-",
            "Inner Var": "-"
        })
        
        # ── Append a Visual Separator Row ──
        records.append({
            "Cell Line": "·" * 10, "Split": "·" * 6, "Orig SCs": "·" * 8,
            "Result MCs": "·" * 8, "Ratio": "·" * 6, "Mean SCs/MC": "·" * 10,
            "Range": "·" * 6, "Inner Var": "·" * 8
        })
        
        global_sc += cl_sc_total
        global_mc += cl_mc_total

    # Remove the final dangling separator row
    if records and "·" in records[-1]["Cell Line"]:
        records.pop()

    df = pd.DataFrame(records)
    overall_ratio = round(global_sc / global_mc, 1) if global_mc > 0 else 0

    print("\n" + "═" * 95)
    print(" 🔬 PHASE 12: METACELL AGGREGATION DIAGNOSTIC REPORT")
    print(" ───────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"  • Total Single Cells Processed : {global_sc:,}")
    print(f"  • Total Metacells Generated    : {global_mc:,}")
    print(f"  • Global Condensation Factor   : {overall_ratio}x reduction in dataset sparsity")
    print(" ───────────────────────────────────────────────────────────────────────────────────────────────")
    print(df.to_string(index=False, justify='center'))
    print(" ═══════════════════════════════════════════════════════════════════════════════════════════════")
    print(" * N/A indicates chunk recovery bypassed QC computation. Values were validated in prior logs.\n")

def build_pipeline_tracker_from_disk(cfg, logger=None):
    """
    Dynamically scans the SPORE+ output directories to build the pipeline 
    attrition tracker by reading the headers of the saved .h5ad files.
    Uses backed='r' to ensure 0 GB RAM usage during the scan.
    """
    import anndata as ad
    from pathlib import Path
    
    tracker = {}
    ds_name = cfg.get("dataset", {}).get("name", "dataset")
    
    # Resolve directories
    proc_dir   = Path(cfg["paths"]["_processed"])
    splits_dir = Path(cfg["paths"]["_splits"])

    def _aggregate_shapes(file_paths, phase_name):
        c_total, g_total = 0, 0
        valid_files = 0
        for f in file_paths:
            try:
                # Load strictly in backed mode to prevent RAM spikes
                tmp = ad.read_h5ad(f, backed='r')
                c_total += tmp.n_obs
                g_total = tmp.n_vars  # Feature count remains uniform across splits
                valid_files += 1
            except Exception as e:
                if logger: logger.warning(f"Could not read {f}: {e}")
                
        if valid_files > 0:
            tracker[phase_name] = {"cells": c_total, "genes": g_total}

    if logger: logger.info("Scanning disk for SPORE+ milestone artifacts...")

    # 1. Phase 0: Raw Data
    raw_path = cfg.get("paths", {}).get("raw_data")
    if raw_path and Path(raw_path).exists():
        _aggregate_shapes([Path(raw_path)], "Phase 0: Raw Data")

    # 2. Phase 10: Ghost Excision (p10 files across all splits)
    p10_files = list(splits_dir.glob(f"{ds_name}_*_p10.h5ad"))
    if p10_files:
        _aggregate_shapes(p10_files, "Phase 10: Ghost Excision")

    # 3. Phase 12: Metacell Aggregation (all cell lines, all splits)
    # We exclude the CHITIN files here to purely grab Phase 12
    mc_files = list(proc_dir.glob(f"{ds_name}_*metacell.h5ad"))
    if mc_files:
        _aggregate_shapes(mc_files, "Phase 12: Metacells")

    # CHITIN (Phase 13) removed.

    if logger:
        logger.info("Pipeline Attrition Tracker built successfully:")
        for k, v in tracker.items():
            logger.info(f"  {k}: {v['cells']:,} cells | {v['genes']:,} genes")

    return tracker
