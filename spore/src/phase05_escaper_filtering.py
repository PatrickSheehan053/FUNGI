"""
SPORE+ · src/phase05_escaper_filtering.py
──────────────────────────────────────────
Phase 5: Escaper Filtering & Knockdown Efficiency Scoring

Engineered for 2M+ cell scCRISPR-seq datasets (e.g., Replogle 2022). 
In CRISPR screens, "escapers" are cells that received a guide but did not 
experience the intended biological perturbation. This phase filters them out 
by comparing target gene expression against the non-targeting control distribution.

Sparse Memory Architecture
──────────────────────────
To prevent OOM crashes on massive matrices, control cells are temporarily 
cached into a CSC (Compressed Sparse Column) matrix. This allows O(1) column 
lookups for target genes. Expression arrays are flattened to 1D dense vectors 
one gene at a time, keeping the memory overhead functionally zero.

Biological Directionality
─────────────────────────
  • CRISPRi / Knockout: Retains cells ≤ Nth percentile of controls.
  • CRISPRa / Activation: Retains cells ≥ (100-N)th percentile of controls.
    (Includes a 1e-5 zero-inflation safeguard for highly sparse genes).

Combinatorial multiplexing (e.g., "GENE_A+GENE_B") is natively supported.
Efficiency metrics are logged to `adata.uns["knockdown_efficiency"]` and 
added as per-cell observations.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import gc

from .utils import (log_phase_header, snapshot, log_memory,
                    force_gc, safe_in_memory_row_subset)


def _get_escaper_direction(cfg: dict, logger) -> str:
    p5_dir    = cfg.get("phase5_escaper_filtering", {}).get("direction", "auto")
    pert_type = cfg.get("dataset", {}).get("perturbation_type", "CRISPRi")
    if p5_dir == "auto":
        direction = "activation" if "CRISPRa" in str(pert_type) else "knockdown"
    else:
        direction = "activation" if p5_dir == "activation" else "knockdown"
    logger.info(f"  Direction: {direction} (perturbation_type={pert_type})")
    return direction


def _parse_target_genes(label: str, cfg: dict) -> list:
    if cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial":
        sep = cfg.get("dataset", {}).get("perturbation_separator", "+")
        if sep and sep in str(label):
            return [p.strip() for p in str(label).split(sep) if p.strip()]
    return [str(label)]


def filter_escapers(adata, cfg: dict, logger):
    log_phase_header(logger, 5, "Escaper Filtering + Efficiency Scoring")
    p5         = cfg.get("phase5_escaper_filtering", {})
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    percentile = p5.get("escaper_percentile", 10)
    direction  = _get_escaper_direction(cfg, logger)
    eff_thresh = p5.get("efficiency_threshold", 0.50)

    pert_values  = adata.obs[pert_col].values
    ctrl_mask    = pert_values == ctrl_label
    ctrl_indices = np.where(ctrl_mask)[0]
    n_ctrl       = len(ctrl_indices)
    logger.info(f"  Control cells ('{ctrl_label}'): {n_ctrl:,}")

    perturbations = [p for p in np.unique(pert_values) if p != ctrl_label]
    logger.info(f"  Perturbation targets: {len(perturbations):,}")

    # ── ALIAS MAPPING FIX: Map var_names AND all var metadata (e.g., old Ensembl IDs) ──
    gene_to_idx = {}
    
    # 1. Map the active column names (Gene Symbols)
    for i, g in enumerate(adata.var_names):
        gene_to_idx[str(g)] = i
        
    # 2. Scan the metadata columns for the original Ensembl IDs and map them to the same index
    for col in adata.var.columns:
        for i, val in enumerate(adata.var[col]):
            if pd.notna(val):
                gene_to_idx[str(val)] = i

    # ── OOM FIREWALL 1: Micro-Chunked Control Caching ──
    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000
    
    if is_large:
        logger.info("  [Escaper] Large/Backed mode: caching control cells in micro-chunks for safe I/O...")
        ctrl_chunks = []
        chunk_size = 50000
        for i in range(0, len(ctrl_indices), chunk_size):
            idx_chunk = ctrl_indices[i:i+chunk_size]
            c = adata.X[idx_chunk]
            if not sp.issparse(c):
                c = sp.csr_matrix(c)
            ctrl_chunks.append(c)
        ctrl_X_csc = sp.vstack(ctrl_chunks).tocsc()
        del ctrl_chunks
    else:
        ctrl_X_csc = adata.X[ctrl_indices, :].tocsc()
        
    log_memory(logger, "after control CSC cache")

    cells_to_keep   = list(ctrl_indices)
    escaper_records = []
    efficiency_records = {}   # pert → {efficiency, knockdown_depth, low_efficiency}
    n_skipped = 0

    for target in perturbations:
        pert_indices = np.where(pert_values == target)[0]
        n_total      = len(pert_indices)
        constituents = _parse_target_genes(target, cfg)
        genes_in_mat = [g for g in constituents if g in gene_to_idx]

        if not genes_in_mat:
            escaper_records.append({
                "perturbation": target, "n_total": n_total,
                "n_escaped": 0, "n_kept": n_total,
                "status": "bypassed_no_gene_in_features"})
            cells_to_keep.extend(pert_indices)
            efficiency_records[target] = {
                "efficiency_score": 1.0, "knockdown_depth": float("nan"),
                "low_efficiency": False, "n_genes_checked": 0}
            n_skipped += 1
            continue

        pass_mask = np.ones(len(pert_indices), dtype=bool)
        knockdown_depths = []

        # ── OOM FIREWALL 2: Safe Perturbation Block Extraction ──
        if is_large:
            pert_chunks = []
            chunk_size = 50000
            for i in range(0, len(pert_indices), chunk_size):
                idx_chunk = pert_indices[i:i+chunk_size]
                c = adata.X[idx_chunk]
                if not sp.issparse(c):
                    c = sp.csr_matrix(c)
                pert_chunks.append(c)
            pert_X_csc = sp.vstack(pert_chunks).tocsc()
            del pert_chunks
        else:
            pert_X_csc = adata.X[pert_indices, :]
            if not sp.issparse(pert_X_csc):
                pert_X_csc = sp.csr_matrix(pert_X_csc)
            pert_X_csc = pert_X_csc.tocsc()

        pass_mask = np.ones(len(pert_indices), dtype=bool)
        knockdown_depths = []

        for gene in genes_in_mat:
            gene_idx  = gene_to_idx[gene]
            ctrl_expr = ctrl_X_csc[:, gene_idx].toarray().flatten()
            
            # Instantly slice the column from the RAM-cached CSC matrix
            pert_expr = pert_X_csc[:, gene_idx].toarray().flatten()

            # ── FIX 1: Calculate TRUE Knockdown Depth on ALL cells (No Survivor Bias) ──
            ctrl_mean = ctrl_expr.mean()
            if ctrl_mean > 1e-10:
                # This represents the actual biological efficiency of the guide
                true_fold_change = pert_expr.mean() / ctrl_mean
                knockdown_depths.append(true_fold_change)

            # ── FIX 2: Apply Escaper Filter safely ──
            if direction == "knockdown":
                threshold = np.percentile(ctrl_expr, percentile)
                gene_pass = pert_expr <= threshold
            else:
                threshold = np.percentile(ctrl_expr, 100 - percentile)
                # Prevent the zero-inflation bug for sparse genes in CRISPRa
                if threshold == 0:
                    threshold = 1e-5 
                gene_pass = pert_expr >= threshold

            pass_mask &= gene_pass

        kept      = pert_indices[pass_mask]
        n_escaped = n_total - len(kept)
        efficiency = len(kept) / max(n_total, 1)
        kd_depth   = float(np.mean(knockdown_depths)) if knockdown_depths else float("nan")

        cells_to_keep.extend(kept)
        escaper_records.append({
            "perturbation": target, "n_total": n_total,
            "n_escaped": n_escaped, "n_kept": len(kept),
            "n_genes_checked": len(genes_in_mat),
            "status": "filtered"})
        efficiency_records[target] = {
            "efficiency_score":  efficiency,
            "knockdown_depth":   kd_depth,
            "low_efficiency":    efficiency < eff_thresh}

        del ctrl_expr, pert_expr, pass_mask, pert_X_csc

    del ctrl_X_csc
    gc.collect()

    # Build per-cell efficiency columns
    eff_scores = np.ones(adata.n_obs, dtype=np.float32)
    kd_depths  = np.full(adata.n_obs, float("nan"), dtype=np.float32)
    for i, pv in enumerate(pert_values):
        if pv in efficiency_records:
            eff_scores[i] = efficiency_records[pv]["efficiency_score"]
            kd_val = efficiency_records[pv]["knockdown_depth"]
            if not np.isnan(kd_val):
                kd_depths[i] = kd_val

    escaper_stats = pd.DataFrame(escaper_records)
    total_escaped = escaper_stats["n_escaped"].sum()
    n_low_eff     = sum(1 for r in efficiency_records.values() if r["low_efficiency"])
    logger.info(f"  Escapers removed: {total_escaped:,}  |  {n_skipped:,} bypassed (gene not in features)")
    logger.info(f"  Efficiency: {n_low_eff:,} perturbations below {eff_thresh*100:.0f}% threshold")

    keep_mask = np.zeros(adata.n_obs, dtype=bool)
    keep_mask[cells_to_keep] = True

    adata_new = safe_in_memory_row_subset(adata, keep_mask, logger)

    # Store efficiency metadata
    adata_new.obs["escaper_efficiency"]     = eff_scores[keep_mask]
    adata_new.obs["escaper_knockdown_depth"] = kd_depths[keep_mask]
    adata_new.uns["knockdown_efficiency"]   = {
        k: {kk: (float(vv) if isinstance(vv, (float, np.floating))
                 else bool(vv) if isinstance(vv, (bool, np.bool_))
                 else vv)
            for kk, vv in v.items()}
        for k, v in efficiency_records.items()}

    snapshot(adata_new, "Post escaper filter", logger)
    return adata_new, escaper_stats


def filter_undersized_perturbations(adata, cfg: dict, logger):
    p5        = cfg.get("phase5_escaper_filtering", {})
    pert_col  = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    min_cells  = p5.get("min_cells_per_perturbation", 50)

    sizes      = adata.obs[pert_col].value_counts()
    undersized = sizes[(sizes < min_cells) & (sizes.index != ctrl_label)]
    if len(undersized) > 0:
        drop_labels = set(undersized.index)
        keep_mask   = ~adata.obs[pert_col].isin(drop_labels).values
        logger.info(f"  Perturbation triage (min {min_cells}): "
                    f"dropped {len(undersized):,} groups ({undersized.sum():,} cells)")
        adata = safe_in_memory_row_subset(adata, keep_mask, logger)
    else:
        logger.info(f"  Perturbation triage: all groups meet minimum ({min_cells})")

    final_sizes = adata.obs[pert_col].value_counts()
    snapshot(adata, "Post perturbation triage", logger)
    return adata, final_sizes


def run_phase5(adata, cfg: dict, logger):
    adata, escaper_stats = filter_escapers(adata, cfg, logger)
    adata, pert_sizes    = filter_undersized_perturbations(adata, cfg, logger)
    return adata, escaper_stats, pert_sizes
