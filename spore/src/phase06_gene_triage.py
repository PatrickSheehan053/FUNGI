"""
SPORE+ · src/phase06_gene_triage.py
────────────────────────────────────
Phase 6: Gene-Level Triage
Adapted from SPORE phase3_gene_triage.py.
Config key: phase6_gene_triage (was phase3_gene_triage).

KEY FIX: combinatorial gene rescue — compound labels like "GENE_A+GENE_B"
are parsed to extract individual gene names before rescue.
"""

import numpy as np
import anndata as ad
import scipy.sparse as sp
from collections import OrderedDict
from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    safe_in_memory_gene_subset, get_cell_cycle_genes)


def _get_perturbation_targets(adata, cfg: dict) -> set:
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    raw_labels = set(adata.obs[pert_col].unique()) - {ctrl_label}

    # COMBINATORIAL FIX: parse compound labels to get individual gene names
    is_comb = cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial"
    sep     = cfg.get("dataset", {}).get("perturbation_separator", "+")

    targets = set()
    for lbl in raw_labels:
        if is_comb and sep and sep in str(lbl):
            for part in str(lbl).split(sep):
                targets.add(part.strip())
        else:
            targets.add(str(lbl))
    return targets


def filter_genes(adata, cfg: dict, logger):
    log_phase_header(logger, 6, "Gene-Level Triage")
    p6       = cfg.get("phase6_gene_triage", {})
    waterfall = OrderedDict()
    waterfall["Starting genes"] = adata.n_vars

    X = adata.X
    log_memory(logger, "Before feature metric calculation")

    # ── OOM FIREWALL: Chunked Metric Calculation ──
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    is_large = getattr(adata, 'isbacked', False) or n_cells > 1000000

    if is_large:
        logger.info("  [Gene Triage] Large/Backed mode: computing gene metrics in safe chunks...")
        cells_per_gene = np.zeros(n_genes, dtype=np.int64)
        sum_expr = np.zeros(n_genes, dtype=np.float64)
        chunk_size = 50000
        
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = X[start:end]
            
            if sp.issparse(chunk):
                cells_per_gene += chunk.getnnz(axis=0)
                sum_expr += np.asarray(chunk.sum(axis=0)).flatten()
            else:
                cells_per_gene += (chunk > 0).sum(axis=0)
                sum_expr += np.asarray(chunk.sum(axis=0)).flatten()
            del chunk # Free memory instantly
            
        mean_expr = (sum_expr / n_cells).astype(np.float32)
        del sum_expr
    else:
        # Fast path for small, fully in-memory datasets
        if sp.issparse(X):
            cells_per_gene = X.getnnz(axis=0)
            mean_expr = np.array(X.sum(axis=0)).flatten() / n_cells
        else:
            cells_per_gene = (X > 0).sum(axis=0)
            mean_expr      = X.mean(axis=0)

    keep_genes = np.ones(adata.n_vars, dtype=bool)

    # ── 1. Ambient RNA filter ──────────────────────────────────────────────
    min_cells     = p6.get("min_cells_expressing", 10)
    ambient_fail  = cells_per_gene < min_cells
    n_ambient     = ambient_fail.sum()
    keep_genes   &= ~ambient_fail
    logger.info(
        f"  Ambient RNA filter (≥{min_cells} cells): "
        f"{n_ambient:,} genes flagged")
    waterfall["After ambient filter"] = int(keep_genes.sum())

    # ── 2. Scale-adaptive filter (REFINED) ─────────────────────────────────
    n_cells = adata.n_obs
    
    # Standard: Keep genes expressed in at least 1% of cells 
    # (If your YAML passes 0.05, it will use that, but 0.01 is recommended for 400k+ cells)
    pct_thresh = p6.get("pct_filter", 0.01) 
    min_cells_adaptive = max(min_cells, int(n_cells * pct_thresh))
    
    # We combine both: Gene must pass a basic count floor OR a mean UMI floor
    umi_thresh = p6.get("mean_umi_threshold", 0.05) 
    
    # A gene FAILS only if it fails BOTH consistency (cells_per_gene) 
    # AND intensity (mean_expr)
    adaptive_fail = (cells_per_gene < min_cells_adaptive) & (mean_expr < umi_thresh)
    
    logger.info(
        f"  Scale-adaptive: Keep genes in >{pct_thresh*100:.1f}% cells "
        f"({min_cells_adaptive:,} cells) OR mean UMI > {umi_thresh}")

    # ── THE FIX: Declare the missing variable and update the waterfall ──
    would_remove = ~keep_genes | adaptive_fail
    waterfall["After adaptive (pre-rescue)"] = int((~would_remove).sum())

    # ── 3. Perturbation target rescue ──────────────────────────────────────
    rescued = []
    if p6.get("rescue_perturbation_targets", True):
        targets = _get_perturbation_targets(adata, cfg)
        var_names = set(adata.var_names)
        targets_in_features = targets & var_names
        gene_name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        targets_to_rescue = {
            t for t in targets_in_features
            if would_remove[gene_name_to_idx[t]]}
        if targets_to_rescue:
            rescued = sorted(targets_to_rescue)
            for t in targets_to_rescue:
                would_remove[gene_name_to_idx[t]] = False
            logger.info(
                f"  ⚡ Target rescue: {len(rescued)} perturbation targets saved")

    # ── 4. Cell Cycle Ghost Rescue (Error 033 fix) ─────────────────────────
    # Cell cycle genes have low variance and get wiped by the adaptive filter.
    # They MUST survive to Phase 8 (HVG ghost rescue) and Phase 10 (regression).
    cc_s, cc_g2m = get_cell_cycle_genes()
    cc_genes = set(cc_s + cc_g2m)
    var_upper_map = {v.upper(): i for i, v in enumerate(adata.var_names)}

    ghosts_rescued = []
    for g in cc_genes:
        if g in var_upper_map:
            idx = var_upper_map[g]
            if would_remove[idx]:
                would_remove[idx] = False
                ghosts_rescued.append(adata.var_names[idx])

    if ghosts_rescued:
        logger.info(
            f"  👻 Ghost rescue: {len(ghosts_rescued)} cell cycle genes "
            f"shielded from sparsity cut")

    final_keep = ~would_remove
    waterfall["After adaptive (post-rescue)"] = int(final_keep.sum())

    # CRITICAL (Error 010): row-by-row C-buffer column reconstructor
    adata_new = safe_in_memory_gene_subset(adata, keep_mask=final_keep, logger=logger)

    # ── THE FIX: Permanently attach the pre-filter stats to the metadata ──
    # OOM FIREWALL: Re-use the safely chunked `cells_per_gene` array we already
    # built at the top of the file instead of pulling the whole matrix again
    orig_pct_cells = (cells_per_gene / adata.n_obs) * 100
        
    adata_new.uns["phase6_waterfall"] = dict(waterfall)
    adata_new.uns["phase6_gene_penetrance"] = np.float32(orig_pct_cells)
    # ──────────────────────────────────────────────────────────────────────

    snapshot(adata_new, "Post Phase 6", logger)
    return adata_new, waterfall, rescued


def run_phase6(adata, cfg: dict, logger):
    return filter_genes(adata, cfg, logger)
