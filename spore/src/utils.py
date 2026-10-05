"""
SPORE+ · src/utils.py
─────────────────────
Config loading, logging, memory monitoring, and shared helpers.
Merged from SPORE utils.py and CHITIN utils.py.

18JUNE patch: _sniff_var_format, load_protected_genes
"""

import re
import json
import yaml
import logging
import os
import gc
import psutil
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "sporeplus_config.yaml") -> dict:
    """Load and validate the SPORE+ YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"].get("project_root") or ".")
    cfg["paths"]["_root"]            = root
    raw_h5ad = cfg["paths"].get("raw_h5ad") or ""
    cfg["paths"]["_raw_h5ad"]        = root / raw_h5ad if raw_h5ad else None
    cfg["paths"]["_processed"]       = root / cfg["paths"]["processed_dir"]
    cfg["paths"]["_splits"]          = root / cfg["paths"]["splits_dir"]
    cfg["paths"]["_figures"]         = root / cfg["paths"]["figures_dir"]
    cfg["paths"]["_logs"]            = root / cfg["paths"]["log_dir"]
    # CHITIN (Phase 13) removed — no chitin_output_dir.

    for key in ["_processed", "_splits", "_figures", "_logs"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logger(cfg: dict, name: str = "SPORE+") -> logging.Logger:
    """Configure SPORE+ logger with console + file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cfg["paths"]["_logs"] / f"sporeplus_run_{timestamp}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s"))
    logger.addHandler(fh)

    logger.info(f"SPORE+ log initialized → {log_path}")
    return logger


def log_phase_header(logger: logging.Logger, phase, title: str):
    """Print a clean phase header to the log and console."""
    bar = "═" * 65
    logger.info(bar)
    logger.info(f"  PHASE {phase} · {title}")
    logger.info(bar)


# alias used by CHITIN sub-components
def log_phase(logger, title: str):
    bar = "═" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def get_memory_usage() -> str:
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    if mem_bytes >= 1e9:
        return f"{mem_bytes / 1e9:.1f} GB"
    return f"{mem_bytes / 1e6:.0f} MB"


def log_memory(logger, label: str = ""):
    mem = get_memory_usage()
    tag = f" ({label})" if label else ""
    logger.info(f"  💾 Memory{tag}: {mem}")


def force_gc(logger=None):
    collected = gc.collect()
    if logger:
        mem = get_memory_usage()
        logger.info(f"  🗑️  GC collected {collected} objects  |  RAM: {mem}")


# ═══════════════════════════════════════════════════════════════════════════
#  SNAPSHOT / PROGRESS
# ═══════════════════════════════════════════════════════════════════════════

def snapshot(adata, label: str, logger):
    """Log shape snapshot with memory and sparsity info."""
    n_cells, n_genes = adata.shape
    mem = get_memory_usage()
    sparse_tag = ""
    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        total = n_cells * n_genes
        sparsity = (1 - nnz / total) * 100 if total > 0 else 0
        sparse_tag = f"  (sparse, {sparsity:.1f}% zeros)"
    logger.info(
        f"  [{label}] → {n_cells:,} cells  ×  {n_genes:,} genes"
        f"{sparse_tag}  |  RAM: {mem}")


# ═══════════════════════════════════════════════════════════════════════════
#  SPARSE SAFETY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def ensure_sparse(adata, logger=None):
    """
    Convert adata.X to CSR sparse if it isn't already.
    CRITICAL: Never skip this — a dense 2M-cell matrix can be 80+ GB.
    """
    if not sp.issparse(adata.X):
        if logger:
            n_cells, n_genes = adata.shape
            dense_gb = (n_cells * n_genes * 4) / 1e9
            logger.info(
                f"  ⚡ Converting dense → CSR sparse "
                f"(dense would be ~{dense_gb:.1f} GB)")
        adata.X = sp.csr_matrix(adata.X)
        force_gc(logger)
        if logger:
            nnz = adata.X.nnz
            sparse_gb = (nnz * 12) / 1e9
            logger.info(f"  ⚡ Sparse: {nnz:,} non-zeros (~{sparse_gb:.1f} GB)")
    return adata


def safe_subset(adata, cell_mask=None, gene_mask=None, logger=None):
    """
    Memory-safe subsetting of AnnData.

    CRITICAL LESSON (Error 001, 004, 006):
    adata[mask] always triggers a 2x RAM spike. SciPy's C++ backend
    cannot subset a sparse matrix without allocating a full duplicate.
    This function bypasses Python's memory allocator entirely.
    """
    import anndata as ad

    if cell_mask is None:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    if gene_mask is None:
        gene_mask = np.ones(adata.n_vars, dtype=bool)

    if cell_mask.all() and gene_mask.all():
        if logger:
            logger.info("  No filtering required. Bypassing slice to save RAM.")
        return adata

    cell_idx = np.where(cell_mask)[0]
    gene_idx  = np.where(gene_mask)[0]

    if logger:
        logger.info(
            f"  Subsetting: {cell_idx.shape[0]:,} cells × {gene_idx.shape[0]:,} genes")

    X_new = adata.X[cell_idx, :][:, gene_idx]
    if sp.issparse(X_new):
        X_new = X_new.tocsr()

    obs_new = adata.obs.iloc[cell_idx].copy()
    var_new = adata.var.iloc[gene_idx].copy()

    layers = {}
    if adata.layers:
        for lname, ldata in adata.layers.items():
            lsub = ldata[cell_idx, :][:, gene_idx]
            if sp.issparse(lsub):
                lsub = lsub.tocsr()
            layers[lname] = lsub

    uns = adata.uns.copy() if hasattr(adata, "uns") and adata.uns else {}
    adata_new = ad.AnnData(
        X=X_new, obs=obs_new, var=var_new, layers=layers, uns=uns)
    return adata_new


# ═══════════════════════════════════════════════════════════════════════════
#  CELL CYCLE GENES (Tirosh 2016)
# ═══════════════════════════════════════════════════════════════════════════

def get_cell_cycle_genes():
    """
    Tirosh et al. 2016 S-phase and G2M-phase gene lists.
    Used by Phase 6 Ghost Rescue, Phase 8 Ghost Rescue, Phase 10 regression.
    """
    s_genes = [
        "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2",
        "MCM6","CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2",
        "NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3",
        "MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1",
        "BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
    ]
    g2m_genes = [
        "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80",
        "CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A",
        "SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E",
        "TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK",
        "CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8",
        "ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5",
        "CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
    ]
    return s_genes, g2m_genes


# ═══════════════════════════════════════════════════════════════════════════
#  PROTECTED GENE ID HARMONIZATION  (18JUNE patch)
# ═══════════════════════════════════════════════════════════════════════════

def _sniff_var_format(adata) -> str:
    """Return 'ensembl' if var_names look like ENSG IDs, else 'symbol'."""
    sample = list(adata.var_names[:10])
    n_ens = sum(1 for g in sample if re.match(r'^ENSG\d{11}$', str(g)))
    return 'ensembl' if n_ens >= 5 else 'symbol'


def load_protected_genes(protected_genes_path, adata, logger=None):
    """
    Load a protected-genes JSON and return gene IDs in the format of adata.var_names.

    Handles symbol→Ensembl translation using adata.var['gene_name'] as the
    authoritative map (no external API calls). Includes None/missing-file guard.
    Deduplicates preserving tier order (tier 1 genes come first).

    Returns (list_of_gene_ids, report_dict).
    report_dict keys: n_requested, n_resolved, n_missing, format, sniffed_format
    """
    report = {
        "n_requested": 0, "n_resolved": 0, "n_missing": 0,
        "format": "symbol", "sniffed_format": "symbol",
    }

    if protected_genes_path is None:
        return [], report

    path = Path(protected_genes_path)
    if not path.exists():
        if logger:
            logger.warning(f"  Protected genes file not found: {path} — skipping")
        return [], report

    with open(path) as f:
        data = json.load(f)

    # Collect genes in tier order, deduplicate
    by_tier = {}
    for cat_name, cat in data.get("protected_genes", {}).items():
        tier = cat.get("tier", 3)
        by_tier.setdefault(tier, []).extend(cat.get("genes", []))

    raw_symbols = []
    seen = set()
    for tier in sorted(by_tier):
        for g in by_tier[tier]:
            if g not in seen:
                raw_symbols.append(g)
                seen.add(g)

    report["n_requested"] = len(raw_symbols)

    declared_ds_format = data.get("metadata", {}).get("dataset_gene_id_format", "symbol")
    sniffed_format     = _sniff_var_format(adata)
    report["format"]         = declared_ds_format
    report["sniffed_format"] = sniffed_format

    if sniffed_format == 'symbol':
        # Direct symbol lookup — Phase 1 already translated Ensembl → symbols
        var_set  = set(adata.var_names)
        resolved = [g for g in raw_symbols if g in var_set]
        report["n_resolved"] = len(resolved)
        report["n_missing"]  = len(raw_symbols) - len(resolved)
        if logger:
            logger.info(
                f"  Protected genes: {len(raw_symbols)} unique "
                f"(json_format=symbol, declared_ds_format={declared_ds_format})")
            logger.info(
                f"  adata.var_names sniffed as: symbol    "
                f"← Phase 1 already translated to symbols")
            logger.info(
                f"  TF rescue: {len(resolved)} genes resolved (direct symbol match), "
                f"{len(raw_symbols)-len(resolved)} not in matrix")
        return resolved, report

    else:
        # var_names are still Ensembl IDs — translate via adata.var['gene_name']
        if 'gene_name' not in adata.var.columns:
            if logger:
                logger.warning(
                    "  Protected genes: var_names are Ensembl but 'gene_name' "
                    "column not in adata.var — cannot translate. Returning empty list.")
            return [], report

        sym_to_ens = {}
        for ens_id, row in adata.var.iterrows():
            sym = str(row.get('gene_name', ''))
            if sym and sym not in sym_to_ens:
                sym_to_ens[sym] = ens_id

        resolved = []
        for sym in raw_symbols:
            if sym in sym_to_ens:
                resolved.append(sym_to_ens[sym])

        report["n_resolved"] = len(resolved)
        report["n_missing"]  = len(raw_symbols) - len(resolved)
        if logger:
            logger.info(
                f"  Protected genes: {len(raw_symbols)} unique "
                f"(json_format=symbol, declared_ds_format={declared_ds_format})")
            logger.info(
                f"  adata.var_names sniffed as: ensembl")
            logger.info(
                f"  TF rescue: {len(resolved)} resolved via gene_name map, "
                f"{len(raw_symbols)-len(resolved)} not in matrix")
        return resolved, report


# ═══════════════════════════════════════════════════════════════════════════
#  IN-MEMORY C-BUFFER ROW SUBSET (Error 006 / 007 fix)
# ═══════════════════════════════════════════════════════════════════════════

def safe_in_memory_row_subset(adata, keep_mask, logger):
    """
    Destructively filters ROWS of a CSR matrix IN-PLACE using C-buffer shift.

    CRITICAL LESSON (Errors 006, 007):
    SciPy's C++ backend cannot row-subset without a full duplicate.
    This function slides the underlying data/indices/indptr arrays in-place,
    then constructs a fresh AnnData wrapper — zero extra RAM overhead.
    """
    import anndata as ad

    if keep_mask.all():
        if logger:
            logger.info("  No row filtering required. Bypassing C-buffer mutation.")
        return adata

    n_kept = keep_mask.sum()

    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr  = adata.X.indptr
        indices = adata.X.indices
        data    = adata.X.data

        new_indptr = np.zeros(n_kept + 1, dtype=indptr.dtype)

        padded = np.concatenate(([False], keep_mask, [False]))
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
        if logger:
            logger.warning("  Matrix is not CSR. Falling back to default slicing.")
        new_X = adata.X[keep_mask]

    new_obs  = adata.obs.iloc[keep_mask].copy()
    new_var  = adata.var.copy()
    new_obsm = {k: v[keep_mask] for k, v in adata.obsm.items()} \
               if hasattr(adata, "obsm") else {}
    new_varm = {k: v.copy() for k, v in adata.varm.items()} \
               if hasattr(adata, "varm") else {}
    new_uns  = adata.uns.copy() if hasattr(adata, "uns") else {}

    del adata
    force_gc(logger)

    return ad.AnnData(
        X=new_X, obs=new_obs, var=new_var,
        obsm=new_obsm, varm=new_varm, uns=new_uns)


# ═══════════════════════════════════════════════════════════════════════════
#  IN-MEMORY C-BUFFER GENE (COLUMN) SUBSET (Error 010 / 008 fix)
# ═══════════════════════════════════════════════════════════════════════════

def safe_in_memory_gene_subset(adata, keep_mask, logger):
    """
    Destructively filters COLUMNS (genes) of a CSR matrix IN-PLACE.

    CRITICAL LESSON (Errors 008, 010):
    Column-slicing on CSR is even worse than row-slicing — SciPy must
    reconstruct all row pointers across n_obs rows. The vectorized
    column-remapping allocates O(nnz) boolean arrays → 30-40 GB spikes.
    This row-by-row C-buffer shift has O(1) memory overhead.
    """
    import anndata as ad

    if keep_mask.all():
        return adata

    n_kept_genes = keep_mask.sum()

    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr  = adata.X.indptr
        indices = adata.X.indices
        data    = adata.X.data

        col_mapping = np.full(adata.X.shape[1], -1, dtype=np.int32)
        col_mapping[keep_mask] = np.arange(n_kept_genes, dtype=np.int32)

        new_indptr = np.zeros_like(indptr)
        write_ptr  = 0

        for i in range(len(indptr) - 1):
            start, end      = indptr[i], indptr[i + 1]
            new_indptr[i]   = write_ptr

            if start < end:
                r_ind  = indices[start:end]
                r_dat  = data[start:end]
                m_cols = col_mapping[r_ind]
                mask   = m_cols >= 0
                nnz_row = mask.sum()

                if nnz_row > 0:
                    indices[write_ptr:write_ptr + nnz_row] = m_cols[mask]
                    data[write_ptr:write_ptr + nnz_row]    = r_dat[mask]
                write_ptr += nnz_row

        new_indptr[-1] = write_ptr

        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr),
            shape=(adata.n_obs, n_kept_genes))
    else:
        if logger:
            logger.warning("  Matrix is not CSR. Falling back to default slicing.")
        new_X = adata.X[:, keep_mask]

    new_var  = adata.var.iloc[keep_mask].copy()
    new_obs  = adata.obs.copy()
    new_obsm = {k: v.copy() for k, v in adata.obsm.items()} \
               if hasattr(adata, "obsm") else {}
    new_varm = {k: v[keep_mask] for k, v in adata.varm.items()} \
               if hasattr(adata, "varm") else {}
    new_uns  = adata.uns.copy() if hasattr(adata, "uns") else {}

    del adata
    force_gc(logger)

    return ad.AnnData(
        X=new_X, obs=new_obs, var=new_var,
        obsm=new_obsm, varm=new_varm, uns=new_uns)


# ═══════════════════════════════════════════════════════════════════════════
#  DARK / LIGHT THEME (shared by all plotting modules)
# ═══════════════════════════════════════════════════════════════════════════

_DARK = {
    "bg": "#0D1117", "panel": "#161B22", "text": "#E6EDF3",
    "grid": "#21262D", "accent": "#58A6FF", "warn": "#F85149",
    "good": "#3FB950", "muted": "#8B949E", "highlight": "#D2A8FF",
    "ctrl": "#79C0FF", "pert": "#F0883E", "delta": "#3FB950",
}

_LIGHT = {
    "bg": "#FFFFFF", "panel": "#F6F8FA", "text": "#1F2328",
    "grid": "#D1D9E0", "accent": "#0969DA", "warn": "#CF222E",
    "good": "#1A7F37", "muted": "#656D76", "highlight": "#8250DF",
    "ctrl": "#0550AE", "pert": "#BC4C00", "delta": "#1A7F37",
}

_CELL_LINE_COLORS = [
    "#58A6FF", "#F0883E", "#3FB950", "#D2A8FF",
    "#F85149", "#79C0FF", "#E3B341", "#BC4C00",
    "#8B949E", "#A5D6FF", "#56D364", "#FFA657",
]


def get_theme(cfg: dict) -> dict:
    return _DARK if cfg["plotting"]["style"] == "dark" else _LIGHT


def get_cell_line_color(idx: int) -> str:
    return _CELL_LINE_COLORS[idx % len(_CELL_LINE_COLORS)]


def apply_sporeplus_style(cfg: dict):
    """Apply the SPORE+ matplotlib global style."""
    theme = get_theme(cfg)
    plt.rcParams.update({
        "figure.facecolor":  theme["bg"],
        "axes.facecolor":    theme["panel"],
        "axes.edgecolor":    theme["grid"],
        "axes.labelcolor":   theme["text"],
        "text.color":        theme["text"],
        "xtick.color":       theme["text"],
        "ytick.color":       theme["text"],
        "grid.color":        theme["grid"],
        "grid.alpha":        0.5,
        "figure.dpi":        cfg["plotting"]["dpi"],
        "font.family":       "monospace",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "legend.facecolor":  theme["panel"],
        "legend.edgecolor":  theme["grid"],
        "savefig.facecolor": theme["bg"],
        "savefig.bbox":      "tight",
        "savefig.dpi":       cfg["plotting"]["dpi"],
    })


def save_fig(fig, cfg: dict, filename: str):
    if cfg["plotting"]["save_figures"]:
        fmt  = cfg["plotting"]["figure_format"]
        path = cfg["paths"]["_figures"] / f"{filename}.{fmt}"
        fig.savefig(path, facecolor=fig.get_facecolor())
        return path
    return None


def format_ax(ax, theme: dict, title: str,
              xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])