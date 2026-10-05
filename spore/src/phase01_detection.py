"""
SPORE+ · src/phase1_detection.py
──────────────────────────────────
Phase 1: Detection

Runs BEFORE any destructive processing. Inspects the raw .h5ad and detects:
  1. Data modality (scRNA-seq | snRNA-seq | CITE-seq | Multiome)
  2. Gene ID format (HGNC symbol | Ensembl | Entrez) + harmonization
  3. Perturbation structure (single | combinatorial)
  4. Cell line column (if labels already exist in .obs)

Detection results are stored in cfg['_detection'] and propagate
downstream — no manual config editing required for standard datasets.

CRITICAL: This phase is READ-ONLY. It never modifies adata.X.
Gene ID translation modifies adata.var_names and adata.var only.
"""

import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from .utils import log_phase_header, snapshot, log_memory, force_gc


# ═══════════════════════════════════════════════════════════════════════════
#  MODALITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _detect_modality(adata, cfg, logger):
    """
    Detect data modality from .var 'feature_types', var_names patterns,
    .obsm keys, and MT gene presence.

    Returns: modality string and notes dict
    """
    p1 = cfg.get("phase1_detection", {})
    notes = {}

    # --- CITE-seq detection ---
    # 10x Genomics stores 'feature_types' in .var with values
    # 'Gene Expression' and 'Antibody Capture'
    if "feature_types" in adata.var.columns:
        ftypes = adata.var["feature_types"].unique().tolist()
        if "Antibody Capture" in ftypes:
            notes["feature_types_found"] = ftypes
            logger.info(f"  Modality: CITE-seq detected (feature_types: {ftypes})")
            return "cite_seq", notes

    # --- Multiome / ATAC detection ---
    # 10x Genomics Multiome stores peaks with chr:start-end format
    peak_pattern = re.compile(r"^chr\w+:\d+-\d+$")
    n_sample = min(50, adata.n_vars)
    sample_genes = list(adata.var_names[:n_sample])
    n_peaks = sum(1 for g in sample_genes if peak_pattern.match(str(g)))
    if n_peaks / max(n_sample, 1) > 0.3:
        logger.info("  Modality: Multiome (RNA + ATAC) detected via peak var_names")
        notes["peak_fraction"] = n_peaks / n_sample
        return "multiome", notes

    # Also check obsm for ATAC signals
    if hasattr(adata, "obsm"):
        atac_keys = [k for k in adata.obsm.keys()
                     if "atac" in k.lower() or "peak" in k.lower()]
        if atac_keys:
            logger.info(f"  Modality: Multiome detected via obsm keys: {atac_keys}")
            return "multiome", notes

    # --- snRNA-seq detection ---
    # snRNA-seq has NO mitochondrial signal (mitochondria excluded from nuclear prep)
    # Heuristic: if top 500 genes contain 0 MT-/mt- genes AND
    # total_counts per cell are lower than typical scRNA-seq
    organism = cfg.get("dataset", {}).get("organism", "human")
    mt_prefix = "MT-" if organism == "human" else "mt-"
    n_mt = sum(1 for g in adata.var_names[:500] if g.startswith(mt_prefix))
    if n_mt == 0:
        # Could be snRNA-seq OR just a dataset with MT genes already removed
        # Check uns for assay annotations
        assay_hint = str(adata.uns.get("assay", "")).lower()
        if "nucleus" in assay_hint or "snrna" in assay_hint or "snuc" in assay_hint:
            logger.info("  Modality: snRNA-seq detected via uns['assay']")
            notes["mt_genes_absent"] = True
            notes["assay_hint"] = assay_hint
            return "snrna_seq", notes
        else:
            notes["mt_genes_absent"] = True
            logger.info(
                f"  Modality: No MT genes found in first 500 var_names. "
                f"May be snRNA-seq or pre-filtered. Defaulting to scRNA-seq. "
                f"Set modality manually in config if needed.")

    # --- Default ---
    logger.info("  Modality: scRNA-seq (default)")
    return "scrna_seq", notes


# ═══════════════════════════════════════════════════════════════════════════
#  GENE ID FORMAT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

_ENSEMBL_PREFIXES = {
    "human":     "ENSG",
    "mouse":     "ENSMUSG",
    "zebrafish": "ENSDARG",
    "fly":       "FBgn",
    "worm":      "WBGene",
}


def _detect_gene_id_format(adata, cfg, logger):
    """
    Detect whether var_names are HGNC symbols, Ensembl IDs, or Entrez IDs.

    CRITICAL LESSON from SPORE development:
    The Replogle K562 dataset has Ensembl IDs in var_names but gene symbols
    in perturbation labels → escaper filtering silently skips ALL targets.
    This detection prevents that silent failure.
    """
    p1  = cfg.get("phase1_detection", {})
    cfg_fmt = cfg.get("dataset", {}).get("gene_id_format", "auto")

    if cfg_fmt != "auto":
        logger.info(f"  Gene ID format: {cfg_fmt} (from config)")
        return cfg_fmt

    organism = cfg.get("dataset", {}).get("organism", "human").lower()
    sample   = list(adata.var_names[:200])

    # Check Ensembl prefixes
    ensembl_prefixes = p1.get("ensembl_prefixes", _ENSEMBL_PREFIXES)
    ens_prefix = ensembl_prefixes.get(organism, "ENSG")
    n_ensembl  = sum(1 for g in sample
                     if str(g).startswith(ens_prefix) and g[len(ens_prefix):].isdigit())
    ensembl_frac = n_ensembl / max(len(sample), 1)

    if ensembl_frac > 0.5:
        logger.info(
            f"  Gene ID format: Ensembl ({ens_prefix}...) detected "
            f"({ensembl_frac*100:.0f}% of first 200 var_names)")
        return "ensembl"

    # Check Entrez (all numeric)
    n_entrez = sum(1 for g in sample if str(g).isdigit())
    entrez_frac = n_entrez / max(len(sample), 1)
    if entrez_frac > 0.5:
        logger.info(
            f"  Gene ID format: Entrez (numeric) detected "
            f"({entrez_frac*100:.0f}% of first 200 var_names)")
        return "entrez"

    # Default: HGNC symbol
    logger.info("  Gene ID format: HGNC symbol (default)")
    return "symbol"


def _harmonize_gene_ids(adata, gene_id_format, cfg, logger):
    """
    Translate Ensembl or Entrez IDs → HGNC symbols in var_names.

    Preserves original IDs in adata.var['original_gene_id'].
    Uses mygene.py for online lookup OR a local mapping file if provided.

    CRITICAL: var_names translation only — never touches adata.X.
    """
    if gene_id_format == "symbol":
        logger.info("  Gene ID harmonization: already symbols, skipping")
        return adata

    mapping_file = cfg.get("dataset", {}).get("gene_id_mapping_file")
    organism     = cfg.get("dataset", {}).get("organism", "human").lower()
    species_map  = {"human": "human", "mouse": "mouse",
                    "zebrafish": "zebrafish", "fly": "fruitfly"}
    species      = species_map.get(organism, "human")

    # Preserve original IDs
    adata.var["original_gene_id"] = adata.var_names.tolist()

    # exp_009.1 rebuild: prefer an in-file symbol column (offline, deterministic, no network) BEFORE
    # falling back to the online mygene query. Many Perturb-seq h5ads (Replogle RPE1/K562) carry the
    # HGNC symbol in a var column (gene_name/gene_symbol/feature_name/symbol) alongside Ensembl var_names.
    # Querying mygene for thousands of genes online is slow + can silently no-op on a network failure
    # (Defect 5: "worked only because translation no-ops"), so use the local map whenever it exists.
    symbol_col = None
    for _cand in ("gene_name", "gene_symbol", "gene_symbols", "feature_name", "symbol", "hgnc_symbol"):
        if _cand in adata.var.columns:
            symbol_col = _cand
            break

    if mapping_file and Path(mapping_file).exists():
        logger.info(f"  Gene ID harmonization: loading mapping from {mapping_file}")
        mapping_df = pd.read_csv(mapping_file)
        # Expect columns: 'gene_id' (Ensembl/Entrez), 'symbol' (HGNC)
        mapping = dict(zip(
            mapping_df["gene_id"].astype(str),
            mapping_df["symbol"].astype(str)))
    elif symbol_col is not None:
        logger.info(
            f"  Gene ID harmonization: using in-file var['{symbol_col}'] as the symbol map "
            f"(offline; no mygene query)")
        mapping = {}
        for orig_id, sym in zip(adata.var["original_gene_id"], adata.var[symbol_col].astype(str)):
            if sym and sym.lower() not in ("nan", "none", ""):
                mapping[str(orig_id)] = str(sym)
        n_mapped = len(mapping)
        logger.info(
            f"  Gene ID harmonization: var['{symbol_col}'] provides {n_mapped:,}/"
            f"{len(adata.var_names):,} symbol mappings")
    else:
        logger.info(
            f"  Gene ID harmonization: querying mygene.py "
            f"({len(adata.var_names):,} genes, organism={species})")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            id_type = "ensembl.gene" if gene_id_format == "ensembl" else "entrezgene"
            results = mg.querymany(
                list(adata.var_names),
                scopes=id_type,
                fields="symbol",
                species=species,
                as_dataframe=True,
                verbose=False)
            mapping = {}
            if "symbol" in results.columns:
                for query_id, row in results.iterrows():
                    sym = row.get("symbol")
                    if pd.notna(sym):
                        mapping[str(query_id)] = str(sym)
        except ImportError:
            logger.warning(
                "  ⚠ mygene not installed. "
                "Run: pip install mygene --break-system-packages\n"
                "  Gene IDs will remain as-is for now.")
            return adata
        except Exception as e:
            logger.warning(f"  ⚠ mygene query failed: {e}. Gene IDs unchanged.")
            return adata

    # Apply mapping; unmapped genes keep their original ID
    old_names    = list(adata.var_names)
    new_names    = [mapping.get(str(g), str(g)) for g in old_names]
    n_translated = sum(1 for o, n in zip(old_names, new_names) if o != n)

    # Handle duplicates after translation by appending suffix
    seen = {}
    deduped = []
    for name in new_names:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    adata.var_names = deduped
    adata.var_names_make_unique()
    logger.info(
        f"  Gene ID harmonization: {n_translated:,}/{len(old_names):,} genes "
        f"translated to HGNC symbols")

    return adata


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINATORIAL PERTURBATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _detect_combinatorial(adata, cfg, logger):
    """
    Detect whether perturbation labels contain compound identifiers
    (e.g. "GENE_A+GENE_B"), indicating a high-MOI combinatorial screen.

    Returns: (is_combinatorial: bool, separator: str or None,
              constituent_genes: set)
    """
    p1         = cfg.get("phase1_detection", {})
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    seps       = p1.get("combinatorial_separators", ["+", ",", "|", ";"])
    min_frac   = p1.get("combinatorial_min_fraction", 0.05)

    if pert_col not in adata.obs.columns:
        logger.warning(f"  Combinatorial detection: '{pert_col}' not in .obs")
        return False, None, set()

    labels      = adata.obs[pert_col].values
    non_ctrl    = [str(l) for l in labels if str(l) != ctrl_label]
    n_non_ctrl  = max(len(non_ctrl), 1)

    for sep in seps:
        n_compound = sum(1 for l in non_ctrl if sep in l)
        frac = n_compound / n_non_ctrl
        if frac >= min_frac:
            # Found compound labels — extract all constituent genes
            all_genes = set()
            for l in non_ctrl:
                if sep in l:
                    parts = [p.strip() for p in l.split(sep) if p.strip()]
                    all_genes.update(parts)
                else:
                    all_genes.add(l)
            logger.info(
                f"  Combinatorial perturbations detected: "
                f"separator='{sep}', {n_compound:,} compound labels "
                f"({frac*100:.1f}% of non-control), "
                f"{len(all_genes):,} constituent genes")
            return True, sep, all_genes

    logger.info("  Perturbation structure: single-guide (no compound labels found)")
    return False, None, set()


# ═══════════════════════════════════════════════════════════════════════════
#  CELL LINE COLUMN DETECTION (Tier 1)
# ═══════════════════════════════════════════════════════════════════════════

# Known cell line names for Tier 1 bonus annotation
# This list is supplementary — detection doesn't NEED these names.
# It works purely on column structure.
_KNOWN_CELL_LINES = {
    "k562", "rpe1", "hesc", "h1", "jurkat", "ipsc", "hela", "a549",
    "293t", "hek293", "thp1", "mcf7", "u2os", "hct116", "pc9", "h358",
    "h1299", "h1792", "h2009", "lncap", "du145", "pc3", "hl60", "molt4",
    "ramos", "raji", "daudi", "hcc1954", "bt474", "skbr3", "t47d", "mbda231",
}

_CELL_LINE_COL_VOCAB = [
    "cell_line", "cell_type", "cellline", "cell_line_id", "line",
    "sample_type", "condition_cell", "clone", "subline", "tissue",
    "cell_line_name", "cell_origin", "sample_origin",
]


def _detect_cell_line_column(adata, cfg, logger):
    """
    Tier 1: check if .obs already has a cell line label column.

    Uses structure-based detection (cardinality, dtype, value length)
    rather than vocabulary matching — works for unknown/novel cell line names.

    Returns: (cell_line_col: str or None, unique_cell_lines: list)
    """
    p1  = cfg.get("phase1_detection", {})
    cfg_col = cfg.get("phase11_cell_line", {}).get("cell_line_col")

    # User-specified column takes priority
    if cfg_col and cfg_col in adata.obs.columns:
        unique = sorted(adata.obs[cfg_col].unique().tolist())
        logger.info(
            f"  Cell line column: user-specified '{cfg_col}', "
            f"{len(unique)} unique values: {unique[:8]}")
        return cfg_col, unique

    min_u   = p1.get("cell_line_col_min_unique",    2)
    max_u   = p1.get("cell_line_col_max_unique",    20)
    max_len = p1.get("cell_line_col_max_label_length", 40)
    pert_col = cfg.get("dataset", {}).get("perturbation_col", "gene")
    batch_col = cfg.get("dataset", {}).get("batch_col", "gem_group")

    candidates = []
    for col in adata.obs.columns:
        # Skip perturbation and batch columns — those are already handled
        if col in (pert_col, batch_col):
            continue
        # Must be string or categorical
        if str(adata.obs[col].dtype) not in ("object", "category"):
            continue
        n_uniq = adata.obs[col].nunique()
        if not (min_u <= n_uniq <= max_u):
            continue

        # Check value lengths — cell line labels are typically short
        sample_vals = adata.obs[col].dropna().astype(str).unique()[:20]
        if any(len(v) > max_len for v in sample_vals):
            continue
        # Must not look like perturbation labels (gene symbols)
        gene_like = sum(
            1 for v in sample_vals
            if re.match(r"^[A-Z][A-Z0-9\-]{1,15}$", v))
        if gene_like / max(len(sample_vals), 1) > 0.6 and n_uniq > 5:
            continue  # Looks like a gene column

        # Score: vocab match is a bonus, not a requirement
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        vocab_score = sum(
            1 for v in _CELL_LINE_COL_VOCAB if v in col_lower)
        # Check if any values are known cell line names (bonus)
        known_score = sum(
            1 for v in sample_vals
            if v.lower().replace("-", "").replace("_", "").replace(" ", "")
            in _KNOWN_CELL_LINES)
        total_score = vocab_score * 2 + known_score
        candidates.append((col, n_uniq, total_score, sample_vals.tolist()))

    if not candidates:
        logger.info("  Cell line column: not found in .obs")
        return None, []

    # Sort by score descending
    candidates.sort(key=lambda x: -x[2])
    best_col, n_uniq, score, sample_vals = candidates[0]

    unique = sorted(adata.obs[best_col].unique().tolist())
    logger.info(
        f"  Cell line column: '{best_col}' detected "
        f"(score={score}, {n_uniq} unique values: {unique[:8]}) "
        f"{'[KNOWN NAMES]' if score > 2 else '[STRUCTURE ONLY]'}")

    return best_col, unique


# ═══════════════════════════════════════════════════════════════════════════
#  CITE-SEQ MODALITY HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def _handle_cite_seq(adata, logger):
    import scipy.sparse as sp

    if "feature_types" not in adata.var.columns:
        return adata

    rna_mask = adata.var["feature_types"] == "Gene Expression"
    adt_mask = adata.var["feature_types"] == "Antibody Capture"
    n_rna = rna_mask.sum()
    n_adt = adt_mask.sum()

    if n_adt == 0:
        return adata

    logger.info(f"  CITE-seq: separating {n_rna:,} RNA features from {n_adt:,} ADT features")

    # Park ADT in obsm natively as sparse to prevent RAM explosion
    X_adt = adata.X[:, adt_mask]
    if sp.issparse(X_adt) and not sp.isspmatrix_csr(X_adt):
        X_adt = X_adt.tocsr()
    
    adata.obsm["X_adt"] = X_adt
    adata.uns["adt_var_names"] = list(adata.var_names[adt_mask])
    logger.info(f"  CITE-seq: ADT matrix parked natively in obsm['X_adt']")

    from .utils import safe_in_memory_gene_subset
    adata = safe_in_memory_gene_subset(adata, keep_mask=rna_mask, logger=logger)
    return adata

# ═══════════════════════════════════════════════════════════════════════════
#  MULTIOME HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def _handle_multiome(adata, logger):
    import re
    import scipy.sparse as sp

    peak_pattern = re.compile(r"^chr\w+:\d+-\d+$")
    rna_mask  = np.array([not peak_pattern.match(str(g)) for g in adata.var_names])
    atac_mask = ~rna_mask

    if atac_mask.sum() == 0:
        return adata

    logger.info(f"  Multiome: separating {rna_mask.sum():,} RNA features from {atac_mask.sum():,} ATAC peaks")

    # Park ATAC in obsm natively as sparse to prevent OOM crash
    X_atac = adata.X[:, atac_mask]
    if sp.issparse(X_atac) and not sp.isspmatrix_csr(X_atac):
        X_atac = X_atac.tocsr()

    adata.obsm["X_atac_peaks"] = X_atac
    adata.uns["atac_var_names"] = list(adata.var_names[atac_mask])
    logger.info("  Multiome: ATAC peaks parked natively in obsm['X_atac_peaks']")

    from .utils import safe_in_memory_gene_subset
    adata = safe_in_memory_gene_subset(adata, keep_mask=rna_mask, logger=logger)
    return adata


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 1
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1(adata, cfg: dict, logger):
    """
    Run all detection steps. Stores results in cfg['_detection'] and
    modifies adata.var_names if gene ID translation is needed.

    Returns: (adata, detection_results_dict)
    """
    log_phase_header(logger, 1, "Detection")
    p1 = cfg.get("phase1_detection", {})

    if not p1.get("enabled", True):
        logger.info("  Phase 1 Detection: DISABLED in config")
        cfg["_detection"] = {"skipped": True}
        return adata, cfg["_detection"]

    results = {}
    log_memory(logger, "Phase 1 start")

    # ── 1a. Modality detection ─────────────────────────────────────────────
    if p1.get("modality_detection", True):
        modality, mod_notes = _detect_modality(adata, cfg, logger)
        results["modality"] = modality
        results["modality_notes"] = mod_notes

        # Adapt Phase 2 cell triage for snRNA-seq
        if modality == "snrna_seq":
            logger.info(
                "  ⚠ snRNA-seq mode: MT% filter will be disabled in Phase 2")
            cfg.setdefault("_detection_overrides", {})
            cfg["_detection_overrides"]["disable_mt_filter"] = True

        # Handle CITE-seq: park ADT, keep RNA
        if modality == "cite_seq":
            adata = _handle_cite_seq(adata, logger)
            results["adt_parked"] = True

        # Handle Multiome: park ATAC peaks, keep RNA
        if modality == "multiome":
            adata = _handle_multiome(adata, logger)
            results["atac_parked"] = True

    # ── 1b. Gene ID format detection & harmonization ───────────────────────
    if p1.get("gene_id_harmonization", True):
        gene_id_format = _detect_gene_id_format(adata, cfg, logger)
        results["gene_id_format"] = gene_id_format

        # Update config so downstream phases know the format
        cfg["dataset"]["gene_id_format"] = gene_id_format

        if gene_id_format != "symbol":
            logger.info(
                f"  Gene IDs are {gene_id_format} — translating to HGNC symbols...")
            adata = _harmonize_gene_ids(adata, gene_id_format, cfg, logger)
            results["gene_ids_harmonized"] = True
        else:
            results["gene_ids_harmonized"] = False

    # ── 1c. Combinatorial perturbation detection ───────────────────────────
    if p1.get("combinatorial_detection", True):
        is_comb, sep, constituents = _detect_combinatorial(adata, cfg, logger)
        results["is_combinatorial"] = is_comb
        results["combinatorial_separator"] = sep
        results["combinatorial_constituents"] = constituents

        if is_comb:
            cfg["dataset"]["perturbation_structure"] = "combinatorial"
            cfg["dataset"]["perturbation_separator"] = sep
            cfg["_detection_overrides"] = cfg.get("_detection_overrides", {})
            cfg["_detection_overrides"]["combinatorial_constituents"] = constituents
        else:
            cfg["dataset"]["perturbation_structure"] = "single"

    # ── 1d. Cell line column detection (Tier 1) ────────────────────────────
    if p1.get("cell_line_detection", True):
        cell_line_col, unique_lines = _detect_cell_line_column(adata, cfg, logger)
        results["detected_cell_line_col"] = cell_line_col
        results["detected_cell_lines"] = unique_lines
        results["n_cell_lines_labeled"] = len(unique_lines)

        if cell_line_col:
            # Propagate to Phase 11 config
            cfg["phase11_cell_line"]["cell_line_col"] = cell_line_col
            logger.info(
                f"  → Phase 11 will use existing labels: "
                f"{unique_lines[:6]}{'...' if len(unique_lines)>6 else ''}")
        else:
            logger.info(
                "  → Phase 11 will attempt automatic detection "
                "(Tier 2/3 clustering)")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("═" * 65)
    logger.info("  Phase 1 Detection Summary:")
    logger.info(f"    Modality           : {results.get('modality', 'unknown')}")
    logger.info(f"    Gene ID format     : {results.get('gene_id_format', 'unknown')}")
    logger.info(f"    IDs harmonized     : {results.get('gene_ids_harmonized', False)}")
    logger.info(f"    Perturbation type  : {cfg['dataset'].get('perturbation_structure', 'single')}")
    cl_col = results.get('detected_cell_line_col', None)
    n_cl   = results.get('n_cell_lines_labeled', 0)
    logger.info(f"    Cell line column   : {cl_col or 'NOT FOUND'} ({n_cl} lines)")
    logger.info("═" * 65)

    cfg["_detection"] = results
    snapshot(adata, "Post Phase 1 Detection", logger)
    log_memory(logger, "Phase 1 end")
    return adata, results


from pathlib import Path
