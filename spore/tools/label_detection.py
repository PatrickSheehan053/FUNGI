"""
SPORE+ · label_detection.py
──────────────────────────────
Standalone detection script. Run this BEFORE opening the notebook
to auto-populate key fields in sporeplus_config.yaml.

Usage:
    python label_detection.py /path/to/your/file.h5ad

What it does:
    1. Scans .obs columns and reports the best perturbation, control,
       batch, and cell line column candidates.
    2. Detects gene ID format (symbol / Ensembl / Entrez).
    3. Detects data modality (scRNA-seq / snRNA-seq / CITE-seq / Multiome).
    4. Detects combinatorial perturbation labels.
    5. Prints a pre-filled YAML config block that can be pasted directly
       into sporeplus_config.yaml.

IMPORTANT: This script only READS the file — never modifies it.
"""

import sys
import re
import numpy as np
import anndata as ad
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  KNOWN VOCABULARIES
# ─────────────────────────────────────────────────────────────────────────────

# Vocabulary for perturbation column detection
_PERT_COL_VOCAB = [
    "gene", "perturbation", "target", "guide", "knockout",
    "crispr", "sgrna", "guide_rna", "perturb", "perturbagen",
    "gene_target", "target_gene", "gene_name", "symbol",
    "pert", "pert_name", "gene_id", "guide_id", "perturbation_target",
    "CRISPR", "guide_assignment", "gene_symbol",
]

# Vocabulary for control label detection
_CTRL_VOCAB = [
    "non-targeting", "non_targeting", "nontargeting",
    "control", "ctrl", "scramble", "scrambled", "negative",
    "NT", "ntc", "gfp", "lacz", "safe_harbor", "safe-harbor",
    "AAVS1", "INTERGENIC", "non targeting", "nontarget",
    "neg_ctrl", "neg_control", "CTRL", "NEGATIVE", "GFP",
]

# Vocabulary for batch/gem group columns
_BATCH_COL_VOCAB = [
    "gem_group", "batch", "run", "lane", "library",
    "experiment", "sample", "replicate", "10x_run",
    "batch_id", "run_id", "sample_id", "library_id",
    "sequencing_run", "prep_batch", "gem",
]

# Vocabulary for cell line columns  
_CELL_LINE_COL_VOCAB = [
    "cell_line", "cell_type", "cellline", "cell_line_id",
    "line", "sample_type", "cell_line_name", "condition",
    "clone", "subline", "tissue", "cell_origin",
]

# Ensembl prefixes per organism
_ENSEMBL_PREFIXES = {
    "ENSG":     "human",
    "ENSMUSG":  "mouse",
    "ENSDARG":  "zebrafish",
    "FBgn":     "fly",
    "WBGene":   "worm",
}


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_gene_id_format(adata):
    """Detect var_names format: symbol | ensembl | entrez"""
    sample = list(adata.var_names[:200])
    for prefix, organism in _ENSEMBL_PREFIXES.items():
        n = sum(1 for g in sample
                if str(g).startswith(prefix) and g[len(prefix):].isdigit())
        if n / max(len(sample), 1) > 0.5:
            return "ensembl", organism
    n_entrez = sum(1 for g in sample if str(g).isdigit())
    if n_entrez / max(len(sample), 1) > 0.5:
        return "entrez", "unknown"
    return "symbol", "unknown"


def detect_modality(adata):
    """Detect: scrna_seq | snrna_seq | cite_seq | multiome"""
    if "feature_types" in adata.var.columns:
        ftypes = adata.var["feature_types"].unique().tolist()
        if "Antibody Capture" in ftypes:
            return "cite_seq"

    peak_pat  = re.compile(r"^chr\w+:\d+-\d+$")
    n_sample  = min(50, adata.n_vars)
    n_peaks   = sum(1 for g in list(adata.var_names[:n_sample])
                    if peak_pat.match(str(g)))
    if n_peaks / max(n_sample, 1) > 0.3:
        return "multiome"

    n_mt = sum(1 for g in adata.var_names[:500]
               if g.startswith("MT-") or g.startswith("mt-"))
    if n_mt == 0:
        assay = str(adata.uns.get("assay", "")).lower()
        if any(kw in assay for kw in ["nucleus", "snrna", "snuc"]):
            return "snrna_seq"
        return "scrna_seq_or_snrna_seq"  # ambiguous

    return "scrna_seq"


def detect_perturbation_col(adata):
    """Find the perturbation label column in .obs."""
    obs_cols = list(adata.obs.columns)
    scored   = []
    for col in obs_cols:
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        vocab_hit = sum(1 for v in _PERT_COL_VOCAB
                        if v.lower() in col_lower)
        if vocab_hit == 0:
            continue
        # Must be string/category
        if str(adata.obs[col].dtype) not in ("object", "category"):
            continue
        n_uniq = adata.obs[col].nunique()
        if n_uniq < 2 or n_uniq > 50000:
            continue
        scored.append((col, vocab_hit, n_uniq))

    if not scored:
        return None, []
    scored.sort(key=lambda x: -x[1])
    best_col = scored[0][0]
    unique   = sorted(adata.obs[best_col].dropna().unique().tolist())
    return best_col, unique


def detect_control_label(adata, pert_col):
    """Find the control label within a perturbation column."""
    if pert_col is None or pert_col not in adata.obs.columns:
        return None
    unique = adata.obs[pert_col].dropna().unique().tolist()
    for ctrl in _CTRL_VOCAB:
        for label in unique:
            if str(label).lower().replace(" ", "").replace("_", "").replace("-", "") == \
               ctrl.lower().replace(" ", "").replace("_", "").replace("-", ""):
                return str(label)
    return None


def detect_batch_col(adata):
    """Find the batch/gem_group column in .obs."""
    obs_cols = list(adata.obs.columns)
    scored   = []
    for col in obs_cols:
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        vocab_hit = sum(1 for v in _BATCH_COL_VOCAB
                        if v.lower() in col_lower)
        if vocab_hit == 0:
            continue
        n_uniq = adata.obs[col].nunique()
        if n_uniq < 2 or n_uniq > 10000:
            continue
        scored.append((col, vocab_hit, n_uniq))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


def detect_cell_line_col(adata, pert_col, batch_col):
    """Find an existing cell line label column in .obs."""
    obs_cols = list(adata.obs.columns)
    candidates = []
    for col in obs_cols:
        if col in (pert_col, batch_col):
            continue
        if str(adata.obs[col].dtype) not in ("object", "category"):
            continue
        n_uniq = adata.obs[col].nunique()
        if not (2 <= n_uniq <= 20):
            continue
        sample_vals = adata.obs[col].dropna().astype(str).unique()[:20]
        if any(len(v) > 40 for v in sample_vals):
            continue

        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        vocab_hit = sum(1 for v in _CELL_LINE_COL_VOCAB
                        if v.lower() in col_lower)
        candidates.append((col, vocab_hit, n_uniq,
                           sorted(adata.obs[col].unique().tolist())[:6]))

    if not candidates:
        return None, []
    candidates.sort(key=lambda x: -x[1])
    best_col, _, _, unique = candidates[0]
    return best_col, sorted(adata.obs[best_col].unique().tolist())


def detect_combinatorial(adata, pert_col, ctrl_label):
    """Detect combinatorial perturbation labels."""
    if pert_col is None:
        return False, None
    labels = [str(l) for l in adata.obs[pert_col].dropna().unique()
              if str(l) != (ctrl_label or "")]
    seps = ["+", ",", "|", ";"]
    for sep in seps:
        n_comb = sum(1 for l in labels if sep in l)
        if n_comb / max(len(labels), 1) >= 0.05:
            return True, sep
    return False, None


def detect_perturbation_type(adata, pert_col, ctrl_label):
    """
    Attempt to detect CRISPRi / CRISPRa / CRISPRko from .uns or .obs metadata.
    """
    # Check uns
    for key in ["perturbation_type", "crispr_type", "assay", "perturbation"]:
        val = str(adata.uns.get(key, "")).lower()
        if "crispra" in val or "activation" in val:
            return "CRISPRa"
        if "crispri" in val or "interference" in val or "inhibition" in val:
            return "CRISPRi"
        if "ko" in val or "knockout" in val:
            return "CRISPRko"

    # Check .obs columns
    for col in adata.obs.columns:
        if "type" in col.lower() or "mode" in col.lower():
            vals = adata.obs[col].dropna().unique()
            for v in vals:
                vl = str(v).lower()
                if "crispra" in vl or "activation" in vl:
                    return "CRISPRa"
                if "crispri" in vl or "interference" in vl:
                    return "CRISPRi"
                if "knockout" in vl or "crispko" in vl:
                    return "CRISPRko"

    return "CRISPRi"  # most common default


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(h5ad_path: str):
    print(f"\n{'='*65}")
    print(f"  SPORE+ label_detection.py")
    print(f"  Scanning: {h5ad_path}")
    print(f"{'='*65}\n")

    print("  Loading .h5ad (backed mode to avoid RAM spike)...")
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
    except Exception as e:
        print(f"  ERROR: Could not load file: {e}")
        sys.exit(1)

    n_obs  = adata.n_obs
    n_vars = adata.n_vars
    print(f"  Dataset shape: {n_obs:,} cells × {n_vars:,} genes")
    print(f"  .obs columns ({len(adata.obs.columns)}): {list(adata.obs.columns)[:15]}")
    print(f"  .var columns ({len(adata.var.columns)}): {list(adata.var.columns)[:10]}")
    if adata.uns:
        print(f"  .uns keys: {list(adata.uns.keys())[:10]}")

    print("\n  ── Running detections ──\n")

    gene_fmt, organism_hint = detect_gene_id_format(adata)
    print(f"  Gene ID format    : {gene_fmt}"
          + (f"  (organism hint: {organism_hint})" if organism_hint != "unknown" else ""))

    modality = detect_modality(adata)
    print(f"  Modality          : {modality}")

    pert_col, pert_unique = detect_perturbation_col(adata)
    print(f"  Perturbation col  : {pert_col}  ({len(pert_unique)} unique values)")
    if pert_unique:
        print(f"    Sample values   : {pert_unique[:8]}")

    ctrl_label = detect_control_label(adata, pert_col)
    print(f"  Control label     : {ctrl_label}")

    batch_col = detect_batch_col(adata)
    print(f"  Batch column      : {batch_col}")

    cl_col, cl_unique = detect_cell_line_col(adata, pert_col, batch_col)
    print(f"  Cell line col     : {cl_col}"
          + (f"  → {cl_unique[:6]}" if cl_unique else ""))

    is_comb, comb_sep = detect_combinatorial(adata, pert_col, ctrl_label)
    print(f"  Combinatorial     : {is_comb}"
          + (f"  separator='{comb_sep}'" if is_comb else ""))

    pert_type = detect_perturbation_type(adata, pert_col, ctrl_label)
    print(f"  Perturbation type : {pert_type}")

    adata.file.close()

    # ── Print suggested config block ─────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUGGESTED YAML CONFIG (paste into sporeplus_config.yaml):")
    print(f"{'='*65}\n")
    print("dataset:")
    print(f'  name:              ""  # fill in your dataset name')
    print(f'  organism:          "{organism_hint if organism_hint != "unknown" else "human"}"')
    print(f'  cell_line:         ""')
    print(f'  perturbation_type: "{pert_type}"')
    print(f'  perturbation_col:  "{pert_col or "gene"}"')
    print(f'  control_label:     "{ctrl_label or "non-targeting"}"')
    print(f'  batch_col:         "{batch_col or "gem_group"}"')
    print(f'  perturbation_structure: "{"combinatorial" if is_comb else "single"}"')
    if is_comb:
        print(f'  perturbation_separator: "{comb_sep}"')
    print(f'  gene_id_format: "{gene_fmt}"')
    print("")
    print("phase11_cell_line:")
    if cl_col:
        print(f'  cell_line_col: "{cl_col}"')
    else:
        print(f'  cell_line_col: null  # no existing cell line column found')
        print(f'  expected_n_cell_lines: null  # set if you know the number')
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_detection.py /path/to/file.h5ad")
        sys.exit(1)
    run_detection(sys.argv[1])
