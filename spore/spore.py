#!/usr/bin/env python3
"""
spore_light.py
──────────────
Laptop-scale SPORE pipeline runner for MYCELIUM/CHITIN investigations.

Runs phases 0-12 from SPORE_plus/src. CHITIN (Phase 13) is intentionally
excluded — apply it separately as a configurable downstream step with
multiple parameter variants (k, n_pcs, metric, correction_mode).

For datasets that exceed 64GB laptop RAM (K562-scale, >1M cells), use
--subset-first to run the memory-safe two-pass cell subsetter before
the pipeline.

Relationship to SPORE+_express.py:
  SPORE+_express.py : HPC batch runner, 32 CPUs, 512 GB, phases 0-13
  spore_light.py    : Laptop runner, 8 CPUs, 64 GB, phases 0-12 only

Usage:
  # Standard run (RPE1, VCC — fits in 64GB RAM):
  python spore_light.py --config spore_light_config_RPE1.yaml

  # Large dataset: subset to N perturbations first, then pipeline:
  python spore_light.py --config config.yaml --subset-first

  # Subsetter only (produces a subsetted h5ad, does not run the pipeline):
  python spore_light.py --config config.yaml --subset-only

  # Print estimated RAM and time for the configured dataset and exit:
  python spore_light.py --config config.yaml --estimate

EXIT CODES
  0  Success
  1  Config / argument error
  2  Pipeline failed
"""

import argparse
import gc
import logging
import os
import sys
import time
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psutil
import yaml

# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP
#  Must run before any scipy/sklearn/numpy import to prevent thread
#  over-subscription on the 8-core i7.
# ══════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("OMP_NUM_THREADS",          "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",          "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",   "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",      "1")
os.environ.setdefault("MALLOC_ARENA_MAX",         "2")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_",   "134217728")


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS  (laptop-tuned — mirrors express.py but scaled for 8 cores)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict = {
    "runtime": {
        "n_jobs":                   6,
        "sparse_on_load":           True,
        "memory_monitor":           True,
        "large_dataset_mode":       "auto",
        "large_dataset_threshold":  500_000,  # cells above this use subprocess paths
        "checkpointing":            True,
    },
    "phase0_ingestion":  {"chunk_size": 50_000},
    "phase1_detection": {
        "enabled":                  True,
        "modality_detection":       False,
        "gene_id_harmonization":    True,
        "combinatorial_detection":  False,
        "cell_line_detection":      False,
        "ensembl_prefixes": {
            "human":     "ENSG",
            "mouse":     "ENSMUSG",
            "zebrafish": "ENSDARG",
        },
        "combinatorial_min_fraction": 0.05,
    },
    "phase2_cell_triage": {
        "mt_method":          "explicit_list",
        "mt_threshold":       0.20,
        "mt_genes": [
            "MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-ND4L","MT-ND5","MT-ND6",
            "MT-CO1","MT-CO2","MT-CO3","MT-ATP6","MT-ATP8","MT-CYB",
        ],
        "ribo_prefixes":       ["RPL", "RPS"],
        "min_genes_per_cell":  200,
        "max_genes_per_cell":  10000,
        "min_counts_per_cell": 500,
        "max_counts_per_cell": 80000,
    },
    "phase3_ambient":  {"enabled": False},
    "phase4_doublets": {"enabled": False},
    "phase5_escaper_filtering": {
        "efficiency_threshold":        0.50,
        "escaper_percentile":          10,
        "min_cells_per_perturbation":  50,
        "direction":                   "auto",
    },
    "phase6_gene_triage": {
        "min_cells_expressing":              10,
        "pct_filter":                        0.01,
        "mean_umi_threshold":                0.25,
        "rescue_perturbation_targets":       True,
        "rescue_combinatorial_constituents": True,
    },
    "phase7_splits": {
        "mode":          "zero_shot",
        "random_seed":   42,
        "test_mode":     False,
        "stratify_bins": 4,
        "test_n":        20,
        "val_n":         20,
        "test_ratio":    0.15,
        "val_ratio":     0.10,
    },
    "phase8_hvg": {
        "enabled":                    True,
        "n_top_genes":                1000,
        "method":                     "seurat_v3",
        "rescue_perturbation_targets": True,
        "compute_aware_budget":        True,
        "protected_gene_list":         None,
    },
    "phase9_normalization": {
        "target_sum":    10000,
        "log_transform": True,
        "method":        "lognorm",
    },
    "phase10_confounders": {
        "cell_cycle_regression": True,
        "s_genes":               "tirosh2016",
        "g2m_genes":             "tirosh2016",
        "batch_correction": {
            "enabled":                   True,
            "method":                    "harmony",
            "batch_key":                 "batch",
            "confounding_ari_threshold": 0.5,
            "min_cells_per_batch":       50,
        },
        "imputation_prohibited": True,
        "fit_on_train_only":     True,
    },
    "phase11_cell_line": {
        "enabled":               "false",
        "cell_line_col":         None,
        "expected_n_cell_lines": None,
        "auto_detect": {
            "enabled":            True,
            "max_k_to_try":       15,
            "n_bootstrap":        50,
            "bootstrap_frac":     0.80,
            "min_stability_ari":  0.85,
            "min_bic_improvement": 50.0,
        },
        "min_cells_per_cell_line":    500,
        "output_single_cell_splits":  False,
    },
    "phase12_metacell": {
        "enabled":                     True,
        "target_cells_per_metacell":   10,
        "min_metacells_per_pert":      5,
        "min_cells_per_metacell":      3,
        "systema_calibration":         True,
        "compute_quality_metrics":     True,
        "inner_variance_warn_threshold": 2.0,
        "suffix":                      "_metacell",
    },
    "plotting": {
        "style":         "dark",
        "dpi":           150,
        "save_figures":  False,
        "figure_format": "png",
    },
    "output": {
        "cleanup_intermediates": False,
        "generate_report":       True,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_paths(cfg: dict) -> dict:
    root = Path(cfg["paths"].get("project_root") or ".")
    cfg["paths"]["_root"]          = root
    raw = cfg["paths"].get("raw_h5ad") or ""
    cfg["paths"]["_raw_h5ad"]      = root / raw if raw else None
    cfg["paths"]["_processed"]     = root / cfg["paths"]["processed_dir"]
    cfg["paths"]["_splits"]        = root / cfg["paths"]["splits_dir"]
    cfg["paths"]["_figures"]       = root / cfg["paths"].get("figures_dir",       "figures")
    cfg["paths"]["_logs"]          = root / cfg["paths"].get("log_dir",           "logs")
    # CHITIN (Phase 13) removed — no chitin_output_dir.
    for key in ["_processed", "_splits", "_figures", "_logs"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def _apply_light_overrides(cfg: dict) -> dict:
    # save_figures is controlled by the YAML, not forced off here.
    # Express mode forced it off (no display on HPC), but on the 2070
    # we want plots saved to the figures directory.
    cfg.setdefault("runtime", {}).setdefault("checkpointing", True)
    # Phase 13 section intentionally absent — CHITIN is a separate tool
    return cfg


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _apply_light_overrides(_resolve_paths(_deep_merge(_DEFAULTS, raw)))
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG-vs-DATA VALIDATION  (Defect 5 — fail at second zero, never silently)
# ══════════════════════════════════════════════════════════════════════════════

class ConfigDataMismatch(Exception):
    """Raised when the config contradicts the actual raw h5ad. Hard-fails the run."""


def _detect_var_id_format(var_names, organism: str = "human") -> str:
    """Return 'ensembl' | 'entrez' | 'symbol' from a sample of var_names."""
    prefixes = {"human": "ENSG", "mouse": "ENSMUSG", "zebrafish": "ENSDARG",
                "fly": "FBgn", "worm": "WBGene"}
    ens_prefix = prefixes.get(str(organism).lower(), "ENSG")
    sample = [str(g) for g in list(var_names[:200])]
    if not sample:
        return "symbol"
    n_ens = sum(1 for g in sample if g.startswith(ens_prefix) and g[len(ens_prefix):].split(".")[0].isdigit())
    if n_ens / len(sample) > 0.5:
        return "ensembl"
    n_entrez = sum(1 for g in sample if g.isdigit())
    if n_entrez / len(sample) > 0.5:
        return "entrez"
    return "symbol"


def validate_config_against_data(cfg: dict, logger) -> None:
    """
    Open the raw h5ad (backed, obs/var only — no matrix) and HARD-FAIL on any config/data mismatch
    BEFORE the pipeline burns compute. Checks (Defect 5):
      1. perturbation_col is a real obs column.
      2. control_label actually appears in that column.
      3. if batch_correction.enabled, its batch_key is a real obs column.
      4. gene_id_format matches the var_names (auto-detect; error on contradiction; fill in 'auto').
      5. test_n + val_n leave room for a train split (soft warn here on raw perts; hard-checked post-triage
         in phase07).
    A wrong config stops the run at second zero, not after an hour of silent degradation.
    """
    import anndata as ad

    raw_path = cfg["paths"].get("_raw_h5ad")
    if not raw_path or not Path(raw_path).exists():
        raise ConfigDataMismatch(
            f"raw_h5ad not found: {raw_path}. Set paths.raw_h5ad (relative to project_root).")

    ds        = cfg.get("dataset", {})
    pert_col  = ds.get("perturbation_col", "gene")
    ctrl      = ds.get("control_label", "non-targeting")
    organism  = ds.get("organism", "human")

    logger.info("═" * 65)
    logger.info("  CONFIG ↔ DATA VALIDATION")
    logger.info(f"  raw_h5ad : {raw_path}")
    A = ad.read_h5ad(str(raw_path), backed="r")
    obs_cols = list(A.obs.columns)

    # Capture the MEASURED-gene symbol set (every gene ever in the raw expression matrix) — used later to
    # compute the "perturbed-but-never-measured" ceiling in the coverage report/final gate. A held-out
    # target that is missing from the panel but also absent here is an unavoidable data ceiling, not a bug.
    measured = set()
    for _col in ("gene_name", "gene_symbol", "gene_symbols", "feature_name", "symbol", "hgnc_symbol"):
        if _col in A.var.columns:
            measured = set(map(str, A.var[_col].astype(str)))
            break
    if not measured:
        measured = set(map(str, A.var_names))
    cfg["_measured_symbols"] = measured

    errors = []

    # 1. perturbation column
    if pert_col not in obs_cols:
        errors.append(
            f"perturbation_col '{pert_col}' is NOT an obs column. Available obs columns: {obs_cols}")
        pert_vals = None
    else:
        pert_vals = set(map(str, A.obs[pert_col].unique()))
        logger.info(f"  ✓ perturbation_col '{pert_col}' present ({len(pert_vals):,} unique labels)")

    # 2. control label
    if pert_vals is not None:
        if str(ctrl) not in pert_vals:
            errors.append(
                f"control_label '{ctrl}' not found in obs['{pert_col}']. "
                f"Examples present: {sorted(list(pert_vals))[:8]}")
        else:
            logger.info(f"  ✓ control_label '{ctrl}' present in obs['{pert_col}']")

    # 3. batch key (only when batch correction is enabled)
    bc = cfg.get("phase10_confounders", {}).get("batch_correction", {})
    if bc.get("enabled", True):
        batch_key = bc.get("batch_key") or ds.get("batch_col") or "gem_group"
        if batch_key not in obs_cols:
            errors.append(
                f"batch_correction.enabled=true but batch_key '{batch_key}' is NOT an obs column "
                f"(Defect 5: this used to silently skip correction). Available: {obs_cols}. "
                f"Set phase10_confounders.batch_correction.batch_key to a real column, "
                f"or set batch_correction.enabled=false.")
        else:
            logger.info(f"  ✓ batch_key '{batch_key}' present ({A.obs[batch_key].nunique()} batches)")
    else:
        logger.info("  · batch_correction disabled — batch_key not validated")

    # 4. gene ID format vs var_names
    p1_harm = cfg.get("phase1_detection", {}).get("gene_id_harmonization", True)
    detected = _detect_var_id_format(A.var_names, organism)
    declared = ds.get("gene_id_format", "auto")
    logger.info(f"  · gene_id_format: declared='{declared}', detected='{detected}' (var_names look {detected})")
    if declared == "auto":
        cfg["dataset"]["gene_id_format"] = detected
        logger.info(f"  ✓ gene_id_format auto-set to '{detected}'")
    elif declared != detected:
        # A real contradiction: e.g. config says symbols but var_names are Ensembl → downstream symbol
        # lookups (protected list, held-out target coverage) would silently miss. Only tolerate the case
        # where harmonization is off and we simply proceed on whatever the data is.
        if p1_harm:
            errors.append(
                f"gene_id_format='{declared}' in config but var_names look like '{detected}'. "
                f"This silently breaks symbol-based rescue/coverage. Set gene_id_format: {detected} "
                f"(or 'auto'), or fix the data.")
        else:
            logger.warning(
                f"  ⚠ gene_id_format='{declared}' but var_names look '{detected}' "
                f"(harmonization disabled — proceeding, but symbol lookups may miss)")

    # 5. split sizes vs perturbation count (soft — surviving count only known post-triage)
    if pert_vals is not None:
        n_perts_raw = len(pert_vals - {str(ctrl)})
        zs = cfg.get("phase7_splits", {})
        test_n, val_n = zs.get("test_n"), zs.get("val_n")
        if test_n is not None and val_n is not None:
            if int(test_n) + int(val_n) >= n_perts_raw:
                errors.append(
                    f"test_n ({test_n}) + val_n ({val_n}) = {int(test_n)+int(val_n)} >= raw perturbation "
                    f"count ({n_perts_raw}); no perturbations left to train on. Lower test_n/val_n.")
            elif int(test_n) + int(val_n) > 0.6 * n_perts_raw:
                logger.warning(
                    f"  ⚠ test_n+val_n ({int(test_n)+int(val_n)}) is >60% of raw perts ({n_perts_raw}); "
                    f"the surviving count after triage will be lower — verify the split is sane.")
            else:
                logger.info(f"  ✓ split sizes ok: test_n={test_n} + val_n={val_n} of ~{n_perts_raw} raw perts")

    try:
        A.file.close()
    except Exception:
        pass

    if errors:
        logger.error("═" * 65)
        logger.error("  CONFIG ↔ DATA VALIDATION FAILED — fix the config and re-run:")
        for e in errors:
            logger.error(f"    ✗ {e}")
        logger.error("═" * 65)
        raise ConfigDataMismatch("; ".join(errors))

    logger.info("  CONFIG ↔ DATA VALIDATION PASSED")
    logger.info("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _make_logger(cfg: dict) -> logging.Logger:
    name   = cfg["dataset"]["name"]
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"spore_light_{name}_{ts}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = cfg["paths"]["_logs"] / f"spore_light_{name}_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"SPORE_light log → {log_path}")
    return logger


@contextmanager
def _timed(label: str, logger: logging.Logger):
    logger.info("═" * 65)
    logger.info(f"  {label}")
    logger.info("═" * 65)
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        mins, sec   = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        ram = psutil.Process().memory_info().rss / 1e9
        t_str = f"{hours}h {mins:02}m {sec:02}s" if hours else f"{mins}m {sec:02}s"
        logger.info(f"  ✓ {label}  —  {t_str}  |  RAM: {ram:.1f} GB")


# ══════════════════════════════════════════════════════════════════════════════
#  SUBSETTER UTILITIES  (two-pass, no full-matrix load)
#  For K562-scale datasets (>500k cells). RPE1/VCC don't need this.
# ══════════════════════════════════════════════════════════════════════════════

def scan_obs_metadata(h5ad_path: str, pert_col: str, ctrl_label: str,
                      min_genes_per_cell: int = 0, logger=None) -> dict:
    """Pass 1: read obs metadata only — no expression matrix access."""
    import h5py
    import numpy as np

    if logger:
        logger.info(f"[Subsetter] Pass 1: scanning obs from {h5ad_path}")
    t0 = time.time()

    with h5py.File(h5ad_path, "r") as f:
        obs_group = f["obs"]
        n_cells = f["X"].shape[0] if "X" in f else f["raw/X"].shape[0]

        # Perturbation labels
        if pert_col not in obs_group:
            raise KeyError(
                f"Column '{pert_col}' not found in obs. "
                f"Available: {list(obs_group.keys())}")

        pert_raw = obs_group[pert_col][:]
        if f"obs/{pert_col}/categories" in f:
            cats = f[f"obs/{pert_col}/categories"][:]
            cats = np.array([c.decode() if isinstance(c, bytes) else str(c)
                             for c in cats])
            codes = f[f"obs/{pert_col}/codes"][:]
            pert_labels = np.where(codes >= 0, cats[codes], ctrl_label)
        elif hasattr(pert_raw, "dtype") and pert_raw.dtype.kind in ("i", "u"):
            pert_labels = np.array([str(x) for x in pert_raw])
        else:
            pert_labels = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                    for x in pert_raw])

        # Optional QC filter
        qc_filter = None
        if min_genes_per_cell > 0:
            for qc_col in ("n_genes_by_counts", "n_genes", "n_features"):
                if qc_col in obs_group:
                    qc_vals   = obs_group[qc_col][:]
                    qc_filter = qc_vals >= min_genes_per_cell
                    if logger:
                        logger.info(
                            f"[Subsetter] QC filter ({qc_col} >= {min_genes_per_cell}): "
                            f"{int(qc_filter.sum()):,}/{n_cells:,} pass")
                    break

    pert_to_indices: dict = {}
    for i, lbl in enumerate(pert_labels):
        if qc_filter is not None and not qc_filter[i]:
            continue
        pert_to_indices.setdefault(lbl, []).append(i)

    elapsed = time.time() - t0
    n_perts = len([k for k in pert_to_indices if k != ctrl_label])
    n_ctrl  = len(pert_to_indices.get(ctrl_label, []))
    if logger:
        logger.info(
            f"[Subsetter] {n_perts:,} perturbations, {n_ctrl:,} controls "
            f"({elapsed:.1f}s scan)")

    return pert_to_indices


def _reservoir_sample(indices: list, k: int, rng) -> list:
    if len(indices) <= k:
        return list(indices)
    reservoir = list(indices[:k])
    for i in range(k, len(indices)):
        j = int(rng.integers(0, i + 1))
        if j < k:
            reservoir[j] = indices[i]
    return reservoir


def select_cell_indices(pert_to_indices: dict, ctrl_label: str,
                        n_perts: int, max_cells_per_pert: int,
                        max_ctrl_cells: int, rng) -> "np.ndarray":
    import numpy as np
    pert_counts  = {k: len(v) for k, v in pert_to_indices.items() if k != ctrl_label}
    ranked       = sorted(pert_counts, key=pert_counts.get, reverse=True)
    selected     = ranked[:n_perts]

    all_indices = []
    for pert in selected:
        all_indices.extend(_reservoir_sample(pert_to_indices[pert], max_cells_per_pert, rng))
    ctrl_idxs = pert_to_indices.get(ctrl_label, [])
    all_indices.extend(_reservoir_sample(ctrl_idxs, max_ctrl_cells, rng))

    return np.sort(np.array(all_indices, dtype=np.int64))


def write_subset(h5ad_path: str, selected_idx, output_path: str,
                 chunk_size: int = 5_000, logger=None):
    """Pass 2: write only selected rows to a new h5ad."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    if logger:
        logger.info(f"[Subsetter] Pass 2: writing {len(selected_idx):,}-cell subset → {output_path}")

    adata_backed = ad.read_h5ad(h5ad_path, backed="r")
    obs_subset   = adata_backed.obs.iloc[selected_idx].copy()
    var_subset   = adata_backed.var.copy()

    chunks = []
    for start in range(0, len(selected_idx), chunk_size):
        end      = min(start + chunk_size, len(selected_idx))
        chunk_X  = adata_backed.X[selected_idx[start:end], :]
        if sp.issparse(chunk_X):
            chunk_X = chunk_X.tocsr().astype(np.float32)
        else:
            chunk_X = sp.csr_matrix(chunk_X.astype(np.float32))
        chunks.append(chunk_X)
        gc.collect()

    adata_backed.file.close()
    X_out = sp.vstack(chunks, format="csr")
    del chunks
    gc.collect()

    adata_out = ad.AnnData(X=X_out, obs=obs_subset, var=var_subset)
    adata_out.write_h5ad(output_path)
    if logger:
        logger.info(f"[Subsetter] Done: {adata_out.n_obs:,} × {adata_out.n_vars:,}")
    return adata_out


def run_subsetter(cfg: dict, logger: logging.Logger):
    """
    Run the two-pass memory-safe subsetter.
    Writes the subset h5ad to cfg['paths']['subset_h5ad'] (or auto-named).
    Optionally updates cfg to point raw_h5ad at the subset for the pipeline.
    """
    import numpy as np

    sub_cfg   = cfg.get("subsetter", {})
    raw_path  = str(cfg["paths"]["_raw_h5ad"])
    pert_col  = cfg["dataset"]["perturbation_col"]
    ctrl_lbl  = cfg["dataset"]["control_label"]
    n_perts   = sub_cfg.get("n_perts",            250)
    max_cells = sub_cfg.get("max_cells_per_pert",  60)
    max_ctrl  = sub_cfg.get("max_ctrl_cells",    3000)
    min_genes = sub_cfg.get("min_genes_per_cell",  200)
    chunk     = sub_cfg.get("chunk_size",         5000)
    seed      = sub_cfg.get("seed",                 42)

    raw_p = Path(raw_path)
    out_path = cfg["paths"].get("subset_h5ad") or str(
        raw_p.parent / f"{raw_p.stem}_subset_{n_perts}perts.h5ad")

    if Path(out_path).exists():
        logger.info(f"[Subsetter] Subset already exists at {out_path} — skipping scan")
        cfg["paths"]["_raw_h5ad"] = Path(out_path)
        return

    rng = np.random.default_rng(seed)
    pert_to_idx = scan_obs_metadata(raw_path, pert_col, ctrl_lbl,
                                    min_genes_per_cell=min_genes, logger=logger)
    n_available = len([k for k in pert_to_idx if k != ctrl_lbl])
    n_use       = min(n_perts, n_available)
    if n_use < n_perts:
        logger.warning(f"[Subsetter] Only {n_available} perts available, requested {n_perts}")

    selected_idx = select_cell_indices(pert_to_idx, ctrl_lbl, n_use, max_cells, max_ctrl, rng)
    write_subset(raw_path, selected_idx, out_path, chunk_size=chunk, logger=logger)
    cfg["paths"]["_raw_h5ad"] = Path(out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER  (phases 0-12, no Phase 13)
# ══════════════════════════════════════════════════════════════════════════════

def _find_spore_plus(cfg: dict) -> Path:
    """
    Find the SPORE_plus root directory. Checks (in order):
      1. 'spore_plus_path' key in config
      2. Sibling directory named 'SPORE_plus' relative to spore_light.py
    """
    configured = cfg.get("spore_plus_path", None)
    if configured:
        p = Path(configured)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"spore_plus_path from config not found: {p}")

    # Repo layout: src/ sits next to this script
    here = Path(__file__).resolve().parent
    if (here / "src").exists():
        return here

    # Legacy layout: SPORE_plus is a sibling directory of this script
    sibling = Path(__file__).resolve().parent.parent / "SPORE_plus"
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        f"Cannot find SPORE_plus. Set 'spore_plus_path' in your config YAML, "
        f"or place SPORE_plus as a sibling of the SPORE_light directory. "
        f"Looked for: {sibling}")


# ══════════════════════════════════════════════════════════════════════════════
#  ONE-STOP SUBSTRATE HELPERS  (Defects 1, 2, 7)
# ══════════════════════════════════════════════════════════════════════════════

def emit_hvgraw_splits(splits: dict, cfg: dict, logger) -> dict:
    """
    Defect 1: write {name}_{split}_hvgraw.h5ad = the HVG panel in RAW counts — the canonical substrate
    input. Must be called right after Phase 8 and BEFORE Phase 9 normalizes the in-memory splits (Phase 8
    only subsets genes, so at this point `splits` still holds RAW counts on the HVG panel).
    """
    splits_dir = cfg["paths"]["_splits"]
    name       = cfg["dataset"]["name"]
    paths = {}
    logger.info("  Emitting HVG-panel RAW-count splits (Defect 1 — THE substrate builder input):")
    for sk in ["train", "val", "test"]:
        a = splits.get(sk)
        if a is None:
            continue
        p = splits_dir / f"{name}_{sk}_hvgraw.h5ad"
        a.write_h5ad(p)
        paths[sk] = str(p)
        logger.info(f"    → {sk}: {a.n_obs:,} × {a.n_vars:,} genes (raw) → {p}")
    return paths


def _split_targets(labels, ctrl, sep, is_comb):
    out = set()
    for lbl in labels:
        if str(lbl) == str(ctrl):
            continue
        if is_comb and sep and sep in str(lbl):
            out.update(p.strip() for p in str(lbl).split(sep) if p.strip())
        else:
            out.add(str(lbl))
    return out


def report_heldout_coverage(splits: dict, cfg: dict, logger) -> dict:
    """
    Defect 2/7: report how many held-out (val+test) perturbation-target genes are present as NODES in the
    HVG panel, plus the perturbed-but-never-measured ceiling. missing_measured should be 0 once the Phase-8
    force-carry works (that is the whole point). Loud, not silent.
    """
    import json as _json
    splits_dir = cfg["paths"]["_splits"]
    name       = cfg["dataset"]["name"]
    ds         = cfg["dataset"]
    ctrl       = ds["control_label"]
    sep        = ds.get("perturbation_separator", "+")
    is_comb    = ds.get("perturbation_structure") == "combinatorial"

    train = splits.get("train")
    panel = set(map(str, train.var_names)) if train is not None else set()

    sj = splits_dir / f"{name}_split_indices.json"
    meta = _json.load(open(sj))
    va = _split_targets(meta.get("val_labels", []),  ctrl, sep, is_comb)
    te = _split_targets(meta.get("test_labels", []), ctrl, sep, is_comb)
    heldout = va | te
    present = {g for g in heldout if g in panel}
    missing = sorted(heldout - present)

    measured        = cfg.get("_measured_symbols", set())
    never_measured  = sorted(g for g in missing if measured and g not in measured)
    missing_measured = sorted(g for g in missing if measured and g in measured)

    cov    = 100.0 * len(present) / max(1, len(heldout))
    va_cov = 100.0 * len({g for g in va if g in panel}) / max(1, len(va))
    te_cov = 100.0 * len({g for g in te if g in panel}) / max(1, len(te))

    logger.info("  ── HELD-OUT TARGET COVERAGE (zero-shot node coverage) ──")
    logger.info(f"    val+test targets present as nodes: {len(present)}/{len(heldout)} ({cov:.1f}%)")
    logger.info(f"    val {va_cov:.1f}% ({len(va)} targets) · test {te_cov:.1f}% ({len(te)} targets)")
    logger.info(f"    missing: {len(missing)}  (never-measured ceiling: {len(never_measured)}; "
                f"measured-but-filtered: {len(missing_measured)})")
    if missing_measured:
        logger.warning(f"    ⚠ {len(missing_measured)} MEASURED held-out targets are NOT nodes "
                       f"(force-carry should make this 0): {missing_measured[:15]}")

    return dict(
        val_test_targets=len(heldout), present_as_nodes=len(present), missing=missing,
        coverage_pct=round(cov, 2), val_coverage_pct=round(va_cov, 2), test_coverage_pct=round(te_cov, 2),
        never_measured=never_measured, missing_measured=missing_measured,
        n_val_targets=len(va), n_test_targets=len(te))


def final_verification_gate(cfg: dict, coverage: dict, substrate_paths: dict, logger) -> bool:
    """
    Defect 7: SPORE proves its own output before handoff. Refuses success unless:
      (1) held-out target coverage is at the measurable ceiling (missing ⊆ never-measured);
      (2) the train-hybrid firewall holds (obs['split'] ⊆ {train, control}; 0 val/test units);
      (3) the emitted substrate is the HVG panel in RAW counts (control units are integer-valued).
    Writes a PASS/FAIL block + COVERAGE_REPORT.{md,json}. Returns True (PASS) / False (FAIL).
    """
    import json as _json
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    name    = cfg["dataset"]["name"]
    out_dir = cfg["paths"]["_processed"]
    checks  = []

    # (1) coverage at ceiling
    missing_measured = coverage.get("missing_measured", [])
    c1 = (len(missing_measured) == 0)
    checks.append(("coverage at measurable ceiling (missing ⊆ never-measured)", c1,
                   f"{len(missing_measured)} measured held-out targets missing"))

    # (2) firewall on the train hybrid
    c2, c2_detail = False, "train_hybrid not found"
    hyb = substrate_paths.get("train_hybrid")
    if hyb and Path(hyb).exists():
        Ah = ad.read_h5ad(hyb, backed="r")
        if "split" in Ah.obs.columns:
            leak = int(Ah.obs["split"].astype(str).isin(["val", "test"]).sum())
            c2 = (leak == 0)
            c2_detail = f"{leak} val/test units in train hybrid"
        else:
            c2, c2_detail = True, "no split column (assumed train+control only)"
        try:
            Ah.file.close()
        except Exception:
            pass
    checks.append(("train-hybrid firewall (0 val/test units)", c2, c2_detail))

    # (3) substrate is HVG-panel RAW counts (control units integer-valued)
    c3, c3_detail = False, "allsplits substrate not found"
    allsp = substrate_paths.get("allsplits_metacell")
    if allsp and Path(allsp).exists():
        A = ad.read_h5ad(allsp)
        n_genes = A.n_vars
        ctrl = cfg["dataset"]["control_label"]
        is_ctrl = (A.obs[cfg["dataset"]["perturbation_col"]].astype(str) == str(ctrl)).to_numpy()
        Xc = A.X[is_ctrl][:200] if is_ctrl.any() else A.X[:200]
        Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
        nz = Xc[Xc > 0]
        near_int = bool(nz.size == 0 or np.mean(np.abs(nz - np.round(nz)) < 1e-4) > 0.98)
        maxval = float(nz.max()) if nz.size else 0.0
        c3 = near_int and (maxval > 10.0 or nz.size == 0)
        c3_detail = f"{n_genes} genes; control units integer={near_int}; max={maxval:.1f}"
    checks.append(("substrate is HVG-panel RAW counts", c3, c3_detail))

    passed = all(c for _, c, _ in checks)

    lines = [f"# {name} — SPORE substrate coverage + firewall verification",
             f"_(generated by spore_light.py final gate)_\n",
             f"- **held-out (val+test) target node coverage: {coverage['coverage_pct']}%** "
             f"({coverage['present_as_nodes']}/{coverage['val_test_targets']})",
             f"  - val {coverage['val_coverage_pct']}% ({coverage['n_val_targets']} targets) · "
             f"test {coverage['test_coverage_pct']}% ({coverage['n_test_targets']} targets)",
             f"- missing held-out targets: {len(coverage['missing'])} "
             f"(never-measured ceiling: {len(coverage['never_measured'])}; "
             f"measured-but-filtered: {len(coverage['missing_measured'])})",
             ""]
    for label, ok, detail in checks:
        lines.append(f"- **{'PASS ✓' if ok else 'FAIL ✗'}** — {label}  ({detail})")
    lines.append("")
    lines.append(f"## {'✅ HYPHAE-READY' if passed else '❌ NOT READY'}")
    (out_dir / "COVERAGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _json.dump({"passed": passed, "coverage": {k: v for k, v in coverage.items() if k != "missing"},
                "checks": [{"check": l, "pass": ok, "detail": d} for l, ok, d in checks],
                "substrate_paths": substrate_paths},
               open(out_dir / "COVERAGE_REPORT.json", "w"), indent=2)

    logger.info("═" * 65)
    logger.info("  FINAL VERIFICATION GATE")
    for label, ok, detail in checks:
        logger.info(f"    [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    logger.info(f"  → {'HYPHAE-READY ✓' if passed else 'NOT READY ✗ — see COVERAGE_REPORT.md'}")
    logger.info(f"  Report: {out_dir / 'COVERAGE_REPORT.md'}")
    logger.info("═" * 65)
    return passed


def run_pipeline(cfg: dict, logger: logging.Logger) -> bool:
    """
    Execute SPORE phases 0-12 and emit the HYPHAE-ready substrate + verification gate in one run.
    Returns True on success (all gates PASS), False on failure.
    """
    spore_root = _find_spore_plus(cfg)
    if str(spore_root) not in sys.path:
        sys.path.insert(0, str(spore_root))

    logger.info(f"  SPORE_plus root : {spore_root}")

    # Apply plotting style early so any diagnostic plots use the right theme
    try:
        from src.utils import apply_sporeplus_style
        apply_sporeplus_style(cfg)
        if cfg.get("plotting", {}).get("save_figures", False):
            logger.info(f"  Figures → {cfg['paths']['_figures']}")
    except Exception:
        pass  # plotting style is cosmetic; don't fail the run

    # ── Import phase modules ──────────────────────────────────────────────────
    from src.phase00_sparse_convert       import run_phase0
    from src.phase01_detection            import run_phase1
    from src.phase02_cell_triage          import run_phase2
    from src.phase03_ambient              import run_phase3
    from src.phase04_doublets             import run_phase4
    from src.phase05_escaper_filtering    import run_phase5
    from src.phase06_gene_triage          import run_phase6
    from src.phase07_splits               import run_phase7
    from src.phase08_hvg                  import run_phase8
    from src.phase09_normalization        import run_phase9, extract_p9_data
    from src.phase10_confounders          import run_phase10
    from src.phase11_cell_line_separation import run_phase11
    # Phase 12 rewritten (Defect 3): native control-preserving MBK k=2 substrate builder.
    from src.phase12_metacell             import run_phase12

    # Diagnostics are optional — use no-ops if the module isn't present
    try:
        from src.diagnostics import (
            report_phase0_diagnostics,
            report_phase1_diagnostics,
            report_phase2_diagnostics,
            report_phase3_diagnostics,
            report_phase4_diagnostics,
            report_phase5_diagnostics,
            report_phase6_diagnostics,
            report_phase7_diagnostics,
            report_phase8_diagnostics,
            report_phase9_diagnostics,
            report_phase10_diagnostics,
            report_phase11_diagnostics,
            report_phase12_diagnostics,
        )
    except ImportError:
        logger.warning("  src.diagnostics not found — diagnostic reports skipped")
        def _noop(*a, **k): pass
        (report_phase0_diagnostics, report_phase1_diagnostics,
         report_phase2_diagnostics, report_phase3_diagnostics,
         report_phase4_diagnostics, report_phase5_diagnostics,
         report_phase6_diagnostics, report_phase7_diagnostics,
         report_phase8_diagnostics, report_phase9_diagnostics,
         report_phase10_diagnostics, report_phase11_diagnostics,
         report_phase12_diagnostics) = [_noop] * 13

    import anndata as ad
    import numpy   as np

    # exp_022 plotting restoration: SPORE_light's driver never called plotting.py (only the spore+.ipynb
    # notebook did). Restore per-phase figure generation, gated by plotting.save_figures + wrapped so a
    # plotting failure can NEVER kill the pipeline. plotting.py itself is unchanged (identical to SPORE_plus).
    try:
        import src.plotting as splt
    except Exception:
        splt = None
    def _fig(fn_name, *args):
        if splt is None or not cfg.get("plotting", {}).get("save_figures", False):
            return
        fn = getattr(splt, fn_name, None)
        if fn is None:
            return
        try:
            fn(*args, cfg)
            logger.info(f"  [plot] {fn_name} -> {cfg['paths']['_figures']}")
        except Exception as e:
            logger.warning(f"  [plot] {fn_name} failed ({type(e).__name__}: {e}) — skipped")

    ds_name       = cfg["dataset"]["name"]
    n_jobs        = cfg.get("runtime", {}).get("n_jobs", 6)
    splits_dir    = cfg["paths"]["_splits"]
    output_dir    = cfg["paths"]["_processed"]

    logger.info("═" * 65)
    logger.info(f"  SPORE_LIGHT  ·  {ds_name}  (phases 0-12 → HYPHAE-ready substrate; no CHITIN)")
    logger.info(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  n_jobs       : {n_jobs}")
    logger.info(f"  n_top_genes  : {cfg.get('phase8_hvg', {}).get('n_top_genes', 1000)}")
    logger.info(f"  Output       : {output_dir}")
    logger.info("═" * 65)

    t_start = time.time()

    adata             = None
    p7_splits         = [{}]
    p10_results       = {}
    detection_results = {}
    cell_line_meta    = {"n_cell_lines": 1}
    coverage          = {}
    substrate_paths   = {}
    p2_waterfall      = {}
    p5_escaper_stats  = p5_pert_sizes = None
    p6_waterfall      = {}
    p6_rescued        = []
    p9_cache          = {}

    try:
        # ── Phase 0 ──────────────────────────────────────────────────────────
        with _timed("PHASE 0 · Data Ingestion & Sparsification", logger):
            adata = run_phase0(cfg, logger)
        report_phase0_diagnostics(adata)

        # ── Phase 1 ──────────────────────────────────────────────────────────
        with _timed("PHASE 1 · Detection", logger):
            adata, detection_results = run_phase1(adata, cfg, logger)
        report_phase1_diagnostics(adata, detection_results)
        _fig("plot_sparsity_overview", adata)

        # ── Phase 2 ──────────────────────────────────────────────────────────
        with _timed("PHASE 2 · Cell-Level Triage", logger):
            adata, p2_waterfall = run_phase2(adata, cfg, logger)
        report_phase2_diagnostics(p2_waterfall)
        _fig("plot_qc_violin", adata); _fig("plot_mt_scatter", adata)

        # ── Phase 3 (disabled by default) ────────────────────────────────────
        with _timed("PHASE 3 · Ambient RNA", logger):
            adata = run_phase3(adata, cfg, logger)
        report_phase3_diagnostics(adata, cfg)

        # ── Phase 4 (disabled by default) ────────────────────────────────────
        with _timed("PHASE 4 · Doublet Detection", logger):
            adata = run_phase4(adata, cfg, logger)
        report_phase4_diagnostics(adata, cfg)

        # ── Phase 5 ──────────────────────────────────────────────────────────
        with _timed("PHASE 5 · Escaper Filtering", logger):
            adata, p5_escaper_stats, p5_pert_sizes = run_phase5(adata, cfg, logger)
        report_phase5_diagnostics(adata, p5_escaper_stats, p5_pert_sizes)
        _fig("plot_perturbation_efficiency", adata)

        # ── Phase 6 ──────────────────────────────────────────────────────────
        with _timed("PHASE 6 · Gene-Level Triage", logger):
            adata, p6_waterfall, p6_rescued = run_phase6(adata, cfg, logger)
        report_phase6_diagnostics(adata, p6_waterfall, p6_rescued)
        _fig("plot_phase6_diagnostics", adata)

        # ── Phase 7 ──────────────────────────────────────────────────────────
        with _timed("PHASE 7 · Data Splits", logger):
            p7_splits = run_phase7(adata, cfg, logger)
        report_phase7_diagnostics(p7_splits)
        _fig("plot_phase7_splits", p7_splits)

        del adata
        gc.collect()

        # ── Phase 8 ──────────────────────────────────────────────────────────
        with _timed("PHASE 8 · HVG Selection", logger):
            splits, hvg_names = run_phase8(p7_splits[0], cfg, logger)
            p7_splits = [splits]
        report_phase8_diagnostics(
            splits.get("train", list(splits.values())[0]) if splits
            else p7_splits[0].get("train"))
        _fig("plot_phase8_hvg", splits.get("train") if splits else p7_splits[0].get("train"))

        # ── Defect 1 + 2: emit HVG-panel RAW splits + coverage report ─────────
        # Phase 8 only subset genes, so `splits` still holds RAW counts on the HVG panel here. Emit them
        # for the Phase-12 substrate builder BEFORE Phase 9 normalizes the in-memory copies. Then report
        # held-out target node coverage (loudly).
        emit_hvgraw_splits(splits, cfg, logger)
        coverage = report_heldout_coverage(splits, cfg, logger)

        # ── Phase 9 ──────────────────────────────────────────────────────────
        with _timed("PHASE 9 · Normalization", logger):
            _p9 = run_phase9(p7_splits[0], cfg, logger)

        # run_phase9 has two return paths:
        #   Large-dataset path: writes p9 h5ads to disk, returns (splits_dict, cache)
        #   Small-dataset path: normalizes in-memory, returns (train, val, test) AnnData tuple
        import anndata as _ad
        if isinstance(_p9, tuple) and len(_p9) == 2 and isinstance(_p9[0], dict):
            # Large-dataset path
            splits, p9_cache = _p9
        elif (isinstance(_p9, tuple) and len(_p9) == 3
              and hasattr(_p9[0], 'n_obs')):
            # Small-dataset path: got (train_adata, val_adata, test_adata)
            splits   = {"train": _p9[0], "val": _p9[1], "test": _p9[2]}
            p9_cache = {}
        elif isinstance(_p9, tuple) and len(_p9) == 2:
            splits, p9_cache = _p9
        elif isinstance(_p9, dict):
            splits   = _p9
            p9_cache = extract_p9_data(splits)
        else:
            splits   = p7_splits[0]
            p9_cache = {}
            logger.warning("  Phase 9 returned unexpected type — using pre-P9 splits")
        p7_splits = [splits]

        # Phase 10 (subprocess) reads {ds_name}_{split}_p9.h5ad from splits_dir.
        # Small-dataset path does NOT write these files. Ensure they exist on disk.
        for _key in ["train", "val", "test"]:
            _p9_path = splits_dir / f"{ds_name}_{_key}_p9.h5ad"
            if not _p9_path.exists() and _key in splits and splits[_key] is not None:
                logger.info(f"  Saving p9 {_key} → {_p9_path}")
                splits[_key].write_h5ad(_p9_path)

        report_phase9_diagnostics(p9_cache, cfg)
        _fig("plot_phase9_normalization", p9_cache)

        # ── Phase 10 ─────────────────────────────────────────────────────────
        with _timed("PHASE 10 · Confounder Mitigation (PCA + Harmony)", logger):
            p10_results = run_phase10(None, cfg, logger)
        report_phase10_diagnostics(p10_results.get("train"), cfg)
        _fig("plot_phase10_integration", p10_results.get("train"))

        # ── Phase 11 ─────────────────────────────────────────────────────────
        with _timed("PHASE 11 · Cell Line Detection", logger):
            p10_results["train"], cell_line_meta = run_phase11(
                p10_results["train"], cfg, logger)
        report_phase11_diagnostics(p10_results["train"], cell_line_meta, cfg)

        # ── Phase 12 (Defect 3): control-preserving MBK k=2 substrate ────────
        # Reads the Phase-8 HVG-panel RAW splits (hvgraw) and emits the HYPHAE-ready substrate directly:
        # {name}_allsplits_metacell_ctrlpreserved.h5ad + {name}_train_hybrid.h5ad. No p10, no Harmony
        # embedding, no cell-line loop — the blessed recipe computes its own per-group PCA.
        with _timed("PHASE 12 · Control-preserving MBK k=2 Substrate", logger):
            substrate_paths = run_phase12(cfg, logger)

        # ── Defect 7: final verification gate (SPORE proves its own output) ──
        gate_pass = final_verification_gate(cfg, coverage, substrate_paths, logger)

        # ── Pipeline complete ────────────────────────────────────────────────
        elapsed      = time.time() - t_start
        hours, rem   = divmod(int(elapsed), 3600)
        mins, sec    = divmod(rem, 60)
        final_ram    = psutil.Process().memory_info().rss / 1e9
        n_top_genes  = cfg.get("phase8_hvg", {}).get("n_top_genes", 1000)

        logger.info("═" * 65)
        logger.info(f"  SPORE_LIGHT COMPLETE  ·  {ds_name}  "
                    f"({'HYPHAE-READY ✓' if gate_pass else 'GATE FAILED ✗'})")
        logger.info(f"  Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Duration     : {hours}h {mins:02}m {sec:02}s")
        logger.info(f"  Final RAM    : {final_ram:.1f} GB")
        logger.info(f"  n_top_genes  : {n_top_genes}")
        logger.info(f"  Output       : {output_dir}")
        logger.info("  ── HYPHAE-ready deliverables ──")
        for label, key in [("all-splits substrate", "allsplits_metacell"),
                           ("train hybrid (graph substrate)", "train_hybrid"),
                           ("aligned split JSON", "allsplits_split_json")]:
            p = substrate_paths.get(key)
            if p and Path(p).exists():
                size_mb = Path(p).stat().st_size / 1e6
                logger.info(f"    {label}: {p}  ({size_mb:.0f} MB)")
        logger.info(f"    coverage/firewall report: {output_dir / 'COVERAGE_REPORT.md'}")
        logger.info("═" * 65)

        if not gate_pass:
            logger.error("  Final verification gate FAILED — substrate is NOT HYPHAE-ready. "
                         "See COVERAGE_REPORT.md.")
            return False

    except Exception as exc:
        logger.error("═" * 65)
        logger.error(f"  PIPELINE FAILED  ·  {ds_name}")
        logger.error(f"  Error: {exc}")
        logger.error("═" * 65, exc_info=True)
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  RAM ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

def estimate_resources(cfg: dict, logger: logging.Logger):
    """Print a rough RAM and wall-time estimate for the configured dataset."""
    raw_path = cfg["paths"].get("_raw_h5ad")
    if raw_path and Path(raw_path).exists():
        size_gb = Path(raw_path).stat().st_size / 1e9
        logger.info(f"  Raw h5ad size: {size_gb:.1f} GB on disk")
        # Sparse h5ad: on-disk size ≈ actual sparse RAM footprint
        logger.info(f"  Phase 0 RAM est: ~{size_gb*1.2:.1f} GB (sparse + obs)")
        logger.info(f"  Phase 9 peak est (train, 5k genes): "
                    f"~{size_gb*0.75*1.5:.1f} GB (normalized in memory)")
    n_top = cfg.get("phase8_hvg", {}).get("n_top_genes", 1000)
    n_jobs = cfg.get("runtime", {}).get("n_jobs", 6)
    logger.info(f"  n_top_genes: {n_top}")
    logger.info(f"  n_jobs: {n_jobs}")
    logger.info("  Phase 12 (metacell aggregation) wall time depends on n_perturbations.")
    logger.info("  RPE1 (~1500 perts, 8 workers): estimated 30-90 min")
    logger.info("  K562 GWPS (~9000 perts, 8 workers): estimated 4+ hours — use --subset-first")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="spore_light.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",        required=True, metavar="YAML",
                   help="Path to spore_light_config.yaml")
    p.add_argument("--subset-first",  action="store_true",
                   help="Run the cell subsetter before the pipeline "
                        "(for K562-scale datasets > 500k cells)")
    p.add_argument("--subset-only",   action="store_true",
                   help="Run the subsetter only, do not run the pipeline")
    p.add_argument("--estimate",      action="store_true",
                   help="Print RAM and time estimates for this config and exit")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"ERROR loading config: {exc}", file=sys.stderr)
        return 1

    logger = _make_logger(cfg)

    if args.estimate:
        estimate_resources(cfg, logger)
        return 0

    # Defect 5: fail at second zero on any config/data mismatch (before compute).
    try:
        validate_config_against_data(cfg, logger)
    except ConfigDataMismatch as exc:
        print(f"ERROR: config/data validation failed: {exc}", file=sys.stderr)
        return 1

    if args.subset_first or args.subset_only:
        with _timed("SUBSETTER", logger):
            run_subsetter(cfg, logger)
        if args.subset_only:
            logger.info("--subset-only: done. Exiting without running pipeline.")
            return 0

    success = run_pipeline(cfg, logger)
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())