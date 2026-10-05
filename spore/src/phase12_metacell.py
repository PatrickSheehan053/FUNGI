"""
SPORE+ · src/phase12_metacell.py
─────────────────────────────────
Phase 12: Control-preserving MBK k=2 metacell substrate  (the BLESSED recipe, folded into SPORE)

exp_009.1 rebuild (Defect 3). The OLD Phase 12 ran a "dynamic-graining" metacell aggregation on the
NORMALIZED p10 splits, and its output was NEVER used downstream — the real substrate was produced by a
separate hand-run script (`build_allsplits_metacell_input.py` + `metacell_sweep.py`). That is now folded
in here so `python spore_light.py --config ...` emits the HYPHAE-ready substrate DIRECTLY.

Recipe (control-preserving MBK k=2 — verbatim from the blessed builder + metacell_sweep.py):
  • Read the Phase-8 HVG-panel RAW-count splits ({name}_{split}_hvgraw.h5ad, all three).
  • non-targeting CONTROL cells are kept SINGLE-CELL everywhere (they anchor the co-expression backbone).
  • Each perturbed group -> MiniBatchKMeans(n_cells // k) metacells (k=2 default = the 2x ceiling), where
    clustering uses a per-group TruncatedSVD-10 embedding if the group has >= 50 cells, else a whole-dataset
    normalize->log1p->PCA-50 fallback embedding. Metacells aggregate the MEAN of RAW counts.
  • Aggregation is WITHIN a single perturbation, so it never crosses the train/val/test boundary.

Outputs (into the processed dir):
  • {name}_allsplits_metacell_ctrlpreserved.h5ad — units x HVG genes, RAW counts; obs: {pert_col}, split,
    n_cells_in_metacell.  (the full-coverage HYPHAE input, all splits)
  • {name}_train_hybrid.h5ad                      — the firewall-safe graph substrate: split in
    {train, control} ONLY (asserts 0 val/test units).
  • {name}_allsplits_metacell_split_indices.json  — the aligned perturbation split (labels unchanged).

Hard-fail guards (the leakage/consistency gate — from the blessed builder):
  (1) every metacell is built from exactly ONE perturbation (structural — within-group aggregation);
  (2) the train/val/test perturbation-label partition is pairwise DISJOINT + COMPLETE over present perts;
  (3) every non-targeting control unit is SINGLE-CELL (n_cells_in_metacell == 1).
"""

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

try:
    from .utils import log_phase_header, snapshot, force_gc
except ImportError:  # worker / bare-module import context
    from utils import log_phase_header, snapshot, force_gc

_PER_GROUP_PCA_THRESHOLD = 50  # groups with >= this many cells get a per-group mini-PCA (blessed default)


# ═══════════════════════════════════════════════════════════════════════════
#  BLESSED EMBEDDING + AGGREGATION PRIMITIVES  (verbatim from metacell_sweep.py / agg_variants.py)
# ═══════════════════════════════════════════════════════════════════════════

def _mean_rows(sub):
    """Mean expression over rows of a (sparse or dense) submatrix -> float32 vector."""
    return (np.asarray(sub.mean(axis=0)).ravel() if sp.issparse(sub) else sub.mean(axis=0)).astype(np.float32)


def _compute_group_pca(X_sub, n_comps: int = 10, seed: int = 42):
    n_cells, n_genes = X_sub.shape
    n_comps_actual = min(n_comps, n_cells - 1, n_genes - 1)
    if n_comps_actual < 2:
        return None
    try:
        svd = TruncatedSVD(n_components=n_comps_actual, algorithm="randomized",
                           n_iter=2, random_state=seed)
        return svd.fit_transform(X_sub).astype(np.float32)
    except Exception:
        return None


def _compute_global_embedding(adata, n_comps: int = 50, seed: int = 42, logger=None):
    import scanpy as sc
    if logger:
        logger.info(f"  [Phase 12] computing global PCA-{n_comps} fallback embedding "
                    f"(for small perturbation groups)...")
    tmp = ad.AnnData(X=adata.X.copy())
    with threadpool_limits(limits=1):
        sc.pp.normalize_total(tmp, target_sum=None)
        sc.pp.log1p(tmp)
        sc.pp.pca(tmp, n_comps=n_comps, zero_center=True, svd_solver="randomized", random_state=seed)
    emb = np.asarray(tmp.obsm["X_pca"], dtype=np.float32)
    del tmp
    return emb


def _compute_one_embedding(pert, X, pert_values, seed):
    """Per-group TruncatedSVD-10 (k-independent). Slices its own group lazily."""
    X_sub = X[pert_values == pert]
    if X_sub.shape[0] >= _PER_GROUP_PCA_THRESHOLD:
        return pert, _compute_group_pca(X_sub, n_comps=10, seed=seed)
    return pert, None


def _get_group_embeddings(adata, pert_values, unique_perts, seed, n_jobs, logger=None):
    if logger:
        logger.info(f"  [Phase 12] per-group PCA embeddings for {len(unique_perts)} groups "
                    f"(n_jobs={n_jobs}, threading)")
    X = adata.X
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_compute_one_embedding)(pert, X, pert_values, seed) for pert in unique_perts)
    return {str(p): e for p, e in results}


def _aggregate_group_mbk(X_sub, cluster_emb, k, seed):
    """MiniBatchKMeans metacells: n_metacells = n_cells // k; aggregate MEAN of raw counts."""
    n_cells = X_sub.shape[0]
    n_metacells = max(1, n_cells // k)
    if n_metacells > 1 and cluster_emb is not None:
        km = MiniBatchKMeans(n_clusters=n_metacells, random_state=seed, n_init="auto")
        labels = km.fit_predict(cluster_emb)
    else:
        labels = np.zeros(n_cells, dtype=int)
    exprs, ncells = [], []
    for c in range(n_metacells):
        mask = labels == c
        if int(mask.sum()) == 0:
            continue
        exprs.append(_mean_rows(X_sub[mask]))
        ncells.append(int(mask.sum()))
    return exprs, ncells


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 12
# ═══════════════════════════════════════════════════════════════════════════

def _load_split_map(split_json_path, ctrl_label, logger):
    """pert(str) -> split from the Phase-7 split JSON. Returns (map, json_dict)."""
    with open(split_json_path) as f:
        meta = json.load(f)
    pert_to_split = {}
    for key, s in (("train_labels", "train"), ("val_labels", "val"), ("test_labels", "test")):
        for p in meta.get(key, []):
            if str(p) != str(ctrl_label):
                pert_to_split[str(p)] = s
    logger.info(f"  [Phase 12] split JSON: "
                f"{sum(v=='train' for v in pert_to_split.values())} train / "
                f"{sum(v=='val' for v in pert_to_split.values())} val / "
                f"{sum(v=='test' for v in pert_to_split.values())} test perturbations")
    return pert_to_split, meta


def run_phase12(cfg: dict, logger):
    """
    Build the control-preserving MBK k=2 substrate from the Phase-8 HVG-panel RAW-count splits.
    Returns a dict of output paths (or {} if disabled).
    """
    log_phase_header(logger, 12, "Control-preserving MBK k=2 Metacell Substrate")
    p12          = cfg.get("phase12_metacell", {})
    ds           = cfg.get("dataset", {})
    dataset_name = ds.get("name", "dataset")
    pert_col     = ds.get("perturbation_col", "gene")
    ctrl_label   = ds.get("control_label", "non-targeting")
    out_dir      = cfg["paths"]["_processed"]
    splits_dir   = cfg["paths"]["_splits"]
    k            = int(p12.get("mbk_k", 2))
    seed         = int(p12.get("seed", cfg.get("phase7_splits", {}).get("random_seed", 42)))
    n_jobs       = int(cfg.get("runtime", {}).get("n_jobs", 6))

    if not p12.get("enabled", True):
        logger.info("  Phase 12: DISABLED")
        return {}

    t0 = time.perf_counter()

    # ── 1. locate the HVG-panel raw split files (Defect 1 output) ────────────────────────────────────
    hvgraw = {sk: splits_dir / f"{dataset_name}_{sk}_hvgraw.h5ad" for sk in ("train", "val", "test")}
    present = {sk: p for sk, p in hvgraw.items() if p.exists()}
    if "train" not in present:
        raise FileNotFoundError(
            f"[Phase 12] HVG-panel raw split not found: {hvgraw['train']}. "
            f"Phase 8 must emit {dataset_name}_{{split}}_hvgraw.h5ad before Phase 12.")

    split_json = splits_dir / f"{dataset_name}_split_indices.json"
    if not split_json.exists():
        raise FileNotFoundError(f"[Phase 12] split JSON not found: {split_json} (Phase 7 must run first).")
    pert_to_split, src_meta = _load_split_map(split_json, ctrl_label, logger)

    # ── 2. load + concat the HVG-panel raw splits ───────────────────────────────────────────────────
    parts, var0 = [], None
    for sk in ("train", "val", "test"):
        if sk not in present:
            logger.warning(f"  [Phase 12] {sk} hvgraw missing — skipping (substrate will lack {sk} units)")
            continue
        A = ad.read_h5ad(present[sk])
        if var0 is None:
            var0 = list(A.var_names)
        else:
            assert list(A.var_names) == var0, f"var_names mismatch in {sk} — cannot concat safely"
        logger.info(f"  [Phase 12] {sk}: {A.n_obs:,} cells x {A.n_vars:,} genes")
        parts.append(A)
    adata = ad.concat(parts, join="inner", index_unique=None)
    adata.obs_names_make_unique()
    var_df = parts[0].var.copy()
    del parts
    force_gc(logger)
    n_genes = adata.n_vars
    logger.info(f"  [Phase 12] concat: {adata.n_obs:,} cells x {n_genes:,} genes")

    pert_values = adata.obs[pert_col].astype(str).to_numpy()
    unique = np.unique(pert_values)
    perturbed_labels = [p for p in unique if p != ctrl_label]

    # every present perturbed label must be assignable to a split
    missing_from_json = sorted(set(perturbed_labels) - set(pert_to_split))
    if missing_from_json:
        raise ValueError(
            f"[Phase 12] {len(missing_from_json)} perturbations present in the data are absent from the "
            f"split JSON: e.g. {missing_from_json[:8]}")

    n_ctrl = int((pert_values == ctrl_label).sum())
    logger.info(f"  [Phase 12] {len(perturbed_labels)} perturbation groups | {n_ctrl:,} control cells "
                f"(single-cell) | {adata.n_obs - n_ctrl:,} perturbed cells | k={k}")

    # ── 3. per-group embeddings (blessed recipe) ────────────────────────────────────────────────────
    emb_by_pert = _get_group_embeddings(adata, pert_values, unique, seed, n_jobs, logger)
    group_sizes = {p: int((pert_values == p).sum()) for p in unique}
    needs_global = any(emb_by_pert.get(str(p)) is None and max(1, n // k) > 1
                       for p, n in group_sizes.items() if p != ctrl_label)
    global_emb = _compute_global_embedding(adata, seed=seed, logger=logger) if needs_global else None

    # ── 4. aggregate: controls single-cell; perturbed -> MBK k ──────────────────────────────────────
    X = adata.X
    all_exprs, all_perts, all_ncells, all_split = [], [], [], []
    n_perturbed_units = 0
    for pi, pert in enumerate(unique):
        mask = pert_values == pert
        X_sub = X[mask]
        if pert == ctrl_label:
            sub = X_sub.toarray() if sp.issparse(X_sub) else np.asarray(X_sub)
            for r in range(sub.shape[0]):
                all_exprs.append(sub[r].astype(np.float32))
                all_ncells.append(1)
            all_perts.extend([pert] * sub.shape[0])
            all_split.extend(["control"] * sub.shape[0])
        else:
            local = emb_by_pert.get(str(pert))
            cl = local if local is not None else (global_emb[mask] if global_emb is not None else None)
            exprs, ncells = _aggregate_group_mbk(X_sub, cl, k, seed)
            all_exprs.extend(exprs)
            all_ncells.extend(ncells)
            all_perts.extend([pert] * len(exprs))
            all_split.extend([pert_to_split[str(pert)]] * len(exprs))
            n_perturbed_units += len(exprs)
        if (pi + 1) % 250 == 0:
            logger.info(f"  [Phase 12] aggregated {pi + 1}/{len(unique)} groups "
                        f"({time.perf_counter() - t0:.0f}s)")

    X_meta = np.vstack(all_exprs).astype(np.float32)
    obs = pd.DataFrame({pert_col: all_perts, "n_cells_in_metacell": all_ncells, "split": all_split})
    am = ad.AnnData(X=X_meta, obs=obs, var=var_df)

    # ── 5. HARD-FAIL GUARDS ─────────────────────────────────────────────────────────────────────────
    assert not obs["split"].isna().any(), "some units have no split assignment"
    ctrl_units = obs[obs[pert_col] == ctrl_label]
    assert (ctrl_units["n_cells_in_metacell"] == 1).all(), "a control unit is NOT single-cell"
    assert len(ctrl_units) == n_ctrl, f"control unit count {len(ctrl_units)} != input controls {n_ctrl}"
    out_perturbed = set(obs.loc[obs[pert_col] != ctrl_label, pert_col].astype(str))
    tr = {p for p in out_perturbed if pert_to_split[p] == "train"}
    va = {p for p in out_perturbed if pert_to_split[p] == "val"}
    te = {p for p in out_perturbed if pert_to_split[p] == "test"}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "split label sets NOT disjoint"
    assert (tr | va | te) == out_perturbed, "split partition does not cover all present perturbations"
    logger.info(f"  [Phase 12] GUARDS PASS: {len(ctrl_units):,} single-cell controls; split partition "
                f"{len(tr)}/{len(va)}/{len(te)} disjoint+complete over {len(out_perturbed)} perts")

    reduction = round(adata.n_obs / am.n_obs, 3)
    am.uns["metacell_metadata"] = {
        "recipe": "control_preserving_mbk", "k_perturbed": k, "control_single_cell": True,
        "splits": "all (train+val+test), within-perturbation", "seed": seed,
        "n_input_cells": int(adata.n_obs), "n_units": int(am.n_obs), "n_ctrl_units": int(n_ctrl),
        "n_perturbed_units": int(n_perturbed_units), "cell_reduction_x": reduction,
        "n_genes": int(n_genes),
    }
    del adata
    force_gc(logger)

    # ── 6. write the all-splits substrate + aligned split JSON ──────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    allsplits_path = out_dir / f"{dataset_name}_allsplits_metacell_ctrlpreserved.h5ad"
    am.write_h5ad(allsplits_path)
    logger.info(f"  [Phase 12] ✓ all-splits substrate → {allsplits_path}  "
                f"({am.n_obs:,} units, {reduction}x fewer than input cells)")

    aligned = {
        "train_labels": [p for p in src_meta.get("train_labels", []) if str(p) in out_perturbed],
        "val_labels":   [p for p in src_meta.get("val_labels", [])   if str(p) in out_perturbed],
        "test_labels":  [p for p in src_meta.get("test_labels", [])  if str(p) in out_perturbed],
        "seed": int(src_meta.get("seed", seed)),
        "_source_split_json": str(split_json),
        "_note": ("Labels UNCHANGED from source: metacells keep their perturbation label. Control cells are "
                  "single-cell here and re-split proportionally by the downstream dataloader."),
    }
    split_out = out_dir / f"{dataset_name}_allsplits_metacell_split_indices.json"
    json.dump(aligned, open(split_out, "w"), indent=2)

    # ── 7. firewall-safe train substrate (split in {train, control}) ────────────────────────────────
    keep = np.isin(am.obs["split"].astype(str).to_numpy(), ["train", "control"])
    train_hybrid = am[keep].copy()
    leaked = int(train_hybrid.obs["split"].isin(["val", "test"]).sum())
    assert leaked == 0, f"FIREWALL BREACH: {leaked} val/test units in the train substrate"
    hyb_path = out_dir / f"{dataset_name}_train_hybrid.h5ad"
    train_hybrid.write_h5ad(hyb_path)
    n_hc = int((train_hybrid.obs["split"] == "control").sum())
    n_ht = int((train_hybrid.obs["split"] == "train").sum())
    logger.info(f"  [Phase 12] ✓ train hybrid (firewall-safe) → {hyb_path}  "
                f"({train_hybrid.n_obs:,} units = {n_hc:,} control SC + {n_ht:,} train metacells)")

    logger.info(f"  [Phase 12] complete in {time.perf_counter() - t0:.0f}s")
    return {
        "allsplits_metacell": str(allsplits_path),
        "train_hybrid":       str(hyb_path),
        "allsplits_split_json": str(split_out),
        "n_units":            int(am.n_obs),
        "n_genes":            int(n_genes),
        "cell_reduction_x":   reduction,
    }
