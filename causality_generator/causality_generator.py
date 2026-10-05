"""
SHROOM: PSGRN-SelfTrain with GPU pseudolabels (production, promoted from the
shroom_2.py candidate, session 4, 23 June 2026).

Replaces the previous production SHROOM (the batched elastic-net/FISTA
method, archived unchanged as shroom_1.py) after a head-to-head FUNGI
comparison: shroom_1.py's champion scored utopia loss 7.13 with 1/6 organic
topology targets and a wrong-signed (positive) assortativity; this method's
champion scored 0.25 with 3/6 targets and the first correctly-signed
(negative) assortativity this project has produced on any SHROOM-family
substrate. Full investigation, including two negative-result attempts at a
GPU PyTorch PSGRN-Causal candidate (shroom_4.py, kept on disk) and a
per-target-LightGBM candidate (shroom_3.py, kept on disk, ~22h/run, not
fully run): markdowns/claude_code/claude_code_session_4.md.

Song et al. 2026 self-training formulation: one global LightGBM/XGBoost binary
classifier trained on all directed gene pairs against correlation-derived
pseudolabels, then re-scored by the same model. Four features per directed pair
(gi -> gj): control-mean expression of gi and gj, the perturbed-cell mean of gi
when gi itself was knocked down, and the perturbed-cell mean of gj specifically
in gi's knockdown cells (NaN when gi was never perturbed). Pseudolabel
correlation matrix computed on GPU (PyTorch): a single control-only matmul for
non-perturbed sources, then per-perturbed-source row overwrites using a
matched control + interventional cell sample. Output: [Target, Regulator,
Importance], same column convention as the archived shroom_1.py.

No 2_psgrn_selftrain.py exists on this machine to adapt -- that file only ever
lived on the (currently offline) HPC. This is a fresh build from the
architecture spec in markdowns/handoffs/Handoff - 30MAY2026.md Part 2 and the
GPU pseudolabel code in the session-4 SHROOM-Eval-3 deep research reports.

Session 6 (23 June 2026) note: this file stays the unmodified production
champion throughout the session-6 SHROOM_2-family fidelity bake-off. Three
candidate variants are kept as separate sibling files, each a full copy of
this one plus exactly one isolated change, matching this project's existing
shroom_1/3/4.py convention rather than flag-gating everything onto the
champion:
  - shroom_2a.py: --ablation correlation_only (skip the classifier entirely,
    score by raw pseudolabel correlation -- replicates the paper's own
    "Baseline" comparator).
  - shroom_2c.py: --coverage-feature (adds a 5th, explicit
    source_has_coverage feature).
  - shroom_2d.py: --self-train-rounds N (multi-round self-training, an
    explicit deviation from the paper's single-round design).
See markdowns/claude_code/claude_code_session_6.md for the audit and
rationale behind each.
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pick_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.clip(values, 0.0, None)
    if np.allclose(values.sum(), 0.0):
        return 0.0
    values = np.sort(values)
    count = values.size
    index = np.arange(1, count + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * values) / (count * np.sum(values))) - (count + 1.0) / count)


def load_singlecell_panel(sc_path: str, hvg_genes: list, pert_col: str) -> tuple:
    log(f"Loading single-cell AnnData from {sc_path}")
    adata = ad.read_h5ad(sc_path)
    log(f"  Full panel: {adata.n_obs:,} cells x {adata.n_vars:,} genes; "
        f"normalizing (median target_sum) before HVG subset. Note: this deliberately differs "
        f"from SPORE_light's own Phase 9 (target_sum=10000, fixed CP10K) -- this file is the "
        f"pre-Phase-9 raw split, so SHROOM's own normalize_total call is the only normalization "
        f"ever applied to it, not a second pass on top of SPORE_light's. The target_sum CHOICE "
        f"itself (median vs. a fixed 10000) is a single global multiplicative constant applied "
        f"uniformly to every cell, so it has no effect on the z-scored Path A/B correlation "
        f"pseudolabels or on LightGBM/XGBoost's tree splits (both invariant to that kind of "
        f"affine rescaling) -- session 6 audit, see claude_code_session_6.md.")
    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)

    var_set = set(adata.var_names)
    keep_genes = [gene for gene in hvg_genes if gene in var_set]
    if len(keep_genes) != len(hvg_genes):
        missing = len(hvg_genes) - len(keep_genes)
        log(f"  WARNING: {missing} HVG panel genes not found in single-cell var_names; dropping them.")
    adata = adata[:, keep_genes].copy()

    X_norm = adata.X
    if hasattr(X_norm, "toarray"):
        X_norm = X_norm.toarray()
    X_norm = np.asarray(X_norm, dtype=np.float32, order="C")
    gene_names = np.asarray(adata.var_names, dtype=object)
    pert_values = adata.obs[pert_col].astype(str).to_numpy()
    return X_norm, gene_names, pert_values


def build_pert_map(pert_values: np.ndarray, control_label: str, gene_lookup: dict, min_cells: int) -> dict:
    pert_map = {}
    skipped_not_in_panel = 0
    skipped_too_few = 0
    for gene in np.unique(pert_values):
        if gene == control_label:
            continue
        if gene not in gene_lookup:
            skipped_not_in_panel += 1
            continue
        idx = np.flatnonzero(pert_values == gene)
        if idx.size < min_cells:
            skipped_too_few += 1
            continue
        pert_map[gene] = idx
    log(f"Perturbation map: {len(pert_map)} usable source genes "
        f"({skipped_not_in_panel} not in HVG panel, {skipped_too_few} below min_cells={min_cells})")
    return pert_map


@torch.no_grad()
def compute_pseudolabels_and_features_gpu(
    X_norm: np.ndarray,
    gene_names: np.ndarray,
    pert_map: dict,
    ctrl_idx: np.ndarray,
    corr_threshold: float,
    device: torch.device,
    seed: int,
) -> dict:
    n_genes = len(gene_names)
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}

    ctrl_idx = np.asarray(ctrl_idx, dtype=np.int64)
    n_ctrl_total = ctrl_idx.size
    log(f"Path A: control-only correlation matrix ({n_ctrl_total:,} control cells x {n_genes:,} genes)")

    X_ctrl = torch.as_tensor(X_norm[ctrl_idx], device=device, dtype=torch.float32)
    obs_mean = X_ctrl.mean(dim=0)
    ctrl_std = X_ctrl.std(dim=0, unbiased=False)
    ctrl_std_safe = torch.where(ctrl_std < 1e-8, torch.ones_like(ctrl_std), ctrl_std)
    X_ctrl_std = (X_ctrl - obs_mean) / ctrl_std_safe

    corr_abs = (X_ctrl_std.T @ X_ctrl_std) / max(1, X_ctrl_std.shape[0])
    corr_abs = corr_abs.abs()
    corr_abs.fill_diagonal_(0.0)

    obs_mean_np = obs_mean.detach().cpu().numpy().astype(np.float32, copy=False)
    del X_ctrl, X_ctrl_std, ctrl_std, ctrl_std_safe, obs_mean
    if device.type == "cuda":
        torch.cuda.empty_cache()

    int_mean_self = np.zeros(n_genes, dtype=np.float32)
    int_mean_other = np.full((n_genes, n_genes), np.nan, dtype=np.float32)

    rng = np.random.default_rng(seed)
    n_pert = len(pert_map)
    log(f"Path B: overwriting correlation rows + interventional means for {n_pert} perturbed source genes")
    t_start = time.perf_counter()

    for step, (gene, int_idx) in enumerate(pert_map.items(), start=1):
        src_i = gene_lookup[gene]
        int_idx = np.asarray(int_idx, dtype=np.int64)
        n_int = int_idx.size
        n_sample = min(n_int, n_ctrl_total)
        sampled_ctrl = rng.choice(ctrl_idx, size=n_sample, replace=False)
        eff_idx = np.concatenate([sampled_ctrl, int_idx])

        eff = torch.as_tensor(X_norm[eff_idx], device=device, dtype=torch.float32)
        interv_only = eff[n_sample:]
        interv_mean = interv_only.mean(dim=0).detach().cpu().numpy()
        int_mean_other[src_i] = interv_mean
        int_mean_self[src_i] = float(interv_mean[src_i])

        source_vals = eff[:, src_i]
        source_centered = source_vals - source_vals.mean()
        source_norm = torch.linalg.norm(source_centered)
        if source_norm < 1e-8:
            del eff, interv_only, source_vals, source_centered
            continue

        target_centered = eff - eff.mean(dim=0, keepdim=True)
        target_norm = torch.linalg.norm(target_centered, dim=0)
        target_norm_safe = torch.where(target_norm < 1e-8, torch.ones_like(target_norm), target_norm)
        row_corr = torch.abs(
            (source_centered[:, None] * target_centered).sum(dim=0) / (source_norm * target_norm_safe)
        )
        row_corr[src_i] = 0.0
        corr_abs[src_i, :] = row_corr

        del eff, interv_only, source_vals, source_centered, target_centered, target_norm, target_norm_safe, row_corr

        if step % 100 == 0 or step == n_pert:
            elapsed = time.perf_counter() - t_start
            log(f"  Path B: {step}/{n_pert} source genes processed ({elapsed:.1f}s elapsed)")

    labels = (corr_abs > corr_threshold).to(torch.uint8)
    labels.fill_diagonal_(0)

    for alt_threshold in (0.05, corr_threshold, 0.15):
        positive_rate = float((corr_abs > alt_threshold).float().mean().item())
        log(f"  Pseudolabel positive rate at threshold={alt_threshold:.2f}: {positive_rate:.4%}")

    return dict(
        labels=labels.detach().cpu().numpy(),
        corr_abs=corr_abs.detach().cpu().numpy().astype(np.float32, copy=False),
        obs_mean=obs_mean_np,
        int_mean_self=int_mean_self,
        int_mean_other=int_mean_other,
    )


def build_training_arrays(pseudo: dict, n_genes: int) -> tuple:
    off_diag = ~np.eye(n_genes, dtype=bool)
    src_idx, tgt_idx = np.where(off_diag)

    obs_mean = pseudo["obs_mean"]
    int_mean_self = pseudo["int_mean_self"]

    feature_matrix = np.empty((src_idx.size, 4), dtype=np.float32)
    feature_matrix[:, 0] = np.broadcast_to(obs_mean.reshape(-1, 1), (n_genes, n_genes))[off_diag]
    feature_matrix[:, 1] = np.broadcast_to(obs_mean.reshape(1, -1), (n_genes, n_genes))[off_diag]
    feature_matrix[:, 2] = np.broadcast_to(int_mean_self.reshape(-1, 1), (n_genes, n_genes))[off_diag]
    feature_matrix[:, 3] = pseudo["int_mean_other"][off_diag]

    label_vector = pseudo["labels"][off_diag]
    return feature_matrix, label_vector, src_idx, tgt_idx


def train_classifier(feature_matrix: np.ndarray, label_vector: np.ndarray, backend: str,
                      num_threads: int, n_estimators: int, num_leaves: int, max_depth: int,
                      min_data_in_leaf: int, learning_rate: float, seed: int) -> tuple:
    if backend == "lgbm_cpu":
        import lightgbm as lgb
        params = dict(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            learning_rate=learning_rate,
            min_gain_to_split=0.01,
            num_threads=num_threads,
            verbose=-1,
            seed=seed,
            device_type="cpu",
        )
        dtrain = lgb.Dataset(feature_matrix, label=label_vector, free_raw_data=False)
        model = lgb.train(params, dtrain, num_boost_round=n_estimators)
        return "lgbm", model

    if backend == "xgb_gpu":
        import xgboost as xgb
        dtrain = xgb.QuantileDMatrix(feature_matrix, label_vector, missing=np.nan)
        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            device="cuda",
            tree_method="hist",
            max_depth=max_depth,
            eta=learning_rate,
            min_child_weight=1.0,
            seed=seed,
        )
        model = xgb.train(params, dtrain, num_boost_round=n_estimators)
        return "xgb", model

    raise ValueError(f"Unknown classifier backend: {backend}")


def predict_all(model_kind: str, model, feature_matrix: np.ndarray) -> np.ndarray:
    if model_kind == "lgbm":
        return model.predict(feature_matrix).astype(np.float32, copy=False)
    if model_kind == "xgb":
        import xgboost as xgb
        dmatrix = xgb.QuantileDMatrix(feature_matrix, missing=np.nan)
        return model.predict(dmatrix).astype(np.float32, copy=False)
    raise ValueError(model_kind)


def calibration_stats(importance: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray, n_genes: int) -> dict:
    out_degree = np.bincount(src_idx, minlength=n_genes)
    in_degree = np.bincount(tgt_idx, minlength=n_genes)
    positive = importance[importance > 0.0]
    return {
        "total_edges": int(importance.size),
        "density": float(importance.size / (n_genes * (n_genes - 1))),
        "gini": gini_coefficient(positive) if positive.size else 0.0,
        "max_out_degree": int(out_degree.max()) if n_genes > 0 else 0,
        "all_targets_present": bool(np.all(in_degree >= 1)),
        "all_regulators_present": bool(np.all(out_degree >= 1)),
    }


def write_dense_parquet(importance: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray,
                         gene_names: np.ndarray, output_path: str, min_importance: float) -> dict:
    if min_importance > 0.0:
        keep = importance > min_importance
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]
        importance = importance[keep]

    gene_names_obj = np.asarray(gene_names, dtype=object)
    df = pd.DataFrame({
        "Target": pd.Categorical.from_codes(tgt_idx.astype(np.int32), categories=gene_names_obj),
        "Regulator": pd.Categorical.from_codes(src_idx.astype(np.int32), categories=gene_names_obj),
        "Importance": importance.astype(np.float64, copy=False),
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return calibration_stats(importance, src_idx, tgt_idx, len(gene_names))


def print_calibration_report(calib: dict) -> None:
    print("\n" + "=" * 65)
    print(" SHROOM CALIBRATION REPORT")
    print("=" * 65)
    print(f"  Edge count        : {calib['total_edges']:,}")
    print(f"  Density           : {calib['density'] * 100:.2f}%")
    print(f"  Gini coefficient  : {calib['gini']:.4f}")
    print(f"  Max out-degree    : {calib['max_out_degree']}")
    print("-" * 65)
    checks = {
        "Gini >= 0.3": calib["gini"] >= 0.3,
        "Max out-degree >= 50": calib["max_out_degree"] >= 50,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("=" * 65 + "\n")


def pseudolabel_checkpoint_file(output_path: str, checkpoint_dir: str) -> Path:
    stem = Path(output_path).stem
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.shroom2_pseudolabels.npz"


def model_checkpoint_file(output_path: str, checkpoint_dir: str, backend: str) -> Path:
    stem = Path(output_path).stem
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "txt" if backend == "lgbm_cpu" else "json"
    return out_dir / f"{stem}.shroom2_model_{backend}.{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="SHROOM: PSGRN-SelfTrain with GPU pseudolabels")
    parser.add_argument("--mc-input", required=True, help="Metacell .h5ad (source of the HVG gene panel/order)")
    parser.add_argument("--sc-input", required=True, help="Single-cell .h5ad (raw counts, perturbation labels)")
    parser.add_argument("--output", required=True, help="Output dense graph parquet path")
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--corr-threshold", type=float, default=0.1)
    parser.add_argument("--min-pert-cells", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                         help="Device for the GPU pseudolabel correlation stage")
    parser.add_argument("--classifier", type=str, default="lgbm_cpu", choices=["lgbm_cpu", "xgb_gpu"])
    parser.add_argument("--num-threads", type=int, default=6, help="LightGBM CPU thread count (6 = physical cores)")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--num-leaves", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-data-in-leaf", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-importance", type=float, default=0.0,
                         help="Drop predicted edges at or below this probability before writing parquet")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                         help="Directory for checkpoint files (default: <output_dir>/checkpoints)")
    parser.add_argument("--resume", action="store_true",
                         help="Reuse an existing pseudolabel/model checkpoint if present")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                         help="Use first --benchmark-genes HVG genes only, for a fast end-to-end smoke test")
    parser.add_argument("--benchmark-genes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = pick_device(args.device)

    log(f"Reading HVG gene panel from {args.mc_input}")
    mc_adata = ad.read_h5ad(args.mc_input, backed="r")
    hvg_genes = mc_adata.var_names.tolist()
    if args.benchmark:
        hvg_genes = hvg_genes[:args.benchmark_genes]
        log(f"Benchmark mode: restricting to first {len(hvg_genes)} HVG genes.")

    t_start_all = time.perf_counter()
    t0 = time.perf_counter()
    X_norm, gene_names, pert_values = load_singlecell_panel(args.sc_input, hvg_genes, args.pert_col)
    n_genes = len(gene_names)
    log(f"Data ready: {X_norm.shape[0]:,} cells x {n_genes:,} genes "
        f"(load+normalize: {time.perf_counter() - t0:.1f}s)")

    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    ctrl_idx = np.flatnonzero(pert_values == args.control_label)
    if ctrl_idx.size == 0:
        raise ValueError(f"No control cells found for control-label='{args.control_label}'")
    pert_map = build_pert_map(pert_values, args.control_label, gene_lookup, args.min_pert_cells)

    checkpoint_dir = args.checkpoint_dir or str(Path(args.output).resolve().parent / "checkpoints")
    pseudo_ckpt = pseudolabel_checkpoint_file(args.output, checkpoint_dir)

    if args.resume and pseudo_ckpt.exists():
        log(f"Resuming pseudolabels from checkpoint {pseudo_ckpt}")
        cached = np.load(pseudo_ckpt, allow_pickle=True)
        pseudo = {key: cached[key] for key in ("labels", "corr_abs", "obs_mean", "int_mean_self", "int_mean_other")}
    else:
        t0 = time.perf_counter()
        pseudo = compute_pseudolabels_and_features_gpu(
            X_norm, gene_names, pert_map, ctrl_idx, args.corr_threshold, device, args.seed)
        log(f"Pseudolabel stage finished in {time.perf_counter() - t0:.1f}s")
        np.savez(pseudo_ckpt, **pseudo)
        log(f"Saved pseudolabel checkpoint to {pseudo_ckpt}")

    t0 = time.perf_counter()
    feature_matrix, label_vector, src_idx, tgt_idx = build_training_arrays(pseudo, n_genes)
    positive_rate = float(label_vector.mean())
    log(f"Built training arrays: {feature_matrix.shape[0]:,} pairs, "
        f"positive rate={positive_rate:.4%} ({time.perf_counter() - t0:.1f}s)")

    model_ckpt = model_checkpoint_file(args.output, checkpoint_dir, args.classifier)
    if args.resume and model_ckpt.exists():
        log(f"Resuming trained classifier from checkpoint {model_ckpt}")
        if args.classifier == "lgbm_cpu":
            import lightgbm as lgb
            model_kind, model = "lgbm", lgb.Booster(model_file=str(model_ckpt))
        else:
            import xgboost as xgb
            model_kind, model = "xgb", xgb.Booster()
            model.load_model(str(model_ckpt))
    else:
        t0 = time.perf_counter()
        log(f"Training {args.classifier} classifier "
            f"(num_leaves={args.num_leaves}, max_depth={args.max_depth}, "
            f"n_estimators={args.n_estimators})")
        model_kind, model = train_classifier(
            feature_matrix, label_vector, args.classifier, args.num_threads,
            args.n_estimators, args.num_leaves, args.max_depth,
            args.min_data_in_leaf, args.learning_rate, args.seed)
        log(f"Classifier training finished in {time.perf_counter() - t0:.1f}s")
        if model_kind == "lgbm":
            model.save_model(str(model_ckpt))
        else:
            model.save_model(str(model_ckpt))
        log(f"Saved model checkpoint to {model_ckpt}")

    t0 = time.perf_counter()
    importance = predict_all(model_kind, model, feature_matrix)
    log(f"Scored {importance.size:,} pairs in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    calib = write_dense_parquet(importance, src_idx, tgt_idx, gene_names, args.output, args.min_importance)
    log(f"Wrote parquet in {time.perf_counter() - t0:.1f}s")

    elapsed_all = time.perf_counter() - t_start_all
    metrics = {
        "n_cells": int(X_norm.shape[0]),
        "n_genes": int(n_genes),
        "n_pert_source_genes": len(pert_map),
        "corr_threshold": float(args.corr_threshold),
        "pseudolabel_positive_rate": positive_rate,
        "classifier": args.classifier,
        "n_estimators": int(args.n_estimators),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(args.max_depth),
        "device": str(device),
        "elapsed_seconds": float(elapsed_all),
        "edges_written": calib["total_edges"],
        "density": calib["density"],
        "importance_gini": calib["gini"],
        "max_out_degree": calib["max_out_degree"],
        "all_targets_present": calib["all_targets_present"],
        "all_regulators_present": calib["all_regulators_present"],
    }
    stem = Path(args.output).with_suffix("")
    metrics_file = stem.with_name(stem.name + ".shroom2_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    log(f"Wrote metrics JSON to {metrics_file}")

    if args.calibrate:
        print_calibration_report(calib)
        calib_file = stem.with_name(stem.name + ".shroom2_calibration.json")
        with open(calib_file, "w", encoding="utf-8") as handle:
            json.dump(calib, handle, indent=2)
        log(f"Wrote calibration JSON to {calib_file}")

    log(
        f"Wrote parquet to {args.output} with {calib['total_edges']:,} edges, "
        f"density={calib['density']:.4f}, gini={calib['gini']:.4f}, "
        f"all_targets_present={calib['all_targets_present']}, "
        f"all_regulators_present={calib['all_regulators_present']}"
    )


if __name__ == "__main__":
    main()
