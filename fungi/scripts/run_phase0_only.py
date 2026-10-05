"""
Standalone Phase 0 runner -- mirrors FUNGI_plus.ipynb cells 2/3/5/6/8/10/11
exactly, for inspecting probe confidence on a candidate dense graph before
committing to a full headless Phase 0-7 notebook run. Same convention as
session 3's _run_phase0_only.py (built, used, deleted -- recreated here for
session 4's SHROOM candidates).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

SRC_DIR = str(Path.cwd() / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--config", default="fungi_config.yaml")
    parser.add_argument("--no-override", action="store_true",
                         help="Run probes raw, ignoring fungi_config.yaml's bounds_override section")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    fungi_mode = cfg.get("fungi_mode", "organic").lower()
    assert fungi_mode == "organic", "this standalone runner only mirrors the organic-mode branch"

    sc_data_path = Path(cfg["input"]["sc_data_path"])
    adata = sc.read_h5ad(str(sc_data_path))
    n_genes = adata.n_vars
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
    print(f"Expression : {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    graph_path = Path(args.graph_path)
    gdf = pd.read_parquet(str(graph_path))
    cols = {column.lower(): column for column in gdf.columns}
    src_col = cols.get("regulator", cols.get("source", gdf.columns[0]))
    tgt_col = cols.get("target", gdf.columns[1])
    w_col = cols.get("importance", cols.get("weight", gdf.columns[2]))
    print(f"Columns: source='{src_col}' target='{tgt_col}' weight='{w_col}'")

    rows = gdf[src_col].map(gene_to_idx).to_numpy()
    cols_idx = gdf[tgt_col].map(gene_to_idx).to_numpy()
    vals = gdf[w_col].to_numpy(dtype=np.float64)
    mask = ~(pd.isna(rows) | pd.isna(cols_idx))
    raw_sparse_mat = sp.csr_matrix(
        (vals[mask], (rows[mask].astype(int), cols_idx[mask].astype(int))),
        shape=(n_genes, n_genes))
    raw_sparse_mat.eliminate_zeros()
    print(f"Parent GRN : {n_genes:,} genes | {raw_sparse_mat.nnz:,} edges | "
          f"density {raw_sparse_mat.nnz / n_genes**2:.4%}")
    if (~mask).any():
        print(f"  dropped {int((~mask).sum()):,} edges with genes not in adata.var_names")

    from diagnostics import run_diagnostics
    bounds_override_cfg = None if args.no_override else cfg.get("bounds_override")
    utopian_bounds, loss_weights, diagnostic_report = run_diagnostics(
        adata=adata,
        n_genes=n_genes,
        cfg_diagnostics=cfg["diagnostics"],
        cfg_input=cfg["input"],
        raw_sparse_mat=raw_sparse_mat,
        lambda_user_cfg=cfg.get("lambda_user"),
        bounds_override_cfg=bounds_override_cfg)

    print("\nUtopian bounds + loss weights + raw confidence:")
    raw_confidences = diagnostic_report.get("raw_confidences", {})
    probes_used = diagnostic_report.get("probes_used", {})
    for key in sorted(utopian_bounds):
        bound = utopian_bounds[key]
        weight = loss_weights.get(key, float("nan"))
        confidence = raw_confidences.get(key, float("nan"))
        probe = probes_used.get(key, "?")
        print(f"  {key:10s} [{bound[0]:.4f}, {bound[1]:.4f}]  weight={weight:.3f}  "
              f"conf={confidence:.3f}  probe={probe}")

    print(f"\nlam_eff={diagnostic_report.get('lam_eff')}  "
          f"lam_q25={diagnostic_report.get('lam_q25')}  lam_q75={diagnostic_report.get('lam_q75')}")
    print(f"proceed={diagnostic_report.get('proceed')}")

    out = {k: v for k, v in diagnostic_report.items() if not str(k).startswith("_")}
    out_path = Path(args.graph_path).with_suffix("").name + (
        "_phase0_no_override.json" if args.no_override else "_phase0_with_override.json")
    out_path = Path("../DATA/FUNGI_outputs") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    print(f"Saved diagnostic report to {out_path}")


if __name__ == "__main__":
    main()
