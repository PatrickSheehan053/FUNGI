"""
systema_graph_eval.py -- orchestrator (CLI + SystemaGraphEvaluator class + batch mode)
for obj_004_systema_graph_eval, the targeted systematic-variation evaluator, parallel
to obj_003's grn_eval_v2.py.

Reports, per top-K, the SYSTEMA edge-level decontamination metrics:
  spec_prec   -- regulator-SPECIFIC precision (R-knock vs GLOBAL-PERTURBED, not control)
  sysvar_cosine -- top-K mean cosine of regulator shift with the systematic axis
  sysvar_gap  -- stat_prec - spec_prec (the contamination fraction; needs --also-stat-prec)
  stat_prec   -- obj_003-equivalent (R-knock vs control), inline when --also-stat-prec

Leakage-safe: the same held-out test split obj_003 uses (test_size/random_state/stratify
from the cell-config), so centroids + tests use ONLY test cells. The normalized expression
matrix is read from the SHARED obj_003 expr_cache, so the cells are bit-identical.

Usage:
  python systema_graph_eval.py --grn-parquet <g.parquet> --cell-config <rpe1.yaml> \
    --output-json <out.json> [--topk 1000,5000,...] [--also-stat-prec] \
    [--input-prenormalized] [--candidate-id NAME]

Batch:
  python systema_graph_eval.py --batch <manifest.csv> --cell-config <rpe1.yaml> \
    --out-dir <dir> [--also-stat-prec] [--topk ...]      # manifest cols: candidate_id,grn_parquet[,input_prenormalized]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from systema_metrics import spec_prec_topk, stat_prec_topk, sysvar_cosine_topk  # noqa: E402

EVALUATOR_VERSION = "obj_004_v1.0"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_config(path):
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    sysb = raw.get("systema", {})
    return dict(
        cell_line=raw.get("cell_line", "?"), sc_input=raw["sc_input"],
        gold_dir=raw.get("gold_dir", "DATA/gold_standards"),
        pert_col=raw.get("pert_col", "gene"),
        control_label=raw.get("control_label", "non-targeting"),
        test_size=float(raw.get("split", {}).get("test_size", 0.2)),
        random_state=int(raw.get("split", {}).get("random_state", 0)),
        stat_p=float(raw.get("stat_metrics", {}).get("p_threshold", 0.05)),
        stat_min_cells=int(raw.get("stat_metrics", {}).get("min_cells", 3)),
        sys_min_cells=int(sysb.get("min_cells_per_perturbation", 25)),
        sys_p=float(sysb.get("p_threshold", 0.05)),
        cosine_on=sysb.get("cosine_on", "full_vector"),
        source_path=str(path))


def load_grn_edges(parquet_path, max_edges=None):
    """Identical column auto-detect + Importance-desc sort + dup check as obj_003.
    max_edges: if set, keep only the top-N by Importance BEFORE stringifying the
    Regulator/Target columns -- avoids a multi-GB string allocation on dense (25M-edge)
    graphs. Safe for top-K metrics as long as max_edges >= max(topk). Default None =
    obj_003-identical behavior (full graph)."""
    df = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in df.columns}
    reg = cols.get("regulator", df.columns[1])
    tgt = cols.get("target", df.columns[0])
    w = cols.get("importance", cols.get("weight", df.columns[2]))
    out = df[[reg, tgt, w]].copy()
    out.columns = ["Regulator", "Target", "Importance"]
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    if max_edges is not None and len(out) > max_edges:
        out = out.head(max_edges).copy()
    out["Regulator"] = out["Regulator"].astype(str)
    out["Target"] = out["Target"].astype(str)
    ndup = int(out.duplicated(subset=["Regulator", "Target"]).sum())
    if ndup:
        raise ValueError(f"{parquet_path}: {ndup} duplicate (Regulator,Target) rows.")
    return out


def _load_expr(sc_input, pert_col, gold_dir, prenorm):
    if prenorm:
        import anndata as ad
        a = ad.read_h5ad(sc_input)
        X = a.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, np.float32), {g: i for i, g in enumerate(a.var_names)}, \
            a.obs[pert_col].astype(str).to_numpy()
    # reuse obj_003's loader + cache (bit-identical normalized matrix). Production layout: obj_003's src.
    o3 = Path(__file__).resolve().parents[2] / "obj_003_grn_bio_evaluator_v2" / "src"
    sys.path.insert(0, str(o3))
    from grn_eval_v2 import load_singlecell_for_evaluation
    return load_singlecell_for_evaluation(sc_input, pert_col, Path(gold_dir) / "expr_cache")


def _guanlab_split(pert_values, test_size, random_state):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(pert_values))
    return train_test_split(idx, test_size=test_size, random_state=random_state,
                            stratify=pert_values)


class SystemaGraphEvaluator:
    """Stateful only in caching the test-split expression + centroids across candidate
    evals against the same cell-config (the obj_003 batch pattern)."""

    def __init__(self, cfg, input_prenormalized=False, device="cpu", cooldown_ms=0):
        self.cfg = cfg
        self.prenorm = input_prenormalized
        self.device = device  # obj_003.2 GPU stat kernel
        self.cooldown_ms = cooldown_ms
        self._gpu_ctx = None   # obj_003.2 resident-GPU reference cache (lazy)
        self._ready = False

    def _ensure_loaded(self):
        if self._ready:
            return
        c = self.cfg
        X, gene_to_col, pert = _load_expr(c["sc_input"], c["pert_col"], c["gold_dir"], self.prenorm)
        _, test_idx = _guanlab_split(pert, c["test_size"], c["random_state"])
        self._X_test = np.asarray(X[test_idx], dtype=np.float64)
        self._pert_test = np.asarray(pert)[test_idx]
        self._gene_to_col = gene_to_col
        ctrl = c["control_label"]
        ctrl_mask = self._pert_test == ctrl
        pert_mask = ~ctrl_mask
        self._X_ctrl_test = self._X_test[ctrl_mask]
        self._X_pert_test = self._X_test[pert_mask]                       # global perturbed cells
        self._mu_ctrl = self._X_ctrl_test.mean(axis=0)
        self._mu_pert = self._X_pert_test.mean(axis=0)                    # global perturbed centroid
        self._sys_shift = self._mu_pert - self._mu_ctrl                   # systematic axis
        # per-regulator shift s_R = mu_R - mu_ctrl (regulators with >= sys_min_cells)
        reg_shift = {}
        uniq = np.unique(self._pert_test)
        for r in uniq:
            if r == ctrl:
                continue
            m = self._pert_test == r
            if int(m.sum()) >= c["sys_min_cells"]:
                reg_shift[r] = self._X_test[m].mean(axis=0) - self._mu_ctrl
        self._reg_shift = reg_shift
        log(f"loaded: {len(self._pert_test):,} test cells "
            f"({int(ctrl_mask.sum()):,} control / {int(pert_mask.sum()):,} perturbed); "
            f"{len(reg_shift):,} regulators with >= {c['sys_min_cells']} test cells")
        self._ready = True

    def evaluate(self, grn_parquet, topk, candidate_id=None, also_stat_prec=False, max_edges=None):
        self._ensure_loaded()
        c = self.cfg
        edges = load_grn_edges(grn_parquet, max_edges=max_edges)
        graph_genes = frozenset(pd.unique(pd.concat([edges["Regulator"], edges["Target"]])))
        report = {"candidate": candidate_id or Path(grn_parquet).stem,
                  "evaluator_version": EVALUATOR_VERSION, "cell_config": c["cell_line"],
                  "n_edges_total": len(edges), "panel_size": len(graph_genes),
                  "distance": "mannwhitney", "split": {"test_size": c["test_size"],
                  "random_state": c["random_state"]}, "by_k": []}
        for k in topk:
            # spec_prec uses the SAME min_cells + p as stat_prec so sysvar_gap is a clean,
            # same-edge contamination measure (control-ref vs perturbed-ref on identical edges).
            # sys_min_cells (25) is used only for the cosine's per-regulator centroid (mu_R).
            if self.device == "cuda" and self._gpu_ctx is None:
                from systema_metrics import make_gpu_ctx
                self._gpu_ctx = make_gpu_ctx(self.device, self.cooldown_ms)
            sp = spec_prec_topk(edges, k, self._X_test, self._gene_to_col, self._pert_test,
                                self._X_pert_test, c["stat_min_cells"], c["stat_p"], device=self.device,
                                gpu_ctx=self._gpu_ctx)
            cos, n_cos = sysvar_cosine_topk(edges, k, self._reg_shift, self._sys_shift,
                                            c["cosine_on"], self._gene_to_col)
            row = {"k": k, "spec_prec": round(sp["prec"], 6), "sysvar_cosine": cos,
                   "n_scored": sp["n_scored"], "n_skipped_no_coverage": sp["n_skip_cov"],
                   "n_skipped_target_not_in_panel": sp["n_not_in_panel"],
                   "fraction_scored": sp["fraction_scored"], "n_regulators_in_cosine": n_cos}
            if also_stat_prec:
                st = stat_prec_topk(edges, k, self._X_test, self._gene_to_col, self._pert_test,
                                    self._X_ctrl_test, c["stat_min_cells"], c["stat_p"], device=self.device,
                                    gpu_ctx=self._gpu_ctx)
                row["stat_prec"] = round(st["prec"], 6)
                row["sysvar_gap"] = round(st["prec"] - sp["prec"], 6)
            report["by_k"].append(row)
            msg = f"k={k}: spec_prec={sp['prec']:.4f} sysvar_cos={cos if cos is None else round(cos,4)}"
            if also_stat_prec:
                msg += f" stat_prec={row['stat_prec']:.4f} sysvar_gap={row['sysvar_gap']:+.4f}"
            log(msg)
        return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grn-parquet")
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--output-json")
    ap.add_argument("--topk", default="1000,5000")
    ap.add_argument("--also-stat-prec", action="store_true")
    ap.add_argument("--input-prenormalized", action="store_true")
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--max-edges", type=int, default=None, help="keep top-N by Importance before stringify (dense-graph memory guard; must be >= max topk)")
    ap.add_argument("--batch", default=None, help="manifest CSV: candidate_id,grn_parquet[,input_prenormalized]")
    ap.add_argument("--out-dir", default=None)
    # obj_003.2 GPU stat kernel. Default cpu (byte-identical). cuda opts in; falls back if no GPU.
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--cooldown-ms", type=int, default=0, help="sleep between GPU batches (thermal guard)")
    args = ap.parse_args()
    cfg = load_config(args.cell_config)
    topk = [int(k) for k in args.topk.split(",")]

    device = args.device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                log("WARNING: --device cuda requested but no CUDA GPU available; falling back to cpu.")
                device = "cpu"
        except Exception as e:
            log(f"WARNING: torch/CUDA unavailable ({e}); falling back to cpu."); device = "cpu"

    if args.batch:
        man = pd.read_csv(args.batch)
        out_dir = Path(args.out_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        # group by prenorm so each loads its expression once
        for prenorm_val, grp in man.groupby(man.get("input_prenormalized", pd.Series([False] * len(man))).fillna(False).astype(bool)):
            ev = SystemaGraphEvaluator(cfg, input_prenormalized=bool(prenorm_val), device=device, cooldown_ms=args.cooldown_ms)
            for _, r in grp.iterrows():
                outp = out_dir / f"{r['candidate_id']}_systema_graph.json"
                if outp.exists():
                    log(f"SKIP {r['candidate_id']} (exists)"); continue
                rep = ev.evaluate(r["grn_parquet"], topk, candidate_id=r["candidate_id"],
                                  also_stat_prec=args.also_stat_prec, max_edges=args.max_edges)
                outp.write_text(json.dumps(rep, indent=2, default=str))
                log(f"wrote {outp}")
        return

    ev = SystemaGraphEvaluator(cfg, input_prenormalized=args.input_prenormalized, device=device, cooldown_ms=args.cooldown_ms)
    rep = ev.evaluate(args.grn_parquet, topk, candidate_id=args.candidate_id,
                      also_stat_prec=args.also_stat_prec, max_edges=args.max_edges)
    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, default=str))
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
