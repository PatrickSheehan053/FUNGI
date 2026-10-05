"""
grn_eval_v2.py -- orchestrator (CLI + GRNEvaluatorV2 class) for
obj_003_grn_bio_evaluator_v2, the canonical GRN evaluator going forward.

Reports stat_prec/wass_test/FOR@K (primary, causal, held-out) alongside
bio_prec_raw/AUPR/AUROC/EPR (secondary, database-overlap) and bio_prec_pbs
(diagnostic only) per evaluation -- see obj_003 design doc Purpose section.
obj_001_grn_bio_evaluator is not modified; this is a new, independent object.

Usage:
    python grn_eval_v2.py \
        --grn-parquet  <path/to/grn.parquet> \
        --cell-config  data/cell_configs/rpe1.yaml \
        --output-json  <path/to/results.json> \
        [--topk 1000,5000] [--panel-source <path/to/panel.h5ad>] \
        [--enable-go-enrichment] [--go-min-regulon-size 5] [--go-pval 0.05] \
        [--skip-stat-metrics] [--skip-bio-metrics] \
        [--candidate-id NAME] [--method NAME]

Python API:
    from grn_eval_v2 import GRNEvaluatorV2
    ev = GRNEvaluatorV2.from_config("data/cell_configs/rpe1.yaml")
    results = ev.evaluate(grn_parquet="path/to/grn.parquet", topk=[1000, 5000])
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bio_metrics
import stat_metrics
from cell_config import CellConfig, load_cell_config
from databases_v2 import DatabaseLoaderV2

EVALUATOR_VERSION = "obj_003_v2.2"  # v2.1 causal panel + obj_003.2 GPU stat kernel (--device cuda; cpu default byte-identical)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_grn_edges(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in df.columns}
    reg_col = cols.get("regulator", df.columns[1])
    tgt_col = cols.get("target", df.columns[0])
    w_col = cols.get("importance", cols.get("weight", df.columns[2]))
    out = df[[reg_col, tgt_col, w_col]].copy()
    out.columns = ["Regulator", "Target", "Importance"]
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    n_dupes = int(out.duplicated(subset=["Regulator", "Target"]).sum())
    if n_dupes:
        raise ValueError(f"{parquet_path}: {n_dupes} duplicate (Regulator, Target) rows.")
    return out


def _expr_cache_paths(sc_path: str, pert_col: str, cache_dir: Path) -> tuple:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(sc_path).stem
    return (cache_dir / f"{stem}_{pert_col}_X.npy",
            cache_dir / f"{stem}_{pert_col}_meta.npz")


def load_singlecell_for_evaluation(sc_path: str, pert_col: str, cache_dir: Path) -> tuple:
    x_path, meta_path = _expr_cache_paths(sc_path, pert_col, cache_dir)
    if x_path.exists() and meta_path.exists():
        log(f"Loading cached normalized expression matrix from {x_path}")
        X = np.load(x_path, mmap_mode="r")
        meta = np.load(meta_path, allow_pickle=True)
        gene_to_col = {g: i for i, g in enumerate(meta["gene_names"])}
        return X, gene_to_col, meta["pert_values"]

    log(f"Loading single-cell AnnData for evaluation from {sc_path}")
    adata = ad.read_h5ad(sc_path)
    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gene_names = np.asarray(adata.var_names, dtype=object)
    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    pert_values = adata.obs[pert_col].astype(str).to_numpy()

    log(f"Caching normalized expression matrix to {x_path}")
    np.save(x_path, X)
    np.savez(meta_path, gene_names=gene_names, pert_values=pert_values)
    return X, gene_to_col, pert_values


def guanlab_split(pert_values: np.ndarray, test_size: float, random_state: int) -> tuple:
    idx = np.arange(len(pert_values))
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=pert_values)


class GRNEvaluatorV2:
    """Stateful only in caching the single-cell matrix/split (expensive to
    reload) across multiple `evaluate()` calls against the same cell_config --
    this is what makes batch_eval.py's repeated calls fast."""

    def __init__(self, cell_config: CellConfig):
        self.cfg = cell_config
        self._X = None
        self._gene_to_col = None
        self._pert_values = None
        self._train_idx = None
        self._test_idx = None
        self._sc_ready = False
        self._panel_cache = {}
        self._gpu_ctx = None  # obj_003.2 resident-GPU reference cache (created lazily on --device cuda)

    @classmethod
    def from_config(cls, cell_config_path: str) -> "GRNEvaluatorV2":
        return cls(load_cell_config(cell_config_path))

    def _ensure_sc_loaded(self) -> bool:
        """Returns True if stat metrics are computable (sc_input staged)."""
        if self._sc_ready:
            return self._X is not None
        if self.cfg.sc_input is None:
            log("sc_input is null for this cell config -- stat metrics (stat_prec/"
                "wass_test/FOR@K/bio_prec_pbs) will be skipped; bio_prec_raw/AUPR/"
                "AUROC/EPR are unaffected.")
            self._sc_ready = True
            return False
        cache_dir = Path(self.cfg.gold_dir) / "expr_cache"
        X, gene_to_col, pert_values = load_singlecell_for_evaluation(
            self.cfg.sc_input, self.cfg.pert_col, cache_dir)
        train_idx, test_idx = guanlab_split(pert_values, self.cfg.split_test_size,
                                             self.cfg.split_random_state)
        log(f"GuanLab-style split: {len(train_idx):,} train cells / {len(test_idx):,} test "
            f"cells (test_size={self.cfg.split_test_size}, random_state={self.cfg.split_random_state})")
        self._X, self._gene_to_col, self._pert_values = X, gene_to_col, pert_values
        self._train_idx, self._test_idx = train_idx, test_idx
        # obj_003.2: materialize the test-cell matrices ONCE (stable objects) so (a) they aren't
        # recomputed per k, and (b) the gpu_context resident-reference cache uploads X_ctrl_test a
        # single time (keyed by object id). Identical values to the per-k slicing it replaces.
        self._X_test = self._X[self._test_idx]
        self._pert_test = self._pert_values[self._test_idx]
        self._X_ctrl_test = self._X_test[self._pert_test == self.cfg.control_label]
        self._sc_ready = True
        return True

    def _panel_genes_for(self, edges_df: pd.DataFrame, panel_source: str = None) -> frozenset:
        if panel_source:
            panel_adata = ad.read_h5ad(panel_source, backed="r")
            return frozenset(panel_adata.var_names)
        return frozenset(pd.unique(pd.concat([edges_df["Regulator"], edges_df["Target"]])))

    def _gold_standard_for_panel(self, panel_genes: frozenset) -> dict:
        key = panel_genes
        if key not in self._panel_cache:
            loader = DatabaseLoaderV2(self.cfg.databases, self.cfg.gold_dir)
            bio = loader.build_pooled(panel_genes)
            pbs_set, pbs_stats = None, None
            if self._ensure_sc_loaded():
                pbs_set, pbs_stats = loader.build_pbs(
                    bio["pooled"], self._X, self._gene_to_col, self._pert_values,
                    self._test_idx, self.cfg.control_label, self.cfg.stat_min_cells,
                    self.cfg.stat_p_threshold)
            # obj_003.1 causal-evidence panel: pure set algebra over per_db (no reload)
            cc = self.cfg.causal_config or {}
            causal = loader.build_causal_pool(
                bio["per_db"],
                cc.get("causal_databases", ["collectri", "trrust_v2", "knocktf", "chipseq"]),
                cc.get("coexpression_databases", ["string_network", "string_physical"]))
            self._panel_cache[key] = dict(pooled=bio["pooled"], per_db=bio["per_db"],
                                           sources_in_panel=bio["sources_in_panel"],
                                           pbs_set=pbs_set, pbs_stats=pbs_stats,
                                           n_panel=len(panel_genes),
                                           causal_pool=causal["causal_pool"],
                                           coexpr_pool=causal["coexpr_pool"],
                                           causal_per_db=causal["causal_per_db"],
                                           causal_sources=causal["causal_sources"])
        return self._panel_cache[key]

    def evaluate(self, grn_parquet: str, topk: list, panel_source: str = None,
                 enable_go_enrichment: bool = False, go_min_regulon_size: int = None,
                 go_pval: float = None, skip_stat_metrics: bool = False,
                 skip_bio_metrics: bool = False, candidate_id: str = None,
                 method: str = None, causal_panel: bool = True,
                 report_coverage: bool = True, motif: bool = False,
                 device: str = "cpu", cooldown_ms: int = 0) -> dict:
        edges_df = load_grn_edges(grn_parquet)
        if device == "cuda" and getattr(self, "_gpu_ctx", None) is None:
            self._gpu_ctx = stat_metrics.make_gpu_ctx(device, cooldown_ms)  # resident refs, once per evaluator
        panel_genes = self._panel_genes_for(edges_df, panel_source)
        graph_genes = frozenset(pd.unique(pd.concat([edges_df["Regulator"], edges_df["Target"]])))
        log(f"GRN: {len(edges_df):,} directed edges over {len(graph_genes):,} graph nodes; "
            f"gold-standard panel = {len(panel_genes):,} genes "
            f"({'fixed from ' + panel_source if panel_source else 'graph node set'})")

        gold = self._gold_standard_for_panel(panel_genes)
        stat_ready = self._ensure_sc_loaded() and not skip_stat_metrics

        go_config = dict(self.cfg.go_config) if self.cfg.go_config else {}
        go_config.setdefault("organism", "hsapiens")
        go_config.setdefault("sources", ["GO:BP"])
        go_config.setdefault("min_regulon_size", 5)
        go_config.setdefault("significance_threshold", 0.05)
        go_config.setdefault("correction_method", "fdr")
        if go_min_regulon_size is not None:
            go_config["min_regulon_size"] = go_min_regulon_size
        if go_pval is not None:
            go_config["significance_threshold"] = go_pval

        report = {
            "candidate": candidate_id or Path(grn_parquet).stem,
            "method": method,
            "panel_size": len(panel_genes),
            "n_edges_total": len(edges_df),
            "n_gold_pairs_in_panel": len(gold["pooled"]),
            "panel_filtering_source": panel_source if panel_source else "graph_node_set",
            "evaluator_version": EVALUATOR_VERSION,
            "cell_config": self.cfg.cell_line,
            "gold_standard_sources": gold["sources_in_panel"],
            "by_k": [],
        }

        full_aupr = bio_metrics.aupr(edges_df, gold["pooled"]) if not skip_bio_metrics else None
        full_auroc = bio_metrics.auroc(edges_df, gold["pooled"]) if not skip_bio_metrics else None

        # obj_003.1 causal-evidence panel: full-graph AUPR/AUROC vs the causal pool
        # (perturbation-independent, STRING-co-expression excluded). Purely additive.
        panel_on = causal_panel and not skip_bio_metrics and self.cfg.causal_config.get("enabled", True)
        causal_full_aupr = bio_metrics.aupr(edges_df, gold["causal_pool"]) if panel_on else None
        causal_full_auroc = bio_metrics.auroc(edges_df, gold["causal_pool"]) if panel_on else None
        # obj_003.1 Tier 2: cisTarget motif validator (perturbation- + literature-independent).
        # Independent of the database panel (needs only the graph + ranking DB), so it does NOT
        # require bio metrics -- runs whenever --motif is set and the resources are staged.
        motif_cfg = (self.cfg.causal_config or {}).get("motif", {}) or {}
        motif_on = motif
        if motif_on:
            rdb, m2tf = motif_cfg.get("ranking_db"), motif_cfg.get("motif2tf")
            if not (rdb and m2tf and Path(rdb).exists() and Path(m2tf).exists()):
                log(f"--motif requested but ranking_db/motif2tf not found (ranking_db={rdb}, "
                    f"motif2tf={m2tf}); skipping motif metrics. Stage the cisTarget resources.")
                motif_on = False
            else:
                import motif_metrics
                log(f"Tier 2 motif validator ON (nes>={motif_cfg.get('nes_threshold', 3.0)}, "
                    f"ranking_db={Path(rdb).name})")

        for k in topk:
            row = {"k": k}
            if stat_ready:
                # obj_003.2: use the cached, STABLE test-cell matrices (materialized once in
                # _ensure_sc_loaded) so the gpu_context reference-cache uploads X_ctrl_test one time.
                pert_test = self._pert_test
                X_test = self._X_test
                X_ctrl_test = self._X_ctrl_test
                sm = stat_metrics.compute_stat_metrics(
                    edges_df, k, X_test, self._gene_to_col, pert_test, X_ctrl_test,
                    self.cfg.stat_min_cells, self.cfg.stat_p_threshold, gold["pooled"],
                    self.cfg.control_label, device=device, gpu_ctx=getattr(self, "_gpu_ctx", None))
                row.update(sm)
                log(f"k={k}: stat_prec={sm['stat_prec']:.4f} wass_test={sm['wass_test']:.4f} "
                    f"for_k={sm['for_k']:.4f} fraction_scored={sm['fraction_scored']:.4f}")
            else:
                row.update(dict(stat_prec=None, wass_test=None, for_k=None,
                                 for_n_evaluable=None, fraction_scored=None))

            if not skip_bio_metrics:
                bio = bio_metrics.biological_topk(edges_df, gold["pooled"], k)
                row["bio_prec_raw"] = bio["precision"]
                row["bio_prec_recall"] = bio["recall"]
                row["bio_prec_f1"] = bio["f1"]
                row["aupr"] = full_aupr
                row["auroc"] = full_auroc
                row["epr_k"] = bio_metrics.epr_topk(edges_df, gold["pooled"],
                                                     len(panel_genes), k)["epr"]
                row["per_db"] = bio_metrics.biological_topk_perdb(edges_df, gold["per_db"], k)
                if gold["pbs_set"] is not None:
                    pbs_bio = bio_metrics.biological_topk(edges_df, gold["pbs_set"], k)
                    row["bio_prec_pbs"] = pbs_bio["precision"]
                else:
                    row["bio_prec_pbs"] = None
                log(f"k={k}: bio_prec_raw={bio['precision']:.4f} bio_prec_pbs="
                    f"{row['bio_prec_pbs'] if row['bio_prec_pbs'] is None else round(row['bio_prec_pbs'], 4)} "
                    f"aupr={full_aupr:.6f} auroc={full_auroc if full_auroc is None else round(full_auroc, 4)}")
            else:
                row.update(dict(bio_prec_raw=None, bio_prec_recall=None, bio_prec_f1=None,
                                 aupr=None, auroc=None, epr_k=None, per_db=None, bio_prec_pbs=None))

            # ---- obj_003.1 causal-evidence panel (Tier 1) — appended, additive ----
            if panel_on:
                cz = bio_metrics.biological_topk(edges_df, gold["causal_pool"], k)
                row["causal_prec"] = cz["precision"]
                row["causal_epr"] = bio_metrics.epr_topk(edges_df, gold["causal_pool"],
                                                          len(panel_genes), k)["epr"]
                row["causal_aupr"] = causal_full_aupr
                row["causal_auroc"] = causal_full_auroc
                row["causal_per_db"] = bio_metrics.biological_topk_perdb(
                    edges_df, gold["causal_per_db"], k)
                # renamed context view of bio_prec_raw, but vs STRING co-expression ONLY
                row["coexpr_prec"] = bio_metrics.biological_topk(edges_df, gold["coexpr_pool"], k)["precision"]
                if report_coverage:
                    cov = bio_metrics.causal_coverage(edges_df, gold["causal_sources"], k)
                    row["causal_coverage"] = cov["causal_coverage"]
                    row["causal_n_covered"] = cov["n_covered"]
                    # stat_prec's coverage (fraction_scored) surfaced alongside for direct compare
                    row["stat_prec_coverage"] = row.get("fraction_scored")
                log(f"k={k}: causal_prec={cz['precision']:.4f} coexpr_prec={row['coexpr_prec']:.4f} "
                    f"causal_cov={row.get('causal_coverage')} stat_cov={row.get('stat_prec_coverage')}")

            # ---- obj_003.1 Tier 2 motif validator (optional) ----
            if motif_on:
                mm = motif_metrics.motif_topk(
                    edges_df, k, motif_cfg["ranking_db"], motif_cfg["motif2tf"],
                    nes_threshold=float(motif_cfg.get("nes_threshold", 3.0)))
                row["motif_prec"] = mm["motif_prec"]
                row["motif_coverage"] = mm["motif_coverage"]
                row["motif_n_validated_regulons"] = mm["n_validated_regulons"]
                log(f"k={k}: motif_prec={mm['motif_prec']:.4f} motif_cov={mm['motif_coverage']:.4f} "
                    f"validated_regulons={mm['n_validated_regulons']}")

            if enable_go_enrichment:
                import go_enrichment
                cache_dir = Path(__file__).resolve().parent.parent / "intermediate" / "go_cache"
                row["go_enrichment"] = go_enrichment.compute_go_enrichment(edges_df, k, go_config, cache_dir)

            report["by_k"].append(row)

        if gold["pbs_stats"] is not None:
            report["pbs_stats"] = gold["pbs_stats"]

        # obj_003.1: per-graph panel summary (the three axes side by side + coverage honesty)
        if panel_on:
            report["panel_summary"] = {
                "causal_pool_size": len(gold["causal_pool"]),
                "coexpr_pool_size": len(gold["coexpr_pool"]),
                "n_causal_tf_sources": len(gold["causal_sources"]),
                "causal_databases": self.cfg.causal_config.get("causal_databases"),
                "by_k": [{"k": r["k"], "stat_prec": r.get("stat_prec"),
                          "causal_prec": r.get("causal_prec"), "coexpr_prec": r.get("coexpr_prec"),
                          "causal_coverage": r.get("causal_coverage"),
                          "stat_prec_coverage": r.get("stat_prec_coverage")} for r in report["by_k"]],
                # axis_agreement compares TWO graphs' rankings (stat_prec vs causal_prec); it is
                # undefined for a single graph. exp_013 / the de-bias test compute it across graphs.
                "axis_agreement": None,
                "axis_agreement_note": "cross-graph metric; not defined for a single graph "
                                        "(compare candidates' stat_prec-ranking vs causal_prec-ranking).",
            }
        return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grn-parquet", required=True)
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--topk", default="1000,5000")
    ap.add_argument("--panel-source", default=None)
    ap.add_argument("--enable-go-enrichment", action="store_true")
    ap.add_argument("--go-min-regulon-size", type=int, default=None)
    ap.add_argument("--go-pval", type=float, default=None)
    ap.add_argument("--skip-stat-metrics", action="store_true")
    ap.add_argument("--skip-bio-metrics", action="store_true")
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--method", default=None)
    # obj_003.1 causal-evidence panel (v2.1). Default ON; purely additive.
    ap.add_argument("--causal-panel", action=argparse.BooleanOptionalAction, default=True,
                    help="compute the causal_pool metrics (default ON; --no-causal-panel to disable)")
    ap.add_argument("--report-coverage", action=argparse.BooleanOptionalAction, default=True,
                    help="emit per-metric scoreable-edge coverage (default ON)")
    ap.add_argument("--motif", action="store_true", help="Tier 2 motif_prec (not built in v2.1 clone)")
    # obj_003.2 GPU stat kernel. Default cpu (byte-identical to production). cuda opts into the GPU
    # MWU/Wasserstein kernels (validated: 0 stat_prec threshold-flips); falls back to cpu if no GPU.
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--cooldown-ms", type=int, default=0, help="sleep between GPU batches (thermal guard)")
    args = ap.parse_args()

    device = args.device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                log("WARNING: --device cuda requested but no CUDA GPU available; falling back to cpu.")
                device = "cpu"
        except Exception as e:
            log(f"WARNING: torch/CUDA unavailable ({e}); falling back to cpu.")
            device = "cpu"

    topk = [int(k) for k in args.topk.split(",")]
    ev = GRNEvaluatorV2.from_config(args.cell_config)
    t_eval = time.perf_counter()
    report = ev.evaluate(
        grn_parquet=args.grn_parquet, topk=topk, panel_source=args.panel_source,
        enable_go_enrichment=args.enable_go_enrichment,
        go_min_regulon_size=args.go_min_regulon_size, go_pval=args.go_pval,
        skip_stat_metrics=args.skip_stat_metrics, skip_bio_metrics=args.skip_bio_metrics,
        candidate_id=args.candidate_id, method=args.method,
        causal_panel=args.causal_panel, report_coverage=args.report_coverage, motif=args.motif,
        device=device, cooldown_ms=args.cooldown_ms)
    report["_gpu_meta"] = {"device": device, "wall_time_s": round(time.perf_counter() - t_eval, 2)}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
