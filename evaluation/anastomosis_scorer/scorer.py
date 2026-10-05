"""
obj_010 ANASTOMOSIS — scorer.py : orchestrate one tag  (adapter -> ScoredUnit -> panel -> record -> upsert).
Crash-resilient: each seed's JSON written immediately; CSV upsert is atomic (temp-file + os.replace).
Device-agnostic (mean tiers are CPU here; GPU tiers wire in via gpu_kernels when run_cell + device=cuda).
"""
from __future__ import annotations
import os, sys, json, time, hashlib, tempfile
from pathlib import Path
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import io_adapters as IO
import panel as PANEL
import metrics_v3f as M3

OBJECT_VERSION = "obj_010_v1"
PROV_COLS = ["object_version", "generation", "dataset", "substrate", "input_kind", "device",
             "n_perts_scored", "n_genes", "coverage_frac", "seed", "split", "config_hash",
             "source_path", "timestamp"]


def _cfg_hash(d): return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _zfit_from_npz(true_path):
    """Build the train-only co-expression baseline Z for coexpr_resid (RHIZO contract npz has train_LFC)."""
    z = np.load(true_path, allow_pickle=True)
    return M3.build_coexpr_Z(np.asarray(z["train_LFC"], np.float64)) if "train_LFC" in z.files else None


def _atomic_upsert_csv(row: dict, csv_path: Path, key_cols):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.read_parquet(csv_path) if csv_path.suffix == ".parquet" else pd.read_csv(csv_path)
        # drop any existing row with the same key, then append (idempotent upsert)
        mask = np.ones(len(df), bool)
        for k in key_cols:
            if k in df.columns: mask &= (df[k].astype(str) == str(row.get(k)))
        df = pd.concat([df[~mask], df_new], ignore_index=True)
    else:
        df = df_new
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=str(csv_path.parent), suffix=".tmp", newline="")
    df.to_csv(tmp.name, index=False); tmp.close(); os.replace(tmp.name, csv_path)


class Anastomosis:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scoring = cfg.get("scoring", {})
        self.device = self.scoring.get("device", "cpu")
        self.out = Path(cfg["paths"]["output_dir"]); (self.out / "json").mkdir(parents=True, exist_ok=True)
        self.csv = self.out / cfg["paths"].get("results_csv", "anastomosis_results.csv")

    def _run_meta(self, run):
        return {k: run.get(k) for k in ("tag", "input_kind", "path", "pred_path", "generation",
                                        "substrate", "n_seeds", "notes")}

    def score_run(self, run: dict, device=None) -> list:
        """Score one run entry (config `runs[]`). Returns list of per-seed/split record dicts (also upserted)."""
        device = device or self.device
        kind = run["input_kind"]; tag = run["tag"]; rows = []
        run_cell = self.scoring.get("run_cell_tiers", "auto")
        splits = run.get("splits", ["test"])
        if kind == "mean_lfc":
            true_path = run["path"]; pred_path = run.get("pred_path")
            zfit = _zfit_from_npz(true_path) if self.scoring.get("coexpr_resid", True) else None
            for split in splits:
                try:
                    u = IO.load_mean_lfc(true_path, split=split, pred_path=pred_path)
                except Exception as e:
                    print(f"[skip] {tag}/{split}: {e}"); continue
                rc = False if run_cell in (False, "off") else (run_cell is True)  # mean_lfc: cell tiers off unless forced (they'll skip)
                res, meta = PANEL.run_panel(u, Zfit=zfit, run_cell=rc, device=device)
                rows.append(self._record(run, res, meta, u, seed=run.get("seed", 0), split=split, device=device,
                                          source_path=true_path))
        elif kind == "per_cell":
            from glob import glob
            d = Path(run["path"])
            seeds = sorted({int(Path(p).stem.split("_")[1]) for p in glob(str(d / "true_*.h5ad"))})
            for seed in seeds:
                try:
                    u = IO.load_per_cell(d, seed, self.cfg["dataset"].get("perturbation_col", "gene"),
                                         self.cfg["dataset"].get("control_label", "non-targeting"))
                except Exception as e:
                    print(f"[skip] {tag}/seed{seed}: {e}"); continue
                rc = True if run_cell in (True, "auto") else False
                res, meta = PANEL.run_panel(u, Zfit=None, run_cell=rc, device=device)
                rows.append(self._record(run, res, meta, u, seed=seed, split="test", device=device,
                                          source_path=str(d)))
        else:
            raise ValueError(f"unknown input_kind {kind}")
        return rows

    def _record(self, run, res, meta, u, seed, split, device, source_path):
        prov = {"object_version": OBJECT_VERSION, "generation": run.get("generation", "unknown"),
                "dataset": self.cfg.get("dataset", {}).get("name", "?"), "substrate": run.get("substrate", "?"),
                "input_kind": run["input_kind"], "device": device, "n_perts_scored": u.n_pert,
                "n_genes": u.n_gene, "coverage_frac": round(u.coverage_frac, 4), "seed": seed, "split": split,
                "config_hash": _cfg_hash(self._run_meta(run)), "source_path": str(source_path),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        row = {**prov, "tag": run["tag"], **{k: (float(v) if np.isscalar(v) and np.isfinite(v) else v)
                                             for k, v in res.items()},
               "skipped_cell_tiers": meta["skipped_cell_tiers"], "is_self": meta["is_self"]}
        # write per-seed JSON immediately (crash-resilient)
        jpath = self.out / "json" / f"{run['tag']}_s{seed}_{split}.json"
        json.dump({"provenance": prov, "metrics": res, "meta": meta}, open(jpath, "w"), indent=2, default=float)
        _atomic_upsert_csv(row, self.csv, key_cols=["tag", "seed", "split", "object_version"])
        return row

    def score_tag(self, tag: str, device=None) -> list:
        run = next((r for r in self.cfg.get("runs", []) if r["tag"] == tag), None)
        if run is None: raise KeyError(f"tag {tag} not in config runs")
        return self.score_run(run, device=device)
