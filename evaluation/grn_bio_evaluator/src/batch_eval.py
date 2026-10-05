"""
batch_eval.py -- batch re-evaluation mode for obj_003_grn_bio_evaluator_v2.

Reads a candidates.json (same schema as SHROOM/shroom_bakeoff.py /
exp_003b/candidates.json), evaluates each candidate with a single shared
GRNEvaluatorV2 instance (so the multi-GB expression cache and the per-panel
gold standard/PBS set are each built at most once per distinct
panel_size/panel_source, not once per candidate), and writes a flat
results_summary.csv with one row per (candidate, k).

candidates.json schema:
[
  {"candidate_id": "shroom_2a_5000", "method": "Pearson correlation (SHROOM_2a)",
   "panel_size": 5000, "coverage_pct": 17.4, "role": "production_baseline",
   "parquet": "DATA/SHROOM_outputs/shroom_2a_5000.parquet",
   "panel_source": null}   # panel_source optional; falls back to graph node set
]

Resume/upsert: if --output-csv already exists, candidates already present
(keyed by candidate_id + k) are skipped, so re-running after a partial batch
does not duplicate rows. Sequential only -- no parallelism (GPU/CPU
contention with concurrent pipeline stages is a real constraint elsewhere
in this project; this evaluator is CPU-only but still runs sequentially for
consistency and because per-candidate log files would otherwise interleave).
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grn_eval_v2 import GRNEvaluatorV2, log

CSV_COLUMNS = [
    "candidate", "method", "panel_size", "coverage_pct", "role", "k",
    "stat_prec", "wass_test", "stat_prec_fraction_scored", "for_k", "for_n_evaluable",
    "bio_prec_raw", "bio_prec_recall", "bio_prec_f1", "bio_prec_pbs",
    "aupr", "auroc", "epr_k",
    "prec_string_network", "prec_string_physical", "prec_chipseq",
    "prec_collectri", "prec_knocktf", "prec_trrust_v2",
    "go_fraction_tfs_enriched", "go_median_top_pval",
    "n_gold_pairs_in_panel", "n_pbs", "fraction_evaluable_for_pbs",
    "evaluator_version",
]


def _report_to_rows(report: dict, candidate_meta: dict) -> list:
    rows = []
    pbs_stats = report.get("pbs_stats") or {}
    for entry in report["by_k"]:
        per_db = entry.get("per_db") or {}
        go = entry.get("go_enrichment") or {}
        rows.append({
            "candidate": report["candidate"], "method": report.get("method"),
            "panel_size": report.get("panel_size"), "coverage_pct": candidate_meta.get("coverage_pct"),
            "role": candidate_meta.get("role"), "k": entry["k"],
            "stat_prec": entry.get("stat_prec"), "wass_test": entry.get("wass_test"),
            "stat_prec_fraction_scored": entry.get("fraction_scored"),
            "for_k": entry.get("for_k"), "for_n_evaluable": entry.get("for_n_evaluable"),
            "bio_prec_raw": entry.get("bio_prec_raw"), "bio_prec_recall": entry.get("bio_prec_recall"),
            "bio_prec_f1": entry.get("bio_prec_f1"), "bio_prec_pbs": entry.get("bio_prec_pbs"),
            "aupr": entry.get("aupr"), "auroc": entry.get("auroc"), "epr_k": entry.get("epr_k"),
            "prec_string_network": per_db.get("string_network"),
            "prec_string_physical": per_db.get("string_physical"),
            "prec_chipseq": per_db.get("chipseq"),
            "prec_collectri": per_db.get("collectri"),
            "prec_knocktf": per_db.get("knocktf"),
            "prec_trrust_v2": per_db.get("trrust_v2"),
            "go_fraction_tfs_enriched": go.get("fraction_tfs_enriched"),
            "go_median_top_pval": go.get("median_top_term_pval"),
            "n_gold_pairs_in_panel": report.get("n_gold_pairs_in_panel"),
            "n_pbs": pbs_stats.get("n_pbs"), "fraction_evaluable_for_pbs": pbs_stats.get("fraction_evaluable"),
            "evaluator_version": report.get("evaluator_version"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--topk", default="1000,5000")
    ap.add_argument("--enable-go-enrichment", action="store_true")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    with open(args.candidates_json, encoding="utf-8") as f:
        candidates = json.load(f)
    topk = [int(k) for k in args.topk.split(",")]
    base_dir = Path(args.candidates_json).resolve().parent

    out_path = Path(args.output_csv)
    log_dir = Path(args.log_dir) if args.log_dir else out_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=CSV_COLUMNS)
    done_keys = set(zip(existing.get("candidate", []), existing.get("k", [])))
    log(f"Loaded {len(existing)} existing rows from {out_path} "
        f"({len(done_keys)} (candidate, k) keys already done)")

    ev = GRNEvaluatorV2.from_config(args.cell_config)
    new_rows = []
    for cand in candidates:
        cid = cand["candidate_id"]
        pending_ks = [k for k in topk if (cid, k) not in done_keys]
        if not pending_ks:
            log(f"Skipping {cid}: all k values already present in {out_path}")
            continue

        parquet_path = cand["parquet"]
        if not Path(parquet_path).is_absolute() and not Path(parquet_path).exists():
            parquet_path = str((base_dir / parquet_path).resolve())
        panel_source = cand.get("panel_source")
        if panel_source and not Path(panel_source).is_absolute() and not Path(panel_source).exists():
            panel_source = str((base_dir / panel_source).resolve())

        log_path = log_dir / f"{cid}.log"
        log(f"=== Evaluating {cid} (k={pending_ks}) -> {log_path} ===")
        with open(log_path, "a", encoding="utf-8") as lf, contextlib.redirect_stdout(lf):
            report = ev.evaluate(
                grn_parquet=parquet_path, topk=pending_ks, panel_source=panel_source,
                enable_go_enrichment=args.enable_go_enrichment,
                candidate_id=cid, method=cand.get("method"))
        rows = _report_to_rows(report, cand)
        new_rows.extend(rows)
        log(f"{cid}: done ({len(rows)} rows)")

    if new_rows and len(existing):
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    elif new_rows:
        combined = pd.DataFrame(new_rows)
    else:
        combined = existing
    combined = combined.drop_duplicates(subset=["candidate", "k"], keep="last")
    combined = combined[CSV_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    log(f"Wrote {len(combined)} total rows ({len(new_rows)} new) -> {out_path}")


if __name__ == "__main__":
    main()
