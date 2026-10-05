"""
go_enrichment.py -- per-TF regulon GO Biological Process enrichment via
g:Profiler (gprofiler-official), for obj_003_grn_bio_evaluator_v2.

Optional, activated only with --enable-go-enrichment. Caches each (TF,
sorted target list, organism) query to `intermediate/go_cache/` keyed by a
hash, so repeated runs over the same candidate/k don't re-hit the network --
g:Profiler is a shared public API and we should not hammer it on every
re-run during development.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _cache_key(tf: str, targets: list, organism: str, sources: list) -> str:
    payload = json.dumps({"tf": tf, "targets": sorted(targets), "organism": organism,
                           "sources": sorted(sources)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _query_one_tf(gp, tf: str, targets: list, go_config: dict, cache_dir: Path) -> dict:
    cache_path = cache_dir / f"{_cache_key(tf, targets, go_config['organism'], go_config['sources'])}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    result_row = {"n_targets": len(targets), "n_terms_significant": 0,
                  "top_term": None, "top_term_pval": 1.0}
    try:
        result = gp.profile(
            organism=go_config["organism"],
            query=targets,
            sources=go_config["sources"],
            significance_threshold_method=go_config["correction_method"],
            user_threshold=go_config["significance_threshold"],
            no_evidences=True,
        )
        if len(result):
            sig = result[result["significant"]] if "significant" in result.columns else result
            result_row["n_terms_significant"] = int(len(sig))
            result_row["top_term"] = str(result.iloc[0]["name"])
            result_row["top_term_pval"] = float(result.iloc[0]["p_value"])
    except Exception as exc:
        result_row = {"n_targets": len(targets), "error": str(exc)}

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result_row), encoding="utf-8")
    return result_row


def compute_go_enrichment(edges_df: pd.DataFrame, k: int, go_config: dict, cache_dir: Path) -> dict:
    """For each TF in the top-K GRN with >= go_config['min_regulon_size']
    targets, run g:Profiler GO:BP enrichment on the target gene list.
    Network failures are caught per-TF and recorded, never allowed to crash
    the whole evaluation -- per the design doc's Test 5 edge case."""
    top = edges_df.head(k)
    try:
        from gprofiler import GProfiler
    except ImportError:
        log("  GO enrichment: gprofiler-official not installed; skipping "
            "(pip install gprofiler-official).")
        return {"enabled": True, "error": "gprofiler_not_installed", "n_tfs_tested": 0,
                "n_tfs_enriched": 0, "fraction_tfs_enriched": float("nan"),
                "median_top_term_pval": float("nan"), "per_tf": {}}

    gp = GProfiler(return_dataframe=True)
    per_tf = {}
    n_network_errors = 0
    for tf, group in top.groupby("Regulator", sort=False, observed=True):
        targets = group["Target"].tolist()
        if len(targets) < go_config["min_regulon_size"]:
            continue
        row = _query_one_tf(gp, str(tf), targets, go_config, cache_dir)
        per_tf[str(tf)] = row
        if "error" in row:
            n_network_errors += 1

    scored = {tf: r for tf, r in per_tf.items() if "error" not in r}
    n_tested = len(scored)
    n_enriched = sum(1 for r in scored.values() if r["n_terms_significant"] > 0)
    med_pval = float(np.median([r["top_term_pval"] for r in scored.values()])) if scored else float("nan")

    pct_enriched = (n_enriched / n_tested * 100) if n_tested else 0.0
    log(f"  GO enrichment @k={k}: {n_tested} TFs tested (>={go_config['min_regulon_size']} "
        f"targets), {n_enriched} enriched ({pct_enriched:.1f}%), "
        f"{n_network_errors} network errors")

    return {
        "enabled": True,
        "n_tfs_tested": n_tested,
        "n_tfs_enriched": n_enriched,
        "fraction_tfs_enriched": n_enriched / n_tested if n_tested else float("nan"),
        "median_top_term_pval": med_pval,
        "n_network_errors": n_network_errors,
        "per_tf": per_tf,
    }
