"""
obj_010 ANASTOMOSIS — anastomosis.py : CLI dispatch (score | firesale) + --legend + --list.
  python src/anastomosis.py score    --config data/anastomosis_config_rpe1.yaml [--tag TAG|--all|--list|--legend] [--device cpu|cuda]
  python src/anastomosis.py firesale --config data/firesale.yaml [--device cpu|cuda] [--resume] [--score]
"""
from __future__ import annotations
import os, sys, argparse, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

LEGEND = {
    "error": "mae/rmse/mse/l2 on delta (mean tier)",
    "correlation": "pearson_delta, cosine_delta, rsc (== arbiter RSC, Fisher-mean spearman over signal genes), coexpr_resid (co-expression-residualized partial corr, obj_009.3)",
    "deg_recovery": "f1_at_k sweep [10..1000] + f1_auc (rank overlap of |delta| top-k)",
    "reference_insensitive": "pearson/spearman/rmse @ top-20 true DEGs",
    "systema": "systematic_variation (dataset), pearson_systema (perturbed-centroid ref), centroid_accuracy",
    "distribution": "energy_distance, MMD  [CELL-ONLY; NaN+skip in Mode B]",
    "cell_wilcoxon": "wilcoxon AUPRC  [CELL-ONLY; NaN+skip in Mode B]",
    "provenance": "object_version, generation, dataset, substrate, input_kind, device, n_perts_scored, n_genes, coverage_frac, seed, split, config_hash, source_path, timestamp",
}


def main():
    ap = argparse.ArgumentParser(prog="anastomosis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--config", required=True)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--tag"); g.add_argument("--all", action="store_true"); g.add_argument("--list", action="store_true")
    s.add_argument("--legend", action="store_true"); s.add_argument("--device", default=None)
    f = sub.add_parser("firesale"); f.add_argument("--config", required=True); f.add_argument("--device", default="cpu")
    f.add_argument("--resume", action="store_true"); f.add_argument("--score", action="store_true"); f.add_argument("--only-kind", default=None)
    a = ap.parse_args()
    import yaml
    if a.cmd == "score":
        if a.legend:
            for k, v in LEGEND.items(): print(f"  {k:22s} {v}")
            return 0
        cfg = yaml.safe_load(open(a.config))
        from scorer import Anastomosis
        an = Anastomosis(cfg)
        if a.list:
            for r in cfg.get("runs", []): print(f"  {r['tag']:24s} kind={r['input_kind']:9s} gen={r.get('generation','?')} path={r.get('path')}")
            return 0
        tags = [r["tag"] for r in cfg.get("runs", [])] if a.all else ([a.tag] if a.tag else [])
        if not tags: print("nothing to score (give --tag/--all/--list/--legend)"); return 1
        for t in tags:
            rows = an.score_tag(t, device=a.device)
            for r in rows: print(f"{t}/{r['split']}: rsc={r.get('rsc_mean')} coexpr={r.get('coexpr_resid_mean')} cov={r['coverage_frac']} skip_cell={r['skipped_cell_tiers']}")
        return 0
    else:
        import firesale
        sys.argv = ["firesale", "--config", a.config, "--device", a.device] + (["--resume"] if a.resume else []) + (["--score"] if a.score else []) + (["--only-kind", a.only_kind] if a.only_kind else [])
        return firesale.main()


if __name__ == "__main__":
    sys.exit(main())
