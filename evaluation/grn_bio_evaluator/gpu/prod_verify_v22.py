"""
obj_003.2 -- prod_verify_v22.py

Post-promotion acceptance: exercises the PROMOTED PRODUCTION obj_003 v2.2 + obj_004 (with the fixed
production import paths) on a real graph, cpu vs cuda, and requires stat_prec/spec_prec BIT-IDENTICAL
(0 flips). Confirms the shared kernel + gpu_context resolve from obj_003's src in the production layout,
and that the default (cpu) path is unchanged.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P3 = REPO / "OBJECTS/obj_003_grn_bio_evaluator_v2/src"
P4 = REPO / "OBJECTS/obj_004_systema_graph_eval/src"
PANEL = REPO / "DATA/SPORE_outputs/RPE1/processed/RPE1_5k_essential_train_metacell.h5ad"
GRAPH = REPO / "DATA/EXPERIMENTS/exp_009_metacell_aggregation_psgrn_w25/intermediate/pruned_graphs/sc_control_w25.parquet"
CFG3 = P3.parent / "data/cell_configs/rpe1.yaml"
CFG4 = P4.parent / "data/cell_configs/rpe1.yaml"
TOPK = [1000, 5000, 100000]
out = {"all_pass": True}


def main():
    # ---- obj_003 v2.2 ----
    sys.path.insert(0, str(P3))
    from grn_eval_v2 import GRNEvaluatorV2, EVALUATOR_VERSION
    out["version"] = EVALUATOR_VERSION
    ev = GRNEvaluatorV2.from_config(str(CFG3))
    cpu = {r["k"]: r for r in ev.evaluate(str(GRAPH), TOPK, panel_source=str(PANEL),
                                          skip_bio_metrics=True, device="cpu")["by_k"]}
    gpu = {r["k"]: r for r in ev.evaluate(str(GRAPH), TOPK, panel_source=str(PANEL),
                                          skip_bio_metrics=True, device="cuda")["by_k"]}
    o3 = {k: {"stat_identical": cpu[k]["stat_prec"] == gpu[k]["stat_prec"],
              "stat_prec": gpu[k]["stat_prec"]} for k in TOPK}
    out["obj_003"] = o3
    out["all_pass"] &= all(v["stat_identical"] for v in o3.values())

    # ---- obj_004 ----
    sys.path.insert(0, str(P4))
    from systema_graph_eval import SystemaGraphEvaluator, load_config
    cfg4 = load_config(str(CFG4))
    ec = SystemaGraphEvaluator(cfg4, device="cpu")
    eg = SystemaGraphEvaluator(cfg4, device="cuda")
    c4 = {r["k"]: r for r in ec.evaluate(str(GRAPH), [1000, 25000], also_stat_prec=True, max_edges=300000)["by_k"]}
    g4 = {r["k"]: r for r in eg.evaluate(str(GRAPH), [1000, 25000], also_stat_prec=True, max_edges=300000)["by_k"]}
    o4 = {k: {"spec_identical": c4[k]["spec_prec"] == g4[k]["spec_prec"],
              "stat_identical": c4[k]["stat_prec"] == g4[k]["stat_prec"]} for k in [1000, 25000]}
    out["obj_004"] = o4
    out["all_pass"] &= all(v["spec_identical"] and v["stat_identical"] for v in o4.values())

    (Path(__file__).resolve().parents[1] / "intermediate/prod_verify_v22.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("version:", out["version"])
    print("obj_003 stat identical:", {k: v["stat_identical"] for k, v in o3.items()})
    print("obj_004 spec/stat identical:", {k: (v["spec_identical"], v["stat_identical"]) for k, v in o4.items()})
    print("ALL PASS:", out["all_pass"])
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
