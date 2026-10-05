"""
obj_003.2 Step 6 -- validate.py  (the end-to-end promotion gate for obj_003's GPU path)

Runs the cloned obj_003 v2.1 evaluator CPU vs GPU on real dense graphs and confirms:
  - stat_prec BIT-IDENTICAL (0 threshold-flips) at every k  -- preserves the additivity guarantee
  - wass_test within rtol 1e-6
  - sc_control_w25 stat_prec@1k reproduces the historical 0.991 baseline
  - GPU wall-time speedup vs CPU (stat stage; --skip-bio-metrics isolates the kernel)
Sequential; loads the gold/expr once (one evaluator instance reused for all runs).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
from pathlib import Path

import numpy as np

OBJ = Path(__file__).resolve().parents[1]
REPO = OBJ.parents[1]
sys.path.insert(0, str(OBJ / "clone" / "obj_003_src"))
from grn_eval_v2 import GRNEvaluatorV2  # noqa: E402  (cloned obj_003 v2.1 + GPU dispatch)

CFG = OBJ / "clone/cell_configs/rpe1.yaml"
PANEL = REPO / "DATA/SPORE_outputs/RPE1/processed/RPE1_5k_essential_train_metacell.h5ad"
E9 = REPO / "DATA/EXPERIMENTS/exp_009_metacell_aggregation_psgrn_w25/intermediate/pruned_graphs"
E7 = REPO / "DATA/EXPERIMENTS/exp_007_chitin_final_assessment/intermediate/pruned_graphs"
GRAPHS = {
    "sc_control_w25": E9 / "sc_control_w25.parquet",
    "mbk_k5_w25": E9 / "mbk_k5_w25.parquet",
    "chitin_2d_a05_w25": E7 / "chitin_2d_a05_w25.parquet",
}
TOPK = [1000, 5000, 10000, 25000, 50000, 100000]  # broadened acceptance gate: all 6 k


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(ev, pq, device):
    t = time.perf_counter()
    rep = ev.evaluate(grn_parquet=str(pq), topk=TOPK, panel_source=str(PANEL),
                      skip_bio_metrics=True, candidate_id="v", device=device)
    return {r["k"]: r for r in rep["by_k"]}, time.perf_counter() - t


def main():
    ev = GRNEvaluatorV2.from_config(str(CFG))
    out = {"graphs": {}, "all_pass": True}
    for name, pq in GRAPHS.items():
        log(f"=== {name} ===")
        cpu, t_cpu = run(ev, pq, "cpu")
        gpu, t_gpu = run(ev, pq, "cuda")
        g = {"speedup": round(t_cpu / t_gpu, 2), "t_cpu_s": round(t_cpu, 1), "t_gpu_s": round(t_gpu, 1),
             "by_k": {}}
        for k in TOPK:
            flips = "n/a"
            sp_cpu, sp_gpu = cpu[k]["stat_prec"], gpu[k]["stat_prec"]
            w_cpu, w_gpu = cpu[k]["wass_test"], gpu[k]["wass_test"]
            stat_identical = (sp_cpu == sp_gpu)  # stat_prec is a rational count/k -> must be exact
            wass_ok = abs(w_cpu - w_gpu) <= 1e-6 * (abs(w_cpu) + 1e-12)
            g["by_k"][k] = dict(stat_prec_cpu=sp_cpu, stat_prec_gpu=sp_gpu, stat_identical=stat_identical,
                                wass_cpu=round(w_cpu, 8), wass_gpu=round(w_gpu, 8), wass_ok=wass_ok)
            out["all_pass"] &= stat_identical and wass_ok
            log(f"  k={k}: stat_prec cpu={sp_cpu:.6f} gpu={sp_gpu:.6f} identical={stat_identical} | "
                f"wass rtol_ok={wass_ok} | speedup {g['speedup']}x")
        out["graphs"][name] = g

    # historical baseline check
    sc1k = out["graphs"]["sc_control_w25"]["by_k"][1000]["stat_prec_gpu"]
    out["baseline_sc_control_stat_prec_1k"] = sc1k
    out["baseline_ok"] = abs(sc1k - 0.991) < 1e-3
    out["all_pass"] &= out["baseline_ok"]
    log(f"baseline sc_control stat_prec@1k gpu={sc1k:.4f} (expect 0.991) -> {out['baseline_ok']}")

    (OBJ / "intermediate").mkdir(exist_ok=True)
    (OBJ / "intermediate/validate_obj003.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"\nALL PASS: {out['all_pass']}  | mean speedup: "
        f"{np.mean([g['speedup'] for g in out['graphs'].values()]):.2f}x")
    log(f"wrote {OBJ/'intermediate/validate_obj003.json'}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
