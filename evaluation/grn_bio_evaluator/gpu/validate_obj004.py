"""
obj_003.2 Step 6b -- validate_obj004.py

obj_004 CPU vs GPU on a real graph: spec_prec + stat_prec must be BIT-IDENTICAL (0 flips), and the
GPU speedup measured (obj_004 is the ~12 min/graph bottleneck; its ~28k-cell perturbed reference is
where the GPU kernel should win biggest).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
from pathlib import Path

OBJ = Path(__file__).resolve().parents[1]
REPO = OBJ.parents[1]
sys.path.insert(0, str(OBJ / "clone" / "obj_004_src"))
from systema_graph_eval import SystemaGraphEvaluator, load_config  # noqa: E402

CFG = REPO / "OBJECTS/obj_004_systema_graph_eval/data/cell_configs/rpe1.yaml"
E9 = REPO / "DATA/EXPERIMENTS/exp_009_metacell_aggregation_psgrn_w25/intermediate/pruned_graphs"
E7 = REPO / "DATA/EXPERIMENTS/exp_007_chitin_final_assessment/intermediate/pruned_graphs"
GRAPHS = {"sc_control_w25": E9 / "sc_control_w25.parquet",
          "mbk_k5_w25": E9 / "mbk_k5_w25.parquet",
          "chitin_2d_a05_w25": E7 / "chitin_2d_a05_w25.parquet"}
TOPK = [1000, 5000, 10000, 25000, 50000, 100000]  # broadened acceptance gate: 3 graphs x all 6 k


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cfg, device, pq):
    ev = SystemaGraphEvaluator(cfg, device=device)
    t = time.perf_counter()
    rep = ev.evaluate(str(pq), TOPK, candidate_id="v", also_stat_prec=True, max_edges=300000)
    return {r["k"]: r for r in rep["by_k"]}, time.perf_counter() - t


def main():
    cfg = load_config(str(CFG))
    out = {"graphs": {}, "all_pass": True}
    for name, pq in GRAPHS.items():
        log(f"=== {name} ===")
        cpu, t_cpu = run(cfg, "cpu", pq)
        gpu, t_gpu = run(cfg, "cuda", pq)
        g = {"speedup": round(t_cpu / t_gpu, 2), "t_cpu_s": round(t_cpu, 1), "t_gpu_s": round(t_gpu, 1), "by_k": {}}
        for k in TOPK:
            spec_id = (cpu[k]["spec_prec"] == gpu[k]["spec_prec"])
            stat_id = (cpu[k]["stat_prec"] == gpu[k]["stat_prec"])
            gap_id = (cpu[k]["sysvar_gap"] == gpu[k]["sysvar_gap"])
            g["by_k"][k] = dict(spec_identical=spec_id, stat_identical=stat_id, sysvar_gap_identical=gap_id,
                                spec_cpu=cpu[k]["spec_prec"], spec_gpu=gpu[k]["spec_prec"])
            out["all_pass"] &= spec_id and stat_id and gap_id
            log(f"  k={k}: spec id={spec_id} stat id={stat_id} gap id={gap_id} "
                f"(spec cpu={cpu[k]['spec_prec']:.6f} gpu={gpu[k]['spec_prec']:.6f})")
        out["graphs"][name] = g
        log(f"  {name}: speedup {g['speedup']}x (cpu {g['t_cpu_s']}s -> gpu {g['t_gpu_s']}s)")
    import numpy as np
    (OBJ / "intermediate").mkdir(exist_ok=True)
    (OBJ / "intermediate/validate_obj004.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"ALL PASS: {out['all_pass']} | mean speedup "
        f"{np.mean([g['speedup'] for g in out['graphs'].values()]):.2f}x")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
