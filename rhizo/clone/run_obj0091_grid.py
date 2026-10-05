"""
obj_009.1 — run_obj0091_grid.py : detached overnight orchestrator (HYPHAE-abort armed, spill-guarded, resumable,
DECISION-POINT-FIRST). Serial (freeze-safe: never overlaps CPU-heavy + GPU-heavy work). Order is chosen so the
CORE deliverable (K=4 gap + collapse checks + P2@K4 for EVERY variant) lands FIRST; the K-shape {2,6} fills in
after. Every ablation_v2 invocation is itself resumable (skips completed (variant,rung,K,arm,seed)); re-running
the orchestrator resumes exactly where it stopped.

Steps:
  0 HYPHAE-abort · 1 graph caches (CPU) · 2 psi(0)=0 GATE (CPU) · 3 spill probe (GPU, unless caps exist)
  4 PASS A (K=4, decision point) for base variants: fungi/top_weight @2 seeds + nulls @1 seed
  5 select v_full composition from the K=4 ΔGAP · 6 PASS A for v_full
  7 PASS B (K-shape {2,6}) fungi/top_weight @2 seeds, ALL variants
  8 V7/V8 feat-shuffle controls (K=4) · 9 summarize + verify_v2
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTHONUTF8", "1")
import sys, json, time, glob, subprocess, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
PY = sys.executable
RES = EXP / "results"
INTER = EXP / "intermediate"
LOG = RES / "obj0091_grid.log"
BASE_VARIANTS = ["v1_baseline", "v_edge", "v_film", "v_tele", "v_dir"]
ALL_VARIANTS = BASE_VARIANTS + ["v_full"]
NULL_ARMS = "shuffle,empty,reverse,labelperm"
KGAP_ARMS = "fungi_bio,top_weight"


def log(m):
    line = f"[grid {time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def hyphae_landed():
    hits = []
    for pat in ["HYPHAE/for_chinmaya/output*", "HYPHAE/**/*dense*graph*"]:
        for p in glob.glob(str(ROOT / pat), recursive=True):
            if "input" in p.lower() or "backup" in p.lower():
                continue
            hits.append(p)
    return hits


def abort_if_hyphae(stage):
    hits = hyphae_landed()
    if hits:
        log(f"HYPHAE LANDED {hits[:2]} -> ABORT before: {stage}")
        json.dump({"aborted_before": stage, "hits": hits}, open(RES / "ABORTED_hyphae.json", "w"), indent=2)
        sys.exit(3)


def sh(args):
    log(f"RUN {' '.join(str(a) for a in args)}")
    with open(LOG, "a", encoding="utf-8") as f:
        r = subprocess.run([PY] + [str(a) for a in args], stdout=f, stderr=subprocess.STDOUT)
    log(f"  -> exit {r.returncode}")
    return r.returncode


def abl(variant, arms, Ks, seeds, tag="grid", extra=None):
    a = [HERE / "ablation_v2.py", "--variant", variant, "--tag", tag, "--arms", arms, "--Ks", Ks, "--seeds", seeds]
    if extra:
        a += extra
    return sh(a)


def full_variant(variant, seeds2, seeds_null, vfull_feats=None):
    """K=4 FIRST (decision point), then the K-shape {2,6}; nulls @ K4 for V5/P2. Priority variants."""
    extra = ["--vfull_feats", json.dumps(vfull_feats)] if vfull_feats is not None else None
    abl(variant, KGAP_ARMS, "4,2,6", seeds2, extra=extra)   # ablation loops Ks in order -> K4 lands first
    abl(variant, NULL_ARMS, "4", seeds_null, extra=extra)


def k4_variant(variant, seeds2, seeds_null, vfull_feats=None):
    """K=4 decision point ONLY (for the slow variants v_dir/v_full at their spill-safe batch 8)."""
    extra = ["--vfull_feats", json.dumps(vfull_feats)] if vfull_feats is not None else None
    abl(variant, KGAP_ARMS, "4", seeds2, extra=extra)
    abl(variant, NULL_ARMS, "4", seeds_null, extra=extra)


def select_vfull():
    sh([HERE / "ablation_v2.py", "--summarize", "--tag", "grid"])
    import pandas as pd, yaml
    cfg = yaml.safe_load(open(EXP / "configs" / "obj_009_1.yaml"))
    feat_of = {"v_edge": "edge", "v_film": "film", "v_dir": "dir", "v_tele": "tele"}
    chosen = {"edge": False, "film": False, "dir": False, "tele": False}
    reasons = {}
    try:
        vv = pd.read_csv(RES / "obj_009_1_v1_vs_v2.csv")
        p1 = pd.read_csv(RES / "obj_009_1_P1_summary.csv")
        for variant, feat in feat_of.items():
            sub = vv[(vv.variant == variant) & (vv.K == 4)]
            if not len(sub):
                reasons[variant] = "no K4 rows"; continue
            positive = bool((sub.delta_gap > 0).all())   # ΔGAP up on BOTH rungs at K=4
            if variant == "v_dir":
                bad = False
                for rung in ["rung1", "rung2"]:
                    rv = p1[(p1.variant == "v_dir") & (p1.rung == rung) & (p1.arm == "reverse") & (p1.K == 4)]
                    fu = p1[(p1.variant == "v_dir") & (p1.rung == rung) & (p1.arm == "fungi_bio") & (p1.K == 4)]
                    if len(rv) and len(fu) and float(rv.rsc.iloc[0]) >= 0.5 * max(float(fu.rsc.iloc[0]), 1e-9):
                        bad = True
                if bad:
                    positive = False; reasons[variant] = "EXCLUDED: reverse does not collapse (symmetrization risk)"
            chosen[feat] = positive
            reasons.setdefault(variant, f"K4 ΔGAP both-rungs>0 = {positive}")
    except Exception as e:
        reasons["error"] = str(e)
    if not any(chosen.values()):
        for f in cfg["vfull_default"]:
            chosen[f] = True
        reasons["fallback"] = f"no positive variant -> vfull_default {cfg['vfull_default']}"
    json.dump(dict(feats=chosen, reasons=reasons), open(INTER / "vfull_composition.json", "w"), indent=2)
    log(f"v_full composition = {chosen}  ({reasons})")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")          # fungi/top_weight seeds
    ap.add_argument("--seeds_null", default="0")       # near-zero null arms (low variance)
    ap.add_argument("--skip_probe", action="store_true")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True); INTER.mkdir(exist_ok=True)
    log(f"===== obj_009.1 GRID START seeds={args.seeds} seeds_null={args.seeds_null} =====")

    abort_if_hyphae("cache build")
    log("--- step 1: graph caches (CPU) ---")
    sh([HERE / "graph_caches.py"])

    log("--- step 2: psi(0)=0 GATE (CPU) ---")
    if sh([HERE / "test_invariant_v2.py"]) != 0:
        log("psi(0)=0 GATE FAILED -> abort grid"); sys.exit(2)

    if not args.skip_probe and not (INTER / "spill_caps.json").exists():
        log("--- step 3: spill probe (GPU) ---")
        sh([HERE / "spill_probe.py"])

    # PRIORITY variants get the FULL treatment (K-shape) first; v_dir/v_full (slow, disqualification-risk)
    # get the K=4 decision point only. Each variant does K4 first internally, so value accrues continuously.
    log("--- step 4: priority variants FULL (v1_baseline, v_edge, v_film, v_tele) ---")
    for v in ["v1_baseline", "v_edge", "v_film", "v_tele"]:
        abort_if_hyphae(f"full {v}")
        full_variant(v, args.seeds, args.seeds_null)

    # v_dir + v_full DROPPED per Patrick (2026-07-08): v_dir is the reverse-collapse/symmetrization-risk variant
    # (likely disqualified) and v_full is just their combo — saves ~5 h. Also honored live via
    # intermediate/skip_variants.json so the already-running orchestrator skips them without a relaunch.

    log("--- step 5: V7/V8 feat-shuffle controls (K=4) ---")
    abort_if_hyphae("feat-shuffle")
    abl("v_edge", KGAP_ARMS, "4", args.seeds, tag="grid_shuf", extra=["--feat_shuffle", "edge"])
    abl("v_film", KGAP_ARMS, "4", args.seeds, tag="grid_shuf", extra=["--feat_shuffle", "context"])

    log("--- step 6: summarize + verify ---")
    sh([HERE / "ablation_v2.py", "--summarize", "--tag", "grid"])
    sh([HERE / "verify_v2.py"])
    log("===== GRID DONE =====")


if __name__ == "__main__":
    main()
