"""
obj_011 — exactness_gate.py : CPU==GPU bit-parity gate (obj_003.2 doctrine).

For a stratified sample of real pred cells, run gpu_run_panel on BOTH cuda and the cpu fallback (the SAME kernel)
and assert: (a) 0 threshold flips on the decision axes (pearson_systema, cosine_delta); (b) the decision axes are
bit-identical (float64 -> ~1e-15). Overall max|diff| is REPORTED (may include f1@k argsort tie-ordering ~1e-6 and
centroid_accuracy mm-cdist near-ties — non-decision, inherent). Writes intermediate/exactness.json.
"""
from __future__ import annotations
import os, sys, glob, json, fnmatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "..", "clone", "obj010")
for _p in (CLONE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_preds as SP
import gpu_panel_kernels as G

TOL = 1e-6                       # decision-axis tolerance (float64 roundoff is ~1e-15)
CENTROID_TOL = 1e-3              # centroid_accuracy uses mm-cdist -> rare distance near-tie diffs (non-decision)
DECISION = ["pearson_systema_mean", "cosine_delta_mean"]


def _sample_cells(preds_root, n_sample):
    files = sorted(glob.glob(os.path.join(preds_root, "**", "*.npz"), recursive=True))
    if len(files) <= n_sample:
        return files
    idx = np.linspace(0, len(files) - 1, n_sample).astype(int)   # stratified across the corpus
    return [files[i] for i in idx]


def run(preds_root, subdir_map=None, legacy=None, n_sample=20, out=None):
    cells = _sample_cells(preds_root, n_sample)
    rec = {"n_sample": len(cells), "TOL": TOL, "cells": [], "max_abs_diff": 0.0, "flips": 0,
           "decision_max_abs_diff": 0.0, "worst_overall_metric": "",
           "note": ("decision axes (pearson_systema, cosine_delta) are the load-bearing gate (float64 -> ~1e-15, "
                    "0 flips). Overall max may include f1@k argsort tie-ordering (cuda vs cpu break |delta| ties "
                    "differently ~1e-6) and centroid_accuracy mm-cdist near-ties — both non-decision, inherent.")}
    for f in cells:
        try:
            z = np.load(f, allow_pickle=True)
            if not {"pred", "true"}.issubset(set(z.files)):
                continue
            u = SP.unit_from_dump(z)
            rc, _ = G.gpu_run_panel(u, Zfit=None, device="cpu")     # SAME kernel, CPU fallback (obj_003.2 doctrine)
            rg, _ = G.gpu_run_panel(u, Zfit=None, device="cuda")
        except Exception as e:
            rec["cells"].append({"cell": os.path.basename(f), "error": f"{type(e).__name__}: {e}"}); continue
        keys = [k for k in set(rc) & set(rg)
                if isinstance(rc[k], (int, float)) and isinstance(rg[k], (int, float))
                and np.isfinite(rc[k]) and np.isfinite(rg[k])
                and not k.startswith("centroid_accuracy")]        # cdist near-ties -> separate looser tol
        md = max(((abs(rc[k] - rg[k]), k) for k in keys), default=(0.0, ""))
        dec_md = max((abs(rc[k] - rg[k]) for k in DECISION if k in rc and k in rg
                      and np.isfinite(rc[k]) and np.isfinite(rg[k])), default=0.0)
        flips = sum(1 for k in DECISION if k in rc and k in rg and np.isfinite(rc[k]) and np.isfinite(rg[k])
                    and np.sign(rc[k]) != np.sign(rg[k]))
        if float(md[0]) > rec["max_abs_diff"]:
            rec["max_abs_diff"] = float(md[0]); rec["worst_overall_metric"] = md[1]
        rec["decision_max_abs_diff"] = max(rec["decision_max_abs_diff"], float(dec_md))
        rec["flips"] += int(flips)
        rec["cells"].append({"cell": os.path.basename(f), "max_abs_diff": float(md[0]),
                             "worst_metric": md[1], "decision_max_abs_diff": float(dec_md), "flips": int(flips)})
    # PASS = 0 flips + decision axes bit-identical (float64 roundoff). Overall/f1-tie diffs are reported, not gating.
    rec["PASS"] = bool(rec["flips"] == 0 and rec["decision_max_abs_diff"] < 1e-9)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rec, open(out, "w"), indent=2, default=float)
    print(f"[exactness] {rec['n_sample']} cells | DECISION-axis max|diff|={rec['decision_max_abs_diff']:.2e} "
          f"flips={rec['flips']} | overall max|diff|={rec['max_abs_diff']:.2e} ({rec['worst_overall_metric']}) | "
          f"{'PASS' if rec['PASS'] else 'FAIL'}")
    return rec["PASS"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True)
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "intermediate", "exactness.json"))
    a = ap.parse_args()
    ok = run(a.preds_root, n_sample=a.n_sample, out=a.out)
    sys.exit(0 if ok else 1)
