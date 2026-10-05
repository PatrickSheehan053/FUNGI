"""
Verifies diagnostics.py's bounds_override mechanism (session 3, v16.1):
  - partial override (only listed params change, others stay probe-driven)
  - 2-element [lo,hi] format defaults confidence to 1.00
  - 3-element [lo,hi,conf] format honors the explicit confidence
  - unknown override keys are silently ignored
  - bounds_override_cfg=None and ={} are bit-identical (backward compatible)
  - overridden params are tracked in diagnostic_report["bounds_override_applied"]
    and probes_used[param] == "user_override"
  - _print_summary shows the override marker for overridden params only

build_impact_array is monkeypatched to a tiny deterministic stand-in so this
runs in milliseconds with no real AnnData/Wilcoxon DE -- the override logic
operates purely on probe OUTPUTS (utopian_bounds/raw_confidences dicts), so
this isolates exactly what changed without re-testing probe internals.
"""
import io
import sys
import contextlib
import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252;
                                          # the notebook's real Jupyter kernel
                                          # is already UTF-8, this only affects
                                          # this standalone script's own prints
import diagnostics as D


def _fake_build_impact_array(adata, pert_col, ctrl_label, **kwargs):
    n_genes = 50
    impact_array = np.array([5., 10., 15., 20., 25.])
    pert_labels = np.array(['p1', 'p2', 'p3', 'p4', 'p5'])
    weights_arr = np.ones(5)
    deg_matrix = sp.csr_matrix((n_genes, n_genes))
    lfc_matrix = np.zeros((5, n_genes), dtype=np.float32)
    valid_cq = ['p1', 'p2', 'p3', 'p4', 'p5']
    name_to_idx = {f'g{i}': i for i in range(n_genes)}
    n_tested = 5
    return (impact_array, pert_labels, weights_arr, deg_matrix, lfc_matrix,
            valid_cq, name_to_idx, n_tested)


D.build_impact_array = _fake_build_impact_array

CFG_DIAGNOSTICS = {
    "de_method": "wilcoxon",
    "de_pval_threshold": 0.05,
    "de_lfc_threshold": 0.25,
    "weight_floor": 1.0,
    "weight_ceiling": 50.0,
    "n_jobs": 1,
    "max_perts_for_de": 500,
    "bound_constraints": {
        "alpha":   {"delta_min": 0.28, "delta_max": 0.80, "hard_floor": 1.1, "hard_ceiling": 4.0},
        "gini":    {"delta_min": 0.20, "delta_max": 0.40, "hard_floor": 0.0, "hard_ceiling": 1.0},
        "gini_in": {"delta_min": 0.10, "delta_max": 0.40, "hard_floor": 0.0, "hard_ceiling": 1.0},
        "S_max":   {"delta_min": 0.02, "delta_max": 0.10, "hard_floor": 0.01, "hard_ceiling": 0.30},
        "Q":       {"delta_min": 0.10, "delta_max": 0.40, "hard_floor": -0.5, "hard_ceiling": 1.0},
        "C":       {"delta_min": 0.05, "delta_max": 0.30, "hard_floor": 0.0, "hard_ceiling": 0.35},
        "rho":     {"delta_min": 0.10, "delta_max": 0.40, "hard_floor": -1.0, "hard_ceiling": 1.0},
    },
}
CFG_INPUT = {
    "perturbation_column": "gene",
    "control_label": "non-targeting",
    "is_metacell": False,
}


def _run(bounds_override_cfg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        utopian_bounds, loss_weights, report = D.run_diagnostics(
            adata=None, n_genes=50,
            cfg_diagnostics=CFG_DIAGNOSTICS, cfg_input=CFG_INPUT,
            raw_sparse_mat=None, lambda_user_cfg=None,
            bounds_override_cfg=bounds_override_cfg)
    return utopian_bounds, loss_weights, report, buf.getvalue()


n_fail = 0

# --- Baseline (no override) -------------------------------------------------
base_ub, base_lw, base_rep, base_out = _run(None)
assert base_rep["bounds_override_applied"] == []
print("[1] baseline (no override) ran cleanly, bounds_override_applied == []")

# --- Test: backward compatibility, None vs {} are bit-identical ------------
empty_ub, empty_lw, empty_rep, empty_out = _run({})
if empty_ub != base_ub or empty_lw != base_lw:
    print("[2] FAIL: bounds_override_cfg=None vs ={} produced different results")
    n_fail += 1
else:
    print("[2] PASS: bounds_override_cfg=None and ={} are bit-identical")

# --- Test: partial override (gini + C only) ---------------------------------
ov = {"gini": [0.65, 0.85], "C": [0.008, 0.060]}
ov_ub, ov_lw, ov_rep, ov_out = _run(ov)

checks = []
checks.append(("gini overridden to [0.65, 0.85]",
               ov_ub["gini"] == [0.65, 0.85]))
checks.append(("C overridden to [0.008, 0.06]",
               ov_ub["C"] == [0.008, 0.06]))
checks.append(("gini confidence defaulted to 1.0",
               ov_rep["raw_confidences"]["gini"] == 1.0))
checks.append(("alpha (not overridden) unchanged vs baseline",
               ov_ub["alpha"] == base_ub["alpha"]))
checks.append(("S_max (not overridden) unchanged vs baseline",
               ov_ub["S_max"] == base_ub["S_max"]))
checks.append(("rho (not overridden) unchanged vs baseline",
               ov_ub["rho"] == base_ub["rho"]))
checks.append(("bounds_override_applied == ['C', 'gini']",
               ov_rep["bounds_override_applied"] == ["C", "gini"]))
checks.append(("probes_used['gini'] == 'user_override'",
               ov_rep["probes_used"]["gini"] == "user_override"))
checks.append(("probes_used['alpha'] unchanged (still probe-driven)",
               ov_rep["probes_used"]["alpha"] == base_rep["probes_used"]["alpha"]))
checks.append(("print shows '⊕ gini' marker",
               any("⊕" in line and "gini " in line for line in ov_out.splitlines())))
checks.append(("print shows '[OVERRIDE]' for gini line",
               any("gini " in line and "[OVERRIDE]" in line for line in ov_out.splitlines())))
checks.append(("print does NOT show '⊕' for alpha (still probe-driven)",
               not any("⊕" in line and "alpha" in line for line in ov_out.splitlines())))
checks.append(("S_max line still uses ✓/~ marker, not ⊕",
               any(("✓ S_max" in line or "~ S_max" in line) for line in ov_out.splitlines())))

for desc, ok in checks:
    print(f"[3] {'PASS' if ok else 'FAIL'}: {desc}")
    if not ok:
        n_fail += 1

# --- Test: 3-element format honors explicit confidence ----------------------
ov2 = {"alpha": [2.00, 2.40, 0.42]}
ov2_ub, ov2_lw, ov2_rep, ov2_out = _run(ov2)
ok = (ov2_ub["alpha"] == [2.00, 2.40]
      and ov2_rep["raw_confidences"]["alpha"] == 0.42)
print(f"[4] {'PASS' if ok else 'FAIL'}: 3-element override sets explicit confidence (0.42)")
if not ok:
    n_fail += 1
ok2 = any("conf=0.42" in line and "alpha" in line for line in ov2_out.splitlines())
print(f"[4b] {'PASS' if ok2 else 'FAIL'}: printed line shows conf=0.42 for alpha")
if not ok2:
    n_fail += 1

# --- Test: unknown override key is silently ignored --------------------------
ov3 = {"not_a_real_param": [1.0, 2.0], "gini": [0.6, 0.8]}
ov3_ub, ov3_lw, ov3_rep, ov3_out = _run(ov3)
ok = ("not_a_real_param" not in ov3_ub
      and ov3_ub["gini"] == [0.6, 0.8]
      and ov3_rep["bounds_override_applied"] == ["gini"])
print(f"[5] {'PASS' if ok else 'FAIL'}: unknown override key ignored, valid key still applied")
if not ok:
    n_fail += 1

print()
if n_fail == 0:
    print("ALL PASS")
else:
    print(f"{n_fail} CHECK(S) FAILED")
    raise SystemExit(1)
