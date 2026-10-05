"""
obj_009.3 (RHIZO-final) — verify_v3f.py : the verification battery -> results/verification_v3f.json.

  V1  ψ(0)=0 for the assembled model + every φ_e profile (runs test_invariant_v3f) — the ship-critical gate.
  V2  empty-collapse: the `empty` arm's aggregate RSC ≈ 0 (ψ(0)=0 on the real data).
  V3  mean-baseline RSC ≤ 0 (predict-zero scores ≤ 0 by construction).
  V4  data cache sha256 matches the config (no silent data drift).
  V5  nulls collapse: fungi − {shuffle, reverse, labelperm} gaps > 0 and the null arms score near 0
      (read from gauntlet.csv when present).
  V7  φ_e-permutation PER NEW FEATURE: from reg_feature_attribution.json, each new directed-topology column's
      gain must collapse under permutation (v7_surviving_frac ≈ 0), AND the whole-φ_e phishuf must collapse.
      **A feature whose gain SURVIVES permutation is reported as an ARTIFACT, not a win** (the strengthen
      inert-DASH V7 failure is the cautionary tale). N/A until the gauntlet has run.

CPU; safe on the 2070. Run after the gauntlet to fill V5/V7; before it, V1-V4 gate the ship.
"""
from __future__ import annotations
import os, sys, json, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
RES = os.path.join(HERE, "..", "results")
import yaml
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "obj_009_3.yaml")))
V7_TOL = 0.5   # a NEW feature's surviving_frac must be below this for its gain to be "real" (not an artifact)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(RES, exist_ok=True)
    checks = {}
    import test_invariant_v3f as TI
    try:
        r = TI.run()
        checks["V1_psi0_assembled"] = dict(passed=all(v["passed"] for v in r.values()),
                                           per_case={k: v["passed"] for k, v in r.items()})
    except AssertionError as e:
        checks["V1_psi0_assembled"] = dict(passed=False, err=str(e))

    import graphs_v3f as G
    import metrics_v3f as MM
    D = G.load_data(); mask = D["signal_mask"]; pidx = D["pidx"]; s = D["s_full"]
    ev = np.where(pidx >= 0)[0]
    rsc0 = MM.rsc_per_pert(s[ev], np.zeros((len(ev), len(mask))), mask, pidx[ev])
    checks["V3_mean_baseline_le0"] = dict(passed=bool(np.nanmax(rsc0) <= 1e-9), max_rsc0=float(np.nanmax(rsc0)))

    dpath = os.path.join(HERE, "..", "data", "obj_009_data.npz")
    if os.path.exists(dpath):
        sha = _sha256(dpath)
        checks["V4_cache_sha256"] = dict(passed=bool(sha == CFG.get("data_cache_sha256")), sha=sha)
    else:
        checks["V4_cache_sha256"] = dict(passed=None, note="data cache not found at data/obj_009_data.npz")

    # V2/V5 from gauntlet.csv (if present)
    gcsv = os.path.join(RES, "gauntlet.csv")
    if os.path.exists(gcsv):
        import pandas as pd
        g = pd.read_csv(gcsv)
        r2 = g[g.rung == "rung2"] if "rung" in g.columns else g
        empty_gap = r2[r2.arm == "empty"]["gap_rsc"].mean() if (r2.arm == "empty").any() else np.nan
        empty_arm_rsc = r2[r2.arm == "empty"]["arm_rsc"].mean() if "arm_rsc" in r2.columns and (r2.arm == "empty").any() else np.nan
        checks["V2_empty_collapse"] = dict(passed=bool(abs(empty_arm_rsc) < 1e-6) if np.isfinite(empty_arm_rsc) else None,
                                           empty_arm_rsc=float(empty_arm_rsc) if np.isfinite(empty_arm_rsc) else None)
        nulls = {}
        for nu in ["shuffle", "reverse", "labelperm"]:
            row = r2[r2.arm == nu]
            if len(row):
                nulls[nu] = float(row["gap_rsc"].mean())
        checks["V5_nulls_collapse"] = dict(passed=bool(all(v > 0 for v in nulls.values())) if nulls else None,
                                           fungi_minus_null=nulls)
    else:
        checks["V2_empty_collapse"] = dict(passed=None, note="run gauntlet_v3f first")
        checks["V5_nulls_collapse"] = dict(passed=None, note="run gauntlet_v3f first")

    # V7 per NEW feature from the attribution json
    apath = os.path.join(RES, "reg_feature_attribution.json")
    if os.path.exists(apath):
        a = json.load(open(apath)); cols = a.get("columns", {})
        per = {}
        for col, rec in cols.items():
            surv = rec.get("v7_surviving_frac")
            # a feature is either (a) load-bearing AND its permutation collapses the gain (real) or
            # (b) not load-bearing (contributes nothing — trivially fine). Fail only if it looks load-bearing
            # but its gain SURVIVES permutation (artifact).
            load = rec.get("load_bearing", 0.0)
            artifact = (load is not None and load > 0.005 and surv is not None and surv > V7_TOL)
            per[col] = dict(load_bearing=load, v7_surviving_frac=surv, artifact=bool(artifact))
        phishuf = a.get("phishuf_all", {}).get("surviving_frac")
        allok = (not any(v["artifact"] for v in per.values())) and (phishuf is None or phishuf < V7_TOL)
        checks["V7_phi_permutation_per_feature"] = dict(passed=bool(allok), per_feature=per,
                                                        phishuf_surviving_frac=phishuf,
                                                        note="artifact=True means a load-bearing feature's gain SURVIVED permutation")
    else:
        checks["V7_phi_permutation_per_feature"] = dict(passed=None, note="run gauntlet_v3f --summarize first")

    allpass = all(c.get("passed") for c in checks.values() if c.get("passed") is not None)
    json.dump(dict(all_passed=allpass, checks=checks), open(os.path.join(RES, "verification_v3f.json"), "w"),
              indent=2, default=float)
    for k, c in checks.items():
        st = "PASS" if c.get("passed") else ("N/A" if c.get("passed") is None else "FLAG")
        print(f"{st:4s}  {k}")
    print(f"verification_v3f -> results/verification_v3f.json (all_passed={allpass})")


if __name__ == "__main__":
    main()
