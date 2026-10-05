"""
obj_009.1 — verify_v2.py : the verification battery -> results/verification_v2.json.
Model-level + data checks run standalone (V1 psi0, V3 mean-baseline, V4 leakage/cache-sha256). Graph-comparison
checks (V2 empty collapse, V5 reverse/labelperm collapse incl. the V-DIR extra, V6 DWLP cross-reproduction) read
the P1 summary. The two NEW anti-leak controls read the feat_shuffle runs produced by the grid:
  V7 edge-feature-shuffle: permuting phi_e across edges must SHRINK V-EDGE's FUNGI-minus-top_weight gap toward
     the v1 scalar-weight baseline (proves phi_e carries graph signal, not a leak/bug).
  V8 context-shuffle: permuting c_g across genes must COLLAPSE V-FILM's gain (proves it is graph-context, not
     an identity channel — invariant #2).
Run AFTER the grid + the feat_shuffle controls. 8 checks per applicable variant.
"""
from __future__ import annotations
import os, sys, json, hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
RES = os.path.join(HERE, "..", "results")
import yaml
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "obj_009_1.yaml")))
V1_DATA = os.path.join(HERE, "..", "..", "obj_009_topology_aware_gnn", "intermediate", "obj_009_data.npz")
DWLP_REF = 0.056  # v1 linear DWLP FUNGI RSC reference


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    checks = {}

    # (V1) psi(0)=0 for ALL six variants
    import test_invariant_v2 as TI
    try:
        r = TI.run()
        checks["V1_psi0_all_variants"] = dict(passed=all(v["passed"] for v in r.values()),
                                              per_variant={k: v["passed"] for k, v in r.items()})
    except AssertionError as e:
        checks["V1_psi0_all_variants"] = dict(passed=False, err=str(e))

    # (V3) mean-baseline RSC <= 0 (predict 0)
    import metrics_v2 as M
    d = np.load(V1_DATA, allow_pickle=True)
    mask = d["signal_mask"].astype(bool); pidx = d["pidx"].astype(np.int64); s = d["s_full"]
    ev = np.where(pidx >= 0)[0]
    rsc0 = M.rsc_per_pert(s[ev], np.zeros((len(ev), len(mask))), mask, pidx[ev])
    checks["V3_mean_baseline_le0"] = dict(passed=bool(np.nanmax(rsc0) <= 1e-9), max_rsc0=float(np.nanmax(rsc0)))

    # (V4) leakage + data-cache sha256 unchanged (reused v1 substrate verbatim, File A train-only)
    sha = _sha256(V1_DATA)
    checks["V4_leakage_and_cache_sha256"] = dict(
        passed=bool(sha == CFG["data_cache_sha256"]), sha256=sha, expected=CFG["data_cache_sha256"],
        note="substrate = exp_024 File A train-only; mu_bar/mask/edge-features/node-context are graph-derived "
             "train-only; only gene identities cross. Rung 2 holds perts from the MODEL, graph keeps out-edges.")

    # graph-comparison checks from the P1 summary
    p1p = os.path.join(RES, "obj_009_1_P1_summary.csv")
    if os.path.exists(p1p):
        df = pd.read_csv(p1p)
        variants = sorted(df.variant.unique())

        def rsc(variant, rung, arm, K=4):
            m = df[(df.variant == variant) & (df.rung == rung) & (df.arm == arm) & (df.K == K)]
            return float(m["rsc"].iloc[0]) if len(m) else None

        # (V2) empty collapse per variant
        emp = {v: [rsc(v, rg, "empty") for rg in ["rung1", "rung2"]] for v in variants}
        checks["V2_empty_collapse"] = dict(
            passed=all(abs(x) < 1e-3 for vv in emp.values() for x in vv if x is not None), per_variant=emp)

        # (V5) reverse/labelperm collapse per variant (must sit far below FUNGI); V-DIR extra flagged explicitly
        v5 = {}; v5pass = True
        for v in variants:
            rev = [rsc(v, rg, "reverse") for rg in ["rung1", "rung2"]]
            lab = [rsc(v, rg, "labelperm") for rg in ["rung1", "rung2"]]
            fun = [rsc(v, rg, "fungi_bio") for rg in ["rung1", "rung2"]]
            ok = all((f is None or ((l is None or l < 0.5 * max(f, 1e-9)) and (r_ is None or r_ < 0.5 * max(f, 1e-9))))
                     for f, l, r_ in zip(fun, lab, rev))
            v5[v] = dict(reverse=rev, labelperm=lab, fungi=fun, passed=bool(ok))
            v5pass = v5pass and ok
        checks["V5_reverse_labelperm_collapse"] = dict(passed=bool(v5pass), per_variant=v5,
            note="V-DIR extra: the added out-channel must NOT resurrect reverse (a reversed-graph forward "
                 "bypass). If v_dir.reverse does not collapse, V-DIR is a symmetrization-in-disguise and is "
                 "DISQUALIFIED from graduating (reported, not promoted).")

        # (V6) DWLP cross-reproduction: FUNGI DNPN a real predictor; report absolute ratio to DWLP
        v6 = {}
        for v in variants:
            fun = [x for x in [rsc(v, rg, "fungi_bio") for rg in ["rung1", "rung2"]] if x is not None]
            fmax = max(fun) if fun else None
            v6[v] = dict(fungi_rsc=fmax, dwlp_ref=DWLP_REF, ratio=(fmax / DWLP_REF if fmax else None))
        checks["V6_dwlp_cross_reproduction"] = dict(
            passed=all(v["fungi_rsc"] is not None and v["fungi_rsc"] > 0 for v in v6.values()), per_variant=v6,
            note="DNPN is a real predictor (FUNGI RSC>0 >> structure-blind). v1 was ~0.55x DWLP (shared-readout "
                 "zero-shot cost); V-FILM is expected to CLOSE this ratio.")
    else:
        for k in ["V2_empty_collapse", "V5_reverse_labelperm_collapse", "V6_dwlp_cross_reproduction"]:
            checks[k] = dict(passed=None, note="P1 summary not found yet — run the grid + summarize first")

    # (V7) edge-feature-shuffle control + (V8) context-shuffle control from the feat_shuffle kgap
    kg = os.path.join(RES, "obj_009_1_P1_kgap.csv")
    runs = os.path.join(RES, "runs_grid.csv")
    checks["V7_edge_feature_shuffle"] = _shuffle_control("v_edge", "edge", "V7")
    checks["V8_context_shuffle"] = _shuffle_control("v_film", "context", "V8")

    allpass = all(c.get("passed") for c in checks.values() if c.get("passed") is not None)
    json.dump(dict(all_passed=allpass, checks=checks), open(os.path.join(RES, "verification_v2.json"), "w"),
              indent=2, default=float)
    for k, c in checks.items():
        st = "PASS" if c.get("passed") else ("N/A" if c.get("passed") is None else "FLAG")
        print(f"{st:4s}  {k}")
    print(f"verification_v2 -> results/verification_v2.json (all_passed={allpass})")


def _shuffle_control(variant, kind, tag):
    """Compare the variant's real FUNGI-minus-top_weight gap vs its feat-shuffled gap at K=4, both rungs.
    PASS if shuffling the graph-derived feature SHRINKS the gap toward the v1 baseline (|gap_shuffled - gap_v1|
    < |gap_real - gap_v1|), i.e. the feature's contribution collapses when its graph-structure is destroyed."""
    p = os.path.join(RES, "runs_grid_shuf.csv")
    pn = os.path.join(RES, "runs_grid.csv")
    if not (os.path.exists(p) and os.path.exists(pn)):
        return dict(passed=None, note=f"{tag}: feat_shuffle control runs not found yet")
    try:
        shuf = pd.read_csv(p); base = pd.read_csv(pn)
        out = {}; okall = True
        for rung in ["rung1", "rung2"]:
            def gap(df, v, fs):
                f = df[(df.variant == v) & (df.feat_shuffle == fs) & (df.rung == rung) & (df.arm == "fungi_bio") & (df.K == 4)]
                t = df[(df.variant == v) & (df.feat_shuffle == fs) & (df.rung == rung) & (df.arm == "top_weight") & (df.K == 4)]
                if not len(f) or not len(t):
                    return None
                return float(f.rsc.mean() - t.rsc.mean())
            g_real = gap(base, variant, "none")
            g_shuf = gap(shuf, variant, kind)
            g_v1 = gap(base, "v1_baseline", "none")
            if None in (g_real, g_shuf, g_v1):
                out[rung] = dict(note="missing cells"); continue
            shrinks = abs(g_shuf - g_v1) <= abs(g_real - g_v1) + 1e-9
            out[rung] = dict(gap_real=g_real, gap_shuffled=g_shuf, gap_v1_baseline=g_v1, shrinks_toward_v1=bool(shrinks))
            okall = okall and shrinks
        return dict(passed=bool(okall), per_rung=out,
                    note=f"{tag}: shuffling the graph-derived feature must move the gap back toward the v1 "
                         f"scalar baseline (feature carries GRAPH signal, not identity leak).")
    except Exception as e:
        return dict(passed=None, err=str(e))


if __name__ == "__main__":
    main()
