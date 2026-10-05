"""
obj_010 (exp_031b patch5) — stats_h_a.py : the FDR-confirmed headline H-A table for Ch5. CPU-only.

Reads the CORE 10-seed pred dumps (results/core/preds/*.npz), recomputes the PER-PERTURBATION decision axes
(cosine_delta, pearson_systema) through the PATCHED panel (degenerate rows -> NaN, excluded), averages each arm's
per-pert vector across seeds (aligned by pert index), and runs the paired H-A comparisons:

  fungi_bio vs top_weight ; shroom vs top_weight ; ptf_borda vs top_weight ; borda vs top_weight   (graph-quality)
  shroom    vs fungi_bio                                                                            (the pruning question)

per rung x metric: paired bootstrap (mean diff + one-sided p) + Wilcoxon signed-rank + BH-FDR across the rung's
family. H-A is IN-DISTRIBUTION -> read rung1 (causal arms are unscoreable on zeroshot; those rows are flagged, not
fabricated). Writes stats_h_a.csv + stats_h_a.json.

  python stats_h_a.py --preds results/core/preds --out results/core/stats [--rungs rung1,zeroshot] [--alpha 0.05]
"""
from __future__ import annotations
import os, sys, json, glob, argparse
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from metrics_path import ensure_metrics_v3f_on_path
ensure_metrics_v3f_on_path()
import panel as PANEL
import metrics_v3f as M3
from score_preds import unit_from_dump, parse_cell_key

DEFAULT_COMPARISONS = [   # (test_arm, ref_arm, label)
    ("fungi_bio", "top_weight", "fungi_bio_vs_top_weight"),
    ("shroom", "top_weight", "shroom_vs_top_weight"),
    ("ptf_borda", "top_weight", "ptf_borda_vs_top_weight"),
    ("borda", "top_weight", "borda_vs_top_weight"),
    ("shroom", "fungi_bio", "shroom_vs_fungi_bio_PRUNING"),
]
METRICS = ["cosine_delta", "pearson_systema"]
MIN_PAIRS = 10


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (monotone), NaN-safe (NaNs kept NaN, excluded from the ranking)."""
    p = np.asarray(pvals, float); q = np.full(p.shape, np.nan)
    ok = np.isfinite(p); pv = p[ok]; n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv); ranked = pv[order]
    qr = ranked * n / (np.arange(1, n + 1))
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(qr, 1.0)
    q[ok] = out
    return q


def per_pert_vectors(f):
    """Return {'cosine_delta': {pidx: val}, 'pearson_systema': {pidx: val}} for one pred dump (degenerate->NaN)."""
    z = np.load(f, allow_pickle=True); u = unit_from_dump(z)
    deg = PANEL.degenerate_pred_mask(u)
    co = PANEL.tier_correlation(u, deg=deg)["cosine_delta"]
    ps = PANEL.tier_systema(u, deg=deg)["pearson_systema"]
    pidx = np.asarray(z["pert_idx"], np.int64) if "pert_idx" in z.files else np.arange(u.n_pert)
    return {"cosine_delta": {int(p): float(v) for p, v in zip(pidx, co)},
            "pearson_systema": {int(p): float(v) for p, v in zip(pidx, ps)}}


def collect(preds, rung):
    """arm -> metric -> {pidx: per-seed-mean value} for all cells of `rung`."""
    acc = {}   # arm -> metric -> pidx -> list of per-seed values
    for f in sorted(glob.glob(os.path.join(preds, "*.npz"))):
        c = parse_cell_key(os.path.basename(f)[:-4])
        if c["rung"] != rung or c["tag"] != "main":
            continue
        vecs = per_pert_vectors(f)
        A = acc.setdefault(c["arm"], {m: {} for m in METRICS})
        for m in METRICS:
            for p, v in vecs[m].items():
                A[m].setdefault(p, []).append(v)
    out = {}   # arm -> metric -> {pidx: nanmean-across-seeds}
    for arm, md in acc.items():
        out[arm] = {}
        for m in METRICS:
            out[arm][m] = {p: float(np.nanmean(vs)) for p, vs in md[m].items()
                           if np.isfinite(np.nanmean(vs))}
    return out


def paired(test_map, ref_map):
    shared = sorted(set(test_map) & set(ref_map))
    t = np.array([test_map[p] for p in shared]); r = np.array([ref_map[p] for p in shared])
    m = np.isfinite(t) & np.isfinite(r); return t[m], r[m], int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--rungs", default="rung1,zeroshot"); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--comparisons", default=None,
                    help="override the default H-A set: 'test:ref,test:ref,...' (label auto = test_vs_ref)")
    args = ap.parse_args()
    comparisons = DEFAULT_COMPARISONS
    if args.comparisons:
        comparisons = [(p.split(":")[0].strip(), p.split(":")[1].strip(),
                        f"{p.split(':')[0].strip()}_vs_{p.split(':')[1].strip()}")
                       for p in args.comparisons.split(",") if ":" in p]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.preds)), "stats")
    os.makedirs(out, exist_ok=True)
    from scipy.stats import wilcoxon
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    rows = []
    for rung in rungs:
        arms = collect(args.preds, rung)
        fam = []
        for test, ref, label in comparisons:
            row = dict(rung=rung, comparison=label, test=test, ref=ref)
            if test not in arms or ref not in arms:
                row.update(status="arm_missing", n_pert=0);
                for m in METRICS:
                    rows.append({**row, "metric": m, "mean_diff": np.nan, "boot_p": np.nan, "wilcoxon_p": np.nan})
                continue
            for m in METRICS:
                t, r, n = paired(arms[test][m], arms[ref][m])
                rr = {**row, "metric": m, "n_pert": n}
                if n < MIN_PAIRS:
                    rr.update(status="insufficient_or_unscoreable", mean_diff=np.nan, boot_p=np.nan, wilcoxon_p=np.nan)
                else:
                    pb = M3.paired_bootstrap(t, r, n_boot=args.n_boot, seed=2)
                    try:
                        wp = float(wilcoxon(t, r).pvalue)
                    except Exception:
                        wp = np.nan
                    rr.update(status="ok", mean_diff=round(float(pb["mean_diff"]), 5),
                              boot_p=float(pb.get("p_one_sided", np.nan)), wilcoxon_p=wp,
                              test_mean=round(float(np.mean(t)), 5), ref_mean=round(float(np.mean(r)), 5))
                fam.append(rr); rows.append(rr)
        # BH-FDR within this rung's family (on the ok rows)
        idx = [i for i, rr in enumerate(fam)]
        qb = bh_fdr([fam[i].get("boot_p", np.nan) for i in idx])
        qw = bh_fdr([fam[i].get("wilcoxon_p", np.nan) for i in idx])
        for j, i in enumerate(idx):
            fam[i]["bh_q_boot"] = None if not np.isfinite(qb[j]) else round(float(qb[j]), 5)
            fam[i]["bh_q_wilcoxon"] = None if not np.isfinite(qw[j]) else round(float(qw[j]), 5)
            fam[i]["significant"] = bool(np.isfinite(qb[j]) and np.isfinite(qw[j])
                                         and qb[j] < args.alpha and qw[j] < args.alpha)

    # write CSV + JSON
    cols = ["rung", "comparison", "test", "ref", "metric", "status", "n_pert", "test_mean", "ref_mean",
            "mean_diff", "boot_p", "bh_q_boot", "wilcoxon_p", "bh_q_wilcoxon", "significant"]
    csv_path = os.path.join(out, "stats_h_a.csv")
    try:
        import pandas as pd
        pd.DataFrame(rows).reindex(columns=cols).to_csv(csv_path, index=False)
    except Exception:
        import csv
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in cols})
    json.dump({"alpha": args.alpha, "n_boot": args.n_boot, "comparisons": [c[2] for c in comparisons],
               "rows": rows}, open(os.path.join(out, "stats_h_a.json"), "w"), indent=2, default=float)
    nsig = sum(1 for r in rows if r.get("significant"))
    print(f"[stats_h_a] {len(rows)} tests over rungs={rungs}; {nsig} significant at BH q<{args.alpha} -> {csv_path}")
    # pretty print the headline (rung1, ok rows)
    for r in rows:
        if r["rung"] == "rung1" and r.get("status") == "ok":
            print(f"  rung1 {r['comparison']:28s} {r['metric']:16s} d={r['mean_diff']:+.4f} "
                  f"boot_q={r.get('bh_q_boot')} wilcox_q={r.get('bh_q_wilcoxon')} sig={r.get('significant')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
