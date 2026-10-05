"""
obj_010 P1 driver (exp_031 patch3; metrics_v3f self-resolve added exp_031b patch5) — Mode-B scoring of the
gauntlet's prediction dumps with the degenerate-row rule. Reads results[/<subdir>]/preds/*.npz (keys
pred/true/pert_idx/mu_ctrl/signal_mask, written by gauntlet_v3f run_cell), builds a ScoredUnit, runs the PATCHED
panel.run_panel, writes a per-cell CSV + JSON. THIS IS THE VERDICT surface — the de-confounded decision axes
(cosine_delta, pearson_systema, DEG f1_auc) + degenerate_frac / unscoreable_zero_coverage. CPU, minutes.

  python score_preds.py --preds <dir/of/preds> --out <dir>

patch5: metrics_v3f (which panel.py imports) is resolved reproducibly by metrics_path — no hand-set
PYTHONPATH=.../ship_patch2/src needed (set SHIP_PATCH2_SRC to override the search).

Cell filenames are <arm>__<rung>__<tag>__<cfg>__s<seed>.npz (or the 4-field form); arm/rung/seed are parsed back
out for grouping. An all-degenerate arm (a ψ(0)=0 zero-coverage causal arm on zeroshot) reports
cosine_delta_mean = NaN and unscoreable_zero_coverage = True — NEVER a fabricated 0 (the blueprint rule).
"""
from __future__ import annotations
import os, sys, json, argparse, glob
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from metrics_path import ensure_metrics_v3f_on_path
ensure_metrics_v3f_on_path()                         # patch5: bake metrics_v3f resolution (no manual PYTHONPATH)
import panel as PANEL
from io_adapters import ScoredUnit

F1_KS = (10, 20, 50, 100, 200, 500, 1000)   # patch6: DEG-recovery f1@k sweep (matches panel.tier_deg k_values)


def parse_cell_key(stem):
    parts = stem.split("__")
    arm = parts[0] if parts else stem
    rung = parts[1] if len(parts) > 1 else "?"
    seed = parts[-1]
    seed = int(seed[1:]) if seed.startswith("s") and seed[1:].isdigit() else -1
    tag = parts[2] if len(parts) > 2 else "main"
    return dict(arm=arm, rung=rung, tag=tag, seed=seed)


def unit_from_dump(z):
    """Build a Mode-B ScoredUnit from a gauntlet pred dump. pred/true are per-eval-pert delta-LFC; means =
    mu_ctrl + delta (exact). pert_gene_idx carries the dumped pert row-index (the rsc self-mask is continuity-
    only for this path — the DECISION axes cosine_delta / pearson_systema / degenerate_frac do not use it)."""
    pred = np.asarray(z["pred"], np.float64); true = np.asarray(z["true"], np.float64)
    mu = np.asarray(z["mu_ctrl"], np.float64); sig = np.asarray(z["signal_mask"]).astype(bool)
    pidx = np.asarray(z["pert_idx"], np.int64) if "pert_idx" in z.files else np.full(pred.shape[0], -1, np.int64)
    names = np.arange(pred.shape[1])
    return ScoredUnit(means_true=mu[None, :] + true, means_pred=mu[None, :] + pred, mu_ctrl=mu, signal_mask=sig,
                      pert_gene_idx=pidx, gene_names=names, split="test", input_kind="mean_lfc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="dir of gauntlet pred dumps (*.npz)")
    ap.add_argument("--out", default=None, help="output dir (default: <preds>/../anastomosis)")
    args = ap.parse_args()
    preds = args.preds
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(preds)), "anastomosis")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(preds, "*.npz")))
    if not files:
        print(f"[score_preds] no *.npz under {preds}"); return 1
    rows = []
    for f in files:
        stem = os.path.basename(f)[:-4]
        meta_cell = parse_cell_key(stem)
        try:
            z = np.load(f, allow_pickle=True)
            u = unit_from_dump(z)
            res, meta = PANEL.run_panel(u, Zfit=None, run_cell=False)
        except Exception as e:
            print(f"[skip] {stem}: {type(e).__name__}: {e}"); continue
        # patch6: DEG-AUPRC bug fix — run_panel aggregates the DEG f1 sweep under `f1_auc_mean` (and
        # `f1_at_{k}_mean/_median`), NOT a bare `f1_auc`; the old .get("f1_auc") always returned None -> NaN.
        row = dict(cell=stem, **meta_cell, n_pert=int(u.n_pert),
                   degenerate_frac=meta["degenerate_frac"], n_degenerate=meta["n_degenerate"],
                   unscoreable_zero_coverage=meta["unscoreable_zero_coverage"],
                   cosine_delta_mean=res.get("cosine_delta_mean"),
                   pearson_systema_mean=res.get("pearson_systema_mean"),
                   f1_auc=res.get("f1_auc_mean"), f1_auc_median=res.get("f1_auc_median"),
                   rsc_mean=res.get("rsc_mean"),
                   **{f"f1_at_{k}_mean": res.get(f"f1_at_{k}_mean") for k in F1_KS},
                   **{f"f1_at_{k}_median": res.get(f"f1_at_{k}_median") for k in F1_KS})
        rows.append(row)
        json.dump({"cell": meta_cell, "metrics": res, "meta": meta},
                  open(os.path.join(out, stem + ".json"), "w"), indent=2, default=float)
        tag = " [UNSCOREABLE]" if meta["unscoreable_zero_coverage"] else ""
        cd = row["cosine_delta_mean"]; cd = f"{cd:+.4f}" if cd is not None and np.isfinite(cd) else "nan"
        print(f"  {stem:52s} cosine_delta={cd} degen_frac={row['degenerate_frac']:.3f}{tag}")
    csv_path = os.path.join(out, "anastomosis_preds.csv")
    cols = (["cell", "arm", "rung", "tag", "seed", "n_pert", "degenerate_frac", "n_degenerate",
             "unscoreable_zero_coverage", "cosine_delta_mean", "pearson_systema_mean",
             "f1_auc", "f1_auc_median", "rsc_mean"]
            + [f"f1_at_{k}_mean" for k in F1_KS] + [f"f1_at_{k}_median" for k in F1_KS])
    try:
        import pandas as pd
        pd.DataFrame(rows)[cols].to_csv(csv_path, index=False)
    except Exception:
        import csv
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in cols})
    print(f"[score_preds] {len(rows)} cells -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
