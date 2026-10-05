"""
obj_010 ANASTOMOSIS — firesale.py : corpus discovery + inventory + master ledger.

(a) Discovery: walk configured roots, classify each artifact by SIGNATURE:
      npz with the RHIZO Mode-B keys (panel/mu_ctrl/*_LFC) -> mean_lfc
      pred_*.h5ad + true_*.h5ad pair in a dir        -> per_cell
      else                                            -> UNCLASSIFIED.txt (never silently dropped)
(b) Inventory: firesale_out/inventory.csv (path, kind, generation, n_seeds, mtime). STOP for a one-line human
    confirm on first run (--resume skips the stop).
(c) Score: each classified artifact through scorer, tagged with its generation.
(d) Ledger: concat all rows -> anastomosis_master_ledger.csv (+ .parquet) + rank_within_metric helper cols.
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scorer import Anastomosis

RHIZO_KEYS = {"panel", "mu_ctrl", "signal_mask"}


def _classify_npz(p: Path):
    try:
        z = np.load(p, allow_pickle=True)
        keys = set(z.files)
        if RHIZO_KEYS.issubset(keys) and any(k.endswith("_LFC") for k in keys):
            n_splits = sum(1 for k in keys if k.endswith("_LFC"))
            return ("mean_lfc", n_splits)
    except Exception:
        pass
    return (None, 0)


def _generation_of(path: Path, roots: dict):
    s = str(path).replace("\\", "/")
    for gen, root in roots.items():
        if str(Path(root)).replace("\\", "/") in s:
            return gen
    return "unknown"


def discover(roots: dict, fout: Path):
    fout.mkdir(parents=True, exist_ok=True)
    rows = []; unclassified = []
    seen_percell_dirs = set()
    for gen, root in roots.items():
        root = Path(root)
        if not root.exists():
            continue
        # npz (mean_lfc)
        for p in root.rglob("*.npz"):
            kind, ns = _classify_npz(p)
            if kind == "mean_lfc":
                rows.append(dict(path=str(p), kind="mean_lfc", generation=gen, n_seeds=ns,
                                 mtime=time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))))
            # arm .npz (src/tgt/w) etc. are NOT mean_lfc -> not scoreable as predictions; log softly
            elif p.name.endswith(".npz"):
                unclassified.append(f"{p}  (npz without RHIZO Mode-B keys: {gen})")
        # per_cell (pred_*/true_* pairs)
        for tp in root.rglob("true_*.h5ad"):
            d = tp.parent
            if d in seen_percell_dirs:
                continue
            preds = list(d.glob("pred_*.h5ad")); trues = list(d.glob("true_*.h5ad"))
            if preds and trues:
                seen_percell_dirs.add(d)
                rows.append(dict(path=str(d), kind="per_cell", generation=gen, n_seeds=len(trues),
                                 mtime=time.strftime("%Y-%m-%d", time.localtime(d.stat().st_mtime))))
    inv = pd.DataFrame(rows).drop_duplicates(subset=["path"]).sort_values(["generation", "kind", "path"])
    inv.to_csv(fout / "inventory.csv", index=False)
    (fout / "UNCLASSIFIED.txt").write_text("\n".join(unclassified), encoding="utf-8")
    return inv, unclassified


def score_corpus(cfg: dict, inv: pd.DataFrame, fout: Path, only_kind=None, device="cpu"):
    an = Anastomosis(cfg); all_rows = []
    for _, r in inv.iterrows():
        if only_kind and r["kind"] != only_kind:
            continue
        run = dict(tag=Path(r["path"]).stem, input_kind=r["kind"], path=r["path"],
                   generation=r["generation"], substrate=r["generation"],
                   splits=["val", "test"] if r["kind"] == "mean_lfc" else ["test"])
        try:
            rows = an.score_run(run, device=device); all_rows += rows
            print(f"[scored] {run['tag']} ({r['kind']}, {r['generation']}) -> {len(rows)} rows")
        except Exception as e:
            print(f"[FAIL] {run['tag']}: {e}")
    if all_rows:
        led = pd.DataFrame(all_rows)
        for m in [c for c in led.columns if c.endswith("_mean")]:
            led[f"rank__{m}"] = led[m].rank(ascending=False, method="min")
        (fout).mkdir(parents=True, exist_ok=True)
        led.to_csv(fout / "anastomosis_master_ledger.csv", index=False)
        try: led.to_parquet(fout / "anastomosis_master_ledger.parquet", index=False)
        except Exception: pass
        print(f"master ledger: {len(led)} rows -> {fout/'anastomosis_master_ledger.csv'}")
    return all_rows


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--only-kind", default=None)
    ap.add_argument("--score", action="store_true", help="score after inventory (else inventory-only + STOP)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    fout = Path(cfg["paths"].get("firesale_out", "firesale_out"))
    roots = cfg["firesale"]["roots"]
    inv, unc = discover(roots, fout)
    print(f"\n=== INVENTORY ({len(inv)} artifacts) -> {fout/'inventory.csv'} ===")
    print(inv.to_string(index=False) if len(inv) else "(none found)")
    print(f"UNCLASSIFIED: {len(unc)} (see {fout/'UNCLASSIFIED.txt'})")
    if not (a.resume or a.score):
        print("\n*** STOP: review inventory.csv, then re-run with --score (or --resume) to score the corpus. ***")
        return 0
    score_corpus(cfg, inv, fout, only_kind=a.only_kind, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
