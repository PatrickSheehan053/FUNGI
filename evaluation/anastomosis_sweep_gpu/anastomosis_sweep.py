"""
obj_011 — anastomosis_sweep.py : batched, provenance-tagged, resumable GPU sweep over the exp_033 pred cells.

Walks <preds-root>/<subdir>/preds/*.npz, resolves (experiment, cell_line, density) provenance from the config's
subdir_map (identical arm names across batches are disambiguated ONLY by subdir — never flattened), applies the
exact-token legacy filter (arm.split("__")[0] in legacy_exclude -> drop; bare `knn` only, KEEP knn_causal/...),
scores each cell on GPU via gpu_panel_kernels.gpu_run_panel (bit-identical to the CPU obj_010 panel; degenerate
rule + unscoreable tag preserved, never a fabricated 0), and upserts one tidy row per cell into --out keyed on
(experiment, cell_line, density, arm, rung, tag, seed) — idempotent under --resume.

  python src/anastomosis_sweep.py --preds-root <dir> --out MASTER_RESULTS.csv --config data/sweep_config.yaml \
      --device cuda [--exclude-legacy] [--resume] [--exactness-check]
"""
from __future__ import annotations
import os, sys, glob, argparse, fnmatch, time, re
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "..", "clone", "obj010")
for _p in (CLONE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_preds as SP          # unit_from_dump
import gpu_panel_kernels as G

KEY = ["experiment", "cell_line", "density", "arm", "rung", "tag", "seed"]


def log(m): print(f"[sweep {time.strftime('%H:%M:%S')}] {m}", flush=True)


RUNG_VALUES = ("rung1", "zeroshot")
_LAMK = re.compile(r"^lam\d+k$")     # density-sweep token, e.g. lam120k/lam220k/lam350k


def parse_cell_key(stem):
    """Field-detected (not positional) parse. The density-sweep batches embed a lam###k token
    (arm__lam120k__rung1__tag__...) that the old positional logic (rung=p[1], tag=p[2]) mis-assigned to
    rung/tag. Fix: rung is matched by VALUE (rung1/zeroshot); a lam###k token is recognised as the DENSITY
    token and can never become rung or tag; tag = the token immediately after the rung; seed = trailing
    s<digits>. Backward compatible: primary cells (arm__rung1__tag__cfg__s#) parse IDENTICALLY to the old
    logic (p[1] is already the rung sentinel, p[2] the tag), so existing primary rows/keys are unchanged.
    Density is NOT emitted here — it stays authoritative from the subdir_map provenance (complete + uniform
    across primaries and density batches); the lam###k detection exists only to protect rung/tag."""
    p = stem.split("__")
    arm = p[0]
    rung = next((t for t in p[1:] if t in RUNG_VALUES), None)
    if rung is None:                                   # neither sentinel: first non-arm, non-density token
        rung = next((t for t in p[1:] if not _LAMK.match(t)), "?")
    tag = "main"
    if rung in p:
        ri = p.index(rung)
        if ri + 1 < len(p):
            cand = p[ri + 1]
            if not (cand.startswith("s") and cand[1:].isdigit()):   # not the trailing seed
                tag = cand
    seed = 0
    for tok in p[::-1]:
        if tok.startswith("s") and tok[1:].isdigit():
            seed = int(tok[1:]); break
    return dict(arm=arm, rung=rung, tag=tag, seed=seed)


def provenance(subdir, subdir_map):
    for pat, prov in subdir_map.items():
        if fnmatch.fnmatch(subdir, pat):
            return dict(prov)
    return dict(experiment="?", cell_line="?", density="?")


def is_legacy(arm, legacy_exclude):
    return arm.split("__")[0] in set(legacy_exclude)


def score_subdir(subdir_path, subdir_name, prov, legacy_exclude, exclude_legacy, device, Zfit=None):
    rows = []
    preds_dir = os.path.join(subdir_path, "preds")
    if not os.path.isdir(preds_dir):
        preds_dir = subdir_path                       # allow flat layout
    for f in sorted(glob.glob(os.path.join(preds_dir, "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        ck = parse_cell_key(stem)
        if exclude_legacy and is_legacy(ck["arm"], legacy_exclude):
            continue
        try:
            z = np.load(f, allow_pickle=True)
            if not {"pred", "true"}.issubset(set(z.files)):
                continue
            u = SP.unit_from_dump(z)
            res, meta = G.gpu_run_panel(u, Zfit=Zfit, device=device)
        except Exception as e:
            log(f"  [warn] {stem}: {type(e).__name__}: {e}"); continue
        row = {**prov, **ck, "cell_key": stem, "subdir": subdir_name,
               "n_pert": int(u.n_pert), "n_gene": int(u.n_gene),
               "coverage_frac": round(float(u.coverage_frac), 4),
               "degenerate_frac": meta["degenerate_frac"], "n_degenerate": meta["n_degenerate"],
               "unscoreable_zero_coverage": meta["unscoreable_zero_coverage"]}
        row.update({k: v for k, v in res.items()})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--exclude-legacy", action="store_true", default=True)
    ap.add_argument("--no-exclude-legacy", dest="exclude_legacy", action="store_false")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exactness-check", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    subdir_map = cfg.get("subdir_map", {}); legacy = cfg.get("legacy_exclude", [])

    if args.exactness_check:
        import exactness_gate
        exactness_gate.run(args.preds_root, subdir_map, legacy, n_sample=20,
                           out=os.path.join(HERE, "..", "intermediate", "exactness.json"))

    existing = pd.read_csv(args.out) if (args.resume and os.path.exists(args.out)) else None
    done = set(map(tuple, existing[KEY].astype(str).values.tolist())) if existing is not None else set()

    subdirs = [d for d in sorted(os.listdir(args.preds_root))
               if os.path.isdir(os.path.join(args.preds_root, d))] or ["."]
    all_rows = []
    for sd in subdirs:
        prov = provenance(sd, subdir_map)
        rows = score_subdir(os.path.join(args.preds_root, sd), sd, prov, legacy,
                            getattr(args, "exclude_legacy", True), args.device)
        if args.resume:
            rows = [r for r in rows if tuple(str(r[k]) for k in KEY) not in done]
        all_rows.extend(rows)
        log(f"{sd}: {len(rows)} cells scored ({prov.get('cell_line')}/{prov.get('density')})")

    new = pd.DataFrame(all_rows)
    if existing is not None and len(new):
        combined = pd.concat([existing, new], ignore_index=True).drop_duplicates(subset=KEY, keep="last")
    else:
        combined = new if existing is None else existing
    combined.to_csv(args.out, index=False)
    log(f"DONE: {len(new)} new rows -> {args.out} ({len(combined)} total)")


if __name__ == "__main__":
    main()
