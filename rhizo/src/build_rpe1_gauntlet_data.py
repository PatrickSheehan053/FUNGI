"""
exp_029 RPE1 port (build side) — construct the obj_009.3 gauntlet's data package FOR RPE1.
exp_031 patch3 FIX: SHIP / DATA_OUT / ARM_OUT are now derived from a single `--root` argument instead of the
hardcoded nested paths (EXP/ship + EXP/clone/data) the web agent had to symlink around.

  bundle root IS `ship/`  ->  --root <ship>
    arms live at            <root>/arms/<stem>.npz
    the cell-half LFC at    <root>/lfc_targets_rpe1_cellhalf.npz  (fallback <root>/lfc_targets_rpe1.npz)
    the built package at     <root>/data/  (obj_009_data.npz + arm_graphs/<sem>.npz)

The obj_009.3 gauntlet reads data/obj_009_data.npz + data/arm_graphs/<arm>.npz (src/tgt/w_outnorm/w_raw,
SEMANTIC names). This script builds the RPE1 versions into <root>/data/.

  1. obj_009_data.npz  from the cell-half LFC: panel_genes, mu_ctrl, signal_mask, pidx, s_full,
     s_A/s_B (GENUINE cell halves -> real rung1), var_real, split (0=train,1=val,2=test).
  2. arm_graphs/<semantic>.npz  from <root>/arms/<file>.npz.

  python build_rpe1_gauntlet_data.py --root /path/to/ship
"""
import os, sys, json, argparse
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]

# Module-level path handles — set by configure(root). Defaults keep the bundle self-consistent (root == EXP/ship).
SHIP = None; DATA_OUT = None; ARM_OUT = None; LFC = None

# `shroom` -> the REBUILT true dense parent (contains fungi_bio); nulls are PER-ARM. exp_031: the RHIZOMORPH
# cohort arms (rhizomorph/shroom_grafted/shuffle_of_rhizomorph/reverse_of_rhizomorph) are already in arm_graphs
# format and are dropped straight into <root>/data/arm_graphs by the web agent (RHIZO_EXTRA_ARMS hook) — they do
# NOT go through ARM_MAP (which remaps exp_025 `w`-key stems).
ARM_MAP = {  # gauntlet semantic name -> shipped arm file stem
    "fungi_bio": "shroom_fungi", "top_weight": "hyphae", "knn": "knn", "mst": "borda_mst",
    "empty": "empty",
    "hyphae_fungi": "hyphae_fungi", "ptf_borda": "ptf_borda", "borda": "borda",
    "shroom": "shroom_denseparent",
    "shuffle_of_fungi_bio": "shuffle_of_fungi_bio",
    "reverse_of_fungi_bio": "reverse_of_fungi_bio",
}
NULL_ARMS = {"empty", "shuffle_of_fungi_bio", "reverse_of_fungi_bio"}


def configure(root):
    """Point every path at the bundle whose root IS `ship/` (arms at <root>/arms, package at <root>/data)."""
    global SHIP, DATA_OUT, ARM_OUT, LFC
    SHIP = Path(root)
    DATA_OUT = SHIP / "data"
    ARM_OUT = DATA_OUT / "arm_graphs"
    lfc_ch = SHIP / "lfc_targets_rpe1_cellhalf.npz"
    lfc_agg = SHIP / "lfc_targets_rpe1.npz"
    LFC = lfc_ch if lfc_ch.exists() else lfc_agg
    return SHIP, DATA_OUT, ARM_OUT, LFC


def build_data():
    z = np.load(LFC, allow_pickle=True)
    has_halves = all(f"{sp}_LFC_A" in z.files for sp in ("train", "val", "test"))
    if LFC.name.endswith("cellhalf.npz") and has_halves:
        print(f"[FIX A] using GENUINE cell-half LFC: {LFC.name}")
    else:
        print(f"[FIX A][WARN] cell-half LFC not found -> s_A=s_B=s_full (DEGENERATE). rung1 will be REFUSED by the "
              f"gauntlet's degenerate guard. Provide {SHIP}/lfc_targets_rpe1_cellhalf.npz to enable a real rung1.")
    panel = np.asarray(z["panel"]); N = int(z["N"])
    mu = np.asarray(z["mu_ctrl"], np.float64); sig = np.asarray(z["signal_mask"]).astype(bool)
    parts_LFC, parts_A, parts_B, parts_pidx, parts_split = [], [], [], [], []
    for si, sp in enumerate(("train", "val", "test")):
        L = np.asarray(z[f"{sp}_LFC"], np.float32)
        parts_LFC.append(L)
        parts_A.append(np.asarray(z[f"{sp}_LFC_A"], np.float32) if has_halves else L)
        parts_B.append(np.asarray(z[f"{sp}_LFC_B"], np.float32) if has_halves else L)
        parts_pidx.append(np.asarray(z[f"{sp}_pidx"], np.int64))
        parts_split.append(np.full(len(z[f"{sp}_pidx"]), si, np.int64))
    s_full = np.concatenate(parts_LFC, axis=0)
    s_A = np.concatenate(parts_A, axis=0); s_B = np.concatenate(parts_B, axis=0)
    pidx = np.concatenate(parts_pidx); split = np.concatenate(parts_split)
    var_real = np.var(s_full.astype(np.float64), axis=0) + 1e-8
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_OUT / "obj_009_data.npz",
             panel_genes=panel, mu_ctrl=mu, signal_mask=sig, pidx=pidx,
             s_full=s_full, s_A=s_A, s_B=s_B, var_real=var_real, split=split)
    n_tr = int((split == 0).sum()); n_ho = int((split > 0).sum())
    genuine = has_halves and not np.array_equal(s_A, s_B)
    print(f"obj_009_data.npz: N={N} perts={len(pidx)} (train {n_tr} / held-out {n_ho}) signal={int(sig.sum())} "
          f"| s_A!=s_B (genuine rung1)={genuine}  -> {DATA_OUT/'obj_009_data.npz'}")
    return N, pidx, split


def _coverage(src, pidx, split):
    srcset = set(np.unique(np.asarray(src, np.int64)).tolist())
    train_g = set(pidx[split == 0].tolist()); held_g = set(pidx[split > 0].tolist())
    return len(srcset), len(srcset & train_g), len(srcset & held_g)


def build_arm_graphs(pidx, split):
    ARM_OUT.mkdir(parents=True, exist_ok=True)
    built, missing, manifest_arms = [], [], {}
    for sem, stem in ARM_MAP.items():
        p = SHIP / "arms" / f"{stem}.npz"
        if not p.exists():
            missing.append((sem, stem)); continue
        d = np.load(p)
        s = d["src"].astype(np.int64); t = d["tgt"].astype(np.int64)
        w = (d["w"] if "w" in d.files else d["w_raw"]).astype(np.float64)
        np.savez(ARM_OUT / f"{sem}.npz", src=s, tgt=t,
                 w_outnorm=(d["w_outnorm"] if "w_outnorm" in d.files else w).astype(np.float32),
                 w_raw=w.astype(np.float64))
        n_src, in_train, held_cov = _coverage(s, pidx, split)
        manifest_arms[sem] = dict(stem=stem, n_edges=int(len(s)), n_src=n_src,
                                  in_train_coverage=in_train, held_out_coverage=held_cov,
                                  is_null=bool(sem in NULL_ARMS))
        built.append((sem, stem, len(s), n_src, held_cov))
    # exp_031: pass-through arm-graph .npz already present in <root>/data/arm_graphs (dropped in by the web agent,
    # e.g. the RHIZOMORPH cohort). Record them in the manifest so the coverage pre-flight can see them.
    if ARM_OUT.exists():
        known = set(ARM_MAP.keys())
        for p in sorted(ARM_OUT.glob("*.npz")):
            sem = p.stem
            if sem in known or sem in manifest_arms:
                continue
            d = np.load(p)
            if not {"src", "tgt"}.issubset(set(d.files)):
                continue
            s = d["src"].astype(np.int64); t = d["tgt"].astype(np.int64)
            n_src, in_train, held_cov = _coverage(s, pidx, split)
            manifest_arms[sem] = dict(stem="(passthrough)", n_edges=int(len(s)), n_src=n_src,
                                      in_train_coverage=in_train, held_out_coverage=held_cov,
                                      is_null=bool(sem.startswith(("shuffle_of_", "reverse_of_")) or sem == "empty"))
            built.append((sem, "(passthrough)", len(s), n_src, held_cov))
    for sem, stem, ne, ns, hc in built:
        print(f"  arm_graphs/{sem}.npz <- {stem}  ({ne:,} edges | n_src={ns} | held_out_cov={hc})")
    if missing:
        print(f"  MISSING (build/ship first): {missing}")
    json.dump({"root": str(SHIP), "arm_map": ARM_MAP, "built": [b[0] for b in built],
               "missing": [m[0] for m in missing], "null_arms": sorted(NULL_ARMS), "arms": manifest_arms,
               "_note": "held_out_coverage=0 on a non-null arm => it CANNOT emit for any held-out perturbation "
                        "(zeroshot structural zero). The gauntlet refuses such arms on zeroshot unless "
                        "allow_zero_coverage is set (then it records them as the finding)."},
              open(DATA_OUT / "arm_graphs_manifest.json", "w"), indent=2)
    return [b[0] for b in built]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(EXP / "ship"),
                    help="bundle root that IS ship/ (arms at <root>/arms, LFC at <root>/lfc_*.npz, package "
                         "written to <root>/data). Default: <EXP>/ship.")
    args = ap.parse_args()
    configure(args.root)
    print(f"[build_rpe1_gauntlet_data] root={SHIP}  data_out={DATA_OUT}  lfc={LFC.name if LFC else None}")
    _, pidx, split = build_data()
    built = build_arm_graphs(pidx, split)
    print(f"RPE1 gauntlet data package -> {DATA_OUT}  ({len(built)} arms)")


if __name__ == "__main__":
    main()
