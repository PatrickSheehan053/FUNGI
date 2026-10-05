"""
exp_033 — build_rpe1_graphs.py : assemble every RPE1 arm graph from the FUNGI size-setter champions.

For each density point (label -> causal/coexp FUNGI champion tags) this builds the 3x3 grid:
  FUNGI Causal   (from champion)            | Topweight Causal | KNN Causal   (matched to E_causal)
  FUNGI Co-exp   (from champion)            | Topweight Co-exp | KNN Co-exp    (matched to E_coexp)
  FUNGI Super    = borda(FUNGI pillars,cap) | Topweight Super  | KNN Super     (same borda, same cap)

Every (FUNGI, Topweight, KNN) triplet is EDGE-COUNT IDENTICAL within its substrate (asserted). Causal TW/KNN
are pruned from the SHROOM dense RESTRICTED to train-perturbed sources (out-edges only from perturbed
regulators = the causal definition; keeps the whole Causal column structurally dark at zeroshot). Co-exp
TW/KNN prune the full HYPHAE dense (all 5024 genes are regulators; scores at zeroshot).

Output: intermediate/graphs/rpe1/<armid>__<label>.npz (src/tgt/w_outnorm/w_raw) + a parquet edge-list twin +
intermediate/graphs/rpe1/fungi_sizes.json + manifest.json. CPU-only; run when the GPU is idle (dense-parent
load is CPU-heavy). Resumable: --force to rebuild.

  python build_rpe1_graphs.py --point lam220k     # one density point
  python build_rpe1_graphs.py --all               # every point whose FUNGI champions exist
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_tools as GT

REPO = Path("c:/Users/studi/OneDrive/Documents/thesis")
EXP = HERE.parent
LFC = REPO / "DATA/EXPERIMENTS/exp_030_source_partitioned_supergraph/intermediate/graphs/lfc_targets_rpe1_cellhalf.npz"
DENSE_CAUSAL = REPO / "DATA/EXPERIMENTS/exp_022_substrate_coverage_recovery/intermediate/shroom_recovered/RPE1_recovered2_selftrain_dense_graph.parquet"
DENSE_COEXP = REPO / "HYPHAE/for_chinmaya/HYPHAE_RPE1_hybrid_ctrlpreserved_dense_graph.parquet"
CHAMP = EXP / "intermediate/fungi/champions_full"
GRAPHS = EXP / "intermediate/graphs/rpe1"
CACHE = EXP / "intermediate/dense_cache"
N = 5000

# density points: label -> (causal champion tag, coexp champion tag)
POINTS = {
    "lam120k": ("causal_lam24", "coexp_lam24"),
    "lam220k": ("causal_lam44", "coexp_lam44"),   # primary
    "lam350k": ("causal_lam70", "coexp_lam70"),
}
PRIMARY = "lam220k"


def load_meta():
    z = np.load(LFC, allow_pickle=True)
    panel = [str(x) for x in z["panel"]]; g2i = {g: i for i, g in enumerate(panel)}
    train_pidx = z["train_pidx"].astype(np.int64)
    held_pidx = np.concatenate([z["val_pidx"], z["test_pidx"]]).astype(np.int64)
    return panel, g2i, train_pidx, held_pidx


def get_dense(kind, g2i, train_pidx):
    """Mapped dense parent edges (src,tgt,w), cached to npz. Causal is RESTRICTED to train-perturbed sources."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"dense_{kind}.npz"
    if cp.exists():
        z = np.load(cp); return z["s"], z["t"], z["w"]
    if kind == "causal":
        s, t, w = GT.load_dense(DENSE_CAUSAL, g2i)
        trset = np.zeros(N, bool); trset[np.asarray(train_pidx)] = True
        m = trset[s]                                   # keep out-edges only from train-perturbed regulators
        s, t, w = s[m], t[m], w[m]
        GT.log(f"causal dense restricted to train sources: {len(s):,} edges, {len(np.unique(s))} src")
    else:
        s, t, w = GT.load_dense(DENSE_COEXP, g2i)
        GT.log(f"coexp dense (full): {len(s):,} edges, {len(np.unique(s))} src")
    np.savez(cp, s=s, t=t, w=w)
    return s, t, w


def load_champion(tag, g2i):
    p = CHAMP / f"{tag}.parquet"
    if not p.exists():
        return None
    s, t, w = GT.load_dense(p, g2i, reg="Regulator", tgt="Target", w="Weight")
    return s, t, w


def write_arm(armid, label, s, t, w_raw, train_pidx, held_pidx, manifest, substrate, pruner, lam_label, extra=None):
    GRAPHS.mkdir(parents=True, exist_ok=True)
    stem = f"{armid}__{label}"
    GT.save_arm(GRAPHS / f"{stem}.npz", s, t, w_raw)
    # parquet edge-list twin (portability/inspection)
    cov = GT.coverage_report(s, t, train_pidx, held_pidx)
    rec = dict(arm=armid, label=label, substrate=substrate, pruner=pruner, lam_label=lam_label, **cov)
    if extra: rec.update(extra)
    manifest[stem] = rec
    GT.log(f"  {stem:28s} E={cov['n_edges']:>7,} n_src={cov['n_src']:>4} "
           f"held_cov={cov['held_out_coverage']:.3f} train_cov={cov['train_coverage']:.3f}")
    return stem


def build_point(label, g2i, train_pidx, held_pidx, causal_dense, coexp_dense, manifest, sizes, force=False):
    ctag, htag = POINTS[label]
    fc = load_champion(ctag, g2i); fh = load_champion(htag, g2i)
    if fc is None or fh is None:
        GT.log(f"[skip] {label}: champions missing (causal={fc is not None} coexp={fh is not None})")
        return False
    fcs, fct, fcw = fc; fhs, fht, fhw = fh
    E_c = len(fcs); E_h = len(fhs)
    CAP = E_c                                            # Super cap = causal density (all Super variants share it)
    sizes[label] = {"E_causal": int(E_c), "E_coexp": int(E_h), "super_cap": int(CAP),
                    "causal_tag": ctag, "coexp_tag": htag}
    GT.log(f"[{label}] FUNGI E_causal={E_c:,}  E_coexp={E_h:,}  super_cap={CAP:,}")

    # ---- Causal column (FUNGI + matched TW/KNN on train-restricted causal dense) ----
    write_arm("fungi_causal", label, fcs, fct, fcw, train_pidx, held_pidx, manifest, "Causal", "FUNGI", label)
    cds, cdt, cdw = causal_dense
    tws, twt, tww = GT.topweight_prune(cds, cdt, cdw, E_c)
    write_arm("tw_causal", label, tws, twt, tww, train_pidx, held_pidx, manifest, "Causal", "Topweight", label)
    kns, knt, knw = GT.knn_exact(cds, cdt, cdw, E_c)
    write_arm("knn_causal", label, kns, knt, knw, train_pidx, held_pidx, manifest, "Causal", "KNN", label)

    # ---- Co-exp column (FUNGI + matched TW/KNN on full coexp dense) ----
    write_arm("fungi_coexp", label, fhs, fht, fhw, train_pidx, held_pidx, manifest, "Coexp", "FUNGI", label)
    hds, hdt, hdw = coexp_dense
    htws, htwt, htww = GT.topweight_prune(hds, hdt, hdw, E_h)
    write_arm("tw_coexp", label, htws, htwt, htww, train_pidx, held_pidx, manifest, "Coexp", "Topweight", label)
    hkns, hknt, hknw = GT.knn_exact(hds, hdt, hdw, E_h)
    write_arm("knn_coexp", label, hkns, hknt, hknw, train_pidx, held_pidx, manifest, "Coexp", "KNN", label)

    # ---- Super column (borda-fuse the like-pruned pillars, same cap) ----
    for armid, (a, b) in {
            "fungi_super": ((fcs, fct, fcw), (fhs, fht, fhw)),
            "tw_super":    ((tws, twt, tww), (htws, htwt, htww)),
            "knn_super":   ((kns, knt, knw), (hkns, hknt, hknw))}.items():
        us, ut, uf = GT.borda_fuse(a, b, N, cap=CAP)
        pruner = {"fungi_super": "FUNGI", "tw_super": "Topweight", "knn_super": "KNN"}[armid]
        write_arm(armid, label, us, ut, uf, train_pidx, held_pidx, manifest, "Super", pruner, label,
                  extra={"super_cap": int(CAP)})

    # ---- edge-count match assertions ----
    def E(a): return manifest[f"{a}__{label}"]["n_edges"]
    assert E("fungi_causal") == E("tw_causal") == E("knn_causal") == E_c, "Causal edge-match FAILED"
    assert E("fungi_coexp") == E("tw_coexp") == E("knn_coexp") == E_h, "Coexp edge-match FAILED"
    assert E("fungi_super") == E("tw_super") == E("knn_super") == CAP, "Super edge-match FAILED"
    GT.log(f"[{label}] edge-match asserts PASSED (causal={E_c:,} coexp={E_h:,} super={CAP:,})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default=None, help="one density label e.g. lam220k")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    panel, g2i, train_pidx, held_pidx = load_meta()
    GT.log(f"panel N={len(panel)} train_perts={len(train_pidx)} held_perts={len(held_pidx)}")
    causal_dense = get_dense("causal", g2i, train_pidx)
    coexp_dense = get_dense("coexp", g2i, train_pidx)

    META = GRAPHS / "_meta"; META.mkdir(parents=True, exist_ok=True)   # keep shipment dir npz-only
    man_path = META / "manifest.json"; sizes_path = META / "fungi_sizes.json"
    # migrate any legacy top-level meta (batch 1 wrote these flat) into _meta on next run
    for legacy in (GRAPHS / "manifest.json", GRAPHS / "fungi_sizes.json"):
        if legacy.exists() and not (META / legacy.name).exists():
            try:
                json.dump(json.load(open(legacy)), open(META / legacy.name, "w"), indent=2)
            except Exception:
                pass
    manifest = json.load(open(man_path)) if man_path.exists() and not args.force else {}
    sizes = json.load(open(sizes_path)) if sizes_path.exists() and not args.force else {}

    labels = list(POINTS) if args.all else ([args.point] if args.point else [PRIMARY])
    built = []
    for label in labels:
        if build_point(label, g2i, train_pidx, held_pidx, causal_dense, coexp_dense, manifest, sizes, args.force):
            built.append(label)
        json.dump(manifest, open(man_path, "w"), indent=2, default=float)
        json.dump(sizes, open(sizes_path, "w"), indent=2, default=float)
    GT.log(f"DONE points={built} -> {man_path.name}, {sizes_path.name}")


if __name__ == "__main__":
    main()
