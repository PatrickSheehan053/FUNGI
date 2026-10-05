"""
exp_033 — build_h1hesc_graphs.py : assemble h1-hESC arm graphs from the FUNGI champions.

Mirrors build_rpe1_graphs.py (same proven logic, same graph_tools library) but for the h1-hESC cell line:
  N = 5024; panel = vcc_train_hybrid var_names; train/val/test perturbations from VCC_h1hESC_split_indices.json
  (label-based). Causal dense = the FRESH h1-hESC SHROOM; Co-exp dense = the h1-hESC HYPHAE parent.
FUNGI champion tags are h1_ prefixed (h1_causal_lam44 / h1_coexp_lam44) to avoid the RPE1 tag namespace.
Output arms -> intermediate/graphs/h1_hesc/<armid>__<label>.npz (harness format).

  python build_h1hesc_graphs.py --point lam220k
"""
from __future__ import annotations
import os, sys, json, argparse
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
from pathlib import Path
import numpy as np
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_tools as GT

EXP = HERE.parent
BUNDLE = EXP / "exp033_h1hESC_bundle"
DENSE_CAUSAL = EXP / "intermediate/h1_hesc/shroom/h1hESC_selftrain_dense_graph.parquet"
DENSE_COEXP = BUNDLE / "hyphae_h1hESC_dense_10JUL.parquet"
CHAMP = EXP / "intermediate/fungi/champions_full"
GRAPHS = EXP / "intermediate/graphs/h1_hesc"
CACHE = EXP / "intermediate/dense_cache_h1hesc"
N = 5024

POINTS = {
    "lam120k": ("h1_causal_lam24", "h1_coexp_lam24"),
    "lam220k": ("h1_causal_lam44", "h1_coexp_lam44"),   # primary
    "lam350k": ("h1_causal_lam70", "h1_coexp_lam70"),
}
PRIMARY = "lam220k"


def load_meta():
    a = ad.read_h5ad(BUNDLE / "vcc_train_hybrid.h5ad", backed="r")
    panel = [str(x) for x in a.var_names]; g2i = {g: i for i, g in enumerate(panel)}
    sj = json.load(open(BUNDLE / "VCC_h1hESC_split_indices.json"))
    def to_idx(labels): return np.array(sorted(g2i[l] for l in labels if l in g2i), np.int64)
    train_pidx = to_idx(sj["train_labels"])
    held_pidx = np.concatenate([to_idx(sj["val_labels"]), to_idx(sj["test_labels"])])
    return panel, g2i, train_pidx, held_pidx


def get_dense(kind, g2i, train_pidx):
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"dense_{kind}.npz"
    if cp.exists():
        z = np.load(cp); return z["s"], z["t"], z["w"]
    if kind == "causal":
        s, t, w = GT.load_dense(DENSE_CAUSAL, g2i)
        trset = np.zeros(N, bool); trset[np.asarray(train_pidx)] = True
        m = trset[s]; s, t, w = s[m], t[m], w[m]
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
    return GT.load_dense(p, g2i, reg="Regulator", tgt="Target", w="Weight")


def write_arm(armid, label, s, t, w_raw, train_pidx, held_pidx, manifest, substrate, pruner, extra=None):
    GRAPHS.mkdir(parents=True, exist_ok=True)
    stem = f"{armid}__{label}"
    GT.save_arm(GRAPHS / f"{stem}.npz", s, t, w_raw)
    cov = GT.coverage_report(s, t, train_pidx, held_pidx)
    rec = dict(arm=armid, label=label, substrate=substrate, pruner=pruner, lam_label=label, **cov)
    if extra: rec.update(extra)
    manifest[stem] = rec
    GT.log(f"  {stem:28s} E={cov['n_edges']:>7,} n_src={cov['n_src']:>4} "
           f"held_cov={cov['held_out_coverage']:.3f} train_cov={cov['train_coverage']:.3f}")
    return stem


def build_point(label, g2i, train_pidx, held_pidx, causal_dense, coexp_dense, manifest, sizes):
    ctag, htag = POINTS[label]
    fc = load_champion(ctag, g2i); fh = load_champion(htag, g2i)
    if fc is None or fh is None:
        GT.log(f"[skip] {label}: champions missing (causal={fc is not None} coexp={fh is not None})")
        return False
    fcs, fct, fcw = fc; fhs, fht, fhw = fh
    E_c = len(fcs); E_h = len(fhs); CAP = E_c
    sizes[label] = {"E_causal": int(E_c), "E_coexp": int(E_h), "super_cap": int(CAP),
                    "causal_tag": ctag, "coexp_tag": htag}
    GT.log(f"[{label}] FUNGI E_causal={E_c:,}  E_coexp={E_h:,}  super_cap={CAP:,}")
    write_arm("fungi_causal", label, fcs, fct, fcw, train_pidx, held_pidx, manifest, "Causal", "FUNGI")
    cds, cdt, cdw = causal_dense
    tws, twt, tww = GT.topweight_prune(cds, cdt, cdw, E_c)
    write_arm("tw_causal", label, tws, twt, tww, train_pidx, held_pidx, manifest, "Causal", "Topweight")
    kns, knt, knw = GT.knn_exact(cds, cdt, cdw, E_c)
    write_arm("knn_causal", label, kns, knt, knw, train_pidx, held_pidx, manifest, "Causal", "KNN")
    write_arm("fungi_coexp", label, fhs, fht, fhw, train_pidx, held_pidx, manifest, "Coexp", "FUNGI")
    hds, hdt, hdw = coexp_dense
    htws, htwt, htww = GT.topweight_prune(hds, hdt, hdw, E_h)
    write_arm("tw_coexp", label, htws, htwt, htww, train_pidx, held_pidx, manifest, "Coexp", "Topweight")
    hkns, hknt, hknw = GT.knn_exact(hds, hdt, hdw, E_h)
    write_arm("knn_coexp", label, hkns, hknt, hknw, train_pidx, held_pidx, manifest, "Coexp", "KNN")
    for armid, (a, b) in {
            "fungi_super": ((fcs, fct, fcw), (fhs, fht, fhw)),
            "tw_super":    ((tws, twt, tww), (htws, htwt, htww)),
            "knn_super":   ((kns, knt, knw), (hkns, hknt, hknw))}.items():
        us, ut, uf = GT.borda_fuse(a, b, N, cap=CAP)
        pruner = {"fungi_super": "FUNGI", "tw_super": "Topweight", "knn_super": "KNN"}[armid]
        write_arm(armid, label, us, ut, uf, train_pidx, held_pidx, manifest, "Super", pruner,
                  extra={"super_cap": int(CAP)})
    def E(a): return manifest[f"{a}__{label}"]["n_edges"]
    assert E("fungi_causal") == E("tw_causal") == E("knn_causal") == E_c, "Causal edge-match FAILED"
    assert E("fungi_coexp") == E("tw_coexp") == E("knn_coexp") == E_h, "Coexp edge-match FAILED"
    assert E("fungi_super") == E("tw_super") == E("knn_super") == CAP, "Super edge-match FAILED"
    GT.log(f"[{label}] edge-match asserts PASSED (causal={E_c:,} coexp={E_h:,} super={CAP:,})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    panel, g2i, train_pidx, held_pidx = load_meta()
    GT.log(f"h1-hESC panel N={len(panel)} train_perts={len(train_pidx)} held_perts={len(held_pidx)}")
    causal_dense = get_dense("causal", g2i, train_pidx)
    coexp_dense = get_dense("coexp", g2i, train_pidx)
    META = GRAPHS / "_meta"; META.mkdir(parents=True, exist_ok=True)   # keep the shipment dir npz-ONLY
    man_path = META / "manifest.json"; sizes_path = META / "fungi_sizes.json"
    manifest = json.load(open(man_path)) if man_path.exists() and not args.force else {}
    sizes = json.load(open(sizes_path)) if sizes_path.exists() and not args.force else {}
    labels = list(POINTS) if args.all else ([args.point] if args.point else [PRIMARY])
    for label in labels:
        build_point(label, g2i, train_pidx, held_pidx, causal_dense, coexp_dense, manifest, sizes)
        json.dump(manifest, open(man_path, "w"), indent=2, default=float)
        json.dump(sizes, open(sizes_path, "w"), indent=2, default=float)
    GT.log(f"DONE -> {man_path}")


if __name__ == "__main__":
    main()
