"""
exp_029 FIX B + FIX C — rebuild the per-arm nulls OF fungi_bio and the TRUE dense SHROOM parent.

Uses the SAME arbiter_lib machinery + gene-index space (panel from lfc_targets_rpe1.npz -> g2i) the original
ship/arms were built with, so every output is a drop-in ship/arms/<stem>.npz (keys src/tgt/w/n_src/n_edges).
Writes NEW files only (no overwrite / no deletion of the broken originals). CPU-only.

FIX B — nulls named for provenance, derived FROM fungi_bio (= ship/arms/shroom_fungi.npz):
  shuffle_of_fungi_bio : make_shuffle (target-permute) -> preserves fungi_bio's EXACT src set + out-degree
                         sequence + E (168138); only the edge targets are scrambled.
  reverse_of_fungi_bio : make_reverse (edge-direction flip) -> E (168138); the directional null of fungi_bio.

FIX C — shroom_denseparent = the un-pruned SHROOM parent, guaranteed to CONTAIN the fungi_bio champion:
  the shipped `shroom` was a top-200k-weight truncation of the dense SHROOM parquet that dropped 33 champion
  source genes (n_src 834 < 865) -> shroom_fungi ⊄ shroom, so the "FUNGI pruning contribution" ablation compared
  two different truncations. We rebuild it as  (ALL champion edges) ∪ (top-weight dense fill up to 200k),
  every edge carrying its SHROOM DENSE weight (pre-FUNGI), per-source sum-normed. edges(shroom_fungi) ⊆ this by
  construction. NOTE: still 0/373 held-out coverage on zeroshot (structural); this fix is for the rung1 ablation.

  python build_nulls_and_shroom.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
from pathlib import Path
import numpy as np

REPO = "c:/Users/studi/OneDrive/Documents/thesis"
EXP = Path(__file__).resolve().parents[2]
ARMS = EXP / "ship" / "arms"
LFC = EXP / "ship" / "lfc_targets_rpe1.npz"
SHROOM_DENSE = f"{REPO}/DATA/EXPERIMENTS/exp_022_substrate_coverage_recovery/intermediate/shroom_recovered/RPE1_recovered2_selftrain_dense_graph.parquet"
sys.path.insert(0, f"{REPO}/DATA/EXPERIMENTS/exp_025_hyphae_vs_shroom_ensemble/fungi_hyphae_prep/proto_exp025/src")
import arbiter_lib as AL
K_TOTAL = 200000                                    # shroom_denseparent budget (matches the original shroom cap)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    panel = list(np.load(LFC, allow_pickle=True)["panel"]); g2i = {g: i for i, g in enumerate(panel)}; N = len(panel)
    fz = np.load(ARMS / "shroom_fungi.npz")
    fs, ft, fw = fz["src"].astype(np.int64), fz["tgt"].astype(np.int64), fz["w"].astype(np.float64)
    E = len(fs); assert not (fs == ft).any(), "champion has self-loops (unexpected)"
    log(f"fungi_bio champion: E={E} n_src={len(np.unique(fs))}")

    # ---------------- FIX B: per-arm nulls OF fungi_bio ----------------
    AL.build_single_arm(*AL.make_shuffle(fs, ft, fw, N, seed=0), None, ARMS / "shuffle_of_fungi_bio.npz", mode="raw")
    AL.build_single_arm(*AL.make_reverse(fs, ft, fw), None, ARMS / "reverse_of_fungi_bio.npz", mode="raw")
    sh = np.load(ARMS / "shuffle_of_fungi_bio.npz"); rv = np.load(ARMS / "reverse_of_fungi_bio.npz")
    # verify: shuffle preserves fungi_bio's exact out-degree sequence + E; reverse flips (src<-tgt) + keeps E
    od_f = np.bincount(fs, minlength=N); od_s = np.bincount(sh["src"], minlength=N)
    shuffle_ok = bool(np.array_equal(np.sort(od_f), np.sort(od_s))) and int(sh["n_edges"]) == E
    reverse_ok = int(rv["n_edges"]) == E and bool(np.array_equal(np.sort(np.unique(rv["src"])),
                                                                 np.sort(np.unique(ft))))
    log(f"[FIX B] shuffle_of_fungi_bio  E={int(sh['n_edges'])} n_src={int(sh['n_src'])}  outdeg==fungi & E==E : {shuffle_ok}")
    log(f"[FIX B] reverse_of_fungi_bio  E={int(rv['n_edges'])} n_src={int(rv['n_src'])}  src==fungi.tgt & E==E : {reverse_ok}")
    assert shuffle_ok, "shuffle_of_fungi_bio does not preserve fungi_bio out-degree/E"
    assert reverse_ok, "reverse_of_fungi_bio malformed"

    # ---------------- FIX C: true dense SHROOM parent (contains the champion) ----------------
    log("loading dense SHROOM parent parquet (complete 5000x4999 graph) ...")
    ds, dt, dw = AL.load_dense(SHROOM_DENSE, g2i, reg="Regulator", tgt="Target", w="Importance")
    m = ds != dt; ds, dt, dw = ds[m], dt[m], dw[m]                       # drop self-loops
    dk = ds.astype(np.int64) * N + dt.astype(np.int64)
    ck = fs * N + ft                                                     # champion edge keys
    log(f"dense parent E={len(dk):,} (self-loops dropped); champion keys={len(ck):,}")
    # champion dense weights (searchsorted into sorted dense keys) — all present (complete graph)
    order = np.argsort(dk, kind="stable"); dk_s = dk[order]; dw_s = dw[order]
    pos = np.searchsorted(dk_s, ck); pos = np.clip(pos, 0, len(dk_s) - 1)
    hit = dk_s[pos] == ck
    assert hit.all(), f"{int((~hit).sum())} champion edges absent from dense parent — cannot guarantee containment"
    champ_w = dw_s[pos]
    # fill: top-weight dense edges NOT in champion, up to K_TOTAL total
    in_champ = np.isin(dk, ck, assume_unique=False)
    fdk = dk[~in_champ]; fdw = dw[~in_champ]
    n_fill = max(0, K_TOTAL - len(ck))
    if n_fill and n_fill < len(fdk):
        top = np.argpartition(fdw, -n_fill)[-n_fill:]
        fdk, fdw = fdk[top], fdw[top]
    sel_k = np.concatenate([ck, fdk]); sel_w = np.concatenate([champ_w, fdw])
    ss = (sel_k // N).astype(np.int64); st = (sel_k % N).astype(np.int64)
    AL.build_single_arm(ss, st, sel_w, None, ARMS / "shroom_denseparent.npz", mode="raw")
    dp = np.load(ARMS / "shroom_denseparent.npz")
    # verify containment: edges(fungi_bio) ⊆ edges(shroom_denseparent) AND src(fungi_bio) ⊆ src(shroom_denseparent)
    dpk = set((dp["src"].astype(np.int64) * N + dp["tgt"].astype(np.int64)).tolist())
    edge_sub = set(ck.tolist()) <= dpk
    src_sub = set(np.unique(fs).tolist()) <= set(np.unique(dp["src"]).tolist())
    log(f"[FIX C] shroom_denseparent E={int(dp['n_edges'])} n_src={int(dp['n_src'])}  "
        f"edges(fungi) subset-of edges(shroom)={edge_sub}  src(fungi) subset-of src(shroom)={src_sub}")
    assert edge_sub and src_sub, "shroom_denseparent does NOT contain the champion — containment fix failed"

    rep = dict(
        shuffle_of_fungi_bio=dict(E=int(sh["n_edges"]), n_src=int(sh["n_src"]), outdeg_eq_fungi_and_E=shuffle_ok),
        reverse_of_fungi_bio=dict(E=int(rv["n_edges"]), n_src=int(rv["n_src"]), well_formed=reverse_ok),
        shroom_denseparent=dict(E=int(dp["n_edges"]), n_src=int(dp["n_src"]),
                                edges_contain_champion=bool(edge_sub), src_contain_champion=bool(src_sub),
                                n_champion_edges=int(len(ck)), n_fill_edges=int(len(fdk))),
        secs=round(time.time() - t0, 1))
    json.dump(rep, open(EXP / "fix_source_coverage" / "diagnostics" / "build_nulls_and_shroom_report.json", "w"),
              indent=2, default=int)
    log(f"DONE ({time.time()-t0:.0f}s) -> wrote shuffle_of_fungi_bio / reverse_of_fungi_bio / shroom_denseparent to ship/arms/")


if __name__ == "__main__":
    main()
