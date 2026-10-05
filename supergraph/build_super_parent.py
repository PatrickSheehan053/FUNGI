"""
exp_035 Step 2 — build_super_parent.py : fuse the two hybrid ctrl-preserved dense pillars into ONE directed
dense Super parent (Regulator/Target/Weight, gene names), and (Stage 2) build the fusion-order variants.

NOVEL code (CLONE-ONLY discipline: never edits FUNGI/SHROOM/HYPHAE/OBJECTS). Reuses exp_033 graph_tools.

Fusion (both stages): within-source rank-normalize each pillar (graph_tools.rank_normalize applied per source
group), then RANK-MAX UNION over E_causal u E_coexp (edge weight = max of the two within-source rank scores;
absent-in-a-pillar -> 0; direction preserved; NOT Borda-capped — this is a *dense* parent, FUNGI prunes later).

Leakage firewall: the causal (SHROOM) pillar's SOURCES are restricted to train-perturbed regulators (exactly
as exp_033 build_rpe1_graphs.get_dense("causal")) whenever it enters the fusion, in EVERY fusion order — only
gene identities cross to held-out, never held-out out-edges (zeroshot). Toggle with --no-restrict-causal-train.

Stage-1 (fuse-then-prune / FO0):
  python build_super_parent.py --mode fuse_then_prune --out .../super_dense_rpe1.parquet
Stage-2 variants (the intermediate pillar prunes are produced by fungi_prune.py first; passed here):
  --mode prune_hyphae_first  --hyphae-pruned <champ.parquet>                       (FO2: pruned HYPHAE + dense SHROOM)
  --mode prune_shroom_first  --shroom-pruned <champ.parquet>                       (FO3: pruned SHROOM + dense HYPHAE)
  --mode prune_both_first    --hyphae-pruned <champ.parquet> --shroom-pruned <..>  (FO4: both pruned)
Negative control (fail-on-shuffle): degree-preserving Maslov-Sneppen shuffle of an existing parent:
  python build_super_parent.py --shuffle-of <parent.parquet> --out <parent_shuffled.parquet>

The reusable primitives (rank_max_union, within_source_ranknorm, qc_report) are importable by Stage-2 drivers.
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_tools as GT

REPO = Path("c:/Users/studi/OneDrive/Documents/thesis")
EXP = HERE.parent
LFC = REPO / "DATA/EXPERIMENTS/exp_030_source_partitioned_supergraph/intermediate/graphs/lfc_targets_rpe1_cellhalf.npz"
DENSE_CAUSAL = REPO / "DATA/EXPERIMENTS/exp_022_substrate_coverage_recovery/intermediate/shroom_recovered/RPE1_recovered2_selftrain_dense_graph.parquet"
DENSE_COEXP = REPO / "HYPHAE/for_chinmaya/HYPHAE_RPE1_hybrid_ctrlpreserved_dense_graph.parquet"
DENSE_CACHE = EXP / "intermediate/dense_cache"
N = 5000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- meta / pillars
def load_meta():
    z = np.load(LFC, allow_pickle=True)
    panel = [str(x) for x in z["panel"]]
    g2i = {g: i for i, g in enumerate(panel)}
    train_pidx = z["train_pidx"].astype(np.int64)
    held_pidx = np.concatenate([z["val_pidx"], z["test_pidx"]]).astype(np.int64)
    return panel, g2i, train_pidx, held_pidx


def restrict_to_train_sources(s, t, w, train_pidx):
    """Keep out-edges only from train-perturbed regulators (the causal/leakage firewall)."""
    trset = np.zeros(N, bool); trset[np.asarray(train_pidx)] = True
    m = trset[s]
    return s[m], t[m], w[m]


def get_causal_dense(g2i, train_pidx, restrict=True):
    """Recovered SHROOM dense edges (src,tgt,w), restricted to train-perturbed sources. Cached to npz."""
    DENSE_CACHE.mkdir(parents=True, exist_ok=True)
    tag = "causal_trainrestrict" if restrict else "causal_full"
    cp = DENSE_CACHE / f"dense_{tag}.npz"
    if cp.exists():
        z = np.load(cp); return z["s"], z["t"], z["w"]
    s, t, w = GT.load_dense(DENSE_CAUSAL, g2i)   # cols auto-detected (Regulator/Target/Importance)
    if restrict:
        s, t, w = restrict_to_train_sources(s, t, w, train_pidx)
        log(f"causal dense restricted to train sources: {len(s):,} edges, {len(np.unique(s))} src")
    else:
        log(f"causal dense (full): {len(s):,} edges, {len(np.unique(s))} src")
    np.savez(cp, s=s, t=t, w=w)
    return s, t, w


def get_coexp_dense(g2i):
    """Full HYPHAE co-expression dense edges (all 5000 genes are regulators). Cached to npz."""
    DENSE_CACHE.mkdir(parents=True, exist_ok=True)
    cp = DENSE_CACHE / "dense_coexp_full.npz"
    if cp.exists():
        z = np.load(cp); return z["s"], z["t"], z["w"]
    s, t, w = GT.load_dense(DENSE_COEXP, g2i)
    log(f"coexp dense (full): {len(s):,} edges, {len(np.unique(s))} src")
    np.savez(cp, s=s, t=t, w=w)
    return s, t, w


def load_pruned_pillar(parquet_path, g2i, restrict_train=None):
    """Load a fungi_prune champion pillar (Regulator/Target/Weight). Optionally restrict sources to train."""
    s, t, w = GT.load_dense(Path(parquet_path), g2i, reg="Regulator", tgt="Target", w="Weight")
    if restrict_train is not None:
        s, t, w = restrict_to_train_sources(s, t, w, restrict_train)
        log(f"pruned pillar {Path(parquet_path).name} restricted to train sources: {len(s):,} edges")
    else:
        log(f"pruned pillar {Path(parquet_path).name}: {len(s):,} edges, {len(np.unique(s))} src")
    return s, t, w


# ---------------------------------------------------------------- fusion primitives (reusable by Stage 2)
def within_source_ranknorm(s, t, w):
    """Per-source rank-normalization: apply graph_tools.rank_normalize to each source's out-edge weights.
    Returns a rank score in (0,1] per edge, comparable across pillars regardless of a source's out-degree."""
    s = np.asarray(s); w = np.asarray(w, np.float64)
    out = np.empty(len(w), np.float64)
    order = np.argsort(s, kind="stable"); ss = s[order]; ws = w[order]
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1, [len(ss)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[order[a:b]] = GT.rank_normalize(ws[a:b])
    return out


def rank_union(edgesA, edgesB, n_genes=N, mode="max", w_a=0.5):
    """UNION over E_A u E_B of within-source rank scores (absent-in-a-pillar -> 0). Direction preserved.
      mode="max"      -> fused = max(scoreA, scoreB)                       (Stage-1 default)
      mode="weighted" -> fused = w_a*scoreA + (1-w_a)*scoreB              (Task 2 fusion-balance; A=causal)
      mode="maxw"     -> fused = max(w_a*scoreA, (1-w_a)*scoreB)          (causal-CAPPED max, 2026-07-17)
    Returns (src, tgt, fused_weight, in_A_mask) where in_A_mask marks union edges present in pillar A
    (= causal membership when A is the causal pillar). THE reusable union step for all fusion orders.

    COVERAGE NOTE (exp_035, 2026-07-17): "weighted" is what collapses the Super parent to ~100% causal.
    A causal edge is present in BOTH pillars (the coexp parent is near-complete), so it scores
    w_a*scoreA + (1-w_a)*scoreB, strictly above a coexp-only edge's (1-w_a)*scoreB at equal scoreB --
    causal wins every tie by construction, at EVERY w_a. "max" does not double-count and preserves
    coexp; "maxw" caps the causal pillar's ceiling at w_a so coexp-only edges can outrank it."""
    sA, tA, rA = edgesA; sB, tB, rB = edgesB
    kA = np.asarray(sA, np.int64) * n_genes + np.asarray(tA, np.int64)
    kB = np.asarray(sB, np.int64) * n_genes + np.asarray(tB, np.int64)
    rA = np.asarray(rA, np.float64); rB = np.asarray(rB, np.float64)
    uk = np.union1d(kA, kB)
    scoreA = np.zeros(len(uk)); scoreB = np.zeros(len(uk))
    oA = np.argsort(kA, kind="stable"); kAs = kA[oA]; rAs = rA[oA]
    pA = np.clip(np.searchsorted(kAs, uk), 0, len(kAs) - 1); hA = kAs[pA] == uk; scoreA[hA] = rAs[pA[hA]]
    oB = np.argsort(kB, kind="stable"); kBs = kB[oB]; rBs = rB[oB]
    pB = np.clip(np.searchsorted(kBs, uk), 0, len(kBs) - 1); hB = kBs[pB] == uk; scoreB[hB] = rBs[pB[hB]]
    if mode == "weighted":
        fused = float(w_a) * scoreA + (1.0 - float(w_a)) * scoreB
    elif mode == "maxw":
        fused = np.maximum(float(w_a) * scoreA, (1.0 - float(w_a)) * scoreB)
    else:
        fused = np.maximum(scoreA, scoreB)
    us = (uk // n_genes).astype(np.int64); ut = (uk % n_genes).astype(np.int64)
    return us, ut, fused, hA


def rank_max_union(edgesA, edgesB, n_genes=N):
    """Back-compat wrapper (Stage-1/Stage-2): rank-max union, returns the 3-tuple (src,tgt,fused)."""
    us, ut, fused, _ = rank_union(edgesA, edgesB, n_genes, mode="max")
    return us, ut, fused


def fuse_pillars(causal_edges, coexp_edges, n_genes=N, fusion_wc=None, causal_norm="within_source"):
    """rank-normalize each pillar, then union. fusion_wc=None -> rank-max union (Stage 1); float -> weighted
    w_c*rank_causal + (1-w_c)*rank_coexp (Task 2). causal_norm controls the CAUSAL pillar's normalization:
      'within_source' (Stage-1 default) -> per-source rank (flattens cross-source magnitude -> low S_max);
      'global'        -> global rank of the raw SHROOM weights (PRESERVES causal hub magnitude -> raises S_max).
    Coexp is ALWAYS within-source (prevents the 24.7M-edge coexp pillar from swamping). Returns
    (src, tgt, fused, causal_mask)."""
    cs, ct, cw = causal_edges; hs, ht, hw = coexp_edges
    if causal_norm == "global":
        rC = rankdata(np.asarray(cw, np.float64), method="average") / len(cw) if len(cw) else cw
    else:
        rC = within_source_ranknorm(cs, ct, cw)
    rH = within_source_ranknorm(hs, ht, hw)
    mode = "max" if fusion_wc is None else "weighted"
    return rank_union((cs, ct, rC), (hs, ht, rH), n_genes, mode=mode, w_a=(fusion_wc or 0.5))


# ---------------------------------------------------------------- QC + IO
def reciprocated_fraction(s, t, n_genes=N):
    """Fraction of directed edges (i->j) whose reverse (j->i) also exists in the edge set."""
    if len(s) == 0:
        return 0.0
    k = np.asarray(s, np.int64) * n_genes + np.asarray(t, np.int64)
    ks = np.sort(k)
    rk = np.asarray(t, np.int64) * n_genes + np.asarray(s, np.int64)   # reverse keys
    pos = np.clip(np.searchsorted(ks, rk), 0, len(ks) - 1)
    hit = ks[pos] == rk
    return float(hit.mean())


def degree_summary(idx, n_genes=N):
    d = np.bincount(np.asarray(idx, np.int64), minlength=n_genes)
    nz = d[d > 0]
    q = lambda p: float(np.percentile(d, p))
    return dict(n_nodes_nonzero=int((d > 0).sum()), min=int(d.min()), p10=q(10), median=float(np.median(d)),
                mean=round(float(d.mean()), 3), p90=q(90), max=int(d.max()),
                mean_nonzero=round(float(nz.mean()) if len(nz) else 0.0, 3))


def qc_report(s, t, w, panel, train_pidx, held_pidx, mode, n_genes=N):
    cov = GT.coverage_report(s, t, train_pidx, held_pidx)
    dens = len(s) / (n_genes * n_genes)
    rf = reciprocated_fraction(s, t, n_genes)
    qc = dict(
        mode=mode, n_edges=int(len(s)), n_genes=int(n_genes), density=round(dens, 6),
        n_sources=int(cov["n_src"]),
        train_source_coverage=cov["train_coverage"], held_out_source_coverage=cov["held_out_coverage"],
        n_src_in_train=cov["n_src_in_train"], n_src_in_heldout=cov["n_src_in_heldout"],
        reciprocated_pair_fraction=round(rf, 6),
        out_degree=degree_summary(s, n_genes), in_degree=degree_summary(t, n_genes),
        weight_min=round(float(np.min(w)), 6), weight_max=round(float(np.max(w)), 6),
        weight_mean=round(float(np.mean(w)), 6),
        density_gate_pass=bool(dens >= 0.25),
        coverage_nondegenerate=bool(cov["n_src"] > 0 and cov["train_coverage"] > 0.0))
    qc["QC_PASS"] = bool(qc["density_gate_pass"] and qc["coverage_nondegenerate"])
    return qc


def write_parent(s, t, w, panel, out_path):
    df = pd.DataFrame({"Regulator": [panel[i] for i in s], "Target": [panel[i] for i in t],
                       "Weight": np.asarray(w, np.float32)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log(f"parent -> {out_path} ({len(df):,} edges)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fuse_then_prune",
                    choices=["fuse_then_prune", "prune_hyphae_first", "prune_shroom_first", "prune_both_first"])
    ap.add_argument("--out", required=False, help="output parquet for the fused parent")
    ap.add_argument("--qc-out", default=None, help="QC json path (default: <out>_QC.json)")
    ap.add_argument("--hyphae-pruned", default=None, help="pruned HYPHAE champion parquet (FO2/FO4)")
    ap.add_argument("--shroom-pruned", default=None, help="pruned SHROOM champion parquet (FO3/FO4)")
    ap.add_argument("--no-restrict-causal-train", action="store_true",
                    help="do NOT restrict the causal/SHROOM pillar to train-perturbed sources (default: restrict)")
    ap.add_argument("--shuffle-of", default=None,
                    help="degree-preserving shuffle of an existing parent parquet -> --out")
    ap.add_argument("--shuffle-method", default="weights", choices=["weights", "wiring"],
                    help="weights = per-source weight permutation (correct for a ~dense parent, where a "
                         "Maslov-Sneppen wiring shuffle is a near no-op); wiring = scramble_ms (sparse graphs)")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--fusion-wc", type=float, default=None,
                    help="Task 2 fusion-balance weight on the CAUSAL pillar (w_c in [0,1]); coexp gets 1-w_c. "
                         "None = rank-max union (Stage-1 default). Higher w_c -> causal edges survive pruning.")
    ap.add_argument("--causal-norm", default="within_source", choices=["within_source", "global"],
                    help="CAUSAL pillar normalization. 'global' PRESERVES causal hub magnitude -> raises S_max "
                         "(the real S_max lever; Checkpoint-1 finding). Coexp always within-source.")
    args = ap.parse_args()

    panel, g2i, train_pidx, held_pidx = load_meta()
    restrict = not args.no_restrict_causal_train
    log(f"panel N={len(panel)} train_perts={len(train_pidx)} held_perts={len(held_pidx)} "
        f"mode={args.mode if not args.shuffle_of else 'shuffle'} restrict_causal_train={restrict}")

    # ---- shuffle mode (NEG control) ----
    if args.shuffle_of:
        assert args.out, "--shuffle-of requires --out"
        s, t, w = GT.load_dense(Path(args.shuffle_of), g2i, reg="Regulator", tgt="Target", w="Weight")
        log(f"loaded parent to shuffle: {len(s):,} edges  method={args.shuffle_method}")
        if args.shuffle_method == "wiring":
            ss, st, sw, info = GT.scramble_ms(s, t, w, seed=args.shuffle_seed)
            log(f"Maslov-Sneppen wiring shuffle: {info['accepted_swaps']:,} swaps ({info['passes']} passes)")
        else:
            # per-source weight permutation: preserves out/in-degree AND each source's weight multiset
            # (the scramble_ms invariants), randomizes which target a source strongly weights. Degree-
            # preserving structure-destroying null appropriate for a ~dense parent.
            rng = np.random.default_rng(args.shuffle_seed)
            ss, st, sw = s.copy(), t.copy(), w.astype(np.float64).copy()
            order = np.argsort(ss, kind="stable"); ss_o = ss[order]
            bounds = np.concatenate(([0], np.flatnonzero(np.diff(ss_o)) + 1, [len(ss_o)]))
            for a, b in zip(bounds[:-1], bounds[1:]):
                seg = order[a:b]
                sw[seg] = sw[seg][rng.permutation(len(seg))]
            info = {"method": "persource_weight_perm", "n_sources": int(len(bounds) - 1)}
            log(f"per-source weight permutation over {info['n_sources']} sources")
        write_parent(ss, st, sw, panel, args.out)
        qc = qc_report(ss, st, sw, panel, train_pidx, held_pidx, f"shuffle_{args.shuffle_method}")
        qc["shuffle_of"] = str(args.shuffle_of); qc["shuffle_method"] = args.shuffle_method
        qc["scramble_info"] = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in info.items()}
        qc_out = args.qc_out or (str(Path(args.out).with_suffix("")) + "_QC.json")
        json.dump(qc, open(qc_out, "w"), indent=2)
        log(f"QC -> {qc_out}  QC_PASS={qc['QC_PASS']} density={qc['density']} recip={qc['reciprocated_pair_fraction']}")
        return

    assert args.out, "--out required"
    # ---- select pillar inputs by fusion mode ----
    if args.mode == "fuse_then_prune":
        causal = get_causal_dense(g2i, train_pidx, restrict=restrict)
        coexp = get_coexp_dense(g2i)
    elif args.mode == "prune_hyphae_first":
        assert args.hyphae_pruned, "prune_hyphae_first needs --hyphae-pruned"
        causal = get_causal_dense(g2i, train_pidx, restrict=restrict)
        coexp = load_pruned_pillar(args.hyphae_pruned, g2i)
    elif args.mode == "prune_shroom_first":
        assert args.shroom_pruned, "prune_shroom_first needs --shroom-pruned"
        causal = load_pruned_pillar(args.shroom_pruned, g2i, restrict_train=(train_pidx if restrict else None))
        coexp = get_coexp_dense(g2i)
    elif args.mode == "prune_both_first":
        assert args.hyphae_pruned and args.shroom_pruned, "prune_both_first needs both --hyphae-pruned and --shroom-pruned"
        causal = load_pruned_pillar(args.shroom_pruned, g2i, restrict_train=(train_pidx if restrict else None))
        coexp = load_pruned_pillar(args.hyphae_pruned, g2i)

    us, ut, uf, causal_mask = fuse_pillars(causal, coexp, N, fusion_wc=args.fusion_wc,
                                           causal_norm=args.causal_norm)
    log(f"fused: {len(us):,} edges  (causal={len(causal[0]):,}  coexp={len(coexp[0]):,})  "
        f"fusion_wc={args.fusion_wc}  causal_norm={args.causal_norm}  causal_frac_full={causal_mask.mean():.4f}")
    write_parent(us, ut, uf, panel, args.out)

    qc = qc_report(us, ut, uf, panel, train_pidx, held_pidx, args.mode)
    qc["restrict_causal_train"] = bool(restrict)
    qc["fusion_wc"] = args.fusion_wc
    qc["causal_norm"] = args.causal_norm
    qc["causal_frac_full_union"] = round(float(causal_mask.mean()), 4)
    # Task 2 diagnostic: causal-edge fraction SURVIVING a top-lambda prune (actual balance after pruning,
    # not just the input edge-count ratio). Shows whether causal structure survives to shape topology.
    for lam_probe in (20, 44, 70):
        E = min(int(lam_probe * N), len(uf))
        top = np.argpartition(uf, -E)[-E:]
        qc[f"causal_frac_top_lam{lam_probe}"] = round(float(causal_mask[top].mean()), 4)
    qc["pillars"] = dict(causal_edges=int(len(causal[0])), coexp_edges=int(len(coexp[0])),
                         hyphae_pruned=args.hyphae_pruned, shroom_pruned=args.shroom_pruned)
    qc_out = args.qc_out or (str(Path(args.out).with_suffix("")) + "_QC.json")
    json.dump(qc, open(qc_out, "w"), indent=2)
    log(f"QC -> {qc_out}")
    log(f"  QC_PASS={qc['QC_PASS']} density={qc['density']} (gate>=0.25:{qc['density_gate_pass']}) "
        f"recip_frac={qc['reciprocated_pair_fraction']} n_src={qc['n_sources']} "
        f"train_cov={qc['train_source_coverage']} held_cov={qc['held_out_source_coverage']}")
    if not qc["QC_PASS"]:
        log("!!! QC GATE FAILED — do NOT proceed to caching; report to Patrick.")


if __name__ == "__main__":
    main()
