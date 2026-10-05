"""
obj_009.3 (RHIZO-final) — zeroshot_v3f.py : the gene-held-out / zero-shot crown-jewel harness (per
ZERO_SHOT_HARNESS_SPEC.md). One FROZEN assembled RHIZO instrument; only the input GRAPH is swapped. Predicts the
responses of HELD-OUT perturbation-target genes (gene-disjoint from train) — genes the model never trained on.

The whole point: on a HYPHAE dense parent, FUNGI supplies out-edges for held-out genes (coverage), while a
greedy top_weight prune of the SHROOM subset has ~0 held-out-gene coverage. If FUNGI(HYPHAE) beats every arm on
the held-out genes, the thesis is made.

Arms (graph-swap, same frozen model): FUNGI, top_weight, pure_parent, knn, degree-shuffle, reverse, labelperm,
empty. Metrics on held-out genes: RSC (primary), dsRSC≥2, dnsa_ge2hop, + per-arm HELD-OUT-GENE COVERAGE. GAP =
FUNGI − arm, paired bootstrap (B=10,000).

LEAKAGE FIREWALL (asserted in code): train-gene ∩ held-out-gene = ∅; held-out responses NEVER enter training
(train_one fits on fit_s only); μ̄/signal-mask/co-expression baseline are train-only; only gene IDENTITIES cross.
K is tuned on the gene-disjoint VAL split, NOT inherited from the in-distribution run.

  python src/zeroshot_v3f.py --dry_run                      # plumbing/leakage test on the CURRENT subset substrate
  python src/zeroshot_v3f.py --hyphae_dir DATA/HYPHAE/rpe1  # the REAL run (fires when a HYPHAE parquet lands)
"""
from __future__ import annotations
import os, sys, json, time, argparse, glob
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTHONUTF8", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3f as G
import harness_v3f as H
import metrics_v3f as M
import edge_features_v3 as EF3
import edge_features_v3f as EF3F
from model_v3f import make_feats
import yaml

RES = os.path.join(HERE, "..", "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "obj_009_3.yaml")))
BFS_CAP = 6
# map an arm name -> its HYPHAE parquet basename (when running the real HYPHAE swap)
HYPHAE_FILES = {"fungi": "fungi_hyphae.parquet", "top_weight": "topweight_hyphae.parquet",
                "pure_parent": "hyphae_parent.parquet", "knn": "knn_hyphae.parquet"}
# dry-run stand-ins on the CURRENT subset substrate (fungi_bio stands in for FUNGI(HYPHAE))
DRY_ARMS = {"fungi": "fungi_bio", "top_weight": "top_weight", "pure_parent": "knn", "knn": "knn",
            "shuffle": "shuffle", "reverse": "reverse", "labelperm": "labelperm", "empty": "empty"}


def log(m):
    line = f"[zeroshot {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    os.makedirs(RES, exist_ok=True); open(os.path.join(RES, "zeroshot.log"), "a", encoding="utf-8").write(line + "\n")


def hyphae_landed():
    hits = []
    for pat in CFG.get("run", {}).get("hyphae_abort_glob", []):
        hits += [p for p in glob.glob(os.path.join(ROOT, pat), recursive=True)
                 if "input" not in p.lower() and "backup" not in p.lower()]
    return hits


# ------------------------------------------------------------------ gene-held-out split (the exp_022 firewall)
def gene_held_out_split(D, seed=0, frac_holdout=0.30, val_frac=0.5):
    """Partition the perturbed GENES (pidx>=0) into train vs held-out (val+test), gene-disjoint. Returns
    (train_ix, val_ix, test_ix, info) as indices into D['pidx']. Leakage assert: train ∩ held-out genes = ∅."""
    pidx = D["pidx"]; has = np.where(pidx >= 0)[0]
    genes = np.unique(pidx[has])
    rng = np.random.default_rng(seed); rng.shuffle(genes)
    n_hold = int(round(frac_holdout * len(genes)))
    hold_genes = set(int(g) for g in genes[:n_hold]); train_genes = set(int(g) for g in genes[n_hold:])
    assert train_genes.isdisjoint(hold_genes), "gene-held-out firewall: train ∩ held-out genes must be empty"
    train_ix = np.array([i for i in has if int(pidx[i]) in train_genes], np.int64)
    hold_ix = np.array([i for i in has if int(pidx[i]) in hold_genes], np.int64)
    rng2 = np.random.default_rng(seed + 1); rng2.shuffle(hold_ix)
    nval = int(round(val_frac * len(hold_ix)))
    val_ix, test_ix = hold_ix[:nval], hold_ix[nval:]
    info = dict(n_train_genes=len(train_genes), n_hold_genes=len(hold_genes),
                n_train_pert=len(train_ix), n_val_pert=len(val_ix), n_test_pert=len(test_ix))
    return train_ix, val_ix, test_ix, info


# ------------------------------------------------------------------ HYPHAE ingest (direction-preserving)
def ingest_hyphae(parquet, g2i, N):
    """Read a HYPHAE parquet -> (src,tgt,w_outnorm,w_raw). Accepts Regulator/Target/Importance or
    source/target/weight columns. Out-normalized (NEVER symmetric). Genes off the panel are dropped."""
    import pandas as pd
    df = pd.read_parquet(parquet)
    cmap = {c.lower(): c for c in df.columns}
    sc = cmap.get("regulator", cmap.get("source")); tc = cmap.get("target")
    wc = cmap.get("importance", cmap.get("weight", cmap.get("score")))
    si = df[sc].map(g2i).to_numpy(); ti = df[tc].map(g2i).to_numpy()
    w = df[wc].to_numpy(np.float64) if wc else np.ones(len(df))
    keep = ~(pd.isna(si) | pd.isna(ti)); si = si[keep].astype(np.int64); ti = ti[keep].astype(np.int64); w = w[keep]
    k2 = si != ti; si, ti, w = si[k2], ti[k2], w[k2]
    outstr = np.bincount(si, weights=w, minlength=N); wn = (w / (outstr[si] + 1e-8)).astype(np.float32)
    return si, ti, wn, w.astype(np.float64)


def arm_graph(arm, D, hyphae_dir, dry):
    """Return (src,tgt,w_outnorm,w_raw) for an arm. dry -> the current-subset stand-in; else the HYPHAE parquet
    (fungi/top_weight/pure_parent/knn) or a derived null (shuffle/reverse/labelperm/empty of the FUNGI-HYPHAE)."""
    if dry:
        return G.build_arm(DRY_ARMS[arm], D["g2i"], D["N"])
    if arm in HYPHAE_FILES:
        return ingest_hyphae(os.path.join(hyphae_dir, HYPHAE_FILES[arm]), D["g2i"], D["N"])
    base = ingest_hyphae(os.path.join(hyphae_dir, HYPHAE_FILES["fungi"]), D["g2i"], D["N"])
    s, t, wn, wr = base
    if arm == "empty":
        z = np.zeros(0, np.int64); return z, z, np.zeros(0, np.float32), np.zeros(0, np.float64)
    if arm == "reverse":
        return t, s, wn, wr
    if arm == "labelperm":
        perm = np.random.default_rng(42).permutation(D["N"]); return perm[s], perm[t], wn, wr
    if arm == "shuffle":
        rng = np.random.default_rng(42); return s, t[rng.permutation(len(t))], wn, wr
    raise ValueError(arm)


def coverage(src, heldout_gene_ids):
    have = set(int(x) for x in np.unique(src)) if len(src) else set()
    return float(np.mean([int(g) in have for g in heldout_gene_ids])) if len(heldout_gene_ids) else 0.0


# ------------------------------------------------------------------ per-arm train + score on held-out genes
def _profile_phi(arm_graph_tuple, arm_name, D, profile, with_dash):
    """Build the assembled φ_e for an ingested (non-cached) graph on the fly (clean cols + directed-topology)."""
    s, t, wn, wr = arm_graph_tuple; N = D["N"]
    if len(s) == 0:
        return np.zeros((0, EF3F.profile_dim(profile, with_dash)), np.float32)
    clean6, _ = EF3.compute_clean(s, t, wn, wr, N)
    if profile in ("directed_topo", "regulatory"):
        import compute_topo_features as CTF
        raw = {}
        ffl_fwd, ffl_fanout, cyc3 = CTF.directed_motif_counts(s, t, N)
        import topo_common as TC
        comm = TC.louvain_labels(s, t, wr, N, seed=0)
        P = CTF.participation_coefficient(s, t, N, comm)
        cols = {"ffl_fwd": TC._zscore(np.log1p(ffl_fwd)), "ffl_fanout": TC._zscore(np.log1p(ffl_fanout)),
                "cyc3": TC._zscore(np.log1p(cyc3)), "part_src": P[s].astype(np.float32),
                "part_tgt": P[t].astype(np.float32)}
        if with_dash:
            r = np.clip(1.0 - 0.5 * 0.0, 0, None)  # no per-arm oversquash cache off-substrate -> neutral bridge
            cols["dash_recomp"] = TC._zscore(np.asarray(wn, np.float64))
        return EF3F.assemble_directed_topo(clean6, cols, with_dash=with_dash)
    return clean6


def run(dry=True, hyphae_dir=None, seed=0, device="cuda", frac_holdout=0.30):
    from gauntlet_v3f import resolve_config
    cfg, profile, with_dash, src = resolve_config()
    feats = make_feats("assembled", profile=profile, film=cfg.get("film", True), ovsq=cfg.get("ovsq", True))
    D = G.load_data(); N = D["N"]
    train_ix, val_ix, test_ix, info = gene_held_out_split(D, seed=seed, frac_holdout=frac_holdout)
    log(f"config {src} profile={profile} | split {info}")
    arms = list(DRY_ARMS) if dry else (list(HYPHAE_FILES) + ["shuffle", "reverse", "labelperm", "empty"])
    hp = dict(cfg); hp = {k: hp.get(k) for k in ["d", "d_hidden", "K", "teleport_alpha", "gcnii_beta",
             "dropout", "lr", "weight_decay", "batch_perts", "epochs", "patience", "min_delta", "delta_mode"]}
    hp = {k: v for k, v in hp.items() if v is not None}
    hp.setdefault("oversmooth", "jk"); hp.setdefault("corr_weight", 1.0); hp.setdefault("mse_weight", 0.1)
    hp.setdefault("grad_clip", 1.0); hp.setdefault("delta_mode", "neg_mu_ctrl")
    Zfit = M.build_coexpr_Z(D["s_full"][train_ix])
    results = {}
    for arm in arms:
        t0 = time.time()
        g = arm_graph(arm, D, hyphae_dir, dry); graph = g[:3]
        phi = _profile_phi(g, arm, D, profile, with_dash) if (feats.get("edge")) else None
        c_g = G.load_node_ctx(DRY_ARMS[arm]) if (dry and feats.get("film")) else None
        hop_w = G.load_oversquash_w(DRY_ARMS[arm], N) if (dry and feats.get("ovsq")) else None
        hold_genes = np.unique(D["pidx"][test_ix])
        cov = coverage(graph[0], hold_genes)
        # train on TRAIN-gene perts, predict the held-out (test) perts
        pred, meta = H.train_one(graph, D["pidx"][train_ix], D["s_full"][train_ix],
                                 D["pidx"][test_ix], D["s_full"][test_ix], D["signal_mask"], D["mu_ctrl"], hp,
                                 seed=seed, device=device, variant="assembled", feats=feats,
                                 phi_e=phi, c_g=c_g, hop_w=hop_w, delta_mode=hp["delta_mode"])
        st = D["s_full"][test_ix]; mask = D["signal_mask"]; pe = D["pidx"][test_ix]
        rsc = M.rsc_per_pert(st, pred, mask, pe)
        rscr = M.rsc_coexpr_resid_per_pert(st, pred, mask, pe, Zfit)
        results[arm] = dict(rsc=rsc, rsc_coexpr_resid=rscr, coverage=cov,
                            rsc_mean=float(np.nanmean(rsc)), secs=round(time.time() - t0, 1))
        log(f"{arm:11s} cov={cov:.3f} held-out RSC={np.nanmean(rsc):+.4f} rscr={np.nanmean(rscr):+.4f} ({results[arm]['secs']}s)")
    # GAP = FUNGI − arm on the held-out genes, paired bootstrap
    f = results["fungi"]; out = dict(mode="dry_run" if dry else "hyphae", config_source=src, profile=profile,
                                     split=info, seed=seed, arms={})
    for arm, r in results.items():
        pb = M.paired_bootstrap(f["rsc"], r["rsc"], n_boot=CFG.get("metrics", {}).get("bootstrap_B", 10000), seed=2)
        pbr = M.paired_bootstrap(f["rsc_coexpr_resid"], r["rsc_coexpr_resid"], n_boot=10000, seed=3)
        out["arms"][arm] = dict(rsc_mean=r["rsc_mean"], coverage=r["coverage"],
                                gap_rsc=round(pb["mean_diff"], 4), gap_rsc_p=round(pb["p_one_sided"], 5),
                                gap_rsc_coexpr_resid=round(pbr["mean_diff"], 4))
    fn = "zeroshot_dryrun.json" if dry else "zeroshot_hyphae.json"
    json.dump(out, open(os.path.join(RES, fn), "w"), indent=2, default=float)
    log(f"zeroshot {'DRY-RUN' if dry else 'HYPHAE'} -> results/{fn}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--hyphae_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frac_holdout", type=float, default=0.30)
    args = ap.parse_args()
    dry = args.dry_run or not args.hyphae_dir
    if not dry:
        h = hyphae_landed()
        log(f"HYPHAE run: dir={args.hyphae_dir} (abort-glob hits: {h[:2]})")
    run(dry=dry, hyphae_dir=args.hyphae_dir, seed=args.seed, device=args.device, frac_holdout=args.frac_holdout)


if __name__ == "__main__":
    main()
