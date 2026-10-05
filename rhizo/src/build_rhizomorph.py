"""
exp_031 patch3 (build side, 2070) — build the RHIZOMORPH cohort DELTAS for the A100 windfall campaign.

exp_030's `sp_union` IS the rhizomorph core (already built + H1-verified). This script builds only the deltas
the A100 P2 experiment needs, as obj_009.x arm-graph .npz files (keys: src/tgt/w_outnorm/w_raw) that the
gauntlet's `RHIZO_EXTRA_ARMS` hook consumes without a code edit:

  rhizomorph              = alias of exp_030 sp_union  [edges(shroom_fungi) ∪ HYPHAE graft on the uncovered
                            genes; E=200k, 5000 sources, 373/373 held-out coverage]
  shroom_grafted          = the UNPRUNED SHROOM dense causal core (top-weight, coverage-preserving) ∪ the SAME
                            graft rhizomorph uses. Isolates FUNGI's within-graft contribution: the ONLY thing
                            that differs from rhizomorph is the P-core (FUNGI-pruned vs unpruned). Matched E,
                            matched coverage, byte-identical graft.
  shuffle_of_rhizomorph   = per-arm degree-preserving null OF rhizomorph (src kept, tgt permuted; out-degree
                            sequence identical). Provenance-named (the Report_032 lesson).
  reverse_of_rhizomorph   = per-arm edge-direction-flip null OF rhizomorph.

Four BUILD-TIME ASSERTS, each PAIRED WITH A NEGATIVE CONTROL that would make it fail (the four-defect lesson —
an assert that cannot fail is worthless):
  A1  n_src(rhizomorph)==5000                       NEG: n_src(fungi_bio)==865 (a causal-only core) != 5000
  A2  held_out_coverage(rhizomorph)==373 (count)    NEG: held_out_coverage(fungi_bio)==0 != 373
  A3  edges(shroom_fungi) ⊆ edges(rhizomorph)       NEG: edges(shroom_fungi) ⊄ edges(top_weight)
  A4  sorted_outdeg(shuffle_of_rhizomorph)==sorted_outdeg(rhizomorph) & E match
                                                    NEG: sorted_outdeg(reverse_of_rhizomorph) != sorted_outdeg(rhizomorph)
(shroom_grafted inherits the E=200k / n_src=5000 / coverage=373 / same-graft asserts too.)

Leakage profile is identical to exp_030 (SHROOM causal = train-perturbation-derived; the HYPHAE graft is
train-cell co-expression). Writes the four .npz + build_rhizomorph_report.json.

  python build_rhizomorph.py                         # local 2070 defaults (thesis tree)
  python build_rhizomorph.py --out <dir> --report <json>
"""
from __future__ import annotations
import os, sys, json, time, argparse, hashlib
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")   # cp1252 console guard
except Exception:
    pass
from pathlib import Path
import numpy as np

N_GENES = 5000
CAP = 200000
HELD_OUT_N = 373
DENSITIES = [100_000, 150_000, 200_000, 300_000, 400_000]   # rhizomorph density sweep (coverage-preserving)

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent
THESIS = Path("c:/Users/studi/OneDrive/Documents/thesis")
DEF_GRAPHS = THESIS / "DATA/EXPERIMENTS/exp_030_source_partitioned_supergraph/intermediate/graphs"
DEF_DENSE = THESIS / "DATA/EXPERIMENTS/exp_029_hpc_rhizo_crownjewel_showdown/patch2/ship/arms/shroom_denseparent.npz"
DEF_LFC = DEF_GRAPHS / "lfc_targets_rpe1_cellhalf.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ inlined arm helpers (from exp_030 sp_common)
def outnorm_persource(src, tgt, w):
    """Per-source sum-normalization (row-stochastic) — the locked arm-graph convention (== arbiter_lib / save_arm)."""
    w = np.asarray(w, np.float64); out = np.empty_like(w)
    order = np.argsort(src, kind="stable"); ss = src[order]; ws = w[order]
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1, [len(ss)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = ws[a:b]; tot = seg.sum()
        ws[a:b] = seg / tot if tot > 0 else 1.0 / max(b - a, 1)
    out[order] = ws
    return out


def save_arm(path, src, tgt, w_raw):
    """Write an obj_009.x arm-graph .npz (src/tgt/w_outnorm/w_raw) — the exp_029/030 arm_graphs format."""
    src = np.asarray(src, np.int64); tgt = np.asarray(tgt, np.int64); w_raw = np.asarray(w_raw, np.float64)
    wn = outnorm_persource(src, tgt, w_raw) if len(src) else w_raw.astype(np.float32)
    np.savez(path, src=src, tgt=tgt, w_outnorm=wn.astype(np.float32), w_raw=w_raw.astype(np.float64))
    return dict(n_edges=int(len(src)), n_src=int(len(np.unique(src))) if len(src) else 0)


def edge_keys(s, t, N=N_GENES):
    return (np.asarray(s, np.int64) * N + np.asarray(t, np.int64))


def n_src(s):
    return int(len(np.unique(np.asarray(s, np.int64)))) if len(s) else 0


def held_out_coverage_count(s, held):
    return len(set(np.unique(np.asarray(s, np.int64)).tolist()) & set(np.asarray(held, np.int64).tolist()))


def sorted_outdeg(s, N=N_GENES):
    return np.sort(np.bincount(np.asarray(s, np.int64), minlength=N))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ coverage-preserving top-k on the dense core
def coverage_topk(s, t, w, sources_required, budget):
    """Keep, from (s,t,w), the top-`budget` edges by weight while GUARANTEEING >=1 edge for every source in
    `sources_required` that has any edge (its top-weight edge). Mirrors exp_030 hyphae_fill's coverage guarantee
    so the unpruned P-core covers every one of the 865 perturbed sources (n_src parity with rhizomorph)."""
    s = np.asarray(s, np.int64); t = np.asarray(t, np.int64); w = np.asarray(w, np.float64)
    order = np.lexsort((-w, s)); ss, ts, ws = s[order], t[order], w[order]
    change = np.r_[True, ss[1:] != ss[:-1]]; grp = np.where(change)[0]
    within = np.arange(len(ss)) - np.repeat(grp, np.diff(np.r_[grp, len(ss)]))
    req = set(np.asarray(sources_required, np.int64).tolist())
    keep = np.zeros(len(ss), bool)
    is_req_top = (within == 0) & np.isin(ss, np.fromiter(req, np.int64, len(req)))
    keep[is_req_top] = True
    remaining = budget - int(keep.sum())
    if remaining > 0:
        rest = np.where(~keep)[0]
        rest = rest[np.argsort(-ws[rest])[:remaining]]
        keep[rest] = True
    elif remaining < 0:                                              # budget < n_required (won't happen: 865<<168k)
        raise ValueError(f"budget {budget} < n_required {int(keep.sum())} — cannot preserve coverage")
    return ss[keep], ts[keep], ws[keep]


def build_rhizomorph_at(E, fb, hy, U):
    """Coverage-preserving rhizomorph at edge-cap E (the density sweep). Same construction as exp_030 sp_union —
    causal core (fungi_bio on the perturbed sources) ∪ HYPHAE graft on the uncovered genes U — parameterised by E.
    Two coverage BASES are taken FIRST so BOTH held-out coverage (373/373) AND n_src (5000) hold at EVERY density:
      P-base : the top-weight causal edge of each perturbed source (865)
      U-base : the top-Importance HYPHAE edge of each U source (4135; held-out perts are in U)
    then the budget fills with the causal REST (strongest first) then the HYPHAE graft EXTRA (highest Importance).
    At E=200k the P-base is a subset of the (fully-included) causal core, so this reproduces sp_union exactly."""
    fb_s, fb_t, fb_w = fb
    hy_s, hy_t, hy_w, hy_wi = hy
    # causal: per-source top (P-base) vs the rest (weight-ordered globally)
    order = np.lexsort((-fb_w, fb_s)); ss, ts_, ws = fb_s[order], fb_t[order], fb_w[order]
    is_ptop = np.r_[True, ss[1:] != ss[:-1]]                          # first (= highest-weight) edge per P source
    pb_s, pb_t, pb_w = ss[is_ptop], ts_[is_ptop], ws[is_ptop]
    rc = np.where(~is_ptop)[0]; rc = rc[np.argsort(-ws[rc], kind="stable")]
    rc_s, rc_t, rc_w = ss[rc], ts_[rc], ws[rc]
    # HYPHAE graft on U: base (one per U source) vs extra (Importance-ordered)
    elig = np.isin(hy_s, np.asarray(U, np.int64))
    gs, gt, gw, gwi = hy_s[elig], hy_t[elig], hy_w[elig], hy_wi[elig]
    ub = np.where(gwi == 0)[0]; ge = np.where(gwi != 0)[0]; ge = ge[np.argsort(-gw[ge], kind="stable")]
    # assemble: both coverage bases (5000 edges), then causal rest, then graft extra, up to E
    rem = max(0, E - (len(pb_s) + len(ub)))
    take_core = min(len(rc_s), rem); rem -= take_core
    take_extra = min(len(ge), rem)
    ie = ge[:take_extra]
    s = np.concatenate([pb_s, gs[ub], rc_s[:take_core], gs[ie]])
    t = np.concatenate([pb_t, gt[ub], rc_t[:take_core], gt[ie]])
    w = np.concatenate([pb_w, gw[ub], rc_w[:take_core], gw[ie]])
    return s.astype(np.int64), t.astype(np.int64), w.astype(np.float64)


# ------------------------------------------------------------------ build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default=str(DEF_GRAPHS), help="exp_030 intermediate/graphs (sp_union/fungi_bio/top_weight)")
    ap.add_argument("--dense-parent", default=str(DEF_DENSE), help="unpruned SHROOM dense parent .npz (src/tgt/w)")
    ap.add_argument("--lfc", default=str(DEF_LFC), help="cell-half LFC package (held-out val/test pidx)")
    ap.add_argument("--out", default=str(PATCH / "ship" / "arms"))
    ap.add_argument("--report", default=str(PATCH / "diagnostics" / "build_rhizomorph_report.json"))
    args = ap.parse_args()
    t0 = time.time()
    GR = Path(args.graphs_dir); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    def load(name):
        z = np.load(GR / f"{name}.npz")
        return z["src"].astype(np.int64), z["tgt"].astype(np.int64), z["w_raw"].astype(np.float64)

    su_s, su_t, su_w = load("sp_union")                              # rhizomorph core
    fb_s, fb_t, fb_w = load("fungi_bio")                             # shroom_fungi (P core + subset check)
    tw_s, tw_t, _tw_w = load("top_weight")                          # negative control for the subset assert
    dp = np.load(args.dense_parent)
    dp_s, dp_t = dp["src"].astype(np.int64), dp["tgt"].astype(np.int64)
    dp_w = dp["w"].astype(np.float64) if "w" in dp.files else dp["w_raw"].astype(np.float64)
    zl = np.load(args.lfc, allow_pickle=True)
    held = np.concatenate([zl["val_pidx"], zl["test_pidx"]]).astype(np.int64)
    P = np.unique(fb_s)
    log(f"loaded: sp_union E={len(su_s):,} | fungi_bio E={len(fb_s):,} n_src={len(P)} | dense_parent E={len(dp_s):,} "
        f"n_src={n_src(dp_s)} | held-out={len(held)}")

    # ---- rhizomorph = alias of sp_union (build the arm-graph via the canonical save_arm; edges preserved) ----
    info_rz = save_arm(OUT / "rhizomorph.npz", su_s, su_t, su_w)
    rz = np.load(OUT / "rhizomorph.npz"); rz_s, rz_t = rz["src"].astype(np.int64), rz["tgt"].astype(np.int64)
    assert set(edge_keys(rz_s, rz_t).tolist()) == set(edge_keys(su_s, su_t).tolist()), "rhizomorph != sp_union edges"

    # ---- graft = rhizomorph edges whose source is NOT perturbed (the HYPHAE fill on the uncovered genes) ----
    inP = np.isin(su_s, P)
    graft_s, graft_t, graft_w = su_s[~inP], su_t[~inP], su_w[~inP]
    graft_keys = set(edge_keys(graft_s, graft_t).tolist())
    log(f"graft (src not in P): E={len(graft_s):,} n_src={n_src(graft_s)}  |  budget for dense P-core = {CAP-len(graft_s):,}")

    # ---- shroom_grafted = UNPRUNED dense P-core (coverage-preserving top-weight) ∪ the SAME graft ----
    budget_P = CAP - len(graft_s)
    core_s, core_t, core_w = coverage_topk(dp_s, dp_t, dp_w, P, budget_P)
    sg_s = np.concatenate([core_s, graft_s]); sg_t = np.concatenate([core_t, graft_t])
    sg_w = np.concatenate([core_w, graft_w])
    info_sg = save_arm(OUT / "shroom_grafted.npz", sg_s, sg_t, sg_w)

    # ---- per-arm nulls OF rhizomorph ----
    rng = np.random.default_rng(0)
    info_sh = save_arm(OUT / "shuffle_of_rhizomorph.npz", rz_s.copy(), rng.permutation(rz_t), rz["w_raw"].astype(np.float64))
    info_rv = save_arm(OUT / "reverse_of_rhizomorph.npz", rz_t.copy(), rz_s.copy(), rz["w_raw"].astype(np.float64))

    def loadarm(name):
        z = np.load(OUT / f"{name}.npz"); return z["src"].astype(np.int64), z["tgt"].astype(np.int64)
    sh_s, sh_t = loadarm("shuffle_of_rhizomorph"); rv_s, rv_t = loadarm("reverse_of_rhizomorph")
    sg2_s, sg2_t = loadarm("shroom_grafted")

    # ---- rhizomorph DENSITY SWEEP (coverage-preserving; E in {100k,150k,200k,300k,400k}) ----
    hy_path = GR / "_hyphae_dense_bysource.npz"
    density = {}
    if hy_path.exists():
        hz = np.load(hy_path)
        hy = (hz["src"].astype(np.int64), hz["tgt"].astype(np.int64),
              hz["w"].astype(np.float64), hz["within"].astype(np.int64))
        U = np.setdiff1d(np.arange(N_GENES, dtype=np.int64), P)
        for E in DENSITIES:
            ds, dt_, dw = build_rhizomorph_at(E, (fb_s, fb_t, fb_w), hy, U)
            info = save_arm(OUT / f"rhizomorph_{E//1000}k.npz", ds, dt_, dw)
            hc = held_out_coverage_count(ds, held)
            density[f"rhizomorph_{E//1000}k"] = dict(target_E=E, n_edges=info["n_edges"], n_src=info["n_src"],
                                                     held_out_coverage_count=hc)
            log(f"density rhizomorph_{E//1000}k: E={info['n_edges']:,} (target {E:,}) n_src={info['n_src']} held_cov={hc}")
    else:
        log(f"[WARN] HYPHAE cache {hy_path} absent -> density sweep SKIPPED (rhizomorph@200k canonical still built)")

    # ================================ build-time asserts (each paired with a negative control) =================
    checks = []

    def check(name, passed, neg_name, neg_would_fail, detail):
        ok = bool(passed) and bool(neg_would_fail)   # assert must hold AND its negative control must actually break it
        checks.append(dict(assert_name=name, passed=bool(passed), neg_control=neg_name,
                           neg_control_fails_assert=bool(neg_would_fail), discriminating=bool(neg_would_fail),
                           ok=ok, detail=detail))
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {passed} | NEG {neg_name}: would_fail={neg_would_fail} | {detail}")
        return ok

    # A1 — n_src == 5000
    a1 = check("A1_n_src_5000", n_src(rz_s) == N_GENES,
               "n_src(fungi_bio)!=5000", n_src(fb_s) != N_GENES,
               f"n_src(rhizomorph)={n_src(rz_s)}, n_src(shroom_grafted)={n_src(sg2_s)}, "
               f"NEG n_src(fungi_bio)={n_src(fb_s)}")
    a1b = check("A1b_shroom_grafted_n_src_5000", n_src(sg2_s) == N_GENES,
                "n_src(fungi_bio)!=5000", n_src(fb_s) != N_GENES, f"n_src(shroom_grafted)={n_src(sg2_s)}")

    # A2 — held_out_coverage (count) == 373
    hc_rz = held_out_coverage_count(rz_s, held); hc_sg = held_out_coverage_count(sg2_s, held)
    hc_fb = held_out_coverage_count(fb_s, held)
    a2 = check("A2_held_out_coverage_373", hc_rz == HELD_OUT_N,
               "held_out_coverage(fungi_bio)!=373", hc_fb != HELD_OUT_N,
               f"held_cov(rhizomorph)={hc_rz}, held_cov(shroom_grafted)={hc_sg}, NEG held_cov(fungi_bio)={hc_fb}")
    a2b = check("A2b_shroom_grafted_coverage_373", hc_sg == HELD_OUT_N,
                "held_out_coverage(fungi_bio)!=373", hc_fb != HELD_OUT_N, f"held_cov(shroom_grafted)={hc_sg}")

    # A3 — edges(shroom_fungi) ⊆ edges(rhizomorph)
    fb_keys = set(edge_keys(fb_s, fb_t).tolist())
    rz_keys = set(edge_keys(rz_s, rz_t).tolist())
    tw_keys = set(edge_keys(tw_s, tw_t).tolist())
    miss_in_rz = len(fb_keys - rz_keys); miss_in_tw = len(fb_keys - tw_keys)
    a3 = check("A3_fungi_subset_of_rhizomorph", miss_in_rz == 0,
               "fungi_bio NOT subset of top_weight", miss_in_tw > 0,
               f"missing edges in rhizomorph={miss_in_rz}, NEG missing in top_weight={miss_in_tw}")

    # A4 — sorted_outdeg(shuffle) == sorted_outdeg(rhizomorph) & E match ; NEG: reverse changes out-degree
    od_rz = sorted_outdeg(rz_s); od_sh = sorted_outdeg(sh_s); od_rv = sorted_outdeg(rv_s)
    a4_deg = bool(np.array_equal(od_sh, od_rz)); a4_e = (len(sh_s) == len(rz_s))
    a4 = check("A4_shuffle_outdeg_and_E_match", a4_deg and a4_e,
               "reverse_of_rhizomorph outdeg != rhizomorph", (not np.array_equal(od_rv, od_rz)),
               f"outdeg_equal(shuffle)={a4_deg}, E(shuffle)={len(sh_s)} vs E(rhizomorph)={len(rz_s)}; "
               f"NEG outdeg_equal(reverse)={bool(np.array_equal(od_rv, od_rz))}")

    # A5 — density sweep: rhizomorph_200k reproduces the canonical rhizomorph edge set; NEG: rhizomorph_100k does
    # NOT (a genuinely different, sparser graph). Also validates coverage=373 & n_src=5000 at EVERY density.
    density_ok = True
    if density:
        d200_s, d200_t = loadarm("rhizomorph_200k"); d100_s, d100_t = loadarm("rhizomorph_100k")
        d200_keys = set(edge_keys(d200_s, d200_t).tolist()); d100_keys = set(edge_keys(d100_s, d100_t).tolist())
        ov200 = len(d200_keys & rz_keys) / max(len(rz_keys), 1)
        ov100 = len(d100_keys & rz_keys) / max(len(rz_keys), 1)
        check("A5_density_200k_reproduces_canonical", d200_keys == rz_keys,
              "rhizomorph_100k != canonical rhizomorph", d100_keys != rz_keys,
              f"overlap(200k,canonical)={ov200:.4f} (==1.0 expected), NEG overlap(100k,canonical)={ov100:.4f}")
        for name, info in density.items():
            E = info["target_E"]
            ok = (info["n_edges"] == E and info["n_src"] == N_GENES and info["held_out_coverage_count"] == HELD_OUT_N)
            density_ok = density_ok and ok
            if not ok:
                log(f"[FAIL] density {name}: E={info['n_edges']} (target {E}) n_src={info['n_src']} "
                    f"held_cov={info['held_out_coverage_count']}")

    # ---- supplementary provenance asserts (not among the 4 primaries but strengthen the ablation) ----
    supp = {}
    supp["E_all_200k"] = {a: int(info["n_edges"]) for a, info in
                          [("rhizomorph", info_rz), ("shroom_grafted", info_sg),
                           ("shuffle_of_rhizomorph", info_sh), ("reverse_of_rhizomorph", info_rv)]}
    # same graft: shroom_grafted's non-P edges == rhizomorph's graft
    sg_inP = np.isin(sg2_s, P)
    sg_graft_keys = set(edge_keys(sg2_s[~sg_inP], sg2_t[~sg_inP]).tolist())
    supp["graft_identical"] = bool(sg_graft_keys == graft_keys)
    # P-core is genuinely UNPRUNED (differs from the FUNGI-pruned fungi_bio)
    sg_core_keys = set(edge_keys(sg2_s[sg_inP], sg2_t[sg_inP]).tolist())
    supp["shroom_grafted_core_differs_from_fungi"] = bool(sg_core_keys != fb_keys)
    supp["shroom_grafted_core_overlap_with_fungi"] = len(sg_core_keys & fb_keys)
    supp["shroom_grafted_core_size"] = len(sg_core_keys)
    # P-core and graft are edge-disjoint (P ∩ U = ∅ by construction)
    supp["core_graft_edge_overlap"] = len(sg_core_keys & sg_graft_keys)
    log(f"supp: graft_identical={supp['graft_identical']} core_differs={supp['shroom_grafted_core_differs_from_fungi']} "
        f"core-and-fungi={supp['shroom_grafted_core_overlap_with_fungi']}/{supp['shroom_grafted_core_size']} "
        f"core-and-graft={supp['core_graft_edge_overlap']}")

    all_ok = all(c["ok"] for c in checks) and supp["graft_identical"] and \
        supp["shroom_grafted_core_differs_from_fungi"] and supp["core_graft_edge_overlap"] == 0 and density_ok
    for a, info in [("rhizomorph", info_rz), ("shroom_grafted", info_sg),
                    ("shuffle_of_rhizomorph", info_sh), ("reverse_of_rhizomorph", info_rv)]:
        if info["n_edges"] != CAP:
            all_ok = False; log(f"[FAIL] E({a})={info['n_edges']} != {CAP}")

    manifest = {}
    arm_names = ["rhizomorph", "shroom_grafted", "shuffle_of_rhizomorph", "reverse_of_rhizomorph"] + \
        [f"rhizomorph_{E//1000}k" for E in DENSITIES if (OUT / f"rhizomorph_{E//1000}k.npz").exists()]
    for a in arm_names:
        z = np.load(OUT / f"{a}.npz"); s = z["src"].astype(np.int64); t = z["tgt"].astype(np.int64)
        manifest[a] = dict(n_edges=int(len(s)), n_src=n_src(s),
                           held_out_coverage_count=held_out_coverage_count(s, held),
                           held_out_coverage_frac=round(held_out_coverage_count(s, held) / HELD_OUT_N, 4),
                           is_null=a.startswith(("shuffle_of_", "reverse_of_")),
                           sha256=sha256_of(OUT / f"{a}.npz"))

    report = dict(
        experiment="exp_031_a100_windfall_campaign / patch3",
        built=time.strftime("%Y-%m-%dT%H:%M:%S"),
        inputs=dict(graphs_dir=str(GR), dense_parent=str(args.dense_parent), lfc=str(args.lfc),
                    P_perturbed_sources=int(len(P)), held_out=int(len(held)), graft_edges=int(len(graft_s)),
                    dense_core_budget=int(budget_P)),
        arms=manifest, asserts=checks, supplementary=supp, density_sweep=density, density_ok=bool(density_ok),
        all_ok=bool(all_ok), secs=round(time.time() - t0, 1),
    )
    json.dump(report, open(args.report, "w"), indent=2, default=float)
    log(f"DONE ({report['secs']}s) all_ok={all_ok} -> {OUT}  (report: {args.report})")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
