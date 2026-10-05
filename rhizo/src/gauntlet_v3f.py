"""
obj_009.3 (RHIZO-final) — gauntlet_v3f.py : at the self-HPO WINNER config, the full graph-type gauntlet +
un-confounding controls, cell-based / sharded / resumable (mirrors the obj_009.2 strengthen runner).

Cells (each -> results/cells/<key>.json, resumable):
  MAIN         arm × rung × seed for {fungi_bio, knn, top_weight, mst} + nulls {shuffle, reverse, labelperm,
               empty}, rungs {rung2, rung1}, seeds 0-4. -> the headline table: gap = fungi − arm on RSC,
               rsc_coexpr_resid (residualized), dnsa_ge2hop, dsrsc_ge2, each with paired bootstrap.
  ATTRIBUTION  drop_<col> on fungi_bio AND top_weight (rung2, seeds 0-1): zero one directed_topo NEW column ->
               gap drop = how load-bearing that feature is (which new feature carries signal).
  V7           perm_<col> on fungi_bio (rung2, seeds 0-1): PERMUTE one new column across edges -> the gain must
               collapse toward the drop level; a feature whose gain SURVIVES permutation is an ARTIFACT, not a
               win (the strengthen inert-DASH V7 failure is the cautionary tale). phishuf (whole-φ_e permute) too.

Config: reads the HPO winner from results/hpo_leaderboard.csv (top by gap) unless --config given; else falls back
to configs/obj_009_3.yaml frozen_hp + model. All the wideners fire (FiLM c_g, over-squash hop_w wired via
harness_v3f.arm_inputs). Leakage-safe (gene-disjoint rungs; graph/train-only features; per-fold coexpr baseline).

  python src/gauntlet_v3f.py --dry_run
  python src/gauntlet_v3f.py --shard 0 --n-shards 2 --device cuda
  python src/gauntlet_v3f.py --summarize
"""
from __future__ import annotations
import os, sys, json, time, argparse, hashlib
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v3f as G
import harness_v3f as H
import metrics_v3f as M
import edge_features_v3f as EF3F
from model_v3f import make_feats
import yaml

# exp_031 patch3: results dir is TIER-ISOLATED via --results_subdir (or RHIZO_RESULTS_SUBDIR). Each A100 tier
# (P0 rung1, P2 zeroshot rhizomorph, ...) writes to results/<subdir>/ so cells/preds/logs never blend across
# tiers. Default (no subdir) == results/ (byte-identical to patch2 behaviour).
BASE_RES = os.path.join(HERE, "..", "results")
RES = BASE_RES; CELLS = os.path.join(RES, "cells")
LOG = os.path.join(RES, "gauntlet_v3f.log")
CFG_PATH = os.path.join(HERE, "..", "configs", "obj_009_3.yaml")
BFS_CAP = 6
# item 10: obj_010 Mode-B prediction dumps (pred/true/pert_idx/mu_ctrl/signal_mask per cell). ~360 KB/cell.
PREDS = os.environ.get("RHIZO_PRED_DIR", os.path.join(RES, "preds"))


def _apply_results_subdir(subdir):
    """Re-point RES/CELLS/LOG/PREDS at results/<subdir> (tier isolation). RHIZO_PRED_DIR still overrides PREDS.
    Called from main() before any dispatch so every downstream function sees the reconfigured module globals."""
    global RES, CELLS, LOG, PREDS
    RES = BASE_RES if not subdir else os.path.join(BASE_RES, subdir)
    CELLS = os.path.join(RES, "cells")
    LOG = os.path.join(RES, "gauntlet_v3f.log")
    PREDS = os.environ.get("RHIZO_PRED_DIR", os.path.join(RES, "preds"))
    os.makedirs(RES, exist_ok=True)
    return RES
# item 12: the ge2hop / dsrsc metrics use ONE FIXED reference graph's BFS hop mask for EVERY arm — otherwise the
# super-graph (denser by construction) would be scored on a different gene-pair population than its components.
REF_HOP_ARM = os.environ.get("RHIZO_HOP_REF_ARM", "fungi_bio")
# exp_029 RPE1 cohort (blueprint §Cohort A). fungi_bio<-shroom_fungi (primary FUNGI arm), top_weight<-hyphae
# (co-expr competitor); + the extra cohort arms by their own names. labelperm dropped (not in the exp_025 arm set).
GAUNTLET = ["fungi_bio", "top_weight", "knn", "mst", "hyphae_fungi", "ptf_borda", "borda", "shroom"]
# exp_029 SOURCE-COVERAGE FIX B: the nulls are now PER-ARM and named for their provenance. The old global
# `shuffle`/`reverse` were degree-preserving shuffles of the BORDA consensus (E=200k, 957/3377 src), NOT of
# `fungi_bio` (E=168138, 865 src) — so every fungi_minus_shuffle margin was a comparison to an unrelated graph.
# `shuffle_of_fungi_bio` (target-permute -> preserves fungi_bio's exact src/out-degree set + E) and
# `reverse_of_fungi_bio` (edge-direction flip of fungi_bio) are the correct nulls of the champion.
NULLS = ["shuffle_of_fungi_bio", "reverse_of_fungi_bio", "empty"]
# exp_031 patch3: append extra NULLS WITHOUT a code edit — `export RHIZO_EXTRA_NULLS=shuffle_of_rhizomorph,
# reverse_of_rhizomorph` + drop their data/arm_graphs/<name>.npz in. Registering the per-arm nulls as NULLS (not
# GAUNTLET) is what lets P2 compute the null margin (arm − its shuffle/reverse); without this they would be
# treated as real arms and no null margin exists. Extend NULLS BEFORE the GAUNTLET-extra line so a name listed in
# both is kept as a null (excluded from GAUNTLET).
_extra_nulls = [a.strip() for a in os.environ.get("RHIZO_EXTRA_NULLS", "").split(",") if a.strip()]
NULLS = NULLS + [a for a in _extra_nulls if a not in NULLS and a not in GAUNTLET]
# exp_029 wave-2 hook: append extra arms (e.g. synth super-graphs) WITHOUT a code edit —
# `export RHIZO_EXTRA_ARMS=biologic_mid,synth_mid,...` + drop their data/arm_graphs/<name>.npz in.
_extra = [a.strip() for a in os.environ.get("RHIZO_EXTRA_ARMS", "").split(",") if a.strip()]
GAUNTLET = GAUNTLET + [a for a in _extra if a not in GAUNTLET and a not in NULLS]
# exp_031b patch4: RESTRICT the run to an explicit arm set WITHOUT a code edit — `export RHIZO_ONLY_ARMS=a,b,c`.
# When set, GAUNTLET := exactly that list (in order) and NULLS := only the nulls whose name is in the list. Used by
# the ABLATION/CAPACITY blocks (fungi_bio,top_weight,rhizomorph) so the 3-arm study does not train the full cohort
# (~5x cheaper). No RHIZO_EXTRA_ARMS needed — RHIZO_ONLY_ARMS is self-sufficient (the named .npz just must exist).
# NOTE: keep fungi_bio FIRST in the list — the (RSC-only, non-decision) summarize() anchors its gap on GAUNTLET[0].
_only = [a.strip() for a in os.environ.get("RHIZO_ONLY_ARMS", "").split(",") if a.strip()]
if _only:
    GAUNTLET = list(_only)
    NULLS = [n for n in NULLS if n in _only]
NEW_COLS = EF3F.NEW_FEATURE_COLS                     # [ffl_fwd, ffl_fanout, cyc3, part_src, part_tgt, dash_recomp]
METRICS = ["rsc", "rsc_coexpr_resid", "dnsa_ge2hop", "dsrsc_ge2"]


def log(m):
    line = f"[gauntlet {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    os.makedirs(RES, exist_ok=True); open(LOG, "a", encoding="utf-8").write(line + "\n")


# ------------------------------------------------------------------ winner config resolution
def _apply_row(c, r):
    for k in ["d", "d_hidden", "K", "batch"]:
        if k in r and not (isinstance(r[k], float) and np.isnan(r[k])):
            c["d_hidden" if k == "d_hidden" else ("batch_perts" if k == "batch" else k)] = int(r[k])
    for k, kk in [("tele", "teleport_alpha"), ("gcnii", "gcnii_beta"), ("lr", "lr")]:
        if k in r and not (isinstance(r[k], float) and np.isnan(r[k])):
            c[kk] = float(r[k])
    return r.get("edge_profile")


def resolve_config(path=None, allow_frozen=False):
    """item 9: NEVER silently run frozen_hp. Prefer the compound HPO winner (hpo_winner.json, written by the
    defect-11 summarize); then a compound-ELIGIBLE leaderboard row; only fall back to frozen_hp if explicitly
    allowed (--allow_frozen_fallback). A missing/empty leaderboard must not silently pick an unselected config."""
    base = yaml.safe_load(open(CFG_PATH))
    hp = dict(base["frozen_hp"]); hp["seeds_gauntlet"] = [0, 1, 2, 3, 4]
    prof = base["model"].get("edge_profile", "directed_topo"); wdash = base["model"].get("with_dash", True)
    winner_json = os.path.join(RES, "hpo_winner.json"); lb = os.path.join(RES, "hpo_leaderboard.csv")
    c = dict(hp)
    if path and os.path.exists(path):
        c = json.load(open(path)); src = f"explicit config {path}"
        # BLOCKER 1 fix: take edge_profile/with_dash FROM the config file. Without this, the trailing
        # `c["edge_profile"] = prof` below silently overwrites the file's profile with the YAML model
        # profile (directed_topo) -> --config anchor_t0009.json would still run directed_topo.
        prof = c.get("edge_profile", prof); wdash = c.get("with_dash", wdash)
    elif os.path.exists(winner_json):
        r = json.load(open(winner_json)); p = _apply_row(c, r); prof = p or prof
        src = f"hpo COMPOUND winner {r.get('trial_id')}"
    elif os.path.exists(lb):
        import pandas as pd
        df = pd.read_csv(lb)
        elig = df[df["compound_eligible"] == True] if "compound_eligible" in df.columns else df.iloc[0:0]
        if len(elig) == 0:
            if not allow_frozen:
                raise SystemExit("FATAL(item 9): leaderboard has NO COMPOUND-ELIGIBLE winner and no hpo_winner.json. "
                                 "Refusing frozen_hp fallback (it would run 12h at a config no HPO selected -> a "
                                 "publishable-looking but INVALID result). Pass --allow_frozen_fallback to override, "
                                 "or accept the honest 'co-expression wins on RPE1' outcome and escalate.")
            src = "obj_009_3.yaml frozen_hp (EXPLICITLY ALLOWED)"
        else:
            r = elig.iloc[0].to_dict(); p = _apply_row(c, r); prof = p or prof
            src = f"hpo COMPOUND winner {r.get('trial_id')}"
    else:
        if not allow_frozen:
            raise SystemExit("FATAL(item 9): no hpo_winner.json and no hpo_leaderboard.csv. Run the HPO first, "
                             "or pass --allow_frozen_fallback to deliberately run the frozen_hp config.")
        src = "obj_009_3.yaml frozen_hp (EXPLICITLY ALLOWED)"
    c["edge_profile"] = prof; c["with_dash"] = wdash
    return c, prof, wdash, src


# ------------------------------------------------------------------ config tag (cell-key collision guard, FIX A)
def cfg_tag(cfg, profile, with_dash):
    """Short deterministic tag over the decision-relevant config keys. cell_key encodes it so two DIFFERENT
    configs can never write to (or resume from) the SAME cell file — the collision hazard flagged for cell_key."""
    keys = ("d", "d_hidden", "K", "teleport_alpha", "gcnii_beta", "lr", "batch_perts", "film", "ovsq")
    payload = {"profile": profile, "with_dash": bool(with_dash), **{k: cfg.get(k) for k in keys}}
    h = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{profile}_{h}"


# ------------------------------------------------------------------ cell plan
def build_plan(seeds_main=(0, 1, 2, 3, 4), seeds_ctrl=(0, 1), rungs=("rung2", "rung1"), ctag="cfg"):
    cells = []
    arm_rank = lambda a: {"fungi_bio": 0, "top_weight": 1, "knn": 2, "mst": 3}.get(a, 9)
    for rung in rungs:
        for a in sorted(GAUNTLET + NULLS, key=arm_rank):
            for sd in seeds_main:
                cells.append(dict(arm=a, rung=rung, seed=sd, tag="main", cfg=ctag))
    # BLOCKER 2: the drop_*/perm_* directed-topology column controls are REMOVED. Under the clean profile those
    # 6 columns are absent, so each such cell would train IDENTICALLY to `main` and report v7_surviving_frac=1.000
    # (a false "pure artifact" verdict indistinguishable from a real finding — obj_009.3 burned ~4 GPU-h on this).
    # We keep only `phishuf` (a whole-φ_e row-shuffle null, valid under ANY profile). The directed-topology
    # negative is re-confirmed HONESTLY by the abl_directed_topo Tier-2 arm, not by these guarded no-ops.
    # FIX A: phishuf runs on `rung1` when present (fungi_bio has ~865/878 source coverage there) — NEVER on
    # `zeroshot`, where fungi_bio is 0-coverage so a φ_e shuffle of an all-zero prediction is meaningless.
    r_phi = "rung1" if "rung1" in rungs else rungs[0]
    for sd in seeds_ctrl:
        cells.append(dict(arm="fungi_bio", rung=r_phi, seed=sd, tag="phishuf", cfg=ctag))
    return cells


def cell_key(c):
    # FIX A: encode the config tag (collision guard). Falls back cleanly for cells built without one.
    ct = c.get("cfg")
    return (f"{c['arm']}__{c['rung']}__{c['tag']}__{ct}__s{c['seed']}" if ct
            else f"{c['arm']}__{c['rung']}__{c['tag']}__s{c['seed']}")


# ------------------------------------------------------------------ source-coverage pre-flight (FIX D)
def arm_coverage(arm, D):
    """(n_src, train_cov, held_cov) for `arm`: how many of its source nodes are train/held-out perturbations.
    held_cov==0 on a non-null arm means it CANNOT emit for ANY held-out perturbation -> ψ(0)=0 makes its
    zeroshot prediction identically the empty graph. train_cov==0 means the same on the rung1 (train-pert)
    eval set. Computed from the graph alone, at zero GPU cost — the check that would have caught the T0 zero."""
    s, _t, _w = G.build_arm(arm, D["g2i"], D["N"])[:3]
    srcset = set(np.unique(np.asarray(s, np.int64)).tolist())
    pidx = D["pidx"]; split = D["split"]; ev = np.where(pidx >= 0)[0]
    train_g = set(pidx[ev[split[ev] == 0]].tolist()); held_g = set(pidx[ev[split[ev] > 0]].tolist())
    return len(srcset), len(srcset & train_g), len(srcset & held_g)


def _eval_coverage(rung, train_cov, held_cov, n_src):
    return held_cov if rung == "zeroshot" else (train_cov if rung == "rung1" else n_src)


# ------------------------------------------------------------------ φ_e modification for controls
def _apply_tag(phi, tag, profile, with_dash, seed):
    if phi is None or tag == "main" or phi.shape[0] == 0:
        return phi
    cols = EF3F.cols_for(profile, with_dash) or []
    phi = phi.copy()
    if tag.startswith("drop_"):
        col = tag[len("drop_"):]
        if col not in cols:   # BLOCKER 2: RAISE instead of silently no-op'ing (which would fake v7_surviving=1.000)
            raise ValueError(f"_apply_tag: drop_ column '{col}' absent from profile '{profile}' (cols={cols}). "
                             f"A no-op here fakes a 'pure artifact' verdict. drop_/perm_ are directed_topo-only "
                             f"and were removed from build_plan for clean (BLOCKER 2).")
        phi[:, cols.index(col)] = 0.0
    elif tag.startswith("perm_"):
        col = tag[len("perm_"):]
        if col not in cols:
            raise ValueError(f"_apply_tag: perm_ column '{col}' absent from profile '{profile}' (cols={cols}) — "
                             f"a no-op fakes a false v7_surviving. See BLOCKER 2.")
        rng = np.random.default_rng(1234 + seed)
        phi[:, cols.index(col)] = phi[rng.permutation(phi.shape[0]), cols.index(col)]
    elif tag == "phishuf":
        rng = np.random.default_rng(9999 + seed)
        phi = phi[rng.permutation(phi.shape[0])]
    return phi


# ------------------------------------------------------------------ train + score one cell
def _cfg_hp(c):
    return dict(d=int(c["d"]), d_hidden=int(c["d_hidden"]), K=int(c["K"]), oversmooth="jk",
                dropout=c.get("dropout", 0.1), teleport_alpha=c.get("teleport_alpha", 0.05),
                gcnii_beta=c.get("gcnii_beta", 0.1), dropedge_p=c.get("dropedge_p", 0.0),
                lr=c.get("lr", 1e-3), weight_decay=c.get("weight_decay", 1e-4),
                batch_perts=int(c.get("batch_perts", 16)), epochs=c.get("epochs", 130),
                patience=c.get("patience", 12), min_delta=c.get("min_delta", 5e-4),
                grad_clip=1.0, corr_weight=1.0, mse_weight=0.1, delta_mode=c.get("delta_mode", "neg_mu_ctrl"))


def _score(D, s_true, pred, eval_ix, arm, K, Zfit):
    st = s_true[eval_ix]; mask = D["signal_mask"]; pe = D["pidx"][eval_ix]
    o = {"rsc": M.rsc_per_pert(st, pred, mask, pe),
         "rsc_coexpr_resid": M.rsc_coexpr_resid_per_pert(st, pred, mask, pe, Zfit)}
    try:
        dist, sr = G.load_bfs(REF_HOP_ARM)   # item 12: FIXED reference hop mask for ALL arms (not per-arm `arm`)
        ds, _ = M.dsrsc_per_pert(st, pred, mask, pe, dist, sr, 2, BFS_CAP)
        d2, _, _ = M.dnsa_ge2hop_per_pert(st, pred, mask, pe, dist, sr, K, min_hops=2)
        o["dsrsc_ge2"] = ds; o["dnsa_ge2hop"] = d2
    except Exception:
        o["dsrsc_ge2"] = np.full(len(eval_ix), np.nan); o["dnsa_ge2hop"] = np.full(len(eval_ix), np.nan)
    return o


class ZeroCoverageRefusal(RuntimeError):
    """A non-null arm cannot emit for ANY perturbation on this rung's eval set (structural). Raised BEFORE any
    GPU training so a 0-coverage cell never burns compute, UNLESS deliberately allowed to record the finding."""


def _distinguishable_from_empty(pred):
    """The empty graph (E=0) predicts exactly the seed node per row -> <=2 unique values per row. A graph with
    real source coverage reaches other genes via message passing -> >2 unique values in >=1 row. This is the
    self-contained form of the report's `not np.allclose(pred, pred_empty)` gate (no empty array needed)."""
    if pred.shape[0] == 0:
        return False
    return int(max(len(np.unique(pred[r])) for r in range(pred.shape[0]))) > 2


def run_cell(c, D, cfg, feats, profile, with_dash, device, allow_zero_cov=False):
    arm = c["arm"]; seed = c["seed"]; rung = c["rung"]; tag = c["tag"]
    hp = _cfg_hp(cfg); K = hp["K"]; N = D["N"]; pidx = D["pidx"]; eval_ix = np.where(pidx >= 0)[0]
    split = D["split"]
    is_null = arm in NULLS
    # ---- FIX D: source-coverage pre-flight (zero GPU cost) — the check that would have caught the T0 zero ----
    n_src, train_cov, held_cov = arm_coverage(arm, D)
    eval_cov = _eval_coverage(rung, train_cov, held_cov, n_src)
    # On zeroshot a causal arm has held_cov==0 by construction (its sources are the TRAIN perts). That zero IS
    # the structural finding and must be recorded — but only when explicitly allowed, and only on zeroshot; a
    # 0-coverage non-null arm on rung1 (train-pert eval) is a genuine bug and always refuses.
    deliberate_zero = is_null is False and eval_cov == 0 and allow_zero_cov and rung == "zeroshot"
    if (not is_null) and eval_cov == 0 and not deliberate_zero:
        raise ZeroCoverageRefusal(
            f"{arm} has 0 source coverage on the {rung} eval set (n_src={n_src}, train_cov={train_cov}, "
            f"held_cov={held_cov}). ψ(0)=0 => its prediction is identically the empty graph. Refusing to burn "
            f"GPU. On zeroshot set allow_zero_coverage to RECORD this as the structural finding.")
    cov = dict(n_src=n_src, train_cov=train_cov, held_cov=held_cov, eval_cov=eval_cov,
               zero_coverage_finding=bool(deliberate_zero))

    graph = G.build_arm(arm, D["g2i"], N)[:3]
    phi, c_g, hop_w = H.arm_inputs(arm, feats, D, profile=profile, with_dash=with_dash)
    phi = _apply_tag(phi, tag, profile, with_dash, seed)
    kw = dict(variant="assembled", feats=feats, phi_e=phi, c_g=c_g, hop_w=hop_w, delta_mode=hp["delta_mode"])
    if rung == "zeroshot":
        # exp_029 RPE1: faithful train->held-out zero-shot (matches exp_025/lfc_targets, not the native random
        # 2-fold). Fit on the TRAIN perts (split==0), predict the val+test held-out perts (split>0). The coexpr
        # baseline is built TRAIN-ONLY -> leakage-safe. per-pert vectors + the pred dump are over the held-out set.
        # NOTE: causal arms have held_cov==0 here -> a STRUCTURAL zero (documented), not a graph-quality result.
        fit = eval_ix[split[eval_ix] == 0]; ev = eval_ix[split[eval_ix] > 0]
        assert len(np.intersect1d(fit, ev)) == 0, "zeroshot firewall: train ∩ held-out must be empty"
        p, _ = H.train_one(graph, pidx[fit], D["s_full"][fit], pidx[ev], D["s_full"][ev],
                           D["signal_mask"], D["mu_ctrl"], hp, seed=seed, device=device, **kw)
        sc = _score(D, D["s_full"], p, ev, arm, K, M.build_coexpr_Z(D["s_full"][fit]))
        pred = p
    elif rung == "rung2":
        rng = np.random.default_rng(seed); perm = rng.permutation(len(eval_ix)); half = len(perm) // 2
        fo = [perm[:half], perm[half:]]; pred = np.zeros((len(eval_ix), N)); folds = []
        for fa, fb in [(0, 1), (1, 0)]:
            fit = eval_ix[fo[fa]]; ev = eval_ix[fo[fb]]
            assert len(np.intersect1d(fit, ev)) == 0, "rung2 firewall: fit ∩ eval must be empty"
            p, _ = H.train_one(graph, pidx[fit], D["s_full"][fit], pidx[ev], D["s_full"][ev],
                               D["signal_mask"], D["mu_ctrl"], hp, seed=seed, device=device, **kw)
            pred[fo[fb]] = p
            folds.append((fo[fb], _score(D, D["s_full"], p, ev, arm, K, M.build_coexpr_Z(D["s_full"][fit]))))
        sc = {}
        for kmet in folds[0][1]:
            v = np.full(len(eval_ix), np.nan)
            for idxs, s in folds:
                v[idxs] = s[kmet]
            sc[kmet] = v
        ev = eval_ix
    else:  # rung1 — cell-held-out on the COVERAGE-ALIGNED TRAINING perturbations.
        # FIX A + Patrick's constraint: fit on cell-half A / eval on cell-half B of the TRAIN perts (split==0),
        # where fungi_bio covers ~865/878 of the evaluated genes as sources. Evaluating on held-out perts here
        # would reproduce the zeroshot zero in a new costume (causal arms are still 0-coverage there). The two
        # halves are DISJOINT unit sets -> cell-held-out firewall; s_A/s_B must be GENUINE (guard below).
        if np.array_equal(D["s_A"], D["s_B"]):
            raise RuntimeError(
                "rung1 requested but s_A == s_B (degenerate cell-half data). Rebuild the data package with "
                "genuine cell halves (build_cellhalf_lfc.py -> build_rpe1_gauntlet_data.py). Refusing to run a "
                "FAKE in-sample rung1 (train==eval).")
        r1 = eval_ix[split[eval_ix] == 0]                        # TRAIN perts only (coverage-aligned)
        outs = []; _preds = []
        for fs, es in [(D["s_A"], D["s_B"]), (D["s_B"], D["s_A"])]:
            p, _ = H.train_one(graph, pidx[r1], fs[r1], pidx[r1], es[r1],
                               D["signal_mask"], D["mu_ctrl"], hp, seed=seed, device=device, **kw)
            _preds.append(p)
            outs.append(_score(D, es, p, r1, arm, K, M.build_coexpr_Z(fs[r1])))
        sc = {k: np.nanmean(np.stack([outs[0][k], outs[1][k]]), axis=0) for k in outs[0]}
        pred = np.nanmean(np.stack(_preds), axis=0)
        ev = r1
    # ---- FIX D: post-train distinguishability gate (the report's `not allclose(pred, empty)` in self-contained
    # form). Skipped only for nulls and for the deliberately-recorded zeroshot structural zero. ----
    if (not is_null) and (not deliberate_zero) and (not _distinguishable_from_empty(pred)):
        raise ZeroCoverageRefusal(
            f"{arm} {rung} prediction is indistinguishable from the empty graph (<=2 unique values/row) despite "
            f"eval_cov={eval_cov}. This is the T0 failure signature — do not record it as a result.")
    return sc, pred, ev, cov


def _jsonable(sc):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in sc.items()}


def _resolve_rungs(cfg, D):
    """FIX A: rungs are CONFIG-DRIVEN, not hard-forced. A config `rungs` list wins; else default to
    ('zeroshot','rung1') when a split is present (RPE1) — zeroshot DOCUMENTS the structural coverage finding
    across all arms, rung1 is the FAIR cell-held-out graph-quality comparison on the covered training perts."""
    r = cfg.get("rungs")
    if r:
        return tuple(r)
    return ("zeroshot", "rung1") if D.get("split") is not None else ("rung2", "rung1")


def run_shard(shard, n_shards, device, allow_frozen=False, config_path=None, allow_zero_cov=False):
    cfg, profile, with_dash, src = resolve_config(path=config_path, allow_frozen=allow_frozen)
    feats = make_feats("assembled", profile=profile, film=cfg.get("film", True), ovsq=cfg.get("ovsq", True))
    D = G.load_data()
    allow_zero_cov = allow_zero_cov or bool(cfg.get("allow_zero_coverage_zeroshot", False))
    rungs = _resolve_rungs(cfg, D)
    ctag = cfg_tag(cfg, profile, with_dash)
    seeds = tuple(cfg.get("seeds_gauntlet", (0, 1, 2, 3, 4)))   # exp_031 patch3: config-driven seed count
    cells = build_plan(seeds_main=seeds, rungs=rungs, ctag=ctag)  # (resumable 5->10 top-up: cell_key encodes seed)
    os.makedirs(CELLS, exist_ok=True)
    log(f"config: {src} | profile={profile} d{cfg['d']}/dh{cfg['d_hidden']}/K{cfg['K']}/a{cfg.get('teleport_alpha')}"
        f"/g{cfg.get('gcnii_beta')} | rungs={list(rungs)} cfg={ctag} allow_zero_cov={allow_zero_cov} "
        f"| plan {len(cells)} cells | shard {shard}/{n_shards} | device={device}")
    done = 0
    for idx, c in enumerate(cells):
        if idx % n_shards != shard:
            continue
        out = os.path.join(CELLS, cell_key(c) + ".json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            sc, pred, eval_ix, cov = run_cell(c, D, cfg, feats, profile, with_dash, device,
                                              allow_zero_cov=allow_zero_cov)
        except ZeroCoverageRefusal as e:
            log(f"CELL REFUSED (coverage) {cell_key(c)}: {e}"); continue
        except Exception as e:
            log(f"CELL FAILED {cell_key(c)}: {type(e).__name__}: {e}"); continue
        rec = dict(**c, secs=round(time.time() - t0, 1), rsc_mean=float(np.nanmean(sc["rsc"])),
                   n_src=cov["n_src"], train_cov=cov["train_cov"], held_cov=cov["held_cov"],
                   eval_cov=cov["eval_cov"], zero_coverage_finding=cov["zero_coverage_finding"],
                   n_eval=int(len(eval_ix)), perpert=_jsonable(sc))
        tmp = out + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, out); done += 1
        # item 10 (non-negotiable): obj_010 Mode-B prediction dump — makes every downstream metric free + the run
        # re-analysable without a GPU-second. pred/true are per-eval-pert delta-LFC; means = mu_ctrl + delta.
        try:
            os.makedirs(PREDS, exist_ok=True)
            np.savez_compressed(os.path.join(PREDS, cell_key(c) + ".npz"),
                                pred=np.asarray(pred, np.float32), true=np.asarray(D["s_full"][eval_ix], np.float32),
                                pert_idx=np.asarray(eval_ix, np.int64), mu_ctrl=np.asarray(D["mu_ctrl"], np.float64),
                                signal_mask=np.asarray(D["signal_mask"], bool))
        except Exception as e:
            log(f"PRED-DUMP FAILED {cell_key(c)}: {type(e).__name__}: {e}")
        log(f"[{idx}] {cell_key(c):40s} RSC={rec['rsc_mean']:+.4f} ({rec['secs']}s)")
    log(f"shard {shard} complete: wrote {done} new cells")


# ------------------------------------------------------------------ summarize
def _load_cells():
    out = {}
    if os.path.isdir(CELLS):
        for fn in os.listdir(CELLS):
            if fn.endswith(".json"):
                out[fn[:-5]] = json.load(open(os.path.join(CELLS, fn)))
    return out


def _pool(cells, arm, rung, tag, metric):
    vecs = [np.asarray(r["perpert"][metric], float) for r in cells.values()
            if r["arm"] == arm and r["rung"] == rung and r["tag"] == tag and metric in r["perpert"]]
    return np.nanmean(np.stack(vecs), axis=0) if vecs else None


def _cov_of(cells, arm, rung):
    """Pull the recorded coverage for an arm/rung from any matching cell (FIX D reporting)."""
    for r in cells.values():
        if r["arm"] == arm and r["rung"] == rung and "eval_cov" in r:
            return {k: r.get(k) for k in ("n_src", "train_cov", "held_cov", "eval_cov", "zero_coverage_finding")}
    return {}


def summarize():
    cfg, profile, with_dash, src = resolve_config(allow_frozen=True)  # reporting only (not a training run)
    cells = _load_cells()
    if not cells:
        log("no cells found - nothing to summarize"); return
    import pandas as pd
    rows = []
    # FIX A: zeroshot documents the structural coverage finding; rung1 is the fair comparison; rung2 if present.
    for rung in ["zeroshot", "rung1", "rung2"]:
        f = {m: _pool(cells, "fungi_bio", rung, "main", m) for m in METRICS}
        if f["rsc"] is None:
            continue
        for arm in GAUNTLET[1:] + NULLS:
            row = dict(rung=rung, arm=arm)
            cov = _cov_of(cells, arm, rung)
            row.update({f"cov_{k}": cov.get(k) for k in ("eval_cov", "held_cov", "train_cov", "zero_coverage_finding")})
            for m in METRICS:
                fv = f[m]; av = _pool(cells, arm, rung, "main", m)
                if fv is None or av is None:
                    row[f"gap_{m}"] = np.nan; continue
                pb = M.paired_bootstrap(fv, av, n_boot=cfg.get("bootstrap_B", 10000), seed=2)
                row[f"gap_{m}"] = round(pb["mean_diff"], 4)
                if m == "rsc":
                    row["gap_rsc_p"] = round(pb["p_one_sided"], 5)
                    row["fungi_rsc"] = round(float(np.nanmean(fv)), 4); row["arm_rsc"] = round(float(np.nanmean(av)), 4)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "gauntlet.csv"), index=False)

    # attribution + V7: drop_/perm_ columns only exist under rung2 (directed_topo); the RPE1 clean run has no
    # rung2 -> this block is skipped gracefully. `phishuf` (whole-φ_e shuffle) now runs on the RUNG the plan put
    # it on (rung1 when present) and is reported against the same-rung fungi_bio/top_weight base gap.
    attr = {"profile": profile, "with_dash": with_dash, "config_source": src, "columns": {}}
    ft = _pool(cells, "fungi_bio", "rung2", "main", "rsc"); tt = _pool(cells, "top_weight", "rung2", "main", "rsc")
    if ft is not None and tt is not None:
        base_gap = float(np.nanmean(ft) - np.nanmean(tt)); attr["base_gap_vs_top"] = round(base_gap, 4)
        for col in NEW_COLS:
            fd = _pool(cells, "fungi_bio", "rung2", f"drop_{col}", "rsc")
            td = _pool(cells, "top_weight", "rung2", f"drop_{col}", "rsc")
            fp = _pool(cells, "fungi_bio", "rung2", f"perm_{col}", "rsc")
            rec = {}
            if fd is not None and td is not None:
                drop_gap = float(np.nanmean(fd) - np.nanmean(td)); rec["drop_gap"] = round(drop_gap, 4)
                rec["load_bearing"] = round(base_gap - drop_gap, 4)          # how much zeroing the col costs
            if fp is not None:
                perm_gap = float(np.nanmean(fp) - np.nanmean(tt)); rec["perm_gap"] = round(perm_gap, 4)
                rec["perm_delta"] = round(base_gap - perm_gap, 4)            # how much permuting the col costs
                rec["v7_surviving_frac"] = round(perm_gap / (base_gap + 1e-9), 3)  # want << 1 for a real feature
            attr["columns"][col] = rec
    # phishuf: try rung1 first (where the plan puts it), then rung2.
    for r_phi in ("rung1", "rung2"):
        fs = _pool(cells, "fungi_bio", r_phi, "phishuf", "rsc")
        fb = _pool(cells, "fungi_bio", r_phi, "main", "rsc"); tb = _pool(cells, "top_weight", r_phi, "main", "rsc")
        if fs is not None and fb is not None and tb is not None:
            bg = float(np.nanmean(fb) - np.nanmean(tb))
            attr["phishuf_all"] = dict(rung=r_phi, gap=round(float(np.nanmean(fs) - np.nanmean(tb)), 4),
                                       surviving_frac=round(float(np.nanmean(fs) - np.nanmean(tb)) / (bg + 1e-9), 3))
            break
    json.dump(attr, open(os.path.join(RES, "reg_feature_attribution.json"), "w"), indent=2, default=float)
    log(f"summarize -> gauntlet.csv + reg_feature_attribution.json (base_gap_vs_top={attr.get('base_gap_vs_top')})")
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def dry_run(config_path=None):
    # thread --config so the canary's dry-run reflects the REAL run's config (allow_frozen only if no --config)
    cfg, profile, with_dash, src = resolve_config(path=config_path, allow_frozen=(config_path is None))
    try:
        D = G.load_data(); rungs = _resolve_rungs(cfg, D)
    except Exception:
        rungs = tuple(cfg.get("rungs") or ("zeroshot", "rung1"))    # data not built yet: trust the config
    ctag = cfg_tag(cfg, profile, with_dash)
    seeds = tuple(cfg.get("seeds_gauntlet", (0, 1, 2, 3, 4)))   # exp_031 patch3: config-driven seed count
    cells = build_plan(seeds_main=seeds, rungs=rungs, ctag=ctag)
    allow_zero_cov = bool(cfg.get("allow_zero_coverage_zeroshot", False))
    from collections import Counter
    bc = Counter((c["tag"].split("_")[0] if c["tag"] != "main" else "main") for c in cells)
    print(f"\n=== gauntlet PLAN (config: {src}; profile={profile}; d{cfg['d']}/dh{cfg['d_hidden']}/K{cfg['K']}) ===")
    print(f"arms: {GAUNTLET} + nulls {NULLS} ; rungs {list(rungs)} ; cfg={ctag} ; allow_zero_cov={allow_zero_cov} ; "
          f"main seeds 0-4 ; phishuf seeds 0-1")
    print(f"cell tag breakdown: {dict(bc)}")
    # fix (7): no fake flat GPU-h constant. Wall = TOTAL cells x (measured s/cell) x 2 trainings/cell; the CANARY
    # produces the real s/cell on the V100 (obj_009.1 prior: ~90 min/HPO-trial; a gauntlet cell is much smaller).
    _sc = os.environ.get("RHIZO_S_PER_CELL")
    if _sc:
        h = len(cells) * 2 * float(_sc) / 3600.0
        print(f"TOTAL cells: {len(cells)} (each = 2 trainings). Wall ~{h:.1f} GPU-h at {_sc}s/training "
              f"(from RHIZO_S_PER_CELL).")
    else:
        print(f"TOTAL cells: {len(cells)} (each = 2 trainings). Wall = cells x 2 x (s/training); run the canary to "
              f"get the real s/training, then set RHIZO_S_PER_CELL=<sec> for a real estimate. (No fake constant.)")
    print("Resumable: each cell -> results/cells/<key>.json; re-run skips existing.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--config", default=None)
    ap.add_argument("--allow_frozen_fallback", action="store_true",
                    help="item 9: deliberately run the frozen_hp config when no HPO winner exists (default: REFUSE).")
    ap.add_argument("--allow_zero_coverage", action="store_true",
                    help="FIX D: on the zeroshot rung, RECORD (do not refuse) non-null arms with 0 held-out source "
                         "coverage — the deliberate way to document the causal-arm structural zero. Also settable "
                         "via config key allow_zero_coverage_zeroshot: true. Never bypasses the rung1 guard.")
    ap.add_argument("--results_subdir", default=os.environ.get("RHIZO_RESULTS_SUBDIR"),
                    help="exp_031 patch3: write cells/preds/logs under results/<subdir> (per-tier isolation, e.g. "
                         "rung1 / zeroshot). Default: results/ (patch2 behaviour). RHIZO_PRED_DIR still overrides PREDS.")
    args = ap.parse_args()
    _apply_results_subdir(args.results_subdir)
    if args.dry_run:
        return dry_run(config_path=args.config)
    if args.summarize:
        return summarize()
    run_shard(args.shard, args.n_shards, args.device, args.allow_frozen_fallback, config_path=args.config,
              allow_zero_cov=args.allow_zero_coverage)


if __name__ == "__main__":
    main()
