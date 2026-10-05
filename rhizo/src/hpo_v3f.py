"""
obj_009.3 (RHIZO-final) — hpo_v3f.py : the assembled model's OWN HPO driver + VRAM-AWARE DYNAMIC STACKER for
2-3 V100s. Generates trial configs from configs/hpo_v3f.yaml (SEEDED from the known-good region but RE-SEARCHED
— seed 200, no overlap with prior sweeps), estimates each config's VRAM, and PACKS each requested V100: greedily
co-schedules as many run_trial_v3f.py subprocesses as fit in the usable budget, double-checking nvidia-smi before
each launch. Fully resumable (skips trials whose result json exists). Multi-GPU (--gpus 0,1,2), each packed.

  python src/hpo_v3f.py --gpus 0,1          # sweep, packing 2 V100s
  python src/hpo_v3f.py --dry_run           # print schedule + VRAM estimates, run nothing (no GPU needed)
  python src/hpo_v3f.py --summarize         # (re)build results/hpo_leaderboard.csv from finished trials
"""
from __future__ import annotations
import os, sys, json, time, argparse, subprocess, random
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
RES = os.path.join(BUNDLE, "results")
TRIALS = os.path.join(RES, "trials")
CFG_DIR = os.path.join(RES, "trial_configs")
LOG = os.path.join(RES, "hpo_v3f.log")
PY = sys.executable
BATCH_LADDER = [32, 24, 16, 12, 8]

# the known-good region (mapped onto the assembled model) — ANCHORS to beat, NOT inherited (Patrick's rule).
ANCHORS = [
    dict(edge_profile="directed_topo", d=64, d_hidden=128, K=6, teleport_alpha=0.15, gcnii_beta=0.1,
         film=True, ovsq=True, lr=1e-3, weight_decay=1e-4, batch_perts=16, _anchor="v_fuse_t0001"),
    dict(edge_profile="directed_topo", d=32, d_hidden=64, K=12, teleport_alpha=0.05, gcnii_beta=0.0,
         film=True, ovsq=True, lr=1e-3, weight_decay=1e-4, batch_perts=16, _anchor="5090_t0013"),
    dict(edge_profile="clean", d=32, d_hidden=64, K=12, teleport_alpha=0.05, gcnii_beta=0.0,
         film=True, ovsq=True, lr=1e-3, weight_decay=1e-4, batch_perts=16, _anchor="5090_t0013_clean"),
]
SEARCH_KEYS = ["edge_profile", "d", "d_hidden", "K", "teleport_alpha", "gcnii_beta", "film", "ovsq",
               "lr", "weight_decay", "batch_perts"]


def log(m):
    line = f"[hpo_v3f {time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- VRAM admission (DEFECT 3)
# The 5090-fit polynomial has NO edge-count term, yet the dominant activation is the (E,B,d) message tensor
# (incident: E*B*d*4 = 501.8 MiB reconciled the OOM traceback exactly). So: (1) est_vram_gb gains an explicit
# E term as a FALLBACK, and (2) the packer PREFERS an EMPIRICAL measurement from the Step-4a probe cache
# (results/vram_probe/*.json) whenever one exists. Never trust the formula alone across machines/edge counts.
PROBE_DIR = os.path.join(RES, "vram_probe")
_REF_E = None   # HPO reference-graph edge count; set from cfg['hardware']['ref_edge_count'] in run_sweep.


def _cfg_sig(d, d_hidden, K, batch, edge_profile):
    return f"d{d}_dh{d_hidden}_K{K}_b{batch}_{edge_profile}"


def probe_vram_gb(d, d_hidden, K, batch, edge_profile):
    """Return the MEASURED peak_vram_gb from the Step-4a probe cache for this config, else None."""
    p = os.path.join(PROBE_DIR, _cfg_sig(d, d_hidden, K, batch, edge_profile) + ".json")
    if os.path.exists(p):
        try:
            return float(json.load(open(p)).get("peak_vram_gb"))
        except Exception:
            return None
    return None


def est_vram_gb(d, d_hidden, K, batch, edge_profile, E=None):
    """FALLBACK estimate (used only when no empirical probe exists). d32/dh64/K12/b16 -> ~27.5 GB on the 5090
    fit; the E term adds the (E,B,d) message activation across ~K layers with a fwd+grad multiplier."""
    fixed = 1.5
    base = 0.387 * batch * (K / 4.0) * ((d + d_hidden) / 96.0)
    vf = 1.15 if edge_profile in ("directed_topo", "regulatory") else 1.08
    est = fixed + base * vf
    E = _REF_E if E is None else E
    if E:
        est += 3.0 * K * (E * batch * d * 4) / (1024 ** 3)   # (E,B,d) msg tensor × K layers × ~3 (fwd+grad)
    return est


def admit_vram_gb(d, d_hidden, K, batch, edge_profile):
    """Admission VRAM: MEASURED probe if available (authoritative), else the E-aware fallback estimate."""
    m = probe_vram_gb(d, d_hidden, K, batch, edge_profile)
    return (m, "measured") if m is not None else (est_vram_gb(d, d_hidden, K, batch, edge_profile), "estimated")


def fit_batch(c, usable_gb, min_batch, safety):
    start = c["batch_perts"]
    for b in [x for x in BATCH_LADDER if x <= start] or [start]:
        if b < min_batch:
            break
        val, src = admit_vram_gb(c["d"], c["d_hidden"], c["K"], b, c["edge_profile"])
        est = val * safety
        if est <= usable_gb:
            c["batch_perts"] = b
            c["_vram_src"] = src
            return est
    return None


def _feasible_batch(c, hw):
    cc = dict(c)
    est = fit_batch(cc, float(hw["usable_vram_gb"]), int(hw["min_batch"]), float(hw["safety_factor"]))
    return cc["batch_perts"] if est is not None else None


def gen_configs(cfg):
    base = dict(cfg["base"]); s = cfg["search"]; smp = cfg["sampler"]; hw = cfg["hardware"]
    if isinstance(base.get("rung"), list):        # sweep objective on the first rung (rung2); winner confirmed on both
        base["rung"] = base["rung"][0]
    variant = cfg.get("model", {}).get("variant", "assembled")
    rng = random.Random(smp.get("seed", 200))
    combos = []
    for a in ANCHORS:                              # anchors first (depth-0 references to beat)
        c = {k: a[k] for k in SEARCH_KEYS}; c["_anchor"] = a["_anchor"]
        if _feasible_batch(c, hw) is not None:
            combos.append(c)
    seen = set(); tries = 0; n_skipped = 0
    while len([c for c in combos if not c.get("_anchor")]) < smp.get("n_trials", 40) and tries < 500000:
        tries += 1
        c = {k: rng.choice(s[k]) for k in SEARCH_KEYS}
        key = tuple(c[k] for k in SEARCH_KEYS)
        if key in seen:
            continue
        seen.add(key)
        if _feasible_batch(c, hw) is None:
            n_skipped += 1; continue
        combos.append(c)
    log(f"config gen: {len([c for c in combos if not c.get('_anchor')])} feasible random trials "
        f"(+{len([c for c in combos if c.get('_anchor')])} anchors); {n_skipped} draws rejected as >VRAM")
    out = []
    for i, c in enumerate(combos):
        cc = dict(base, variant=variant, **c)
        cc["trial_id"] = f"t{i:04d}"
        out.append(cc)
    return out


# ---------------------------------------------------------------- nvidia-smi live check
def gpu_used_total_mb(idx):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits", "-i", str(idx)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = out.split("\n")[0].split(",")
        return int(used), int(total)
    except Exception:
        return 0, 32768


def reconcile_gpus(running):
    """DEFECT 14: the packer's avail[g]/n_on_g bookkeeping was once internally consistent AND entirely wrong
    (all children stacked on cuda:0 while the packer logged distinct cards). Reconcile our LAUNCHED pids against
    ground truth: nvidia-smi says which physical card each pid actually runs on. If a running trial's real card
    != the card the packer assigned it, abort (defect 2 regressed, or a stacking bug). Returns (ok, msg)."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return True, "nvidia-smi unavailable (skip reconcile)"
    uuid2idx = {}
    for ln in idx.splitlines():
        if "," in ln:
            i, u = ln.split(",", 1); uuid2idx[u.strip()] = int(i.strip())
    pid2card = {}
    for ln in apps.splitlines():
        if "," in ln:
            p, u = ln.split(",", 1)
            try: pid2card[int(p.strip())] = uuid2idx.get(u.strip())
            except Exception: pass
    for r in running:
        pid = r["popen"].pid
        real = pid2card.get(pid)
        if real is not None and real != r["gpu"]:
            return False, f"trial {r['tid']} pid {pid} assigned gpu{r['gpu']} but nvidia-smi runs it on gpu{real}"
    return True, "ok"


def run_sweep(cfg, gpus, dry):
    os.makedirs(TRIALS, exist_ok=True); os.makedirs(CFG_DIR, exist_ok=True); os.makedirs(PROBE_DIR, exist_ok=True)
    global _REF_E
    _REF_E = int(cfg["hardware"].get("ref_edge_count", 200000))  # DEFECT 3: E for the fallback estimate (RPE1=200k)
    log(f"VRAM admission: probe cache {PROBE_DIR} (measured wins) | fallback est E={_REF_E}")
    usable = float(cfg["hardware"]["usable_vram_gb"]); maxc = int(cfg["hardware"]["max_concurrent"])
    minb = int(cfg["hardware"]["min_batch"]); safety = float(cfg["hardware"]["safety_factor"])
    configs = gen_configs(cfg)
    runnable, toobig, done = [], [], []
    for c in configs:
        out = os.path.join(TRIALS, f"{c['trial_id']}.json")
        if os.path.exists(out):
            done.append(c["trial_id"]); continue
        est = fit_batch(c, usable, minb, safety)
        if est is None:
            toobig.append(c); continue
        c["_est"] = est
        cfgp = os.path.join(CFG_DIR, f"{c['trial_id']}.json"); json.dump(c, open(cfgp, "w"), indent=2, default=float)
        runnable.append((c, cfgp, est))
    log(f"configs: {len(configs)} | done {len(done)} | runnable {len(runnable)} | too-big {len(toobig)}")
    if dry:
        log("--- DRY RUN schedule (est VRAM/trial, sorted) ---")
        for c, _, est in sorted(runnable, key=lambda x: -x[2]):
            n_fit = max(1, int(usable // est))
            log(f"  {c['trial_id']} [{c['edge_profile']:13s}] d{c['d']}/dh{c['d_hidden']}/K{c['K']}/"
                f"a{c['teleport_alpha']}/g{c['gcnii_beta']}/b{c['batch_perts']} -> est {est:.1f}GB -> stacks "
                f"{min(n_fit, maxc)}-up" + (f"  [anchor {c['_anchor']}]" if c.get("_anchor") else ""))
        return
    avail = {g: usable for g in gpus}; running = []; queue = list(runnable)
    n_completed = 0; n_no_output = 0; _recon = 0   # DEFECT 8 fail-fast + DEFECT 14 reconcile counter
    while queue or running:
        for g in gpus:
            n_on_g = sum(1 for r in running if r["gpu"] == g)
            while queue and n_on_g < maxc:
                c, cfgp, est = queue[0]
                if avail[g] < est:
                    break
                used_mb, total_mb = gpu_used_total_mb(g)
                if used_mb + est * 1000 > total_mb * 0.97:
                    break
                out = os.path.join(TRIALS, f"{c['trial_id']}.json")
                # DEFECT 2 (the false-result defect): SLURM pre-sets CUDA_VISIBLE_DEVICES="0,1,.." under
                # --gres=gpu:N, so a `setdefault` downstream is a silent no-op and every child stacks on cuda:0
                # -> OOM while the packer logs the right gpu. Set it EXPLICITLY here so the child sees ONLY card g.
                env = dict(os.environ, HPO_GPU=str(g), CUDA_VISIBLE_DEVICES=str(g))
                p = subprocess.Popen([PY, os.path.join(HERE, "run_trial_v3f.py"), "--config", cfgp,
                                      "--out", out, "--gpu", str(g)], env=env,
                                     stdout=open(os.path.join(RES, f"trial_{c['trial_id']}.log"), "w"),
                                     stderr=subprocess.STDOUT)
                running.append(dict(popen=p, gpu=g, est=est, tid=c["trial_id"], out=out))
                avail[g] -= est; n_on_g += 1; queue.pop(0)
                log(f"launch {c['trial_id']} gpu{g} est {est:.1f}GB (avail {avail[g]:.1f}GB, {n_on_g} on gpu{g})")
        for r in list(running):
            if r["popen"].poll() is not None:
                avail[r["gpu"]] += r["est"]
                ok = os.path.exists(r["out"])
                log(f"  done {r['tid']} rc={r['popen'].returncode} {'ok' if ok else 'NO-OUTPUT'}")
                running.remove(r)
                n_completed += 1
                if not ok:
                    n_no_output += 1
                # DEFECT 8: if the FIRST 2 completed trials both produced no output, something is
                # systemically broken (GPU stacking / bad env / import failure) -> abort NOW rather than
                # burning the whole sweep to an empty leaderboard that silently falls back to frozen_hp.
                if n_completed >= 2 and n_no_output >= 2 and n_no_output == n_completed:
                    for rr in running:
                        try: rr["popen"].terminate()
                        except Exception: pass
                    raise SystemExit(f"FAIL-FAST: first {n_completed} trials all NO-OUTPUT — sweep aborted. "
                                     f"Check GPU isolation (defect 2), PY path (defect 4), and results/trial_*.log.")
        _recon += 1
        if len(gpus) > 1 and running and _recon % 10 == 0:   # DEFECT 14: reconcile every ~30s on multi-GPU
            ok, msg = reconcile_gpus(running)
            if not ok:
                for rr in running:
                    try: rr["popen"].terminate()
                    except Exception: pass
                raise SystemExit(f"FAIL-FAST(defect 14): GPU bookkeeping diverged from nvidia-smi -> {msg}. "
                                 f"Aborting rather than trusting a packer state that does not match reality.")
        time.sleep(3)
    log("sweep complete"); summarize()


def summarize():
    import pandas as pd, glob
    rows = []
    for p in glob.glob(os.path.join(TRIALS, "*.json")):
        d = json.load(open(p)); c = d.get("config", {})
        gap = d.get("gap"); dn = d.get("dnsa_ge2hop_gap")
        rows.append(dict(trial_id=d.get("trial_id"), edge_profile=d.get("edge_profile", c.get("edge_profile")),
                         d=c.get("d"), d_hidden=c.get("d_hidden"), K=c.get("K"), tele=c.get("teleport_alpha"),
                         gcnii=c.get("gcnii_beta"), batch=c.get("batch_perts"), lr=c.get("lr"),
                         fungi_rsc=d.get("fungi_rsc"), topweight_rsc=d.get("topweight_rsc"), knn_rsc=d.get("knn_rsc"),
                         gap=gap, gap_p=d.get("gap_p"), gap_vs_knn=d.get("gap_vs_knn"),
                         gap_coexpr_resid_vs_top=d.get("gap_coexpr_resid_vs_top"),
                         gap_coexpr_resid_vs_knn=d.get("gap_coexpr_resid_vs_knn"),
                         dnsa_ge2hop_gap=dn, dnsa_ge2hop_gap_vs_knn=d.get("dnsa_ge2hop_gap_vs_knn"),
                         fungi_minus_shuffle=d.get("fungi_minus_shuffle"),
                         peak_vram_gb=d.get("peak_vram_gb"), secs=d.get("secs")))
    if not rows:
        log("no finished trials yet"); return
    df = pd.DataFrame(rows)
    # DEFECT 11: COMPOUND objective — a config is only ELIGIBLE if it beats BOTH baselines AND the shuffle null,
    # at K>=8 (the K=6 causal cliff). Rank eligible configs by the PRODUCT of the three margins. Never select on
    # aggregate gap alone (that promoted configs that lose to kNN / don't beat shuffle). dnsa is NOT in the objective.
    def _f(x):
        try: return float(x)
        except Exception: return float("nan")
    import numpy as _np
    g   = df["gap"].map(_f)
    gk  = df["gap_vs_knn"].map(_f) if "gap_vs_knn" in df else _np.full(len(df), _np.nan)
    fs  = df["fungi_minus_shuffle"].map(_f) if "fungi_minus_shuffle" in df else _np.full(len(df), _np.nan)
    kk  = df["K"].map(_f)
    df["compound_eligible"] = (g > 0) & (gk > 0) & (fs > 0) & (kk >= 8)
    df["compound_score"] = _np.where(df["compound_eligible"], g * gk * fs, _np.nan)
    df = df.sort_values(["compound_eligible", "compound_score", "gap"], ascending=[False, False, False],
                        na_position="last")
    df.to_csv(os.path.join(RES, "hpo_leaderboard.csv"), index=False)
    n_elig = int(df["compound_eligible"].sum())
    if n_elig > 0:
        top = df.iloc[0].to_dict()
        json.dump({k: top.get(k) for k in ("trial_id", "edge_profile", "d", "d_hidden", "K", "tele",
                   "gcnii", "batch", "lr", "gap", "gap_vs_knn", "fungi_minus_shuffle", "compound_score")},
                  open(os.path.join(RES, "hpo_winner.json"), "w"), indent=2, default=float)
        log(f"leaderboard -> hpo_leaderboard.csv ({len(df)} trials, {n_elig} COMPOUND-ELIGIBLE). "
            f"WINNER {top['trial_id']} [{top['edge_profile']}] K={top['K']} gap={top['gap']:+.4f} "
            f"gap_vs_knn={_f(top.get('gap_vs_knn')):+.4f} f-shuffle={_f(top.get('fungi_minus_shuffle')):+.4f} "
            f"compound={_f(top.get('compound_score')):.5f} -> hpo_winner.json")
    else:
        log(f"leaderboard -> hpo_leaderboard.csv ({len(df)} trials). *** NO COMPOUND-ELIGIBLE WINNER *** "
            f"(no config beats BOTH top_weight AND knn AND shuffle at K>=8). Gauntlet MUST NOT run frozen_hp — "
            f"this is the honest 'co-expression wins on RPE1' outcome; escalate to Patrick.")
    try:
        ab = df.groupby("edge_profile")["gap"].mean().to_dict()
        log(f"A/B mean gap by edge_profile: {ab}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(BUNDLE, "configs", "hpo_v3f.yaml"))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if args.summarize:
        return summarize()
    cfg = yaml.safe_load(open(args.config))
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else list(cfg["hardware"].get("gpus", [0]))
    log(f"=== HPO v3f START gpus={gpus} usable={cfg['hardware']['usable_vram_gb']}GB "
        f"max_concurrent={cfg['hardware']['max_concurrent']} dry={args.dry_run} ===")
    run_sweep(cfg, gpus, args.dry_run)


if __name__ == "__main__":
    main()
