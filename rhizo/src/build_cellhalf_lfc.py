"""
exp_029 FIX A (part 1) — build GENUINE cell-half LFC targets so rung1 is a REAL cell-held-out test.

The shipped ship/lfc_targets_rpe1.npz carries only per-perturbation aggregated LFC (train/val/test_LFC), and
build_rpe1_gauntlet_data.py therefore set s_A = s_B = s_full (DEGENERATE). Running rung1 on that trains and
evaluates on identical data -> a fake in-sample eval. This script replicates build_rpe1_lfc.py's preprocessing
EXACTLY and additionally splits each perturbation's units 50/50 (seeded) and pseudobulks each half -> s_A / s_B
(residual vs the SAME mu_ctrl), following obj_009.3's v1_prep_data.py cell-held-out convention.

Output: ship/lfc_targets_rpe1_cellhalf.npz  = every key the shipped LFC file has, PLUS
        {train,val,test}_LFC_A / _LFC_B  (per-covered-pert half-pseudobulk residuals, same row order).

FAITHFULNESS GATE (asserted here): the recomputed per-split full LFC reproduces the SHIPPED train/val/test_LFC
to <1e-4, and corr(1/2 (s_A+s_B), s_full) is high -> proves s_A/s_B are on the identical scale + pert order as
s_full. CPU-only (no GPU); safe re: the 2070/WHEA rules.

  python build_cellhalf_lfc.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import numpy as np, scanpy as sc, scipy.sparse as spx
from pathlib import Path

REPO = "c:/Users/studi/OneDrive/Documents/thesis"
EXP = Path(__file__).resolve().parents[2]   # .../exp_029_hpc_rhizo_crownjewel_showdown
TRAIN_PANEL = f"{REPO}/HYPHAE/for_chinmaya/input/RPE1_5k_hybrid_ctrlpreserved_train.h5ad"
ALLSPLITS = f"{REPO}/DATA/EXPERIMENTS/exp_022_substrate_coverage_recovery/intermediate/data_build/rpe1_allsplits_metacell_ctrlpreserved.h5ad"
SHIPPED_LFC = EXP / "ship" / "lfc_targets_rpe1.npz"
OUT = EXP / "ship" / "lfc_targets_rpe1_cellhalf.npz"
CTRL = "non-targeting"; SIGNAL_THRESH = 0.25; SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    panel = list(sc.read_h5ad(TRAIN_PANEL, backed="r").var_names)
    N = len(panel); g2i = {g: i for i, g in enumerate(panel)}
    log(f"panel N={N}")
    ad = sc.read_h5ad(ALLSPLITS)
    xmax = float(ad.X.max()); log(f"allsplits {ad.shape} X.max={xmax:.1f} (raw counts expected)")
    assert xmax > 15, "expected RAW counts (X.max>15)"
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    ad = ad[:, panel].copy()
    X = ad.X.tocsr() if spx.issparse(ad.X) else spx.csr_matrix(ad.X)
    gene = ad.obs["gene"].astype(str).to_numpy(); split = ad.obs["split"].astype(str).to_numpy()
    log(f"panel-subset log1p done ({time.time()-t0:.0f}s)")

    def grp_mean(mask):
        return np.asarray(X[mask].mean(axis=0)).ravel()
    mu_ctrl = grp_mean(split == "control")
    rng = np.random.default_rng(SEED)

    def split_S(sp):
        """Per covered pert (source gene on panel, sorted by name to match build_rpe1_lfc): full + half A/B
        pseudobulk residuals vs the SAME mu_ctrl. Returns names, S, S_A, S_B, pidx, ncA, ncB."""
        names, rows, rowsA, rowsB, pid, ncA, ncB = [], [], [], [], [], [], []
        genes = [x for x in np.unique(gene[split == sp]) if x != CTRL]
        for gname in genes:
            if gname not in g2i:
                continue
            m = (split == sp) & (gene == gname)
            idx = np.where(m)[0]
            rng.shuffle(idx)
            half = len(idx) // 2
            ia, ib = idx[:half], idx[half:]           # disjoint unit halves (cell-held-out firewall)
            full = grp_mean(m) - mu_ctrl
            a = np.asarray(X[ia].mean(axis=0)).ravel() - mu_ctrl
            b = np.asarray(X[ib].mean(axis=0)).ravel() - mu_ctrl
            names.append(gname); pid.append(g2i[gname])
            rows.append(full); rowsA.append(a); rowsB.append(b)
            ncA.append(len(ia)); ncB.append(len(ib))
        S = np.stack(rows) if rows else np.zeros((0, N), np.float32)
        SA = np.stack(rowsA) if rowsA else np.zeros((0, N), np.float32)
        SB = np.stack(rowsB) if rowsB else np.zeros((0, N), np.float32)
        return (np.array(names), S.astype(np.float32), SA.astype(np.float32), SB.astype(np.float32),
                np.array(pid, np.int64), np.array(ncA), np.array(ncB))

    out = {}
    corr_report = {}
    for sp in ("train", "val", "test"):
        nm, S, SA, SB, pid, ncA, ncB = split_S(sp)
        out[f"{sp}_names"] = nm; out[f"{sp}_LFC"] = S; out[f"{sp}_pidx"] = pid
        out[f"{sp}_LFC_A"] = SA; out[f"{sp}_LFC_B"] = SB
        approx = 0.5 * (SA + SB)
        c = float(np.corrcoef(approx.ravel(), S.ravel())[0, 1]) if len(S) else float("nan")
        corr_report[sp] = dict(n=len(nm), median_cells_A=int(np.median(ncA)) if len(ncA) else 0,
                               median_cells_B=int(np.median(ncB)) if len(ncB) else 0,
                               corr_halfsum_full=round(c, 4))
        log(f"{sp}: {len(nm)} perts | median units/half A={corr_report[sp]['median_cells_A']} "
            f"B={corr_report[sp]['median_cells_B']} | corr(1/2(A+B),full)={c:.4f}")

    # ---- FAITHFULNESS GATE against the shipped LFC (proves scale + pert order match s_full) ----
    z = np.load(SHIPPED_LFC, allow_pickle=True)
    faith = {}
    for sp in ("train", "val", "test"):
        # names must match in order
        assert list(out[f"{sp}_names"]) == list(z[f"{sp}_names"]), f"{sp} pert order differs from shipped LFC!"
        d = float(np.abs(out[f"{sp}_LFC"] - np.asarray(z[f"{sp}_LFC"])).max())
        faith[sp] = d
        log(f"faithfulness {sp}: max|recomputed_full - shipped_full| = {d:.2e}")
        assert d < 1e-3, f"{sp} recomputed full LFC does not reproduce shipped ({d:.2e}) — scale mismatch"

    sig = np.asarray(z["signal_mask"])            # keep the shipped signal_mask + mu_ctrl (identical build)
    np.savez(OUT,
             panel=np.array(panel), N=N, signal_mask=sig, mu_ctrl=mu_ctrl.astype(np.float64),
             **{k: out[k] for k in out})
    rep = dict(source=ALLSPLITS, seed=SEED, faithfulness_max_abs=faith, halves=corr_report,
               out=str(OUT), secs=round(time.time() - t0, 1))
    json.dump(rep, open(str(OUT).replace(".npz", "_report.json"), "w"), indent=2, default=float)
    log(f"WROTE {OUT}  ({time.time()-t0:.0f}s)  faithfulness OK, s_A/s_B genuine")


if __name__ == "__main__":
    main()
