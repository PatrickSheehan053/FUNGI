"""
obj_009 — prep_data.py : build the DNPN data cache from exp_024 File A (the same substrate as exp_024/Report_028).

Reuses the exp_024 File-A eval cache (μ_ctrl, μ̄ [train-only], the train-only signal mask S, panel genes,
per-perturbation FULL pseudobulk residual s_full) and adds the RUNG-1 CELL-HELD-OUT split: each perturbation's
cells are split 50/50 (seeded), pseudobulk each half → s_A, s_B (residuals vs the SAME μ̄). Leakage firewall:
File A is train-only; μ̄ and S are train-only; only gene identities ever cross.

Rungs:
  Rung 1 (cell-held-out): train on s_A, evaluate vs s_B (and symmetric) — cell-sampling robustness, all perts
    in both train and test. VALID because DNPN is seed-only / no node features / no per-pert params / ψ(0)=0.
  Rung 2 (perturbation-held-out CV): 2-fold over perturbations on s_full — unseen-perturbation generalization.

Output -> intermediate/obj_009_data.npz. CPU-only.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP24 = os.path.join(ROOT, "DATA", "EXPERIMENTS", "exp_024_multisubstrate_pruning_and_obj008_synthetic")
ADATA = os.path.join(EXP24, "intermediate", "substrate", "train_hybrid_oldpanel.h5ad")
EVAL_CACHE = os.path.join(EXP24, "intermediate", "eval_cache", "pseudobulk_fileA.npz")
OUT = os.path.join(HERE, "..", "intermediate", "obj_009_data.npz")
CTRL = "non-targeting"


def log(m): print(f"[prep_data {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pb = np.load(EVAL_CACHE, allow_pickle=True)
    panel = [str(g) for g in pb["panel_genes"]]; g2i = {g: i for i, g in enumerate(panel)}
    mu_ctrl = pb["mu_ctrl"]; mu_bar = pb["mu_bar"]; signal_mask = pb["signal_mask"].astype(bool)
    train_perts = [str(p) for p in pb["train_perts"]]
    s_full = pb["s_train"]; pidx = pb["pidx_train"].astype(np.int64)
    N = len(panel); n_pert = len(train_perts)
    log(f"panel={N} perts={n_pert} signal_genes={int(signal_mask.sum())}")

    import anndata as ad
    log("reading File A adata (cells) for the cell-held-out split ...")
    A = ad.read_h5ad(ADATA)
    X = A.X; X = np.asarray(X.todense(), np.float32) if sp.issparse(X) else np.asarray(X, np.float32)
    gv = A.obs["gene"].astype(str).values

    rng = np.random.default_rng(42)
    s_A = np.zeros((n_pert, N), np.float64); s_B = np.zeros((n_pert, N), np.float64)
    ncA = np.zeros(n_pert, np.int64); ncB = np.zeros(n_pert, np.int64)
    for i, p in enumerate(train_perts):
        cells = np.where(gv == p)[0]
        rng.shuffle(cells)
        half = len(cells) // 2
        ca, cb = cells[:half], cells[half:]
        ncA[i] = len(ca); ncB[i] = len(cb)
        s_A[i] = X[ca].mean(0).astype(np.float64) - mu_ctrl - mu_bar
        s_B[i] = X[cb].mean(0).astype(np.float64) - mu_ctrl - mu_bar
    log(f"cell-split done: median cells/half A={int(np.median(ncA))} B={int(np.median(ncB))}")

    # reproducibility check: mean of the two half-pseudobulks ≈ the full pseudobulk residual (weighted)
    approx = 0.5 * (s_A + s_B)
    corr_halfsum_full = float(np.corrcoef(approx.ravel(), s_full.ravel())[0, 1])
    log(f"corr(½(s_A+s_B), s_full) = {corr_halfsum_full:.4f}")

    np.savez_compressed(
        OUT, panel_genes=np.array(panel, dtype=object), mu_ctrl=mu_ctrl, mu_bar=mu_bar,
        signal_mask=signal_mask, train_perts=np.array(train_perts, dtype=object), pidx=pidx,
        s_full=s_full, s_A=s_A, s_B=s_B, ncA=ncA, ncB=ncB,
        var_real=pb["var_real"] if "var_real" in pb else np.zeros(N))
    json.dump(dict(N=N, n_pert=n_pert, n_signal=int(signal_mask.sum()),
                   n_fungi_source=int((pidx >= 0).sum()), corr_halfsum_full=corr_halfsum_full,
                   median_cells_halfA=int(np.median(ncA))),
              open(os.path.join(HERE, "..", "intermediate", "obj_009_data_summary.json"), "w"), indent=2)
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
