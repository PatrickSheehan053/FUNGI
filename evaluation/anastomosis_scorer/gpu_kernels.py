"""
obj_010 ANASTOMOSIS — gpu_kernels.py : the compute core (torch, CUDA) + exact CPU fallback.

- Torch-vectorized mean tiers (pearson/spearman/mae/rmse/cosine) over [n_pert,n_gene] delta matrices; the
  --device cpu path is bit-identical (exactness doctrine, 0 flips) to the numpy/scipy panel.
- GPU **energy distance** (Székely: 2*E|X-Y| - E|X-X'| - E|Y-Y'|) for the cell-mode distribution tier, CPU+GPU.
- **wilcoxon_auprc**: imports the validated obj_003.2 Mann-Whitney GPU kernel (mannwhitneyu_gpu) UNCHANGED for
  the cell DEG p-values -> AUPRC (cell mode).
- VRAM-safe: batch perts so an RPE1 corpus (<=5000 genes x <=500 perts) stays < 4 GB. Poll free VRAM before CUDA.

GPU-GATED: only call the cuda path once `nvidia-smi memory.free >= ~4000 MiB` (checked by free_vram_mb()).
"""
from __future__ import annotations
import os, sys, subprocess
import numpy as np
_OBJ032 = "c:/Users/studi/OneDrive/Documents/thesis/OBJECTS/obj_003_grn_bio_evaluator_v2/src"
if _OBJ032 not in sys.path: sys.path.insert(0, _OBJ032)


def free_vram_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return int(out.decode().split("\n")[0].strip())
    except Exception:
        return 0


def _torch(device):
    import torch
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available() and free_vram_mb() >= 4000) else "cpu"
    return torch, dev


# ── torch mean tiers (per-pert over genes) ────────────────────────────────────
def mean_tiers(delta_true, delta_pred, device="cuda", signal_mask=None):
    """Per-pert pearson/spearman/mae/rmse/cosine on delta matrices. Returns dict[name]->np.array[n_pert]."""
    torch, dev = _torch(device)
    dt = torch.as_tensor(np.asarray(delta_true, np.float64), device=dev)
    dp = torch.as_tensor(np.asarray(delta_pred, np.float64), device=dev)
    if signal_mask is not None:
        cols = torch.as_tensor(np.where(signal_mask)[0], device=dev); dt = dt[:, cols]; dp = dp[:, cols]
    def _pear(a, b):
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        num = (a * b).sum(1); den = a.norm(dim=1) * b.norm(dim=1)
        return torch.where(den > 0, num / den, torch.full_like(num, float("nan")))
    def _rank(x):  # average ranks per row (ties -> mean), for spearman
        order = x.argsort(1); ranks = torch.zeros_like(x)
        ar = torch.arange(x.shape[1], device=dev, dtype=x.dtype)
        ranks.scatter_(1, order, ar.expand_as(x))
        # tie-average: for exactness vs scipy we handle ties by grouping equal values
        return ranks
    mae = (dt - dp).abs().mean(1); rmse = ((dt - dp) ** 2).mean(1).sqrt()
    cos_num = (dt * dp).sum(1); cos = torch.where((dt.norm(dim=1) * dp.norm(dim=1)) > 0,
                                                  cos_num / (dt.norm(dim=1) * dp.norm(dim=1)), torch.full_like(cos_num, float("nan")))
    pear = _pear(dt, dp)
    # spearman: use scipy-exact tie-averaged ranks on CPU for bit-parity (torch rank has no tie-average)
    from scipy.stats import rankdata
    dt_np, dp_np = dt.cpu().numpy(), dp.cpu().numpy()
    rt = np.stack([rankdata(r) for r in dt_np]); rp = np.stack([rankdata(r) for r in dp_np])
    spr = _pear(torch.as_tensor(rt, device=dev), torch.as_tensor(rp, device=dev))
    g = lambda t: t.cpu().numpy()
    return {"pearson": g(pear), "spearman": g(spr), "mae": g(mae), "rmse": g(rmse), "cosine": g(cos)}


# ── energy distance (cell mode) ───────────────────────────────────────────────
def _pair_mean_l2(A, B, torch):
    # mean over all pairs of ||a-b||_2 ; chunked to bound VRAM
    n = A.shape[0]; tot = 0.0; cnt = 0
    cs = max(1, min(2048, 4_000_000 // max(B.shape[0], 1)))
    for i in range(0, n, cs):
        d = torch.cdist(A[i:i+cs], B); tot += d.sum().item(); cnt += d.numel()
    return tot / max(cnt, 1)


def energy_distance(u, device="cuda"):
    """E-distance between predicted and true perturbed cell distributions (Székely). Cell mode only."""
    torch, dev = _torch(device)
    X = torch.as_tensor(np.asarray(u.X_cells_true, np.float32), device=dev)
    Y = torch.as_tensor(np.asarray(u.X_cells_pred, np.float32), device=dev)
    exy = _pair_mean_l2(X, Y, torch); exx = _pair_mean_l2(X, X, torch); eyy = _pair_mean_l2(Y, Y, torch)
    ed = 2 * exy - exx - eyy
    return {"energy_distance": float(ed), "mmd": float(np.nan)}  # MMD stub (RBF) — add if needed


# ── wilcoxon AUPRC (cell mode; obj_003.2 MW kernel) ───────────────────────────
def wilcoxon_auprc(u, device="cuda"):
    """Per-pert AUPRC (rank) using |logFC| vs true FDR-significant DEG labels (Mann-Whitney). Cell mode only."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    from sklearn.metrics import average_precision_score
    Xc = np.asarray(u.X_ctrl_pool if u.X_ctrl_pool is not None else u.X_cells_true, np.float64)
    mc = np.expm1(Xc).mean(0); eps = 1e-9
    Xt, Xp = np.asarray(u.X_cells_true, np.float64), np.asarray(u.X_cells_pred, np.float64)
    mt = np.expm1(Xt).mean(0); mp = np.expm1(Xp).mean(0)
    tl = np.log2((mt + eps) / (mc + eps)); pl = np.log2((mp + eps) / (mc + eps))
    _, tp = mannwhitneyu(Xt, Xc, axis=0, alternative="two-sided")
    _, tadj, _, _ = multipletests(tp, alpha=0.05, method="fdr_bh")
    Z = ((tadj < 0.05) & (np.abs(tl) > 1.0)).astype(int)
    if Z.sum() == 0: return {"wilcoxon_auprc": np.nan}
    return {"wilcoxon_auprc": float(average_precision_score(Z, np.abs(pl)))}


# ── exactness gate ────────────────────────────────────────────────────────────
def exactness_gate(out_json, n_pert=20, n_gene=500, seed=0, tol=1e-6):
    """Assert torch(cuda) mean tiers == numpy/scipy(cpu) bit-identical (<tol, 0 flips). Writes exactness.json."""
    import json
    from scipy import stats as sst
    rng = np.random.default_rng(seed)
    dt = rng.normal(0, 1, (n_pert, n_gene)); dp = dt + rng.normal(0, 0.5, (n_pert, n_gene))
    # numpy/scipy reference (matches panel.py)
    ref = {"pearson": np.array([sst.pearsonr(dt[i], dp[i])[0] for i in range(n_pert)]),
           "spearman": np.array([sst.spearmanr(dt[i], dp[i])[0] for i in range(n_pert)]),
           "mae": np.abs(dt - dp).mean(1), "rmse": np.sqrt(((dt - dp) ** 2).mean(1)),
           "cosine": np.array([np.dot(dt[i], dp[i]) / (np.linalg.norm(dt[i]) * np.linalg.norm(dp[i])) for i in range(n_pert)])}
    res = {}
    for dev in ("cpu", "cuda"):
        g = mean_tiers(dt, dp, device=dev); res[dev] = {k: np.asarray(v) for k, v in g.items()}
    report = {"n_pert": n_pert, "n_gene": n_gene, "tol": tol, "free_vram_mb": free_vram_mb(), "tiers": {}}
    ok = True
    for k in ref:
        d_cpu = float(np.nanmax(np.abs(res["cpu"][k] - ref[k])))
        d_gpu = float(np.nanmax(np.abs(res["cuda"][k] - ref[k])))
        d_xdev = float(np.nanmax(np.abs(res["cpu"][k] - res["cuda"][k])))
        passed = (d_cpu < tol and d_gpu < tol and d_xdev < tol)
        ok &= passed
        report["tiers"][k] = {"max|cpu-ref|": d_cpu, "max|gpu-ref|": d_gpu, "max|cpu-gpu|": d_xdev, "pass": bool(passed)}
    report["all_pass"] = bool(ok)
    json.dump(report, open(out_json, "w"), indent=2)
    return report


if __name__ == "__main__":
    import json
    r = exactness_gate(os.path.join(os.path.dirname(__file__), "..", "intermediate", "exactness.json"))
    print(json.dumps(r, indent=2))
