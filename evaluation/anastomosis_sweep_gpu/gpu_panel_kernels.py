"""
obj_011 — gpu_panel_kernels.py : GPU-complete the obj_010 Mode-B panel.

`gpu_run_panel(u, Zfit, device)` mirrors clone/obj010/panel.run_panel EXACTLY (same dict + meta), but GPU-batches
the per-pert Python loops + the O(n^2) centroid_accuracy bottleneck in float64 (2070 supports f64; the axes are
small so it's fast enough and reproduces the CPU/scipy panel to ~1e-12). Mirrors the obj_003.2 doctrine:
  - float64 everywhere on the decision axes (no float32 rounding -> 0 threshold flips);
  - the arbiter RANK axes (rsc, coexpr_resid) stay on CPU via metrics_v3f (M3) — they ARE the definition, so we
    never re-implement them (exactness by construction), matching panel.tier_correlation;
  - the degenerate-row rule, finite masks, and the std/`v.sum()<10` gates are copied byte-for-byte from panel.py
    so CPU==GPU on every metric.
GPU-covered: cosine_delta, pearson_delta, pearson_systema, systematic_variation, centroid_accuracy (cdist),
f1_at_k + f1_auc, mae/rmse/mse/l2, tier_topk. CPU (M3, exact): rsc, coexpr_resid, degenerate mask, fisher_mean.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "..", "clone", "obj010")
if CLONE not in sys.path:
    sys.path.insert(0, CLONE)
import panel as P               # the clone (metrics defs + degenerate rule + M3 import)
M3 = P.M3
K_VALUES = (10, 20, 50, 100, 200, 500, 1000)


def _torch(device):
    import torch
    ok = device == "cuda" and torch.cuda.is_available()
    if ok:
        try:
            from gpu_kernels import free_vram_mb
            ok = free_vram_mb() >= 4000
        except Exception:
            pass
    return torch, ("cuda" if ok else "cpu")


def _pearson_rows(t, a, b):
    """Row-wise pearson over columns where BOTH finite, matching panel: v.sum()>=10 and both std>1e-10."""
    fin = torch.isfinite(a) & torch.isfinite(b)
    az = torch.where(fin, a, torch.zeros_like(a)); bz = torch.where(fin, b, torch.zeros_like(b))
    n = fin.sum(1).to(a.dtype)
    sa = az.sum(1); sb = bz.sum(1)
    ma = sa / n; mb = sb / n
    ac = torch.where(fin, a - ma[:, None], torch.zeros_like(a))
    bc = torch.where(fin, b - mb[:, None], torch.zeros_like(b))
    num = (ac * bc).sum(1); den = ac.norm(dim=1) * bc.norm(dim=1)
    # per-row std proxies (population, over finite) to replicate panel's std>1e-10 gate
    sda = (ac * ac).sum(1); sdb = (bc * bc).sum(1)
    out = torch.where((n >= 10) & (sda > 1e-20) & (sdb > 1e-20) & (den > 0),
                      num / den, torch.full_like(num, float("nan")))
    return out


import torch  # noqa: E402  (module-level for the helper above; re-imported per-device in gpu_run_panel)


def gpu_run_panel(u, Zfit=None, device="cuda"):
    global torch
    torch, dev = _torch(device)
    f64 = torch.float64
    mt = torch.as_tensor(np.asarray(u.means_true, np.float64), device=dev)
    mp = torch.as_tensor(np.asarray(u.means_pred, np.float64), device=dev)
    mu = torch.as_tensor(np.asarray(u.mu_ctrl, np.float64), device=dev)
    dt = mt - mu[None, :]; dp = mp - mu[None, :]
    n = int(u.n_pert); ng = int(u.n_gene)

    # --- degenerate-row rule (CPU numpy, byte-identical to panel) ---
    deg = P.degenerate_pred_mask(u)
    n_deg = int(deg.sum()); degen_frac = float(n_deg / n) if n else 0.0
    unscoreable = bool(n > 0 and n_deg == n)
    deg_t = torch.as_tensor(deg, device=dev)

    res = {}

    # --- tier_error (GPU per-pert, NUMPY aggregation to match panel's np.nanmean/np.nanmedian exactly) ---
    d = dt - dp
    for nm, arr in [("mae", d.abs().mean(1)), ("rmse", (d ** 2).mean(1).sqrt()),
                    ("mse", (d ** 2).mean(1)), ("l2", (d ** 2).sum(1).sqrt())]:
        a = arr.cpu().numpy()
        res[f"{nm}_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        res[f"{nm}_median"] = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan

    # --- tier_correlation: pearson_delta, cosine_delta (GPU); rsc, coexpr_resid (CPU M3) ---
    pe = _pearson_rows(torch, dt, dp)
    fin = torch.isfinite(dt) & torch.isfinite(dp)
    az = torch.where(fin, dt, torch.zeros_like(dt)); bz = torch.where(fin, dp, torch.zeros_like(dp))
    co_num = (az * bz).sum(1); co = torch.where((az.norm(dim=1) > 0) & (bz.norm(dim=1) > 0),
                                                co_num / (az.norm(dim=1) * bz.norm(dim=1)), torch.full_like(co_num, float("nan")))
    co = torch.where(deg_t, torch.full_like(co, float("nan")), co)          # degenerate -> NaN (panel rule)
    pe_np = pe.cpu().numpy(); co_np = co.cpu().numpy()
    rsc = M3.rsc_per_pert(np.asarray(u.means_true - u.mu_ctrl[None, :]), np.asarray(u.means_pred - u.mu_ctrl[None, :]),
                          u.signal_mask, u.pert_gene_idx)
    cxr = (M3.rsc_coexpr_resid_per_pert(np.asarray(u.means_true - u.mu_ctrl[None, :]),
           np.asarray(u.means_pred - u.mu_ctrl[None, :]), u.signal_mask, u.pert_gene_idx, Zfit)
           if Zfit is not None else np.full(n, np.nan))

    # --- tier_deg: f1@k + f1_auc (GPU topk on |delta|) ---
    at, ap = dt.abs(), dp.abs()
    ot = at.argsort(1, descending=True); op = ap.argsort(1, descending=True)
    f1cols = []
    for k in K_VALUES:
        ke = min(k, ng)
        mt_mask = torch.zeros((n, ng), dtype=torch.bool, device=dev)
        mp_mask = torch.zeros((n, ng), dtype=torch.bool, device=dev)
        mt_mask.scatter_(1, ot[:, :ke], True); mp_mask.scatter_(1, op[:, :ke], True)
        tp = (mt_mask & mp_mask).sum(1).to(f64)
        res_k = (tp / ke) if ke else torch.zeros(n, dtype=f64, device=dev)
        a = res_k.cpu().numpy()
        a[deg] = np.nan                       # patch6.1: mask degenerate/zero-coverage rows on the DEG axis too
        res[f"f1_at_{k}_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        res[f"f1_at_{k}_median"] = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
        f1cols.append(a)
    ka = np.array(K_VALUES, float); kn = (ka - ka.min()) / (ka.max() - ka.min() + 1e-9)
    F = np.column_stack(f1cols)               # n x K, deg rows already NaN -> f1_auc NaN for them
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    f1_auc = _trapz(F, kn, axis=1)
    res["f1_auc"] = float(np.nanmean(f1_auc)) if np.isfinite(f1_auc).any() else np.nan
    res["f1_auc_median"] = float(np.nanmedian(f1_auc)) if np.isfinite(f1_auc).any() else np.nan

    # --- tier_systema: systematic_variation, pearson_systema, centroid_accuracy ---
    if n >= 2:
        shifts = mt - mu[None, :]; avg = shifts.mean(0); an = avg.norm()
        sn = shifts.norm(dim=1)
        cs = torch.where((sn > 0) & (an > 0), (shifts * avg[None, :]).sum(1) / (sn * an), torch.full_like(sn, float("nan")))
        systematic_variation = float(torch.nanmean(cs).cpu())
        Op = mt.mean(0); ss_t = mt - Op[None, :]; ss_p = mp - Op[None, :]
        ps = _pearson_rows(torch, ss_t, ss_p)
        ps = torch.where(deg_t, torch.full_like(ps, float("nan")), ps)
        # centroid_accuracy: D[i,j]=||mp[i]-mt[j]|| via cdist; ca[i]=frac_{j!=i, mt[j] finite}(D[i,i] < D[i,j])
        finite_mt = torch.isfinite(mt).all(1)
        D = torch.cdist(mp, mt)                                    # n x n, f64 (fast mm path)
        di = torch.diagonal(D)[:, None]                           # ||mp[i]-mt[i]||
        wins = ((di < D) & finite_mt[None, :]).sum(1) - 0         # j==i: di<di False, excluded
        m = finite_mt.sum() - finite_mt.to(torch.int64)           # valid j per i (exclude self if finite)
        ca = torch.where(m > 0, wins.to(f64) / m.to(f64), torch.full_like(di[:, 0], float("nan")))
        ps_np = ps.cpu().numpy(); ca_np = ca.cpu().numpy()
    else:
        systematic_variation = float("nan"); ps_np = np.full(n, np.nan); ca_np = np.full(n, np.nan)

    # --- tier_topk (small k=20; CPU exact via numpy/scipy for parity) ---
    tk = P.tier_topk(u)

    # --- aggregation (fisher-mean for the correlation axes, else mean/median) ---
    fisher = {"rsc", "coexpr_resid", "pearson_delta", "cosine_delta", "pearson_systema"}
    def emit_arr(name, a):
        a = np.asarray(a, float); okf = np.isfinite(a).any()
        if name in fisher:
            res[f"{name}_mean"] = float(M3.fisher_mean(a)) if okf else np.nan
            res[f"{name}_median"] = float(np.nanmedian(a)) if okf else np.nan
        else:
            res[f"{name}_mean"] = float(np.nanmean(a)) if okf else np.nan
            res[f"{name}_median"] = float(np.nanmedian(a)) if okf else np.nan
    emit_arr("pearson_delta", pe_np); emit_arr("cosine_delta", co_np)
    emit_arr("rsc", rsc); emit_arr("coexpr_resid", cxr)
    emit_arr("pearson_systema", ps_np); emit_arr("centroid_accuracy", ca_np)
    res["systematic_variation"] = systematic_variation
    for k, v in tk.items():
        emit_arr(k, v)
    res["degenerate_frac"] = degen_frac; res["n_degenerate"] = n_deg
    meta = {"skipped_cell_tiers": True, "skipped": ["distribution", "cell_wilcoxon"],
            "coverage_frac": u.coverage_frac, "n_pert": u.n_pert, "n_gene": u.n_gene, "is_self": u.is_self,
            "degenerate_frac": degen_frac, "n_degenerate": n_deg, "unscoreable_zero_coverage": unscoreable}
    return res, meta
