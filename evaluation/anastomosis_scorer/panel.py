"""
obj_010 ANASTOMOSIS — metrics.py : the metric panel (PerturBench taxonomy), pure functions on a ScoredUnit.

Mean/rank/Systema tiers ported verbatim from anastomosis_proto (CPU numpy). The RANK axis (RSC) and the
co-expression-residualized axis are IMPORTED from obj_009.3 metrics_v3f (unchanged) — so obj_010's rank axis
IS the arbiter's operator (faithfulness by construction). Cell-only tiers raise CellTierUnavailable; the scorer
catches it and records NaN + skipped_cell_tiers. GPU tiers (energy-distance, GPU MW/Wass) are added in
gpu_kernels.py during the GPU window; the mean tiers here are device-agnostic CPU (bit-identical on GPU later).
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import stats as sstats

# import the arbiter's rank + coexpr axes UNCHANGED (import, don't fork). exp_031 patch3: prefer an already-on-path
# metrics_v3f (the HPC bundle / gauntlet puts obj_009.3 on sys.path); only fall back to a local tree if that fails
# (the hardcoded Windows path was a Linux/A100 deployment blocker — OBJ093_DIR overrides it).
try:
    import metrics_v3f as M3   # rsc_per_pert, fisher_mean, build_coexpr_Z, rsc_coexpr_resid_per_pert
except ImportError:
    _OBJ093 = os.environ.get("OBJ093_DIR", "c:/Users/studi/OneDrive/Documents/thesis/OBJECTS/obj_009.3_rhizo_final")
    for _p in (f"{_OBJ093}/src", f"{_OBJ093}/clone"):
        if _p not in sys.path: sys.path.insert(0, _p)
    import metrics_v3f as M3


class CellTierUnavailable(Exception):
    """Raised by a cell-only metric when a ScoredUnit has no cells (Mode B)."""


PERTURBENCH = {  # metric_group -> needs_cells
    "error": False, "correlation": False, "deg_recovery": False,
    "reference_insensitive": False, "systema": False, "distribution": True, "cell_wilcoxon": True,
}


def _delta(u):
    return u.means_true - u.mu_ctrl[None, :], u.means_pred - u.mu_ctrl[None, :]


# ── degenerate-row rule (exp_031 patch3) ──────────────────────────────────────
# A constant / near-constant PREDICTION delta row carries no usable structure — e.g. a ψ(0)=0 zero-coverage arm
# (fungi_bio on zeroshot) predicts an all-equal delta for every held-out perturbation. Its cosine_delta and
# pearson_systema are then MEANINGLESS (cosine of a constant vector is ill-defined; pearson_systema would report
# a spurious correlation with -mu_ctrl / the true centroid). Such a row is reported as NaN and EXCLUDED from the
# (Fisher-)nanmean; the fraction of degenerate rows is `degenerate_frac`; an arm whose rows are ALL degenerate is
# tagged `unscoreable_zero_coverage` — NEVER a fabricated 0. Decided rule: std < DEGEN_STD OR fewer than
# DEGEN_MIN_UNIQUE distinct finite values in the predicted delta row.
DEGEN_STD = 1e-8
DEGEN_MIN_UNIQUE = 3


def degenerate_pred_mask(u):
    """Bool mask over perturbations whose PREDICTED delta row (means_pred - mu_ctrl) is degenerate."""
    _, dp = _delta(u)
    n = dp.shape[0]
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        row = dp[i]; row = row[np.isfinite(row)]
        if row.size == 0 or float(row.std()) < DEGEN_STD or np.unique(row).size < DEGEN_MIN_UNIQUE:
            mask[i] = True
    return mask


# ── error ───────────────────────────────────────────────────────────────────
def tier_error(u):
    dt, dp = _delta(u); d = dt - dp
    return {"mae": np.nanmean(np.abs(d), axis=1), "rmse": np.sqrt(np.nanmean(d ** 2, axis=1)),
            "mse": np.nanmean(d ** 2, axis=1), "l2": np.sqrt(np.nansum(d ** 2, axis=1))}


# ── correlation (delta pearson/spearman/cosine + arbiter RSC + coexpr_resid) ──
def tier_correlation(u, Zfit=None, deg=None):
    dt, dp = _delta(u); n = u.n_pert
    pe = np.full(n, np.nan); co = np.full(n, np.nan)
    for i in range(n):
        a, b = dt[i], dp[i]; v = np.isfinite(a) & np.isfinite(b)
        if v.sum() < 10: continue
        if a[v].std() > 1e-10 and b[v].std() > 1e-10:
            pe[i] = sstats.pearsonr(a[v], b[v])[0]
        na, nb = np.linalg.norm(a[v]), np.linalg.norm(b[v])
        if na > 0 and nb > 0: co[i] = np.dot(a[v], b[v]) / (na * nb)
    if deg is not None:                        # exp_031 patch3: degenerate prediction rows -> cosine_delta NaN
        co[np.asarray(deg, bool)] = np.nan
    # RANK axis == arbiter RSC (spearman over signal genes, per pert): imported verbatim
    rsc = M3.rsc_per_pert(dt, dp, u.signal_mask, u.pert_gene_idx)
    cxr = M3.rsc_coexpr_resid_per_pert(dt, dp, u.signal_mask, u.pert_gene_idx, Zfit) if Zfit is not None \
        else np.full(n, np.nan)
    return {"pearson_delta": pe, "cosine_delta": co, "rsc": rsc, "coexpr_resid": cxr}


# ── DEG recovery (rank overlap F1@k sweep + f1_auc) ───────────────────────────
def tier_deg(u, k_values=(10, 20, 50, 100, 200, 500, 1000)):
    dt, dp = np.abs(_delta(u)[0]), np.abs(_delta(u)[1]); n, ng = dt.shape
    st, sp = np.argsort(-dt, axis=1), np.argsort(-dp, axis=1)
    out = {}; f1cols = []
    for k in k_values:
        ke = min(k, ng); f1 = np.full(n, np.nan)
        for i in range(n):
            tp = len(set(st[i, :ke]) & set(sp[i, :ke])); f1[i] = tp / ke if ke else 0.0
        out[f"f1_at_{k}"] = f1; f1cols.append(f1)
    ka = np.array(k_values, float); kn = (ka - ka.min()) / (ka.max() - ka.min() + 1e-9)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np.trapz removed in NumPy 2.0
    out["f1_auc"] = _trapz(np.column_stack(f1cols), kn, axis=1)
    return out


# ── reference-insensitive (top-20 true DEGs) ──────────────────────────────────
def tier_topk(u, k=20):
    dt, dp = _delta(u); n = u.n_pert
    pk = np.full(n, np.nan); sk = np.full(n, np.nan); rk = np.full(n, np.nan)
    for i in range(n):
        a, b = dt[i], dp[i]; v = np.isfinite(a) & np.isfinite(b)
        if v.sum() < k: continue
        idx = np.argpartition(np.abs(a), -k)[-k:]; ak, bk = a[idx], b[idx]
        if ak.std() < 1e-10 or bk.std() < 1e-10: continue
        pk[i] = sstats.pearsonr(ak, bk)[0]; sk[i] = sstats.spearmanr(ak, bk)[0]
        rk[i] = np.sqrt(np.mean((ak - bk) ** 2))
    return {f"pearson_{k}": pk, f"spearman_{k}": sk, f"rmse_{k}": rk}


# ── Systema (systematic variation + perturbed-centroid pearson + centroid acc) ─
def tier_systema(u, deg=None):
    mt, mp, c = u.means_true, u.means_pred, u.mu_ctrl; n = mt.shape[0]
    if n < 2:
        return {"systematic_variation": np.nan, "pearson_systema": np.full(n, np.nan),
                "centroid_accuracy": np.full(n, np.nan)}
    shifts = mt - c[None, :]; avg = shifts.mean(0); an = np.linalg.norm(avg)
    cs = np.full(n, np.nan)
    for i in range(n):
        sn = np.linalg.norm(shifts[i])
        if sn > 0 and an > 0: cs[i] = np.dot(shifts[i], avg) / (sn * an)
    Op = mt.mean(0); ss_t = mt - Op[None, :]; ss_p = mp - Op[None, :]
    ps = np.full(n, np.nan)
    for i in range(n):
        a, b = ss_t[i], ss_p[i]; v = np.isfinite(a) & np.isfinite(b)
        if v.sum() < 10 or a[v].std() < 1e-10 or b[v].std() < 1e-10: continue
        ps[i] = sstats.pearsonr(a[v], b[v])[0]
    if deg is not None:                        # exp_031 patch3: degenerate prediction rows -> pearson_systema NaN
        ps[np.asarray(deg, bool)] = np.nan
    ca = np.full(n, np.nan)
    for i in range(n):
        di = np.linalg.norm(mp[i] - mt[i]); wins = 0; m = 0
        for j in range(n):
            if j == i or not np.isfinite(mt[j]).all(): continue
            wins += (di < np.linalg.norm(mp[i] - mt[j])); m += 1
        if m: ca[i] = wins / m
    return {"systematic_variation": float(np.nanmean(cs)), "pearson_systema": ps, "centroid_accuracy": ca}


# ── cell-only tiers (Mode A) — raise in Mode B ────────────────────────────────
def tier_distribution(u, device="cpu"):
    if not u.has_cells: raise CellTierUnavailable("energy_distance/MMD need per-cell data (Mode A)")
    from gpu_kernels import energy_distance  # implemented in the GPU window
    return energy_distance(u, device=device)


def tier_cell_wilcoxon(u, device="cpu"):
    if not u.has_cells: raise CellTierUnavailable("wilcoxon AUPRC needs per-cell data (Mode A)")
    from gpu_kernels import wilcoxon_auprc
    return wilcoxon_auprc(u, device=device)


# ── aggregation ───────────────────────────────────────────────────────────────
def _agg(name, arr, out):
    a = np.asarray(arr, float)
    if a.ndim == 0: out[name] = float(a); return
    out[f"{name}_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    out[f"{name}_median"] = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def run_panel(u, Zfit=None, run_cell=None, device="cpu"):
    """Full panel on a ScoredUnit. run_cell None => auto (per_cell only). Returns (metrics_dict, meta)."""
    run_cell = (u.has_cells) if run_cell is None else run_cell
    res = {}; skipped = []
    # exp_031 patch3: degenerate-row rule — compute once, thread into the correlation + systema tiers so their
    # degenerate rows become NaN (excluded from the Fisher-nanmean below), and surface degenerate_frac + the
    # unscoreable tag. NEVER fabricate a 0 for an all-degenerate (zero-coverage) arm.
    deg = degenerate_pred_mask(u)
    n_pert = int(u.n_pert); n_deg = int(deg.sum())
    degenerate_frac = float(n_deg / n_pert) if n_pert else 0.0
    unscoreable = bool(n_pert > 0 and n_deg == n_pert)
    # aggregate the rank correlations with Fisher-mean (== the arbiter); others plain mean/median
    fisher = {"rsc", "coexpr_resid", "pearson_delta", "cosine_delta", "pearson_systema"}
    def emit(d):
        for k, v in d.items():
            if np.ndim(v) == 0: res[k] = v; continue
            a = np.asarray(v, float); ok = np.isfinite(a).any()
            if k in fisher:  # correlation axes aggregated with Fisher-mean == the arbiter RSC
                res[f"{k}_mean"] = float(M3.fisher_mean(a)) if ok else np.nan
                res[f"{k}_median"] = float(np.nanmedian(a)) if ok else np.nan
            else: _agg(k, a, res)
    emit(tier_error(u)); emit(tier_correlation(u, Zfit, deg=deg)); emit(tier_deg(u)); emit(tier_topk(u))
    emit(tier_systema(u, deg=deg))
    res["degenerate_frac"] = degenerate_frac; res["n_degenerate"] = n_deg
    for tier, fn in [("distribution", tier_distribution), ("cell_wilcoxon", tier_cell_wilcoxon)]:
        if run_cell:
            try: emit(fn(u, device=device))
            except CellTierUnavailable: skipped.append(tier)
        else: skipped.append(tier)
    meta = {"skipped_cell_tiers": bool(skipped), "skipped": skipped, "coverage_frac": u.coverage_frac,
            "n_pert": u.n_pert, "n_gene": u.n_gene, "is_self": u.is_self,
            "degenerate_frac": degenerate_frac, "n_degenerate": n_deg,
            "unscoreable_zero_coverage": unscoreable}
    return res, meta
