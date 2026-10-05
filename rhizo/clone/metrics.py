"""
obj_008 — metrics.py

The scoreboard. Every primary/secondary metric scores the mean baseline at <= 0 by construction, so a
"constant mean-prediction" run (predict s_p = 0 for all genes) must score <= 0 on every test perturbation
(verification check #1). Targets/predictions are the SPECIFIC RESIDUAL s_p = Delta_p - mu_bar.

PRIMARY
  RSC  (Residual-Specific Correlation)  — Pearson(s_p, s_hat_p) over train-signal genes S, excluding the
        perturbed gene; Fisher-averaged over test perts. Mean baseline (s_hat=0) => RSC = 0.
CO-PRIMARY (for power)
  DNSA (Directed-Neighbor Sign Accuracy) — over the perturbed gene's OUT-neighbors in the graph under test
        (intersect S, exclude self): fraction with sign(s_hat)==sign(s_true), rescaled random->0 (2*acc-1).
        Reported per-pert AND per-pair (thousands of pairs => power at n~304). Mean baseline => -1.
SECONDARY (report, do not select)
  weighted-R2 over S (predicting-zero baseline => 0), top-k specific-DEG sign accuracy, AUPRC over specific DEGs.

SIGNIFICANCE: paired bootstrap over test perts (B, percentile CI + one-sided p) + Wilcoxon signed-rank.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon


# ------------------------------------------------------------------ helpers
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


def fisher_mean(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


# ------------------------------------------------------------------ PRIMARY: RSC
def rsc_per_pert(S_true, S_pred, signal_mask, pert_gene_idx):
    """Per-pert Pearson over signal genes S, excluding the perturbed gene. Returns array[n_pert]."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        cols = sig[sig != pert_gene_idx[i]] if pert_gene_idx[i] >= 0 else sig
        out[i] = _pearson(S_true[i, cols], S_pred[i, cols])
    return out


# ------------------------------------------------------------------ CO-PRIMARY: DNSA
def dnsa(S_true, S_pred, signal_mask, pert_gene_idx, A_csc, hops2=False, A_csc2=None):
    """
    A_csc: CSC of the RAW adjacency (A[t,s]=w(s->t)); column s => out-neighbors (targets) of source s.
    Returns dict: per_pert (array[n_pert], rescaled 2*acc-1), pair_acc, dnsa_pair (2*pair_acc-1), n_pairs.
    """
    n = S_true.shape[0]
    per_pert = np.full(n, np.nan)
    tot_match = 0
    tot = 0
    for i in range(n):
        g = pert_gene_idx[i]
        if g < 0:
            continue
        tg = A_csc.indices[A_csc.indptr[g]:A_csc.indptr[g + 1]]
        if hops2 and A_csc2 is not None:
            tg2 = A_csc2.indices[A_csc2.indptr[g]:A_csc2.indptr[g + 1]]
            tg = np.unique(np.concatenate([tg, tg2]))
        tg = tg[signal_mask[tg]]
        tg = tg[tg != g]
        if len(tg) == 0:
            continue
        st = np.sign(S_true[i, tg])
        sp_ = np.sign(S_pred[i, tg])
        valid = st != 0
        if valid.sum() == 0:
            continue
        match = (st[valid] == sp_[valid])
        per_pert[i] = 2.0 * match.mean() - 1.0
        tot_match += int(match.sum())
        tot += int(valid.sum())
    pair_acc = tot_match / tot if tot > 0 else float("nan")
    return dict(per_pert=per_pert, pair_acc=pair_acc,
                dnsa_pair=(2.0 * pair_acc - 1.0) if tot > 0 else float("nan"), n_pairs=tot)


# ------------------------------------------------------------------ SECONDARY
def weighted_r2_per_pert(S_true, S_pred, signal_mask, pert_gene_idx, weights=None):
    """R2 of the specific residual over S with the predicting-zero baseline (mean baseline => 0)."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    w_full = weights if weights is not None else np.ones(S_true.shape[1])
    for i in range(n):
        cols = sig[sig != pert_gene_idx[i]] if pert_gene_idx[i] >= 0 else sig
        y = S_true[i, cols]; yh = S_pred[i, cols]; w = w_full[cols]
        denom = float((w * y * y).sum())
        if denom < 1e-12:
            continue
        num = float((w * (y - yh) ** 2).sum())
        out[i] = 1.0 - num / denom
    return out


def topk_sign_acc(S_true, S_pred, signal_mask, pert_gene_idx, k=20):
    """Sign accuracy on each pert's true top-k specific DEGs (by |s_true|) over S. Random ~ 0.5."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        cols = sig[sig != pert_gene_idx[i]] if pert_gene_idx[i] >= 0 else sig
        if len(cols) < k:
            kk = len(cols)
        else:
            kk = k
        if kk == 0:
            continue
        y = S_true[i, cols]
        top = cols[np.argsort(np.abs(y))[::-1][:kk]]
        st = np.sign(S_true[i, top]); sp_ = np.sign(S_pred[i, top])
        valid = st != 0
        if valid.sum() == 0:
            continue
        out[i] = float((st[valid] == sp_[valid]).mean())
    return out


def auprc_per_pert(S_true, S_pred, signal_mask, pert_gene_idx, k=50):
    """AUPRC ranking |s_hat| against the true top-k specific-DEG set over S. Baseline = prevalence."""
    from sklearn.metrics import average_precision_score
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        cols = sig[sig != pert_gene_idx[i]] if pert_gene_idx[i] >= 0 else sig
        if len(cols) <= k:
            continue
        y = np.abs(S_true[i, cols])
        thresh = np.sort(y)[::-1][k - 1]
        ytrue = (y >= thresh).astype(int)
        score = np.abs(S_pred[i, cols])
        if ytrue.sum() == 0 or ytrue.sum() == len(ytrue) or score.std() < 1e-12:
            continue
        out[i] = float(average_precision_score(ytrue, score))
    return out


# ------------------------------------------------------------------ SIGNIFICANCE
def paired_bootstrap(a, b, n_boot=10000, seed=42):
    """Paired FUNGI(a) - baseline(b) over perts. One-sided p = Pr(mean diff <= 0). Returns dict."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    n = len(d)
    if n == 0:
        return dict(n=0, mean_diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    p_one_sided=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    return dict(n=int(n), mean_diff=float(d.mean()),
                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                p_one_sided=float(np.mean(boot <= 0.0)))


def one_sample_bootstrap(a, n_boot=10000, seed=42):
    """CI + one-sided p (Pr(mean <= 0)) for a single metric vector (e.g. FUNGI RSC > 0)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return dict(n=0, mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p_one_sided=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1)
    return dict(n=int(n), mean=float(a.mean()),
                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                p_one_sided=float(np.mean(boot <= 0.0)))


def wilcoxon_paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    d = d[d != 0]
    if len(d) < 5:
        return dict(stat=float("nan"), p=float("nan"), n=int(len(d)))
    try:
        s, p = wilcoxon(d, alternative="greater")
        return dict(stat=float(s), p=float(p), n=int(len(d)))
    except Exception as e:
        return dict(stat=float("nan"), p=float("nan"), n=int(len(d)), err=str(e))


# ------------------------------------------------------------------ mean-baseline verification (check #1)
def mean_baseline_scores(S_true, signal_mask, pert_gene_idx):
    """Score a constant mean-prediction (s_hat = 0). Returns per-pert RSC, weighted-R2, DNSA-free scalars.
    RSC and weighted-R2 => 0; DNSA (needs a graph) computed by caller. All must be <= 0."""
    S_pred = np.zeros_like(S_true)
    rsc = rsc_per_pert(S_true, S_pred, signal_mask, pert_gene_idx)
    wr2 = weighted_r2_per_pert(S_true, S_pred, signal_mask, pert_gene_idx)
    return dict(rsc=rsc, wr2=wr2)
