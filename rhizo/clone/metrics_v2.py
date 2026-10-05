"""
obj_009.1 — metrics_v2.py : the v1 obj_008 scoreboard (RSC / DNSA / wr2 / bootstrap) RE-EXPORTED byte-identical
(imported from clone/metrics.py so P1 is a true 1-to-1 reproduction) PLUS the P2 topology-isolating metrics
that isolate the multi-hop / long-range / directional regime where topology is supposed to win:

  dsRSC_ge2 / dsRSC_ge3   RSC computed ONLY over signal genes at directed BFS distance ≥2 (resp ≥3) AND ≤cap
                          hops from the perturbed source (i.e. the genuinely multi-hop-reachable set on THIS
                          arm's graph). Greedy's fragmentation starves the multi-hop set -> lower dsRSC AND
                          fewer scorable genes (n reported alongside). Mean baseline (pred 0) -> 0.
  reach_at_K              purely-structural: fraction of a pert's true top-20 specific DEGs reachable within K
                          directed hops of the perturbed node on this arm's graph (no prediction involved).
  eff_resistance         mean effective resistance (undirected weighted Laplacian pseudo-inverse) from the
                          perturbed node to its true top-20 DEGs; lower = better multi-path connectivity. NaN
                          DEGs (different component) are excluded and their fraction tracked. Structural.
  DNSA_ge2hop            directed-neighbor sign accuracy extended from v1's 1-hop out-neighbors to genes at
                          distance 2..K (multi-hop). Rescaled 2·acc−1; mean baseline -> undefined/negative.

All P2 metrics take a per-arm directed-BFS distance cache (dist[row, :] for the perturbed source's row) built
CPU-side by graph_caches.py. The GAP is reported on every P2 metric by the ablation driver.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clone"))
from metrics import (  # noqa: F401  (re-export v1 metrics byte-identical)
    rsc_per_pert, dnsa, weighted_r2_per_pert, topk_sign_acc, auprc_per_pert,
    paired_bootstrap, one_sample_bootstrap, wilcoxon_paired, fisher_mean, _pearson,
)

UNREACH = 999  # distance sentinel for "beyond cap"


# ---------------------------------------------------------------- helpers
def _dist_row(dist, src_row, g):
    """distances (N,) from perturbed gene g on this arm; None if g is not a cached source."""
    r = src_row.get(int(g), -1)
    return dist[r] if r >= 0 else None


def true_top_degs(s_true_i, signal_mask, pert_gene, k=20):
    """This pert's true top-k specific DEGs (by |s_true|) over signal genes, excluding the perturbed gene."""
    sig = np.where(signal_mask)[0]
    cols = sig[sig != pert_gene] if pert_gene >= 0 else sig
    if len(cols) == 0:
        return np.array([], np.int64)
    kk = min(k, len(cols))
    order = np.argsort(np.abs(s_true_i[cols]))[::-1][:kk]
    return cols[order]


# ---------------------------------------------------------------- dsRSC (distance-stratified RSC)
def dsrsc_per_pert(S_true, S_pred, signal_mask, pert_gene_idx, dist, src_row, min_hops, cap=6):
    """RSC over signal genes with min_hops <= directed_dist(g,.) <= cap, excluding the perturbed gene.
    Returns (rsc array[n_pert], n array[n_pert]). n<3 -> nan (too few scorable multi-hop genes)."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    out = np.full(n, np.nan); ncols = np.zeros(n, np.int64)
    for i in range(n):
        g = pert_gene_idx[i]
        drow = _dist_row(dist, src_row, g) if g >= 0 else None
        if drow is None:
            continue
        cand = sig[sig != g]
        dd = drow[cand]
        cols = cand[(dd >= min_hops) & (dd <= cap)]
        ncols[i] = len(cols)
        if len(cols) < 3:
            continue
        out[i] = _pearson(S_true[i, cols], S_pred[i, cols])
    return out, ncols


# ---------------------------------------------------------------- reach@K (structural)
def reach_at_k_per_pert(S_true, signal_mask, pert_gene_idx, dist, src_row, K, k_deg=20):
    """Fraction of each pert's true top-k_deg DEGs reachable within K directed hops. Structural (no pred)."""
    n = S_true.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        g = pert_gene_idx[i]
        drow = _dist_row(dist, src_row, g) if g >= 0 else None
        if drow is None:
            continue
        degs = true_top_degs(S_true[i], signal_mask, g, k=k_deg)
        if len(degs) == 0:
            continue
        out[i] = float((drow[degs] <= K).mean())
    return out


# ---------------------------------------------------------------- effective resistance (structural)
def eff_resistance_per_pert(S_true, signal_mask, pert_gene_idx, Lpinv, comp, k_deg=20):
    """Mean effective resistance perturbed-node -> its true top-k_deg DEGs (undirected). Lower=better.
    Returns (mean_R array[n_pert], reach_frac array[n_pert] = same-component DEG fraction)."""
    n = S_true.shape[0]
    R = np.full(n, np.nan); frac = np.full(n, np.nan)
    diag = np.diag(Lpinv)
    for i in range(n):
        g = pert_gene_idx[i]
        if g < 0:
            continue
        degs = true_top_degs(S_true[i], signal_mask, g, k=k_deg)
        if len(degs) == 0:
            continue
        same = comp[degs] == comp[g]
        frac[i] = float(same.mean())
        d2 = degs[same]
        if len(d2) == 0:
            continue
        r = diag[g] + diag[d2] - 2.0 * Lpinv[g, d2]
        R[i] = float(np.mean(r))
    return R, frac


# ---------------------------------------------------------------- DNSA >= 2 hop
def dnsa_ge2hop_per_pert(S_true, S_pred, signal_mask, pert_gene_idx, dist, src_row, K, min_hops=2):
    """Sign accuracy over signal genes at min_hops <= dist <= K (multi-hop reachable), excl self.
    Rescaled 2·acc−1; nan if no such gene. Returns (per_pert array, pair_dnsa scalar, n_pairs)."""
    sig = np.where(signal_mask)[0]
    n = S_true.shape[0]
    per = np.full(n, np.nan); tot_match = 0; tot = 0
    for i in range(n):
        g = pert_gene_idx[i]
        drow = _dist_row(dist, src_row, g) if g >= 0 else None
        if drow is None:
            continue
        cand = sig[sig != g]
        dd = drow[cand]
        cols = cand[(dd >= min_hops) & (dd <= K)]
        if len(cols) == 0:
            continue
        st = np.sign(S_true[i, cols]); sp_ = np.sign(S_pred[i, cols])
        valid = st != 0
        if valid.sum() == 0:
            continue
        match = st[valid] == sp_[valid]
        per[i] = 2.0 * match.mean() - 1.0
        tot_match += int(match.sum()); tot += int(valid.sum())
    pair = (2.0 * tot_match / tot - 1.0) if tot > 0 else float("nan")
    return per, pair, tot
