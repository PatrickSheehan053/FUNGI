"""
FUNGI v9.1 — DASH Kernel Engine

Changes from v9.0:
  - Effective Resistance (ER) scoring integrated as a multiplicative term:
      ω(s→t) = Wq^β × exp(δ×T̃) × π_s^ψ × G_st × R_st^η
    where R_st is the rank-normalized approximate effective resistance and
    η is a fixed constant (default 0.3), not a tunable hyperparameter.
  - run_dash_and_score and build_graph_from_params accept optional
    er_scores (np.ndarray) and er_eta (float) keyword arguments.
  - er_scores=None (default) is a strict no-op: factor collapses to 1.0,
    preserving full backward compatibility with existing saved results.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import warnings
import graphblas as gb
import igraph as ig
import powerlaw

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _safe(x, fb=0.0):
    return float(x) if (x is not None and np.isfinite(x)) else fb


# ---------------------------------------------------------------------------
# Pre-computation helpers (called once in Phase 2, not per-evaluation)
# ---------------------------------------------------------------------------

def compute_source_quantile_weights(sources, weights, n_genes):
    """
    For each edge, compute its rank within its source gene's outgoing weight
    distribution, normalized to [0, 1].

    High-weight edges within their source get values near 1.0.
    This makes β a meaningful gradient across the actual signal range
    rather than amplifying arbitrary magnitude differences between sources.

    Returns array of same length as weights, in same order.
    """
    n_edges = len(weights)
    if n_edges == 0:
        return np.ones(0, dtype=np.float64)

    order = np.lexsort((-weights, sources))
    src_s = sources[order]

    src_change = np.concatenate([[True], src_s[1:] != src_s[:-1]])
    group_start = np.where(src_change)[0]
    group_sizes = np.diff(np.concatenate([group_start, [n_edges]]))

    pos_in_group = (np.arange(n_edges) -
                    np.repeat(group_start, group_sizes))
    n_in_group = np.repeat(group_sizes, group_sizes)

    W_q_sorted = 1.0 - (pos_in_group + 0.5) / np.maximum(n_in_group, 1.0)

    W_quantile = np.empty(n_edges, dtype=np.float64)
    W_quantile[order] = W_q_sorted
    return W_quantile


def compute_pagerank_kappa_multipliers(csr_matrix, n_genes,
                                       alpha=0.85, n_iter=60,
                                       hub_percentile=99.0,
                                       hub_multiplier=3.0):
    """
    Run power-iteration PageRank on the filtered parent graph.
    Returns a per-gene multiplier array: top hub_percentile% of genes
    get hub_multiplier × the base κ; all others get 1.0×.
    """
    n = csr_matrix.shape[0]
    out_deg = np.asarray(csr_matrix.sum(axis=1)).ravel().astype(np.float64)
    out_deg = np.where(out_deg > 0, out_deg, 1.0)

    D_inv = sp.diags(1.0 / out_deg)
    M = (D_inv @ csr_matrix).astype(np.float64)

    pr = np.ones(n, dtype=np.float64) / n
    teleport = (1.0 - alpha) / n
    for _ in range(n_iter):
        pr_new = alpha * (M.T @ pr) + teleport
        if np.linalg.norm(pr_new - pr, 1) < 1e-6:
            break
        pr = pr_new

    threshold = np.percentile(pr, hub_percentile)
    multipliers = np.ones(n_genes, dtype=np.float64)
    hub_mask = pr >= threshold
    multipliers[:len(hub_mask)][hub_mask] = hub_multiplier
    return multipliers


def compute_source_pert_impact(impact_array, perturbation_labels,
                               name_to_idx, n_genes,
                               pert_efficiency_map=None,
                               deg_out_parent=None):
    """
    Compute per-gene perturbation impact prior (π_s).

    v9.3 change (Gemini normalization):
        π_s is now normalized by the source gene's out-degree in the parent
        graph, converting from raw DEG footprint to regulatory efficiency
        (DEGs produced per outgoing edge). This prevents hub genes with
        thousands of outgoing edges from dominating purely through volume.

        π_s = (impact_i / mean_impact) / (deg_out_i / mean_deg_out)

        Genes not in the perturbation set keep π_s = 1.0 as before.
        If deg_out_parent is not provided, falls back to v9.2 behavior.
    """
    source_impact = np.ones(n_genes, dtype=np.float64)
    if len(impact_array) == 0 or len(perturbation_labels) == 0:
        return source_impact

    active = impact_array[impact_array > 0]
    if len(active) == 0:
        return source_impact
    mean_impact = float(np.mean(active))

    # Gemini normalization: compute efficiency-adjusted deg signal
    if deg_out_parent is not None and len(deg_out_parent) == n_genes:
        deg_out = np.asarray(deg_out_parent, dtype=np.float64)
        # Mean out-degree only over genes that ARE perturbation targets
        target_idxs = [name_to_idx.get(str(l)) for l in perturbation_labels]
        target_idxs = [i for i in target_idxs if i is not None and i < n_genes]
        if len(target_idxs) > 0:
            mean_deg_out = float(np.mean(deg_out[target_idxs]))
            mean_deg_out = max(mean_deg_out, 1.0)
        else:
            mean_deg_out = 1.0
    else:
        deg_out = None
        mean_deg_out = 1.0

    deg_signal = np.ones(n_genes, dtype=np.float64)
    for i, label in enumerate(perturbation_labels):
        gene_idx = name_to_idx.get(str(label))
        if gene_idx is not None and gene_idx < n_genes:
            raw_ratio = float(impact_array[i]) / max(mean_impact, 1.0)
            if deg_out is not None:
                # Normalize by out-degree: efficiency = DEGs_caused / edges_available
                gene_deg_out = max(float(deg_out[gene_idx]), 1.0)
                deg_out_ratio = gene_deg_out / mean_deg_out
                # Efficiency: penalizes genes whose DEG count is explained purely
                # by having more edges to fire from
                deg_signal[gene_idx] = raw_ratio / deg_out_ratio
            else:
                deg_signal[gene_idx] = raw_ratio

    if pert_efficiency_map:
        eff_signal = np.ones(n_genes, dtype=np.float64)
        eff_vals = np.array([v for v in pert_efficiency_map.values()
                             if np.isfinite(v) and v > 0], dtype=np.float64)
        if len(eff_vals) > 0:
            eff_max = float(np.percentile(eff_vals, 95))
            for gene_name, eff in pert_efficiency_map.items():
                gene_idx = name_to_idx.get(str(gene_name))
                if gene_idx is not None and gene_idx < n_genes and eff > 0:
                    eff_signal[gene_idx] = float(np.clip(eff / max(eff_max, 1e-6), 0.1, 2.0))
        source_impact = 0.5 * deg_signal + 0.5 * eff_signal
    else:
        source_impact = deg_signal

    # Clip to prevent extreme values from any combination of signals
    source_impact = np.clip(source_impact, 0.0, 10.0)
    return source_impact


def compute_chi_prior(deg_col_sums, n_genes, zeta=0.5):
    """
    Compute the perturbation pleiotropy prior χ for all n_genes genes.

    χ(s) captures how frequently gene s appears as a DEG across ALL
    training perturbations. This is derived from the column sums of the
    Phase 0 DEG matrix:

        col_sum[s] = number of training perturbations for which gene s
                     is a statistically significant DEG

    A gene that is a DEG under many independent CRISPRi knockdowns is a
    regulatory convergence point — many upstream pathways run through it,
    and it likely has its own significant downstream regulatory output.

    This signal is:
    - Global: defined for ALL 5,024 genes, not just the 96 perturbed ones
    - Data-derived: comes directly from the Phase 0 DE results
    - Biologically grounded: high col_sum = regulatory hub (cf. PANDA
      gene targeting score, Sonawane et al. 2017; hub-centred GRN priors,
      van Someren et al. 2006)
    - Generalizable: val/test genes that happen to be frequent DEGs in
      training perturbations will also receive this boost organically

    Formula:
        g  = mean(col_sum[col_sum > 0])    # mean over genes ever observed
        chi_raw(s) = max(1.0, 1 + log(col_sum[s] / g))  if col_sum[s] > g
                   = 1.0                                  otherwise
        chi(s) = chi_raw(s) ^ zeta          # dampen with fixed exponent

    At zeta=0.5 (default, fixed — not a hyperparameter):
        col_sum = g   → chi = 1.00  (average, neutral)
        col_sum = 2g  → chi = 1.30
        col_sum = 4g  → chi = 1.55
        col_sum = 8g  → chi = 1.76
        col_sum = 0   → chi = 1.00  (floor, no penalty)

    The log-damping ensures the boost is bounded even for extreme hubs.
    The zeta=0.5 power further compresses the range to a conservative
    [1.0, ~1.8] spread — meaningful but cannot dominate other DASH terms.

    Parameters
    ----------
    deg_col_sums : np.ndarray, shape (n_genes,)
        Column sums of the Phase 0 DEG matrix. From diagnostic_report.
    n_genes : int
    zeta : float
        Exponent applied to chi_raw. Fixed at 0.5 (not a hyperparameter).
        Increasing to 1.0 doubles the boost strength if needed.

    Returns
    -------
    chi : np.ndarray, shape (n_genes,), dtype float64
        Per-gene chi values in [1.0, ~1.8] for zeta=0.5.
    """
    chi = np.ones(n_genes, dtype=np.float64)
    if deg_col_sums is None or len(deg_col_sums) != n_genes:
        return chi

    col = np.asarray(deg_col_sums, dtype=np.float64)
    nonzero = col[col > 0]
    if len(nonzero) == 0:
        return chi

    g = float(np.mean(nonzero))  # mean over genes ever observed as a DEG

    # Log-damped boost for genes above the mean
    above = col > g
    chi[above] = np.maximum(1.0, 1.0 + np.log(col[above] / g))

    # Apply zeta exponent (fixed at 0.5 — conservative, not a hyperparameter)
    chi = np.power(chi, float(zeta))

    return chi

import numpy as np
 
 
def build_experimental_modifiers(experimental_df, sources, targets, gene_names,
                                  n_genes, alpha_md=0.5, alpha_stab=0.3,
                                  n_bootstraps=20, tau_shrinkage=0.5):
    """
    Build a per-edge multiplicative gate from experimental GRN columns.
 
    Returns
    -------
    total_gate          : np.ndarray float64 shape (len(sources),)
    pert_efficiency_map : dict {gene_name: float}
    """
    import pandas as pd
    from scipy.special import digamma
 
    n_edges = len(sources)
    pert_efficiency_map = {}
 
    if experimental_df is None or len(experimental_df) == 0:
        return np.ones(n_edges, dtype=np.float64), pert_efficiency_map
 
    src_col     = experimental_df.columns[0]
    tgt_col     = experimental_df.columns[1]
    exp_col_set = set(experimental_df.columns[2:])
    name_to_idx = {name: i for i, name in enumerate(gene_names)}
 
    exp_src_idx = (experimental_df[src_col].map(name_to_idx)
                   .fillna(-1).values.astype(np.int64))
    exp_tgt_idx = (experimental_df[tgt_col].map(name_to_idx)
                   .fillna(-1).values.astype(np.int64))
    valid         = (exp_src_idx >= 0) & (exp_tgt_idx >= 0)
    exp_src_valid = exp_src_idx[valid]
    exp_tgt_valid = exp_tgt_idx[valid]
    valid_row_idx = np.where(valid)[0]
 
    exp_keys    = exp_src_valid * np.int64(n_genes) + exp_tgt_valid
    sort_order  = np.argsort(exp_keys)
    keys_sorted = exp_keys[sort_order]
    query_keys  = (sources.astype(np.int64) * np.int64(n_genes)
                   + targets.astype(np.int64))
    ins       = np.searchsorted(keys_sorted, query_keys)
    ins       = np.clip(ins, 0, len(keys_sorted) - 1)
    matched   = keys_sorted[ins] == query_keys
    valid_pos = sort_order[ins]
 
    stability_gate = np.ones(n_edges, dtype=np.float64)
 
    if 'stability' in exp_col_set:
        stab_valid = experimental_df['stability'].values.astype(
            np.float64)[valid_row_idx]
 
        stab_arr          = np.full(n_edges, np.nan)
        stab_arr[matched] = stab_valid[valid_pos[matched]]
        has_stab          = ~np.isnan(stab_arr)
        median_stab       = (float(np.median(stab_arr[has_stab]))
                             if has_stab.sum() > 100 else 0.9)
        stab_filled       = np.where(has_stab, stab_arr, median_stab)
 
        K        = float(n_bootstraps)
        S        = np.clip(stab_filled * K, 0.0, K)
        log_odds = digamma(S + 1.0) - digamma(K - S + 1.0)
 
        L_max = max(float(digamma(K + 1.0) - digamma(1.0)), 1e-6)
 
        stability_gate = np.clip(
            1.0 + alpha_stab * log_odds / L_max,
            1.0 - alpha_stab,
            1.0 + alpha_stab)
        stability_gate[~has_stab] = 1.0
 
    md_gate = np.ones(n_edges, dtype=np.float64)
 
    if 'md_score' in exp_col_set and 'sign_agreement' in exp_col_set:
        md_valid   = experimental_df['md_score'].values.astype(
            np.float64)[valid_row_idx]
        sign_valid = experimental_df['sign_agreement'].values.astype(
            np.float64)[valid_row_idx]
 
        has_eff   = 'pert_efficiency' in exp_col_set
        eff_valid = (experimental_df['pert_efficiency'].values.astype(
            np.float64)[valid_row_idx] if has_eff
            else np.zeros(len(exp_src_valid), dtype=np.float64))
 
        active_mask    = md_valid > 0
        active_sources = np.unique(exp_src_valid[active_mask])
        n_active       = len(active_sources)
        panel_coverage = n_active / max(n_genes, 1)
 
        max_eff = 1e-6
        for src in active_sources:
            src_active = (exp_src_valid == src) & active_mask
            if src_active.sum() > 0:
                max_eff = max(max_eff, float(eff_valid[src_active].max()))
 
        lambda_per_src = {}
        for src in active_sources:
            src_active = (exp_src_valid == src) & active_mask
            if src_active.sum() == 0:
                lambda_per_src[int(src)] = 0.0
                continue
            eff_norm  = float(eff_valid[src_active].max()) / max_eff
            sign_cons = float(sign_valid[src_active].mean())
            kappa     = eff_norm * sign_cons * (1.0 + 10.0 * panel_coverage)
            lambda_per_src[int(src)] = kappa / (kappa + tau_shrinkage)
 
        md_rank_norm = np.zeros(len(exp_src_valid), dtype=np.float64)
        for src in active_sources:
            src_pos = (exp_src_valid == src) & active_mask
            n_src   = int(src_pos.sum())
            if n_src == 0:
                continue
            if n_src == 1:
                md_rank_norm[src_pos] = 1.0
                continue
            ranks = np.argsort(np.argsort(md_valid[src_pos])).astype(np.float64)
            md_rank_norm[src_pos] = ranks / (n_src - 1)
 
        matched_idx = np.where(matched)[0]
        if len(matched_idx) > 0:
            pos        = valid_pos[matched]
            src_at_pos = exp_src_valid[pos]
            lam_arr    = np.array([lambda_per_src.get(int(s), 0.0)
                                   for s in src_at_pos], dtype=np.float64)
            contrib    = (1.0 + alpha_md * lam_arr
                          * md_rank_norm[pos] * sign_valid[pos])
            boost      = (md_rank_norm[pos] > 0) & (lam_arr > 0)
            md_gate[matched_idx[boost]] = contrib[boost]
 
    if 'pert_efficiency' in exp_col_set:
        eff_sub = (experimental_df[[src_col, 'pert_efficiency']]
                   .copy()
                   .pipe(lambda df: df[df['pert_efficiency'] > 0])
                   .dropna())
        if len(eff_sub) > 0:
            max_eff_df = eff_sub.groupby(src_col)['pert_efficiency'].max()
            pert_efficiency_map = {
                str(k): float(v) for k, v in max_eff_df.items()
                if np.isfinite(v) and v > 0}
 
    total_gate = np.clip(stability_gate * md_gate, 0.5, 2.0)
    return total_gate, pert_efficiency_map

# ---------------------------------------------------------------------------
# GraphBLAS FFL topology (degree-normalized T̃_st)
# ---------------------------------------------------------------------------

def compute_dynamic_topology(W_sorted, src_sorted, tgt_sorted, k_core, n):
    """
    Compute degree-normalized FFL triangle counts T̃_st for each edge.
    """
    te = min(int(n * k_core), len(W_sorted))
    T = np.zeros(len(W_sorted), dtype=np.float64)
    if te < 1:
        return T

    cr, cc = src_sorted[:te], tgt_sorted[:te]
    A = gb.Matrix.from_coo(
        cr.astype(np.uint64), cc.astype(np.uint64),
        np.ones(te, dtype=np.float64), nrows=n, ncols=n)

    Tgb = A.mxm(A, gb.semiring.plus_times).new(mask=A.S)

    do = np.zeros(n, dtype=np.float64)
    oi, ov = A.reduce_rowwise(gb.monoid.plus).new().to_coo()
    do[oi] = ov

    di = np.zeros(n, dtype=np.float64)
    ii, iv = A.reduce_columnwise(gb.monoid.plus).new().to_coo()
    di[ii] = iv

    tr, tc, zv = Tgb.to_coo()
    if len(zv) > 0:
        cf = cr.astype(np.int64) * n + cc.astype(np.int64)
        tf = tr.astype(np.int64) * n + tc.astype(np.int64)
        to = np.argsort(tf)
        tfs, zvs = tf[to], zv[to]
        si = np.searchsorted(tfs, cf)
        vi = np.clip(si, 0, len(tfs) - 1)
        hm = (tfs[vi] == cf)
        mz = np.zeros(te, dtype=np.float64)
        mz[hm] = zvs[vi[hm]]
    else:
        mz = np.zeros(te, dtype=np.float64)

    denom = np.sqrt(np.maximum(do[cr] * di[cc], 1.0))
    v = mz > 0
    r = np.zeros(te, dtype=np.float64)
    r[v] = mz[v] / denom[v]
    np.clip(r, 0.0, 1.0, out=r)
    T[:te] = r
    return T


# ---------------------------------------------------------------------------
# Edge selection with per-gene soft kappa
# ---------------------------------------------------------------------------

def select_edges(omega, W, src, tgt, pert_nodes, n, lam,
                 per_gene_kappa, kappa_base):
    """
    Select edges by descending omega score, respecting per-gene hub caps.
    """
    budget = int(np.round(n * lam))

    effective_caps = np.maximum(
        (per_gene_kappa * kappa_base * n).astype(np.int64), 1)

    prot = []
    for p in pert_nodes:
        pe = np.where(src == p)[0]
        if len(pe) > 0:
            k = min(3, len(pe))
            prot.extend(pe[np.argsort(omega[pe])[-k:]])
    prot = np.unique(np.array(prot, dtype=int)) if prot else np.array([], dtype=int)

    rem = budget - len(prot)
    if rem > 0:
        m = np.ones(len(omega), dtype=bool)
        if len(prot) > 0:
            m[prot] = False
        av = np.where(m)[0]
        if rem < len(av):
            fi = av[np.argpartition(omega[av], -rem)[-rem:]]
        else:
            fi = av
        sel = np.concatenate([prot, fi]) if len(prot) > 0 else fi
    else:
        sel = prot[:budget]

    ss = src[sel]
    nc = np.bincount(ss, minlength=n)
    ov = np.where(nc > effective_caps[np.arange(n)])[0]

    if len(ov) > 0:
        km = np.ones(len(sel), dtype=bool)
        for nd in ov:
            ex = nc[nd] - effective_caps[nd]
            if ex <= 0:
                continue
            np_ = np.where(ss == nd)[0]
            nd_ = min(ex, len(np_))
            if nd_ > 0:
                km[np_[np.argsort(omega[sel[np_]])[:nd_]]] = False
        sel = sel[km]
        ss = src[sel]
        freed = budget - len(sel)
        if freed > 0:
            used = set(sel.tolist())
            cm = np.ones(len(omega), dtype=bool)
            for i in used:
                cm[i] = False
            for nd in ov:
                cm[src == nd] = False
            cands = np.where(cm)[0]
            if len(cands) > 0:
                nf = min(freed, len(cands))
                sel = np.concatenate(
                    [sel, cands[np.argpartition(omega[cands], -nf)[-nf:]]])

    return src[sel], tgt[sel], W[sel]


# ---------------------------------------------------------------------------
# Shatter checks
# ---------------------------------------------------------------------------

def check_shatter(ss, st, sw, od, n, active, cfg):
    ne = len(ss)
    if ne > cfg.get("max_edge_count", 500000):
        return True, "density_collapse"
    if (n - active) / max(n, 1) > cfg.get("max_orphan_fraction", 0.70):
        return True, "orphan_collapse"
    if ne > 0 and active > 0:
        try:
            G = sp.coo_matrix((np.ones(ne), (ss, st)), shape=(n, n))
            _, lb = connected_components(csgraph=G, directed=False,
                                         return_labels=True)
            if np.bincount(lb).max() / n < cfg.get("min_gwcc_fraction", 0.30):
                return True, "gwcc_percolation"
        except Exception:
            return True, "gwcc_percolation"
    min_clust = cfg.get("min_clustering", None)
    if min_clust is not None and ne > 50:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_u = ig.Graph(n=n, edges=edges, directed=True).as_undirected(
                mode="collapse")
            cc = ig_u.transitivity_undirected()
            if np.isfinite(cc) and cc < min_clust:
                return True, "clustering_collapse"
        except Exception:
            pass
    return False, None


# ---------------------------------------------------------------------------
# Utopia loss
# ---------------------------------------------------------------------------

def calculate_utopia_loss(ss, st, sw, n, od, active, kappa_base, ub, lw):
    ne = len(ss)

    def _p_smooth(par, obs, ub, lw, buffer_frac=0.10, sharpness=5.0):
        b = ub[par]
        w = _safe(lw[par], 1.)
        o = _safe(obs, 0.)
        bound_width = max(abs(b[1] - b[0]), 1e-6)
        buffer = bound_width * buffer_frac
        if b[0] <= o <= b[1]:
            return 0.
        if o < b[0]:
            raw_dist = (b[0] - o) / max(abs(b[0]), 1e-6)
            dist_beyond = max(0., (b[0] - o) - buffer)
        else:
            raw_dist = (o - b[1]) / max(abs(b[1]), 1e-6)
            dist_beyond = max(0., (o - b[1]) - buffer)
        base_penalty = raw_dist ** 2
        onset = 1.0 / (1.0 + np.exp(-sharpness * (dist_beyond / bound_width)))
        return w * min(base_penalty, 4.0) * onset

    ao = 1.0
    try:
        cap = int(n * 0.15)
        cd = od[(od > 0) & (od < cap)]
        if len(cd) > 10 and len(np.unique(cd)) >= 3:
            ao = _safe(powerlaw.Fit(
                cd, xmin=2, discrete=True, verbose=False).power_law.alpha, 1.)
    except Exception:
        pass
    ta = _p_smooth("alpha", ao, ub, lw)

    go = 1.0
    try:
        if active > 1 and np.sum(od) > 0:
            sd = np.sort(od)
            nn = len(sd)
            go = _safe((2. * np.sum(np.arange(1, nn + 1) * sd)) /
                       (nn * np.sum(sd)) - (nn + 1) / nn, 1.)
    except Exception:
        pass
    tg = _p_smooth("gini", go, ub, lw)

    smo = _safe((np.max(od) / n) if len(od) > 0 else 0.)
    kappa_frac = 0.25
    full_w = _safe(lw.get("S_max", 1.0), 1.0)
    bound_penalty = _p_smooth("S_max", smo, ub, {"S_max": full_w * (1 - kappa_frac)})
    kappa_excess = max(0., (smo - kappa_base) / max(kappa_base, 1e-6))
    kappa_penalty = full_w * kappa_frac * kappa_excess ** 2
    ts = bound_penalty + kappa_penalty

    co, qo, ro = 0., 0., 1.
    tc = _safe(lw["C"], 1.)
    tq = _safe(lw["Q"], 1.)
    tr = _safe(lw["rho"], 1.)

    if ne > 100:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_g = ig.Graph(n=n, edges=edges, directed=True,
                            edge_attrs={'weight': sw.tolist()})
            try:
                rc = ig_g.assortativity_degree(directed=True)
                if np.isfinite(rc):
                    ro = rc
                    try:
                        od_pos = od[od > 0].astype(np.float64)
                        if len(od_pos) > 10:
                            k1 = float(np.mean(od_pos))
                            k2 = float(np.mean(od_pos ** 2))
                            k3 = float(np.mean(od_pos ** 3))
                            if k1 > 1e-6 and k3 > 1e-6:
                                rho_base = _safe(
                                    -(k2 / k1) ** 2 / max(k3 / k1, 1e-6), -0.05)
                                rho_base = float(np.clip(rho_base, -0.50, 0.0))
                            else:
                                rho_base = -0.05
                            rho_excess = ro - rho_base
                            rho_excess_ub = ub["rho"][1] - rho_base
                            rho_excess_lb = ub["rho"][0] - rho_base
                            tr = _p_smooth(
                                "rho", rho_excess,
                                {**ub, "rho": [rho_excess_lb, rho_excess_ub]}, lw)
                        else:
                            tr = _p_smooth("rho", ro, ub, lw)
                    except Exception:
                        tr = _p_smooth("rho", ro, ub, lw)
            except Exception:
                pass
            try:
                ig_u = ig_g.as_undirected(
                    mode="collapse", combine_edges=dict(weight="sum"))
                cc_val = ig_u.transitivity_undirected()
                if np.isfinite(cc_val):
                    co = cc_val
                    tc = _p_smooth("C", co, ub, lw)
                pt = ig_u.community_multilevel()
                qm = pt.modularity
                if np.isfinite(qm):
                    qo = qm
                    tq = _p_smooth("Q", qo, ub, lw)
            except Exception:
                pass
        except Exception:
            pass

    raw = (_safe(ta) + _safe(tc) + _safe(tq) + _safe(tg) +
           _safe(tr) + _safe(ts))
    return np.sqrt(max(raw, 0.)), {
        'alpha': _safe(ao, 1.), 'C': _safe(co), 'Q': _safe(qo),
        'Gini': _safe(go, 1.), 'rho': _safe(ro, 1.), 'S_max': _safe(smo)}


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_dash_and_score(params, W, W_q, D, sources, targets, n_genes,
                       perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, per_gene_kappa, source_pert_impact,
                       md_gate=None, er_scores=None, er_eta=0.3,
                       inter_mask=None, chi_prior=None):
    """
    Score a single hyperparameter configuration via the DASH kernel.

    Parameters
    ----------
    ... (unchanged from v9.0) ...
    er_scores : np.ndarray or None
        Per-edge rank-normalized effective resistance scores in [0.05, 1.0],
        same order as the presorted W/sources/targets arrays.
        If None, the ER factor is 1.0 for all edges (backward-compatible).
    er_eta : float
        Exponent for R_st^η in the DASH score (default 0.3).
    """
    try:
        beta, delta, kappa_base, k_core, lam, psi = params
        param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
        rng = np.random.default_rng(param_hash)
        k_core = max(k_core, max(5.0, lam * 0.4))
        T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
        Nm = shatter_cfg.get("max_edge_count", 500000)
        Ne = min(len(W), Nm + 10000)
        Ws = W[:Ne]
        Wqs = W_q[:Ne]
        ss = sources[:Ne]
        ts = targets[:Ne]
        Ts = T_local[:Ne]
        pi_s = np.power(source_pert_impact[ss], psi)
        gate = md_gate[:Ne] if md_gate is not None else np.ones(Ne, dtype=np.float64)
        # ── SCBER factor ─────────────────────────────────────────────────────
        # inter_mask=None  → flat ER (all edges, backward-compatible)
        # inter_mask given → SCBER: ER boost only for inter-module edges
        if er_scores is not None:
            base_er = np.power(er_scores[:Ne], er_eta)
            er = (np.where(inter_mask[:Ne], base_er, 1.0)
                  if inter_mask is not None else base_er)
        else:
            er = np.ones(Ne, dtype=np.float64)
        # ── χ prior (perturbation pleiotropy — global, all genes) ─────────────
        chi = chi_prior[ss] if chi_prior is not None else np.ones(Ne, dtype=np.float64)
        # ── DASH score ───────────────────────────────────────────────────────
        num = (Wqs ** beta) * np.exp(delta * Ts) * pi_s * gate * er * chi

        order = np.lexsort((-num, ss))
        so, to, Wo, no = ss[order], ts[order], Ws[order], num[order]

        mhs = shatter_cfg.get("max_hub_saturation", 0.15)
        surv_s, surv_t, surv_W = select_edges(
            no, Wo, so, to, perturbed_nodes, n_genes, lam,
            per_gene_kappa, kappa_base)

        surv_s, surv_t, surv_W = _motif_repair_swap(
            surv_s, surv_t, surv_W, no, so, to, n_genes,
            max_swap_fraction=0.03, rng=rng)

        od = np.bincount(surv_s, minlength=n_genes)
        idd = np.bincount(surv_t, minlength=n_genes)
        active = int(np.count_nonzero(od + idd > 0))

        sh, reason = check_shatter(
            surv_s, surv_t, surv_W, od, n_genes, active, shatter_cfg)

        if sh:
            return {
                'beta': beta, 'delta': delta, 'kappa': kappa_base,
                'k_core': k_core, 'lambda': lam, 'psi': psi,
                'utopia_loss': 999., 'is_shattered': 1,
                'shatter_reason': reason, 'n_edges': len(surv_s),
                'active_nodes': active, 'alpha': 1., 'Gini': 1.,
                'rho': 1., 'C': 0., 'Q': 0.,
                'S_max': _safe((np.max(od) / n_genes) if len(od) > 0 else 0)}

        loss, topo = calculate_utopia_loss(
            surv_s, surv_t, surv_W, n_genes, od, active,
            kappa_base, utopian_bounds, loss_weights)

        gf = 0.
        try:
            G = sp.coo_matrix(
                (np.ones(len(surv_W)), (surv_s, surv_t)),
                shape=(n_genes, n_genes))
            _, lb = connected_components(csgraph=G, directed=False,
                                          return_labels=True)
            gf = _safe(np.bincount(lb).max() / n_genes)
        except Exception:
            pass

        gp = 0.
        if gf < 0.45:
            gp = 8. * ((0.45 - gf) / 0.45) ** 2
        loss = _safe(np.sqrt(max(loss ** 2 + gp, 0.)), 999.)

        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core, 'lambda': lam, 'psi': psi,
            'utopia_loss': loss, 'is_shattered': 0, 'shatter_reason': None,
            'n_edges': len(surv_s), 'active_nodes': active,
            'gwcc_fraction': gf,
            'alpha': topo['alpha'], 'Gini': topo['Gini'],
            'rho': topo['rho'], 'C': topo['C'], 'Q': topo['Q'],
            'S_max': topo['S_max']}

    except Exception as e:
        beta, delta, kappa_base, k_core, lam, psi = params
        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core, 'lambda': lam, 'psi': psi,
            'utopia_loss': 999., 'is_shattered': 1,
            'shatter_reason': f'crash:{str(e)[:60]}',
            'n_edges': 0, 'active_nodes': 0,
            'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'Q': 0., 'S_max': 0.}


# ---------------------------------------------------------------------------
# Motif repair swap (budget 3%)
# ---------------------------------------------------------------------------

def _motif_repair_swap(surv_s, surv_t, surv_W, omega_full, src_full, tgt_full,
                        n_genes, max_swap_fraction=0.03, rng=None):
    n_selected = len(surv_s)
    budget_swaps = max(1, int(n_selected * max_swap_fraction))

    if n_selected < 10 or budget_swaps < 1:
        return surv_s, surv_t, surv_W

    try:
        adj = sp.coo_matrix(
            (np.ones(n_selected), (surv_s, surv_t)),
            shape=(n_genes, n_genes)).tocsr()
        A2 = adj @ adj
        A2_coo = A2.tocoo()
        selected_set = set(zip(surv_s.tolist(), surv_t.tolist()))

        candidate_src, candidate_tgt = [], []
        for ci, cj, cv in zip(A2_coo.row, A2_coo.col, A2_coo.data):
            if cv > 0 and (ci, cj) not in selected_set and ci != cj:
                candidate_src.append(ci)
                candidate_tgt.append(cj)

        if len(candidate_src) == 0:
            return surv_s, surv_t, surv_W

        omega_lookup = {}
        for idx in range(len(src_full)):
            omega_lookup[(int(src_full[idx]), int(tgt_full[idx]))] = float(omega_full[idx])

        close_scores = sorted(
            [(omega_lookup.get((ci, cj), 0.0), ci, cj)
             for ci, cj in zip(candidate_src, candidate_tgt)],
            reverse=True)

        selected_scores = sorted(
            [(omega_lookup.get((int(surv_s[idx]), int(surv_t[idx])), 0.0), idx)
             for idx in range(n_selected)])

        selected_mask = np.ones(n_selected, dtype=bool)
        new_src, new_tgt, new_W = [], [], []
        n_swapped = 0

        for (close_om, ci, cj), (sel_om, sel_idx) in zip(
                close_scores[:budget_swaps], selected_scores[:budget_swaps]):
            if close_om <= sel_om:
                break
            selected_mask[sel_idx] = False
            new_src.append(ci)
            new_tgt.append(cj)
            new_W.append(omega_lookup.get((ci, cj), 0.0))
            n_swapped += 1

        if n_swapped == 0:
            return surv_s, surv_t, surv_W

        keep_idx = np.where(selected_mask)[0]
        final_s = np.concatenate([surv_s[keep_idx],
                                   np.array(new_src, dtype=surv_s.dtype)])
        final_t = np.concatenate([surv_t[keep_idx],
                                   np.array(new_tgt, dtype=surv_t.dtype)])
        final_W = np.concatenate([surv_W[keep_idx],
                                   np.array(new_W, dtype=surv_W.dtype)])
        return final_s, final_t, final_W

    except Exception:
        return surv_s, surv_t, surv_W


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------

def build_graph_from_params(params, W, W_q, D, sources, targets, n_genes,
                            perturbed_nodes, shatter_cfg, per_gene_kappa,
                            source_pert_impact, md_gate=None,
                            er_scores=None, er_eta=0.3, inter_mask=None,
                            chi_prior=None):
    """
    Reconstruct the final edge set for a given hyperparameter recipe.

    er_scores / er_eta: same semantics as run_dash_and_score.
    """
    beta, delta, kappa_base, k_core, lam, psi = params
    param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
    rng = np.random.default_rng(param_hash)
    k_core = max(k_core, max(5.0, lam * 0.4))
    T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
    Nm = shatter_cfg.get("max_edge_count", 500000)
    Ne = min(len(W), Nm + 10000)
    Ws, Wqs = W[:Ne], W_q[:Ne]
    ss, ts, Ts = sources[:Ne], targets[:Ne], T_local[:Ne]
    pi_s = np.power(source_pert_impact[ss], psi)
    gate = md_gate[:Ne] if md_gate is not None else np.ones(Ne, dtype=np.float64)
    if er_scores is not None:
        base_er = np.power(er_scores[:Ne], er_eta)
        er = (np.where(inter_mask[:Ne], base_er, 1.0)
              if inter_mask is not None else base_er)
    else:
        er = np.ones(Ne, dtype=np.float64)
    chi = chi_prior[ss] if chi_prior is not None else np.ones(Ne, dtype=np.float64)
    num = (Wqs ** beta) * np.exp(delta * Ts) * pi_s * gate * er * chi

    order = np.lexsort((-num, ss))
    so, to, Wo, no = ss[order], ts[order], Ws[order], num[order]

    surv_s, surv_t, surv_W = select_edges(
        no, Wo, so, to, perturbed_nodes, n_genes, lam,
        per_gene_kappa, kappa_base)

    surv_s, surv_t, surv_W = _motif_repair_swap(
        surv_s, surv_t, surv_W, no, so, to, n_genes,
        max_swap_fraction=0.03, rng=rng)

    return surv_s, surv_t, surv_W


def recompute_loss_from_metrics(metrics, utopian_bounds, loss_weights,
                                kappa_base):
    """
    Recompute utopia loss from pre-recorded topology metrics.
    (Unchanged from v9.0 — ER does not affect loss computation.)
    """

    def _p_smooth(par, obs, ub, lw, buffer_frac=0.10, sharpness=5.0):
        b = ub[par]
        w = _safe(lw.get(par, 1.0), 1.0)
        o = _safe(obs, 0.)
        bound_width = max(abs(b[1] - b[0]), 1e-6)
        buffer = bound_width * buffer_frac
        if b[0] <= o <= b[1]:
            return 0.
        if o < b[0]:
            raw_dist = (b[0] - o) / max(abs(b[0]), 1e-6)
            dist_beyond = max(0., (b[0] - o) - buffer)
        else:
            raw_dist = (o - b[1]) / max(abs(b[1]), 1e-6)
            dist_beyond = max(0., (o - b[1]) - buffer)
        base_penalty = raw_dist ** 2
        onset = 1.0 / (1.0 + np.exp(-sharpness * (dist_beyond / bound_width)))
        return w * min(base_penalty, 4.0) * onset

    ta = _p_smooth("alpha", metrics.get("alpha", 1.0), utopian_bounds, loss_weights)
    tg = _p_smooth("gini", metrics.get("Gini", 1.0), utopian_bounds, loss_weights)

    smo = metrics.get("S_max", 0.0)
    kf = 0.25
    fw = _safe(loss_weights.get("S_max", 1.0), 1.0)
    ts_bound = _p_smooth("S_max", smo, utopian_bounds, {"S_max": fw * (1 - kf)})
    kappa_excess = max(0., (smo - kappa_base) / max(kappa_base, 1e-6))
    ts = ts_bound + fw * kf * kappa_excess ** 2

    tc = _p_smooth("C", metrics.get("C", 0.0), utopian_bounds, loss_weights)
    tq = _p_smooth("Q", metrics.get("Q", 0.0), utopian_bounds, loss_weights)
    tr = _p_smooth("rho", metrics.get("rho", 1.0), utopian_bounds, loss_weights)

    raw = _safe(ta) + _safe(tc) + _safe(tq) + _safe(tg) + _safe(tr) + _safe(ts)

    gf = metrics.get("gwcc_fraction", 1.0)
    gp = 0.
    if gf < 0.45:
        gp = 8. * ((0.45 - gf) / 0.45) ** 2

    return float(np.sqrt(max(raw + gp, 0.)))
