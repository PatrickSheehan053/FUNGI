"""
FUNGI — Synthetic Phase 0 Diagnostics
alt_diagnostics.py

Drop-in replacement for diagnostics.run_diagnostics() when
fungi_mode = "synthetic" in the pipeline config.

Instead of probing biological network topology (α, Gini, S_max, Q, C, ρ),
this module probes six graph properties chosen to directly improve
SPECTRA/FAGCN perturbation-prediction performance:

  1. EPR@k          — fraction of outgoing edges from perturbed sources
                      that point to known DEGs (edge biological precision)
  2. Weight entropy — Shannon entropy of the DASH score distribution
                      (ensures FAGCN attention has informative gradients)
  3. Path length    — mean BFS distance from perturbed source to its DEGs
                      (bounded above by the GNN propagation depth L)
  4. Spectral gap   — λ₂ of the normalised Laplacian, calibrated to balance
                      over-squashing and over-smoothing (Rusch et al. 2022;
                      NeurIPS 2024 Spectral Graph Pruning)
  5. Source conc.   — mean outdegree of active regulatory source genes
                      (prevents degenerate flat-source topology)
  6. Heterophily    — cross-community edge fraction (FAGCN's adaptive
                      frequency mixing works best at intermediate values;
                      Bo et al. 2021)

All six target windows are DATA-DERIVED from the dense parent graph at
λ_eff density — probed once in Phase 0, exactly like the organic mode.
No hard-coded dataset-specific constants are used.

Shared infrastructure (build_impact_array, _compute_lam_eff, etc.) is
imported directly from diagnostics.py to avoid duplication.

Public API — identical return format to diagnostics.run_diagnostics():
  run_synthetic_diagnostics(
      adata, n_genes, cfg_diagnostics, cfg_input,
      raw_sparse_mat, spectra_L=3, community_labels=None)
  → (utopian_bounds, loss_weights, diagnostic_report, deg_matrix_csr)

  NOTE: utopian_bounds contains the 6 synthetic keys plus a loose "S_max"
  key [0.001, 0.30] required by the downstream build_shatter_config() call.
  "S_max" receives zero loss weight so it never affects optimisation.
"""

import gc
import warnings
import numpy as np
import scipy.sparse as sp
def _sp_shannon(pk, base=None):
    pk = np.asarray(pk, dtype=np.float64)
    pk = pk[pk > 0]
    if len(pk) == 0:
        return 0.0
    pk = pk / pk.sum()
    h = float(-np.sum(pk * np.log(pk)))
    if base is not None:
        h /= float(np.log(base))
    return h

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Shared infrastructure imported from organic diagnostics ──────────────────
# We reuse build_impact_array, _compute_lam_eff, _safe_float, _normalize_weights
# so synthetic and organic modes always derive λ_eff identically.
from diagnostics import (
    build_impact_array,
    _compute_lam_eff,
    _safe_float,
    _normalize_weights,
    FALLBACK_CONFIDENCE,
)

# ── Synthetic physical limits ─────────────────────────────────────────────────
# Used only inside this file; no overlap with organic PHYSICAL dict.
_SYN_PHYSICAL = {
    "epr_k":          (0.0,  1.0),
    "weight_entropy": (0.0, 10.0),
    "path_length":    (0.5, 10.0),
    "spectral_gap":   (0.001, 0.6),
    "source_conc":    (1.0, 200.0),
    "heterophily":    (0.0,  2.0),
}

# Fallback bounds — used when a probe fails or data is insufficient
_SYN_FALLBACK = {
    "epr_k":          [0.10, 1.00],
    "weight_entropy": [3.80, 7.00],
    "path_length":    [1.50, 3.00],
    "spectral_gap":   [0.05, 0.15],
    "source_conc":    [10.0, 60.0],
    "heterophily":    [0.40, 0.80],
}

_SYN_DESCRIPTIONS = {
    "epr_k":          "Edge biological precision (DEG hit rate)",
    "weight_entropy": "DASH weight Shannon entropy (attention signal)",
    "path_length":    "Mean DEG path length (GNN reachability)",
    "spectral_gap":   "Fiedler value λ₂ (mixing vs smoothing balance)",
    "source_conc":    "Mean outdegree of active source genes",
    "heterophily":    "Edge feature cosine-distance (FAGCN regime; community-fraction fallback)",
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _clip_syn(lo, hi, param):
    """Clip (lo, hi) to the physical range for param and ensure lo < hi."""
    floor, ceiling = _SYN_PHYSICAL.get(param, (0.0, 1e6))
    lo = float(np.clip(lo, floor, ceiling))
    hi = float(np.clip(hi, floor, ceiling))
    if lo >= hi:
        mid = (lo + hi) / 2.0
        span = (ceiling - floor) * 0.02
        lo = max(mid - span, floor)
        hi = min(mid + span, ceiling)
    return lo, hi


def _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes):
    """
    Return (src_arr, tgt_arr, weight_arr) for the top ⌊λ_eff × n_genes⌋ edges
    in the parent graph, sorted descending by weight.
    This is the same substrate used by the organic S_max and C probes.
    """
    coo = raw_sparse_mat.tocoo()
    n_keep = max(1, int(lam_eff * n_genes))

    if n_keep >= len(coo.data):
        order = np.argsort(coo.data)[::-1]
    else:
        # Partial sort — faster than full sort for large graphs
        order = np.argpartition(coo.data, -n_keep)[-n_keep:]
        order = order[np.argsort(coo.data[order])[::-1]]

    return (coo.row[order].copy().astype(np.int32),
            coo.col[order].copy().astype(np.int32),
            coo.data[order].copy().astype(np.float32))


# ── Probe 1 — EPR@k ──────────────────────────────────────────────────────────

def _probe_epr_k(raw_sparse_mat, deg_matrix_csr, perturbed_nodes,
                  lam_eff, n_genes, floor_frac=0.70):
    """
    Probe EPR@k (Expected Precision at k) on the parent graph at λ_eff density.

    For each perturbed gene that appears as a source in the λ_eff substrate,
    compute the fraction of its outgoing edges that point to known DEGs for
    that perturbation. Report mean EPR@k across all such sources.

    The synthetic target lower bound is floor_frac × observed_parent_epr_k
    so DASH is asked to preserve at least that fraction of biological precision.

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["epr_k"]
    if (raw_sparse_mat is None or deg_matrix_csr is None
            or len(perturbed_nodes) == 0):
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        src_arr, tgt_arr, _ = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)

        # Build per-source adjacency as dict for fast lookup
        from collections import defaultdict
        adj = defaultdict(list)
        for s, t in zip(src_arr.tolist(), tgt_arr.tolist()):
            adj[s].append(t)

        n_rows = deg_matrix_csr.shape[0]
        epr_scores = []

        for src in perturbed_nodes:
            src = int(src)
            out_tgts = adj.get(src, [])
            k = len(out_tgts)
            if k == 0 or src >= n_rows:
                continue
            # Sparse row lookup: which of out_tgts are DEGs for this source?
            deg_row_dense = np.asarray(
                deg_matrix_csr[src, :].todense()).ravel().astype(np.float32)
            hits = int(np.sum(deg_row_dense[out_tgts] > 0))
            epr_scores.append(hits / k)

        if len(epr_scores) < 3:
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        observed = float(np.mean(epr_scores))
        conf = min(len(epr_scores) / max(len(perturbed_nodes), 1), 1.0) * 0.85
        lo = max(observed * floor_frac, 0.01)
        hi = 1.0
        lo, hi = _clip_syn(lo, hi, "epr_k")
        return lo, hi, _safe_float(conf, FALLBACK_CONFIDENCE)

    except Exception:
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ── Probe 2 — Weight entropy ──────────────────────────────────────────────────

def _probe_weight_entropy(raw_sparse_mat, lam_eff, n_genes,
                           absolute_floor=3.8):
    """
    Probe Shannon entropy of the log-scaled edge weight distribution.

    Mirrors the W_spectra computation in the main notebook:
        W_spectra = log1p(w) / log1p(max_w)

    A floor of ≥3.8 bits ensures FAGCN's attention mechanism has
    meaningful input gradients. Near-binary weights (~2.5 bits) degrade
    FAGCN to a standard GCN.

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["weight_entropy"]
    if raw_sparse_mat is None:
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        _, _, weights = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)
        weights = weights.astype(np.float64)

        w_max = float(weights.max())
        if w_max <= 0:
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        w_scaled = np.log1p(weights) / np.log1p(w_max)
        counts, _ = np.histogram(w_scaled, bins=50, range=(0.0, 1.0))
        p = counts.astype(np.float64)
        p /= p.sum()
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))

        lo = max(entropy * 0.85, absolute_floor)
        hi = 7.0
        lo, hi = _clip_syn(lo, hi, "weight_entropy")
        return lo, hi, 0.85

    except Exception:
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ── Probe 3 — Mean DEG path length ───────────────────────────────────────────

def _probe_path_length(raw_sparse_mat, deg_matrix_csr, perturbed_nodes,
                        lam_eff, n_genes, L=3, n_sample=30, seed=42):
    """
    Probe mean BFS distance from perturbed sources to their known DEGs.

    The hard upper bound is L (number of FAGCN message-passing layers):
    an L-layer GNN cannot propagate information beyond L hops regardless
    of weights or training. The lower bound prevents degenerate bipartite
    structure (all paths = 1).

    BFS is run from a random sample of perturbed sources for efficiency.

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["path_length"]
    if (raw_sparse_mat is None or deg_matrix_csr is None
            or len(perturbed_nodes) == 0):
        return fb[0], float(L), FALLBACK_CONFIDENCE

    try:
        src_arr, tgt_arr, _ = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)

        # Build forward adjacency list
        adj = [[] for _ in range(n_genes)]
        for s, t in zip(src_arr.tolist(), tgt_arr.tolist()):
            adj[s].append(t)

        rng = np.random.default_rng(seed)
        n_samp = min(n_sample, len(perturbed_nodes))
        sample = rng.choice(perturbed_nodes, size=n_samp, replace=False)

        n_rows = deg_matrix_csr.shape[0]
        all_lengths = []

        for src in sample:
            src = int(src)
            if src >= n_rows:
                continue
            deg_row = np.asarray(
                deg_matrix_csr[src, :].todense()).ravel().astype(np.float32)
            deg_targets = set(int(i) for i in np.where(deg_row > 0)[0])
            if not deg_targets:
                continue

            # BFS — stop at depth L+1 (no benefit searching deeper)
            dist = {src: 0}
            queue = [src]
            head = 0
            while head < len(queue):
                node = queue[head]; head += 1
                d = dist[node]
                if d >= L + 1:
                    break
                for nb in adj[node]:
                    if nb not in dist:
                        dist[nb] = d + 1
                        queue.append(nb)
                        if nb in deg_targets:
                            all_lengths.append(d + 1)

        if len(all_lengths) < 3:
            return 1.5, float(L), FALLBACK_CONFIDENCE

        observed = float(np.mean(all_lengths))
        conf = min(len(all_lengths) / max(n_samp * 5, 1), 0.80)
        lo = max(observed * 0.80, 1.5)
        hi = float(L)
        # If observed is already above L, the graph has poor connectivity
        # for the given GNN depth — set lo conservatively
        if observed >= float(L):
            lo = float(L) * 0.70
        lo, hi = _clip_syn(lo, hi, "path_length")
        return lo, hi, _safe_float(conf, FALLBACK_CONFIDENCE)

    except Exception:
        return 1.5, float(L), FALLBACK_CONFIDENCE


# ── Probe 4 — Spectral gap ───────────────────────────────────────────────────

def _probe_spectral_gap(raw_sparse_mat, lam_eff, n_genes, window_frac=0.35):
    """
    Probe λ₂ of the symmetrised normalised Laplacian (Fiedler value).

    Returns a symmetric window around the observed parent-graph spectral gap
    rather than a one-sided floor, because over-squashing and over-smoothing
    are opposing objectives governed by the same spectral property:
    maximising the gap reduces over-squashing but promotes over-smoothing
    (Rusch et al. 2022; Di Giovanni et al. 2023). A window target forces
    DASH to find graphs balanced between the two failure modes.

    NeurIPS 2024 (Caro et al.) shows edge deletion — exactly what DASH does
    — can simultaneously improve both properties, validating this approach.

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["spectral_gap"]
    if raw_sparse_mat is None:
        return fb[0], fb[1], 0.5

    try:
        from scipy.sparse.linalg import eigsh

        src_arr, tgt_arr, _ = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)

        A = sp.csr_matrix(
            (np.ones(len(src_arr), dtype=np.float32), (src_arr, tgt_arr)),
            shape=(n_genes, n_genes))
        A = (A + A.T).multiply(0.5)
        A.data[A.data == 0] = 0
        A.eliminate_zeros()

        deg_arr = np.array(A.sum(axis=1)).ravel()
        deg_arr[deg_arr == 0] = 1.0
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_arr))
        L_norm = sp.eye(n_genes, format='csr') - d_inv_sqrt @ A @ d_inv_sqrt

        k_eig = min(4, n_genes - 2)
        eigs = eigsh(L_norm, k=k_eig, which='SM',
                     return_eigenvectors=False,
                     tol=1e-3, maxiter=800)
        eigs = np.sort(np.abs(eigs))

        if len(eigs) < 2:
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        observed = float(eigs[1])
        if not np.isfinite(observed) or observed <= 0:
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        lo = max(observed * (1.0 - window_frac), 0.005)
        hi = min(observed * (1.0 + window_frac), 0.5)
        lo, hi = _clip_syn(lo, hi, "spectral_gap")
        return lo, hi, 0.72

    except Exception:
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ── Probe 5 — Source concentration ───────────────────────────────────────────

def _probe_source_concentration(raw_sparse_mat, lam_eff, n_genes,
                                  window_frac=0.40):
    """
    Probe mean outdegree of genes that have at least one outgoing edge.

    Source concentration = n_edges / n_active_sources.

    High concentration (few, highly-connected hubs) gives FAGCN a clear
    signal hierarchy and prevents the degenerate case where every gene
    acts as a regulator with equal, low connectivity (which collapses
    per-perturbation specificity).

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["source_conc"]
    if raw_sparse_mat is None:
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        src_arr, _, _ = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)

        n_edges = len(src_arr)
        n_sources = len(np.unique(src_arr))

        if n_sources == 0:
            return fb[0], fb[1], FALLBACK_CONFIDENCE

        observed = n_edges / n_sources
        lo = max(observed * (1.0 - window_frac), 2.0)
        hi = observed * (1.0 + window_frac)
        lo, hi = _clip_syn(lo, hi, "source_conc")
        return lo, hi, 0.82

    except Exception:
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ── Probe 6 — Heterophily ─────────────────────────────────────────────────────

def _probe_heterophily(raw_sparse_mat, lam_eff, n_genes,
                        community_labels=None, window=0.10, gene_features=None):
    """
    Probe edge heterophily.

    FAGCN's self-gating mechanism adaptively mixes low-frequency (homophilic)
    and high-frequency (heterophilic) graph signals (Bo et al. 2021).
    Intermediate heterophily is optimal: purely homophilic graphs degrade
    FAGCN to a standard low-pass GCN; purely heterophilic graphs suppress
    the low-frequency component needed for stable training.

    Two measurement modes, MUST match engine.calculate_synthetic_loss:
      * FEATURE-BASED (preferred): mean cosine DISTANCE (1 - cos) between the
        endpoint feature vectors of the top-lambda parent edges. Requires
        gene_features (n_genes x D, unit-norm rows). This is what the loss
        measures, so the probe target and the achieved value are on the same
        scale ([0, 2]). No dependence on community labels.
      * COMMUNITY-BASED (fallback): cross-community edge fraction ([0, 1]).
        Only used if gene_features is None; the loss must then also be run
        with gene_community_labels (not gene_features) to stay on-scale.

    Returns (lo, hi, confidence).
    """
    fb = _SYN_FALLBACK["heterophily"]
    if raw_sparse_mat is None:
        return fb[0], fb[1], FALLBACK_CONFIDENCE

    try:
        src_arr, tgt_arr, _ = _top_lambda_edges(raw_sparse_mat, lam_eff, n_genes)

        # ── Feature-based (preferred) — same metric as the synthetic loss ──
        if gene_features is not None:
            gf = np.asarray(gene_features, dtype=np.float64)
            norms = np.linalg.norm(gf, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            gfn = gf / norms
            cos = np.sum(gfn[src_arr] * gfn[tgt_arr], axis=1)
            observed = float(np.mean(np.clip(1.0 - cos, 0.0, 2.0)))
            lo, hi = observed - window, observed + window
            lo, hi = _clip_syn(lo, hi, "heterophily")
            return lo, hi, 0.78

        # ── Community-based (fallback) ─────────────────────────────────────
        if community_labels is None:
            try:
                import igraph as ig
                edges = list(zip(src_arr.tolist(), tgt_arr.tolist()))
                ig_g = ig.Graph(n=n_genes, edges=edges, directed=False)
                part = ig_g.community_leiden(
                    objective_function='modularity',
                    resolution=0.5, n_iterations=3)
                community_labels = np.array(part.membership, dtype=np.int32)
            except Exception:
                od = np.bincount(src_arr, minlength=n_genes)
                community_labels = np.digitize(od, np.percentile(od[od > 0], [20, 40, 60, 80]))

        comm_src = community_labels[src_arr]
        comm_tgt = community_labels[tgt_arr]
        observed = float(np.mean(comm_src != comm_tgt))

        lo = max(observed - window, 0.05)
        hi = min(observed + window, 0.95)
        lo, hi = _clip_syn(lo, hi, "heterophily")
        return lo, hi, 0.75

    except Exception:
        return fb[0], fb[1], FALLBACK_CONFIDENCE


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_synthetic_summary(utopian_bounds, loss_weights, raw_confidences,
                               lam_eff, spectra_L, n_active, n_tested, n_genes):
    """Print formatted synthetic diagnostic summary to stdout."""
    sep = "─" * 72
    print(f"\n  {sep}")
    print(f"  SYNTHETIC DIAGNOSTIC SUMMARY  (λ_center={lam_eff:.2f}, L={spectra_L})")
    print(f"  {sep}")
    print(f"  {'Parameter':20s}  {'Target window':20s}  {'Conf':6s}  {'Weight':7s}  Description")
    print(f"  {sep}")

    display_keys = ["epr_k", "weight_entropy", "path_length",
                    "spectral_gap", "source_conc", "heterophily"]
    for k in display_keys:
        if k not in utopian_bounds:
            continue
        lo, hi = utopian_bounds[k]
        conf = raw_confidences.get(k, 0.0)
        wt   = loss_weights.get(k, 0.0)
        desc = _SYN_DESCRIPTIONS.get(k, "")
        print(f"  {k:20s}  [{lo:.4f},  {hi:.4f}]      {conf:.2f}    {wt:6.1f}   {desc}")

    print(f"  {sep}")
    print(f"  Perturbations active  : {n_active} / {n_tested} tested")
    print(f"  Gene universe         : {n_genes:,} genes")
    print(f"  S_max                 : [0.001, 0.300]  — loose (shatter guard only, "
          f"weight=0.0)")
    print(f"  {sep}\n")

    # PROCEED/ABORT logic — synthetic mode passes if EPR@k is non-trivial
    epr_lo = utopian_bounds.get("epr_k", [0, 0])[0]
    if epr_lo < 0.05:
        print("  ⚠ WARNING: EPR@k floor is very low. The parent graph may not have"
              " enough biological precision to guide DASH meaningfully.")
        print("  RECOMMEND: Check that deg_matrix has non-zero entries and that"
              " perturbation genes appear as sources in the parent graph.\n")


# ── Public API ────────────────────────────────────────────────────────────────

def run_synthetic_diagnostics(adata, n_genes, cfg_diagnostics, cfg_input,
                               raw_sparse_mat=None, spectra_L=3,
                               community_labels=None, gene_features=None):
    """
    Full synthetic Phase 0 pipeline.

    Parameters
    ----------
    adata              : AnnData — training expression (metacells or single-cell).
    n_genes            : int — number of genes in the working gene universe.
    cfg_diagnostics    : dict — diagnostics section of fungi_config.yaml.
    cfg_input          : dict — input section of fungi_config.yaml.
    raw_sparse_mat     : scipy.sparse — dense parent GRN (PSGRN output).
    spectra_L          : int — number of FAGCN message-passing layers.
                         Determines the hard upper bound on path_length target.
                         Default 3 (confirmed from alt_spectra.py architecture).
    community_labels   : np.ndarray or None — per-gene Leiden community IDs from
                         Phase 2 SCBER computation. If None, a quick Leiden run
                         is performed on the substrate during Phase 0.

    Returns
    -------
    utopian_bounds : dict — 6 synthetic target windows + loose S_max for
                    build_shatter_config compatibility.
    loss_weights   : dict — confidence-normalised optimisation weights.
    diagnostic_report : dict — full metadata for downstream use and logging.
    deg_matrix_csr : scipy.sparse.csr_matrix — DEG matrix (reused in Phase 3/5
                    to avoid rebuild, the primary advantage of returning it here).
    """
    pert_col    = cfg_input["perturbation_column"]
    ctrl_label  = cfg_input["control_label"]
    is_metacell = cfg_input.get("is_metacell", False)
    mc_pool     = cfg_input.get("metacell_pooling_factor", None)

    bc       = cfg_diagnostics["bound_constraints"]
    w_floor  = cfg_diagnostics["weight_floor"]
    w_ceiling = cfg_diagnostics["weight_ceiling"]
    n_jobs   = cfg_diagnostics.get("n_jobs", 6)
    max_perts = cfg_diagnostics.get("max_perts_for_de", 500)

    print(f"\n  ═══ Synthetic Phase 0: SPECTRA-targeted diagnostics (L={spectra_L}) ═══")

    # ── Shared: DEG matrix + λ_eff ─────────────────────────────────────────────
    (impact_array, pert_labels, sample_weights,
     deg_matrix_csr, lfc_matrix, valid_cq, name_to_idx,
     n_tested) = build_impact_array(
        adata, pert_col, ctrl_label,
        de_method       = cfg_diagnostics["de_method"],
        pval_threshold  = cfg_diagnostics["de_pval_threshold"],
        lfc_threshold   = cfg_diagnostics["de_lfc_threshold"],
        n_jobs          = n_jobs,
        max_perts_for_de = max_perts,
        is_metacell     = is_metacell,
        metacell_pooling_factor = mc_pool,
    )

    n_active = len(impact_array)

    lam_eff, lam_q25, lam_q75, erank_diag = _compute_lam_eff(
        deg_matrix_csr, lfc_matrix, valid_cq, name_to_idx,
        n_active, n_tested, n_genes,
        sample_weights=sample_weights)

    print(f"\n  λ_center = {lam_eff:.2f} edges/gene  "
          f"[λ_q25={lam_q25:.2f}  λ_q75={lam_q75:.2f}]")

    # Derive perturbed_nodes: gene indices for active perturbation targets
    gene_list = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    perturbed_nodes = np.array(
        [gene_to_idx[g] for g in valid_cq if g in gene_to_idx],
        dtype=np.int32)
    print(f"  Active perturbation sources: {len(perturbed_nodes)}")
    print("\n  Running synthetic probes...")

    # ── Six probes ─────────────────────────────────────────────────────────────
    epr_lo,  epr_hi,  epr_conf  = _probe_epr_k(
        raw_sparse_mat, deg_matrix_csr, perturbed_nodes, lam_eff, n_genes)
    print(f"    [1/6] EPR@k          → [{epr_lo:.4f}, {epr_hi:.4f}]  "
          f"(conf={epr_conf:.2f})")

    ent_lo,  ent_hi,  ent_conf  = _probe_weight_entropy(
        raw_sparse_mat, lam_eff, n_genes)
    print(f"    [2/6] Weight entropy → [{ent_lo:.4f}, {ent_hi:.4f}]  "
          f"(conf={ent_conf:.2f})")

    path_lo, path_hi, path_conf = _probe_path_length(
        raw_sparse_mat, deg_matrix_csr, perturbed_nodes, lam_eff, n_genes,
        L=spectra_L)
    print(f"    [3/6] Path length    → [{path_lo:.4f}, {path_hi:.4f}]  "
          f"(conf={path_conf:.2f})  [hard ceiling=L={spectra_L}]")

    gap_lo,  gap_hi,  gap_conf  = _probe_spectral_gap(
        raw_sparse_mat, lam_eff, n_genes)
    print(f"    [4/6] Spectral gap   → [{gap_lo:.5f}, {gap_hi:.5f}]  "
          f"(conf={gap_conf:.2f})")

    conc_lo, conc_hi, conc_conf = _probe_source_concentration(
        raw_sparse_mat, lam_eff, n_genes)
    print(f"    [5/6] Source conc.   → [{conc_lo:.2f}, {conc_hi:.2f}]  "
          f"(conf={conc_conf:.2f})")

    het_lo,  het_hi,  het_conf  = _probe_heterophily(
        raw_sparse_mat, lam_eff, n_genes,
        community_labels=community_labels,
        window=float(cfg_diagnostics.get('heterophily_window', 0.10)),
        gene_features=gene_features)
    print(f"    [6/6] Heterophily    → [{het_lo:.4f}, {het_hi:.4f}]  "
          f"(conf={het_conf:.2f})")

    # ── Assemble bounds ─────────────────────────────────────────────────────────
    # The "S_max" entry is a loose passthrough required by build_shatter_config().
    # It is given zero loss weight so it never contributes to optimisation.
    utopian_bounds = {
        "epr_k":          [_safe_float(epr_lo,  0.10), _safe_float(epr_hi,  1.00)],
        "weight_entropy": [_safe_float(ent_lo,  3.80), _safe_float(ent_hi,  7.00)],
        "path_length":    [_safe_float(path_lo, 1.50), _safe_float(path_hi, float(spectra_L))],
        "spectral_gap":   [_safe_float(gap_lo,  0.05), _safe_float(gap_hi,  0.15)],
        "source_conc":    [_safe_float(conc_lo, 10.0), _safe_float(conc_hi, 60.0)],
        "heterophily":    [_safe_float(het_lo,  0.30), _safe_float(het_hi,  0.70)],
        "S_max":          [0.001, 0.300],   # passthrough for build_shatter_config
    }

    raw_confidences = {
        "epr_k":          _safe_float(epr_conf,  FALLBACK_CONFIDENCE),
        "weight_entropy": _safe_float(ent_conf,  FALLBACK_CONFIDENCE),
        "path_length":    _safe_float(path_conf, FALLBACK_CONFIDENCE),
        "spectral_gap":   _safe_float(gap_conf,  FALLBACK_CONFIDENCE),
        "source_conc":    _safe_float(conc_conf, FALLBACK_CONFIDENCE),
        "heterophily":    _safe_float(het_conf,  FALLBACK_CONFIDENCE),
        "S_max":          0.01,  # near-zero so it gets minimal weight
    }

    loss_weights = _normalize_weights(raw_confidences, w_floor, w_ceiling)
    loss_weights["S_max"] = 0.0  # ensure S_max never affects the DASH loss

    # ── Print summary ───────────────────────────────────────────────────────────
    _print_synthetic_summary(utopian_bounds, loss_weights, raw_confidences,
                              lam_eff, spectra_L, n_active, n_tested, n_genes)

    # ── Diagnostic report (same structure as organic for downstream compatibility)
    diagnostic_report = {
        "mode":                       "synthetic",
        "version":                    "1.0",
        "lam_eff":                    lam_eff,
        "lam_q25":                    lam_q25,
        "lam_q75":                    lam_q75,
        "lam_eff_erank_diagnostic":   erank_diag,
        "substrate_lam":              lam_eff,
        "lam_method":                 "specificity_weighted_deg_count",
        "spectra_L":                  spectra_L,
        "n_active":                   n_active,
        "n_tested":                   n_tested,
        "deg_matrix_nnz":             int(deg_matrix_csr.nnz),
        "is_metacell":                is_metacell,
        "probes_used": {
            "epr_k":          "top_lambda_substrate_outgoing_precision",
            "weight_entropy": "log_scaled_weight_shannon_entropy",
            "path_length":    "bfs_sampled_source_to_deg",
            "spectral_gap":   "arpack_normalised_laplacian_fiedler",
            "source_conc":    "top_lambda_edges_per_source",
            "heterophily":    "feature_cosine_distance_top_lambda",
        },
        "raw_confidences":  {k: _safe_float(v) for k, v in raw_confidences.items()},
        "utopian_bounds":   utopian_bounds,
        "loss_weights":     loss_weights,
        "proceed":          True,
        # Underscore keys for downstream notebook cells (same as organic)
        "_deg_col_sums": np.asarray(
            deg_matrix_csr.sum(axis=0)).ravel().tolist(),
        "_deg_row_sums": np.asarray(
            deg_matrix_csr.sum(axis=1)).ravel().tolist(),
        "_impact_array": impact_array.tolist() if len(impact_array) > 0 else [],
        "_perturbation_labels": pert_labels.tolist() if len(pert_labels) > 0 else [],
        "_name_to_idx":  name_to_idx,
    }

    gc.collect()
    return utopian_bounds, loss_weights, diagnostic_report, deg_matrix_csr