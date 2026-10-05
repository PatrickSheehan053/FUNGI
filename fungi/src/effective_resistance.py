"""
FUNGI — Source-Conditioned Bridge Effective Resistance (SCBER)

The original flat ER implementation (v1, η=0.3) broke all topology targets because
effective resistance is high for inter-community edges by construction — applying
a global ER boost therefore systematically over-selected bridges and suppressed the
intra-module hub→effector edges that build real GRN topology (Gini, Q, C, ρ).

This module implements Source-Conditioned Bridge ER (SCBER), which fixes this by
applying the ER boost selectively:

    R_factor(s→t) = R_st^η_inter   if community(s) ≠ community(t)
                  = 1.0             if community(s) == community(t)

The updated DASH score is:
    ω(s→t) = Wq^β × exp(δ×T̃) × π_s^ψ × G_st × SCBER(s→t)

Design rationale
----------------
* Intra-module edges are left completely untouched (factor = 1.0 exactly).
  DASH's weight+FFL+prior signals already do an excellent job of selecting
  the right intra-module edges. No intervention needed there.

* Inter-module edges get R_st^η_inter, where η_inter ∈ [0.1, 0.4].
  At η_inter=0.20: max spread 1.0/0.05^0.20 = 1.82× between the most and
  least structurally important bridges. Strong enough to meaningfully prefer
  bridges with no alternative paths over redundant cross-module connections,
  while not dominating the weight/FFL signals.

* Community detection uses igraph's built-in Leiden (C++, ~2-5s on VCC-5k).
  The membership array is computed once in Phase 2 and reused throughout.

* Resolution selection REDESIGNED (session 4, 23 June 2026). The original
  design swept [0.3, 0.5, 0.8] and picked whichever resolution's achieved
  modularity was closest to 0.5 -- the center of the "Q (modularity)"
  organic topology target's utopian bound [0.30, 0.70]. That topology
  target has since been removed entirely (FUNGI/src/diagnostics.py) because
  SCBER's own bridge promotion structurally fights modularity, so no probe
  substrate could ever satisfy it -- the target-coupling rationale here is
  now moot. Worse, on G_work's prefilter density (~25-38% edge density, far
  denser than the sparse real-world graphs modularity is normally tuned
  for), the TRIVIAL single-community partition's modularity always equals
  exactly (1 - resolution) -- a mathematical identity, not a measurement --
  which made resolution=0.5 a "perfect" (distance-0) match to target 0.5
  whenever Leiden found nothing better. The selection criterion was
  rewarding the degenerate outcome, not real structure, and on every
  candidate dense graph tested this session (both the SHROOM_1 elastic-net
  and SHROOM_2 PSGRN-SelfTrain substrates) it did exactly that --
  `communities=1` every time, SCBER inert. CPM objective was tried as an
  alternative and is *worse* here, not better: it finds one moderate "core"
  community plus thousands of singleton leftovers (a known CPM-on-dense-
  graphs failure mode), not meaningful multi-community structure.

  Current approach: widen the resolution sweep, reject any resolution
  whose result is degenerate (fewer than `leiden_min_communities`, or one
  community holding more than `leiden_max_dominant_fraction` of all
  nodes), and take the FIRST (lowest, ascending) resolution that clears
  both bars -- this finds genuine, reasonably-distributed structure as
  soon as it emerges, instead of chasing an absolute Q value that has no
  structural meaning on a graph this dense. Empirically (session 4, the
  RPE1 5k SelfTrain/elastic-net G_work, 25% prefilter density): resolution
  1.3 is the first to produce 304 communities with a sane size
  distribution (largest 850/5000 genes, real tail below that) -- nothing
  below it finds more than 2 (lopsided) communities, and resolution >= 2.5
  over-fragments into many near-singletons with negative modularity.

Public API
----------
compute_scber_scores(G_csr, sources, targets, cfg)
    -> er_normalized, inter_mask, er_raw, diagnostics

    er_normalized : np.ndarray float64 shape (n_edges,) — ER scores in [0.05, 1.0]
    inter_mask    : np.ndarray bool    shape (n_edges,) — True for inter-module edges
    er_raw        : np.ndarray float64 shape (n_edges,) — raw ER values (diagnostics)
    diagnostics   : dict
"""

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)


# ---------------------------------------------------------------------------
# Community detection via igraph Leiden
# ---------------------------------------------------------------------------

def _detect_communities(G_csr, n, cfg):
    """
    Run Leiden community detection on the symmetrized parent graph.

    Uses igraph's built-in Leiden (C++, fast). Falls back to igraph's
    multilevel Louvain if leidenalg is not installed and igraph's built-in
    Leiden is not available (older igraph versions).

    Resolution selection (redesigned session 4 -- see module docstring for
    the full rationale): sweep `leiden_resolutions` in ascending order and
    return the FIRST one whose result is non-degenerate -- at least
    `leiden_min_communities` communities, and no single community holding
    more than `leiden_max_dominant_fraction` of all nodes. Ascending order
    means this finds the COARSEST resolution where real structure first
    emerges, not the most fragmented. Falls back to whichever resolution
    in the sweep produced the most communities if none clears both bars
    (better than silently returning a trivial single-community result).

    Returns
    -------
    membership : np.ndarray int, shape (n,) — community index per gene
    Q          : float — achieved modularity
    n_comm     : int   — number of communities found
    """
    import igraph as ig

    A = G_csr.tocsr().astype(np.float64)
    A_sym = (A + A.T) / 2.0
    A_sym.data = np.abs(A_sym.data)
    A_sym.eliminate_zeros()

    coo = A_sym.tocoo()
    rows, cols, data = coo.row.tolist(), coo.col.tolist(), coo.data.tolist()

    # Build undirected weighted igraph
    G = ig.Graph(n=n, edges=list(zip(rows, cols)), directed=False,
                 edge_attrs={'weight': data})
    G.simplify(combine_edges='sum')

    resolutions = sorted(cfg.get('leiden_resolutions', [0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]))
    min_communities = int(cfg.get('leiden_min_communities', 10))
    max_dominant_fraction = float(cfg.get('leiden_max_dominant_fraction', 0.5))

    best_part, best_res = None, None
    fallback_part, fallback_res, fallback_n_comm = None, None, -1

    for res in resolutions:
        try:
            # igraph >= 0.10 has community_leiden built-in
            part = G.community_leiden(
                weights='weight',
                objective_function='modularity',
                n_iterations=5,
                resolution=float(res),
            )
        except AttributeError:
            # Older igraph — fall back to Louvain
            part = G.community_multilevel(weights='weight')

        membership_candidate = np.array(part.membership)
        n_comm_candidate = len(set(membership_candidate.tolist()))
        largest_fraction = float(np.bincount(membership_candidate).max()) / n

        if n_comm_candidate > fallback_n_comm:
            fallback_part, fallback_res, fallback_n_comm = part, res, n_comm_candidate

        if n_comm_candidate >= min_communities and largest_fraction <= max_dominant_fraction:
            best_part, best_res = part, res
            break

    if best_part is None:
        best_part, best_res = fallback_part, fallback_res

    membership = np.array(best_part.membership, dtype=np.int32)
    Q = float(best_part.modularity)
    n_comm = len(set(membership.tolist()))
    largest_fraction = float(np.bincount(membership).max()) / n

    print(f"    Leiden: resolution={best_res}, Q={Q:.3f}, communities={n_comm}, "
          f"largest_community_fraction={largest_fraction:.3f}")

    return membership, Q, n_comm


# ---------------------------------------------------------------------------
# Laplacian + LU factorization (same as before)
# ---------------------------------------------------------------------------

def _build_grounded_laplacian(G_csr, n):
    A = G_csr.tocsr().astype(np.float64)
    A_sym = (A + A.T) / 2.0
    A_sym.data = np.abs(A_sym.data)
    A_sym.eliminate_zeros()

    degree = np.asarray(A_sym.sum(axis=1)).ravel()
    L = sp.diags(degree) - A_sym
    ground = sp.diags(np.full(n, max(degree.mean() * 1e-4, 1e-9)))
    return (L + ground).tocsc()


def _build_jl_sketch(L_csc, n, k, seed=42):
    rng = np.random.default_rng(seed)

    try:
        lu = spla.splu(L_csc)
    except Exception as e:
        print(f"    WARNING: LU failed ({e}). Increasing grounding.")
        bump = sp.diags(np.full(L_csc.shape[0], 1e-3)).tocsc()
        lu = spla.splu(L_csc + bump)

    Z = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        y = rng.choice(np.array([-1.0, 1.0]), size=n) / np.sqrt(float(k))
        y -= y.mean()
        x = lu.solve(y)
        x -= x.mean()
        Z[i] = x
    return Z


def _er_from_sketch(Z, sources, targets):
    diff = Z[:, sources] - Z[:, targets]
    return np.sum(diff ** 2, axis=0)


def _normalize_er(er_raw, clip_percentile=99.5):
    n = len(er_raw)
    if n == 0:
        return np.ones(0, dtype=np.float64)
    upper = np.percentile(er_raw, clip_percentile)
    er_clipped = np.minimum(er_raw, upper)
    order = np.argsort(er_clipped)
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    return (0.05 + 0.95 * rank).astype(np.float64)


# ---------------------------------------------------------------------------
# obj_006.1 — source->DEG effective resistance (oversquashing metric)
# ---------------------------------------------------------------------------

def source_deg_effective_resistance(ss, st, n_genes, perturbed_nodes, deg_matrix_csr,
                                    k=64, seed=42, reduce="mean",
                                    max_pairs=20000, max_sources=200):
    """
    obj_006.1: mean (or high-quantile) effective resistance between perturbed
    SOURCES and their DEGs, measured ON THE CANDIDATE (pruned) GRAPH — the graph
    SPECTRA/FAGCN will actually message-pass over.

    THIN WRAPPER, not a new algorithm (obj_006.1 manuscript): it reuses the exact
    SCBER machinery already in this module — `_build_grounded_laplacian` +
    `_build_jl_sketch` (Spielman-Srivastava Johnson-Lindenstrauss ER estimator) +
    `_er_from_sketch` — restricted to (perturbed-source, DEG) node pairs instead of
    to graph edges. Direction of merit (Di Giovanni et al. 2023, arXiv:2302.06835):
    LOWER source->DEG ER = better information flow (oversquashing relieved); too-low
    ER means the graph has densified/oversmoothed — hence obj_006.1 pairs this with
    a sweet-spot window and the biologic-0-loss gate rather than pure minimisation.

    Parameters
    ----------
    ss, st          : survived-graph edge endpoints (int arrays)
    perturbed_nodes : candidate perturbed-source node indices
    deg_matrix_csr  : csr (n_pert_rows x n_genes); deg_matrix_csr[src,:].indices = DEGs of src
    k               : JL sketch dimension (kept small for a scalar mean; the SCBER
                      Phase-2 ER uses ceil(24 ln n / eps^2), overkill for one scalar)
    reduce          : "mean" | "median" | "p90" over the source->DEG pairs
    max_pairs       : cap on total (source,DEG) pairs scored (subsampled, seeded)
    max_sources     : cap on distinct perturbed sources used (subsampled, seeded)

    Returns a float (nan if no scoreable source->DEG pair or empty graph).
    """
    ne = len(ss)
    if ne == 0 or perturbed_nodes is None or len(perturbed_nodes) == 0 or deg_matrix_csr is None:
        return float("nan")
    n = int(n_genes)
    try:
        A = sp.csr_matrix((np.ones(ne, dtype=np.float64),
                           (np.asarray(ss, np.int64), np.asarray(st, np.int64))), shape=(n, n))
        L_csc = _build_grounded_laplacian(A, n)
        Z = _build_jl_sketch(L_csc, n, int(k), seed=int(seed))
        del L_csc

        n_rows = deg_matrix_csr.shape[0]
        rng = np.random.default_rng(int(seed))
        pn = np.asarray(perturbed_nodes, dtype=np.int64)
        if len(pn) > max_sources:
            pn = rng.choice(pn, size=max_sources, replace=False)
        src_list, tgt_list = [], []
        for src in pn:
            src = int(src)
            if src >= n_rows or src >= n:
                continue
            degs = deg_matrix_csr[src, :].indices
            degs = degs[(degs != src) & (degs < n)]
            if len(degs) == 0:
                continue
            src_list.append(np.full(len(degs), src, dtype=np.int64))
            tgt_list.append(degs.astype(np.int64))
        if not src_list:
            return float("nan")
        src_all = np.concatenate(src_list); tgt_all = np.concatenate(tgt_list)
        if len(src_all) > max_pairs:
            sel = rng.choice(len(src_all), size=max_pairs, replace=False)
            src_all, tgt_all = src_all[sel], tgt_all[sel]
        er = _er_from_sketch(Z, src_all, tgt_all)
        er = er[np.isfinite(er)]
        if len(er) == 0:
            return float("nan")
        if reduce == "median":
            return float(np.median(er))
        if reduce == "p90":
            return float(np.percentile(er, 90))
        return float(np.mean(er))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_scber_scores(G_csr, sources, targets, cfg=None, gene_features=None):
    """
    Compute Source-Conditioned Bridge ER (SCBER) scores, optionally biased
    toward feature-heterophilic bridges (GOKU-style feature-aware ER).

    Parameters
    ----------
    G_csr         : scipy.sparse.csr_matrix — pre-filtered candidate graph
    sources       : np.ndarray int          — edge source indices (pre-presort order)
    targets       : np.ndarray int          — edge target indices
    cfg           : dict from effective_resistance section of fungi_config.yaml
    gene_features : np.ndarray float (n_genes, D) or None — per-gene feature
        vectors (e.g. expression profiles), assumed L2-normalised to unit rows.
        If provided and eta_feat>0, inter-module ER is re-weighted by edge
        feature DISSIMILARITY (1 - cosine), so structurally-critical bridges
        that ALSO connect dissimilar genes are preferred.

    Config keys
    -----------
    epsilon          : float, default 0.5  — JL approximation quality
    seed             : int,   default 42
    eta_inter        : float, default 0.20 — exponent for inter-module R^η (= η_bridge)
    eta_feat         : float, default 0.0  — feature-dissimilarity weight in the
        inter-module bridge blend. 0.0 → pure ER (original behaviour). For
        synthetic/FAGCN graphs, set 0.5–1.0 to prefer heterophilic bridges.
    leiden_resolutions : list, default [0.3, 0.5, 0.8]
    leiden_target_q  : float, default 0.5 — target Q for resolution selection

    Why feature-aware ER (GOKU, arXiv:2506.16110, 2025)
    ---------------------------------------------------
    Plain ER ranks a bridge purely by how irreplaceable it is structurally
    (high ER = no alternative path). It is blind to whether that bridge is a
    genuine heterophilic regulatory edge (TF -> repressed target, expression-
    dissimilar) or a redundant link between two arbitrary modules. Folding edge
    feature dissimilarity into the ER ranking aligns edge SELECTION with the
    property FAGCN exploits at inference time (its high-pass filters act on
    dissimilar — heterophilic — neighbours). So the synthetic heterophily target
    is no longer only MEASURED post-hoc; the kernel is biased toward producing
    it. The effect is strong for synthetic mode and mild for organic (it nudges
    ρ more disassortative); hence eta_feat defaults to 0.0 and is opt-in.

    Returns
    -------
    er_normalized : np.ndarray float64 (n_edges,) — ER scores in [0.05, 1.0]
    inter_mask    : np.ndarray bool    (n_edges,) — True = inter-module edge
    er_raw        : np.ndarray float64 (n_edges,) — raw ER (for diagnostics)
    diagnostics   : dict
    """
    if cfg is None:
        cfg = {}

    epsilon   = float(cfg.get('epsilon',   0.5))
    seed      = int(cfg.get('seed',       42))
    eta_inter = float(cfg.get('eta_inter', cfg.get('eta', 0.20)))
    eta_feat  = float(cfg.get('eta_feat', 0.0))

    n       = G_csr.shape[0]
    n_edges = len(sources)

    if n_edges == 0:
        empty = np.ones(0, dtype=np.float64)
        return empty, np.ones(0, dtype=bool), empty, {}

    # ── Step 1: Community detection ───────────────────────────────────────
    print("    Running Leiden community detection on G_work...")
    try:
        membership, Q_achieved, n_comm = _detect_communities(G_csr, n, cfg)
    except ImportError:
        # igraph not available — fall back to flat ER with gentle η
        print("    WARNING: igraph not available — falling back to flat ER (η=0.05)")
        eta_flat = min(eta_inter, 0.05)
        k = int(np.ceil(24.0 * np.log(max(n, 2)) / (epsilon ** 2)))
        k = max(k, 10); k = min(k, 300)
        L_csc = _build_grounded_laplacian(G_csr, n)
        Z = _build_jl_sketch(L_csc, n, k, seed)
        er_raw = _er_from_sketch(Z, sources.astype(np.int64), targets.astype(np.int64))
        del Z, L_csc
        er_norm = _normalize_er(er_raw)
        inter_mask = np.ones(n_edges, dtype=bool)  # treat all as inter
        return er_norm, inter_mask, er_raw, {
            'mode': 'flat_fallback', 'eta_inter': eta_flat,
            'n_inter': n_edges, 'n_intra': 0}

    # ── Step 2: Build inter-module mask ───────────────────────────────────
    sources_int = sources.astype(np.int64)
    targets_int = targets.astype(np.int64)
    src_comm = membership[sources_int]
    tgt_comm = membership[targets_int]
    inter_mask = (src_comm != tgt_comm)

    n_inter = int(inter_mask.sum())
    n_intra = n_edges - n_inter
    frac_inter = n_inter / max(n_edges, 1)

    print(f"    Inter-module edges: {n_inter:,} ({frac_inter*100:.1f}%)  "
          f"Intra-module: {n_intra:,} ({(1-frac_inter)*100:.1f}%)")

    # ── Step 3: ER sketch (only computed if any inter-module edges exist) ─
    if n_inter == 0:
        print("    No inter-module edges found — SCBER factor = 1.0 everywhere")
        er_norm = np.ones(n_edges, dtype=np.float64)
        er_raw  = np.zeros(n_edges, dtype=np.float64)
        return er_norm, inter_mask, er_raw, {
            'mode': 'no_inter', 'n_communities': n_comm, 'Q_achieved': Q_achieved,
            'membership': membership}

    k = int(np.ceil(24.0 * np.log(max(n, 2)) / (epsilon ** 2)))
    k = max(k, 10); k = min(k, 300)

    print(f"    Computing ER sketch: k={k}, ε={epsilon}")
    L_csc = _build_grounded_laplacian(G_csr, n)
    Z = _build_jl_sketch(L_csc, n, k, seed=seed)
    del L_csc

    er_raw = _er_from_sketch(Z, sources_int, targets_int)
    del Z

    # ── Step 3b: feature-aware bridge re-weighting (GOKU-style, opt-in) ───
    # For inter-module edges, blend in feature DISSIMILARITY so structurally
    # critical bridges that also connect dissimilar genes rank higher:
    #     importance = ER_raw * (1 + eta_feat * (1 - cos_sim(x_s, x_t)))
    # eta_feat=0 -> pure ER (original behaviour). Only inter-module edges are
    # touched; intra-module edges keep factor 1.0 in the kernel regardless.
    feat_dissim_mean = None
    er_for_norm = er_raw.copy()
    if (gene_features is not None and eta_feat > 0.0 and n_inter > 0):
        try:
            gf = np.asarray(gene_features, dtype=np.float64)
            # cosine similarity assuming unit-normalised rows; guard anyway
            norms = np.linalg.norm(gf, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            gfn = gf / norms
            cos = np.sum(gfn[sources_int] * gfn[targets_int], axis=1)
            dissim = np.clip(1.0 - cos, 0.0, 2.0)
            blend = 1.0 + eta_feat * dissim
            # apply only to inter-module edges
            er_for_norm = np.where(inter_mask, er_raw * blend, er_raw)
            feat_dissim_mean = float(dissim[inter_mask].mean())
            print(f"    Feature-aware ER: eta_feat={eta_feat}, "
                  f"mean inter dissimilarity={feat_dissim_mean:.3f}")
        except Exception as e:
            print(f"    WARNING: feature-aware ER skipped ({e}); using pure ER")

    er_norm = _normalize_er(er_for_norm)

    # ── Diagnostics ───────────────────────────────────────────────────────
    inter_er = er_norm[inter_mask]
    intra_er = er_norm[~inter_mask] if n_intra > 0 else np.array([1.0])

    # Expected DASH factor for inter-module edges at eta_inter
    inter_factor_mean = float(np.mean(np.power(inter_er, eta_inter)))
    inter_factor_min  = float(0.05 ** eta_inter)
    inter_factor_max  = 1.0

    print(f"    SCBER at η_inter={eta_inter}:")
    print(f"      Inter-module factor: [{inter_factor_min:.3f}, {inter_factor_max:.3f}] "
          f"(mean {inter_factor_mean:.3f})")
    print(f"      Intra-module factor: 1.000 exactly (untouched)")
    print(f"      High-ER bridges (R>0.9): {int((inter_er > 0.9).sum()):,} edges")

    diagnostics = {
        'mode':              'scber',
        'k_sketches':        k,
        'epsilon':           epsilon,
        'eta_inter':         eta_inter,
        'eta_feat':          eta_feat,
        'feat_dissim_mean':  feat_dissim_mean,
        'n_communities':     n_comm,
        'Q_achieved':        Q_achieved,
        # session 5: expose the partition itself so m_intra/Q (engine.py)
        # can reuse it without re-running Leiden -- per-gene community index,
        # shape (n_genes,), same membership _detect_communities() returned.
        'membership':        membership,
        'n_edges':           n_edges,
        'n_inter':           n_inter,
        'n_intra':           n_intra,
        'frac_inter':        float(frac_inter),
        'inter_er_mean':     float(inter_er.mean()),
        'inter_er_p95':      float(np.percentile(inter_er, 95)),
        'intra_er_mean':     float(intra_er.mean()),
        'inter_factor_mean': inter_factor_mean,
        'inter_factor_min':  inter_factor_min,
        'n_high_er_bridges': int((inter_er > 0.9).sum()),
    }

    return er_norm, inter_mask, er_raw, diagnostics