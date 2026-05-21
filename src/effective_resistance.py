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

* The resolution parameter is set to match the topology target Q range.
  Lower resolution → fewer, larger communities → higher Q.
  We target the center of the Q utopian bound: [0.30, 0.70] → center 0.50.
  Resolution ≈ 0.5 typically gives Q ≈ 0.35-0.50 for GRN graphs.
  A brief 3-resolution sweep picks the resolution whose Q is closest
  to 0.5 (the center of the target range).

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

    Resolution parameter is swept over [0.3, 0.5, 0.8] and the value
    closest to Q=0.5 (center of the [0.30, 0.70] utopian range) is used.
    This makes community detection loosely coupled to the topology targets
    without being rigidly dependent on them.

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

    # Try resolutions and pick the one closest to Q=0.5
    resolutions = cfg.get('leiden_resolutions', [0.3, 0.5, 0.8])
    target_q    = float(cfg.get('leiden_target_q', 0.5))
    best_part, best_q_dist = None, float('inf')

    for res in resolutions:
        try:
            # igraph >= 0.10 has community_leiden built-in
            part = G.community_leiden(
                weights='weight',
                objective_function='modularity',
                n_iterations=5,
                resolution_parameter=float(res),
            )
        except AttributeError:
            # Older igraph — fall back to Louvain
            part = G.community_multilevel(weights='weight')

        q = part.modularity
        q_dist = abs(q - target_q)
        if q_dist < best_q_dist:
            best_q_dist = q_dist
            best_part = part
            best_res = res

    membership = np.array(best_part.membership, dtype=np.int32)
    Q = float(best_part.modularity)
    n_comm = len(set(membership.tolist()))

    print(f"    Leiden: resolution={best_res}, Q={Q:.3f}, "
          f"communities={n_comm} (target Q≈{target_q})")

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
# Main public function
# ---------------------------------------------------------------------------

def compute_scber_scores(G_csr, sources, targets, cfg=None):
    """
    Compute Source-Conditioned Bridge ER (SCBER) scores.

    Parameters
    ----------
    G_csr   : scipy.sparse.csr_matrix — pre-filtered candidate graph
    sources : np.ndarray int          — edge source indices (pre-presort order)
    targets : np.ndarray int          — edge target indices
    cfg     : dict from effective_resistance section of fungi_config.yaml

    Config keys
    -----------
    epsilon          : float, default 0.5  — JL approximation quality
    seed             : int,   default 42
    eta_inter        : float, default 0.20 — exponent for inter-module R^η
    leiden_resolutions : list, default [0.3, 0.5, 0.8]
    leiden_target_q  : float, default 0.5 — target Q for resolution selection

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
            'mode': 'no_inter', 'n_communities': n_comm, 'Q_achieved': Q_achieved}

    k = int(np.ceil(24.0 * np.log(max(n, 2)) / (epsilon ** 2)))
    k = max(k, 10); k = min(k, 300)

    print(f"    Computing ER sketch: k={k}, ε={epsilon}")
    L_csc = _build_grounded_laplacian(G_csr, n)
    Z = _build_jl_sketch(L_csc, n, k, seed=seed)
    del L_csc

    er_raw = _er_from_sketch(Z, sources_int, targets_int)
    del Z

    er_norm = _normalize_er(er_raw)

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
        'n_communities':     n_comm,
        'Q_achieved':        Q_achieved,
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
