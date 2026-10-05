"""
FUNGI — Phase 5: Refinement (v3.0)
Mode-symmetric desirability secondary + region-capped archive search.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS REWRITE
═══════════════════════════════════════════════════════════════════════════════
Synthetic mode produced thousands of indistinguishable zero-loss graphs. The v2.0
basin machinery (built for tens of zero-loss inputs) detected 31 basins on pass 1,
re-detected up to 82, ran 3+ hours, and crashed. This turns the second search into
a *constrained frontier-expansion over an existing archive*, exactly as the
project's ChatGPT FUNGI Eval 10 recommends, and makes the secondary objective a
mode-symmetric "yin-yang":

  • ORGANIC  → primary loss = biological windows; once 0-loss is held, the secondary
               pulls toward FAGCN-ideal structure (EPR@k, heterophily, spectral
               gap, weight entropy).
  • SYNTHETIC → primary loss = FAGCN windows; once 0-loss is held, the secondary
               pulls toward biology (EPR@k, out-/in-degree Gini → scale-free α).

The secondary is ONE Derringer–Suich desirability scalar (weighted geometric mean
of window- and maximise-desirabilities). One scalar = one coherent "best" used for
the sampling anchor, the improvement test, basin ranking and champion selection.
(Fixes Eval 10 flaw #3: v2.0 sampled around the densest graph but accepted by a
different score.)

NOT a Gaussian-Process / GP-TuRBO method. In 6-D with cheap evaluations a global GP
is the wrong tool (ChatGPT Eval 1; Gemini Eval 1). The surrogate stays the existing
GMM + density-ratio (TPE) sampler. "Trust region" here is pure orchestration
bookkeeping — a clip-box + expand/shrink counters + a hard region cap — layered on
that sampler. No GP is fit anywhere.

Density-phase architecture:
  1. Score the (subsampled) zero-loss archive UP FRONT → global incumbent. (Eval 10
     flaws #1/#2: no zero-initialised incumbent.)
  2. Subsample-before-cluster (cluster_pool_cap); DBSCAN min_samples ≈ 2·dim, eps
     raised; HARD cap of max_basins regions kept by desirability. (Stops 31→82.)
  3. In-region sampler = GMM + density-ratio TPE, samples CLIPPED to a trust-region
     box around each region's best-DESIRABILITY anchor (not its densest member).
  4. Each round scores a SHORTLIST of densest survivors, not one. (Eval 10 flaw #5.)
  5. Expand/shrink: radius grows on improving rounds, shrinks on flat rounds, region
     terminates below min radius. Replaces fixed per-basin round budgets.
  6. GLOBAL early-stop when the global incumbent stalls; re-detection OFF by
     default. The global incumbent is never demoted by a basin-local update.

Also fixed here, all verified against engine.py / search.py:
  • select_champion / select_diverse_cohort were organic-only (hardcoded
    alpha/Gini/Q/C/rho) and would KeyError in synthetic mode — run_dash_and_score
    discards the synthetic columns from the result row. Diversity is now computed in
    PARAMETER space (mode-agnostic, no topo-column dependency); the champion is
    chosen by desirability.
  • Every internal build_graph_from_params call now threads kernel_flags AND er_eta
    (v2.0 dropped kernel_flags, silently using the full kernel even when ablated).
  • The cheap metric helpers mirror engine.py EXACTLY: weight_entropy = 50-bin log2
    histogram entropy; heterophily = unit-norm cosine distance; spectral_gap = the
    Lanczos which='LA' proxy clip(1-μ₂,0,2); out-degree Gini over ALL genes;
    in-degree Gini over NONZERO targets.
  • Edge list presorted ONCE per density phase and reused (Eval 10: wasted overhead).

Public signatures unchanged; new args are keyword-optional and default to the
legacy organic behaviour, so existing call sites keep working.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

_PROGRESS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "DATA", "FUNGI_outputs", "phase5_refinement", "progress.json")


def _write_progress(**kwargs):
    """Best-effort progress marker for external monitoring during long runs --
    must never raise or alter refinement's actual behavior on failure."""
    try:
        kwargs["_written_at"] = time.time()
        os.makedirs(os.path.dirname(_PROGRESS_PATH), exist_ok=True)
        with open(_PROGRESS_PATH, "w") as f:
            json.dump(kwargs, f)
    except Exception:
        pass

warnings.filterwarnings("ignore", category=RuntimeWarning)

PARAM_COLS = ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"]
_EPS = 1e-9


# ===========================================================================
# EPR@k — Early Precision Ratio at k (preserved verbatim from v2.0)
# ===========================================================================

def compute_epr_at_k(ss, st, deg_matrix_csr, perturbed_nodes):
    """Mean per-source precision at k. EPR@k in [0,1]; 0.0 if no qualifying source."""
    if deg_matrix_csr is None or len(ss) == 0:
        return 0.0
    n_genes = deg_matrix_csr.shape[0]
    od = np.bincount(ss, minlength=n_genes)
    hits_total = 0.0
    sources_counted = 0
    for g in perturbed_nodes:
        if g >= deg_matrix_csr.shape[0]:
            continue
        deg_row = deg_matrix_csr.getrow(g)
        if deg_row.nnz == 0:
            continue
        k_g = int(od[g])
        if k_g == 0:
            continue
        deg_targets = set(deg_row.indices.tolist())
        neighbors = set(st[ss == g].tolist())
        hits_total += len(neighbors & deg_targets) / k_g
        sources_counted += 1
    return float(hits_total / sources_counted) if sources_counted > 0 else 0.0


# ===========================================================================
# Cheap topology metrics — mirror engine.py definitions exactly
# ===========================================================================

def _gini(degrees, filter_zero):
    """
    Gini over a degree array.
      out-degree (engine): filter_zero=False  -> over ALL genes (incl. zeros)
      in-degree  (engine): filter_zero=True   -> over NONZERO targets only
    Matches the closed form used in engine.calculate_utopia_loss.
    """
    v = np.asarray(degrees, dtype=np.float64)
    if filter_zero:
        v = v[v > 0]
    nn = v.size
    if nn < 2:
        return 0.0
    sd = np.sort(v)
    total = sd.sum()
    if total <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, nn + 1) * sd)) / (nn * total)
                 - (nn + 1.0) / nn)


def _assortativity(ss, st, n_genes):
    """
    Degree assortativity. Tries igraph directed assortativity (matches engine);
    falls back to a Pearson proxy (source out-degree vs target in-degree across
    edges). Negative => disassortative (typical for GRNs).
    """
    if len(ss) < 3:
        return 0.0
    try:
        import igraph as ig
        g = ig.Graph(n=int(n_genes),
                     edges=list(zip(ss.tolist(), st.tolist())), directed=True)
        r = g.assortativity_degree(directed=True)
        if np.isfinite(r):
            return float(r)
    except Exception:
        pass
    od = np.bincount(ss, minlength=n_genes).astype(np.float64)
    ind = np.bincount(st, minlength=n_genes).astype(np.float64)
    x, y = od[ss], ind[st]
    if np.std(x) < _EPS or np.std(y) < _EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _weight_entropy(sw):
    """50-bin histogram Shannon entropy (log2) — identical to engine."""
    if len(sw) == 0:
        return 0.0
    try:
        counts, _ = np.histogram(sw, bins=50)
        p = counts.astype(np.float64)
        p /= max(p.sum(), 1)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    except Exception:
        return 0.0


def _heterophily(ss, st, gene_features):
    """Mean edge cosine distance (1 - cos). Assumes unit-norm rows (engine match)."""
    if gene_features is None or len(ss) == 0:
        return None
    try:
        cos = np.sum(gene_features[ss] * gene_features[st], axis=1)
        return float(np.mean(1.0 - cos))
    except Exception:
        return None


def _cheap_spectral_gap(ss, st, n_genes):
    """1 - μ₂ via Lanczos which='LA' on the symmetric normalised adjacency.
    Identical parameters to engine.calculate_synthetic_loss. None on failure."""
    ne = len(ss)
    if ne <= 100:
        return None
    try:
        import scipy.sparse as _sp
        from scipy.sparse.linalg import eigsh
        A = _sp.csr_matrix((np.ones(ne, dtype=np.float32), (ss[:ne], st[:ne])),
                           shape=(n_genes, n_genes))
        A = (A + A.T).multiply(0.5)
        A.eliminate_zeros()
        dv = np.array(A.sum(axis=1)).ravel()
        dv[dv == 0] = 1.0
        Dinv = _sp.diags(1.0 / np.sqrt(dv))
        A_norm = (Dinv @ A @ Dinv).tocsr()
        mu = eigsh(A_norm, k=2, which='LA', return_eigenvectors=False,
                   tol=1e-2, maxiter=300)
        mu = np.sort(mu)[::-1]
        if len(mu) > 1 and np.isfinite(mu[1]):
            return float(np.clip(1.0 - mu[1], 0.0, 2.0))
    except Exception:
        pass
    return None


def _source_conc(ss, n_genes):
    od = np.bincount(ss, minlength=n_genes)
    n_src = int(np.count_nonzero(od))
    return float(len(ss)) / max(n_src, 1)


# exp_011 Item 1c: the synthetic->biology cross-pull now targets the finalized
# LIVE biologic set {alpha, gini_in, S_max, C, rho, reciprocity}. _secondary_metrics
# previously computed none of alpha/S_max/C/reciprocity, so they would have been
# silently dropped from the desirability (the scorer skips any metric not in the
# returned dict). These helpers mirror engine.calculate_utopia_loss exactly.
def _smax_frac(ss, n_genes):
    od = np.bincount(ss, minlength=n_genes)
    return float(od.max()) / n_genes if len(od) else 0.0


def _reciprocity(ss, st, n_genes):
    """Directed link reciprocity (Garlaschelli 2004) -- inline numpy identical to
    engine.calculate_utopia_loss's reciprocity term."""
    ne = len(ss)
    if ne == 0:
        return 0.0
    ss = ss.astype(np.int64); st = st.astype(np.int64)
    codes = ss * n_genes + st
    rev = st * n_genes + ss
    order = np.argsort(codes, kind="stable")
    codes_s = codes[order]
    pos = np.clip(np.searchsorted(codes_s, rev), 0, ne - 1)
    has_rev = (codes_s[pos] == rev) & (ss != st)
    return float(has_rev.sum()) / float(ne)


def _alpha_pl(ss, n_genes):
    """Scale-free exponent via the engine's fast CSN auto-xmin fit (same as the
    biologic primary)."""
    try:
        from engine import fast_auto_xmin_alpha
        od = np.bincount(ss, minlength=n_genes).astype(np.float64)
        return float(fast_auto_xmin_alpha(od, n_genes, cap_frac=0.15,
                                          min_tail=50, fallback_xmin=6))
    except Exception:
        return 1.0


def _clustering(ss, st, n_genes):
    """Directed -> undirected (collapse, sum) transitivity, mirroring engine's C."""
    try:
        import igraph as ig
        if len(ss) <= 100:
            return 0.0
        g = ig.Graph(n=n_genes, edges=list(zip(ss.tolist(), st.tolist())),
                     directed=True, edge_attrs={'weight': np.ones(len(ss)).tolist()})
        gu = g.as_undirected(mode="collapse", combine_edges=dict(weight="sum"))
        c = gu.transitivity_undirected()
        return float(c) if np.isfinite(c) else 0.0
    except Exception:
        return 0.0


def _secondary_metrics(ss, st, sw, n_genes, perturbed_nodes, deg_matrix,
                       gene_features, want, need_spectral):
    """Compute only the metrics in `want` for a rebuilt graph. Always n_edges."""
    out = {"n_edges": int(len(ss))}
    if "epr_k" in want:
        out["epr_k"] = compute_epr_at_k(ss, st, deg_matrix, perturbed_nodes)
    if "gini" in want:
        out["gini"] = _gini(np.bincount(ss, minlength=n_genes), filter_zero=False)
    if "gini_in" in want:
        out["gini_in"] = _gini(np.bincount(st, minlength=n_genes), filter_zero=True)
    if "rho" in want:
        out["rho"] = _assortativity(ss, st, n_genes)
    if "weight_entropy" in want:
        out["weight_entropy"] = _weight_entropy(sw)
    if "source_conc" in want:
        out["source_conc"] = _source_conc(ss, n_genes)
    if "heterophily" in want:
        h = _heterophily(ss, st, gene_features)
        if h is not None:
            out["heterophily"] = h
    if "spectral_gap" in want and need_spectral:
        g = _cheap_spectral_gap(ss, st, n_genes)
        if g is not None:
            out["spectral_gap"] = g
    # obj_006.1: source->DEG effective resistance (oversquashing sweet-spot). Reuses
    # the SCBER JL-sketch on the CANDIDATE graph, restricted to (pert-source, DEG)
    # pairs. Needs deg_matrix; skipped gracefully if unavailable (nan not stored ->
    # SecondaryObjective.score drops the metric rather than tanking the geo-mean).
    if "eff_resist" in want and deg_matrix is not None:
        try:
            from effective_resistance import source_deg_effective_resistance
            er = source_deg_effective_resistance(
                ss, st, n_genes, perturbed_nodes, deg_matrix, k=64, seed=42, reduce="mean")
            if er == er:  # exclude nan
                out["eff_resist"] = float(er)
        except Exception:
            pass
    # exp_011 Item 1c: finalized live biologic metrics (synthetic->biology pull)
    if "alpha" in want:
        out["alpha"] = _alpha_pl(ss, n_genes)
    if "S_max" in want:
        out["S_max"] = _smax_frac(ss, n_genes)
    if "C" in want:
        out["C"] = _clustering(ss, st, n_genes)
    if "reciprocity" in want:
        out["reciprocity"] = _reciprocity(ss, st, n_genes)
    return out


# ===========================================================================
# Secondary objective — mode-symmetric desirability (Derringer–Suich)
# ===========================================================================

def _default_secondary_cfg(mode):
    # density is NOT a desirability pillar (weight 0). It was a proxy for DEG
    # reach, now handled directly by the path-length gate; density_floor_edges
    # still applies as a hard sanity guard (a collapsed graph scores 0).
    if mode == "synthetic":
        # pull 0-loss FAGCN graphs toward biology. exp_011 Item 1: the cross-pull
        # is now the FINALIZED LIVE biologic set {alpha, gini_in, S_max, C, rho,
        # reciprocity} (gini_out + Q DISABLED -> EXCLUDED per the exp_010 lock),
        # plus epr_k kept as the DEG-precision graph-utility anchor. Windows here
        # are FALLBACKS -- PROBE-OVERRIDDEN at runtime by re-running the
        # diagnostics_v2 biologic probes (+ reciprocity prior) on the dense parent.
        return {
            "metrics": ["epr_k", "alpha", "gini_in", "S_max", "C", "rho", "reciprocity"],
            "windows": {"alpha": [2.22, 2.50], "gini_in": [0.55, 0.69],
                        "S_max": [0.089, 0.18], "C": [0.097, 0.157],
                        "rho": [-0.31, 0.0], "reciprocity": [0.02, 0.12]},
            "weights": {"epr_k": 1.5, "alpha": 0.8, "gini_in": 0.6, "S_max": 0.6,
                        "C": 0.6, "rho": 0.4, "reciprocity": 0.6, "density": 0.0},
            "compute_spectral_on": "never",
        }
    # biologic (obj_006.1 REFIT): pull 0-loss biological graphs toward the two
    # oversquashing-grounded, theory-certain-direction targets ONLY.
    #   epr_k     = MAXIMIZE (reachability; _MAX_METRIC -> higher-is-better desirability,
    #               no floor = anti-laziness).
    #   eff_resist= SWEET-SPOT window (source->DEG effective resistance; lower relieves
    #               oversquashing (Di Giovanni 2023), too-low = densified/oversmoothed).
    # REMOVED (mis-grounded, session-18 lit pass): heterophily (FAGCN frequency-ADAPTIVE),
    #   weight_entropy + source_conc (unsupported), spectral_gap (superseded by eff_resist).
    # eff_resist window is a SUBSTRATE-CALIBRATED placeholder; PROBE-OVERRIDDEN at run time
    # via secondary_windows (same mechanism the old FAGCN windows used). Kept in sync with
    # fungi_config.yaml biologic_to_synthetic.
    return {
        "metrics": ["epr_k", "eff_resist"],
        "windows": {"eff_resist": [0.0, 0.0]},
        "weights": {"epr_k": 1.0, "eff_resist": 1.0, "density": 0.0},
        "compute_spectral_on": "never",
    }


class SecondaryObjective:
    """score(metrics, density_ref) -> desirability in [0,1]; 0 below density floor."""

    _MAX_METRICS = {"epr_k"}
    _FLOOR_METRICS = {"weight_entropy"}

    def __init__(self, mode, full_cfg, probe_windows=None):
        """
        probe_windows: dict {metric_name: [lo, hi]} from re-running the OTHER
        domain's probes on the dense parent graph (target-specific). When present
        for a metric, it OVERRIDES the config/literature window for that metric —
        the secondary pull then aims at FUNGI's own probe-derived ideal, not a
        generic one. Config windows remain a fallback if a probe is missing.
        """
        full_cfg = full_cfg or {}
        # exp_011 Item 6: canonical config keys are biologic_to_synthetic /
        # synthetic_to_biologic; the legacy organic_* keys are accepted as aliases
        # so existing configs/shards don't break.
        key = ("synthetic_to_biologic" if mode == "synthetic"
               else "biologic_to_synthetic")
        legacy = ("synthetic_to_organic" if mode == "synthetic"
                  else "organic_to_synthetic")
        d = _default_secondary_cfg(mode)
        sub = full_cfg.get(key) or full_cfg.get(legacy) or d
        self.mode = mode
        self.metrics = list(sub.get("metrics", d["metrics"]))
        self.windows = dict(sub.get("windows", d["windows"]))
        if probe_windows:
            win_metrics = set(self.metrics) - self._MAX_METRICS
            for k, v in probe_windows.items():
                if v is not None and k in win_metrics:
                    self.windows[k] = list(v)
        self.weights = dict(sub.get("weights", d["weights"]))
        self.compute_spectral_on = sub.get("compute_spectral_on",
                                           d["compute_spectral_on"])
        self.density_floor = int(full_cfg.get("density_floor_edges", 50_000))
        self.density_ref_quantile = float(full_cfg.get("density_ref_quantile", 0.95))
        self.metric_names = set(self.metrics)

    @property
    def needs_spectral_in_score(self):
        return self.compute_spectral_on == "all"

    @staticmethod
    def _window_d(v, lo, hi):
        if lo <= v <= hi:
            return 1.0
        width = max(hi - lo, 1e-6)
        dist = (lo - v) if v < lo else (v - hi)
        return float(np.exp(-0.5 * (dist / (0.5 * width)) ** 2))

    @staticmethod
    def _max_d(v, ref):
        return float(np.clip(v / max(ref, _EPS), 0.0, 1.0))

    @staticmethod
    def _floor_d(v, floor):
        return float(np.clip(v / max(floor, _EPS), 0.0, 1.0))

    def score(self, metrics, density_ref):
        n_edges = int(metrics.get("n_edges", 0))
        if n_edges < self.density_floor:
            return 0.0
        d_list, w_list = [], []
        for name in self.metrics:
            if name not in metrics:
                continue
            v = float(metrics[name])
            w = float(self.weights.get(name, 1.0))
            if name in self._MAX_METRICS:
                d = self._max_d(v, 1.0)
            elif name in self._FLOOR_METRICS:
                lo = self.windows.get(name, [1.0, 99.0])[0]
                d = self._floor_d(v, lo)
            else:
                win = self.windows.get(name)
                if win is None:
                    continue
                d = self._window_d(v, win[0], win[1])
            d_list.append(max(d, 1e-6))
            w_list.append(w)
        w_dens = float(self.weights.get("density", 0.5))
        if w_dens > 0:
            d_list.append(max(self._max_d(n_edges, density_ref), 1e-6))
            w_list.append(w_dens)
        if not d_list:
            return 0.0
        w_arr = np.asarray(w_list, dtype=np.float64)
        d_arr = np.asarray(d_list, dtype=np.float64)
        return float(np.exp(np.sum(w_arr * np.log(d_arr)) / np.sum(w_arr)))


def _pareto_score(epr, n_edges, max_edges, alpha=0.7):
    """Backward-compat shim (unused internally)."""
    e = float(np.clip(epr, 0.0, 1.0))
    de = float(np.clip(n_edges / max(max_edges, 1), 0.0, 1.0))
    return (e ** alpha) * (de ** (1.0 - alpha))


# ===========================================================================
# Online surrogate — learn (params -> desirability) in real time, screen samples
# ===========================================================================

class _OnlineSurrogate:
    """
    A cheap RandomForest that learns the secondary-desirability surface over the
    6-D hyperparameter space from EVERY scored point (winners and losers) as the
    refinement accumulates them, then pre-screens GMM proposals so evaluations
    concentrate where the surface looks promising. This is the "search becomes
    aware and learns" layer: it is refit each sweep on the growing archive.

    Deliberately NOT a Gaussian Process — an RF on a few hundred 6-D points fits
    in milliseconds, has no line-search failure modes on flat plateaus, and is the
    right complexity for cheap low-D evaluations. Losers (desirability 0: shattered
    graphs and zero-loss graphs that scored poorly) are real signal: they teach the
    RF which corners to avoid, which a winners-only GMM cannot.
    """

    def __init__(self, enabled, min_points, oversample, lower, upper, seed=42):
        self.enabled = bool(enabled)
        self.min_points = int(min_points)
        self.oversample = max(int(oversample), 1)
        self.lower = lower
        self.upper = upper
        self.X = []
        self.y = []
        self.model = None
        self._seed = seed
        self._dirty = False

    def add(self, params, desirability):
        self.X.append(np.asarray(params, dtype=np.float64))
        self.y.append(float(desirability))
        self._dirty = True

    def add_many(self, param_rows, desirabilities):
        for p, d in zip(param_rows, desirabilities):
            self.add(p, d)

    def n(self):
        return len(self.y)

    def fit(self):
        if not (self.enabled and self._dirty and self.n() >= self.min_points):
            return
        try:
            from sklearn.ensemble import RandomForestRegressor
            X = np.vstack(self.X)
            y = np.asarray(self.y, dtype=np.float64)
            m = RandomForestRegressor(
                n_estimators=120, max_depth=None, min_samples_leaf=3,
                n_jobs=-1, random_state=self._seed)
            m.fit(X, y)
            self.model = m
            self._dirty = False
        except Exception:
            self.model = None

    def screen(self, candidates, k):
        """Return the k candidates with highest predicted desirability.
        Falls back to candidates[:k] before the model is trained."""
        if self.model is None or len(candidates) <= k:
            return candidates[:k]
        try:
            pred = self.model.predict(candidates)
            return candidates[np.argsort(pred)[-k:]]
        except Exception:
            return candidates[:k]


# ===========================================================================
# Evaluator plumbing — presort once, rebuild + score
# ===========================================================================

def _evaluator_context(evaluator, mode, gene_features, kernel_flags, spectra_L):
    ctx = {
        "mode": mode or getattr(evaluator, "mode", "biologic"),
        "gene_features": (gene_features if gene_features is not None
                          else getattr(evaluator, "gene_features", None)),
        "kernel_flags": (kernel_flags if kernel_flags is not None
                         else getattr(evaluator, "kernel_flags", None)),
        "spectra_L": (spectra_L if spectra_L is not None
                      else getattr(evaluator, "spectra_L", 3)),
        "n_genes": evaluator.n_genes,
        "extra_hp_names": tuple(getattr(evaluator, "extra_hp_names", ()) or ()),
        "perturbed": evaluator._perturbed_nodes,
        "shatter_cfg": evaluator._shatter_cfg,
        "per_gene_kappa": evaluator._per_gene_kappa,
        "source_pert_impact": evaluator._source_pert_impact,
        "md_gate": getattr(evaluator, "md_gate", None),
        "er_scores": getattr(evaluator, "er_scores", None),
        "er_eta": float(getattr(evaluator, "er_eta", 0.3)),
        "inter_mask": getattr(evaluator, "inter_mask", None),
        "intra_mask": getattr(evaluator, "intra_mask", None),
        "chi_prior": getattr(evaluator, "chi_prior", None),
        "rho_prior": getattr(evaluator, "rho_prior", None),
        "chi_t_prior": getattr(evaluator, "chi_t_prior", None),
        "rdf_prior": getattr(evaluator, "rdf_prior", None),
    }
    try:
        from engine import build_graph_from_params
        # evaluator.Ws/Wqs/Ds/srcs/tgts are already presorted by
        # SearchEvaluator.__init__ -- use directly, don't re-presort (a
        # redundant presort_edges call here used to be able to land on a
        # different Ne-truncated candidate pool than init_gpu_context's;
        # presort_edges is now idempotent, but there's no reason to pay
        # the O(N log N) cost again either).
        (ctx["Ws"], ctx["Wqs"], ctx["Ds"], ctx["srcs"], ctx["tgts"]) = \
            (evaluator.Ws, evaluator.Wqs, evaluator.Ds,
             evaluator.srcs, evaluator.tgts)
        ctx["_build"] = build_graph_from_params
        ctx["ok"] = True
    except Exception:
        ctx["ok"] = False
    return ctx


def _build_graph(ctx, params):
    return ctx["_build"](
        tuple(np.asarray(params, dtype=float)),
        ctx["Ws"], ctx["Wqs"], ctx["Ds"], ctx["srcs"], ctx["tgts"],
        ctx["n_genes"], ctx["perturbed"], ctx["shatter_cfg"],
        ctx["per_gene_kappa"], ctx["source_pert_impact"],
        md_gate=ctx["md_gate"], er_scores=ctx["er_scores"], er_eta=ctx["er_eta"],
        inter_mask=ctx["inter_mask"], intra_mask=ctx["intra_mask"],
        chi_prior=ctx["chi_prior"],
        rho_prior=ctx["rho_prior"], chi_t_prior=ctx["chi_t_prior"],
        rdf_prior=ctx["rdf_prior"], kernel_flags=ctx["kernel_flags"],
        extra_hp_names=ctx.get("extra_hp_names", ()))


def _rebuild_and_score(ctx, params, obj, deg_matrix, density_ref, need_spectral):
    """-> (desirability, metrics, (ss,st,sw)). Safe: returns (0.0,{},None) on error."""
    if not ctx.get("ok"):
        return 0.0, {}, None
    try:
        ss, st, sw = _build_graph(ctx, params)
        metrics = _secondary_metrics(ss, st, sw, ctx["n_genes"], ctx["perturbed"],
                                     deg_matrix, ctx["gene_features"],
                                     obj.metric_names, need_spectral)
        return obj.score(metrics, density_ref), metrics, (ss, st, sw)
    except Exception:
        return 0.0, {}, None


# ===========================================================================
# Mode-aware topo map (used ONLY for the non-zero-loss margin fallback)
# ===========================================================================

def _topo_map(mode):
    if mode == "synthetic":
        # synthetic metrics are not in the result df; margin fallback is rarely
        # reached in synthetic mode (it saturates 0-loss). Use what's available.
        return [("epr_k", "epr_k"), ("spectral_gap", "spectral_gap"),
                ("heterophily", "heterophily")]
    # exp_011 Item 4: biologic margin fallback = finalized LIVE set. gini_out (Gini)
    # dropped (disabled in the exp_010 lock); reciprocity added. (Used only to rank
    # graphs when NO zero-loss graph exists; the df carries 'reciprocity' since the
    # search worker now surfaces it.)
    return [("alpha", "alpha"), ("gini_in", "gini_in"), ("S_max", "S_max"),
            ("C", "C"), ("rho", "rho"), ("reciprocity", "reciprocity")]


# ===========================================================================
# Pre-round generators (preserved from v2.0)
# ===========================================================================

def _generate_boundary_probe_params(lower, upper, n_probes=200, seed=99):
    rng = np.random.default_rng(seed)
    hp_range = upper - lower
    n_dim = len(lower)  # was hardcoded 6 -- broke once nu became the 7th HP
    probes = np.empty((n_probes, n_dim), dtype=np.float64)
    third = n_probes // 3
    for i in range(n_probes):
        p = lower + rng.random(n_dim) * hp_range
        if i < third:
            p[0] = upper[0] - 0.2 * hp_range[0]
            p[4] = lower[4] + 0.15 * hp_range[4]
        elif i < 2 * third:
            p[2] = lower[2] + 0.15 * hp_range[2]
            p[4] = lower[4] + 0.20 * hp_range[4]
        else:
            corner = rng.integers(0, 2, size=n_dim)
            p = np.where(corner == 0, lower + 0.10 * hp_range, upper - 0.10 * hp_range)
        probes[i] = np.clip(p, lower, upper)
    return probes


def _generate_delta_kcore_grid(lower, upper, df_phase3, n_points=196):
    df_viable = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        return None
    top_n = max(int(len(df_viable) * 0.05), 10)
    elite = df_viable.nsmallest(top_n, "utopia_loss")
    beta_fix = float(elite["beta"].median())
    kappa_fix = float(elite["kappa"].median())
    lam_fix = float(elite["lambda"].median())
    psi_fix = float(elite["psi"].median()) if "psi" in elite.columns else 0.5
    nu_fix = float(elite["nu"].median()) if "nu" in elite.columns else 0.0
    m_intra_fix = float(elite["m_intra"].median()) if "m_intra" in elite.columns else 1.0
    grid_n = int(np.sqrt(n_points))
    ng2 = grid_n * grid_n
    dv, kv = np.meshgrid(np.linspace(lower[1], upper[1], grid_n),
                         np.linspace(lower[3], upper[3], grid_n))
    cols = [
        np.full(ng2, beta_fix), dv.ravel(),
        np.full(ng2, kappa_fix), kv.ravel(),
        np.full(ng2, lam_fix), np.full(ng2, psi_fix),
        np.full(ng2, nu_fix),
        np.full(ng2, m_intra_fix)]  # base 8 cols
    # exp_008 levers (exp_035): extend the grid to the full search dim (len(lower)) so it vstacks
    # with the n_dim-D boundary probes. Each lever dim fixed at its elite median (df stores _lever{j}
    # via the patched _topology_worker), else the lower bound.
    for j in range(8, len(lower)):
        col = f"_lever{j - 8}"
        val = (float(elite[col].median()) if (col in elite.columns and np.isfinite(elite[col].median()))
               else float(lower[j]))
        cols.append(np.full(ng2, val))
    return np.column_stack(cols)


# ===========================================================================
# DBSCAN basin detection (preserved; called on a SUBSAMPLED pool, sane params)
# ===========================================================================

def _detect_basins(df_zero, param_cols, eps=0.9, min_samples=12,
                   n_fallback_clusters=6):
    """
    DBSCAN basin detection on the (subsampled) zero-loss pool.

    Falls back to KMeans (n_fallback_clusters, default = max_basins) if DBSCAN
    finds zero core clusters -- everything labeled noise under the configured
    eps/min_samples. Without this fallback that degenerate case collapses the
    whole region-capped multi-basin search to exploring a single
    undifferentiated basin (the entire zero-loss pool), silently defeating the
    diversity/cap machinery built around it. This replaces the old standalone
    niching.py Phase (KMeans anchors that were never actually consumed here);
    folding the fallback in directly means basin diversity degrades gracefully
    instead of depending on a separate, disconnected step.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    df_zero = df_zero.copy()
    df_zero["n_edges"] = pd.to_numeric(
        df_zero["n_edges"], errors="coerce").fillna(0).astype(int)
    if len(df_zero) < min_samples:
        return [df_zero]
    X = df_zero[param_cols].values.astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    uniq = [l for l in np.unique(labels) if l >= 0]
    if len(uniq) == 0:
        from sklearn.cluster import KMeans
        k = max(1, min(n_fallback_clusters, len(df_zero) - 1))
        km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
        return [df_zero[km_labels == l].copy() for l in np.unique(km_labels)]
    basins = [df_zero[labels == l].copy() for l in uniq]
    noise = labels == -1
    if noise.any():
        sc = StandardScaler().fit(X)
        cents = np.array([sc.transform(b[param_cols].values.astype(np.float64)).mean(0)
                          for b in basins])
        nX = sc.transform(X[noise])
        for i, nx_ in enumerate(nX):
            j = int(np.argmin(np.linalg.norm(cents - nx_, axis=1)))
            basins[j] = pd.concat([basins[j],
                                   df_zero.iloc[np.where(noise)[0][i]].to_frame().T],
                                  ignore_index=True)
    return basins


# ===========================================================================
# k-distance plot — empirical validation/re-tuning of dbscan_eps
# ===========================================================================

def plot_kdistance_for_eps_tuning(df_zero, param_cols=PARAM_COLS, k=None,
                                  save_path=None):
    """
    Run this once a real local Phase 3 (expansive search) result exists, to
    validate or re-pick dbscan_eps/dbscan_min_samples in fungi_config.yaml.

    The config's dbscan_eps is a dimensional-scaling placeholder (eps_6d *
    sqrt(7/6)), not a value re-validated against real data. Sort each point's
    distance to its k-th nearest neighbor (k = dbscan_min_samples by default)
    in standardized PARAM_COLS space; the "elbow" where the sorted curve bends
    sharply upward is the conventional DBSCAN eps pick (Ester et al. 1996).

    Usage:
        from refinement import plot_kdistance_for_eps_tuning
        dv = df_expansive[df_expansive["is_shattered"] == 0]
        zero = dv[dv["utopia_loss"] <= 1e-6]  # or dv.nsmallest(2000, "utopia_loss")
        plot_kdistance_for_eps_tuning(zero)
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    k = k or 14
    X = df_zero[param_cols].values.astype(np.float64)
    if len(X) < k + 1:
        raise ValueError(f"Need at least {k + 1} points for k={k}, got {len(X)}.")
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kth_dist = np.sort(dists[:, -1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kth_dist)
    ax.set_xlabel("Points sorted by k-distance")
    ax.set_ylabel(f"Distance to {k}th nearest neighbor")
    ax.set_title(f"k-distance plot ({len(param_cols)}D, k={k}) "
                 "-- pick dbscan_eps at the elbow")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved k-distance plot to {save_path}")
    return fig, kth_dist


# ===========================================================================
# Sobol fallback (preserved)
# ===========================================================================

def _sobol_refinement_fallback(df_phase3, lower, upper, evaluator, cfg, verbose):
    from scipy.stats.qmc import Sobol
    n_per = cfg.get("n_samples_per_round", 400)
    dv = df_phase3[df_phase3["is_shattered"] == 0]
    seeds = dv.nsmallest(min(5, len(dv)), "utopia_loss")[PARAM_COLS].values
    hp_range = upper - lower
    n_dim = len(lower)
    rng = np.random.default_rng(42)
    parts = []
    for s in seeds:
        m = int(np.ceil(np.log2(max(n_per, 2))))
        # d must match len(lower) (PARAM_COLS) -- was hardcoded to 6, same
        # latent bug as search.py's generate_sobol_samples.
        smp = Sobol(d=n_dim, scramble=True, seed=int(rng.integers(0, 9999)))
        lo = np.maximum(lower, s - 0.12 * hp_range)
        hi = np.minimum(upper, s + 0.12 * hp_range)
        parts.append(lo + smp.random(n=2 ** m)[:n_per] * (hi - lo))
    samples = np.vstack(parts)
    if verbose:
        print(f"  Refinement (Sobol fallback): {len(samples):,} points")
    return evaluator.evaluate(param_list=samples,
                              chunk_size=cfg.get("chunk_size", 50),
                              desc="  Refinement: Sobol fallback",
                              show_progress=verbose), False


# ===========================================================================
# Champion selection — mode-aware, desirability-ranked
# ===========================================================================

def select_champion(df_all, utopian_bounds, n_genes, mode="biologic",
                    desirability_map=None):
    dv = df_all[df_all["is_shattered"] == 0].copy()
    if len(dv) == 0:
        raise ValueError("No viable graphs found.")
    dv["utopia_loss"] = pd.to_numeric(dv["utopia_loss"], errors="coerce").fillna(999.0)
    dv["n_edges"] = pd.to_numeric(dv["n_edges"], errors="coerce").fillna(0).astype(int)
    zero = dv[dv["utopia_loss"] <= 1e-6].copy()
    if len(zero) > 1:
        if desirability_map:
            sc = np.array([desirability_map.get(int(zero.index[i]), -1.0)
                           for i in range(len(zero))])
            return zero.iloc[int(sc.argmax())]
        return zero.iloc[int(zero["n_edges"].values.argmax())]
    if len(zero) == 1:
        return dv.loc[zero.index[0]]
    margins = []
    for _, row in dv.iterrows():
        mm = float("inf")
        for bk, col in _topo_map(mode):
            if bk not in utopian_bounds or col not in row:
                continue
            lo, hi = utopian_bounds[bk]
            w = max(hi - lo, 1e-6)
            val = row.get(col, (lo + hi) / 2)
            mm = min(mm, min(val - lo, hi - val) / w)
        margins.append(mm if mm != float("inf") else 0.0)
    dv = dv.reset_index(drop=True)
    dv["_margin"] = margins
    return dv.iloc[int(dv["_margin"].values.argmax())]


# ===========================================================================
# Unified per-point scoring (session 5 refinement redesign)
# ===========================================================================
#
# One comparable [0, 1] scalar for both the loss-minimization and
# desirability-pull regimes, so a SINGLE online surrogate (_OnlineSurrogate)
# can train continuously across the whole refinement run regardless of which
# regime a given region currently is in -- previously the surrogate was only
# ever trained inside the density-mode (zero-loss) code path, which made it
# dead weight on any substrate that never reaches exact zero loss (the common
# case once there are several simultaneous hard biological targets). See
# markdowns/claude_code/claude_code_session_5.md Sections 8-9 for the full
# audit and approved redesign this implements.

def _unified_score(loss, secondary_D=None):
    """
    secondary_D (from SecondaryObjective.score(), already in [0,1], capped at
    1.0) is used as-is when available. Otherwise the loss-based fallback,
    1/(1+loss), equals exactly 1.0 at loss=0 -- the same ceiling desirability
    is capped at -- so the two scales meet continuously at the zero-loss
    boundary instead of needing two separately-calibrated surrogates.
    """
    if secondary_D is not None:
        return float(secondary_D)
    return float(1.0 / (1.0 + max(float(loss), 0.0)))


def _score_to_loss(score):
    """Inverse of _unified_score's loss-based branch (score = 1/(1+loss)) --
    display-only convenience so log lines can show the actual loss number a
    human reads naturally, not just the abstract [0,1] unified score. Only
    meaningful for a "loss"-mode region; desirability-mode scores are real
    SecondaryObjective values with no loss to invert."""
    return float(1.0 / max(float(score), 1e-9) - 1.0)


def _row_unified_scores(df):
    """
    Vectorized _unified_score over an evaluated-results dataframe (must have
    utopia_loss/is_shattered columns). Shattered rows score exactly 0.0 (the
    same convention the surrogate's existing loser-training already used) --
    worse than any viable row's loss-based score, which is always > 0.
    """
    losses = pd.to_numeric(df["utopia_loss"], errors="coerce").fillna(999.0).values
    shattered = (df["is_shattered"].values == 1)
    scores = 1.0 / (1.0 + np.clip(losses, 0.0, None))
    scores[shattered] = 0.0
    return scores


# ===========================================================================
# MAIN — trust-region archive search
# ===========================================================================

def run_ml_gmm_refinement(df_phase3, lower, upper, evaluator, refinement_cfg,
                          verbose=True, deg_matrix=None, pareto_alpha=0.3,
                          mode=None, gene_features=None, kernel_flags=None,
                          spectra_L=None, secondary_cfg=None,
                          secondary_windows=None):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    # exp_008 levers (exp_035): the search space is len(lower)-D. Extend the module-level
    # PARAM_COLS to include the _lever{j} columns the evaluator now stores, so every
    # df[PARAM_COLS]->param-vector reconstruction in refinement is full-dim (matches lower/upper
    # and the boundary/grid probes). No-lever runs leave PARAM_COLS at the base 8. Recomputed per
    # call from the run's own df, so it is always consistent with this refinement's search dim.
    global PARAM_COLS
    _base_pc = ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"]
    _lever_cols = sorted([c for c in df_phase3.columns if str(c).startswith("_lever")],
                         key=lambda x: int(str(x)[6:]))
    PARAM_COLS = _base_pc + _lever_cols
    if verbose and _lever_cols:
        print(f"  [exp_035] refinement search dim = {len(PARAM_COLS)} (levers: {_lever_cols})")

    C = refinement_cfg
    eps = float(C.get("density_mode_epsilon", 1e-6))
    top_fraction = float(C.get("top_fraction", 0.05))
    n_gmm = int(C.get("n_gmm_components", 5))
    chunk_size = int(C.get("chunk_size", 50))
    n_boundary_probes = int(C.get("n_boundary_probes", 200))
    run_grid = bool(C.get("run_delta_kcore_grid", True))
    dbscan_eps = float(C.get("dbscan_eps", 0.9))
    dbscan_min_samples = int(C.get("dbscan_min_samples", 12))
    good_q = float(C.get("good_loss_quantile", 0.15))
    bad_q = float(C.get("bad_loss_quantile", 0.70))
    # region / archive (session 5: now governs BOTH loss-mode and
    # desirability-mode regions uniformly -- see module docstring above
    # _unified_score for why a single set of knobs now covers both)
    max_basins = int(C.get("max_basins", 6))
    cluster_pool_cap = int(C.get("cluster_pool_cap", 2000))
    frontier_k = int(C.get("frontier_shortlist_k", 12))
    global_patience = int(C.get("global_improve_patience", 2))
    min_improve = float(C.get("min_fitness_improvement", 0.005))
    redetection_passes = int(C.get("redetection_passes", 0))
    region_samples = int(C.get("region_samples_per_round", 128))
    tr = C.get("trust_region", {}) or {}
    tr_init = float(tr.get("init_radius_frac", 0.25))
    tr_min = float(tr.get("min_radius_frac", 0.02))
    tr_exp = float(tr.get("expand_factor", 1.5))
    tr_shr = float(tr.get("shrink_factor", 0.5))
    tr_succ = int(tr.get("success_streak_to_expand", 2))
    tr_fail = int(tr.get("failure_streak_to_shrink", 2))
    tr_rounds_cap = int(tr.get("max_rounds_per_region", 12))
    # online surrogate (learns params -> unified score from winners AND
    # losers, EVERY round, regardless of region mode -- session 5 fix)
    sg = C.get("surrogate", {}) or {}
    sg_enabled = bool(sg.get("enabled", True))
    sg_min_pts = int(sg.get("min_points", 40))
    sg_oversample = int(sg.get("oversample", 6))
    sg_score_full = bool(sg.get("score_full_round", False))

    mode = mode or getattr(evaluator, "mode", "biologic")
    obj = SecondaryObjective(mode, secondary_cfg, probe_windows=secondary_windows)
    surrogate = _OnlineSurrogate(sg_enabled, sg_min_pts, sg_oversample, lower, upper)
    hp_range = upper - lower

    dv = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(dv) == 0:
        if verbose:
            print("  Refinement: no viable graphs from expansive search. Skipping.")
        return None, True
    dv["utopia_loss"] = pd.to_numeric(dv["utopia_loss"], errors="coerce").fillna(999.0)
    dv["n_edges"] = pd.to_numeric(dv["n_edges"], errors="coerce").fillna(0).astype(int)

    best_phase3 = float(dv["utopia_loss"].min())
    n_zero_phase3 = int((dv["utopia_loss"] <= eps).sum())

    # Built unconditionally (session 5): any region may transition into
    # desirability mode the instant it finds a zero-loss survivor, so the
    # rebuild context can't be deferred behind a single global gate anymore.
    ctx = _evaluator_context(evaluator, mode, gene_features, kernel_flags, spectra_L)
    need_spec = (obj.compute_spectral_on in ("survivors", "all"))
    if verbose and not ctx.get("ok"):
        print("     [warn] could not import engine/search rebuild — "
              "desirability scoring disabled for any region that reaches zero-loss.")

    if verbose:
        print(f"\n  ── Refinement [parallel region search]")
        print(f"     Expansive best loss: {best_phase3:.6f} | "
              f"Zero-loss: {n_zero_phase3:,}")

    t0 = time.time()
    _write_progress(stage="starting", best_loss=best_phase3, n_evals=0, elapsed=0.0)

    # ── Pre-rounds (boundary + δ×k_core grid) ──────────────────────────────
    boundary = _generate_boundary_probe_params(lower, upper, n_probes=n_boundary_probes)
    grid = _generate_delta_kcore_grid(lower, upper, df_phase3) if run_grid else None
    pre_parts = [boundary] + ([grid] if grid is not None else [])
    pre = np.vstack(pre_parts)
    if verbose:
        print(f"\n     [Pre-rounds] {len(pre)} configs (boundary {len(boundary)}"
              + (f", grid {len(grid)}" if grid is not None else "") + ")")
    df_pre = evaluator.evaluate(param_list=pre, chunk_size=chunk_size,
                                desc="  Refinement pre-rounds", show_progress=verbose)
    df_pre["utopia_loss"] = pd.to_numeric(df_pre["utopia_loss"], errors="coerce").fillna(999.0)
    df_pre["n_edges"] = pd.to_numeric(df_pre["n_edges"], errors="coerce").fillna(0).astype(int)
    if verbose:
        nsh = int(df_pre.iloc[:len(boundary)]["is_shattered"].sum())
        print(f"     [Boundary] {nsh}/{len(boundary)} shattered "
              f"({nsh/len(boundary)*100:.0f}%)")
    _write_progress(stage="pre_rounds_done", best_loss=best_phase3,
                    n_evals=len(pre), elapsed=time.time() - t0)

    # ── Dual-KDE good/bad split for the acquisition (cheap, helps avoid shatter)
    train = pd.concat([df_phase3, df_pre], ignore_index=True)
    vl = pd.to_numeric(train.loc[train["is_shattered"] == 0, "utopia_loss"],
                       errors="coerce").dropna()
    thr_good = float(vl.quantile(good_q)) if len(vl) else 0.0
    thr_bad = float(vl.quantile(bad_q)) if len(vl) else 1.0
    bad_df = train[((train["is_shattered"] == 0) &
                    (pd.to_numeric(train["utopia_loss"], errors="coerce") >= thr_bad))
                   | (train["is_shattered"] == 1)].copy()
    gmm_bad = sc_bad = None
    if len(bad_df) >= 2:
        Xb = bad_df[PARAM_COLS].values.astype(np.float64)
        sc_bad = StandardScaler().fit(Xb)
        gmm_bad = GaussianMixture(n_components=max(1, min(n_gmm, len(bad_df))),
                                  covariance_type="full", n_init=3, random_state=42)
        gmm_bad.fit(sc_bad.transform(Xb))

    all_results = df_pre.to_dict("records")
    # _is_pre_round marks rows from the boundary/grid pre-rounds so basin
    # detection can exclude them (see _CLUSTERING_EXCLUDE_PRE_ROUND comment at
    # the region-seeding block below) while everything else -- surrogate
    # training, df_all_viable's own loss minimum, the final report -- still
    # uses them normally; they're real, legitimately-evaluated data.
    dv = dv.copy()
    dv["_is_pre_round"] = False
    df_pre_viable = df_pre[df_pre["is_shattered"] == 0].copy()
    df_pre_viable["_is_pre_round"] = True
    df_all_viable = pd.concat([dv, df_pre_viable], ignore_index=True)
    df_all_viable["utopia_loss"] = pd.to_numeric(
        df_all_viable["utopia_loss"], errors="coerce").fillna(999.0)
    df_all_viable["n_edges"] = pd.to_numeric(
        df_all_viable["n_edges"], errors="coerce").fillna(0).astype(int)

    # Seed the online surrogate from Phase 3 + pre-rounds (winners AND losers)
    # BEFORE region search starts (session 5 fix for root cause #1 -- the
    # surrogate previously never received a single training example outside
    # density mode). Cheap: _row_unified_scores needs no graph rebuild.
    surrogate.add_many(train[PARAM_COLS].values.astype(np.float64),
                       _row_unified_scores(train))
    surrogate.fit()
    if verbose:
        print(f"     [Surrogate] seeded with {surrogate.n():,} points from "
              f"Phase 3 + pre-rounds (trained: {surrogate.model is not None})")

    # ── helpers ─────────────────────────────────────────────────────────────
    def _fit_gmm(df_pool, rank_by="n_edges"):
        d = df_pool.copy()
        d["n_edges"] = pd.to_numeric(d["n_edges"], errors="coerce").fillna(0).astype(int)
        n_el = max(int(len(d) * max(top_fraction, 0.20)), n_gmm + 1)
        if rank_by in d.columns and d[rank_by].notna().any():
            el = d.nlargest(min(n_el, len(d)), rank_by)
        else:
            el = d.nlargest(min(n_el, len(d)), "n_edges")
        X = el[PARAM_COLS].values.astype(np.float64)
        if len(X) < 2:
            rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 31))
            pad = np.clip(X[0:1] + rng.normal(0, 0.01, (max(2, n_gmm + 1) - len(X),
                          X.shape[1])) * hp_range, lower, upper)
            X = np.vstack([X, pad])
        sc = StandardScaler()
        gm = GaussianMixture(n_components=max(1, min(n_gmm, len(X))),
                             covariance_type="full", n_init=5, random_state=42)
        gm.fit(sc.fit_transform(X))
        return gm, sc

    def _sample(gm, sc, n_want, anchor, radius):
        raw, _ = gm.sample(n_want * 6)
        cand = np.clip(sc.inverse_transform(raw), lower, upper)
        if gmm_bad is not None:
            acq = (gm.score_samples(sc.transform(cand))
                   - gmm_bad.score_samples(sc_bad.transform(cand)))
            keep = max(n_want * 2, n_want)  # leave headroom for surrogate screen
            cand = cand[np.argsort(acq)[-keep:]]
        if anchor is not None and radius is not None:
            box_lo = np.maximum(lower, anchor - radius * hp_range)
            box_hi = np.minimum(upper, anchor + radius * hp_range)
            cand = np.clip(cand, box_lo, box_hi)
        # online surrogate: keep the n_want with highest predicted desirability
        cand = surrogate.screen(cand, n_want)
        return cand

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED PARALLEL-REGION SEARCH (session 5 redesign)
    # ─────────────────────────────────────────────────────────────────────────
    # Replaces the old split between a single-point "loss minimization" drill
    # (shrink-only, then 2 short alt-starts on stall) and a separate "density
    # mode" region/GMM/trust-region archive search that only ever activated if
    # Phase 3 found an EXACT zero-loss solution -- which never happens once
    # there are several simultaneous hard biological targets (see
    # markdowns/claude_code/claude_code_session_5.md Section 8 for the full
    # audit of why that gating left the online surrogate untrained in
    # practice). Every region now runs the SAME trust-region search from
    # round 1, in parallel, and independently switches from "loss" to
    # "desirability" scoring the instant it finds its own zero-loss survivor.
    drill_oversample = 4  # generate this multiple, surrogate screens to n_want

    def _sample_local(center, scale, n_want, seed_bump=0):
        """Gaussian perturbation around `center`. Scale is fraction of hp_range.
        Used by loss-mode regions -- GMM-fit-then-sample is the wrong tool for
        drilling toward a single best point (it returns the elite-cloud
        centroid, not the tip); desirability-mode regions use _sample (GMM)
        instead, for the different problem of exploring an already-equally-
        good manifold."""
        rng_local = np.random.default_rng(
            int(time.time() * 1e6) % (2**31) + seed_bump)
        sigma = scale * hp_range
        n_draw = n_want * drill_oversample
        noise = rng_local.normal(0, 1, (n_draw, len(center))) * sigma
        cand = np.clip(center + noise, lower, upper)
        cand = surrogate.screen(cand, n_want)
        return cand

    def _density_ref():
        zp = df_all_viable[df_all_viable["utopia_loss"] <= eps]
        return max(float(zp["n_edges"].quantile(obj.density_ref_quantile))
                  if len(zp) else 1.0, 1.0)

    def _seed_region(idx, b):
        """Build one region's state dict from a basin dataframe `b`."""
        b = b.copy()
        b["utopia_loss"] = pd.to_numeric(b["utopia_loss"], errors="coerce").fillna(999.0)
        b["n_edges"] = pd.to_numeric(b["n_edges"], errors="coerce").fillna(0).astype(int)
        b["_uscore"] = _row_unified_scores(b)
        bi = b["utopia_loss"].idxmin()
        anchor = b.loc[bi, PARAM_COLS].values.astype(np.float64)
        anchor_loss = float(b.loc[bi, "utopia_loss"])
        region_mode = "desirability" if anchor_loss <= eps else "loss"
        rank_by = "n_edges" if region_mode == "desirability" else "_uscore"
        gm, sc = _fit_gmm(b, rank_by=rank_by)
        return {
            "idx": idx, "anchor": anchor, "mode": region_mode,
            "best_score": float(b.loc[bi, "_uscore"]), "metrics": {},
            "best_edges": int(b["n_edges"].max()), "gmm": gm, "scaler": sc,
            "radius": tr_init, "succ": 0, "fail": 0, "rounds": 0, "done": False,
            "df": b,
        }

    def _try_redetection(regions):
        """
        Re-cluster the full grown archive (bigger now than at seeding time --
        every region's rounds have added to df_all_viable) to check whether a
        better-scoring region exists that the original max_basins cap missed.
        Cheap: the surrogate's prediction (no rebuild) decides whether a new
        candidate is worth adding; replaces the worst active/done region if
        so. Mutates `regions` in place. Returns True if a region was added.
        """
        # Same pre-round exclusion as the initial seeding above (and for the
        # same reason: _generate_delta_kcore_grid's fixed-dimension corridor
        # can bridge otherwise-distinct basins) -- the grid rows never leave
        # df_all_viable, so without this they'd keep corrupting every
        # re-detection pass too, not just the first one.
        clustering_pool_rd = df_all_viable[~df_all_viable["_is_pre_round"]]
        cand_pool = clustering_pool_rd.nsmallest(
            min(cluster_pool_cap, len(clustering_pool_rd)), "utopia_loss").copy()
        cand_basins = _detect_basins(cand_pool, PARAM_COLS, eps=dbscan_eps,
                                     min_samples=dbscan_min_samples,
                                     n_fallback_clusters=max_basins)
        worst = min(regions, key=lambda r: r["best_score"], default=None)
        for b in cand_basins:
            b2 = b.copy()
            b2["utopia_loss"] = pd.to_numeric(b2["utopia_loss"], errors="coerce").fillna(999.0)
            bi = b2["utopia_loss"].idxmin()
            cand_anchor = b2.loc[bi, PARAM_COLS].values.astype(np.float64)
            cand_uscore = _unified_score(float(b2.loc[bi, "utopia_loss"]))
            cand_score = (float(surrogate.model.predict(cand_anchor.reshape(1, -1))[0])
                         if surrogate.model is not None else cand_uscore)
            if worst is None or cand_score > worst["best_score"] + min_improve:
                new_region = _seed_region(len(regions), b2)
                if worst is not None:
                    worst["done"] = True
                regions.append(new_region)
                if verbose:
                    _loss_str = (f", anchor loss={float(b2.loc[bi, 'utopia_loss']):.4f}"
                                if new_region["mode"] == "loss" else "")
                    print(f"       [Re-detection] new region {new_region['idx']+1} "
                          f"(mode={new_region['mode']}, predicted score "
                          f"{cand_score:.4f}{_loss_str}) replaces a stalled region")
                return True
        if verbose:
            print("       [Re-detection] no better region found.")
        return False

    # ── Region seeding: best-loss quantile of the accumulated pool, NOT a
    # zero-loss-only filter -- this is what lets the search explore multiple
    # basins in parallel from round 1 regardless of whether any of them ever
    # reaches exact zero loss. ──────────────────────────────────────────────
    #
    # Pre-round rows (_is_pre_round) are EXCLUDED from the clustering input
    # specifically -- confirmed on real production data that mixing them in
    # collapses genuinely distinct basins into one. Root cause:
    # _generate_delta_kcore_grid's 196 points all share the SAME fixed value
    # in 6 of 8 dimensions (only delta/k_core vary), forming a dense corridor
    # that DBSCAN's transitive connectivity uses to bridge separate clusters
    # into one connected blob -- a real basin in delta/k_core-only terms, but
    # not in the full 8-D space the search actually operates in. This was
    # always a latent risk in _generate_delta_kcore_grid's design, but never
    # triggered before session 5: the OLD density-mode code only ever
    # clustered an EXACT-zero-loss pool, which the grid's coarse points
    # essentially never qualified for. Seeding from a best-LOSS quantile
    # (this redesign) newly exposes it, since the grid's points (built from
    # elite medians) usually score well enough to make the cut. Pre-round
    # evaluations remain real data for every other purpose (surrogate
    # training, df_all_viable's own loss minimum) -- only excluded from
    # basin-shape detection. See markdowns/claude_code/claude_code_session_5.md
    # Section 13 for the full diagnosis (reproduced and confirmed on the real
    # saved Phase 3 shards: 6 clean basins with pre-rounds excluded, 1 with
    # them included).
    clustering_pool = df_all_viable[~df_all_viable["_is_pre_round"]]
    seed_pool = clustering_pool.nsmallest(
        min(cluster_pool_cap, len(clustering_pool)), "utopia_loss")
    basins = _detect_basins(seed_pool, PARAM_COLS, eps=dbscan_eps,
                            min_samples=dbscan_min_samples,
                            n_fallback_clusters=max_basins)
    # _detect_basins does NOT itself cap the number of DBSCAN-detected
    # clusters (n_fallback_clusters only bounds its own KMeans fallback) --
    # max_basins is enforced here instead: seed every detected basin cheaply
    # (loss-mode seeding needs no rebuild), then keep only the top max_basins
    # by score. Any basin that happens to already be desirability-mode at
    # seeding time (rare -- needs a zero-loss row already in Phase 3) gets a
    # real single-point rebuild so its score is comparable, not just an
    # arbitrary tie-broken loss pick among its zero-loss rows.
    candidate_regions = [_seed_region(i, b) for i, b in enumerate(basins)]
    for cr in candidate_regions:
        if cr["mode"] == "desirability":
            D, m, _ = _rebuild_and_score(ctx, cr["anchor"], obj, deg_matrix,
                                         _density_ref(), need_spec)
            cr["best_score"], cr["metrics"] = D, m
            surrogate.add(cr["anchor"], D)
    candidate_regions.sort(key=lambda r: r["best_score"], reverse=True)
    regions = candidate_regions[:max_basins]
    for i, r in enumerate(regions):
        r["idx"] = i
    if verbose:
        print(f"\n     [Regions] seeded {len(regions)} region(s) from the best "
              f"{len(seed_pool):,} of {len(clustering_pool):,} non-pre-round viable "
              f"configs ({len(df_all_viable) - len(clustering_pool):,} pre-round rows "
              f"excluded from clustering) | best loss overall: "
              f"{float(df_all_viable['utopia_loss'].min()):.6f}")
        if len(candidate_regions) > len(regions):
            print(f"     [Region cap] keeping top {len(regions)} of "
                  f"{len(candidate_regions)} basins by score "
                  f"(dropped {len(candidate_regions)-len(regions)})")
        for r in regions:
            li = PARAM_COLS.index("lambda")
            _loss_str = f" (loss={_score_to_loss(r['best_score']):.4f})" if r["mode"] == "loss" else ""
            print(f"       Region {r['idx']+1} [{r['mode']}]: score={r['best_score']:.4f}"
                  f"{_loss_str} "
                  f"λ={r['anchor'][li]:.1f} δ={r['anchor'][PARAM_COLS.index('delta')]:.2f} "
                  f"β={r['anchor'][PARAM_COLS.index('beta')]:.2f}")

    global_best_score = max((r["best_score"] for r in regions), default=0.0)
    global_no_improve = 0
    round_num_d = 0
    redetect_left = redetection_passes

    while True:
        active = [r for r in regions if not r["done"]]
        if not active:
            break
        if global_no_improve >= global_patience:
            if redetect_left > 0:
                redetect_left -= 1
                if _try_redetection(regions):
                    global_no_improve = 0
                    continue
            break
        improved_sweep = False
        for r in active:
            if r["rounds"] >= tr_rounds_cap or r["radius"] < tr_min:
                r["done"] = True
                continue
            r["rounds"] += 1
            round_num_d += 1
            seed_bump = r["idx"] * 100_000 + r["rounds"]

            if r["mode"] == "loss":
                samp = _sample_local(r["anchor"], r["radius"], region_samples, seed_bump)
            else:
                samp = _sample(r["gmm"], r["scaler"], region_samples, r["anchor"], r["radius"])

            dfr = evaluator.evaluate(
                param_list=samp, chunk_size=chunk_size,
                desc=f"  Region {r['idx']+1} r{r['rounds']} [{r['mode']}]",
                show_progress=verbose)
            dfr["utopia_loss"] = pd.to_numeric(dfr["utopia_loss"], errors="coerce").fillna(999.0)
            dfr["n_edges"] = pd.to_numeric(dfr["n_edges"], errors="coerce").fillna(0).astype(int)
            dfr["region_idx"] = r["idx"]
            all_results.extend(dfr.to_dict("records"))

            # Surrogate trains on EVERY evaluated point, every region, every
            # mode (session 5 fix for root cause #1) -- cheap, no rebuild.
            surrogate.add_many(dfr[PARAM_COLS].values.astype(np.float64),
                               _row_unified_scores(dfr))

            viable = dfr[dfr["is_shattered"] == 0].copy()
            viable["_is_pre_round"] = False
            zero_rows = viable[viable["utopia_loss"] <= eps]
            if len(viable) > 0:
                df_all_viable = pd.concat([df_all_viable, viable], ignore_index=True)

            just_transitioned = len(zero_rows) > 0 and r["mode"] == "loss"
            if just_transitioned:
                r["mode"] = "desirability"
                if verbose:
                    print(f"       Region {r['idx']+1}: reached zero-loss -- "
                          f"switching to desirability-pull scoring")

            # ---- scoring this round ----
            round_score, round_anchor, round_metrics, round_edges = -1.0, None, {}, 0
            if r["mode"] == "desirability" and len(zero_rows) > 0:
                # Existing frontier-shortlist-with-rebuild pattern, unchanged:
                # score a DIVERSE mix of zero-loss survivors (dense, sparse,
                # random) -- densest graphs are not necessarily the best for
                # biology, sparser graphs with fewer but more targeted edges
                # can have much higher EPR@k.
                density_ref = _density_ref()
                k3 = max(frontier_k // 3, 1)
                _dense  = zero_rows.nlargest(min(k3, len(zero_rows)), "n_edges")
                _sparse = zero_rows.nsmallest(min(k3, len(zero_rows)), "n_edges")
                _rest   = zero_rows.drop(_dense.index.union(_sparse.index))
                _rand   = _rest.sample(min(frontier_k - 2*k3, len(_rest)),
                                       random_state=round_num_d) if len(_rest) > 0 \
                          else pd.DataFrame()
                _shortlist = pd.concat([_dense, _sparse, _rand]).drop_duplicates(
                    subset=PARAM_COLS)
                for _, rr in _shortlist.iterrows():
                    p = rr[PARAM_COLS].values.astype(float)
                    D, m, _ = _rebuild_and_score(ctx, p, obj, deg_matrix,
                                                 density_ref, need_spec)
                    surrogate.add(p, D)
                    if D > round_score:
                        round_score, round_anchor, round_metrics = D, p, m
                round_edges = int(round_metrics.get("n_edges", 0))
            elif len(viable) > 0:
                # Loss-mode scoring: no rebuild -- utopia_loss is already a
                # column on every evaluated row.
                bi = viable["utopia_loss"].idxmin()
                best_round_loss = float(viable.loc[bi, "utopia_loss"])
                round_score = _unified_score(best_round_loss)
                round_anchor = viable.loc[bi, PARAM_COLS].values.astype(float)
                round_edges = int(viable.loc[bi, "n_edges"])
                round_metrics = {"utopia_loss": best_round_loss, "n_edges": round_edges}

            # ---- trust-region expand/shrink, shared by both modes ----
            # just_transitioned always counts as a success: _unified_score
            # compresses small losses very close to 1.0 (e.g. loss=0.005 ->
            # score=0.995), so the exact-zero score (1.0) can fail to clear
            # best_score + min_improve by a hair even though reaching exact
            # zero loss is unambiguously the best possible outcome for that
            # region -- don't let a fixed absolute threshold miss it.
            if just_transitioned or round_score > r["best_score"] + min_improve:
                r["best_score"], r["anchor"], r["metrics"] = round_score, round_anchor, round_metrics
                r["best_edges"] = max(r["best_edges"], round_edges)
                if r["mode"] == "desirability":
                    r["df"] = pd.concat([r["df"], zero_rows], ignore_index=True)
                    refit_rank_by = "n_edges"
                else:
                    viable_scored = viable.copy()
                    viable_scored["_uscore"] = _row_unified_scores(viable_scored)
                    r["df"] = pd.concat([r["df"], viable_scored], ignore_index=True)
                    refit_rank_by = "_uscore"
                r["succ"] += 1
                r["fail"] = 0
                try:
                    r["gmm"], r["scaler"] = _fit_gmm(r["df"], rank_by=refit_rank_by)
                except Exception:
                    pass
                if r["succ"] >= tr_succ:
                    r["radius"] = min(1.0, r["radius"] * tr_exp)
                    r["succ"] = 0
                if round_score > global_best_score + min_improve:
                    global_best_score = round_score
                    improved_sweep = True
                if verbose:
                    _loss_str = (f", loss={_score_to_loss(round_score):.4f}"
                                if r["mode"] == "loss" else "")
                    print(f"       ↑ Region {r['idx']+1} [{r['mode']}]: "
                          f"score={round_score:.4f}{_loss_str} ({round_edges:,} edges, "
                          f"radius={r['radius']:.3f})")
            else:
                r["fail"] += 1
                r["succ"] = 0
                if r["fail"] >= tr_fail:
                    r["radius"] *= tr_shr
                    r["fail"] = 0
                    if r["radius"] < tr_min:
                        r["done"] = True
                if verbose:
                    _loss_str = (f", loss={_score_to_loss(r['best_score']):.4f}"
                                if r["mode"] == "loss" else "")
                    print(f"       Region {r['idx']+1} [{r['mode']}]: no gain "
                          f"(score={r['best_score']:.4f}{_loss_str}, "
                          f"radius={r['radius']:.3f}, "
                          f"viable={len(viable)}/{len(dfr)})")
            import gc; gc.collect()
        global_no_improve = 0 if improved_sweep else global_no_improve + 1
        surrogate.fit()
        _write_progress(stage="region_search", best_score=global_best_score,
                        global_no_improve=global_no_improve,
                        global_patience=global_patience,
                        n_evals=len(all_results), elapsed=time.time() - t0)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_zero_final = int((df_all_viable["utopia_loss"] <= eps).sum())
    final_best_loss = (float(df_all_viable["utopia_loss"].min())
                       if len(df_all_viable) else best_phase3)
    if verbose:
        print(f"\n  ── Refinement complete: {len(all_results):,} evals in {elapsed:.1f}s")
        print(f"     Best loss: {final_best_loss:.6f} | zero-loss: {n_zero_final:,} | "
              f"regions: {len(regions)}")
        for r in regions:
            _loss_str = (f" (loss={_score_to_loss(r['best_score']):.4f})"
                        if r["mode"] == "loss" else "")
            print(f"       Region {r['idx']+1} [{r['mode']}]: score={r['best_score']:.4f}"
                  f"{_loss_str}, best={r['best_edges']:,} edges, rounds={r['rounds']}"
                  f"{' (done)' if r['done'] else ''}")
    _write_progress(stage="complete", best_loss=final_best_loss, zero_loss=n_zero_final,
                    n_evals=len(all_results), elapsed=elapsed)

    return pd.DataFrame(all_results) if all_results else None, False


# ===========================================================================
# Diverse cohort — PARAMETER-space diversity (mode-agnostic), desirability champ
# ===========================================================================

def select_diverse_cohort(df_all, utopian_bounds, n_genes, cohort_size=5,
                          max_loss_multiplier=3.0, epr_scores_map=None,
                          pareto_alpha=0.3, mode=None, evaluator=None,
                          gene_features=None, kernel_flags=None, spectra_L=None,
                          deg_matrix=None, secondary_cfg=None,
                          secondary_windows=None):
    """
    Champion = highest secondary desirability among zero-loss graphs (rebuilt and
    scored when an evaluator is supplied); else densest. Remaining slots = farthest
    points in normalised PARAMETER space (robust for both modes, since synthetic
    topo columns are not present in the result df).
    """
    dv = df_all[df_all["is_shattered"] == 0].copy()
    if len(dv) == 0:
        raise ValueError("No viable graphs found.")
    dv["utopia_loss"] = pd.to_numeric(dv["utopia_loss"], errors="coerce").fillna(999.0)
    dv["n_edges"] = pd.to_numeric(dv["n_edges"], errors="coerce").fillna(0).astype(int)
    dv = dv.reset_index(drop=True)

    mode = mode or getattr(evaluator, "mode", "biologic")
    obj = SecondaryObjective(mode, secondary_cfg, probe_windows=secondary_windows)

    # normalised parameter matrix for diversity
    P = dv[PARAM_COLS].values.astype(np.float64)
    pmin, pmax = P.min(0), P.max(0)
    Pn = (P - pmin) / np.maximum(pmax - pmin, 1e-9)

    champ_loss = float(dv["utopia_loss"].min())
    zero = dv[dv["utopia_loss"] <= 1e-6]

    champion_pos = None
    if len(zero) >= 1 and evaluator is not None:
        ctx = _evaluator_context(evaluator, mode, gene_features, kernel_flags, spectra_L)
        density_ref = max(float(zero["n_edges"].quantile(obj.density_ref_quantile)), 1.0)
        need_spec = obj.compute_spectral_on in ("survivors", "all")
        # score top-density zero-loss candidates (bounded)
        cand = zero.nlargest(min(64, len(zero)), "n_edges")
        best_D = -1.0
        for idx, r in cand.iterrows():
            D, _, _ = _rebuild_and_score(ctx, r[PARAM_COLS].values.astype(float),
                                         obj, deg_matrix, density_ref, need_spec)
            if D > best_D:
                best_D, champion_pos = D, int(idx)
    if champion_pos is None:
        if len(zero) >= 1:
            champion_pos = int(zero["n_edges"].values.argmax()
                               if len(zero) > 1 else zero.index[0])
            # map argmax of zero back to dv position
            if len(zero) > 1:
                champion_pos = int(zero.index[int(zero["n_edges"].values.argmax())])
        else:
            champion_pos = int(dv["utopia_loss"].values.argmin())

    if champ_loss > 1e-6:
        max_loss = champ_loss * max_loss_multiplier
    else:
        max_loss = float(dv["utopia_loss"].quantile(0.15))
    eligible = np.where(dv["utopia_loss"].values <= max_loss)[0].tolist()
    if champion_pos not in eligible:
        eligible.append(champion_pos)

    selected = [champion_pos]
    sel_vecs = [Pn[champion_pos].copy()]
    for _ in range(min(cohort_size - 1, len(eligible) - 1)):
        best_d, best_pos = -1.0, None
        for pos in eligible:
            if pos in selected:
                continue
            md = min(float(np.linalg.norm(Pn[pos] - s)) for s in sel_vecs)
            if md > best_d:
                best_d, best_pos = md, pos
        if best_pos is None:
            break
        selected.append(best_pos)
        sel_vecs.append(Pn[best_pos].copy())

    cohort = dv.iloc[selected].copy().reset_index(drop=True)
    cohort["cohort_rank"] = list(range(1, len(selected) + 1))
    cohort["is_champion"] = [True] + [False] * (len(selected) - 1)
    return cohort