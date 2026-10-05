"""
FUNGI v9.1 — Search Space Generation and Parallel Execution

Changes from v9.0:
  - SearchEvaluator accepts er_scores (np.ndarray or None) and er_eta (float).
  - er_scores is ray.put() once alongside all other arrays and passed through
    to run_dash_and_score in every _eval_chunk call.
  - er_scores=None is a strict no-op (backward-compatible with saved shards).
"""
import os
import glob
import re
import logging
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.qmc import Sobol

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _choose_gpu_batch_size(ctx, vram_budget_gb=7.0, max_batch=512, min_batch=1):
    """
    Size the GPU batch B from THIS run's actual Ne/n_genes/max_seg_len, not a
    fixed assumption -- Ne varies a lot across datasets and config (prefilter
    density, lambda_max, n_genes), so a hardcoded batch size from any one
    reference scale would be wrong for others.

    Dominant cost is the padded segmented-sort tensor: (B, n_genes,
    max_seg_len), once as float32 (omega) and twice as int64 (local_order,
    global_idx). A 1.5x safety margin covers allocator overhead and the
    smaller omega_batch/order_batch tensors.
    """
    # obj_006 FUNGI-Fast: with the flat two-pass sort the padded (B, n_genes,
    # max_seg_len) tensor no longer exists; per-config VRAM is dominated by a
    # handful of (B, Ne) tensors (omega f32 + idx/src_perm/idx2 i64 + gathered
    # so/to/no). Budget for those instead so the freed VRAM raises B (the padded
    # formula capped B at ~10 by over-budgeting for 97%-padding). Same 1.5x
    # margin. Results are B-invariant (each eval is independent), verified in the
    # obj_006 cross-batch-size determinism check.
    try:
        from engine import _OBJ006_FAST_SORT as _fast
    except Exception:
        _fast = False
    if _fast:
        # VRAM is no longer the constraint (the padded tensor is gone). But a huge B
        # HURTS the Ray path: each batch ray.put's the full (B, Ne) sorted arrays, so
        # big B inflates per-batch IPC (measured: auto B=212 = 1.21x vs B=32 = 1.29x).
        # The GPU sort is now ~1 ms so there's no GPU reason to grow B either. Cap at a
        # modest FAST_MAX_B; the flat sort's win is B-independent (it removes the sort).
        FAST_MAX_B = 48
        bytes_per_config = ctx.Ne * (4 + 8 + 8 + 8 + 4 + 4 + 4) * 1.5  # omega + 3 i64 sort tmps + so/to/no
        budget_bytes = vram_budget_gb * 1e9
        b = int(budget_bytes / max(bytes_per_config, 1.0))
        return int(np.clip(b, min_batch, min(max_batch, FAST_MAX_B)))
    padded_elems = ctx.n_genes * ctx.max_seg_len
    bytes_per_config = (padded_elems * (4 + 8 + 8) + ctx.Ne * (4 + 8)) * 1.5
    budget_bytes = vram_budget_gb * 1e9
    b = int(budget_bytes / max(bytes_per_config, 1.0))
    return int(np.clip(b, min_batch, max_batch))


# obj_006 FUNGI-Fast: chunked Ray dispatch. When True, each batch dispatches
# n_workers tasks (each processing a round-robin slice of the batch) instead of B
# separate per-candidate tasks. Bit-exact (verified: CHUNK==per-row max|delta|=0).
# BUT it is a MEASURED NEGATIVE (0.91x vs per-row): Ray's dynamic per-task scheduler
# load-balances the highly variable per-candidate eval time (fast shatter vs ~700ms
# full eval) better than 6 fixed round-robin chunks, where one slow candidate makes
# its chunk a straggler. So DISABLED by default ("disable, don't delete" -- the code
# + exactness proof stay for reference). Keep per-row dispatch.
_OBJ006_CHUNKED_RAY = False


def _fungi_process_row(row_idx, src_batch, tgt_batch, omega_batch, params, ng,
                       perturbed_nodes, utopian_bounds, loss_weights, shatter_cfg,
                       per_gene_kappa, mode, deg_matrix_csr, gene_community_labels,
                       gene_features, spectra_L, exact_spectral):
    """Exact per-candidate topology+score for one batch row. Byte-for-byte the body
    the original nested `_topology_worker` ran, hoisted to module scope so a Ray
    chunk-worker can loop it. numpy + engine only (no torch) for Windows worker
    safety."""
    import numpy as _np
    from engine import (select_edges, _motif_repair_swap, check_shatter,
                        calculate_utopia_loss, calculate_synthetic_loss,
                        _synthetic_path_gate, _safe)

    so = src_batch[row_idx]
    to = tgt_batch[row_idx]
    no = omega_batch[row_idx]
    Wo = no  # placeholder -- fast_topology=True never reads edge weight

    if len(params) >= 8:
        beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
    elif len(params) >= 7:
        beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
        m_intra = 1.0
    else:
        beta, delta, kappa_base, k_core, lam, psi = params[:6]
        nu = 0.0
        m_intra = 1.0
    k_core_eff = max(k_core, max(5.0, lam * 0.4))
    # exp_008 levers (exp_035): store searched lever values (params[8:]) by position
    # so the champion can be rebuilt exactly (build_graph_from_params needs them).
    # No-lever arms have len(params)==8 -> no _lever* columns. fungi_prune maps
    # _lever{j} back to its canonical lever name.
    _levers = {f'_lever{_k}': float(params[8 + _k]) for _k in range(max(0, len(params) - 8))}

    def _shattered(reason, od_arr, active_n, n_edges):
        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
            'm_intra': m_intra,
            'utopia_loss': 999., 'is_shattered': 1,
            'shatter_reason': reason, 'n_edges': n_edges,
            'active_nodes': active_n, 'alpha': 1., 'Gini': 1.,
            'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1., 'reciprocity': 1.,
            'S_max': _safe((_np.max(od_arr) / ng) if len(od_arr) > 0 else 0),
            'epr_k': 0., 'weight_entropy': 0., 'source_conc': 0.,
            'heterophily': 0., 'spectral_gap': 0., **_levers}

    try:
        param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
        rng = _np.random.default_rng(param_hash)

        surv_s, surv_t, surv_W = select_edges(
            no, Wo, so, to, perturbed_nodes, ng, lam,
            per_gene_kappa, kappa_base)
        surv_s, surv_t, surv_W = _motif_repair_swap(
            surv_s, surv_t, surv_W, no, so, to, ng,
            max_swap_fraction=0.03, rng=rng)

        od = _np.bincount(surv_s, minlength=ng)
        idd = _np.bincount(surv_t, minlength=ng)
        active = int(_np.count_nonzero(od + idd > 0))

        sh, reason, gwcc_fraction = check_shatter(
            surv_s, surv_t, surv_W, od, ng, active, shatter_cfg)
        if sh:
            return _shattered(reason, od, active, len(surv_s))

        if mode == 'synthetic':
            if deg_matrix_csr is not None and perturbed_nodes is not None:
                bad_path = _synthetic_path_gate(
                    surv_s, surv_t, ng, perturbed_nodes,
                    deg_matrix_csr, int(spectra_L),
                    min_reach_frac=shatter_cfg.get('min_reach_frac', 0.5))
                if bad_path:
                    return _shattered('deg_path_exceeds_L', od, active, len(surv_s))
            loss, topo = calculate_synthetic_loss(
                surv_s, surv_t, surv_W, ng, od, active, kappa_base,
                utopian_bounds, loss_weights,
                deg_matrix_csr=deg_matrix_csr,
                perturbed_nodes=perturbed_nodes,
                gene_community_labels=gene_community_labels,
                gene_features=gene_features,
                exact_spectral=exact_spectral)
        else:
            loss, topo = calculate_utopia_loss(
                surv_s, surv_t, surv_W, ng, od, active, kappa_base,
                utopian_bounds, loss_weights, fast_topology=True,
                gene_community_labels=gene_community_labels)

        gf = gwcc_fraction if gwcc_fraction is not None else 0.
        gp = 8. * ((0.45 - gf) / 0.45) ** 2 if gf < 0.45 else 0.
        loss = _safe(_np.sqrt(max(loss ** 2 + gp, 0.)), 999.)

        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
            'm_intra': m_intra,
            'utopia_loss': loss, 'is_shattered': 0, 'shatter_reason': None,
            'n_edges': len(surv_s), 'active_nodes': active,
            'gwcc_fraction': gf,
            'alpha': topo['alpha'], 'Gini': topo['Gini'],
            'rho': topo['rho'], 'C': topo['C'], 'gini_in': topo['gini_in'],
            'Q': topo['Q'],
            'reciprocity': topo.get('reciprocity', _safe(0.)),
            'S_max': topo['S_max'],
            'epr_k':          topo.get('epr_k', _safe(0.)),
            'weight_entropy': topo.get('weight_entropy', _safe(0.)),
            'source_conc':    topo.get('source_conc', _safe(0.)),
            'heterophily':    topo.get('heterophily', _safe(0.)),
            'spectral_gap':   topo.get('spectral_gap', _safe(0.)), **_levers}
    except Exception as e:
        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
            'm_intra': m_intra,
            'utopia_loss': 999., 'is_shattered': 1,
            'shatter_reason': f'crash:{str(e)[:60]}',
            'n_edges': 0, 'active_nodes': 0,
            'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'gini_in': 1.,
            'Q': 1., 'reciprocity': 1., 'S_max': 0.,
            'epr_k': 0., 'weight_entropy': 0., 'source_conc': 0.,
            'heterophily': 0., 'spectral_gap': 0., **_levers}


def compute_lambda_search_bounds(lam_eff, n_genes, static_lo=2.0, static_hi=40.0,
                                  lam_q25=None, lam_q75=None):
    if lam_q25 is not None and lam_q75 is not None:
        lo = max(static_lo, lam_q25)
        hi = min(static_hi, lam_q75)
    else:
        lo = max(static_lo, lam_eff * 0.4)
        hi = min(static_hi, lam_eff * 2.5)
    if hi - lo < 6.0:
        mid = (lo + hi) / 2.0
        lo = max(static_lo, mid - 3.0)
        hi = min(static_hi, mid + 3.0)
    return lo / n_genes, hi / n_genes


def get_static_bounds(n_genes, hp_cfg, lam_eff=None, lam_q25=None, lam_q75=None):
    if hp_cfg.get("lambda_density") is not None:
        ld = hp_cfg["lambda_density"]
    elif lam_eff is not None:
        ld = list(compute_lambda_search_bounds(lam_eff, n_genes,
                                               lam_q25=lam_q25, lam_q75=lam_q75))
    else:
        ld = [2.0 / n_genes, 40.0 / n_genes]

    # v10.0: nu (RDF exponent) added as 7th HP. If absent from config,
    # defaults to [0, 0] which fixes nu=0 (no RDF effect, backward compatible).
    nu_bounds = hp_cfg.get("nu", [0.0, 0.0])
    # session 5: m_intra (8th HP) -- intra-community edge boost. Default
    # [1.0, 1.0] fixes m_intra=1.0 (identity, backward compatible) if absent.
    m_intra_bounds = hp_cfg.get("m_intra", [1.0, 1.0])

    names = ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"]
    lower = np.array([
        hp_cfg["beta"][0],
        hp_cfg["delta"][0],
        hp_cfg["kappa"][0],
        hp_cfg["k_core"][0],
        n_genes * ld[0],
        hp_cfg["psi"][0],
        nu_bounds[0],
        m_intra_bounds[0],
    ])
    upper = np.array([
        hp_cfg["beta"][1],
        hp_cfg["delta"][1],
        hp_cfg["kappa"][1],
        hp_cfg["k_core"][1],
        n_genes * ld[1],
        hp_cfg["psi"][1],
        nu_bounds[1],
        m_intra_bounds[1],
    ])
    # exp_008 levers ported (exp_035): append any searched kernel-firepower levers
    # present in hp_cfg, after m_intra, in canonical order. A lever absent from
    # hp_cfg is simply not a Sobol dim (its no-op default is applied in engine).
    for _nm in ["sigma_scber", "m_inter", "eta_out", "zeta_s", "gamma_md", "tau_indeg", "theta_pa"]:
        if _nm in hp_cfg:
            names.append(_nm)
            lower = np.append(lower, float(hp_cfg[_nm][0]))
            upper = np.append(upper, float(hp_cfg[_nm][1]))
    return lower, upper, names


def generate_sobol_samples(n_genes, n_samples, hp_cfg, seed=42, lam_eff=None,
                            lam_q25=None, lam_q75=None):
    lower, upper, names = get_static_bounds(n_genes, hp_cfg, lam_eff=lam_eff,
                                            lam_q25=lam_q25, lam_q75=lam_q75)
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    ns = 2 ** m
    n_dim = len(lower)
    # d must match len(lower)/len(upper) (PARAM_COLS) -- was hardcoded to 6,
    # which silently broke (shape-mismatch crash) once nu became the 7th HP.
    sampler = Sobol(d=n_dim, scramble=True, seed=seed)
    scaled = lower + sampler.random(n=ns) * (upper - lower)
    scaled = scaled[:n_samples]
    print(f"  Sobol sequence: {n_samples:,} points in {n_dim}D space.")
    print(f"  Parameter space: β[{lower[0]:.1f},{upper[0]:.1f}]  "
          f"δ[{lower[1]:.2f},{upper[1]:.2f}]  "
          f"κ[{lower[2]:.3f},{upper[2]:.3f}]  "
          f"k_core[{lower[3]:.0f},{upper[3]:.0f}]  "
          f"λ[{lower[4]:.1f},{upper[4]:.1f}]  "
          f"ψ[{lower[5]:.1f},{upper[5]:.1f}]  "
          f"ν[{lower[6]:.1f},{upper[6]:.1f}]  "
          f"m_intra[{lower[7]:.2f},{upper[7]:.2f}]")
    return scaled, lower, upper


def presort_edges(W, W_q, D, src, tgt):
    """Sort all edge arrays descending by weight. One-time O(N log N).
    W_q (source-quantile weights) is sorted in the same order as W.

    Idempotent by construction (stable sort on -W, no [::-1] reversal trick):
    re-applying this to already-presorted arrays reproduces the identical
    order. This matters because several call sites (the cohort-rebuild
    notebook cell, refinement.py's _evaluator_context) historically called
    this AGAIN on SearchEvaluator's already-presorted self.Ws/srcs/tgts --
    with a plain `np.argsort(W)[::-1]`, that redundant second call could
    silently re-shuffle tied-weight edges into a different relative order
    (argsort's default/reversed-stable order is not idempotent under
    repeated application), producing a DIFFERENT Ne-truncated candidate
    pool than the one init_gpu_context uses for the SAME hyperparameters.
    Confirmed empirically as the root cause of a champion-graph rho/gini
    discrepancy -- see markdowns/claude_code/claude_code_session_3.md.
    """
    order = np.argsort(-W, kind="stable")
    return (W[order].copy(), W_q[order].copy(),
            D[order].copy(), src[order].copy(), tgt[order].copy())


def _try_ray():
    try:
        import ray
        return ray, True
    except ImportError:
        return None, False


class SearchEvaluator:
    """
    Persistent evaluator: presorts edges and ray.put's all data ONCE.

    v9.1 additions:
      - er_scores  : per-edge ER scores (same presorted order as W).
                     If None, ER factor is 1.0 everywhere (no change to
                     existing behavior).
      - er_eta     : exponent for R_st^η (default 0.3).
    """

    def __init__(self, W_arr, W_q_arr, D_arr, sources_arr, targets_arr,
                    n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                    shatter_cfg, per_gene_kappa, source_pert_impact,
                    n_workers=6, src_dir=None,
                    md_gate=None,
                    er_scores=None, er_eta=0.3, inter_mask=None,
                    intra_mask=None,
                    chi_prior=None, w_causal=None,
                    rho_prior=None, chi_t_prior=None, rdf_prior=None,
                    mode='biologic', deg_matrix_csr=None,
                    gene_community_labels=None,
                    kernel_flags=None, gene_features=None,
                    spectra_L=3, exact_spectral=False,
                    use_ray_pipeline=True):

        self.n_genes = n_genes
        self.n_workers = n_workers
        self.src_dir = src_dir or os.path.dirname(os.path.abspath(__file__))
        self.er_eta = float(er_eta)
        # GPU+Ray pipelined topology (search.py's _evaluate_gpu_ray): only
        # takes effect when a GPU context is active (checked in evaluate()).
        # Verified 6.43x faster than single-process GPU on the real 4096-eval
        # Sobol sweep with bit-identical results -- see
        # markdowns/claude_code/claude_code_session_2.md. Set False to force
        # the single-process GPU path (e.g. if Ray is unavailable/misbehaves
        # on a given machine).
        self.use_ray_pipeline = bool(use_ray_pipeline)

        # Presort once — W_q in same order as W
        (self.Ws, self.Wqs, self.Ds,
         self.srcs, self.tgts) = presort_edges(
            W_arr, W_q_arr, D_arr, sources_arr, targets_arr)

        # If er_scores provided, apply the same presort order so the array
        # stays aligned with Ws/srcs/tgts after presort_edges.
        if er_scores is not None:
            order = np.argsort(W_arr)[::-1]
            self.er_scores = er_scores[order].astype(np.float64)
        else:
            self.er_scores = None

        # inter_mask: presort in the same order as er_scores
        if inter_mask is not None:
            order = np.argsort(W_arr)[::-1]
            self.inter_mask = inter_mask[order].astype(bool)
        else:
            self.inter_mask = None

        # intra_mask (session 5, m_intra's static per-edge factor): same
        # presort discipline as inter_mask -- SCBER's own community partition,
        # reused not recomputed.
        if intra_mask is not None:
            order = np.argsort(W_arr)[::-1]
            self.intra_mask = intra_mask[order].astype(bool)
        else:
            self.intra_mask = None

        # chi_prior: per-gene array, NOT edge-level — no presort needed.
        # chi_prior[s] is looked up by source node index during DASH scoring.
        self.chi_prior = (chi_prior.astype(np.float64)
                          if chi_prior is not None else None)
        self.rho_prior = (rho_prior.astype(np.float64)
                          if rho_prior is not None else None)
        self.chi_t_prior = (chi_t_prior.astype(np.float64)
                            if chi_t_prior is not None else None)
        # rdf_prior: per-gene array, NOT edge-level -- no presort needed,
        # same lookup-by-source-index pattern as chi_prior/rho_prior. GPU
        # path doesn't read this attribute at all -- it's baked into
        # FungiGPUContext once at init_gpu_context() time instead; this is
        # only consumed by the CPU/Ray fallback path's run_dash_and_score
        # calls and by refinement.py's champion/cohort rebuild.
        self.rdf_prior = (rdf_prior.astype(np.float64)
                          if rdf_prior is not None else None)

        # md_gate: presort in the same order
        if md_gate is not None:
            order = np.argsort(W_arr)[::-1]
            self.md_gate = md_gate[order].astype(np.float64)
        else:
            self.md_gate = None

        if w_causal is not None:
            order = np.argsort(W_arr)[::-1]
            self.w_causal = w_causal[order].astype(np.float64)
        else:
            self.w_causal = None

        self._perturbed_nodes = perturbed_nodes
        self._utopian_bounds = utopian_bounds
        self._loss_weights = loss_weights
        self._shatter_cfg = shatter_cfg
        self._per_gene_kappa = per_gene_kappa
        self._source_pert_impact = source_pert_impact

        self.mode                 = mode
        self.deg_matrix_csr       = deg_matrix_csr
        self.gene_community_labels = gene_community_labels
        self.kernel_flags         = kernel_flags
        self.gene_features        = gene_features
        self.spectra_L            = int(spectra_L)
        self.exact_spectral       = bool(exact_spectral)

        self._ray = None
        self._ray_refs = None
        self._ray_topology_refs = None

    def _init_ray(self):
        ray, ok = _try_ray()
        if not ok:
            return False
        self._ray = ray
        os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "1"
        logging.getLogger("ray").setLevel(logging.ERROR)

        # Always restart Ray so workers reimport engine.py from disk.
        # Reusing a live cluster risks stale module caches when engine.py
        # changes between runs — all workers keep the version they first loaded.
        if ray.is_initialized():
            ray.shutdown()
            ray.init(
                num_cpus=self.n_workers,
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                configure_logging=True,
                logging_level=logging.ERROR,
                runtime_env={"env_vars": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }},
            )

        self._ray_refs = {
            "W":      ray.put(self.Ws),
            "W_q":    ray.put(self.Wqs),
            "D":      ray.put(self.Ds),
            "src":    ray.put(self.srcs),
            "tgt":    ray.put(self.tgts),
            "pert":   ray.put(self._perturbed_nodes),
            "bounds": ray.put(self._utopian_bounds),
            "weights": ray.put(self._loss_weights),
            "shatter": ray.put(self._shatter_cfg),
            "kappa":  ray.put(self._per_gene_kappa),
            "impact": ray.put(self._source_pert_impact),
            "md":     ray.put(self.md_gate),
            # er_scores / inter_mask / chi_prior may be None — ray.put(None) is valid
            "er":         ray.put(self.er_scores),
            "er_eta":     ray.put(self.er_eta),
            "inter_mask": ray.put(self.inter_mask),
            "intra_mask": ray.put(self.intra_mask),
            "chi":        ray.put(self.chi_prior),
            "wcausal":    ray.put(self.w_causal),
            "rho":        ray.put(self.rho_prior),
            "chi_t":      ray.put(self.chi_t_prior),
            "rdf":        ray.put(self.rdf_prior),
            # Synthetic mode additions — ray.put(None) is valid no-op
            "mode":       ray.put(self.mode),
            "deg":        ray.put(self.deg_matrix_csr),
            "comm":       ray.put(self.gene_community_labels),
            "kflags":     ray.put(self.kernel_flags),
            "feats":      ray.put(self.gene_features),
            "spectra_L":  ray.put(self.spectra_L),
            "exact_spec": ray.put(self.exact_spectral),
        }
        return True

    def evaluate(self, param_list, chunk_size=50, shard_dir=None,
                 desc="Evaluating", show_progress=True):
        from engine import get_gpu_context
        gpu_ctx = get_gpu_context()
        if gpu_ctx is not None and gpu_ctx.enabled:
            if self.use_ray_pipeline:
                try:
                    return self._evaluate_gpu_ray(
                        param_list, n_workers=self.n_workers, shard_dir=shard_dir,
                        desc=desc, show_progress=show_progress)
                except Exception as e:
                    warnings.warn(f"[FUNGI GPU+Ray] _evaluate_gpu_ray failed ({e}), "
                                  "falling back to single-process GPU for this call")
            try:
                return self._evaluate_gpu(param_list, shard_dir=shard_dir,
                                          desc=desc, show_progress=show_progress)
            except Exception as e:
                warnings.warn(f"[FUNGI GPU] Batched GPU evaluate() failed ({e}), "
                              "falling back to CPU/Ray for this call")

        if self._ray is None:
            ray_ok = self._init_ray()
            if not ray_ok:
                return self._evaluate_joblib(param_list)

        ray = self._ray
        refs = self._ray_refs
        n_genes = self.n_genes
        src_dir = self.src_dir

        @ray.remote
        def _eval_chunk(chunk, Wr, Wqr, Dr, sr, tr, ng,
                        pr, br, wr, shr, kr, ir, mdr, err, er_eta_r, imr, intramr,
                        chir, wcr, rhor, chittr,
                        rdfr, mode_r, degr, commr, kflagsr, featsr, spectraLr, exactspecr, sd):
            import sys
            if sd not in sys.path:
                sys.path.insert(0, sd)
            from engine import run_dash_and_score
            return [run_dash_and_score(
                p, Wr, Wqr, Dr, sr, tr, ng, pr, br, wr, shr, kr, ir,
                md_gate=mdr, er_scores=err, er_eta=er_eta_r,
                inter_mask=imr, intra_mask=intramr, chi_prior=chir, w_causal=wcr,
                rho_prior=rhor, chi_t_prior=chittr, rdf_prior=rdfr,
                mode=mode_r, deg_matrix_csr=degr,
                gene_community_labels=commr,
                kernel_flags=kflagsr, gene_features=featsr,
                spectra_L=spectraLr, exact_spectral=exactspecr)
                for p in chunk]

        # Shard recovery
        all_results = []
        graphs_done = 0

        if shard_dir is not None:
            os.makedirs(shard_dir, exist_ok=True)
            existing = sorted(
                glob.glob(os.path.join(shard_dir, "shard_*.csv")),
                key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
                if re.search(r'\d+', os.path.basename(f)) else 0)
            if existing:
                for sf in existing:
                    all_results.extend(pd.read_csv(sf).to_dict('records'))
                graphs_done = len(all_results)
                if graphs_done > 0:
                    print(f"  Resuming: {graphs_done:,} graphs from {len(existing)} shards")

            remaining = param_list[graphs_done:]
            if len(remaining) == 0:
                return pd.DataFrame(all_results)
        else:
            remaining = param_list

        chunks = [remaining[i:i + chunk_size]
                  for i in range(0, len(remaining), chunk_size)]

        from tqdm.auto import tqdm as tqdm_auto
        futures = [
            _eval_chunk.remote(
                c, refs["W"], refs["W_q"], refs["D"],
                refs["src"], refs["tgt"], n_genes,
                refs["pert"], refs["bounds"], refs["weights"],
                refs["shatter"], refs["kappa"], refs["impact"],
                refs["md"], refs["er"], refs["er_eta"],
                refs["inter_mask"], refs["intra_mask"], refs["chi"], refs["wcausal"],
                refs["rho"], refs["chi_t"], refs["rdf"],
                refs["mode"], refs["deg"], refs["comm"],
                refs["kflags"], refs["feats"], refs["spectra_L"],
                refs["exact_spec"], src_dir)
            for c in chunks
        ]

        n_total = len(param_list)
        if show_progress:
            pbar = tqdm_auto(
                total=n_total, initial=graphs_done, desc=desc, unit="graph",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        else:
            pbar = None

        completed = 0
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            cr = ray.get(done[0])
            if shard_dir is not None:
                pd.DataFrame(cr).to_csv(
                    os.path.join(shard_dir, f"shard_{completed:05d}.csv"),
                    index=False)
            all_results.extend(cr)
            completed += 1
            if pbar is not None:
                pbar.update(len(cr))

        if pbar is not None:
            pbar.close()

        return pd.DataFrame(all_results)

    def _evaluate_gpu(self, param_list, chunk_size=None, shard_dir=None,
                      desc="Evaluating", show_progress=True):
        """
        Single-process GPU batched path (no Ray): omega+sort for a whole
        batch computed together on the GPU context set up once before this
        loop began, then each row's select_edges/_motif_repair_swap/shatter/
        loss runs through the same CPU functions run_dash_and_score uses.

        Deliberately not a Ray remote -- Ray workers are separate processes
        and would each need their own CUDA context on the same 8GB GPU.
        """
        from engine import run_dash_and_score_gpu_batch, get_gpu_context
        ctx = get_gpu_context()
        B = chunk_size or _choose_gpu_batch_size(ctx)

        all_results = []
        graphs_done = 0
        if shard_dir is not None:
            os.makedirs(shard_dir, exist_ok=True)
            existing = sorted(
                glob.glob(os.path.join(shard_dir, "shard_*.csv")),
                key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
                if re.search(r'\d+', os.path.basename(f)) else 0)
            if existing:
                for sf in existing:
                    all_results.extend(pd.read_csv(sf).to_dict('records'))
                graphs_done = len(all_results)
                if graphs_done > 0:
                    print(f"  Resuming: {graphs_done:,} graphs from {len(existing)} shards")
            remaining = param_list[graphs_done:]
            if len(remaining) == 0:
                return pd.DataFrame(all_results)
        else:
            remaining = param_list

        chunks = [remaining[i:i + B] for i in range(0, len(remaining), B)]

        from tqdm.auto import tqdm as tqdm_auto
        pbar = tqdm_auto(
            total=len(param_list), initial=graphs_done, desc=desc, unit="graph",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) if show_progress else None

        for ci, chunk in enumerate(chunks):
            results = run_dash_and_score_gpu_batch(
                list(chunk), ctx, self.n_genes, self._perturbed_nodes,
                self._utopian_bounds, self._loss_weights, self._shatter_cfg,
                self._per_gene_kappa, mode=self.mode,
                deg_matrix_csr=self.deg_matrix_csr,
                gene_community_labels=self.gene_community_labels,
                gene_features=self.gene_features, spectra_L=self.spectra_L,
                exact_spectral=self.exact_spectral)
            if shard_dir is not None:
                pd.DataFrame(results).to_csv(
                    os.path.join(shard_dir, f"shard_{graphs_done + ci:05d}.csv"),
                    index=False)
            all_results.extend(results)
            if pbar is not None:
                pbar.update(len(results))

        if pbar is not None:
            pbar.close()

        return pd.DataFrame(all_results)

    def _init_ray_topology(self, n_workers=6):
        """
        Sets up Ray and puts the STATIC per-run data (same for every eval)
        into the object store exactly once. Separate from _init_ray()'s refs
        (the old CPU/Ray fallback path's refs include the full presorted
        edge arrays, which the GPU+Ray path doesn't need -- those come from
        the per-batch GPU step instead, via compute_gpu_omega_sort_batch).
        """
        ray, ok = _try_ray()
        if not ok:
            raise RuntimeError("ray is not installed in this environment")

        os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "1"
        logging.getLogger("ray").setLevel(logging.ERROR)

        # Ray worker processes are separate Python interpreters and do NOT
        # inherit the driver's sys.path mutations (sys.path.insert() in the
        # notebook only affects the driver's own already-running interpreter,
        # not freshly spawned workers). They DO inherit PYTHONPATH via
        # runtime_env's env_vars (read at interpreter startup), which is what
        # lets `from engine import ...` resolve correctly inside workers
        # regardless of the driver's current working directory.
        _search_module_dir = os.path.dirname(os.path.abspath(__file__))

        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            num_cpus=n_workers,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            configure_logging=True,
            logging_level=logging.ERROR,
            runtime_env={"env_vars": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONPATH": _search_module_dir,
            }},
        )

        refs = {
            "pert":       ray.put(self._perturbed_nodes),
            "bounds":     ray.put(self._utopian_bounds),
            "weights":    ray.put(self._loss_weights),
            "shatter":    ray.put(self._shatter_cfg),
            "kappa":      ray.put(self._per_gene_kappa),
            "mode":       ray.put(self.mode),
            "deg":        ray.put(self.deg_matrix_csr),
            "comm":       ray.put(self.gene_community_labels),
            "feats":      ray.put(self.gene_features),
            "spectra_L":  ray.put(self.spectra_L),
            "exact_spec": ray.put(self.exact_spectral),
        }
        self._ray = ray
        self._ray_topology_refs = refs
        return ray, refs

    def _evaluate_gpu_ray(self, param_list, n_workers=6, chunk_size=None,
                          shard_dir=None, desc="Evaluating", show_progress=True):
        """
        GPU+Ray pipelined path: omega+sort for a whole batch computed on the
        GPU in the main process (identical math to _evaluate_gpu's GPU step,
        factored into engine.compute_gpu_omega_sort_batch), but per-eval
        topology (select_edges/_motif_repair_swap/check_shatter/
        calculate_utopia_loss) is dispatched to n_workers Ray worker
        processes in parallel instead of a sequential Python loop in the
        main process. Routed to automatically by evaluate() when
        self.use_ray_pipeline is set and a GPU context is active; falls back
        to _evaluate_gpu on any exception.

        Batch arrays (so/to/no, already int32/float32 from
        compute_gpu_omega_sort_batch) are ray.put ONCE per batch, not once
        per eval -- each worker receives the shared refs plus its own row
        index and slices its row via Ray's zero-copy Plasma read.

        Verified bit-identical against _evaluate_gpu at production scale
        (n_genes=5000, real RPE1 perturbed_nodes=868) and on the full real
        4096-eval Sobol sweep (best loss matched to the decimal: 6.43x
        speedup, 108.4min -> 16.86min). See
        markdowns/claude_code/claude_code_session_2.md for the full
        verification record.
        """
        from engine import compute_gpu_omega_sort_batch, get_gpu_context
        ctx = get_gpu_context()
        B = chunk_size or _choose_gpu_batch_size(ctx)
        n_genes = self.n_genes

        if self._ray is None or not getattr(self, "_ray_topology_refs", None):
            ray, refs = self._init_ray_topology(n_workers=n_workers)
        else:
            ray, refs = self._ray, self._ray_topology_refs

        @ray.remote
        def _topology_worker(row_idx, src_batch, tgt_batch, omega_batch, params,
                             ng, perturbed_nodes, utopian_bounds, loss_weights,
                             shatter_cfg, per_gene_kappa, mode, deg_matrix_csr,
                             gene_community_labels, gene_features, spectra_L,
                             exact_spectral):
            # Direct ObjectRef arguments to .remote() are auto-resolved by Ray
            # before this function runs -- src_batch/tgt_batch/omega_batch and
            # every other parameter here are already plain values, not refs.
            # No CUDA/torch import anywhere in this function or its engine
            # imports below -- must stay true for Windows worker-process safety.
            import numpy as _np
            from engine import (select_edges, _motif_repair_swap, check_shatter,
                                calculate_utopia_loss, calculate_synthetic_loss,
                                _synthetic_path_gate, _safe)

            so = src_batch[row_idx]
            to = tgt_batch[row_idx]
            no = omega_batch[row_idx]
            Wo = no  # placeholder -- fast_topology=True never reads edge weight

            if len(params) >= 8:
                beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
            elif len(params) >= 7:
                beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
                m_intra = 1.0
            else:
                beta, delta, kappa_base, k_core, lam, psi = params[:6]
                nu = 0.0
                m_intra = 1.0
            k_core_eff = max(k_core, max(5.0, lam * 0.4))
            # exp_008 levers (exp_035): store searched lever values (params[8:]) by position so the
            # champion can be rebuilt exactly and refinement can reconstruct full param vectors.
            _levers = {f'_lever{_k}': float(params[8 + _k]) for _k in range(max(0, len(params) - 8))}

            def _shattered(reason, od_arr, active_n, n_edges):
                return {
                    'beta': beta, 'delta': delta, 'kappa': kappa_base,
                    'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                    'm_intra': m_intra,
                    'utopia_loss': 999., 'is_shattered': 1,
                    'shatter_reason': reason, 'n_edges': n_edges,
                    'active_nodes': active_n, 'alpha': 1., 'Gini': 1.,
                    'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1., 'reciprocity': 1.,
                    'S_max': _safe((_np.max(od_arr) / ng) if len(od_arr) > 0 else 0),
                    'epr_k': 0., 'weight_entropy': 0., 'source_conc': 0.,
                    'heterophily': 0., 'spectral_gap': 0., **_levers}

            try:
                param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
                rng = _np.random.default_rng(param_hash)

                surv_s, surv_t, surv_W = select_edges(
                    no, Wo, so, to, perturbed_nodes, ng, lam,
                    per_gene_kappa, kappa_base)
                surv_s, surv_t, surv_W = _motif_repair_swap(
                    surv_s, surv_t, surv_W, no, so, to, ng,
                    max_swap_fraction=0.03, rng=rng)

                od = _np.bincount(surv_s, minlength=ng)
                idd = _np.bincount(surv_t, minlength=ng)
                active = int(_np.count_nonzero(od + idd > 0))

                sh, reason, gwcc_fraction = check_shatter(
                    surv_s, surv_t, surv_W, od, ng, active, shatter_cfg)
                if sh:
                    return _shattered(reason, od, active, len(surv_s))

                if mode == 'synthetic':
                    if deg_matrix_csr is not None and perturbed_nodes is not None:
                        bad_path = _synthetic_path_gate(
                            surv_s, surv_t, ng, perturbed_nodes,
                            deg_matrix_csr, int(spectra_L),
                            min_reach_frac=shatter_cfg.get('min_reach_frac', 0.5))
                        if bad_path:
                            return _shattered('deg_path_exceeds_L', od, active, len(surv_s))
                    loss, topo = calculate_synthetic_loss(
                        surv_s, surv_t, surv_W, ng, od, active, kappa_base,
                        utopian_bounds, loss_weights,
                        deg_matrix_csr=deg_matrix_csr,
                        perturbed_nodes=perturbed_nodes,
                        gene_community_labels=gene_community_labels,
                        gene_features=gene_features,
                        exact_spectral=exact_spectral)
                else:
                    loss, topo = calculate_utopia_loss(
                        surv_s, surv_t, surv_W, ng, od, active, kappa_base,
                        utopian_bounds, loss_weights, fast_topology=True,
                        gene_community_labels=gene_community_labels)

                gf = gwcc_fraction if gwcc_fraction is not None else 0.
                gp = 8. * ((0.45 - gf) / 0.45) ** 2 if gf < 0.45 else 0.
                loss = _safe(_np.sqrt(max(loss ** 2 + gp, 0.)), 999.)

                return {
                    'beta': beta, 'delta': delta, 'kappa': kappa_base,
                    'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                    'm_intra': m_intra,
                    'utopia_loss': loss, 'is_shattered': 0, 'shatter_reason': None,
                    'n_edges': len(surv_s), 'active_nodes': active,
                    'gwcc_fraction': gf,
                    'alpha': topo['alpha'], 'Gini': topo['Gini'],
                    'rho': topo['rho'], 'C': topo['C'], 'gini_in': topo['gini_in'],
                    'Q': topo['Q'],
                    'reciprocity': topo.get('reciprocity', _safe(0.)),
                    'S_max': topo['S_max'],
                    # exp_012 P2: surface the 5 synthetic metrics so synthetic-mode
                    # runs report them (calculate_synthetic_loss puts them in topo;
                    # organic topo lacks them -> .get defaults). Fixes null
                    # champion_metrics on synthetic runs. Schema-consistent across modes.
                    'epr_k':          topo.get('epr_k', _safe(0.)),
                    'weight_entropy': topo.get('weight_entropy', _safe(0.)),
                    'source_conc':    topo.get('source_conc', _safe(0.)),
                    'heterophily':    topo.get('heterophily', _safe(0.)),
                    'spectral_gap':   topo.get('spectral_gap', _safe(0.)), **_levers}
            except Exception as e:
                return {
                    'beta': beta, 'delta': delta, 'kappa': kappa_base,
                    'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                    'm_intra': m_intra,
                    'utopia_loss': 999., 'is_shattered': 1,
                    'shatter_reason': f'crash:{str(e)[:60]}',
                    'n_edges': 0, 'active_nodes': 0,
                    'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'gini_in': 1.,
                    'Q': 1., 'reciprocity': 1., 'S_max': 0.,
                    'epr_k': 0., 'weight_entropy': 0., 'source_conc': 0.,
                    'heterophily': 0., 'spectral_gap': 0., **_levers}

        @ray.remote
        def _topology_chunk_worker(row_idxs, params_list, src_batch, tgt_batch,
                                   omega_batch, ng, perturbed_nodes, utopian_bounds,
                                   loss_weights, shatter_cfg, per_gene_kappa, mode,
                                   deg_matrix_csr, gene_community_labels, gene_features,
                                   spectra_L, exact_spectral):
            # obj_006 chunked dispatch: process a round-robin slice of the batch in
            # ONE Ray task (n_workers tasks/batch instead of B) -> fewer handshakes.
            # Bit-exact: loops the SAME _fungi_process_row the per-candidate worker used.
            from search import _fungi_process_row
            out = []
            for ri, params in zip(row_idxs, params_list):
                out.append((ri, _fungi_process_row(
                    ri, src_batch, tgt_batch, omega_batch, params, ng, perturbed_nodes,
                    utopian_bounds, loss_weights, shatter_cfg, per_gene_kappa, mode,
                    deg_matrix_csr, gene_community_labels, gene_features, spectra_L,
                    exact_spectral)))
            return out

        all_results = []
        graphs_done = 0
        if shard_dir is not None:
            os.makedirs(shard_dir, exist_ok=True)
            existing = sorted(
                glob.glob(os.path.join(shard_dir, "shard_*.csv")),
                key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
                if re.search(r'\d+', os.path.basename(f)) else 0)
            if existing:
                for sf in existing:
                    all_results.extend(pd.read_csv(sf).to_dict('records'))
                graphs_done = len(all_results)
            remaining = param_list[graphs_done:]
            if len(remaining) == 0:
                return pd.DataFrame(all_results)
        else:
            remaining = param_list

        chunks = [remaining[i:i + B] for i in range(0, len(remaining), B)]

        from tqdm.auto import tqdm as tqdm_auto
        pbar = tqdm_auto(
            total=len(param_list), initial=graphs_done, desc=desc, unit="graph",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) if show_progress else None

        for ci, chunk in enumerate(chunks):
            chunk = list(chunk)
            so_batch, to_batch, no_batch = compute_gpu_omega_sort_batch(chunk, ctx)[:3]

            src_ref = ray.put(so_batch)
            tgt_ref = ray.put(to_batch)
            omega_ref = ray.put(no_batch)

            if _OBJ006_CHUNKED_RAY:
                # n_workers round-robin slices -> n_workers Ray tasks per batch
                # (balances load across shatter-fast vs full-eval candidates).
                nb = len(chunk)
                assign = [list(range(w, nb, n_workers)) for w in range(n_workers)]
                futures = [
                    _topology_chunk_worker.remote(
                        rows, [chunk[i] for i in rows], src_ref, tgt_ref, omega_ref,
                        n_genes, refs["pert"], refs["bounds"], refs["weights"],
                        refs["shatter"], refs["kappa"], refs["mode"], refs["deg"],
                        refs["comm"], refs["feats"], refs["spectra_L"], refs["exact_spec"])
                    for rows in assign if rows
                ]
                results = [None] * nb
                for wr in ray.get(futures):
                    for ri, res in wr:
                        results[ri] = res
            else:
                futures = [
                    _topology_worker.remote(
                        i, src_ref, tgt_ref, omega_ref, chunk[i], n_genes,
                        refs["pert"], refs["bounds"], refs["weights"], refs["shatter"],
                        refs["kappa"], refs["mode"], refs["deg"], refs["comm"],
                        refs["feats"], refs["spectra_L"], refs["exact_spec"])
                    for i in range(len(chunk))
                ]
                results = ray.get(futures)

            if shard_dir is not None:
                pd.DataFrame(results).to_csv(
                    os.path.join(shard_dir, f"shard_{graphs_done + ci:05d}.csv"),
                    index=False)
            all_results.extend(results)
            if pbar is not None:
                pbar.update(len(results))

        if pbar is not None:
            pbar.close()

        return pd.DataFrame(all_results)

    def evaluate_single(self, params):
        from engine import run_dash_and_score
        return run_dash_and_score(
            params, self.Ws, self.Wqs, self.Ds, self.srcs, self.tgts,
            self.n_genes, self._perturbed_nodes,
            self._utopian_bounds, self._loss_weights, self._shatter_cfg,
            self._per_gene_kappa, self._source_pert_impact,
            md_gate=self.md_gate,
            er_scores=self.er_scores, er_eta=self.er_eta,
            inter_mask=self.inter_mask, intra_mask=self.intra_mask,
            chi_prior=self.chi_prior,
            w_causal=self.w_causal,
            rho_prior=self.rho_prior, chi_t_prior=self.chi_t_prior,
            rdf_prior=self.rdf_prior,
            mode=self.mode, deg_matrix_csr=self.deg_matrix_csr,
            gene_community_labels=self.gene_community_labels,
            kernel_flags=self.kernel_flags, gene_features=self.gene_features,
            spectra_L=self.spectra_L, exact_spectral=self.exact_spectral)

    def evaluate_batch_for_optimizer(self, batch_params):
        """Used by refinement.py — returns list of result dicts."""
        return [self.evaluate_single(p) for p in batch_params]

    def _evaluate_joblib(self, param_list):
        from joblib import Parallel, delayed
        from engine import run_dash_and_score
        print(f"  Executing with joblib ({self.n_workers} workers)...")
        results = Parallel(n_jobs=self.n_workers)(
            delayed(run_dash_and_score)(
                p, self.Ws, self.Wqs, self.Ds, self.srcs, self.tgts,
                self.n_genes, self._perturbed_nodes,
                self._utopian_bounds, self._loss_weights, self._shatter_cfg,
                self._per_gene_kappa, self._source_pert_impact,
                md_gate=self.md_gate,
                er_scores=self.er_scores, er_eta=self.er_eta,
                inter_mask=self.inter_mask, intra_mask=self.intra_mask,
                chi_prior=self.chi_prior,
                w_causal=self.w_causal,
                rho_prior=self.rho_prior, chi_t_prior=self.chi_t_prior,
            rdf_prior=self.rdf_prior,
                mode=self.mode, deg_matrix_csr=self.deg_matrix_csr,
                gene_community_labels=self.gene_community_labels,
                kernel_flags=self.kernel_flags, gene_features=self.gene_features,
                spectra_L=self.spectra_L, exact_spectral=self.exact_spectral)
            for p in param_list)
        return pd.DataFrame(results)