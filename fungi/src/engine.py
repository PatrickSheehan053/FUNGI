"""
FUNGI v9.2 — DASH Kernel Engine

Changes from v9.1:
  ─────────────────────────────────────────────────────────────────────
  E1. Alpha xmin consistency fix in calculate_utopia_loss
        Replaced fixed xmin=2 with the same KS-minimising auto-selection
        (50-node tail guard, xmin=6 fallback) used in diagnostics v11.0.
        Previously the probe set targets using auto-xmin (corrected values)
        while the loss measured alpha with xmin=2 (biased ~0.30 units low),
        creating a silent probe/loss mismatch. With xmin=2 the optimizer
        saw no upper-bound penalty even when true alpha exceeded the ceiling.
  E2. Perturbation protection budget cap in select_edges
        The top-3-edges-per-pert protection is now capped at 5% of total
        edge budget. On VCC (96 perts) this is identical to previous
        behaviour (<0.3% of budget). On K562 (3454 perts) the old code
        reserved up to 32.7% of the budget as forced inclusions regardless
        of omega score, suppressing hub concentration and depressing S_max.
        Cap formula: max_prot = max(1, int(budget * 0.05)).
        Protected edges are the highest-omega ones within the cap.
  E3. Rho penalty uses absolute observed value (probe/loss consistency)
        Removed the rho_base shift in calculate_utopia_loss. The old code
        estimated rho_base from the degree distribution via a BA-style
        formula and then evaluated (rho_obs - rho_base) against shifted
        bounds. The probe sets absolute rho targets via Spearman correlation.
        The shift introduced a theory-motivated but inconsistent offset that
        caused the optimizer to navigate toward different rho values than the
        probe intended. Now both probe and loss evaluate absolute rho.
        Literature: Newman (2002) Phys Rev Lett 89:208701; the directed
        assortativity is correctly computed as assortativity_degree() in
        igraph which returns the Pearson correlation of out-degree at source
        vs in-degree at target, consistent with the probe's Spearman proxy.
  E4. _motif_repair_swap: vectorized omega lookup (performance)
        Replaced the O(N_edges) dict build (previously ~13.8B iterations
        per full run) with a sorted binary-search lookup. Same result,
        ~100× faster on 2.5M-edge pools.
  E5. _motif_repair_swap: correct W assignment for swapped edges
        Swapped-in edges now carry W=0.0 as a placeholder instead of the
        DASH score. W is used only for graph output (Regulator/Target/Weight
        parquet), not for loss computation. Using the DASH score as W was
        mixing units and inflating apparent edge importance for motif edges.
  E6. Ne evaluation window widened to 4× lambda_max headroom
        Previously Ne = Nm + 10_000 where Nm = lambda_max × n_genes.
        At lambda_max=40 and n_genes=5000 this gave Ne=210_000, leaving
        only 10k slack for kappa backfill at the top of the search range.
        Now Ne = min(len(W), max(Nm * 2, Nm + 50_000)) which provides
        ample backfill candidates at all lambda values while remaining
        memory-efficient on the presorted array.
  E7. select_edges: kappa backfill no longer blanket-excludes capped sources
        Old backfill: after kappa enforcement, ALL edges from any source
        that was over-cap were excluded from backfill candidates, meaning
        sources that were capped by exactly 1 edge had all remaining edges
        blocked. New backfill: only edges from already-selected nodes are
        masked, allowing capped sources to contribute their next-best edges
        in the backfill pass. This prevents systematic underfill at high λ.

Changes from v9.0 (v9.1 additions, preserved):
  - Effective Resistance (ER) scoring integrated as a multiplicative term.
  - run_dash_and_score and build_graph_from_params accept optional
    er_scores (np.ndarray) and er_eta (float) keyword arguments.
  - er_scores=None (default) is a strict no-op (backward-compatible).
"""

import time
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import warnings
import graphblas as gb
import igraph as ig
import powerlaw

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _safe(x, fb=0.0):
    return float(x) if (x is not None and np.isfinite(x)) else fb


# --- exp_008 (DASH kernel firepower) searched kernel levers -- ported into exp_035 ---
# New/converted hyperparameters searched in addition to the base 8
# (beta,delta,kappa,k_core,lambda,psi,nu,m_intra). Each mirrors m_intra: a
# static per-edge array on the GPU ctx x a per-eval searched scalar. They are
# appended to the Sobol param vector AFTER m_intra in this canonical order; the
# default value reproduces the stock kernel exactly (so an arm that does not
# search a lever is byte-equivalent to production at that lever). See
# markdowns/claude_code/claude_code_session_13.md.
_EXTRA_HP_ORDER = ["sigma_scber", "m_inter", "eta_out", "zeta_s", "gamma_md", "tau_indeg", "theta_pa"]
_EXTRA_HP_DEFAULTS = {
    "sigma_scber": 0.28,  # = effective_resistance.eta_inter (baked SCBER exponent)
    "m_inter": 1.0,       # log(1)=0 -> no inter-community penalty
    "eta_out": 0.0,       # 0 -> no out-degree anti-hub term
    "zeta_s": 0.5,        # = chi_prior.zeta (baked chi_s exponent)
    "gamma_md": 0.0,      # 0 -> no MD gate
    "tau_indeg": 0.0,     # exp_035 Task 4: 0 -> no target in-degree (anti-promiscuity) penalty
    "theta_pa": 0.0,      # followupA: GRADED preferential-attachment boost omega*=(outdeg/max)^theta_pa; 0->identity
}


def _extra_coefs_from_params(params_list, ctx, device):
    """Build {lever_name: (B,) float32 cuda tensor} from params_list[i][8:],
    ordered by ctx.extra_hp_names. Returns None if the ctx has no extra levers.
    Missing trailing entries fall back to the no-op default for that lever."""
    names = tuple(getattr(ctx, "extra_hp_names", ()) or ())
    if not names:
        return None
    extra = {}
    for j, nm in enumerate(names):
        col = 8 + j
        vals = [(float(p[col]) if len(p) > col else _EXTRA_HP_DEFAULTS[nm])
                for p in params_list]
        extra[nm] = torch.tensor(vals, device=device, dtype=torch.float32)
    return extra


# Module-level countdown for the GPU batch path's per-eval profiling print --
# decremented in run_dash_and_score_gpu_batch, read (not decremented) inside
# calculate_utopia_loss via its profile= flag. Auto-silences itself after the
# first 10 evaluations of the process so it's safe to leave permanently on.
_PROFILE_EVALS_LEFT = 10


# ═══════════════════════════════════════════════════════════════════════════
#  GPU ACCELERATION (RTX 2070 Max-Q / Turing SM 7.5, Windows 11, CUDA 12.1)
#
#  Optional fast path for run_dash_and_score's omega + ordering steps. The
#  CPU code throughout the rest of this file is the fallback and ground
#  truth -- run_dash_and_score's GPU branch falls through to it on any
#  exception. Built once per search run via init_gpu_context(), called from
#  search.py's single-process GPU batched loop (NOT from a Ray worker --
#  Ray workers are separate processes with their own copy of this module's
#  globals, so a context set here would not be visible to them).
#
#  Per-edge static arrays are stored PRE-GROUPED by source gene: sources[:Ne]
#  is presorted by descending weight (search.presort_edges), not by source,
#  so segment boundaries are not naturally fixed. A one-time stable argsort
#  at init groups everything by source once; every per-eval segmented sort
#  then directly yields "grouped by source asc, omega desc within source"
#  order -- the same semantics as the CPU path's np.lexsort((-num, ss)) --
#  with no per-eval unpermute step needed.
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Optional

@dataclass
class FungiGPUContext:
    enabled: bool = False
    device: str = "cuda"
    n_genes: int = 0
    Ne: int = 0

    # Per-edge tensors, GPU-resident float32, length Ne, grouped by source
    log_wq: Optional["torch.Tensor"] = None
    log_pi_base: Optional["torch.Tensor"] = None
    log_rdf_base: Optional["torch.Tensor"] = None
    log_static: Optional["torch.Tensor"] = None
    # m_intra (8th searched hyperparameter, session 5): per-eval scalar boost
    # on intra-community edges, applied as log(m_intra) * intra_mask_f. Unlike
    # log_static's factors (fixed exponents baked in once), m_intra varies per
    # Sobol sample -- same reason delta*T_batch isn't baked into log_static.
    intra_mask_f: Optional["torch.Tensor"] = None
    # exp_008 (DASH kernel firepower) -- static per-edge lever tensors, grouped
    # order, each x a per-eval searched scalar in compute_omega_batch_gpu. None
    # => that lever is not active for this run. extra_hp_names gives the ordered
    # names of the params appended after m_intra (params[8:]).
    log_R_inter: Optional["torch.Tensor"] = None       # sigma_scber: log(R).inter
    inter_mask_f: Optional["torch.Tensor"] = None      # m_inter: 0/1 inter mask
    log1p_outdeg_src: Optional["torch.Tensor"] = None  # eta_out: log1p(parent_outdeg[src])
    log_outdeg_norm: Optional["torch.Tensor"] = None    # theta_pa: log(parent_outdeg[src]/max_outdeg) <= 0
    log_chi_raw_s: Optional["torch.Tensor"] = None     # zeta_s: log(chi_raw[src])
    log_md: Optional["torch.Tensor"] = None            # gamma_md: log(md_score)
    log_indeg_tgt: Optional["torch.Tensor"] = None     # exp_035 tau_indeg: log1p(promiscuity[tgt])
    extra_hp_names: tuple = ()
    ss: Optional["torch.Tensor"] = None
    ts: Optional["torch.Tensor"] = None
    W: Optional["torch.Tensor"] = None

    # Segment metadata derived from the grouped ss
    seg_offsets: Optional["torch.Tensor"] = None
    seg_lengths: Optional["torch.Tensor"] = None
    max_seg_len: int = 0
    # row_ids[i]/col_ids[i]: the (gene, local-position-within-segment) padded-
    # tensor coordinates for grouped-array position i. Known-valid by
    # construction (never point into padding) -- used for both the scatter
    # into the padded tensor and the gather back out, so no value-based
    # validity inference (e.g. "!= sentinel") is needed anywhere.
    row_ids: Optional["torch.Tensor"] = None
    col_ids: Optional["torch.Tensor"] = None

    # FFL bucket cache: (n_buckets, Ne) stacked, grouped order. Different
    # configs in the same batch can land in different buckets (effective
    # k_core varies per-eval), so this is gathered per-row, not per-batch --
    # see lookup_T_bucket_rows().
    T_bucket_stack: Optional["torch.Tensor"] = None
    k_core_bucket_values: Optional[np.ndarray] = None  # the kc each row of the stack was built at

    group_perm: Optional[np.ndarray] = None  # kept for verification only

_GPU_CTX: Optional[FungiGPUContext] = None

def get_gpu_context() -> Optional[FungiGPUContext]:
    return _GPU_CTX

def release_gpu_context() -> None:
    global _GPU_CTX
    _GPU_CTX = None
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

def _build_segment_metadata(ss_grouped: np.ndarray, n_genes: int):
    """ss_grouped must already be sorted ascending (grouped by source)."""
    counts = np.bincount(ss_grouped, minlength=n_genes).astype(np.int64)
    seg_offsets = np.zeros(n_genes + 1, dtype=np.int64)
    seg_offsets[1:] = np.cumsum(counts)
    max_seg_len = int(counts.max()) if len(counts) > 0 else 0
    return seg_offsets, counts, max_seg_len


def init_gpu_context(W, W_q, sources, targets, n_genes, shatter_cfg,
                     source_pert_impact, k_core_bounds,
                     rdf_prior=None, er_scores=None, er_eta=0.3,
                     inter_mask=None, chi_prior=None, chi_t_prior=None,
                     rho_prior=None, kernel_flags=None, device="cuda",
                     n_ffl_buckets=8, intra_mask=None,
                     extra_hp_names=(), externalize_scber=False,
                     externalize_chi_s=False, parent_outdeg=None,
                     md_edge=None, promiscuity_prior=None) -> FungiGPUContext:
    """
    Build the persistent GPU context for one search run. Call once, from the
    main process, before the Sobol/TPE loop begins.

    W, W_q, sources, targets: full arrays as passed to run_dash_and_score /
        SearchEvaluator (already globally presorted by descending weight).
    k_core_bounds: (lo, hi) -- the configured hyperparameter_bounds.k_core
        range. FFL bucket te values scale with k_core x n_genes, not with
        lambda_max/Ne -- compute_dynamic_topology's te depends only on
        k_core.
    kernel_flags: dash_kernel on/off flags (utils.build_kernel_flags). A
        disabled factor's static log-array is zeroed here once, so the
        per-eval omega kernel needs no per-factor branching.
    """
    global _GPU_CTX
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        _GPU_CTX = FungiGPUContext(enabled=False)
        return _GPU_CTX

    dev = torch.device(device)
    major, _ = torch.cuda.get_device_capability(dev)
    torch.backends.cuda.matmul.allow_tf32 = bool(major >= 8)

    kf = kernel_flags or {}
    _on = lambda name: bool(kf.get(name, True))

    Nm = shatter_cfg.get("max_edge_count", 500000)
    Ne = min(len(W), max(Nm * 2, Nm + 50_000))
    eps = 1e-6

    W_full  = np.asarray(W, dtype=np.float64)
    Wq_ne   = np.clip(np.asarray(W_q[:Ne], dtype=np.float64), eps, None)
    ss_ne   = np.asarray(sources[:Ne], dtype=np.int64)
    ts_ne   = np.asarray(targets[:Ne], dtype=np.int64)
    W_ne    = W_full[:Ne]

    pi_base = np.clip(np.asarray(source_pert_impact, dtype=np.float64)[ss_ne], eps, None)
    rdf_base = (np.clip(np.asarray(rdf_prior, dtype=np.float64)[ss_ne], eps, None)
                if rdf_prior is not None else np.ones(Ne, dtype=np.float64))

    er_arr = (np.asarray(er_scores[:Ne], dtype=np.float64) if er_scores is not None
              else np.ones(Ne, dtype=np.float64))
    if inter_mask is not None:
        er_arr = np.where(np.asarray(inter_mask[:Ne], dtype=bool), er_arr, 1.0)
    er_arr = np.power(np.clip(er_arr, eps, None), float(er_eta))

    chi_s_arr = (np.clip(np.asarray(chi_prior, dtype=np.float64)[ss_ne], eps, None)
                 if chi_prior is not None else np.ones(Ne, dtype=np.float64))
    chi_t_arr = (np.clip(np.asarray(chi_t_prior, dtype=np.float64)[ts_ne], eps, None)
                 if chi_t_prior is not None else np.ones(Ne, dtype=np.float64))
    rho_arr   = (np.clip(np.asarray(rho_prior, dtype=np.float64)[ss_ne], eps, None)
                 if rho_prior is not None else np.ones(Ne, dtype=np.float64))

    # m_intra (session 5): boolean edge mask, true for intra-community edges
    # under SCBER's own community partition. Reused, not recomputed -- this
    # is the same partition compute_scber_scores already detected. Gated by
    # kernel_flags["m_intra"] here (not per-eval) since this static array is
    # baked once at context-init time -- an all-False mask makes the
    # per-eval log(m_intra)*intra_mask_f term exactly zero regardless of the
    # searched m_intra value, i.e. forced identity, same effect _on() has on
    # every other kernel-chain factor.
    intra_arr = (np.asarray(intra_mask[:Ne], dtype=bool)
                 if (_on('m_intra') and intra_mask is not None)
                 else np.zeros(Ne, dtype=bool))

    # Bake kernel_flags into the static arrays: a disabled factor's log
    # contributes exactly 0 (i.e. the factor multiplies to 1), regardless
    # of whatever per-eval scalar exponent would otherwise apply to it.
    log_wq_full      = np.log(Wq_ne)   if _on("weight")      else np.zeros(Ne)
    log_pi_base_full = np.log(pi_base) if _on("pert_impact") else np.zeros(Ne)
    log_rdf_full     = np.log(rdf_base) if _on("rdf")        else np.zeros(Ne)
    # exp_008: when a factor is externalized into a searched per-eval coefficient
    # (sigma_scber for SCBER, zeta_s for chi_s), it is DROPPED from log_static here
    # and re-added per-eval in compute_omega_batch_gpu. Default coefficient values
    # (0.28 / 0.5) reproduce the baked math, so a non-externalized arm is identical.
    log_static_full = (
        (np.log(er_arr)    if (_on("scber") and not externalize_scber) else 0.0) +
        (np.log(chi_s_arr) if (_on("chi_s") and not externalize_chi_s) else 0.0) +
        (np.log(chi_t_arr) if _on("chi_t") else 0.0) +
        (np.log(rho_arr)   if _on("rho")   else 0.0))
    if np.isscalar(log_static_full):
        log_static_full = np.full(Ne, float(log_static_full))

    # exp_008 lever static per-edge arrays (presorted order, length Ne; grouped
    # below alongside the other _full arrays). Built only for the levers this arm
    # searches (names in extra_hp_names); each mirrors intra_mask_f.
    _eps_lev = 1e-12
    inter_ne = (np.asarray(inter_mask[:Ne], dtype=bool)
                if inter_mask is not None else np.zeros(Ne, dtype=bool))
    if "sigma_scber" in extra_hp_names:
        er_raw_ne = (np.clip(np.asarray(er_scores[:Ne], dtype=np.float64), _eps_lev, None)
                     if er_scores is not None else np.ones(Ne, dtype=np.float64))
        log_R_inter_full = np.where(inter_ne, np.log(er_raw_ne), 0.0)
    else:
        log_R_inter_full = None
    inter_mask_f_full = (inter_ne.astype(np.float64)
                         if "m_inter" in extra_hp_names else None)
    if "eta_out" in extra_hp_names:
        po = (np.asarray(parent_outdeg, dtype=np.float64)
              if parent_outdeg is not None else np.zeros(n_genes, dtype=np.float64))
        log1p_outdeg_full = np.log1p(po[ss_ne])
    else:
        log1p_outdeg_full = None
    # followupA theta_pa: GRADED preferential-attachment boost = (outdeg_src/max_outdeg)^theta_pa. In log-omega:
    # + theta_pa*(log(outdeg_src) - log(max_outdeg)) <= 0, i.e. a bounded (<=1) suppression of low-out-degree
    # sources that amplifies hubs PROPORTIONALLY (preserving the power-law tail) rather than eta_out's unbounded
    # (1+outdeg)^|eta| top-hub blow-up. theta_pa=0 -> log term x 0 -> identity.
    if "theta_pa" in extra_hp_names:
        po2 = (np.asarray(parent_outdeg, dtype=np.float64)
               if parent_outdeg is not None else np.ones(n_genes, dtype=np.float64))
        _maxod = max(float(po2.max()), 1.0)
        log_outdeg_norm_full = np.log(np.maximum(po2[ss_ne], 1.0)) - np.log(_maxod)
    else:
        log_outdeg_norm_full = None
    if "zeta_s" in extra_hp_names:
        # chi_prior passed in is chi_raw^0.5 (config chi_prior.zeta=0.5), so
        # chi_raw = chi_prior^2 and log(chi_raw) = 2*log(chi_prior).
        chi_raw_s_ne = (np.clip(np.asarray(chi_prior, dtype=np.float64)[ss_ne], _eps_lev, None) ** 2
                        if chi_prior is not None else np.ones(Ne, dtype=np.float64))
        log_chi_raw_s_full = np.log(chi_raw_s_ne)
    else:
        log_chi_raw_s_full = None
    if "gamma_md" in extra_hp_names:
        md_ne = (np.clip(np.asarray(md_edge[:Ne], dtype=np.float64), _eps_lev, None)
                 if md_edge is not None else np.ones(Ne, dtype=np.float64))
        log_md_full = np.log(md_ne)
    else:
        log_md_full = None
    # exp_035 tau_indeg: per-TARGET anti-promiscuity penalty (target-side analog of eta_out). Down-weights
    # edges pointing into high-in-degree ("promiscuous housekeeping hub") targets -> spreads in-edges -> gini_in down.
    if "tau_indeg" in extra_hp_names:
        pin = (np.asarray(promiscuity_prior, dtype=np.float64)
               if promiscuity_prior is not None else np.zeros(n_genes, dtype=np.float64))
        log_indeg_tgt_full = np.log1p(pin[ts_ne])
    else:
        log_indeg_tgt_full = None

    group_perm = np.argsort(ss_ne, kind="stable")
    ss_g  = ss_ne[group_perm]
    ts_g  = ts_ne[group_perm]
    W_g   = W_ne[group_perm]
    log_wq_g       = log_wq_full[group_perm]
    log_pi_base_g  = log_pi_base_full[group_perm]
    log_rdf_g      = log_rdf_full[group_perm]
    log_static_g   = log_static_full[group_perm]
    intra_mask_g   = intra_arr[group_perm]

    # exp_008 lever arrays into grouped order (same permutation as everything else)
    def _grp(a):
        return a[group_perm] if a is not None else None
    log_R_inter_g   = _grp(log_R_inter_full)
    inter_mask_f_g  = _grp(inter_mask_f_full)
    log1p_outdeg_g  = _grp(log1p_outdeg_full)
    log_outdeg_norm_g = _grp(log_outdeg_norm_full)
    log_chi_raw_s_g = _grp(log_chi_raw_s_full)
    log_md_g        = _grp(log_md_full)
    log_indeg_tgt_g = _grp(log_indeg_tgt_full)

    def _t(a):
        return (torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev,
                                dtype=torch.float32) if a is not None else None)

    seg_offsets_np, seg_lengths_np, max_seg_len = _build_segment_metadata(ss_g, n_genes)
    # row_ids = ss_g directly (ss_g IS the gene id per grouped position).
    # col_ids = position within that gene's segment.
    row_ids_np = ss_g
    col_ids_np = np.arange(Ne, dtype=np.int64) - seg_offsets_np[:-1][ss_g]

    ctx = FungiGPUContext(
        enabled=True, device=device, n_genes=n_genes, Ne=Ne,
        log_wq=torch.as_tensor(log_wq_g, device=dev, dtype=torch.float32),
        log_pi_base=torch.as_tensor(log_pi_base_g, device=dev, dtype=torch.float32),
        log_rdf_base=torch.as_tensor(log_rdf_g, device=dev, dtype=torch.float32),
        log_static=torch.as_tensor(log_static_g, device=dev, dtype=torch.float32),
        intra_mask_f=torch.as_tensor(intra_mask_g.astype(np.float32), device=dev,
                                     dtype=torch.float32),
        ss=torch.as_tensor(ss_g, device=dev, dtype=torch.int64),
        ts=torch.as_tensor(ts_g, device=dev, dtype=torch.int64),
        W=torch.as_tensor(W_g, device=dev, dtype=torch.float32),
        seg_offsets=torch.as_tensor(seg_offsets_np, device=dev, dtype=torch.int64),
        seg_lengths=torch.as_tensor(seg_lengths_np, device=dev, dtype=torch.int64),
        max_seg_len=int(max_seg_len),
        row_ids=torch.as_tensor(row_ids_np, device=dev, dtype=torch.int64),
        col_ids=torch.as_tensor(col_ids_np, device=dev, dtype=torch.int64),
        group_perm=group_perm,
        # exp_008 levers
        log_R_inter=_t(log_R_inter_g),
        inter_mask_f=_t(inter_mask_f_g),
        log1p_outdeg_src=_t(log1p_outdeg_g),
        log_outdeg_norm=_t(log_outdeg_norm_g),
        log_chi_raw_s=_t(log_chi_raw_s_g),
        log_md=_t(log_md_g),
        log_indeg_tgt=_t(log_indeg_tgt_g),
        extra_hp_names=tuple(extra_hp_names),
    )

    if _on("ffl"):
        k_lo, k_hi = k_core_bounds
        k_lo = max(float(k_lo), 5.0)
        bucket_kcores = np.linspace(k_lo, float(k_hi), n_ffl_buckets)
        sources_full = np.asarray(sources, dtype=np.int64)
        targets_full = np.asarray(targets, dtype=np.int64)
        T_rows = []
        for kc in bucket_kcores:
            T_full = compute_dynamic_topology(W_full, sources_full, targets_full, kc, n_genes)
            T_rows.append(T_full[:Ne][group_perm].astype(np.float32))
        ctx.k_core_bucket_values = bucket_kcores
        ctx.T_bucket_stack = torch.as_tensor(
            np.stack(T_rows), device=dev, dtype=torch.float32)

    _GPU_CTX = ctx
    vram_gb = torch.cuda.memory_allocated(dev) / 1e9
    print(f"[FUNGI GPU] Context initialized: Ne={Ne:,} n_genes={n_genes:,} "
          f"max_seg_len={max_seg_len:,} VRAM={vram_gb:.2f} GB")
    return ctx

def lookup_T_bucket_rows(ctx: FungiGPUContext, k_core_eff: np.ndarray) -> "torch.Tensor":
    """
    Nearest-bucket FFL lookup, one row per config. k_core_eff: (B,) ndarray
    of effective k_core (after the max(k_core, max(5, lam*0.4)) floor).
    Linear interpolation across k_core is not valid here -- FFL counts
    change piecewise with the underlying edge set -- so this is nearest-
    bucket only, matching the CPU-equivalent precompute strategy.
    """
    bucket_vals = ctx.k_core_bucket_values
    idx = np.argmin(np.abs(k_core_eff[:, None] - bucket_vals[None, :]), axis=1)
    idx_t = torch.as_tensor(idx, device=ctx.T_bucket_stack.device, dtype=torch.long)
    return ctx.T_bucket_stack[idx_t]

@torch.no_grad()
def compute_omega_batch_gpu(ctx: FungiGPUContext, beta, delta, psi, nu, m_intra,
                            k_core_eff: np.ndarray, extra=None) -> "torch.Tensor":
    """
    Batched DASH omega, log-space. beta/delta/psi/nu/m_intra: (B,) float32
    CUDA tensors. kernel_flags zeroing is already baked into ctx's static
    arrays at init -- no per-factor branching needed here. Returns (B, Ne)
    float32, grouped order (see FungiGPUContext docstring for what "grouped"
    means).

    m_intra (session 5, 8th searched hyperparameter) varies per-eval like
    beta/delta/psi/nu, so it cannot be baked into ctx.log_static (which only
    holds FIXED-exponent factors). Its per-edge static piece is
    ctx.intra_mask_f (0/1, true for intra-community edges); the per-eval
    scalar log(m_intra) is broadcast against it exactly like delta*T_batch.
    """
    if ctx.T_bucket_stack is not None:
        T_batch = lookup_T_bucket_rows(ctx, k_core_eff)
    else:
        T_batch = torch.zeros((len(beta), ctx.Ne), device=ctx.log_wq.device,
                              dtype=torch.float32)

    b, d, p, n, m = (beta.view(-1, 1), delta.view(-1, 1), psi.view(-1, 1),
                      nu.view(-1, 1), m_intra.view(-1, 1))
    log_num = (b * ctx.log_wq.unsqueeze(0)
               + d * T_batch
               + p * ctx.log_pi_base.unsqueeze(0)
               + n * ctx.log_rdf_base.unsqueeze(0)
               + torch.log(m) * ctx.intra_mask_f.unsqueeze(0)
               + ctx.log_static.unsqueeze(0))

    # exp_008 searched kernel levers -- each a static per-edge tensor x a per-eval
    # (B,) scalar, added in log-space (mirrors log(m_intra)*intra_mask_f above).
    if extra:
        if ctx.log_R_inter is not None and "sigma_scber" in extra:
            log_num = log_num + extra["sigma_scber"].view(-1, 1) * ctx.log_R_inter.unsqueeze(0)
        if ctx.inter_mask_f is not None and "m_inter" in extra:
            log_num = log_num + torch.log(extra["m_inter"].view(-1, 1)) * ctx.inter_mask_f.unsqueeze(0)
        if ctx.log1p_outdeg_src is not None and "eta_out" in extra:
            log_num = log_num - extra["eta_out"].view(-1, 1) * ctx.log1p_outdeg_src.unsqueeze(0)
        if ctx.log_outdeg_norm is not None and "theta_pa" in extra:  # followupA graded pref-attachment boost
            log_num = log_num + extra["theta_pa"].view(-1, 1) * ctx.log_outdeg_norm.unsqueeze(0)
        if ctx.log_chi_raw_s is not None and "zeta_s" in extra:
            log_num = log_num + extra["zeta_s"].view(-1, 1) * ctx.log_chi_raw_s.unsqueeze(0)
        if ctx.log_md is not None and "gamma_md" in extra:
            log_num = log_num + extra["gamma_md"].view(-1, 1) * ctx.log_md.unsqueeze(0)
        if ctx.log_indeg_tgt is not None and "tau_indeg" in extra:  # exp_035: target anti-promiscuity penalty
            log_num = log_num - extra["tau_indeg"].view(-1, 1) * ctx.log_indeg_tgt.unsqueeze(0)
    return torch.exp(log_num)

# obj_006 FUNGI-Fast (integrated with obj_006.1): flat two-pass sort dispatcher.
_OBJ006_FAST_SORT = True


@torch.no_grad()
def segmented_argsort_batch_gpu(ctx: FungiGPUContext, omega_batch: "torch.Tensor") -> "torch.Tensor":
    """Dispatcher (obj_006). flat fast path or padded reference; identical output."""
    if _OBJ006_FAST_SORT:
        return _segmented_argsort_batch_gpu_flat(ctx, omega_batch)
    return _segmented_argsort_batch_gpu_padded(ctx, omega_batch)


@torch.no_grad()
def _segmented_argsort_batch_gpu_flat(ctx: FungiGPUContext, omega_batch: "torch.Tensor") -> "torch.Tensor":
    """obj_006 flat two-pass stable sort — bit-identical grouped order to the padded reference,
    ~110x faster, ~16x less VRAM (see obj_006 report). Two stable sorts of the real Ne edges."""
    B = omega_batch.shape[0]
    src_ids = ctx.row_ids
    idx = torch.argsort(omega_batch, dim=1, descending=True, stable=True)
    src_perm = src_ids.unsqueeze(0).expand(B, -1).gather(1, idx)
    idx2 = torch.argsort(src_perm, dim=1, descending=False, stable=True)
    return torch.gather(idx, 1, idx2)


@torch.no_grad()
def _segmented_argsort_batch_gpu_padded(ctx: FungiGPUContext, omega_batch: "torch.Tensor") -> "torch.Tensor":
    """REFERENCE (equivalence anchor) — original production padded segmented argsort."""
    B = omega_batch.shape[0]
    n_genes, max_len, Ne = ctx.n_genes, ctx.max_seg_len, ctx.Ne
    device = omega_batch.device

    row_b = ctx.row_ids.unsqueeze(0).expand(B, -1)
    col_b = ctx.col_ids.unsqueeze(0).expand(B, -1)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, Ne)

    padded = torch.full((B, n_genes, max_len), float("-inf"), device=device, dtype=torch.float32)
    padded[b_idx, row_b, col_b] = omega_batch

    local_order = torch.argsort(padded, dim=2, descending=True, stable=True)
    global_idx_3d = ctx.seg_offsets[:-1].view(1, n_genes, 1) + local_order  # (B, n_genes, max_len)

    return global_idx_3d[b_idx, row_b, col_b]  # (B, Ne)


# ---------------------------------------------------------------------------
# Pre-computation helpers (called once in Phase 2, not per-evaluation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synthetic loss function  [called when mode='synthetic']
# ---------------------------------------------------------------------------

def _synthetic_path_gate(ss, st, n, perturbed_nodes, deg_matrix_csr, L,
                         min_reach_frac=0.5, sample_sources=30):
    """
    Hard reachability gate for synthetic mode.

    An L-layer message-passing GNN can move information at most L hops. For a
    sample of perturbed sources, BFS to depth L and measure the fraction of
    each source's DEGs reachable within L hops. If the mean reachable fraction
    is below min_reach_frac, most perturbation->DEG relationships are beyond
    the GNN's structural reach and the graph is rejected.

    Cheap BFS-to-depth-L (frontier expansion), not full shortest paths.
    Returns True to shatter.
    """
    ne = len(ss)
    if ne == 0 or perturbed_nodes is None or len(perturbed_nodes) == 0:
        return False
    A = sp.csr_matrix((np.ones(ne, dtype=np.int8), (ss, st)), shape=(n, n))
    n_rows = deg_matrix_csr.shape[0]

    pn = np.asarray(perturbed_nodes)
    if len(pn) > sample_sources:
        rng = np.random.default_rng(0)
        pn = rng.choice(pn, size=sample_sources, replace=False)

    reach_fracs = []
    for src in pn:
        src = int(src)
        if src >= n_rows:
            continue
        degs = deg_matrix_csr[src, :].indices
        if len(degs) == 0:
            continue
        visited = np.zeros(n, dtype=bool)
        frontier = np.array([src], dtype=np.int64)
        visited[src] = True
        for _ in range(int(L)):
            if frontier.size == 0:
                break
            nbrs = np.unique(A[frontier].indices)
            new = nbrs[~visited[nbrs]]
            if new.size == 0:
                break
            visited[new] = True
            frontier = new
        reach_fracs.append(float(visited[degs].mean()))

    if not reach_fracs:
        return False
    return float(np.mean(reach_fracs)) < float(min_reach_frac)

def _p_smooth_syn(par, obs, ub, lw, buffer_frac=0.10, sharpness=5.0):
    """Smooth boundary penalty — same logic as the organic _p_smooth closure
    but defined at module scope so calculate_synthetic_loss can call it."""
    b = ub.get(par, [0., 1.])
    w = _safe(lw.get(par, 1.))
    o = _safe(obs, 0.)
    bound_width = max(abs(b[1] - b[0]), 1e-6)
    buffer = bound_width * buffer_frac
    if b[0] <= o <= b[1]:
        return 0.
    if o < b[0]:
        raw_dist   = (b[0] - o) / max(abs(b[0]), 1e-6)
        dist_beyond = max(0., (b[0] - o) - buffer)
    else:
        raw_dist   = (o - b[1]) / max(abs(b[1]), 1e-6)
        dist_beyond = max(0., (o - b[1]) - buffer)
    base_penalty = raw_dist ** 2
    onset = 1.0 / (1.0 + np.exp(-sharpness * (dist_beyond / bound_width)))
    return w * min(base_penalty, 4.0) * onset

def calculate_synthetic_loss(ss, st, sw, n, od, active, kappa_base, ub, lw,
                              deg_matrix_csr=None, perturbed_nodes=None,
                              gene_community_labels=None, gene_features=None,
                              compute_expensive=False, exact_spectral=False):
    """
    Synthetic utopian loss for SPECTRA/FAGCN-targeted graphs.

    Evaluates six graph properties against synthetic_utopian_bounds:
      epr_k          — edge precision against known DEGs
      weight_entropy — Shannon entropy of edge weight distribution
      source_conc    — mean outdegree of active source genes
      heterophily    — cross-community edge fraction (proxy)
      spectral_gap   — λ₂ of normalised Laplacian (if compute_expensive=True)

    Returns (loss_scalar, topology_dict) — same structure as
    calculate_utopia_loss so run_dash_and_score can dispatch transparently.

    Backward-compatible: all new parameters are optional kwargs.
    """
    ne = len(ss)

    # 1. Weight entropy — trivially fast
    w_ent = 0.0
    if ne > 0 and len(sw) > 0:
        try:
            counts, _ = np.histogram(sw, bins=50)
            p = counts.astype(np.float64); p /= max(p.sum(), 1); p = p[p > 0]
            w_ent = float(-np.sum(p * np.log2(p)))
        except Exception:
            pass

    # 2. Source concentration — O(n_genes) bincount already in od
    n_src  = int(np.count_nonzero(od[:n])) if n > 0 else 1
    s_conc = float(ne) / max(n_src, 1)

    # 3. EPR@k — O(perturbed × k), requires deg_matrix_csr
    epr_k = 0.0
    if (deg_matrix_csr is not None and perturbed_nodes is not None
            and ne > 0):
        try:
            from collections import defaultdict
            adj = defaultdict(list)
            for src, tgt in zip(ss.tolist(), st.tolist()):
                adj[src].append(tgt)
            n_rows  = deg_matrix_csr.shape[0]
            scores  = []
            for src in perturbed_nodes:
                src = int(src)
                out = adj.get(src, [])
                k   = len(out)
                if k == 0 or src >= n_rows:
                    continue
                row_dense = np.asarray(
                    deg_matrix_csr[src, :].todense()).ravel()
                scores.append(int(np.sum(row_dense[out] > 0)) / k)
            epr_k = float(np.mean(scores)) if scores else 0.0
        except Exception:
            pass

    # 4. Heterophily — feature-based (preferred) or community-based (fallback).
    #    Feature-based: mean cosine DISTANCE between source/target gene feature
    #    vectors across each edge. This measures heterophily in the same feature
    #    space SPECTRA consumes, with no dependence on arbitrary community
    #    labels. gene_features must be pre-L2-normalised (unit rows) so that the
    #    row dot product is the cosine similarity. (GOKU, arXiv:2506.16110, 2025,
    #    selects edges with exactly this feature-similarity x effective-resistance
    #    coupling; FAGCN's high-pass filters exploit the heterophilic — high
    #    cosine-distance — edges this rewards.)
    het = 0.0
    if gene_features is not None and ne > 0:
        try:
            fs  = gene_features[ss]
            ft  = gene_features[st]
            cos = np.sum(fs * ft, axis=1)
            het = float(np.mean(1.0 - cos))   # cosine distance = dissimilarity
        except Exception:
            pass
    elif gene_community_labels is not None and ne > 0:
        try:
            het = float(np.mean(gene_community_labels[ss]
                                != gene_community_labels[st]))
        except Exception:
            pass

    # 5. Spectral gap — obj_006.1 REMOVED from the primary synthetic loss (superseded
    #    by the source->DEG eff_resist quantity, which is the oversquashing metric with
    #    a theory-certain direction). The eigsh solve is also the expensive per-candidate
    #    step; dropping it keeps the primary cheap. `gap` is reported only (fallback).
    gap = _safe(ub.get('spectral_gap', [0.07, 0.09])[0])

    # obj_006.1 REFIT — PRIMARY synthetic loss is the FEASIBILITY gate only: epr_k
    # (reachability) as a REACHABLE floor. The three mis-grounded targets (heterophily,
    # weight_entropy, source_conc) and spectral_gap are REMOVED from the loss (session-18
    # lit pass: FAGCN is frequency-ADAPTIVE -> no ideal heterophily; the others
    # unsupported). epr_k's MAXIMISATION and the eff_resist SWEET-SPOT anti-cheat are the
    # SECONDARY (Derringer-Suich among 0-loss graphs; refinement.py) -- eff_resist needs a
    # per-candidate JL/LU solve (~1-3s) so it CANNOT live in this per-candidate hot loop.
    # This is exactly the design principle: biology/feasibility = flat band (here epr_k
    # floor), synthetic desirability = graded, maximised within it (there).
    t_epr = _p_smooth_syn("epr_k", epr_k, ub, lw)
    raw = _safe(t_epr)
    loss = np.sqrt(max(raw, 0.))
    topo = {
        'epr_k':          _safe(epr_k),
        # reported-but-not-in-loss (disable, don't delete); computed cheaply above
        'weight_entropy': _safe(w_ent),
        'source_conc':    _safe(s_conc),
        'heterophily':    _safe(het),
        'spectral_gap':   _safe(gap),
        'eff_resist':     float('nan'),   # scored in the secondary, not the primary
        # Organic keys set to 0 so result dict always has both key sets
        'alpha': 1., 'Gini': 0., 'gini_in': 0., 'Q': 0., 'C': 0., 'rho': 0., 'S_max': 0.,
    }
    return loss, topo

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
    source_impact = np.clip(source_impact, 1.0, 10.0)
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

def compute_deg_row_prior(deg_row_sums, deg_out_parent, perturbation_labels,
                           name_to_idx, n_genes, phi=0.5):
    """
    DEG row-sum prior rho_s: per-gene causal output efficiency prior.
 
    rho_s captures how causally productive each perturbed gene is per
    outgoing edge — DEGs caused per available outgoing edge. This is the
    direct complement to pi_s, but with a strict floor at 1.0 (boost only,
    never penalty). Genes below the mean efficiency get rho=1.0 (neutral).
 
    This fixes the primary failure mode of pi_s: at psi=1.74, genes with
    below-average DEG counts receive pi_s < 1.0, which actively penalizes
    their outgoing edges. rho_s counteracts this by ensuring that any gene
    with non-zero causal evidence gets at least a neutral score.
 
    Properties:
    - Global: defined for all n_genes, floor = 1.0 for non-perturbed genes
    - Boost-only: strictly >= 1.0, no gene is penalized
    - Bounded: [1.0, ~1.8] at phi=0.5 — same scale as chi_s
    - Generalizable: val/test genes with no row-sum data get 1.0 (neutral)
 
    Parameters
    ----------
    deg_row_sums      : np.ndarray (n_genes,) — row sums of Phase 0 DEG matrix
    deg_out_parent    : np.ndarray (n_genes,) — out-degree in raw parent graph
    perturbation_labels : array-like of str
    name_to_idx       : dict gene_name -> index
    n_genes           : int
    phi               : float, fixed exponent (default 0.5, not a hyperparameter)
    """
    rho = np.ones(n_genes, dtype=np.float64)
 
    if deg_row_sums is None or len(deg_row_sums) != n_genes:
        return rho
 
    row = np.asarray(deg_row_sums, dtype=np.float64)
    deg_out = (np.asarray(deg_out_parent, dtype=np.float64)
               if deg_out_parent is not None else None)
 
    # Get indices of perturbation targets
    target_idxs = [name_to_idx.get(str(l)) for l in perturbation_labels]
    target_idxs = [i for i in target_idxs if i is not None and i < n_genes]
    if len(target_idxs) == 0:
        return rho
 
    # Efficiency: DEG count normalized by out-degree (Gemini normalization)
    # This prevents hub genes with many edges from dominating by volume alone
    if deg_out is not None:
        mean_deg_out = float(np.mean(deg_out[target_idxs]))
        mean_deg_out = max(mean_deg_out, 1.0)
        efficiency = np.zeros(n_genes, dtype=np.float64)
        for idx in target_idxs:
            d = max(float(deg_out[idx]), 1.0)
            efficiency[idx] = float(row[idx]) / (d / mean_deg_out)
    else:
        efficiency = row.copy()
 
    # Mean efficiency over active perturbation targets only
    eff_vals = efficiency[target_idxs]
    active = eff_vals[eff_vals > 0]
    if len(active) == 0:
        return rho
 
    g = float(np.mean(active))
 
    # Log-damped boost, floor at 1.0 — identical pattern to compute_chi_prior
    above = efficiency > g
    rho_raw = np.ones(n_genes, dtype=np.float64)
    rho_raw[above] = np.maximum(1.0, 1.0 + np.log(efficiency[above] / g))
    rho = np.power(rho_raw, float(phi))
 
    n_boosted = int((rho > 1.05).sum())
    print(f"  rho_prior: {n_boosted} genes boosted above 1.05 "
          f"(range [{rho.min():.3f}, {rho.max():.3f}])")

    return rho


def compute_rdf_prior(sources_arr, targets_arr, weights_arr, gene_features,
                       n_genes, top_n=50):
    """
    Regulatory Diversity Factor: for each source gene, measure how spread its
    top-N targets are in expression feature space (the 50-dim SVD gene
    features from utils.build_gene_features, already L2-normalised so a row
    dot product is cosine similarity).

    High RDF = targets span multiple expression programs (a genuine
    multi-program TF, e.g. a GATA/SOX-family regulator coordinating distinct
    metabolic/structural/signalling targets).
    Low RDF  = targets cluster in one co-expression bloc (a chromatin/
    cell-cycle hub like KAT2A/METTL3 whose high importance score reflects
    indirect effects through a single shared program, not genuine regulatory
    breadth). LightGBM/elastic-net importance, kappa, chi_prior and rho_prior
    cannot distinguish these two cases -- both score high on outgoing weight,
    both have wide CRISPRi footprints, both participate in FFLs.

    Deliberately deviates from the original 17 June 2026 design (source-to-
    target distance: mean(1 - cosine_sim(features_s, features_t))). Source-
    to-target distance mostly measures whether a TF is co-expressed with its
    own targets, which is not very informative -- TFs are often not
    co-expressed with their targets by design. What distinguishes a
    multi-program TF from a bloc regulator is whether its OWN targets are
    diverse AMONG THEMSELVES: mean pairwise cosine distance among a source's
    top-N targets (by edge weight) measures this directly. A multi-program
    TF's targets span multiple dissimilar modules (high mean pairwise
    distance); a bloc regulator's targets all sit in the same module (low
    mean pairwise distance).

    This connects to FAGCN's heterophilic message-passing (SPECTRA):
    discounting co-expression-bloc hubs via RDF increases the fraction of
    edges that cross expression-space boundaries, directly increasing the
    heterophily of the resulting graph -- the same biological signal the
    synthetic-mode loss's heterophily term already rewards.

    One-time precomputation (called once per search run, not per-eval) --
    a Python loop over n_genes with a vectorized (top_n, top_n) cosine
    similarity matrix per gene is fast enough here (no scipy.optimize, no
    sklearn call per gene) without needing a fully gene-vectorized rewrite.

    Parameters
    ----------
    sources_arr, targets_arr : np.ndarray (n_edges,) int -- parent graph edges
    weights_arr               : np.ndarray (n_edges,) float -- edge importance
    gene_features             : np.ndarray (n_genes, k) -- L2-normalised rows
    n_genes                   : int
    top_n                     : int -- consider each source's top-N targets by
                                weight (default 50)

    Returns
    -------
    rdf : np.ndarray (n_genes,) float64 in [0.1, 1.0]. Genes with <3 outgoing
    edges keep the neutral default of 1.0 (RDF^nu = 1 regardless of nu).
    """
    rdf = np.ones(n_genes, dtype=np.float64)

    for s in range(n_genes):
        mask = sources_arr == s
        n_out = int(mask.sum())
        if n_out < 3:
            continue
        t_idx = targets_arr[mask]
        t_wgt = weights_arr[mask]
        top_k = min(top_n, len(t_idx))
        top_t = t_idx[np.argsort(-t_wgt)[:top_k]]

        # Mean pairwise cosine distance among the source's own top-N targets.
        # Rows are unit-norm, so cos_sim = t_feats @ t_feats.T directly.
        t_feats = gene_features[top_t]
        sim_matrix = t_feats @ t_feats.T
        n_pairs = top_k * (top_k - 1)
        if n_pairs == 0:
            continue
        total_sim = sim_matrix.sum() - np.trace(sim_matrix)
        rdf[s] = float(1.0 - total_sim / n_pairs)

    rdf_max = rdf.max()
    if rdf_max > 1e-10:
        rdf = np.clip(rdf / rdf_max, 0.1, 1.0)

    n_diverse = int((rdf > 0.7).sum())
    n_bloc = int((rdf < 0.3).sum())
    print(f"  rdf_prior: {n_diverse} genes diverse (RDF>0.7), "
          f"{n_bloc} bloc-like (RDF<0.3), range [{rdf.min():.3f}, {rdf.max():.3f}]")
    return rdf

 
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

def build_causal_weight_array(experimental_df, sources, targets,
                               gene_names, n_genes, W_q_existing):
    """
    Build a W_q replacement array using rank-normalized md_confidence.
 
    For each source gene s, the edges (s->t) that have md_confidence > 0
    are rank-normalized within that source's own md_confidence distribution
    to produce values in (0, 1]. These rank-normalized values replace the
    corresponding W_q entries.
 
    Why rank-normalize per source?
    - md_confidence from 3_consolidate_grn.py uses global min-max normalization,
      which is dominated by a handful of outlier edges. Most nonzero values
      compress near zero (observed mean ~0.011), making direct substitution
      equivalent to zeroing those edges out.
    - Per-source rank normalization maps the md_confidence ordering onto the
      same [0,1] scale as W_q, preserving the relative causal evidence ranking
      within each source gene's outgoing edges without distorting cross-source
      budget competition.
 
    Edges without md_confidence data keep their original W_q value unchanged.
    """
    W_q_final = W_q_existing.copy()
 
    if experimental_df is None or 'md_confidence' not in experimental_df.columns:
        print("  W_q_causal: md_confidence not found — W_q unchanged")
        return W_q_final
 
    src_col = experimental_df.columns[0]
    tgt_col = experimental_df.columns[1]
    name_to_idx = {name: i for i, name in enumerate(gene_names)}
 
    exp_src_idx = (experimental_df[src_col].map(name_to_idx)
                   .fillna(-1).values.astype(np.int64))
    exp_tgt_idx = (experimental_df[tgt_col].map(name_to_idx)
                   .fillna(-1).values.astype(np.int64))
    md_conf_vals = experimental_df['md_confidence'].values.astype(np.float64)
 
    valid = (exp_src_idx >= 0) & (exp_tgt_idx >= 0) & (md_conf_vals > 0)
    exp_src_valid = exp_src_idx[valid]
    exp_tgt_valid = exp_tgt_idx[valid]
    md_valid = md_conf_vals[valid]
 
    # Rank-normalize md_confidence per source gene so values live in (0, 1]
    # matching the W_q scale. Use the same midpoint formula as W_q:
    #   rank_norm = 1 - (rank_ascending + 0.5) / n_in_group
    md_rank_normalized = np.zeros(len(md_valid), dtype=np.float64)
    unique_sources = np.unique(exp_src_valid)
    for src in unique_sources:
        mask = exp_src_valid == src
        vals = md_valid[mask]
        n = len(vals)
        if n == 1:
            md_rank_normalized[mask] = 0.75  # single edge: give it a solid mid-high rank
            continue
        # argsort ascending, then compute rank position
        order = np.argsort(vals)           # ascending: worst md first
        rank_asc = np.empty(n, dtype=np.float64)
        rank_asc[order] = np.arange(n, dtype=np.float64)
        # highest md_confidence gets rank_asc = n-1 → rank_norm near 1.0
        md_rank_normalized[mask] = 1.0 - (rank_asc + 0.5) / n
 
    # Build sorted lookup for matching against the candidate edge pool
    exp_keys = exp_src_valid * np.int64(n_genes) + exp_tgt_valid
    sort_order = np.argsort(exp_keys)
    keys_sorted = exp_keys[sort_order]
    rank_sorted = md_rank_normalized[sort_order]
 
    query_keys = sources.astype(np.int64) * np.int64(n_genes) + targets.astype(np.int64)
    ins = np.searchsorted(keys_sorted, query_keys)
    ins = np.clip(ins, 0, len(keys_sorted) - 1)
    matched = keys_sorted[ins] == query_keys
 
    W_q_final[matched] = rank_sorted[ins[matched]]
 
    n_replaced = int(matched.sum())
    old_mean = float(W_q_existing[matched].mean()) if n_replaced > 0 else 0.0
    new_mean = float(W_q_final[matched].mean()) if n_replaced > 0 else 0.0
    print(f"  W_q_causal: {n_replaced:,}/{len(sources):,} edges have W_q replaced by "
          f"rank-normalized md_confidence (old mean={old_mean:.3f} → new mean={new_mean:.3f})")
    print(f"  W_q_causal: rank-norm range [{W_q_final[matched].min():.3f}, "
          f"{W_q_final[matched].max():.3f}] — comparable to W_q scale")
 
    return W_q_final


# ---------------------------------------------------------------------------
# Edge selection with per-gene soft kappa
# ---------------------------------------------------------------------------

# ══════════════════════════════════════════════════════════════════════════════
# engine.py — CRITICAL FIX: restore compute_dynamic_topology
#
# This function was accidentally deleted during the v9.5/v9.6 edits.
# Its absence causes every single DASH evaluation to crash with:
#   "name 'compute_dynamic_topology' is not defined"
#
# INSERT this entire block immediately before the line:
#   # Edge selection with per-gene soft kappa
#   # ------------------------------------
#   def select_edges(...)
#
# That is: paste it at line 480 in engine__42_.py, just before select_edges.
# ══════════════════════════════════════════════════════════════════════════════

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

def select_edges(omega, W, src, tgt, pert_nodes, n, lam,
                 per_gene_kappa, kappa_base):
    """
    Select edges by descending omega score, respecting per-gene hub caps.

    v9.2 changes (E2, E7):
      E2: Protection budget capped at 5% of total edge budget.
          On VCC (96 perts) this is identical to previous behaviour (<0.3%
          of budget). On K562 (3454 perts) the old code reserved up to
          32.7% of budget as forced inclusions regardless of omega score,
          suppressing hub concentration and depressing S_max.
      E7: Kappa backfill no longer blanket-excludes all edges from capped
          sources. Only already-selected edge indices are masked, so a
          source capped by exactly one edge can still contribute its
          next-best candidates in the backfill pass. This prevents
          systematic underfill at high λ.

    Vectorized (no Python loop over pert_nodes or over over-capacity genes):
    src/omega arrive grouped source-ascending, omega-descending-within-source
    (every caller in this codebase presorts this way -- see
    FungiGPUContext's docstring and run_dash_and_score's
    np.lexsort((-num, ss))), so "top-3-by-omega per pert gene" is just the
    first <=3 positions of that gene's contiguous segment, and per-gene
    kappa rank is a group-boundary cumsum on a fresh (src,-omega) sort of
    just the budget-selected subset. At the real RPE1 5k scale (868
    perturbation nodes) the old Python loop over pert_nodes alone cost
    ~220ms/eval; this is ~3-5x faster there and identical at K562 scale
    (3454 perts, ~13x faster) -- verified exact-edge-set-equivalent against
    the original loop-based version across 30 randomized trials including
    the real n_pert values (_verify_step9_vectorized_select_edges.py).
    """
    budget = int(np.round(n * lam))
    n_total = len(omega)

    effective_caps = np.maximum(
        (per_gene_kappa * kappa_base * n).astype(np.int64), 1)

    if len(pert_nodes) > 0:
        uniq_src, first_idx, counts = np.unique(
            src, return_index=True, return_counts=True)
        pert_arr = np.asarray(pert_nodes, dtype=np.int64)
        pos = np.clip(np.searchsorted(uniq_src, pert_arr), 0, len(uniq_src) - 1)
        found = uniq_src[pos] == pert_arr
        pert_pos = pos[found]
        starts = first_idx[pert_pos]
        sizes = np.minimum(counts[pert_pos], 3)

        offs = np.arange(3)
        idx_grid = starts[:, None] + offs[None, :]
        valid = offs[None, :] < sizes[:, None]
        prot_raw = np.unique(idx_grid[valid].astype(np.int64))
    else:
        prot_raw = np.array([], dtype=np.int64)

    # E2: cap protection at 5% of budget — prevents large pert panels
    # (K562: 3454 genes) from consuming the budget with forced inclusions
    max_prot = max(1, int(budget * 0.05))
    if len(prot_raw) > max_prot:
        # Keep the highest-omega protected edges within the cap
        prot = prot_raw[np.argsort(omega[prot_raw])[-max_prot:]]
    else:
        prot = prot_raw

    rem = budget - len(prot)
    if rem > 0:
        m = np.ones(n_total, dtype=bool)
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
    over_cap = nc > effective_caps

    if np.any(over_cap):
        om_sel = omega[sel]
        order = np.lexsort((-om_sel, ss))
        ss_s = ss[order]
        sel_s = sel[order]

        change = np.empty(len(ss_s), dtype=bool)
        change[0] = True
        change[1:] = ss_s[1:] != ss_s[:-1]
        group_start = np.where(change)[0]
        group_sizes = np.diff(np.append(group_start, len(ss_s)))
        rank = np.arange(len(ss_s)) - np.repeat(group_start, group_sizes)

        caps_s = effective_caps[ss_s]
        sel = sel_s[rank < caps_s]

        freed = budget - len(sel)
        if freed > 0:
            # E7: mask only already-selected indices, not all edges from
            # capped sources. A source capped by 1 edge may still have
            # strong candidates remaining; blanket exclusion caused
            # systematic underfill at high λ.
            cm = np.ones(n_total, dtype=bool)
            cm[sel] = False
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
    """
    Returns (is_shattered, reason, gwcc_fraction). gwcc_fraction is the
    giant-weakly-connected-component fraction computed here (None if never
    reached, e.g. an earlier density/orphan rejection) -- callers reuse this
    instead of recomputing connected_components a second time after scoring.
    """
    ne = len(ss)
    if ne > cfg.get("max_edge_count", 500000):
        return True, "density_collapse", None
    if (n - active) / max(n, 1) > cfg.get("max_orphan_fraction", 0.70):
        return True, "orphan_collapse", None
    gwcc_fraction = None
    if ne > 0 and active > 0:
        try:
            G = sp.coo_matrix((np.ones(ne), (ss, st)), shape=(n, n))
            _, lb = connected_components(csgraph=G, directed=False,
                                         return_labels=True)
            gwcc_fraction = _safe(np.bincount(lb).max() / n)
            if gwcc_fraction < cfg.get("min_gwcc_fraction", 0.30):
                return True, "gwcc_percolation", gwcc_fraction
        except Exception:
            return True, "gwcc_percolation", None
    min_clust = cfg.get("min_clustering", None)
    if min_clust is not None and ne > 50:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_u = ig.Graph(n=n, edges=edges, directed=True).as_undirected(
                mode="collapse")
            cc = ig_u.transitivity_undirected()
            if np.isfinite(cc) and cc < min_clust:
                return True, "clustering_collapse", gwcc_fraction
        except Exception:
            pass
    return False, None, gwcc_fraction


# ---------------------------------------------------------------------------
# Utopia loss
# ---------------------------------------------------------------------------

def assortativity_fast(surv_s, surv_t, n_genes):
    """
    Pearson correlation of (out-degree[source], in-degree[target]) across
    survived edges -- exact replacement for igraph's assortativity_degree().
    igraph's call is slow mainly from building a full Graph object, not the
    underlying math; surv_s/surv_t are already the small final selected-edge
    set by this point in the pipeline (not Ne), so this stays in plain numpy
    rather than round-tripping a tiny array through the GPU.
    """
    if len(surv_s) < 2:
        return 1.0
    out_deg = np.bincount(surv_s, minlength=n_genes).astype(np.float64)
    in_deg = np.bincount(surv_t, minlength=n_genes).astype(np.float64)
    x = out_deg[surv_s]
    y = in_deg[surv_t]
    xm, ym = x.mean(), y.mean()
    denom = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def hill_alpha_fast(od, n_genes, xmin=6, cap_frac=0.15):
    """
    Closed-form Hill/CSN MLE for the power-law exponent at a FIXED xmin --
    not an exact match to calculate_utopia_loss's primary path, which uses
    powerlaw.Fit's KS-minimising auto-xmin search (falling back to xmin=6
    only when that leaves under 50 tail nodes). This always uses xmin=6,
    i.e. always takes that fallback branch. Verify empirically (see
    _verify_step7_topology.py) whether the resulting alpha is close enough
    for your tolerance -- don't assume it from this docstring.
    """
    cap = int(n_genes * cap_frac)
    cd = od[(od > 0) & (od < cap)]
    if len(cd) < 20:
        return 1.0
    tail = cd[cd >= xmin].astype(np.float64)
    if len(tail) < 20:
        return 1.0
    return float(1.0 + len(tail) / np.sum(np.log(tail / (xmin - 0.5))))


def fast_auto_xmin_alpha(od, n_genes, cap_frac=0.15, min_tail=50, fallback_xmin=6):
    """
    Vectorized auto-xmin + alpha fit via CSN discrete MLE (eq. 3.7, xmin-0.5).
    No Python loop over candidates. No scipy.optimize. No powerlaw.Fit.
    Same KS-minimising auto-xmin SELECTION rule and xmin=6 fallback as the
    original loop-based version (verified numerically equivalent on Pareto
    and near-uniform-random synthetic out-degree -- see
    _verify_step8_vectorized_alpha.py). Accurate to ~1% for alpha in
    [2.0, 2.5] and xmin >= 6 -- the biological GRN regime (Clauset, Shalizi
    & Newman 2009).
    """
    cap = int(n_genes * cap_frac)
    cd = np.sort(od[(od > 0) & (od < cap)]).astype(np.float64)
    n_total = cd.size
    if n_total < 20:
        return 1.0

    candidates, first_idx = np.unique(cd, return_index=True)
    if candidates.size < 3:
        return 1.0

    counts_ge = n_total - first_idx
    keep = counts_ge >= 20
    candidates = candidates[keep]
    first_idx = first_idx[keep]
    counts_ge = counts_ge[keep]
    if candidates.size == 0:
        return 1.0

    # Vectorized Hill MLE via suffix log-sum (no loop over candidates):
    # for candidate k with tail starting at first_idx[k]:
    #   sum_i log(x_i / (xmin - 0.5)) = suffix_logsum[first_idx[k]]
    #                                   - n_tail[k] * log(xmin - 0.5)
    log_cd = np.log(cd)
    suffix_logsum = np.cumsum(log_cd[::-1])[::-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = suffix_logsum[first_idx] - counts_ge * np.log(candidates - 0.5)

    valid_fit = np.isfinite(denom) & (denom > 0)
    if not np.any(valid_fit):
        return 1.0

    alpha_arr = np.full(candidates.size, np.nan)
    alpha_arr[valid_fit] = 1.0 + counts_ge[valid_fit] / denom[valid_fit]

    # Vectorized KS distance: (K, n_total) broadcast, restricted to
    # candidates with a valid alpha fit, to limit memory.
    valid_idx = np.where(valid_fit)[0]
    fi_v = first_idx[valid_idx]
    cg_v = counts_ge[valid_idx]
    ca_v = candidates[valid_idx]
    al_v = alpha_arr[valid_idx]

    col = np.arange(n_total)
    valid_mask = col[None, :] >= fi_v[:, None]
    rank_in_tail = col[None, :] - fi_v[:, None]

    emp_ccdf = 1.0 - rank_in_tail / cg_v[:, None]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        fit_ccdf = np.power(cd[None, :] / ca_v[:, None],
                            1.0 - al_v[:, None])

    diff = np.abs(emp_ccdf - fit_ccdf)
    diff[~valid_mask] = -np.inf
    diff[~np.isfinite(diff)] = -np.inf
    ks = diff.max(axis=1)

    best_local = int(np.argmin(ks))
    best_alpha = float(al_v[best_local])
    best_xmin = ca_v[best_local]

    # Min-tail guard (same as the original fallback logic).
    if int(np.sum(cd >= best_xmin)) < min_tail:
        fb_s = int(np.searchsorted(cd, float(fallback_xmin), side='left'))
        m = n_total - fb_s
        if m >= 10:
            d_fb = suffix_logsum[fb_s] - m * np.log(float(fallback_xmin) - 0.5)
            if d_fb > 0:
                return float(1.0 + m / d_fb)
    return best_alpha


def clustering_wedge_sample(surv_s, surv_t, n_genes, n_samples=20000, rng=None):
    """
    Approximate global clustering coefficient (transitivity) by wedge
    sampling -- replaces igraph's exact transitivity_undirected() during
    search. Numpy, not GPU, for the same reason as assortativity_fast.
    Exact igraph transitivity is still used for the final champion/finalist
    rebuild in the notebook (that code path doesn't go through here).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(surv_s) < 3:
        return 0.0

    a = np.concatenate([surv_s, surv_t])
    b = np.concatenate([surv_t, surv_s])
    codes = np.minimum(a, b).astype(np.int64) * n_genes + np.maximum(a, b).astype(np.int64)
    codes = np.unique(codes)
    if len(codes) < 3:
        return 0.0
    u = (codes // n_genes).astype(np.int64)
    v = (codes % n_genes).astype(np.int64)

    deg = np.bincount(np.concatenate([u, v]), minlength=n_genes)
    wedge_weight = deg.astype(np.int64) * (deg.astype(np.int64) - 1) // 2
    good = wedge_weight > 0
    if not good.any():
        return 0.0

    all_src = np.concatenate([u, v])
    all_dst = np.concatenate([v, u])
    order = np.argsort(all_src, kind="stable")
    all_src, all_dst = all_src[order], all_dst[order]
    rowptr = np.zeros(n_genes + 1, dtype=np.int64)
    rowptr[1:] = np.cumsum(np.bincount(all_src, minlength=n_genes))

    centers_pool = np.nonzero(good)[0]
    probs = wedge_weight[centers_pool].astype(np.float64)
    probs /= probs.sum()
    chosen = rng.choice(centers_pool, size=n_samples, replace=True, p=probs)
    d = deg[chosen].astype(np.int64)

    r1 = (rng.random(n_samples) * d).astype(np.int64)
    r2 = (rng.random(n_samples) * (d - 1)).astype(np.int64)
    r2 = r2 + (r2 >= r1).astype(np.int64)

    offs = rowptr[chosen]
    nb1 = all_dst[offs + r1]
    nb2 = all_dst[offs + r2]

    a2 = np.minimum(nb1, nb2).astype(np.int64)
    b2 = np.maximum(nb1, nb2).astype(np.int64)
    query_codes = a2 * n_genes + b2
    pos = np.clip(np.searchsorted(codes, query_codes), 0, len(codes) - 1)
    closed = codes[pos] == query_codes
    return float(closed.mean())


def fixed_partition_modularity_fast(surv_s, surv_t, membership, n_genes):
    """
    Directed modularity (Leicht-Newman 2008 form) against a FIXED, already-
    detected community partition -- O(edges + n_communities), no Leiden
    re-run per evaluation.

        Q = sum_c (e_c / m) - sum_c (a_c * b_c) / m^2

    where m = total edges, e_c = edges with both endpoints in community c,
    a_c/b_c = total out-/in-degree of every node assigned to c (not just
    those with surviving edges). This is the exact modularity value for the
    given partition -- not an approximation -- since the partition itself
    is fixed and reused (SCBER's Leiden membership from Phase 2), unlike
    alpha/C/rho which are genuinely recomputed from scratch each eval.

    membership : np.ndarray int, shape (n_genes,) -- community index per
        gene, e.g. SCBER's effective_resistance._detect_communities() output.
    """
    m = len(surv_s)
    if m == 0:
        return 0.0
    n_comm = int(membership.max()) + 1
    od = np.bincount(surv_s, minlength=n_genes).astype(np.float64)
    idd = np.bincount(surv_t, minlength=n_genes).astype(np.float64)
    a_c = np.bincount(membership, weights=od, minlength=n_comm)
    b_c = np.bincount(membership, weights=idd, minlength=n_comm)
    same_comm = membership[surv_s] == membership[surv_t]
    e_frac = float(np.count_nonzero(same_comm)) / m
    expected = float(np.sum(a_c * b_c)) / (m * m)
    return float(e_frac - expected)


def calculate_utopia_loss(ss, st, sw, n, od, active, kappa_base, ub, lw,
                          fast_topology=False, profile=False,
                          gene_community_labels=None):
    """
    fast_topology=False (default): EXACT original behavior, unchanged --
    igraph for assortativity/clustering, powerlaw.Fit for alpha.
    fast_topology=True: used by the GPU batch path. Swaps in
    assortativity_fast/fast_auto_xmin_alpha/clustering_wedge_sample.

    gene_community_labels: per-gene community index array (SCBER's Leiden
    membership, reused not recomputed), used to score modularity (Q) via
    fixed_partition_modularity_fast(). Q was removed entirely in session 4
    because SCBER's bridge promotion structurally fought it with no
    counterbalancing force; session 5 restores it as a genuine 7th organic
    target now that m_intra (the 8th searched hyperparameter, run_dash_and_
    score/build_graph_from_params) gives the optimizer a positive pull
    toward intra-community structure. in-degree Gini (gini_in) remains its
    own separate target -- both are live now, not Q-replaces-gini_in.
    Q falls back to a neutral-but-out-of-window 1.0 when no partition is
    available (SCBER disabled), same defensive pattern as rho's ro=1.0 init.

    profile=True prints a one-line timing breakdown of the alpha sub-step
    (used by run_dash_and_score_gpu_batch for its first few evaluations only).
    """
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

    # E1: alpha measurement — KS-minimising auto-xmin, matching diagnostics v11.0.
    # Fixed xmin=2 biased alpha ~0.30 units low (Clauset, Shalizi & Newman 2009
    # SIAM Rev 51:661). The probe uses auto-xmin; the loss must match.
    #
    # fast_topology=True uses fast_auto_xmin_alpha (vectorized CSN closed-form,
    # eq 3.7) instead of powerlaw.Fit's per-candidate scipy.optimize.minimize --
    # the dominant per-evaluation cost (70-1000ms/call at n_genes=5000). Three
    # independent deep-research passes (FUNGI Eval 16) confirmed the CSN
    # discrete approximation is accurate to ~1% for alpha in [2.0, 2.5] and
    # xmin >= 6 (the biological GRN regime); the earlier session's adversarial
    # near-uniform-random divergence (4.5-6.8) only occurs on degree shapes
    # that check_shatter's GWCC/orphan/density gates reject before reaching
    # this point. fast_topology=False (finalist/champion rebuild) is untouched.
    ao = 1.0
    t_alpha0 = time.perf_counter() if profile else None
    if fast_topology:
        try:
            ao = fast_auto_xmin_alpha(od, n, cap_frac=0.15, min_tail=50,
                                      fallback_xmin=6)
        except Exception:
            pass
    else:
        try:
            cap = int(n * 0.15)
            cd = od[(od > 0) & (od < cap)]
            if len(cd) >= 20 and len(np.unique(cd)) >= 3:
                fit_ao = powerlaw.Fit(cd, discrete=True, verbose=False)
                xmin_ao = fit_ao.power_law.xmin
                if int(np.sum(cd >= xmin_ao)) < 50:
                    fit_ao = powerlaw.Fit(cd, xmin=6, discrete=True, verbose=False)
                ao = _safe(fit_ao.power_law.alpha, 1.)
        except Exception:
            pass
    if profile:
        tag = "fast_auto_xmin_alpha" if fast_topology else "powerlaw.Fit"
        print(f"[FUNGI PROFILE]     alpha ({tag}): "
              f"{1000 * (time.perf_counter() - t_alpha0):.2f} ms")
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
    # kernel_firepower_v2: env-driven S_max loss weighting. Default 0.25 reproduces stock byte-identical
    # (regression gate). Lowering kappa_frac raises the S_max BAND penalty weight (1-kappa_frac) AND
    # reduces the kappa_excess anti-hub penalty -- both help LIFT a too-low S_max toward literature.
    kappa_frac = float(os.environ.get("FUNGI_KAPPA_FRAC", "0.25"))
    full_w = _safe(lw.get("S_max", 1.0), 1.0)
    bound_penalty = _p_smooth("S_max", smo, ub, {"S_max": full_w * (1 - kappa_frac)})
    kappa_excess = max(0., (smo - kappa_base) / max(kappa_base, 1e-6))
    kappa_penalty = full_w * kappa_frac * kappa_excess ** 2
    ts = bound_penalty + kappa_penalty

    # In-degree Gini — regulatory-input concentration. The organic loss's
    # "module" slot (modularity/Q was removed entirely in session 4 -- SCBER's
    # cross-community bridge promotion structurally fights modularity, so no
    # probe substrate could ever close that gap; gini_in is a structural axis
    # no DASH term actively suppresses).
    gini_in = 1.0
    try:
        idd_in = np.bincount(st, minlength=n) if ne > 0 else np.zeros(n)
        idd_nz = idd_in[idd_in > 0]
        if active > 1 and idd_nz.sum() > 0 and len(idd_nz) > 1:
            sdi = np.sort(idd_nz)
            nn = len(sdi)
            gini_in = _safe((2. * np.sum(np.arange(1, nn + 1) * sdi)) /
                            (nn * np.sum(sdi)) - (nn + 1) / nn, 1.)
    except Exception:
        pass

    co, ro = 0., 1.
    tc = _safe(lw["C"], 1.)
    tr = _safe(lw["rho"], 1.)
    t_modslot = _safe(lw.get("gini_in", 1.0), 1.)

    if ne > 100 and fast_topology:
        try:
            rc = assortativity_fast(ss, st, n)
            if np.isfinite(rc):
                ro = rc
                tr = _p_smooth("rho", ro, ub, lw)
        except Exception:
            pass
        try:
            cc_val = clustering_wedge_sample(ss, st, n)
            if np.isfinite(cc_val):
                co = cc_val
                tc = _p_smooth("C", co, ub, lw)
        except Exception:
            pass
    elif ne > 100:
        try:
            edges = list(zip(ss.tolist(), st.tolist()))
            ig_g = ig.Graph(n=n, edges=edges, directed=True,
                            edge_attrs={'weight': sw.tolist()})
            try:
                rc = ig_g.assortativity_degree(directed=True)
                if np.isfinite(rc):
                    ro = rc
                    # E3: evaluate absolute rho directly against probe-set bounds.
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
            except Exception:
                pass
        except Exception:
            pass

    # The "module" loss slot: in-degree Gini.
    t_modslot = _p_smooth("gini_in", gini_in, ub, lw)

    # Q (modularity, session 5 restoration) -- exact given the FIXED SCBER
    # partition, cheap (O(edges)), so it runs identically regardless of
    # fast_topology; no community-detection re-run needed per eval.
    qo = 1.0
    if gene_community_labels is not None and ne > 0:
        try:
            qo = fixed_partition_modularity_fast(ss, st, gene_community_labels, n)
        except Exception:
            pass
    tq = _p_smooth("Q", qo, ub, lw)

    # Reciprocity (exp_010 finalization target; Garlaschelli & Loffredo 2004):
    # fraction of survived edges whose reverse also survived. Always computed
    # (cheap, reported in the topo dict); the smooth penalty is added only when
    # "reciprocity" is an active target (organic_targets.reciprocity.enabled +
    # bounds_override) -- a no-op for any config that does not enable it. Inline
    # numpy verified bit-for-bit against new_target_metrics.reciprocity_raw.
    recip = 0.0
    if ne > 0:
        try:
            codes = ss.astype(np.int64) * n + st.astype(np.int64)
            rev = st.astype(np.int64) * n + ss.astype(np.int64)
            order = np.argsort(codes, kind="stable")
            codes_s = codes[order]
            pos = np.clip(np.searchsorted(codes_s, rev), 0, ne - 1)
            has_rev = (codes_s[pos] == rev) & (ss != st)
            recip = float(has_rev.sum()) / float(ne)
        except Exception:
            pass
    trec = (_p_smooth("reciprocity", recip, ub, lw)
            if ("reciprocity" in lw and "reciprocity" in ub) else 0.0)

    raw = (_safe(ta) + _safe(tc) + _safe(t_modslot) + _safe(tg) +
           _safe(tr) + _safe(ts) + _safe(tq) + _safe(trec))
    return np.sqrt(max(raw, 0.)), {
        'alpha': _safe(ao, 1.), 'C': _safe(co),
        'gini_in': _safe(gini_in, 1.), 'Q': _safe(qo, 1.),
        'Gini': _safe(go, 1.), 'rho': _safe(ro, 1.), 'S_max': _safe(smo),
        'reciprocity': _safe(recip)}


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def compute_gpu_omega_sort_batch(params_list, ctx: "FungiGPUContext"):
    """
    The GPU-only half of run_dash_and_score_gpu_batch, factored out so the
    Ray-pipelined evaluator (search.py's SearchEvaluator._evaluate_gpu_ray)
    can do omega+sort on the GPU in the main process and dispatch the
    per-eval CPU topology work to worker processes. Returns so_batch/
    to_batch/no_batch (B, Ne) numpy arrays plus the per-eval scalar param
    lists -- deliberately omits Wo_batch (W): calculate_utopia_loss's
    fast_topology=True path never reads edge weights (assortativity_fast/
    clustering_wedge_sample take no W argument), so it isn't needed for
    search-time scoring and skipping it roughly halves the per-eval IPC
    payload to Ray workers.
    """
    B = len(params_list)
    device = ctx.log_wq.device

    betas, deltas, kappas, k_cores_eff, lams, psis, nus, m_intras = (
        [], [], [], [], [], [], [], [])
    for params in params_list:
        if len(params) >= 8:
            beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
        elif len(params) >= 7:
            beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
            m_intra = 1.0
        else:
            beta, delta, kappa_base, k_core, lam, psi = params[:6]
            nu = 0.0
            m_intra = 1.0
        k_core = max(k_core, max(5.0, lam * 0.4))
        betas.append(beta); deltas.append(delta); kappas.append(kappa_base)
        k_cores_eff.append(k_core); lams.append(lam); psis.append(psi); nus.append(nu)
        m_intras.append(m_intra)

    k_core_eff_np = np.array(k_cores_eff)
    beta_t   = torch.tensor(betas,   device=device, dtype=torch.float32)
    delta_t  = torch.tensor(deltas,  device=device, dtype=torch.float32)
    psi_t    = torch.tensor(psis,    device=device, dtype=torch.float32)
    nu_t     = torch.tensor(nus,     device=device, dtype=torch.float32)
    mintra_t = torch.tensor(m_intras, device=device, dtype=torch.float32)

    extra = _extra_coefs_from_params(params_list, ctx, device)
    omega_batch = compute_omega_batch_gpu(
        ctx, beta_t, delta_t, psi_t, nu_t, mintra_t, k_core_eff_np, extra=extra)
    order_batch = segmented_argsort_batch_gpu(ctx, omega_batch)

    ss_b = ctx.ss.unsqueeze(0).expand(B, -1)
    ts_b = ctx.ts.unsqueeze(0).expand(B, -1)
    so_batch = ss_b.gather(1, order_batch).cpu().numpy().astype(np.int32)
    to_batch = ts_b.gather(1, order_batch).cpu().numpy().astype(np.int32)
    no_batch = omega_batch.gather(1, order_batch).cpu().numpy().astype(np.float32)

    return (so_batch, to_batch, no_batch, betas, deltas, kappas, k_cores_eff,
            lams, psis, nus, m_intras)


def run_dash_and_score_gpu_batch(params_list, ctx: "FungiGPUContext",
                                 n_genes, perturbed_nodes, utopian_bounds,
                                 loss_weights, shatter_cfg, per_gene_kappa,
                                 mode='biologic', deg_matrix_csr=None,
                                 gene_community_labels=None,
                                 compute_expensive=False, gene_features=None,
                                 spectra_L=3, exact_spectral=False):
    """
    GPU-accelerated batch evaluation. Omega + ordering for all B configs are
    computed together on the GPU; everything after that (select_edges,
    _motif_repair_swap, check_shatter, calculate_utopia_loss/
    calculate_synthetic_loss) runs through the SAME unchanged CPU functions
    run_dash_and_score uses, one config at a time, just fed GPU-sorted
    arrays instead of CPU-np.lexsort ones. Returns a list of B result dicts,
    identical schema to run_dash_and_score's return value.
    """
    global _PROFILE_EVALS_LEFT
    B = len(params_list)
    device = ctx.log_wq.device

    betas, deltas, kappas, k_cores_eff, lams, psis, nus, m_intras = (
        [], [], [], [], [], [], [], [])
    for params in params_list:
        if len(params) >= 8:
            beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
        elif len(params) >= 7:
            beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
            m_intra = 1.0
        else:
            beta, delta, kappa_base, k_core, lam, psi = params[:6]
            nu = 0.0
            m_intra = 1.0
        k_core = max(k_core, max(5.0, lam * 0.4))
        betas.append(beta); deltas.append(delta); kappas.append(kappa_base)
        k_cores_eff.append(k_core); lams.append(lam); psis.append(psi); nus.append(nu)
        m_intras.append(m_intra)

    k_core_eff_np = np.array(k_cores_eff)
    beta_t   = torch.tensor(betas,   device=device, dtype=torch.float32)
    delta_t  = torch.tensor(deltas,  device=device, dtype=torch.float32)
    psi_t    = torch.tensor(psis,    device=device, dtype=torch.float32)
    nu_t     = torch.tensor(nus,     device=device, dtype=torch.float32)
    mintra_t = torch.tensor(m_intras, device=device, dtype=torch.float32)

    do_profile_batch = _PROFILE_EVALS_LEFT > 0
    t_gpu0 = time.perf_counter() if do_profile_batch else None

    extra = _extra_coefs_from_params(params_list, ctx, device)
    omega_batch = compute_omega_batch_gpu(
        ctx, beta_t, delta_t, psi_t, nu_t, mintra_t, k_core_eff_np, extra=extra)
    order_batch = segmented_argsort_batch_gpu(ctx, omega_batch)

    ss_b = ctx.ss.unsqueeze(0).expand(B, -1)
    ts_b = ctx.ts.unsqueeze(0).expand(B, -1)
    W_b  = ctx.W.unsqueeze(0).expand(B, -1)
    so_batch = ss_b.gather(1, order_batch).cpu().numpy()
    to_batch = ts_b.gather(1, order_batch).cpu().numpy()
    Wo_batch = W_b.gather(1, order_batch).cpu().numpy()
    no_batch = omega_batch.gather(1, order_batch).cpu().numpy()

    if do_profile_batch:
        dt_gpu = time.perf_counter() - t_gpu0
        print(f"[FUNGI PROFILE] omega+sort GPU, batch of {B}: "
              f"{1000 * dt_gpu:.1f} ms total, {1000 * dt_gpu / B:.2f} ms/eval")

    results = []
    for i, params in enumerate(params_list):
        beta, delta, kappa_base = betas[i], deltas[i], kappas[i]
        k_core_eff, lam, psi, nu = k_cores_eff[i], lams[i], psis[i], nus[i]
        m_intra = m_intras[i]
        so, to, Wo, no = so_batch[i], to_batch[i], Wo_batch[i], no_batch[i]
        # exp_008 levers (exp_035): store searched lever values (params[8:]) by position
        # so the champion can be rebuilt exactly. Mirrors _fungi_process_row (Ray path).
        _levers = {f'_lever{_k}': float(params[8 + _k]) for _k in range(max(0, len(params) - 8))}

        def _shattered(reason, od_arr, active_n):
            return {
                'beta': beta, 'delta': delta, 'kappa': kappa_base,
                'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                'm_intra': m_intra,
                'utopia_loss': 999., 'is_shattered': 1,
                'shatter_reason': reason, 'n_edges': len(surv_s),
                'active_nodes': active_n, 'alpha': 1., 'Gini': 1.,
                'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1.,
                'S_max': _safe((np.max(od_arr) / n_genes) if len(od_arr) > 0 else 0), **_levers}

        profile_this = _PROFILE_EVALS_LEFT > 0

        try:
            param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
            rng = np.random.default_rng(param_hash)

            t0 = time.perf_counter() if profile_this else None
            surv_s, surv_t, surv_W = select_edges(
                no, Wo, so, to, perturbed_nodes, n_genes, lam,
                per_gene_kappa, kappa_base)
            t1 = time.perf_counter() if profile_this else None
            surv_s, surv_t, surv_W = _motif_repair_swap(
                surv_s, surv_t, surv_W, no, so, to, n_genes,
                max_swap_fraction=0.03, rng=rng)
            t2 = time.perf_counter() if profile_this else None

            od = np.bincount(surv_s, minlength=n_genes)
            idd = np.bincount(surv_t, minlength=n_genes)
            active = int(np.count_nonzero(od + idd > 0))

            sh, reason, gwcc_fraction = check_shatter(
                surv_s, surv_t, surv_W, od, n_genes, active, shatter_cfg)
            t3 = time.perf_counter() if profile_this else None
            if sh:
                results.append(_shattered(reason, od, active))
                if profile_this:
                    print(f"[FUNGI PROFILE] eval (shattered:{reason}): "
                          f"select_edges={1000*(t1-t0):.2f}ms "
                          f"motif_repair={1000*(t2-t1):.2f}ms "
                          f"check_shatter+gwcc={1000*(t3-t2):.2f}ms")
                    _PROFILE_EVALS_LEFT -= 1
                continue

            if mode == 'synthetic':
                if deg_matrix_csr is not None and perturbed_nodes is not None:
                    bad_path = _synthetic_path_gate(
                        surv_s, surv_t, n_genes, perturbed_nodes,
                        deg_matrix_csr, int(spectra_L),
                        min_reach_frac=shatter_cfg.get('min_reach_frac', 0.5))
                    if bad_path:
                        results.append(_shattered('deg_path_exceeds_L', od, active))
                        continue
                loss, topo = calculate_synthetic_loss(
                    surv_s, surv_t, surv_W, n_genes, od, active, kappa_base,
                    utopian_bounds, loss_weights,
                    deg_matrix_csr=deg_matrix_csr,
                    perturbed_nodes=perturbed_nodes,
                    gene_community_labels=gene_community_labels,
                    gene_features=gene_features,
                    compute_expensive=compute_expensive,
                    exact_spectral=exact_spectral)
            else:
                loss, topo = calculate_utopia_loss(
                    surv_s, surv_t, surv_W, n_genes, od, active, kappa_base,
                    utopian_bounds, loss_weights, fast_topology=True,
                    profile=profile_this,
                    gene_community_labels=gene_community_labels)
            t4 = time.perf_counter() if profile_this else None

            # GWCC fraction already computed inside check_shatter -- no need
            # to recompute connected_components a second time here.
            gf = gwcc_fraction if gwcc_fraction is not None else 0.

            gp = 8. * ((0.45 - gf) / 0.45) ** 2 if gf < 0.45 else 0.
            loss = _safe(np.sqrt(max(loss ** 2 + gp, 0.)), 999.)

            if profile_this:
                print(f"[FUNGI PROFILE] eval: select_edges={1000*(t1-t0):.2f}ms "
                      f"motif_repair={1000*(t2-t1):.2f}ms "
                      f"check_shatter+gwcc={1000*(t3-t2):.2f}ms "
                      f"utopia_loss_total={1000*(t4-t3):.2f}ms")
                _PROFILE_EVALS_LEFT -= 1

            results.append({
                'beta': beta, 'delta': delta, 'kappa': kappa_base,
                'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                'm_intra': m_intra,
                'utopia_loss': loss, 'is_shattered': 0, 'shatter_reason': None,
                'n_edges': len(surv_s), 'active_nodes': active,
                'gwcc_fraction': gf,
                'alpha': topo['alpha'], 'Gini': topo['Gini'],
                'rho': topo['rho'], 'C': topo['C'], 'gini_in': topo['gini_in'],
                'Q': topo['Q'],
                'S_max': topo['S_max'], **_levers})
        except Exception as e:
            results.append({
                'beta': beta, 'delta': delta, 'kappa': kappa_base,
                'k_core': k_core_eff, 'lambda': lam, 'psi': psi, 'nu': nu,
                'm_intra': m_intra,
                'utopia_loss': 999., 'is_shattered': 1,
                'shatter_reason': f'crash:{str(e)[:60]}',
                'n_edges': 0, 'active_nodes': 0,
                'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1.,
                'S_max': 0., **_levers})

    return results


def run_dash_and_score(params, W, W_q, D, sources, targets, n_genes,
                       perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, per_gene_kappa, source_pert_impact,
                       md_gate=None, er_scores=None, er_eta=0.3,
                       inter_mask=None, chi_prior=None, w_causal=None,
                       rho_prior=None, chi_t_prior=None, rdf_prior=None,
                       mode='biologic', deg_matrix_csr=None,
                       gene_community_labels=None, compute_expensive=False,
                       kernel_flags=None, gene_features=None,
                       spectra_L=3, exact_spectral=False, intra_mask=None):
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
    if _GPU_CTX is not None and _GPU_CTX.enabled:
        try:
            return run_dash_and_score_gpu_batch(
                [params], _GPU_CTX, n_genes, perturbed_nodes, utopian_bounds,
                loss_weights, shatter_cfg, per_gene_kappa, mode=mode,
                deg_matrix_csr=deg_matrix_csr,
                gene_community_labels=gene_community_labels,
                compute_expensive=compute_expensive, gene_features=gene_features,
                spectra_L=spectra_L, exact_spectral=exact_spectral)[0]
        except Exception as e:
            warnings.warn(f"[FUNGI GPU] GPU path failed ({e}), falling back to CPU")
    try:
        if len(params) >= 8:
            beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
        elif len(params) >= 7:
            beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
            m_intra = 1.0
        else:
            beta, delta, kappa_base, k_core, lam, psi = params[:6]
            nu = 0.0
            m_intra = 1.0
        param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
        rng = np.random.default_rng(param_hash)
        k_core = max(k_core, max(5.0, lam * 0.4))
        T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
        Nm = shatter_cfg.get("max_edge_count", 500000)
        # E6: widen Ne to provide adequate backfill headroom at high λ.
        # Old Ne = Nm + 10_000 left only 10k slack at lambda_max=40 (Nm=200k).
        # New formula: 2× Nm but at least Nm + 50_000 candidates above the budget.
        Ne = min(len(W), max(Nm * 2, Nm + 50_000))
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
        chi   = chi_prior[ss]   if chi_prior   is not None else np.ones(Ne, dtype=np.float64)
        rho   = rho_prior[ss]   if rho_prior   is not None else np.ones(Ne, dtype=np.float64)
        chi_t = chi_t_prior[ts] if chi_t_prior is not None else np.ones(Ne, dtype=np.float64)

        # ── Modular DASH chain ───────────────────────────────────────────────
        # The score is a multiplicative chain; each factor can be switched off
        # from the config by setting its kernel_flag False, in which case that
        # factor contributes 1.0 (multiplicatively neutral). This lets the user
        # ablate any term — e.g. run weight+FFL only, or priors only — without
        # touching code. Exponents (beta, delta, psi, er_eta=eta_bridge, and the
        # zeta baked into chi at precompute) live with their factors.
        #   weight       : W_q^beta          (LightGBM backbone)
        #   ffl          : exp(delta * T)    (feed-forward-loop motif boost)
        #   pert_impact  : pi_s = impact^psi (perturbation impact prior)
        #   rdf          : RDF_s^nu          (regulatory diversity factor — v10.0)
        #   scber        : R^eta_bridge      (source-conditioned bridge ER)
        #   chi_s        : chi_s^zeta_chi    (source pleiotropy prior)
        #   chi_t        : chi_t^zeta_chi    (target pleiotropy prior)
        #   rho          : rho_s             (causal output-efficiency prior)
        kf = kernel_flags if kernel_flags is not None else {}

        def _on(name):
            return bool(kf.get(name, True))   # default: every factor ON

        ones = np.ones(Ne, dtype=np.float64)
        f_weight = (Wqs ** beta)       if _on('weight')      else ones
        f_ffl    = np.exp(delta * Ts)  if _on('ffl')         else ones
        f_pi     = pi_s                if _on('pert_impact') else ones
        f_rdf    = (np.power(rdf_prior[ss], nu)
                    if (_on('rdf') and rdf_prior is not None and nu > 1e-8)
                    else ones)
        f_scber  = er                  if _on('scber')       else ones
        f_chis   = chi                 if _on('chi_s')       else ones
        f_chit   = chi_t               if _on('chi_t')       else ones
        f_rho    = rho                 if _on('rho')         else ones
        # m_intra (session 5, 8th searched HP): multiplicative boost on
        # intra-community edges (SCBER's own partition, reused). m_intra=1.0
        # (the lower bound) is exact identity, backward-compatible.
        f_mintra = (np.where(intra_mask[:Ne], m_intra, 1.0)
                    if (_on('m_intra') and intra_mask is not None) else ones)

        num = (f_weight * f_ffl * f_pi * f_rdf
               * f_scber * f_chis * f_chit * f_rho * f_mintra)
        # ── DASH score ───────────────────────────────────────────────────────

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

        sh, reason, gwcc_fraction = check_shatter(
            surv_s, surv_t, surv_W, od, n_genes, active, shatter_cfg)

        if sh:
            return {
                'beta': beta, 'delta': delta, 'kappa': kappa_base,
                'k_core': k_core, 'lambda': lam, 'psi': psi, 'nu': nu,
                'm_intra': m_intra,
                'utopia_loss': 999., 'is_shattered': 1,
                'shatter_reason': reason, 'n_edges': len(surv_s),
                'active_nodes': active, 'alpha': 1., 'Gini': 1.,
                'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1.,
                'S_max': _safe((np.max(od) / n_genes) if len(od) > 0 else 0)}

        if mode == 'synthetic':
            # Synthetic-only hard gate: an L-layer GNN cannot propagate signal
            # beyond L hops. If the mean shortest path from perturbed sources to
            # their DEGs exceeds spectra_L, SPECTRA structurally cannot model the
            # perturbation->DEG relationship — reject the graph. L is read from
            # config (synthetic_targets.spectra_depth_L), so changing SPECTRA's
            # depth needs only a one-number config edit.
            if deg_matrix_csr is not None and perturbed_nodes is not None:
                bad_path = _synthetic_path_gate(
                    surv_s, surv_t, n_genes, perturbed_nodes,
                    deg_matrix_csr, int(spectra_L),
                    min_reach_frac=shatter_cfg.get('min_reach_frac', 0.5))
                if bad_path:
                    return {
                        'beta': beta, 'delta': delta, 'kappa': kappa_base,
                        'k_core': k_core, 'lambda': lam, 'psi': psi, 'nu': nu,
                        'm_intra': m_intra,
                        'utopia_loss': 999., 'is_shattered': 1,
                        'shatter_reason': 'deg_path_exceeds_L',
                        'n_edges': len(surv_s), 'active_nodes': active,
                        'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'gini_in': 1.,
                        'Q': 1.,
                        'S_max': _safe((np.max(od) / n_genes) if len(od) > 0 else 0)}

            loss, topo = calculate_synthetic_loss(
                surv_s, surv_t, surv_W, n_genes, od, active, kappa_base,
                utopian_bounds, loss_weights,
                deg_matrix_csr=deg_matrix_csr,
                perturbed_nodes=perturbed_nodes,
                gene_community_labels=gene_community_labels,
                gene_features=gene_features,
                compute_expensive=compute_expensive,
                exact_spectral=exact_spectral)
        else:
            loss, topo = calculate_utopia_loss(
                surv_s, surv_t, surv_W, n_genes, od, active, kappa_base,
                utopian_bounds, loss_weights,
                gene_community_labels=gene_community_labels)

        # GWCC fraction already computed inside check_shatter -- no need to
        # recompute connected_components a second time here.
        gf = gwcc_fraction if gwcc_fraction is not None else 0.

        gp = 0.
        if gf < 0.45:
            gp = 8. * ((0.45 - gf) / 0.45) ** 2
        loss = _safe(np.sqrt(max(loss ** 2 + gp, 0.)), 999.)

        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core, 'lambda': lam, 'psi': psi, 'nu': nu,
            'm_intra': m_intra,
            'utopia_loss': loss, 'is_shattered': 0, 'shatter_reason': None,
            'n_edges': len(surv_s), 'active_nodes': active,
            'gwcc_fraction': gf,
            'alpha': topo['alpha'], 'Gini': topo['Gini'],
            'rho': topo['rho'], 'C': topo['C'], 'gini_in': topo['gini_in'],
            'Q': topo['Q'],
            'S_max': topo['S_max']}

    except Exception as e:
        if len(params) >= 8:
            beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
        elif len(params) >= 7:
            beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
            m_intra = 1.0
        else:
            beta, delta, kappa_base, k_core, lam, psi = params[:6]
            nu = 0.0
            m_intra = 1.0
        return {
            'beta': beta, 'delta': delta, 'kappa': kappa_base,
            'k_core': k_core, 'lambda': lam, 'psi': psi, 'nu': nu,
            'm_intra': m_intra,
            'utopia_loss': 999., 'is_shattered': 1,
            'shatter_reason': f'crash:{str(e)[:60]}',
            'n_edges': 0, 'active_nodes': 0,
            'alpha': 1., 'Gini': 1., 'rho': 1., 'C': 0., 'gini_in': 1., 'Q': 1.,
            'S_max': 0.}


# ---------------------------------------------------------------------------
# Motif repair swap (budget 3%)
# ---------------------------------------------------------------------------

def _motif_repair_swap(surv_s, surv_t, surv_W, omega_full, src_full, tgt_full,
                        n_genes, max_swap_fraction=0.03, rng=None):
    """
    Budget-preserving swap: replace up to max_swap_fraction of selected edges
    with unselected edges that close open FFL triangles, if those candidates
    score higher in omega than the weakest selected edges.

    v9.2 changes (E4, E5):
      E4: omega lookup vectorized via sorted binary search — replaces the
          O(N_edges) dict build that previously cost ~13.8B iterations/run.
      E5: W for swapped-in edges is set to 0.0 (placeholder). The old code
          assigned the DASH score as W, mixing units. W is used only in
          graph output (Regulator/Target/Weight parquet), not for loss.
          SPECTRA normalises edge weights before training so a 0.0 default
          is benign; the edge structure is what matters.

    v9.3 change (candidate generation, this session): the sparse matmul
    A2 = adj @ adj is fast (~70-90ms at n_selected~200k), but A2 densifies
    fast (millions of nonzeros even at this scale, since 2-hop reachability
    saturates quickly) -- the old per-nonzero Python loop over A2's COO
    triplets cost 2.9-7.1s/eval at realistic scale, the single largest
    per-evaluation cost in the whole pipeline (bigger than alpha+select_edges
    combined). Candidates can ONLY ever be unselected edges from the full
    omega-scored pool -- anything else defaults to omega=0 in _lookup_omega
    and can never beat a selected edge (always omega>0), so this only ever
    queries A2 for the (much smaller) unselected-pool subset directly instead
    of materializing every one of A2's nonzeros as a candidate first. Verified
    exact-output-equivalent against the old loop across 10 randomized trials
    (_verify_step10_vectorized_motif_repair.py); ~5-8x faster at realistic
    scale (n_selected ~150k-250k).
    """
    n_selected = len(surv_s)
    budget_swaps = max(1, int(n_selected * max_swap_fraction))

    if n_selected < 10 or budget_swaps < 1:
        return surv_s, surv_t, surv_W

    try:
        adj = sp.coo_matrix(
            (np.ones(n_selected, dtype=bool), (surv_s, surv_t)),
            shape=(n_genes, n_genes)).tocsr()
        A2 = adj @ adj  # bool dtype: ~2x faster fancy-indexing below (same nnz
                        # pattern as float -- only A2>0 is ever read, never the
                        # path-count values)

        # E4: vectorized omega lookup — sort src_full keys once, then
        # binary-search for each candidate (O(C log N) vs old O(N) dict)
        src_keys_full = (src_full.astype(np.int64) * n_genes
                         + tgt_full.astype(np.int64))
        sort_order_full = np.argsort(src_keys_full)
        keys_sorted_full = src_keys_full[sort_order_full]
        omega_sorted_full = omega_full[sort_order_full]

        def _lookup_omega(s_arr, t_arr):
            q = s_arr.astype(np.int64) * n_genes + t_arr.astype(np.int64)
            ins = np.searchsorted(keys_sorted_full, q)
            ins = np.clip(ins, 0, len(keys_sorted_full) - 1)
            matched = keys_sorted_full[ins] == q
            scores = np.zeros(len(q), dtype=np.float64)
            scores[matched] = omega_sorted_full[ins[matched]]
            return scores

        selected_keys = np.sort(
            surv_s.astype(np.int64) * n_genes + surv_t.astype(np.int64))
        ins_full = np.clip(np.searchsorted(selected_keys, src_keys_full),
                           0, len(selected_keys) - 1)
        is_selected_full = selected_keys[ins_full] == src_keys_full

        unsel_src = src_full[~is_selected_full]
        unsel_tgt = tgt_full[~is_selected_full]
        unsel_omega = omega_full[~is_selected_full]

        vals = np.asarray(A2[unsel_src, unsel_tgt]).ravel()
        keep = (vals > 0) & (unsel_src != unsel_tgt)
        cand_s = unsel_src[keep].astype(np.int64)
        cand_t = unsel_tgt[keep].astype(np.int64)
        close_scores = unsel_omega[keep]

        if len(cand_s) == 0:
            return surv_s, surv_t, surv_W

        sel_s = surv_s.astype(np.int64)
        sel_t = surv_t.astype(np.int64)
        selected_scores = _lookup_omega(sel_s, sel_t)

        # Sort candidates descending, selected ascending (weakest first)
        cand_order = np.argsort(close_scores)[::-1]
        sel_order  = np.argsort(selected_scores)

        selected_mask = np.ones(n_selected, dtype=bool)
        new_src, new_tgt = [], []
        n_swapped = 0

        for i in range(min(budget_swaps, len(cand_order), len(sel_order))):
            ci_idx  = cand_order[i]
            sel_idx = sel_order[i]
            if close_scores[ci_idx] <= selected_scores[sel_idx]:
                break
            selected_mask[sel_idx] = False
            new_src.append(int(cand_s[ci_idx]))
            new_tgt.append(int(cand_t[ci_idx]))
            n_swapped += 1

        if n_swapped == 0:
            return surv_s, surv_t, surv_W

        keep_idx = np.where(selected_mask)[0]
        final_s = np.concatenate([surv_s[keep_idx],
                                   np.array(new_src, dtype=surv_s.dtype)])
        final_t = np.concatenate([surv_t[keep_idx],
                                   np.array(new_tgt, dtype=surv_t.dtype)])
        # E5: W for swapped edges = 0.0 placeholder (DASH score ≠ LightGBM W)
        final_W = np.concatenate([surv_W[keep_idx],
                                   np.zeros(n_swapped, dtype=surv_W.dtype)])
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
                            chi_prior=None, rho_prior=None, chi_t_prior=None,
                            rdf_prior=None, kernel_flags=None, intra_mask=None,
                            extra_hp_names=(), parent_outdeg=None, md_edge=None,
                            promiscuity_prior=None):
    """
    Reconstruct the final edge set for a given hyperparameter recipe.

    er_scores / er_eta: same semantics as run_dash_and_score.
    intra_mask: same semantics as run_dash_and_score's m_intra factor --
    MUST be passed (along with params' 8th element, m_intra) for the
    reconstructed graph to match the one scored during search/refinement.
    kernel_flags: MUST match the flags used during the search so the
    reconstructed champion graph is identical to the scored one.
    """
    if len(params) >= 8:
        beta, delta, kappa_base, k_core, lam, psi, nu, m_intra = params[:8]
    elif len(params) >= 7:
        beta, delta, kappa_base, k_core, lam, psi, nu = params[:7]
        m_intra = 1.0
    else:
        beta, delta, kappa_base, k_core, lam, psi = params[:6]
        nu = 0.0
        m_intra = 1.0
    param_hash = abs(hash(tuple(float(p) for p in params))) % (2 ** 31)
    rng = np.random.default_rng(param_hash)
    k_core = max(k_core, max(5.0, lam * 0.4))
    T_local = compute_dynamic_topology(W, sources, targets, k_core, n_genes)
    Nm = shatter_cfg.get("max_edge_count", 500000)
    # E6: match widened Ne from run_dash_and_score for consistent reconstruction
    Ne = min(len(W), max(Nm * 2, Nm + 50_000))
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
    rho  = rho_prior[ss]    if rho_prior    is not None else np.ones(Ne, dtype=np.float64)
    chi_t = chi_t_prior[ts] if chi_t_prior  is not None else np.ones(Ne, dtype=np.float64)

    # Modular kernel — must mirror run_dash_and_score exactly.
    kf = kernel_flags if kernel_flags is not None else {}

    def _on(name):
        return bool(kf.get(name, True))

    ones = np.ones(Ne, dtype=np.float64)
    f_weight = (Wqs ** beta)       if _on('weight')      else ones
    f_ffl    = np.exp(delta * Ts)  if _on('ffl')         else ones
    f_pi     = pi_s                if _on('pert_impact') else ones
    f_rdf    = (np.power(rdf_prior[ss], nu)
                if (rdf_prior is not None and nu > 1e-8) else ones)
    f_scber  = er                  if _on('scber')       else ones
    f_chis   = chi                 if _on('chi_s')       else ones
    f_chit   = chi_t               if _on('chi_t')       else ones
    f_rho    = rho                 if _on('rho')         else ones
    f_mintra = (np.where(intra_mask[:Ne], m_intra, 1.0)
                if (_on('m_intra') and intra_mask is not None) else ones)

    # exp_008 searched levers (CPU rebuild mirror of compute_omega_batch_gpu).
    # Read values from params[8:] by extra_hp_names; multiplicative form of the
    # log-space terms. sigma_scber/zeta_s REPLACE the baked f_scber/f_chis.
    _ev = {}
    for _j, _nm in enumerate(extra_hp_names):
        _ev[_nm] = (float(params[8 + _j]) if len(params) > 8 + _j
                    else _EXTRA_HP_DEFAULTS[_nm])
    f_minter = ones
    f_etaout = ones
    f_md = ones
    f_tauindeg = ones
    f_thetapa = ones
    if "sigma_scber" in _ev and er_scores is not None and inter_mask is not None:
        base_er_raw = np.clip(er_scores[:Ne].astype(np.float64), 1e-12, None)
        f_scber = np.where(inter_mask[:Ne], np.power(base_er_raw, _ev["sigma_scber"]), 1.0)
    if "zeta_s" in _ev and chi_prior is not None:
        chi_raw_ne = np.clip(chi.astype(np.float64), 1e-12, None) ** 2  # chi = chi_prior[ss]
        f_chis = np.power(chi_raw_ne, _ev["zeta_s"])
    if "m_inter" in _ev and inter_mask is not None:
        f_minter = np.where(inter_mask[:Ne], _ev["m_inter"], 1.0)
    if "eta_out" in _ev and parent_outdeg is not None:
        f_etaout = np.power(1.0 + parent_outdeg[ss].astype(np.float64), -_ev["eta_out"])
    if "gamma_md" in _ev and md_edge is not None:
        f_md = np.power(np.clip(md_edge[:Ne].astype(np.float64), 1e-12, None), _ev["gamma_md"])
    if "tau_indeg" in _ev and promiscuity_prior is not None:  # exp_035 target anti-promiscuity penalty
        f_tauindeg = np.power(1.0 + promiscuity_prior[ts].astype(np.float64), -_ev["tau_indeg"])
    if "theta_pa" in _ev and parent_outdeg is not None:  # followupA graded preferential-attachment boost
        _maxod = max(float(np.asarray(parent_outdeg, np.float64).max()), 1.0)
        f_thetapa = np.power(np.maximum(parent_outdeg[ss].astype(np.float64), 1.0) / _maxod, _ev["theta_pa"])

    num = (f_weight * f_ffl * f_pi * f_rdf
           * f_scber * f_chis * f_chit * f_rho * f_mintra
           * f_minter * f_etaout * f_md * f_tauindeg * f_thetapa)

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
    t_modslot = _p_smooth("gini_in", metrics.get("gini_in", 1.0), utopian_bounds, loss_weights)
    tr = _p_smooth("rho", metrics.get("rho", 1.0), utopian_bounds, loss_weights)
    tq = _p_smooth("Q", metrics.get("Q", 1.0), utopian_bounds, loss_weights)
    # exp_010 reciprocity target (no-op unless enabled + in bounds)
    trec = (_p_smooth("reciprocity", metrics.get("reciprocity", 0.0),
                      utopian_bounds, loss_weights)
            if ("reciprocity" in loss_weights and "reciprocity" in utopian_bounds) else 0.0)

    raw = (_safe(ta) + _safe(tc) + _safe(t_modslot) + _safe(tg) + _safe(tr) +
           _safe(ts) + _safe(tq) + _safe(trec))

    gf = metrics.get("gwcc_fraction", 1.0)
    gp = 0.
    if gf < 0.45:
        gp = 8. * ((0.45 - gf) / 0.45) ** 2

    return float(np.sqrt(max(raw + gp, 0.)))