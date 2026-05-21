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


def compute_lambda_search_bounds(lam_eff, n_genes, static_lo=2.0, static_hi=35.0):
    lo = max(static_lo, lam_eff * 0.4)
    hi = min(static_hi, lam_eff * 2.5)
    if hi - lo < 6.0:
        mid = (lo + hi) / 2.0
        lo = max(static_lo, mid - 3.0)
        hi = min(static_hi, mid + 3.0)
    return lo / n_genes, hi / n_genes


def get_static_bounds(n_genes, hp_cfg, lam_eff=None):
    if hp_cfg.get("lambda_density") is not None:
        ld = hp_cfg["lambda_density"]
    elif lam_eff is not None:
        ld = list(compute_lambda_search_bounds(lam_eff, n_genes))
    else:
        ld = [2.0 / n_genes, 35.0 / n_genes]

    names = ["beta", "delta", "kappa", "k_core", "lambda", "psi"]
    lower = np.array([
        hp_cfg["beta"][0],
        hp_cfg["delta"][0],
        hp_cfg["kappa"][0],
        hp_cfg["k_core"][0],
        n_genes * ld[0],
        hp_cfg["psi"][0],
    ])
    upper = np.array([
        hp_cfg["beta"][1],
        hp_cfg["delta"][1],
        hp_cfg["kappa"][1],
        hp_cfg["k_core"][1],
        n_genes * ld[1],
        hp_cfg["psi"][1],
    ])
    return lower, upper, names


def generate_sobol_samples(n_genes, n_samples, hp_cfg, seed=42, lam_eff=None):
    lower, upper, names = get_static_bounds(n_genes, hp_cfg, lam_eff=lam_eff)
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    ns = 2 ** m
    sampler = Sobol(d=6, scramble=True, seed=seed)
    scaled = lower + sampler.random(n=ns) * (upper - lower)
    scaled = scaled[:n_samples]
    print(f"  Sobol sequence: {n_samples:,} points in 6D space.")
    print(f"  Parameter space: β[{lower[0]:.1f},{upper[0]:.1f}]  "
          f"δ[{lower[1]:.2f},{upper[1]:.2f}]  "
          f"κ[{lower[2]:.3f},{upper[2]:.3f}]  "
          f"k_core[{lower[3]:.0f},{upper[3]:.0f}]  "
          f"λ[{lower[4]:.1f},{upper[4]:.1f}]  "
          f"ψ[{lower[5]:.1f},{upper[5]:.1f}]")
    return scaled, lower, upper


def presort_edges(W, W_q, D, src, tgt):
    """Sort all edge arrays descending by weight. One-time O(N log N).
    W_q (source-quantile weights) is sorted in the same order as W."""
    order = np.argsort(W)[::-1]
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
                 n_workers=15, src_dir=None,
                 md_gate=None,
                 er_scores=None, er_eta=0.3, inter_mask=None,
                 chi_prior=None):

        self.n_genes = n_genes
        self.n_workers = n_workers
        self.src_dir = src_dir or os.path.dirname(os.path.abspath(__file__))
        self.er_eta = float(er_eta)

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

        # chi_prior: per-gene array, NOT edge-level — no presort needed.
        # chi_prior[s] is looked up by source node index during DASH scoring.
        self.chi_prior = (chi_prior.astype(np.float64)
                          if chi_prior is not None else None)

        # md_gate: presort in the same order
        if md_gate is not None:
            order = np.argsort(W_arr)[::-1]
            self.md_gate = md_gate[order].astype(np.float64)
        else:
            self.md_gate = None

        self._perturbed_nodes = perturbed_nodes
        self._utopian_bounds = utopian_bounds
        self._loss_weights = loss_weights
        self._shatter_cfg = shatter_cfg
        self._per_gene_kappa = per_gene_kappa
        self._source_pert_impact = source_pert_impact

        self._ray = None
        self._ray_refs = None

    def _init_ray(self):
        ray, ok = _try_ray()
        if not ok:
            return False
        self._ray = ray
        os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "1"
        logging.getLogger("ray").setLevel(logging.ERROR)

        if not ray.is_initialized():
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
            "chi":        ray.put(self.chi_prior),
        }
        return True

    def evaluate(self, param_list, chunk_size=50, shard_dir=None,
                 desc="Evaluating", show_progress=True):
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
                        pr, br, wr, shr, kr, ir, mdr, err, er_eta_r, imr, chir, sd):
            import sys
            if sd not in sys.path:
                sys.path.insert(0, sd)
            from engine import run_dash_and_score
            return [run_dash_and_score(
                p, Wr, Wqr, Dr, sr, tr, ng, pr, br, wr, shr, kr, ir,
                md_gate=mdr, er_scores=err, er_eta=er_eta_r,
                inter_mask=imr, chi_prior=chir)
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
                refs["inter_mask"], refs["chi"], src_dir)
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

    def evaluate_single(self, params):
        from engine import run_dash_and_score
        return run_dash_and_score(
            params, self.Ws, self.Wqs, self.Ds, self.srcs, self.tgts,
            self.n_genes, self._perturbed_nodes,
            self._utopian_bounds, self._loss_weights, self._shatter_cfg,
            self._per_gene_kappa, self._source_pert_impact,
            md_gate=self.md_gate,
            er_scores=self.er_scores, er_eta=self.er_eta,
            inter_mask=self.inter_mask, chi_prior=self.chi_prior)

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
                inter_mask=self.inter_mask, chi_prior=self.chi_prior)
            for p in param_list)
        return pd.DataFrame(results)
