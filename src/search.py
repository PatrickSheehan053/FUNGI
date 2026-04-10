"""
FUNGI v6.2 -- Search Space Generation and Ray-Based Parallel Execution

v6.2 fixes:
    - Pre-sorts 5.1M edges descending by weight ONCE before Ray dispatch.
      Workers receive sorted arrays so core slicing is O(1).
    - Default Sobol count reduced to 25,000 (TuRBO converges in ~2k evals,
      100k is computationally wasteful for 6D).
    - Per-graph tqdm via chunk_size default of 50 (small enough for smooth
      progress, large enough to avoid Ray scheduling overhead).
    - Ray telemetry silenced via configure_logging + env vars.
    - Shard recovery with resume from partial runs.
    - Parquet input support added to graph_utils (separate file).
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.qmc import Sobol


# =========================================================================
# Hyperparameter Bounds
# =========================================================================

def get_static_bounds(n_genes, hp_cfg):
    """Returns the static 6D hyperparameter bounds."""
    lam_density = hp_cfg["lambda_density"]
    lam_lower = n_genes * lam_density[0]
    lam_upper = n_genes * lam_density[1]

    names = ["beta", "gamma", "delta", "kappa", "k_core", "lambda"]
    lower = np.array([
        hp_cfg["beta"][0], hp_cfg["gamma"][0], hp_cfg["delta"][0],
        hp_cfg["kappa"][0], hp_cfg["k_core"][0], lam_lower
    ])
    upper = np.array([
        hp_cfg["beta"][1], hp_cfg["gamma"][1], hp_cfg["delta"][1],
        hp_cfg["kappa"][1], hp_cfg["k_core"][1], lam_upper
    ])
    return lower, upper, names


# =========================================================================
# Sobol Sequence Generation
# =========================================================================

def generate_sobol_samples(n_genes, n_samples, hp_cfg, seed=42):
    """Generates a Sobol quasi-random sequence across the 6D parameter space."""
    lower, upper, names = get_static_bounds(n_genes, hp_cfg)

    m = int(np.ceil(np.log2(max(n_samples, 2))))
    n_sobol = 2 ** m
    sampler = Sobol(d=6, scramble=True, seed=seed)
    raw_samples = sampler.random(n=n_sobol)
    scaled = lower + raw_samples * (upper - lower)
    scaled = scaled[:n_samples]

    print(f"Sobol sequence: generated {n_samples:,} points in 6D space.")
    print(f"  Bounds: {dict(zip(names, zip(lower, upper)))}")
    return scaled, lower, upper


# =========================================================================
# Global Pre-Sort (done ONCE before dispatch)
# =========================================================================

def presort_edges(W_arr, D_arr, sources_arr, targets_arr):
    """Sorts all edge arrays descending by weight.

    After this, compute_dynamic_topology can slice [:target_edges]
    instead of running argpartition on 5.1M elements per graph.
    This single sort replaces 100,000 argpartitions.

    Returns sorted copies of W, D, sources, targets.
    """
    order = np.argsort(W_arr)[::-1]  # descending
    return (
        W_arr[order].copy(),
        D_arr[order].copy(),
        sources_arr[order].copy(),
        targets_arr[order].copy(),
    )


# =========================================================================
# Ray Execution
# =========================================================================

def _try_import_ray():
    try:
        import ray
        return ray, True
    except ImportError:
        return None, False


def execute_search_ray(param_list, W_arr, D_arr, sources_arr, targets_arr,
                       n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, n_workers=15, chunk_size=50,
                       shard_dir=None):
    """Distributes the DASH kernel search across Ray workers.

    v6.2: edges are pre-sorted descending by weight before dispatch.
    Workers receive sorted arrays so the core topology slice is O(1).
    Default chunk_size=50 gives near-per-graph tqdm resolution.
    """
    ray, ray_available = _try_import_ray()

    if not ray_available:
        print("Ray not available. Falling back to joblib.")
        return execute_search_joblib(
            param_list, W_arr, D_arr, sources_arr, targets_arr,
            n_genes, perturbed_nodes, utopian_bounds, loss_weights,
            shatter_cfg, n_workers
        )

    # Silence Ray telemetry before init
    os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
    os.environ["RAY_ENABLE_MAC"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "1"

    import logging
    logging.getLogger("ray").setLevel(logging.ERROR)

    if not ray.is_initialized():
        ray.init(
            num_cpus=n_workers,
            ignore_reinit_error=True,
            include_dashboard=False,
            configure_logging=True,
            logging_level=logging.ERROR,
            runtime_env={
                "env_vars": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            }
        )

    print(f"Ray initialized with {n_workers} workers.")

    # Global pre-sort (one-time cost, ~0.5s for 5.1M edges)
    print("Pre-sorting edges by weight (one-time)...")
    W_sorted, D_sorted, src_sorted, tgt_sorted = presort_edges(
        W_arr, D_arr, sources_arr, targets_arr
    )

    # Place sorted data in Object Store
    W_ref = ray.put(W_sorted)
    D_ref = ray.put(D_sorted)
    src_ref = ray.put(src_sorted)
    tgt_ref = ray.put(tgt_sorted)
    pert_ref = ray.put(perturbed_nodes)
    bounds_ref = ray.put(utopian_bounds)
    weights_ref = ray.put(loss_weights)
    shatter_ref = ray.put(shatter_cfg)

    _src_dir = os.path.dirname(os.path.abspath(__file__))

    @ray.remote
    def _evaluate_chunk(chunk_params, W_r, D_r, src_r, tgt_r,
                        ng, pert_r, bounds_r, weights_r, shatter_r,
                        src_dir):
        import sys
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from engine import run_dash_and_score

        results = []
        for params in chunk_params:
            result = run_dash_and_score(
                params, W_r, D_r, src_r, tgt_r, None, ng, pert_r,
                bounds_r, weights_r, shatter_r
            )
            results.append(result)
        return results

    # ---- Shard recovery ----
    all_results = []
    completed_chunks = 0
    graphs_completed = 0
    n_total = len(param_list)

    if shard_dir is not None:
        os.makedirs(shard_dir, exist_ok=True)
        existing_shards = sorted(
            glob.glob(os.path.join(shard_dir, "shard_*.csv")),
            key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
            if re.search(r'\d+', os.path.basename(f)) else 0
        )
        if existing_shards:
            print(f"Found {len(existing_shards)} existing shards. Recovering...")
            for sf in existing_shards:
                df_s = pd.read_csv(sf)
                all_results.extend(df_s.to_dict('records'))
                completed_chunks += 1
            graphs_completed = len(all_results)
            print(f"Recovered {graphs_completed:,} graphs. Resuming...")

        remaining_params = param_list[graphs_completed:]
        if len(remaining_params) == 0:
            print("All graphs already evaluated.")
            return pd.DataFrame(all_results)
    else:
        remaining_params = param_list

    chunks = [
        remaining_params[i:i + chunk_size]
        for i in range(0, len(remaining_params), chunk_size)
    ]

    print(f"Dispatching {len(chunks)} chunks ({chunk_size} graphs each) "
          f"across {n_workers} workers...")

    from tqdm.auto import tqdm

    futures = [
        _evaluate_chunk.remote(
            chunk, W_ref, D_ref, src_ref, tgt_ref,
            n_genes, pert_ref, bounds_ref, weights_ref, shatter_ref,
            _src_dir
        )
        for chunk in chunks
    ]

    pbar = tqdm(
        total=n_total,
        initial=graphs_completed,
        desc="Expansive Search",
        unit="graph",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        chunk_results = ray.get(done[0])

        if shard_dir is not None:
            shard_df = pd.DataFrame(chunk_results)
            shard_file = os.path.join(shard_dir, f"shard_{completed_chunks:05d}.csv")
            shard_df.to_csv(shard_file, index=False)

        all_results.extend(chunk_results)
        completed_chunks += 1
        pbar.update(len(chunk_results))

    pbar.close()
    return pd.DataFrame(all_results)


# =========================================================================
# Joblib Fallback
# =========================================================================

def execute_search_joblib(param_list, W_arr, D_arr, sources_arr, targets_arr,
                          n_genes, perturbed_nodes, utopian_bounds,
                          loss_weights, shatter_cfg, n_jobs=15):
    from joblib import Parallel, delayed
    from engine import run_dash_and_score

    print("Pre-sorting edges for joblib fallback...")
    W_s, D_s, src_s, tgt_s = presort_edges(W_arr, D_arr, sources_arr, targets_arr)

    print(f"Executing search with joblib ({n_jobs} workers)...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_dash_and_score)(
            params, W_s, D_s, src_s, tgt_s, None, n_genes, perturbed_nodes,
            utopian_bounds, loss_weights, shatter_cfg
        )
        for params in param_list
    )
    return pd.DataFrame(results)
