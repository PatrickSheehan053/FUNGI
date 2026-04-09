"""
FUNGI v6 -- Search Space Generation and Ray-Based Parallel Execution

Replaces the v5 LHS + joblib system with:
    - Sobol quasi-random sequences for uniform 6D space-filling (Part A)
    - Ray Core for zero-copy shared-memory parallel execution (Part B)

The module provides both the Sobol generator and the Ray-based batch
executor.  A fallback joblib executor is included for environments
where Ray is unavailable.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.qmc import Sobol


# =========================================================================
# Hyperparameter Bounds
# =========================================================================

def get_static_bounds(n_genes, hp_cfg):
    """Returns the static 6D hyperparameter bounds.

    Lambda is the only parameter that scales with n_genes.
    k_core bounds are corrected to [1.0, 15.0] to prevent the core
    from exceeding the global graph edge count.

    Parameters
    ----------
    n_genes : int
        Number of genes in the graph.
    hp_cfg : dict
        The 'hyperparameter_bounds' section of the YAML config.

    Returns
    -------
    lower : np.ndarray, shape (6,)
    upper : np.ndarray, shape (6,)
    names : list of str
    """
    lam_density = hp_cfg["lambda_density"]
    lam_lower = n_genes * lam_density[0]
    lam_upper = n_genes * lam_density[1]

    k_core_bounds = hp_cfg["k_core"]
    beta_bounds = hp_cfg["beta"]
    gamma_bounds = hp_cfg["gamma"]
    delta_bounds = hp_cfg["delta"]
    kappa_bounds = hp_cfg["kappa"]

    # Order: [beta, gamma, delta, kappa, k_core, lambda]
    names = ["beta", "gamma", "delta", "kappa", "k_core", "lambda"]

    lower = np.array([
        beta_bounds[0], gamma_bounds[0], delta_bounds[0],
        kappa_bounds[0], k_core_bounds[0], lam_lower
    ])
    upper = np.array([
        beta_bounds[1], gamma_bounds[1], delta_bounds[1],
        kappa_bounds[1], k_core_bounds[1], lam_upper
    ])

    return lower, upper, names


# =========================================================================
# Sobol Sequence Generation (Part A)
# =========================================================================

def generate_sobol_samples(n_genes, n_samples, hp_cfg, seed=42):
    """Generates a Sobol quasi-random sequence across the 6D parameter space.

    Sobol sequences maintain superior spatial uniformity compared to LHS
    at high sample counts (50k-100k), avoiding the spatial clumping and
    voiding that degrades LHS in high dimensions.

    Parameters
    ----------
    n_genes : int
        Number of genes (used for lambda scaling).
    n_samples : int
        Number of Sobol points to generate.  Should be a power of 2 for
        optimal uniformity; the function rounds up if needed.
    hp_cfg : dict
        The 'hyperparameter_bounds' section of the YAML config.
    seed : int
        Random seed for the Sobol engine.

    Returns
    -------
    samples : np.ndarray, shape (n_samples, 6)
        Scaled hyperparameter coordinates.
    lower : np.ndarray, shape (6,)
    upper : np.ndarray, shape (6,)
    """
    lower, upper, names = get_static_bounds(n_genes, hp_cfg)

    # Sobol requires sample count to be a power of 2 for optimal balance.
    # We generate the next power of 2 and trim.
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    n_sobol = 2 ** m

    sampler = Sobol(d=6, scramble=True, seed=seed)
    raw_samples = sampler.random(n=n_sobol)

    # Scale from [0, 1]^6 to parameter bounds
    scaled = lower + raw_samples * (upper - lower)

    # Trim to requested count
    scaled = scaled[:n_samples]

    print(f"Sobol sequence: generated {n_samples:,} points in 6D space.")
    print(f"  Bounds: {dict(zip(names, zip(lower, upper)))}")

    return scaled, lower, upper


# =========================================================================
# Ray-Based Parallel Execution (Part B)
# =========================================================================

def _try_import_ray():
    """Attempts to import Ray.  Returns (ray_module, available_bool)."""
    try:
        import ray
        return ray, True
    except ImportError:
        return None, False


def execute_search_ray(param_list, W_arr, D_arr, sources_arr, targets_arr,
                       n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, n_workers=15, chunk_size=5000):
    """Distributes the DASH kernel search across Ray workers with zero-copy
    shared memory for the graph data.

    Parameters
    ----------
    param_list : np.ndarray, shape (N, 6)
        Hyperparameter coordinates to evaluate.
    W_arr, D_arr, sources_arr, targets_arr : np.ndarray
        Normalized edge data from Phase 2.
    n_genes : int
    perturbed_nodes : np.ndarray
    utopian_bounds, loss_weights : dict
        From Phase 0 diagnostics.
    shatter_cfg : dict
        The 'shatter' section of the YAML config.
    n_workers : int
        Number of Ray remote workers.
    chunk_size : int
        Number of graphs per Ray task.

    Returns
    -------
    pd.DataFrame
        Results for all evaluated coordinates.
    """
    ray, ray_available = _try_import_ray()

    if not ray_available:
        print("Ray not available. Falling back to joblib execution.")
        return execute_search_joblib(
            param_list, W_arr, D_arr, sources_arr, targets_arr,
            n_genes, perturbed_nodes, utopian_bounds, loss_weights,
            shatter_cfg, n_workers
        )

    if not ray.is_initialized():
        ray.init(num_cpus=n_workers, ignore_reinit_error=True)

    print(f"Ray initialized with {n_workers} workers.")

    # Build baseline COO matrix once
    G_baseline_csr = sp.csr_matrix(
        (W_arr, (sources_arr, targets_arr)),
        shape=(n_genes, n_genes)
    )
    G_baseline_coo = G_baseline_csr.tocoo()

    # Place large arrays in the Ray Object Store (zero-copy)
    W_ref = ray.put(W_arr)
    D_ref = ray.put(D_arr)
    sources_ref = ray.put(sources_arr)
    targets_ref = ray.put(targets_arr)
    coo_data_ref = ray.put(G_baseline_coo.data)
    coo_row_ref = ray.put(G_baseline_coo.row)
    coo_col_ref = ray.put(G_baseline_coo.col)
    pert_ref = ray.put(perturbed_nodes)
    bounds_ref = ray.put(utopian_bounds)
    weights_ref = ray.put(loss_weights)
    shatter_ref = ray.put(shatter_cfg)

    @ray.remote
    def _evaluate_chunk(chunk_params, W_r, D_r, src_r, tgt_r,
                        coo_data_r, coo_row_r, coo_col_r,
                        ng, pert_r, bounds_r, weights_r, shatter_r):
        """Evaluates a chunk of parameter sets inside a Ray worker."""
        import sys
        import os

        # Identify the absolute path of the folder containing this search.py script (the src/ folder)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Inject the src/ folder into the Ray worker's system path
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Now the worker can safely find and import engine.py
        from engine import run_dash_and_score
        import scipy.sparse as _sp

        
        _coo = _sp.coo_matrix((coo_data_r, (coo_row_r, coo_col_r)), shape=(ng, ng))

        results = []
        for params in chunk_params:
            result = run_dash_and_score(
                params, W_r, D_r, src_r, tgt_r, _coo, ng, pert_r,
                bounds_r, weights_r, shatter_r
            )
            results.append(result)
        return results

    # Split into chunks and dispatch
    n_total = len(param_list)
    chunks = [
        param_list[i:i + chunk_size]
        for i in range(0, n_total, chunk_size)
    ]

    print(f"Dispatching {len(chunks)} chunks ({chunk_size} graphs each) "
          f"across {n_workers} workers...")

    futures = [
        _evaluate_chunk.remote(
            chunk, W_ref, D_ref, sources_ref, targets_ref,
            coo_data_ref, coo_row_ref, coo_col_ref,
            n_genes, pert_ref, bounds_ref, weights_ref, shatter_ref
        )
        for chunk in chunks
    ]

    import os
    import pandas as pd
    
    # Create a directory to hold the safety shards
    shard_dir = "data/output/phase3_expansive_search/shards"
    os.makedirs(shard_dir, exist_ok=True)

    # Collect results with progress tracking AND Sharding
    all_results = []
    completed = 0
    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        chunk_results = ray.get(done[0])
        
        # --- NEW SHARDING LOGIC ---
        # Instantly save this chunk to the hard drive
        shard_df = pd.DataFrame(chunk_results)
        shard_file = os.path.join(shard_dir, f"chunk_{completed}_results.csv")
        shard_df.to_csv(shard_file, index=False)
        # --------------------------

        all_results.extend(chunk_results)
        completed += 1
        
        if completed % 5 == 0 or completed == len(chunks):
            print(f"  Progress: {completed}/{len(chunks)} chunks "
                  f"({len(all_results):,}/{n_total:,} graphs)")

    return pd.DataFrame(all_results)


# =========================================================================
# Joblib Fallback Executor
# =========================================================================

def execute_search_joblib(param_list, W_arr, D_arr, sources_arr, targets_arr,
                          n_genes, perturbed_nodes, utopian_bounds,
                          loss_weights, shatter_cfg, n_jobs=15):
    """Fallback parallel executor using joblib when Ray is not available."""
    from joblib import Parallel, delayed
    from engine import run_dash_and_score

    G_baseline_csr = sp.csr_matrix(
        (W_arr, (sources_arr, targets_arr)),
        shape=(n_genes, n_genes)
    )
    G_baseline_coo = G_baseline_csr.tocoo()

    print(f"Executing search with joblib ({n_jobs} workers)...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_dash_and_score)(
            params, W_arr, D_arr, sources_arr, targets_arr,
            G_baseline_coo, n_genes, perturbed_nodes,
            utopian_bounds, loss_weights, shatter_cfg
        )
        for params in param_list
    )

    return pd.DataFrame(results)
