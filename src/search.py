"""
FUNGI v6.1 -- Search Space Generation and Ray-Based Parallel Execution

v6.1 changes from v6:
    - Fixed: double @ray.remote decorator removed.
    - Fixed: explicit ray.get() calls removed (Ray auto-resolves ObjectRefs).
    - Added: iterative disk sharding (crash-safe checkpoint after each chunk).
    - Added: configurable chunk_size respected from function argument.
    - src/ path injection uses __file__ so workers can find engine.py.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats.qmc import Sobol
import glob
import re

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
# Ray-Based Parallel Execution with Disk Sharding
# =========================================================================

def _try_import_ray():
    try:
        import ray
        return ray, True
    except ImportError:
        return None, False


def execute_search_ray(param_list, W_arr, D_arr, sources_arr, targets_arr,
                       n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, n_workers=15, chunk_size=500,
                       shard_dir=None):
    """Distributes the DASH kernel search across Ray workers.

    v6.1 features:
    - Zero-copy shared memory via Ray Object Store.
    - Iterative disk sharding: each completed chunk is saved to CSV
      immediately so progress survives network drops or crashes.
    - No double-decorator. No explicit ray.get inside workers.

    Parameters
    ----------
    shard_dir : str or None
        Directory for crash-safe shard files. If None, defaults to
        'data/output/phase3_expansive_search/shards'.
    """
    ray, ray_available = _try_import_ray()

    if not ray_available:
        print("Ray not available. Falling back to joblib execution.")
        return execute_search_joblib(
            param_list, W_arr, D_arr, sources_arr, targets_arr,
            n_genes, perturbed_nodes, utopian_bounds, loss_weights,
            shatter_cfg, n_workers
        )

    # Force OS-level silence on Ray's background telemetry
    os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
    os.environ["RAY_ENABLE_MAC"] = "0"

    if not ray.is_initialized():
        ray.init(
            num_cpus=n_workers, 
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env={
                "env_vars": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1"
                }
            }
        )

    print(f"Ray initialized with {n_workers} workers.")

    # Build baseline COO matrix once
    G_baseline_csr = sp.csr_matrix(
        (W_arr, (sources_arr, targets_arr)),
        shape=(n_genes, n_genes)
    )
    G_baseline_coo = G_baseline_csr.tocoo()

    # Place data in Object Store (zero-copy reads by workers)
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

    # Get the absolute path to src/ so workers can find engine.py
    _src_dir = os.path.dirname(os.path.abspath(__file__))

    @ray.remote
    def _evaluate_chunk(chunk_params, W_r, D_r, src_r, tgt_r,
                        ng, pert_r, bounds_r, weights_r, shatter_r,
                        src_dir):
        """Evaluates a chunk of parameter sets inside a Ray worker."""
        import sys
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from engine import run_dash_and_score

        results = []
        for params in chunk_params:
            # We pass None for the old G_baseline_coo argument since the new engine bypasses it!
            result = run_dash_and_score(
                params, W_r, D_r, src_r, tgt_r, None, ng, pert_r,
                bounds_r, weights_r, shatter_r
            )
            results.append(result)
        return results

    # =========================================================
    # STATE RECOVERY & CHUNK SLICING
    # =========================================================
    
    all_results = []
    completed_chunks = 0
    graphs_completed = 0  
    n_total = len(param_list)
    
    if shard_dir is not None:
        os.makedirs(shard_dir, exist_ok=True)
        existing_shards = glob.glob(os.path.join(shard_dir, "shard_*.csv"))
        
        if existing_shards:
            print(f"Found {len(existing_shards)} existing shards. Recovering state...")
            def extract_num(f):
                match = re.search(r'\d+', os.path.basename(f))
                return int(match.group()) if match else 0
            existing_shards = sorted(existing_shards, key=extract_num)
            
            for shard_file in existing_shards:
                df_shard = pd.read_csv(shard_file)
                all_results.extend(df_shard.to_dict('records')) 
                completed_chunks += 1
                
        graphs_completed = len(all_results)
        if graphs_completed > 0:
            print(f"Recovered {graphs_completed:,} previously evaluated graphs. Resuming search...")

        remaining_params = param_list[graphs_completed:]
        if len(remaining_params) == 0:
            print("All graphs already evaluated! Returning recovered results.")
            return pd.DataFrame(all_results)
    else:
        # If shard_dir is None, skip recovery (TuRBO mode)
        remaining_params = param_list

    # 3. Create chunks from the REMAINING params
    chunks = [
        remaining_params[i:i + chunk_size]
        for i in range(0, len(remaining_params), chunk_size)
    ]

    print(f"Dispatching {len(chunks)} chunks ({chunk_size} graphs each) "
          f"across {n_workers} workers...")

    # =========================================================
    # RAY DISPATCH & ITERATIVE SAVING WITH TQDM
    # =========================================================
    from tqdm.auto import tqdm

    futures = [
        _evaluate_chunk.remote(
            chunk, W_ref, D_ref, sources_ref, targets_ref,
            n_genes, pert_ref, bounds_ref, weights_ref, shatter_ref,
            _src_dir
        )
        for chunk in chunks
    ]

    # Initialize the beautiful progress bar
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
        
        # Instantly save this chunk to the hard drive if applicable
        if shard_dir is not None:
            shard_df = pd.DataFrame(chunk_results)
            shard_file = os.path.join(shard_dir, f"shard_{completed_chunks:05d}.csv")
            shard_df.to_csv(shard_file, index=False)

        all_results.extend(chunk_results)
        completed_chunks += 1
        
        # Smoothly update the progress bar with the number of graphs just finished
        pbar.update(len(chunk_results))

    pbar.close()
    return pd.DataFrame(all_results)


# =========================================================================
# Joblib Fallback
# =========================================================================

def execute_search_joblib(param_list, W_arr, D_arr, sources_arr, targets_arr,
                          n_genes, perturbed_nodes, utopian_bounds,
                          loss_weights, shatter_cfg, n_jobs=15):
    """Fallback parallel executor using joblib."""
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
