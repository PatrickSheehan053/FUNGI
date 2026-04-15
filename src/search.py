"""
FUNGI v7 -- Search Space Generation and Ray-Based Parallel Execution

v7: Waterfall immunization REMOVED. presort_and_immunize replaced with
simple presort_edges. The engine no longer expects imm_idx.
"""
import os, glob, re, logging
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats.qmc import Sobol

def get_static_bounds(n_genes, hp_cfg):
    ld = hp_cfg["lambda_density"]
    names = ["beta","gamma","delta","kappa","k_core","lambda"]
    lower = np.array([hp_cfg["beta"][0], hp_cfg["gamma"][0], hp_cfg["delta"][0],
                      hp_cfg["kappa"][0], hp_cfg["k_core"][0], n_genes*ld[0]])
    upper = np.array([hp_cfg["beta"][1], hp_cfg["gamma"][1], hp_cfg["delta"][1],
                      hp_cfg["kappa"][1], hp_cfg["k_core"][1], n_genes*ld[1]])
    return lower, upper, names

def generate_sobol_samples(n_genes, n_samples, hp_cfg, seed=42):
    lower, upper, names = get_static_bounds(n_genes, hp_cfg)
    m = int(np.ceil(np.log2(max(n_samples, 2)))); ns = 2**m
    sampler = Sobol(d=6, scramble=True, seed=seed)
    scaled = lower + sampler.random(n=ns) * (upper - lower)
    scaled = scaled[:n_samples]
    print(f"Sobol sequence: generated {n_samples:,} points in 6D space.")
    print(f"  Bounds: {dict(zip(names, zip(lower, upper)))}")
    return scaled, lower, upper

def presort_edges(W, D, src, tgt):
    """Sort all edge arrays descending by weight. One-time O(N log N)."""
    order = np.argsort(W)[::-1]
    return W[order].copy(), D[order].copy(), src[order].copy(), tgt[order].copy()

def _try_ray():
    try:
        import ray; return ray, True
    except ImportError: return None, False

def execute_search_ray(param_list, W_arr, D_arr, sources_arr, targets_arr,
                       n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                       shatter_cfg, n_workers=15, chunk_size=50, shard_dir=None):
    ray, ok = _try_ray()
    if not ok:
        return execute_search_joblib(param_list, W_arr, D_arr, sources_arr,
                                     targets_arr, n_genes, perturbed_nodes,
                                     utopian_bounds, loss_weights, shatter_cfg, n_workers)
    os.environ["RAY_DISABLE_METRICS_COLLECTION"]="1"
    os.environ["RAY_DEDUP_LOGS"]="1"
    logging.getLogger("ray").setLevel(logging.ERROR)
    if not ray.is_initialized():
        ray.init(num_cpus=n_workers, ignore_reinit_error=True, include_dashboard=False,
                 configure_logging=True, logging_level=logging.ERROR,
                 runtime_env={"env_vars":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
                                          "MKL_NUM_THREADS":"1"}})
    print(f"Ray initialized with {n_workers} workers.")
    print("Pre-sorting edges by weight (one-time)...")
    Ws, Ds, srcs, tgts = presort_edges(W_arr, D_arr, sources_arr, targets_arr)

    W_ref = ray.put(Ws); D_ref = ray.put(Ds)
    src_ref = ray.put(srcs); tgt_ref = ray.put(tgts)
    pert_ref = ray.put(perturbed_nodes)
    bounds_ref = ray.put(utopian_bounds); weights_ref = ray.put(loss_weights)
    shatter_ref = ray.put(shatter_cfg)
    _src_dir = os.path.dirname(os.path.abspath(__file__))

    @ray.remote
    def _eval_chunk(chunk, Wr, Dr, sr, tr, ng, pr, br, wr, shr, sd):
        import sys
        if sd not in sys.path: sys.path.insert(0, sd)
        from engine import run_dash_and_score
        return [run_dash_and_score(p, Wr, Dr, sr, tr, None, ng, pr, br, wr, shr) for p in chunk]

    all_results=[]; completed=0; graphs_done=0; n_total=len(param_list)
    if shard_dir is not None:
        os.makedirs(shard_dir, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(shard_dir,"shard_*.csv")),
                          key=lambda f: int(re.search(r'\d+',os.path.basename(f)).group()) if re.search(r'\d+',os.path.basename(f)) else 0)
        if existing:
            print(f"Found {len(existing)} shards. Recovering...")
            for sf in existing:
                all_results.extend(pd.read_csv(sf).to_dict('records')); completed+=1
            graphs_done=len(all_results)
            print(f"Recovered {graphs_done:,} graphs.")
        remaining = param_list[graphs_done:]
        if len(remaining)==0:
            print("All graphs evaluated."); return pd.DataFrame(all_results)
    else: remaining = param_list

    chunks = [remaining[i:i+chunk_size] for i in range(0, len(remaining), chunk_size)]
    print(f"Dispatching {len(chunks)} chunks ({chunk_size} graphs each) across {n_workers} workers...")
    from tqdm.auto import tqdm
    futures = [_eval_chunk.remote(c, W_ref, D_ref, src_ref, tgt_ref,
                                  n_genes, pert_ref, bounds_ref, weights_ref,
                                  shatter_ref, _src_dir) for c in chunks]
    pbar = tqdm(total=n_total, initial=graphs_done, desc="Expansive Search", unit="graph",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        cr = ray.get(done[0])
        if shard_dir is not None:
            pd.DataFrame(cr).to_csv(os.path.join(shard_dir,f"shard_{completed:05d}.csv"), index=False)
        all_results.extend(cr); completed+=1; pbar.update(len(cr))
    pbar.close()
    return pd.DataFrame(all_results)

def execute_search_joblib(param_list, W_arr, D_arr, sources_arr, targets_arr,
                          n_genes, perturbed_nodes, utopian_bounds, loss_weights,
                          shatter_cfg, n_jobs=15):
    from joblib import Parallel, delayed
    from engine import run_dash_and_score
    Ws,Ds,srcs,tgts = presort_edges(W_arr, D_arr, sources_arr, targets_arr)
    print(f"Executing with joblib ({n_jobs} workers)...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_dash_and_score)(p, Ws, Ds, srcs, tgts, None, n_genes,
                                    perturbed_nodes, utopian_bounds, loss_weights, shatter_cfg)
        for p in param_list)
    return pd.DataFrame(results)
