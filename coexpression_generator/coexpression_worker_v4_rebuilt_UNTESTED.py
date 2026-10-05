"""
Co-expression generator worker (HYPHAE) — v4 REBUILT FROM NOTES, NOT THE HPC ORIGINAL.

The real v4 (9 Jul 2026) lives only on HPC1 at
/scratch/patrick.sheehan/SPORE+/HPC_v2_launch/hyphae/2_run_HYPHAE_worker.py.
This file re-applies the six changes documented in "Handoff - 09JULY2026.md" §1.5 on top of the
local v3 worker (coexpression_worker.py). Diff it against the HPC original before trusting it.

Science is unchanged from v3: same LightGBM params, same per-bootstrap 80/20 split + early stopping,
same gain importances averaged over bootstraps, same Target/Regulator/Importance output.
"""
import os, sys, argparse, time, tempfile, gc, resource
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Manager
from datetime import datetime

import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import lightgbm as lgb

LGBM = {
    'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
    'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 20,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'max_depth': 8, 'n_estimators': 1000, 'num_threads': 1, 'verbose': -1,
}

# max_tasks_per_child only exists on Python >= 3.11; it is the per-gene reaper
HAS_REAPER = sys.version_info >= (3, 11)

_matrix = _gene_names = _gene_index = _n_boot = _shard_dir = _construct_lock = _lgbm_threads = None


def _peak_rss_gb():
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _init(mmap_path, shape, gene_names, n_boot, shard_dir, construct_lock, lgbm_threads):
    global _matrix, _gene_names, _gene_index, _n_boot, _shard_dir, _construct_lock, _lgbm_threads
    # each (possibly spawned) process re-opens the shared matrix by path
    _matrix = np.memmap(mmap_path, dtype='float32', mode='r', shape=shape)
    _gene_names = gene_names
    _gene_index = {name: i for i, name in enumerate(gene_names)}   # change 6: dict, not list.index()
    _n_boot, _shard_dir, _lgbm_threads = n_boot, shard_dir, lgbm_threads
    _construct_lock = construct_lock if construct_lock is not None else nullcontext()


def _train_gene(target_gene):
    out_path = os.path.join(_shard_dir, f"{target_gene}.parquet")
    if os.path.exists(out_path):
        return target_gene, "skipped"
    start = time.time()
    try:
        target_idx = _gene_index[target_gene]
        feature_idx = [i for i in range(len(_gene_names)) if i != target_idx]
        feature_names = [_gene_names[i] for i in feature_idx]
        y = np.ascontiguousarray(_matrix[:, target_idx])
        n_rows = len(y)

        # only construct_slots workers hold the big float32 copy at once (no-op lock when disabled)
        with _construct_lock:
            features = np.ascontiguousarray(_matrix[:, feature_idx])
            full_ds = lgb.Dataset(features, label=y, feature_name=feature_names,
                                  params={'num_threads': _lgbm_threads}, free_raw_data=True)
            full_ds.construct()
            del features

        total_gain = np.zeros(len(feature_names))
        for k in range(_n_boot):
            seed = 42 + k
            rng = np.random.default_rng(seed)
            n_val = max(1, int(n_rows * 0.20))
            val_idx = rng.choice(n_rows, size=n_val, replace=False)
            train_mask = np.ones(n_rows, dtype=bool); train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            params = dict(LGBM, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                          num_threads=_lgbm_threads)
            train_ds = full_ds.subset(train_idx); val_ds = full_ds.subset(val_idx)
            model = lgb.train(params, train_ds, num_boost_round=LGBM['n_estimators'], valid_sets=[val_ds],
                              callbacks=[lgb.log_evaluation(-1), lgb.early_stopping(50, verbose=False)])
            total_gain += model.feature_importance(importance_type='gain')
            model.free_dataset(); del model, train_ds, val_ds
            gc.collect()   # change 1: break the Booster <-> Dataset cycle so the C-side Dataset is actually freed
        del full_ds; gc.collect()

        avg_gain = total_gain / _n_boot
        records = [{'Target': target_gene, 'Regulator': r, 'Importance': s}
                   for r, s in zip(feature_names, avg_gain) if s > 0]
        pd.DataFrame(records or {'Target': [], 'Regulator': [], 'Importance': []}).to_parquet(out_path)
        # change 4: per-gene peak RSS so memory is observed, not assumed
        print(f"[{datetime.now():%H:%M:%S}] {target_gene} ({time.time()-start:.0f}s, {len(records)} edges) "
              f"peakRSS {_peak_rss_gb():.2f} GB", flush=True)
        return target_gene, "ok"
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [FAIL] {target_gene}: {e}", flush=True)
        return target_gene, "fail"


def _remaining(genes, shard_dir):
    return [g for g in genes if not os.path.exists(os.path.join(shard_dir, f"{g}.parquet"))]


def _run_pool(todo, args, mmap_path, shape, gene_names, shard_dir, construct_lock):
    pool_kwargs = dict(max_workers=args.n_jobs, initializer=_init,
                       initargs=(mmap_path, shape, gene_names, args.n_bootstraps, shard_dir,
                                 construct_lock, args.lgbm_threads))
    if HAS_REAPER:
        pool_kwargs['max_tasks_per_child'] = 1   # change 3: fresh process per gene (forces spawn)
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = [pool.submit(_train_gene, g) for g in todo]
        for fut in as_completed(futures):
            fut.result()   # raises BrokenProcessPool if a worker was SIGKILLed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--total_tasks", type=int, required=True)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--lgbm_threads", type=int, default=4, help="LightGBM threads per worker (n_jobs x this = cores)")
    ap.add_argument("--n_bootstraps", type=int, default=10)
    ap.add_argument("--construct_slots", type=int, default=8,
                    help="max workers holding the per-gene feature copy at once; >= n_jobs disables the semaphore")
    ap.add_argument("--max_restarts", type=int, default=5, help="pool rebuilds after a worker is killed")
    ap.add_argument("--tmp_dir", default=os.environ.get("TMPDIR", tempfile.gettempdir()),
                    help="where the shared memmap is written (defaults to $TMPDIR)")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input_file)
    matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    matrix = np.ascontiguousarray(matrix.astype(np.float32))
    gene_names = list(adata.var_names)
    del adata; gc.collect()   # change 6: parent drops the AnnData
    shape = matrix.shape
    n_genes = len(gene_names)

    lock_enabled = args.construct_slots < args.n_jobs   # change 5
    print(f"data {shape} ({matrix.nbytes/1e9:.2f} GB)", flush=True)
    print(f"{args.n_jobs} workers x {args.lgbm_threads} threads | bootstraps={args.n_bootstraps} | "
          f"construct_slots={args.construct_slots if lock_enabled else 'disabled'} | "
          f"per-gene reaper={'on' if HAS_REAPER else 'off'} | py{sys.version_info.major}.{sys.version_info.minor}",
          flush=True)

    per_task, remainder = divmod(n_genes, args.total_tasks)
    if args.task_id < remainder:
        start = args.task_id * (per_task + 1); end = start + per_task + 1
    else:
        start = args.task_id * per_task + remainder; end = start + per_task
    my_genes = gene_names[start:end]

    shard_dir = os.path.join(args.output_dir, "temp_shards")
    os.makedirs(shard_dir, exist_ok=True)
    todo = _remaining(my_genes, shard_dir)
    print(f"task {args.task_id}: genes {start}-{end-1} ({len(my_genes)}); {len(todo)} to do", flush=True)

    if todo:
        os.makedirs(args.tmp_dir, exist_ok=True)
        mm_file = tempfile.NamedTemporaryFile(delete=False, dir=args.tmp_dir); mmap_path = mm_file.name; mm_file.close()
        shared = np.memmap(mmap_path, dtype='float32', mode='w+', shape=shape)
        shared[:] = matrix[:]; shared.flush()
        del matrix, shared; gc.collect()
        try:
            with (Manager() if lock_enabled else nullcontext()) as manager:
                construct_lock = manager.Semaphore(args.construct_slots) if lock_enabled else None
                # change 2: a killed worker raises BrokenProcessPool instead of hanging Pool.map forever;
                # re-derive the to-do list from shards on disk and rebuild the pool
                restarts = 0
                while todo:
                    try:
                        _run_pool(todo, args, mmap_path, shape, gene_names, shard_dir, construct_lock)
                        break
                    except BrokenProcessPool:
                        restarts += 1
                        todo = _remaining(my_genes, shard_dir)
                        print(f"[{datetime.now():%H:%M:%S}] WORKER KILLED (likely OOM) — restart "
                              f"{restarts}/{args.max_restarts}; {len(todo)} genes remain", flush=True)
                        if restarts > args.max_restarts:
                            print("max_restarts exceeded — giving up; resubmit to resume", flush=True)
                            break
        finally:
            try: os.unlink(mmap_path)
            except OSError: pass

    present = [g for g in my_genes if os.path.exists(os.path.join(shard_dir, f"{g}.parquet"))]
    missing = len(my_genes) - len(present)
    out_file = os.path.join(args.output_dir, f"chunk_{args.task_id}.parquet")
    if present:
        df = pd.concat([pd.read_parquet(os.path.join(shard_dir, f"{g}.parquet")) for g in present],
                       ignore_index=True)
        df.to_parquet(out_file)
        print(f"task {args.task_id} -> {out_file} ({len(df):,} edges)", flush=True)
    print(f"task {args.task_id}: {len(present)} genes; {missing} missing", flush=True)


if __name__ == "__main__":
    main()
