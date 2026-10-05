"""
HYPHAE worker — per-gene LightGBM regression -> directed edge importances.

"""
import os, argparse, time, tempfile, gc
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import lightgbm as lgb
from multiprocessing import Pool, Manager
from datetime import datetime

LGBM = {
    'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
    'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 20,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'max_depth': 8, 'n_estimators': 1000, 'num_threads': 1, 'verbose': -1,
}

_X = _NAMES = _NB = _TMP = _SEM = None

def _init(mmap_path, shape, names, n_boot, temp_dir, sem):
    global _X, _NAMES, _NB, _TMP, _SEM
    _X = np.memmap(mmap_path, dtype='float32', mode='r', shape=shape)
    _NAMES, _NB, _TMP, _SEM = names, n_boot, temp_dir, sem

def _train_gene(tgt):
    out = os.path.join(_TMP, f"{tgt}.parquet")
    if os.path.exists(out):
        return
    t0 = time.time()
    try:
        tidx = _NAMES.index(tgt)
        fidx = [i for i in range(len(_NAMES)) if i != tidx]
        fnames = [_NAMES[i] for i in fidx]
        y = np.ascontiguousarray(_X[:, tidx])
        n = len(y)
        # Only construct_slots workers hold the big float32 copy at once (the RAM cap).
        with _SEM:
            Xf = np.ascontiguousarray(_X[:, fidx])
            ds = lgb.Dataset(Xf, label=y, feature_name=fnames,
                             params={'num_threads': 1}, free_raw_data=True)
            ds.construct()
            del Xf
        tot = np.zeros(len(fnames))
        for k in range(_NB):
            seed = 42 + k
            rng = np.random.default_rng(seed)
            nv = max(1, int(n * 0.20))
            vi = rng.choice(n, size=nv, replace=False)
            m = np.ones(n, dtype=bool); m[vi] = False
            ti = np.where(m)[0]
            p = dict(LGBM, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            dtr = ds.subset(ti); dv = ds.subset(vi)
            mdl = lgb.train(p, dtr, num_boost_round=LGBM['n_estimators'], valid_sets=[dv],
                            callbacks=[lgb.log_evaluation(-1),
                                       lgb.early_stopping(50, verbose=False)])
            tot += mdl.feature_importance(importance_type='gain')
            mdl.free_dataset(); del mdl, dtr, dv   # free this bootstrap before the next
        avg = tot / _NB
        rec = [{'Target': tgt, 'Regulator': r, 'Importance': s}
               for r, s in zip(fnames, avg) if s > 0]
        pd.DataFrame(rec or {'Target': [], 'Regulator': [], 'Importance': []}).to_parquet(out)
        print(f"[{datetime.now():%H:%M:%S}] {tgt} ({time.time()-t0:.0f}s, {len(rec)} edges)", flush=True)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [FAIL] {tgt}: {e}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--total_tasks", type=int, required=True)
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--n_bootstraps", type=int, default=20)
    ap.add_argument("--construct_slots", type=int, default=8,
                    help="max workers holding the per-gene feature copy at once (RAM cap)")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input_file)
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    X = np.ascontiguousarray(X.astype(np.float32))
    names = list(adata.var_names)
    shape = X.shape
    ntot = len(names)
    print(f"data {shape} ({X.nbytes/1e9:.2f} GB) | {args.n_jobs} workers x 1 gene/proc | "
          f"construct_slots={args.construct_slots}", flush=True)

    per, rem = divmod(ntot, args.total_tasks)
    if args.task_id < rem:
        s = args.task_id * (per + 1); e = s + per + 1
    else:
        s = args.task_id * per + rem; e = s + per
    my_genes = names[s:e]

    temp_dir = os.path.join(args.output_dir, "temp_shards")
    os.makedirs(temp_dir, exist_ok=True)
    todo = [g for g in my_genes if not os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))]
    print(f"task {args.task_id}: genes {s}-{e-1} ({len(my_genes)}); {len(todo)} to do", flush=True)

    if todo:
        mmf = tempfile.NamedTemporaryFile(delete=False); mmap_path = mmf.name; mmf.close()
        Xmm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=shape)
        Xmm[:] = X[:]; Xmm.flush()
        del X, Xmm; gc.collect()   # parent drops the big array; workers read the memmap
        try:
            with Manager() as mgr:
                sem = mgr.Semaphore(args.construct_slots)
                with Pool(args.n_jobs, initializer=_init,
                          initargs=(mmap_path, shape, names, args.n_bootstraps, temp_dir, sem),
                          maxtasksperchild=1) as pool:
                    pool.map(_train_gene, todo, chunksize=1)
        finally:
            try: os.unlink(mmap_path)
            except OSError: pass

    out_file = os.path.join(args.output_dir, f"chunk_{args.task_id}.parquet")
    files = [os.path.join(temp_dir, f"{g}.parquet") for g in my_genes
             if os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))]
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(out_file)
        print(f"task {args.task_id} -> {out_file} ({len(df):,} edges)", flush=True)
    else:
        print(f"task {args.task_id}: no data generated", flush=True)

if __name__ == "__main__":
    main()
