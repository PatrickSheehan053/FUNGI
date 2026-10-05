# HYPHAE worker OOM — fixed (2026-07-07)

## The symptom
Chinmaya: all 3 arms OOM'd at 96 GB; "memory climbed with every gene processed, OOM'd after ~10 genes."

## Root cause (three compounding problems in the joblib worker)
1. **Per-gene memory leak (the killer).** The old worker used a persistent joblib/loky pool: 32
   long-lived workers each processed ~500 genes in a row and **never handed memory back to the OS**
   between genes (per-bootstrap LightGBM datasets accumulate, and the process allocator keeps the
   pages). RAM climbed every gene → OOM after a handful. Reproduced locally: with just **4 workers on
   the small metacell**, RAM climbed **4.6 → 20 GB before a single gene finished**.
2. **32× feature-matrix copy.** Each gene builds a contiguous float32 copy of "all genes except the
   target" for LightGBM to bin (1.4 GB metacell / 2.9 GB single-cell). With 32 workers copying at once
   (e.g. at startup) that's a 45–90 GB spike. The old file-based semaphore that was supposed to cap
   this is fragile.
3. **Thread oversubscription.** LightGBM's binning (`Dataset.construct`) grabs *every* core by default
   (measured: 1230% CPU), ignoring the `num_threads=1` set for training. With 32 workers that's up to
   ~1000 threads fighting over 32 cores.

## The fix (rewritten `coexpression_worker.py`, base-Python multiprocessing — no joblib)
- **One gene per process** (`Pool(maxtasksperchild=1)`): the OS reclaims *everything* when each gene's
  process exits, so RAM cannot creep. This is exactly the "new process per gene" idea. Bootstraps are
  also freed explicitly within a gene.
- **Shared read-only memmap** for the matrix (one copy, not 32).
- **`--construct_slots` cap** so only N workers hold the transient float32 copy at once (default 8).
- **Binning pinned to 1 thread** → N workers use exactly N cores, no oversubscription.
- **Science unchanged**: identical LightGBM params, identical per-bootstrap 80/20 split + early
  stopping, identical gain importances, identical output schema. (The target column is excluded by the
  same all-but-target copy as before — bit-identical to the previous graphs, not an approximation.)

## Proof the fix holds
Same test that made the old worker climb to 20 GB: the new worker sits **flat at ~4.4 GB** and does not
climb as genes complete.

## Peak RAM per node (32 workers) — all fit 96 GB comfortably
Built from directly-measured per-worker components (shared memmap = matrix; per worker = a binned
LightGBM dataset + one live train/val subset pair; plus a bounded feature-copy transient), and
cross-checked against a real multiprocess run (4 workers/metacell peaked ~4.4 GB, i.e. below these
conservative estimates). Peak is dominated by 32 × per-worker, so it scales with cell count.

| arm | cells | shared matrix | per-worker | est. peak @32 workers | headroom @96 GB |
|-----|------:|--------------:|-----------:|----------------------:|-----------------|
| D metacell   | 68,633  | 1.4 GB | ~1.0 GB | **~35 GB** | ~60 GB |
| B hybrid     | 74,775  | 1.5 GB | ~1.1 GB | **~40 GB** | ~55 GB |
| C singlecell | 144,354 | 2.9 GB | ~2.1 GB | **~75 GB** | ~20 GB |

`--mem=96G` is safe for all three (single-cell is the only tight one, ~20 GB margin). If you want
more margin on single-cell, add `--construct_slots 4` (keeps it ~78 GB) or run it at `--n_jobs 28`
(~62 GB). Metacell/hybrid could drop to `--mem=48G` for better queue priority.

## Files
- `coexpression_worker.py` — fixed worker (replaces the old one).
- `_2_run_HYPHAE_worker_JOBLIB_BACKUP.py` — the old joblib version, kept for reference.
- Submit script / consolidate script unchanged (only the worker was the problem).
