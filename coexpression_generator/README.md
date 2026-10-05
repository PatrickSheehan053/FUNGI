# Co-expression generator (thesis name: HYPHAE)

One LightGBM regression per target gene on all other genes. The averaged gain importances over bootstraps
become directed `Regulator -> Target` edges. Runs as a 10-task SLURM array, then `consolidate_graphs.py`
stitches the chunks into one dense graph.

## Files

| file | what it is |
|---|---|
| `coexpression_worker.py` | **v3 (7 Jul 2026).** One process per gene (`Pool(maxtasksperchild=1)`), shared read-only memmap, `--construct_slots` cap on the feature copy. Produced the RPE1 hybrid and full-metacell graphs (20 bootstraps) on UNC Longleaf. |
| `coexpression_worker_v4_rebuilt_UNTESTED.py` | **v4 rebuilt from notes.** Re-applies the six changes documented in the 9 July handoff on top of v3. Never run on a cluster. |
| `submit_coexpression.sh` | SLURM array script (Longleaf edition, calls v3). |
| `consolidate_graphs.py` | Merges the per-task chunks and runs sanity checks. |
| `environment.yml` | Python 3.11 conda env. |
| `docs/FIX_NOTES_v3_07JUL.md` | Why v3 was written (the joblib memory creep). |

## The memory problem, in order

1. **v1 (joblib):** long-lived workers never returned memory between genes, so RAM climbed until out-of-memory.
2. **v2 (joblib + file semaphore):** still ran out of memory at 96 GB.
3. **v3 (this folder):** fixes the gene-to-gene creep. Measured on HPC1 on 9 July, it still has three problems:
   - When a worker is killed for running out of memory, `Pool.map` hangs forever with no error (a 7.7 h silent stall).
   - The per-bootstrap LightGBM Dataset is not freed until the gene finishes.
   - The real fixed cost is about **7.6 GB per worker**, not about 1 GB, so 32 workers will not fit on a 96–128 GB node.
4. **v4 (HPC1 only):** adds `gc.collect()` per bootstrap, `ProcessPoolExecutor` with an auto-restart on worker death,
   `max_tasks_per_child=1` (Python 3.11+), per-gene `peakRSS` logging, `--lgbm_threads`, `--tmp_dir` and
   `--max_restarts`. Launched as **8 workers x 4 threads, 10 bootstraps, 128 GB**, it produced the h1-hESC graph.

The original v4 is at `/scratch/patrick.sheehan/SPORE+/HPC_v2_launch/hyphae/` on HPC1.

## TODO when HPC access returns

- [ ] Copy the real v4 worker and submit script off HPC1. Diff against `coexpression_worker_v4_rebuilt_UNTESTED.py`.
- [ ] Replace the rebuild with the real file, and update `submit_coexpression.sh` to the v4 launch config.
- [ ] Run one task and confirm `peakRSS` stays flat (about 7.6 GB per worker).

## Notes

- Output columns are `Regulator, Target, Importance`. FUNGI and the baselines expect `source, target, weight`
  or `Regulator, Target, Weight`, so rename before feeding downstream.
- Bootstrap count differs between existing graphs: RPE1 graphs used 20 and h1-hESC used 10.
