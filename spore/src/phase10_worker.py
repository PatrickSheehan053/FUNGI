"""
SPORE+ · src/phase10_worker.py
────────────────────────────────
Thin CLI entry point for Phase 10 subprocess execution.

All logic lives in phase10_confounders.py::run_worker_logic().
This script is a 40-line shim so subprocess.run() has a file to call.

Launched by run_phase10_subprocess() with:
  MALLOC_MMAP_THRESHOLD_=134217728  (all arrays >128 MB use mmap → return on free)
  MALLOC_ARENA_MAX=2                (2 arenas, not 8×nCPU)
  MALLOC_TRIM_THRESHOLD_=131072
  OMP/BLAS/MKL threads = 1

Expected peak RSS in this process: ~49 GB.
On exit, OS unconditionally reclaims all memory.
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="SPORE+ Phase 10 worker")
    parser.add_argument("--input",             required=True)
    parser.add_argument("--output",            required=True)
    parser.add_argument("--embed",             required=True)
    parser.add_argument("--obs",               required=True)
    parser.add_argument("--batch-key",         default="gem_group")
    parser.add_argument("--n-comps",           type=int, default=50)
    parser.add_argument("--n-sample",          type=int, default=100_000)
    parser.add_argument("--chunk-rows",        type=int, default=50_000)
    parser.add_argument("--max-harmony-iter",  type=int, default=20)
    parser.add_argument("--cc-skip-threshold", type=int, default=500_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("phase10_worker")

    import os, psutil
    logger.info("Phase 10 worker starting (fresh heap)")
    logger.info(f"  PID: {os.getpid()}")
    logger.info(f"  MALLOC_MMAP_THRESHOLD_: {os.environ.get('MALLOC_MMAP_THRESHOLD_', 'not set')}")
    logger.info(f"  MALLOC_ARENA_MAX: {os.environ.get('MALLOC_ARENA_MAX', 'not set')}")
    logger.info(f"  RAM at start: {psutil.Process().memory_info().rss / 1e9:.1f} GB")

    # All logic is in phase10_confounders — no duplication
    from phase10_confounders import run_worker_logic

    run_worker_logic(
        input_path           = args.input,
        output_path          = args.output,
        embed_path           = args.embed,
        obs_path             = args.obs,
        batch_key            = args.batch_key,
        n_comps              = args.n_comps,
        n_sample             = args.n_sample,
        chunk_rows           = args.chunk_rows,
        max_harmony_iter     = args.max_harmony_iter,
        cc_skip_threshold    = args.cc_skip_threshold,
        logger               = logger,
    )


if __name__ == "__main__":
    main()
