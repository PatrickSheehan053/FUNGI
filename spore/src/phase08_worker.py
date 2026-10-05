"""
SPORE+ · src/phase08_worker.py
────────────────────────────────
Thin CLI entry point for Phase 8 subprocess execution.
All logic lives in phase08_hvg.py::run_worker_logic().
Launched by run_phase8_subprocess() with allocator-safe env vars.

07JUNE patch: --protected-gene-list argument
"""
import argparse, logging, os, sys, psutil


def main():
    parser = argparse.ArgumentParser(description="SPORE+ Phase 8 HVG worker")
    parser.add_argument("--train",               required=True)
    parser.add_argument("--val",                 required=True)
    parser.add_argument("--test",                required=True)
    parser.add_argument("--splits-dir",          required=True)
    parser.add_argument("--dataset",             required=True)
    parser.add_argument("--pert-col",            default="gene")
    parser.add_argument("--ctrl-label",          default="non-targeting")
    parser.add_argument("--n-top-genes",         type=int, default=5000)
    parser.add_argument("--method",              default="seurat_v3")
    parser.add_argument("--is-combinatorial",    action="store_true")
    parser.add_argument("--sep",                 default="+")
    parser.add_argument("--protected-gene-list", default=None,
                        help="Path to protected genes JSON (optional)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("phase08_worker")
    logger.info("Phase 8 worker starting (fresh heap)")
    logger.info(f"  PID             : {os.getpid()}")
    logger.info(f"  MALLOC_ARENA_MAX: {os.environ.get('MALLOC_ARENA_MAX', 'not set')}")
    logger.info(f"  RAM at start    : {psutil.Process().memory_info().rss / 1e9:.1f} GB")

    from phase08_hvg import run_worker_logic
    run_worker_logic(
        train_path          = args.train,
        val_path            = args.val,
        test_path           = args.test,
        splits_dir          = args.splits_dir,
        dataset             = args.dataset,
        pert_col            = args.pert_col,
        ctrl_label          = args.ctrl_label,
        n_top_genes         = args.n_top_genes,
        method              = args.method,
        is_combinatorial    = args.is_combinatorial,
        sep                 = args.sep,
        protected_gene_list = args.protected_gene_list,
        logger              = logger,
    )


if __name__ == "__main__":
    main()