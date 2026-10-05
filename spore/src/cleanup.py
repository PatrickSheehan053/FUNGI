"""
SPORE+ · src/cleanup.py
─────────────────────────
Post-run cleanup: removes intermediate checkpoint files while keeping
the final metacell outputs and CHITIN delta matrices.

Controlled by `output.cleanup_intermediates: true` in the YAML config.

Files removed (when enabled):
  splits/*_p9.h5ad        — Phase 9 normalized checkpoints (large, no longer needed)
  splits/*_train_p10.h5ad — Phase 10 full output (only embeddings used downstream)
  output/*_p10_embed.npz  — Phase 10 embeddings (only needed for Phase 11/12)
  output/*_p10_obs.parquet— Phase 10 obs metadata
  output/checkpoints/     — Phase 12 shard files

Files KEPT:
  output/*_metacell.h5ad         — final metacell expression matrices for PSGRN/FUNGI
  output/chitin/*_chitin.h5ad    — CHITIN-corrected delta matrices for PSGRN/FUNGI
  splits/*_split_indices.json    — perturbation split assignments (reproducibility)
  output/reports/*.md            — pipeline run report
  figures/                       — all diagnostic plots
"""

import os
import shutil
import glob
from pathlib import Path


def run_cleanup(cfg: dict, logger, dry_run: bool = False) -> dict:
    """
    Remove intermediate files based on config.cleanup_intermediates setting.

    Parameters
    ----------
    cfg      : SPORE+ config dict
    logger   : pipeline logger
    dry_run  : if True, log what would be deleted without deleting anything

    Returns
    -------
    dict with keys 'deleted', 'skipped', 'total_mb_freed'
    """
    enabled = cfg.get("output", {}).get("cleanup_intermediates", False)

    if not enabled:
        logger.info("  Cleanup: DISABLED (set output.cleanup_intermediates: true to enable)")
        return {"deleted": [], "skipped": [], "total_mb_freed": 0.0}

    splits_dir = cfg["paths"]["_splits"]
    output_dir = cfg["paths"]["_processed"]
    dataset    = cfg.get("dataset", {}).get("name", "dataset")

    patterns_to_delete = [
        # Phase 9 checkpoints (normalized, not scaled)
        str(splits_dir / f"{dataset}_train_p9.h5ad"),
        str(splits_dir / f"{dataset}_val_p9.h5ad"),
        str(splits_dir / f"{dataset}_test_p9.h5ad"),
        # Phase 10 intermediates
        str(splits_dir / f"{dataset}_train_p10.h5ad"),
        str(output_dir  / f"{dataset}_train_p10_embed.npz"),
        str(output_dir  / f"{dataset}_train_p10_obs.parquet"),
        # Cell line labels intermediate
        str(output_dir  / f"{dataset}_cell_line_labels.parquet"),
        # Phase 12 shard checkpoint directory
        str(output_dir  / "checkpoints"),
    ]

    deleted   = []
    skipped   = []
    bytes_freed = 0

    for path_str in patterns_to_delete:
        path = Path(path_str)
        if not path.exists():
            skipped.append(path_str)
            continue

        if path.is_dir():
            # Shard checkpoints directory
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            if dry_run:
                logger.info(f"  [DRY RUN] Would delete dir: {path} ({size/1e6:.0f} MB)")
            else:
                shutil.rmtree(path)
                logger.info(f"  🗑  Deleted dir: {path} ({size/1e6:.0f} MB)")
            bytes_freed += size
            deleted.append(path_str)
        else:
            size = path.stat().st_size
            if dry_run:
                logger.info(f"  [DRY RUN] Would delete: {path.name} ({size/1e6:.0f} MB)")
            else:
                path.unlink()
                logger.info(f"  🗑  Deleted: {path.name} ({size/1e6:.0f} MB)")
            bytes_freed += size
            deleted.append(path_str)

    mb_freed = bytes_freed / 1e6
    if dry_run:
        logger.info(f"  [DRY RUN] Would free ~{mb_freed:.0f} MB")
    else:
        logger.info(f"  Cleanup complete: freed ~{mb_freed:.0f} MB across {len(deleted)} items")

    return {
        "deleted":        deleted,
        "skipped":        skipped,
        "total_mb_freed": mb_freed,
    }
