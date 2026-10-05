#!/bin/bash
# ============================================================================
#  HYPHAE — GRNBoost2/GENIE3-style dense GRN builder  (UNC Longleaf edition)
#  Submits a 10-task SLURM array; each task runs LightGBM regression on a
#  slice of the 5,000 genes. Consolidate with consolidate_graphs.py after.
#
#  *** 3-WAY AGGREGATION STUDY: run this ONCE PER INPUT FILE (3 runs total). ***
#  Pick ONE dataset in the "SELECT DATASET" block below, submit, wait, consolidate,
#  then repeat for the next. Each run writes to its own chunks_<name>/ dir and its
#  own output graph, so the three runs never collide.
#
#  BEFORE SUBMITTING (do these once):
#    1. Put the whole HYPHAE/ folder somewhere on Longleaf /work storage, e.g.
#         /work/users/<a>/<b>/<onyen>/HYPHAE      (a,b = first two letters of onyen)
#    2. Create the conda env (one time):
#         module load anaconda/2024.02
#         conda env create -f environment.yml        # creates env "hyphae"
#    3. cd into this folder, pick a dataset below, and submit:
#         sbatch submit_coexpression.sh                                  # hybrid / full-metacell arms
#         sbatch --time=4-00:00:00 submit_coexpression.sh                # SINGLE-CELL arm (just more TIME;
#                                                                     # memory is fine at 96G — worker is
#                                                                     # memory-efficient, no bigmem needed)
#  Longleaf notes: the "general" partition is the default CPU partition (no -p
#  needed); jobs are single-node by default; max walltime is 11 days.
# ============================================================================

#SBATCH --job-name=RPE1_HYPHAE
#SBATCH --array=0-9                  # 10 array tasks == --total_tasks below
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # == --n_jobs below (1 thread per worker)
#SBATCH --mem=96G                    # Fits ALL three arms at 32 workers (the worker is memory-efficient:
                                     # measured 32-worker peak ~40 GB metacell / ~69 GB single-cell, vs
                                     # ~305 GB in the old worker). No bigmem node needed. Drop to 64G with
                                     # --construct_slots 2 if you like.
#SBATCH --time=2-00:00:00            # 48 h; hybrid/full-metacell expected ~17-19 h.
                                     # SINGLE-CELL (144k cells) is MUCH slower -> override to
                                     # 4-00:00:00 at submit:  sbatch --time=4-00:00:00 submit_coexpression.sh
#SBATCH --output=logs/HYPHAE_%A_task_%a.log
#SBATCH --error=logs/HYPHAE_%A_task_%a.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=CHANGE_ME@email.unc.edu   # <-- put your onyen here (or delete these 2 lines)

set -euo pipefail

# ── SELECT DATASET (uncomment EXACTLY ONE; run the script once per dataset) ──
# ARM D — mbk_k2 FULL metacell (perturbed AND controls aggregated):
RUN_NAME="mbk_k2_fullmetacell";   INPUT_BASENAME="RPE1_5k_mbk_k2_fullmetacell_train.h5ad"
# ARM B — mbk_k2 HYBRID (controls kept single-cell):
# RUN_NAME="hybrid_ctrlpreserved"; INPUT_BASENAME="RPE1_5k_hybrid_ctrlpreserved_train.h5ad"
# ARM C — SINGLE CELLS (no aggregation) -> submit with:  sbatch --time=4-00:00:00 submit_coexpression.sh
# RUN_NAME="singlecell";           INPUT_BASENAME="RPE1_5k_singlecell_train.h5ad"

# ── 1. PATHS (all relative to this folder; nothing hard-coded) ──────────────
# $SLURM_SUBMIT_DIR is the directory you ran `sbatch` from == the HYPHAE folder.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
INPUT_H5AD="${PROJECT_DIR}/input/${INPUT_BASENAME}"
CHUNK_DIR="${PROJECT_DIR}/chunks_${RUN_NAME}"     # per-dataset chunk dir (the 3 runs never collide)
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$CHUNK_DIR" "$LOG_DIR"

# ── 2. ENVIRONMENT ──────────────────────────────────────────────────────────
module load anaconda/2024.02
conda activate hyphae

# ── 3. RUN THIS ARRAY TASK'S GENE SLICE ─────────────────────────────────────
echo "Dataset : ${RUN_NAME}  (${INPUT_BASENAME})"
echo "Task    : ${SLURM_ARRAY_TASK_ID} / 10   ->  ${CHUNK_DIR}"
python coexpression_worker.py \
    --input_file   "$INPUT_H5AD" \
    --output_dir   "$CHUNK_DIR" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --total_tasks  10 \
    --n_jobs       32 \
    --n_bootstraps 20

# ── 4. AFTER ALL 10 TASKS OF THIS DATASET FINISH, consolidate (run once): ────
echo ""
echo "When all 10 tasks of '${RUN_NAME}' finish, consolidate with:"
echo "  module load anaconda/2024.02 && conda activate hyphae"
echo "  python consolidate_graphs.py \\"
echo "      --chunk_dir   ${CHUNK_DIR} \\"
echo "      --output_file HYPHAE_RPE1_${RUN_NAME}_dense_graph.parquet \\"
echo "      --total_tasks 10 \\"
echo "      --h5ad_file   ${INPUT_H5AD}"
