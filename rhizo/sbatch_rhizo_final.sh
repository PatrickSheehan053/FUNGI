#!/bin/bash
#SBATCH --job-name=rhizo_final
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
# ============================ SET ME (cluster-specific — ASK PATRICK, do NOT change the footprint on your own) =
#SBATCH --gres=gpu:2                  # SET ME: 2-3 V100s (gpu:2 default; gpu:3 if granted). Packs/shards over ALL visible GPUs.
#SBATCH --time=14:00:00               # SET ME: overnight wall (HPO ~40 trials packed + gauntlet + zero-shot dry-run)
# SET ME: --partition=<gpu_partition>   --account=<acct>   --constraint=v100
# ==============================================================================================================
# obj_009.3 RHIZO-final — the exhaustive final RHIZO run. Resumable + requeue-safe: the HPO skips finished trials,
# the gauntlet skips finished cells, so a wall-kill / requeue loses nothing. `trap summarize EXIT` banks partial
# results even on preempt/timeout.
#   sbatch sbatch_rhizo_final.sh                    # run the full pipeline
#   sbatch --gres=gpu:3 sbatch_rhizo_final.sh       # 3 V100s
#   bash sbatch_rhizo_final.sh --dry_run            # print the HPO + gauntlet plans + VRAM (no GPU, no SLURM)
# NOTE for the operator: paste commands as SINGLE LINES. Agent-authored files do not exist on the HPC until scp'd.
# ==============================================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
PY="${PY:-python}"

if [[ "${1:-}" == "--dry_run" ]]; then $PY src/hpo_v3f.py --dry_run; $PY src/gauntlet_v3f.py --dry_run; exit 0; fi

# ---- ENV (SET ME): load a CUDA-matched torch. V100 = sm_70 -> cu118 OR cu121, NEVER cu128 (that is sm_120). ----
# module load cuda/12.1 2>/dev/null || true
# source /path/to/venv/bin/activate       # a venv with: pip install -r requirements.txt ; torch from the cu121 index

echo "=== node $(hostname) | $(date) | job ${SLURM_JOB_ID:-local} ==="
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPUs', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# ---- HARD GATE: ψ(0)=0 must pass for the assembled model + every profile before any training --------------------
echo "=== psi(0)=0 invariant gate ==="
$PY src/test_invariant_v3f.py || { echo "FATAL: psi(0)=0 invariant FAILED — aborting before training"; exit 2; }

# ---- caches (idempotent; the O(N^3) pinv is the one-time CPU spike; skip if already bundled) --------------------
echo "=== build caches (idempotent) ==="
$PY src/build_caches_v3f.py

NGPUS="$($PY -c 'import torch;print(torch.cuda.device_count())')"
GPULIST="$($PY -c 'import torch;print(",".join(str(i) for i in range(torch.cuda.device_count())))')"
echo "=== $NGPUS GPU(s): [$GPULIST] ==="

summarize() { echo "=== summarize (banked on exit) ==="; $PY src/hpo_v3f.py --summarize || true; $PY src/gauntlet_v3f.py --summarize || true; $PY src/verify_v3f.py || true; }
trap summarize EXIT

# ---- Stage 1: self-HPO (the VRAM-packer stacks ALL visible GPUs internally) ------------------------------------
echo "=== Stage 1: self-HPO sweep (packed across GPUs) ==="
$PY src/hpo_v3f.py --gpus "$GPULIST"
$PY src/hpo_v3f.py --summarize

# ---- Stage 2: gauntlet at the HPO winner (shard one worker per GPU) --------------------------------------------
echo "=== Stage 2: gauntlet at the HPO winner (sharded) ==="
pids=()
for i in $(seq 0 $((NGPUS-1))); do CUDA_VISIBLE_DEVICES=$i $PY src/gauntlet_v3f.py --shard "$i" --n-shards "$NGPUS" --device cuda > "results/gauntlet_worker_${i}.log" 2>&1 & pids+=($!); echo "  gauntlet shard $i/$NGPUS -> results/gauntlet_worker_${i}.log (pid ${pids[-1]})"; done
for p in "${pids[@]}"; do wait "$p"; done
$PY src/gauntlet_v3f.py --summarize

# ---- Stage 3: zero-shot dry-run (plumbing/leakage; fires for real when a HYPHAE parquet lands) ------------------
echo "=== Stage 3: zero-shot gene-held-out dry-run ==="
$PY src/zeroshot_v3f.py --dry_run --device cuda || true

# ---- Stage 4: verification battery (psi0 + nulls-collapse + V7 per feature) ------------------------------------
echo "=== Stage 4: verification ==="
$PY src/verify_v3f.py

echo "=== all stages done | $(date) ==="
# summarize also runs via the EXIT trap
