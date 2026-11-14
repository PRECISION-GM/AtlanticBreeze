#!/bin/bash
#SBATCH --job-name=atlantic-forward
#SBATCH --partition=gpua100largex4,gpua100largex4spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=atlantic-%j.log
# Submit from a frozen detached worktree (one per commit) so edits cannot race a running job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from an AtlanticBreeze run snapshot}"
git diff --quiet HEAD -- scripts gradient Project.toml Manifest.toml || { echo 'Refusing to run from a modified tree'; exit 2; }
export ATLANTIC_COMMIT="$(git rev-parse HEAD)"
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia:"
export JULIA_CUDA_USE_COMPAT=false JULIA_NUM_THREADS=8 JULIA_NUM_PRECOMPILE_TASKS=2
unset LD_LIBRARY_PATH
: "${ATLANTIC_INPUT:?Set ATLANTIC_INPUT to the prepared interval directory}"
: "${ATLANTIC_OUTPUT:?Set ATLANTIC_OUTPUT to a new directory outside the repository}"
# Spot preemption requeues the job: each attempt gets its own directory so partial output is kept
# and a restart never collides with it (the cause of the 15027 failure).
export ATLANTIC_OUTPUT="$ATLANTIC_OUTPUT/attempt-${SLURM_RESTART_COUNT:-0}"
echo "commit $ATLANTIC_COMMIT input $ATLANTIC_INPUT output $ATLANTIC_OUTPUT"
sha256sum scripts/*.jl Project.toml Manifest.toml gradient/*.toml "$ATLANTIC_INPUT"/*.csv
mode="${1:-forward}"
case "$mode" in forward|train) project=. ;; check_gradients|radiation_reproducer) project=gradient ;; *) echo "Expected forward, train, check_gradients or radiation_reproducer"; exit 2;; esac
# ATLANTIC_PROJECT selects another environment (e.g. one developing an isolated upstream checkout).
project="${ATLANTIC_PROJECT:-$project}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Peak device memory is sampled every 10 s alongside the run.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > "atlantic-${SLURM_JOB_ID}-gpumem.csv" &
export XLA_REACTANT_GPU_PREALLOCATE=false
timeout "${ATLANTIC_TIMEOUT:-0}" /usr/bin/time -v julia +1.12.7 --project=$project --compiled-modules=existing "scripts/$mode.jl"
