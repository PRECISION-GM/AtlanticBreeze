#!/bin/bash
#SBATCH --job-name=atlantic-forward
#SBATCH --partition=gpua100largex4,gpua100largex4spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=atlantic-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the AtlanticBreeze repository}"
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia:"
export JULIA_CUDA_USE_COMPAT=false JULIA_NUM_THREADS=8 JULIA_NUM_PRECOMPILE_TASKS=2
unset LD_LIBRARY_PATH
: "${ATLANTIC_INPUT:?Set ATLANTIC_INPUT to the prepared interval directory}"
export ATLANTIC_OUTPUT="${ATLANTIC_OUTPUT:-outputs/forward-${SLURM_JOB_ID}}"
git rev-parse HEAD
sha256sum scripts/model.jl scripts/forward.jl scripts/train.jl Project.toml Manifest.toml
mode="${1:-forward}"
case "$mode" in forward|train) ;; *) echo 'Expected forward or train'; exit 2;; esac
julia +1.12.7 --project=. --compiled-modules=existing "scripts/$mode.jl"
