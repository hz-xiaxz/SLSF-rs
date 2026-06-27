#!/usr/bin/env bash
#SBATCH --job-name=slsf-small
#SBATCH --partition=amd_512
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -euo pipefail

source /public1/soft/modules/module.sh
module load mpi

CONFIG=${CONFIG:-examples/theta_jxy12_prod_small.toml}
BINARY=${BINARY:-target/release/slsf}
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}

mkdir -p runs

"$CARGO" build --release

srun --wait=0 "$BINARY" mpi-run --config "$CONFIG" --restart
