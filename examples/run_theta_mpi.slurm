#!/usr/bin/env bash
#SBATCH --job-name=slsf-theta-mpi
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -euo pipefail

source /public1/soft/modules/module.sh
module load mpi

# Override CONFIG at submission time to choose a split, for example:
#   sbatch --export=ALL,CONFIG=examples/theta_jxy12_prod_mid.toml examples/run_theta_mpi.slurm
#   sbatch --export=ALL,CONFIG=examples/theta_jxy12_prod_large.toml examples/run_theta_mpi.slurm
CONFIG=${CONFIG:-examples/theta_jxy12_prod_small.toml}
TARGET_DIR=${TARGET_DIR:-target/native-nightly}
BINARY=${BINARY:-$TARGET_DIR/release/slsf}
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}

mkdir -p runs

# Build on the allocated compute node so target-cpu=native matches the CPU
# that will run the simulation, including AVX512 when available.
RUSTFLAGS="$RUSTFLAGS" "$CARGO" +nightly build --release --target-dir "$TARGET_DIR"

# The runner reads SLURM_PROCID and SLURM_NTASKS automatically.
# Rank 0 merges rank result files after all ranks finish.
srun --wait=0 "$BINARY" mpi-run --config "$CONFIG" --restart
