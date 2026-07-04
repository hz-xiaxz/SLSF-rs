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
TARGET_DIR=${TARGET_DIR:-target/native-nightly}
BINARY=${BINARY:-$TARGET_DIR/release/slsf}
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-nightly}
RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}

mkdir -p runs

# Build on the allocated compute node before launching MPI so target-cpu=native
# matches the CPU that will run the simulation, including AVX512 when available.
# The sleef SIMD dependency requires nightly because it uses portable_simd.
RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --release --target-dir "$TARGET_DIR"

srun --wait=0 "$BINARY" mpi-run --config "$CONFIG" --restart
