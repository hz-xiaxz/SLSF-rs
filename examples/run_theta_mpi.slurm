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
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-nightly}
RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}
PROFILE=${PROFILE:-none}
PROFILE_EVENTS=${PROFILE_EVENTS:-cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cache-misses}
PERF_FREQ=${PERF_FREQ:-999}
PERF_CALL_GRAPH=${PERF_CALL_GRAPH:-dwarf}
BUILD_PROFILE=release
if [ "$PROFILE" != "none" ]; then
    BUILD_PROFILE=profiling
fi
BINARY=${BINARY:-$TARGET_DIR/$BUILD_PROFILE/slsf}

mkdir -p runs

# Build on the allocated compute node before launching MPI so target-cpu=native
# matches the CPU that will run the simulation, including AVX512 when available.
# The sleef SIMD dependency requires nightly because it uses portable_simd.
if [ "$BUILD_PROFILE" = "release" ]; then
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --release --target-dir "$TARGET_DIR"
else
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --profile profiling --target-dir "$TARGET_DIR"
fi

# The runner reads SLURM_PROCID and SLURM_NTASKS automatically.
# Rank 0 merges rank result files after all ranks finish.
case "$PROFILE" in
    none)
        srun --wait=0 "$BINARY" mpi-run --config "$CONFIG" --restart
        ;;
    stat)
        srun --ntasks=1 perf stat -d -e "$PROFILE_EVENTS" -o "runs/perf-stat-${SLURM_JOB_ID}.txt" \
            "$BINARY" mpi-run --single --config "$CONFIG" --restart
        ;;
    record)
        srun --ntasks=1 perf record -F "$PERF_FREQ" --call-graph "$PERF_CALL_GRAPH" \
            -o "runs/perf-${SLURM_JOB_ID}.data" \
            "$BINARY" mpi-run --single --config "$CONFIG" --restart
        perf report -i "runs/perf-${SLURM_JOB_ID}.data" --stdio > "runs/perf-${SLURM_JOB_ID}.report.txt"
        ;;
    *)
        echo "unsupported PROFILE=$PROFILE; expected none, stat, or record" >&2
        exit 2
        ;;
esac
