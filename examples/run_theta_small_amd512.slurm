#!/usr/bin/env bash
#SBATCH --job-name=slsf-small
#SBATCH --partition=amd_m9_768
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -euo pipefail

MODULE_INIT=${MODULE_INIT:-/publicfs10/fs10-share/soft/share-soft/modules/module.sh}
MPI_MODULE=${MPI_MODULE:-mpi/oneAPI/2022.1}
if [ -r "$MODULE_INIT" ]; then
    source "$MODULE_INIT"
    module load "$MPI_MODULE"
else
    echo "module init script not found: $MODULE_INIT" >&2
    exit 1
fi

CONFIG=${CONFIG:-examples/theta_jxy12_prod_small.toml}
TARGET_DIR=${TARGET_DIR:-target/native-nightly}
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-nightly}
RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}
PROFILE=${PROFILE:-none}
CARGO_FEATURES=${CARGO_FEATURES:-}
BENCH_REPEATS=${BENCH_REPEATS:-3}
BENCH_OUTPUT_DIR=${BENCH_OUTPUT_DIR:-}
PROFILE_EVENTS=${PROFILE_EVENTS:-cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cache-misses}
PERF_FREQ=${PERF_FREQ:-999}
PERF_CALL_GRAPH=${PERF_CALL_GRAPH:-dwarf}
BUILD_PROFILE=release
if [ "$PROFILE" = "stat" ] || [ "$PROFILE" = "record" ]; then
    BUILD_PROFILE=profiling
fi
BINARY=${BINARY:-$TARGET_DIR/$BUILD_PROFILE/slsf}

mkdir -p runs

# Build on the allocated compute node before launching MPI so target-cpu=native
# matches the CPU that will run the simulation, including AVX512 when available.
# The sleef SIMD dependency requires nightly because it uses portable_simd.
if [ -n "$CARGO_FEATURES" ]; then
    FEATURE_ARGS=(--features "$CARGO_FEATURES")
else
    FEATURE_ARGS=()
fi
if [ "$BUILD_PROFILE" = "release" ]; then
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --release --target-dir "$TARGET_DIR" "${FEATURE_ARGS[@]}"
else
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --profile profiling --target-dir "$TARGET_DIR" "${FEATURE_ARGS[@]}"
fi

case "$PROFILE" in
    none)
        srun --wait=0 "$BINARY" mpi-run --config "$CONFIG" --restart
        ;;
    benchmark)
        srun --ntasks=1 "$BINARY" mpi-run --single --config "$CONFIG" --restart
        ;;
    benchmark-repeat)
        if ! [[ "$BENCH_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
            echo "BENCH_REPEATS must be a positive integer" >&2
            exit 2
        fi
        if [ -z "$BENCH_OUTPUT_DIR" ]; then
            echo "BENCH_OUTPUT_DIR is required for PROFILE=benchmark-repeat" >&2
            exit 2
        fi
        mkdir -p "$BENCH_OUTPUT_DIR"
        for ((run = 1; run <= BENCH_REPEATS; run++)); do
            rm -rf runs/theta_l64_s1000_benchmark
            /usr/bin/time -f 'elapsed_seconds=%e user_seconds=%U system_seconds=%S max_rss_kb=%M' \
                -o "$BENCH_OUTPUT_DIR/run-${run}.time.txt" \
                srun --ntasks=1 "$BINARY" mpi-run --single --config "$CONFIG"
            cp runs/theta_l64_s1000_benchmark/theta_l64_s1000_benchmark.rank0.results.json \
                "$BENCH_OUTPUT_DIR/run-${run}.results.json"
        done
        ;;
    benchmark-ab)
        if ! [[ "$BENCH_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
            echo "BENCH_REPEATS must be a positive integer" >&2
            exit 2
        fi
        if [ -z "$BENCH_OUTPUT_DIR" ]; then
            echo "BENCH_OUTPUT_DIR is required for PROFILE=benchmark-ab" >&2
            exit 2
        fi
        MIXED_TARGET_DIR=${MIXED_TARGET_DIR:-${TARGET_DIR}-mixed-vector-exp}
        RUSTFLAGS="$RUSTFLAGS" "$CARGO" +"$RUST_TOOLCHAIN" build --release \
            --target-dir "$MIXED_TARGET_DIR" --features mixed-vector-exp
        MIXED_BINARY="$MIXED_TARGET_DIR/release/slsf"
        mkdir -p "$BENCH_OUTPUT_DIR/baseline" "$BENCH_OUTPUT_DIR/mixed-vector-exp"
        for ((run = 1; run <= BENCH_REPEATS; run++)); do
            if ((run % 2 == 1)); then
                strategies=(baseline mixed-vector-exp)
            else
                strategies=(mixed-vector-exp baseline)
            fi
            for strategy in "${strategies[@]}"; do
                if [ "$strategy" = baseline ]; then
                    run_binary=$BINARY
                else
                    run_binary=$MIXED_BINARY
                fi
                rm -rf runs/theta_l64_s1000_benchmark
                /usr/bin/time -f 'elapsed_seconds=%e user_seconds=%U system_seconds=%S max_rss_kb=%M' \
                    -o "$BENCH_OUTPUT_DIR/$strategy/run-${run}.time.txt" \
                    srun --ntasks=1 "$run_binary" mpi-run --single --config "$CONFIG"
                cp runs/theta_l64_s1000_benchmark/theta_l64_s1000_benchmark.rank0.results.json \
                    "$BENCH_OUTPUT_DIR/$strategy/run-${run}.results.json"
            done
        done
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
        echo "unsupported PROFILE=$PROFILE; expected none, benchmark, benchmark-repeat, benchmark-ab, stat, or record" >&2
        exit 2
        ;;
esac
