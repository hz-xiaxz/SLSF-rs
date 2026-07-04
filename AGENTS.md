# Agent Notes

- This crate uses `std::simd` and `sleef` SIMD entry points, so build and test with the Rust nightly toolchain.
- Prefer `cargo +nightly test` for validation.
- Prefer `cargo +nightly build --release` for production binaries.
- Slurm job scripts should compile the binary on the allocated node before launching the MPI workload, so `target-cpu=native` matches the compute node CPU.
