# SLSF-rs

A classical Monte Carlo simulation of the three-dimensional disordered XY model, written in Rust.

The simulation supports Wolff cluster updates and SIMD-accelerated Metropolis updates using checkerboard decomposition. Its data-management workflow is inspired by [Carlo.jl](https://github.com/lukas-weber/Carlo.jl) and reimplemented in Rust.
