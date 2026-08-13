#![feature(portable_simd)]

#[cfg(feature = "carlo-mc")]
pub use carlo_mc;
#[cfg(feature = "carlo-mc")]
pub use carlo_mc::*;

pub(crate) mod fast_math;
pub mod initialization;
pub mod model;
pub mod cli;
pub mod results;
pub mod observables;
pub mod result_tools;
pub mod simulation;
#[cfg(test)]
mod tests;
pub mod types;
pub mod updates;

pub use initialization::*;
pub use model::*;
pub use cli::*;
pub use results::*;
pub use observables::*;
pub use result_tools::*;
pub use simulation::*;
pub use types::*;
pub use updates::*;
