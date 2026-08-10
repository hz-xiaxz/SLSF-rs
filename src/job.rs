#![allow(clippy::items_after_test_module)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{error::ErrorKind, Args as ClapArgs, Parser, Subcommand};
use hdf5_pure::{
    CharacterSet, Datatype, File as Hdf5File, FileBuilder, Group, GroupBuilder, StringPadding,
};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::initialization::{initialize_angles, initialize_two_point_layer_disorder};
use crate::observables::{
    measure_theta_correlations_with_scratch, measure_theta_observables_with_scratch,
};
use crate::types::{
    FastRng, InitMode, Parameters, ThetaLattice, ThetaObservables, ThetaScratch, WolffScratch,
};
use crate::updates::{metropolis_sweep_with_scratch, wolff_cluster_step_with_theta_scratch};

#[cfg(feature = "carlo-mc")]
pub use carlo_mc::*;

#[cfg(not(feature = "carlo-mc"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobAssignment {
    pub rank: usize,
    pub world_size: usize,
}

#[cfg(not(feature = "carlo-mc"))]
impl JobAssignment {
    pub fn new(rank: usize, world_size: usize) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be positive".to_string());
        }
        if rank >= world_size {
            return Err("rank must be smaller than world_size".to_string());
        }
        Ok(Self { rank, world_size })
    }

    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    pub fn from_env() -> Result<Self, String> {
        Self::new(
            mpi_env_rank().unwrap_or(0),
            mpi_env_world_size().unwrap_or(1),
        )
    }
}

include!("job/model.rs");
include!("job/cli.rs");
include!("job/config.rs");
include!("job/simulation.rs");
include!("job/results.rs");
include!("job/checkpoint.rs");
include!("job/runner.rs");
include!("job/command.rs");
include!("job/paths.rs");
