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
use crate::observables::{measure_theta_correlations, measure_theta_observables_with_scratch};
use crate::types::{
    FastRng, InitMode, Parameters, ThetaLattice, ThetaObservables, ThetaScratch, WolffScratch,
};
use crate::updates::{metropolis_sweep_with_scratch, wolff_cluster_step_with_theta_scratch};

include!("job/model.rs");
include!("job/cli.rs");
include!("job/config.rs");
include!("job/simulation.rs");
include!("job/results.rs");
include!("job/checkpoint.rs");
include!("job/runner.rs");
include!("job/command.rs");
include!("job/paths.rs");
