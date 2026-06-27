use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{error::ErrorKind, Args as ClapArgs, Parser, Subcommand};
use hdf5_pure::{
    CharacterSet, Datatype, File as Hdf5File, FileBuilder, Group, GroupBuilder, StringPadding,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::autocorrelation::mean;
use crate::initialization::{initialize_angles, initialize_disorder};
use crate::observables::{measure_theta_correlations, measure_theta_observables};
use crate::types::{InitMode, Parameters, ThetaLattice, ThetaScratch, WolffScratch};
use crate::updates::{metropolis_sweep_with_scratch, wolff_cluster_step_with_theta_scratch};

#[derive(Debug, Clone, PartialEq)]
pub struct BinnedEstimate {
    pub mean: f64,
    pub stderr: f64,
    pub bins: Vec<f64>,
    pub internal_bins: Vec<f64>,
    pub internal_bin_length: usize,
    pub rebin_length: usize,
}

impl BinnedEstimate {
    pub fn from_samples(samples: &[f64], internal_bin_length: usize) -> Result<Self, String> {
        if internal_bin_length == 0 {
            return Err("binsize must be positive".to_string());
        }
        let usable = samples.len() - samples.len() % internal_bin_length;
        if usable == 0 {
            return Err("binsize is larger than the sample series".to_string());
        }
        let internal_bins = samples[..usable]
            .chunks_exact(internal_bin_length)
            .map(mean)
            .collect::<Vec<_>>();
        let rebin_length = carlo_rebin_length(internal_bins.len());
        let rebin_usable = internal_bins.len() - internal_bins.len() % rebin_length;
        if rebin_usable == 0 {
            return Err("rebin length is larger than the internal bin series".to_string());
        }
        let bins = internal_bins[..rebin_usable]
            .chunks_exact(rebin_length)
            .map(mean)
            .collect::<Vec<_>>();
        Ok(Self {
            mean: mean(&bins),
            stderr: carlo_std_of_mean(&bins),
            bins,
            internal_bins,
            internal_bin_length,
            rebin_length,
        })
    }

    fn jackknife_difference(left: &Self, right: &Self) -> Result<Self, String> {
        let bin_count = left.bins.len().min(right.bins.len());
        if bin_count == 0 {
            return Err("jackknife difference requires at least one common bin".to_string());
        }
        let bins = left
            .bins
            .iter()
            .zip(&right.bins)
            .take(bin_count)
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>();
        let internal_bins = left
            .internal_bins
            .iter()
            .zip(&right.internal_bins)
            .take(left.internal_bins.len().min(right.internal_bins.len()))
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>();
        Ok(Self {
            mean: mean(&bins),
            stderr: carlo_std_of_mean(&bins),
            bins,
            internal_bins,
            internal_bin_length: left.internal_bin_length.min(right.internal_bin_length),
            rebin_length: left.rebin_length.min(right.rebin_length),
        })
    }
}

fn carlo_rebin_count(sample_count: usize) -> usize {
    if sample_count <= 10 {
        sample_count
    } else {
        10 + ((sample_count - 10) as f64).cbrt().round() as usize
    }
    .max(1)
}

fn carlo_rebin_length(total_sample_count: usize) -> usize {
    if total_sample_count == 0 {
        1
    } else {
        (total_sample_count / carlo_rebin_count(total_sample_count)).max(1)
    }
}

fn carlo_std_of_mean(bins: &[f64]) -> f64 {
    if bins.len() <= 1 {
        return f64::NAN;
    }
    let avg = mean(bins);
    let variance = bins.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / (bins.len() - 1) as f64;
    variance.sqrt() / (bins.len() as f64).sqrt()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservableEstimate {
    pub mean: f64,
    pub stderr: f64,
    pub error: f64,
    pub covariance: Option<f64>,
    pub autocorr_time: f64,
    pub bins: usize,
    pub bin_length: usize,
    pub rebin_len: usize,
    pub rebin_count: usize,
    pub internal_bin_len: usize,
}

impl ObservableEstimate {
    fn new(estimate: &BinnedEstimate, bin_length: usize) -> Self {
        Self {
            mean: estimate.mean,
            stderr: estimate.stderr,
            error: estimate.stderr,
            covariance: None,
            autocorr_time: 0.0,
            bins: estimate.bins.len(),
            bin_length,
            rebin_len: estimate.rebin_length,
            rebin_count: estimate.bins.len(),
            internal_bin_len: estimate.internal_bin_length,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScalarAccumulator {
    samples: Vec<f64>,
}

impl ScalarAccumulator {
    pub fn push(&mut self, value: f64) {
        self.samples.push(value);
    }

    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    pub fn estimate(&self, binsize: usize) -> Result<BinnedEstimate, String> {
        BinnedEstimate::from_samples(&self.samples, binsize)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaTask {
    pub name: String,
    pub l: usize,
    pub l_x: usize,
    pub l_y: usize,
    pub l_z: usize,
    pub temperature: f64,
    pub j_xy: f64,
    pub j_z_mean: f64,
    pub delta_j_z: f64,
    pub disorder_seed: u64,
    pub seed: u64,
    pub sample: usize,
    pub sweeps: usize,
    pub thermalization: usize,
    pub binsize: usize,
    pub proposal_width: f64,
    pub wolff_steps: usize,
    pub correlation_rmax: usize,
    pub correlation_rmax_xy: usize,
    pub correlation_rmax_z: usize,
    pub j_z_array: Option<Vec<f64>>,
}

impl ThetaTask {
    pub fn params(&self) -> Parameters {
        Parameters::new(self.j_xy, self.j_z_mean, self.delta_j_z, self.temperature)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaJob {
    pub name: String,
    pub tasks: Vec<ThetaTask>,
}

impl ThetaJob {
    pub fn selected_tasks(
        &self,
        assignment: JobAssignment,
    ) -> impl Iterator<Item = (usize, &ThetaTask)> {
        self.tasks.iter().enumerate().filter(move |(idx, _)| {
            assignment.world_size == 1 || idx % assignment.world_size == assignment.rank
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobAssignment {
    pub rank: usize,
    pub world_size: usize,
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaTaskResult {
    pub task: ThetaTask,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub task_index: usize,
    pub observables: BTreeMap<String, ObservableEstimate>,
    pub acceptance: f64,
    pub measurements: usize,
    #[serde(skip)]
    pub measurement_bins: BTreeMap<String, Vec<f64>>,
    #[serde(skip)]
    pub measurement_samples: BTreeMap<String, Vec<f64>>,
    #[serde(skip)]
    pub final_theta: Vec<f64>,
    #[serde(skip)]
    pub final_j_z: Vec<f64>,
    #[serde(skip)]
    pub rng_word_pos: u128,
    #[serde(skip)]
    pub thermalization_sweeps: usize,
    #[serde(skip)]
    pub measurement_sweeps: usize,
    #[serde(skip)]
    pub acceptance_sum: f64,
    #[serde(skip)]
    pub acceptance_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaJobResult {
    pub job_name: String,
    pub rank: usize,
    pub world_size: usize,
    pub tasks: Vec<ThetaTaskResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRunSummary {
    pub output_path: PathBuf,
    pub task_count: usize,
    pub elapsed_seconds: f64,
    pub checkpoint_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobMergeSummary {
    pub output_path: PathBuf,
    pub input_paths: Vec<PathBuf>,
    pub task_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobMpiRunSummary {
    pub run: JobRunSummary,
    pub merge: Option<JobMergeSummary>,
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaJobConfig {
    pub l: Vec<usize>,
    pub l_x: Option<Vec<usize>>,
    pub l_y: Option<Vec<usize>>,
    pub l_z: Option<Vec<usize>>,
    pub temperatures: Vec<f64>,
    pub delta_j_z: Vec<f64>,
    pub samples: usize,
    pub base_seed: u64,
    pub j_xy: f64,
    pub j_z_mean: f64,
    pub sweeps: usize,
    pub thermalization: usize,
    pub binsize: usize,
    pub proposal_width: f64,
    pub wolff_steps: usize,
    pub correlation_rmax: Option<usize>,
    pub correlation_rmax_xy: Option<usize>,
    pub correlation_rmax_z: Option<usize>,
    pub run_time: Duration,
    pub checkpoint_time: Duration,
    pub job_name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ThetaJobToml {
    pub name: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub merged_output_file: Option<PathBuf>,
    pub measurement_dir: Option<PathBuf>,
    pub checkpoint_dir: Option<PathBuf>,
    pub scheduler_dir: Option<PathBuf>,
    pub checkpoint: Option<bool>,
    pub run_time: Option<String>,
    pub checkpoint_time: Option<String>,
    pub model: Option<ThetaModelToml>,
    pub run: Option<ThetaRunToml>,
    pub measure: Option<ThetaMeasureToml>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ThetaModelToml {
    #[serde(alias = "L")]
    pub l: Option<Vec<usize>>,
    pub l_x: Option<Vec<usize>>,
    pub l_y: Option<Vec<usize>>,
    pub l_z: Option<Vec<usize>>,
    #[serde(alias = "T")]
    pub temperatures: Option<Vec<f64>>,
    pub t: Option<Vec<f64>>,
    pub delta_j_z: Option<Vec<f64>>,
    pub djz: Option<Vec<f64>>,
    pub samples: Option<usize>,
    pub base_seed: Option<u64>,
    pub j_xy: Option<f64>,
    pub j_z_mean: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ThetaRunToml {
    pub sweeps: Option<usize>,
    pub thermalization: Option<usize>,
    pub binsize: Option<usize>,
    pub proposal_width: Option<f64>,
    pub wolff_steps: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ThetaMeasureToml {
    pub corr_rmax: Option<usize>,
    pub corr_rmax_xy: Option<usize>,
    pub corr_rmax_z: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThetaRunOptions {
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub merged_output_file: Option<PathBuf>,
    pub measurement_dir: Option<PathBuf>,
    pub checkpoint_dir: Option<PathBuf>,
    pub scheduler_dir: Option<PathBuf>,
    pub checkpoint: bool,
    pub restart: bool,
    pub single: bool,
    pub rank: Option<usize>,
    pub world_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct ThetaCheckpointRuntime {
    path: PathBuf,
    interval: Duration,
    resume: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct ThetaCheckpointState {
    task: ThetaTask,
    task_index: usize,
    theta: Vec<f64>,
    j_z: Vec<f64>,
    rng_word_pos: u128,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
    acceptance_sum: f64,
    acceptance_count: usize,
    measurement_samples: BTreeMap<String, Vec<f64>>,
}

impl ThetaRunOptions {
    fn with_overrides(mut self, args: &CommandArgs) -> Self {
        if let Some(value) = &args.output_dir {
            self.output_dir = Some(value.clone());
        }
        if let Some(value) = &args.output_file {
            self.output_file = Some(value.clone());
        }
        if let Some(value) = &args.merged_output_file {
            self.merged_output_file = Some(value.clone());
        }
        if let Some(value) = &args.measurement_dir {
            self.measurement_dir = Some(value.clone());
        }
        if let Some(value) = &args.checkpoint_dir {
            self.checkpoint_dir = Some(value.clone());
        }
        if let Some(value) = &args.scheduler_dir {
            self.scheduler_dir = Some(value.clone());
        }
        if args.checkpoint {
            self.checkpoint = true;
        }
        if args.restart {
            self.restart = true;
        }
        if args.single {
            self.single = true;
            self.rank = Some(0);
            self.world_size = Some(1);
        }
        if let Some(value) = args.rank {
            self.rank = Some(value);
        }
        if let Some(value) = args.world_size {
            self.world_size = Some(value);
        }
        self
    }

    fn rank(&self) -> usize {
        if self.single {
            return 0;
        }
        self.rank.or_else(mpi_env_rank).unwrap_or(0)
    }

    fn world_size(&self) -> usize {
        if self.single {
            return 1;
        }
        self.world_size.or_else(mpi_env_world_size).unwrap_or(1)
    }

    fn assignment(&self) -> Result<JobAssignment, String> {
        JobAssignment::new(self.rank(), self.world_size())
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust XY/Theta job runner")]
pub struct ThetaCli {
    #[command(subcommand)]
    command: Option<ThetaCommand>,
}

#[derive(Debug, Subcommand)]
pub enum ThetaCommand {
    #[command(alias = "r")]
    Run(CommandArgs),
    #[command(alias = "m")]
    Merge(CommandArgs),
    RunMerge(CommandArgs),
    RunDynamic(CommandArgs),
    MpiRun(CommandArgs),
    Checkpoint(CommandArgs),
    #[command(alias = "s")]
    Status(CommandArgs),
    #[command(alias = "d")]
    Delete(CommandArgs),
}

#[derive(Debug, Clone, Default, ClapArgs)]
pub struct CommandArgs {
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub from_env: bool,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long)]
    pub output_file: Option<PathBuf>,
    #[arg(long)]
    pub merged_output_file: Option<PathBuf>,
    #[arg(long)]
    pub measurement_dir: Option<PathBuf>,
    #[arg(long)]
    pub checkpoint_dir: Option<PathBuf>,
    #[arg(long)]
    pub scheduler_dir: Option<PathBuf>,
    #[arg(long)]
    pub checkpoint: bool,
    #[arg(short, long)]
    pub single: bool,
    #[arg(short, long)]
    pub restart: bool,
    #[arg(long)]
    pub rank: Option<usize>,
    #[arg(long)]
    pub world_size: Option<usize>,
}

impl Default for ThetaJobConfig {
    fn default() -> Self {
        Self {
            l: vec![16],
            l_x: None,
            l_y: None,
            l_z: None,
            temperatures: vec![2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
            delta_j_z: vec![0.8],
            samples: 16,
            base_seed: 20260414,
            j_xy: 1.0,
            j_z_mean: 1.0,
            sweeps: 20_000,
            thermalization: 5_000,
            binsize: 50,
            proposal_width: std::f64::consts::PI,
            wolff_steps: 1,
            correlation_rmax: None,
            correlation_rmax_xy: None,
            correlation_rmax_z: None,
            run_time: Duration::from_secs(12 * 60 * 60),
            checkpoint_time: Duration::from_secs(30 * 60),
            job_name: "xy_theta_rust".to_string(),
        }
    }
}

impl ThetaJobConfig {
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<(Self, ThetaRunOptions), String> {
        let text = fs::read_to_string(path.as_ref()).map_err(|err| err.to_string())?;
        let spec = toml::from_str::<ThetaJobToml>(&text).map_err(|err| err.to_string())?;
        Ok(Self::from_toml_spec(spec))
    }

    pub fn from_toml_spec(spec: ThetaJobToml) -> (Self, ThetaRunOptions) {
        let mut cfg = Self::default();
        if let Some(name) = spec.name {
            cfg.job_name = name;
        }
        if let Some(run_time) = spec.run_time {
            cfg.run_time = parse_duration(&run_time).unwrap_or(cfg.run_time);
        }
        if let Some(checkpoint_time) = spec.checkpoint_time {
            cfg.checkpoint_time = parse_duration(&checkpoint_time).unwrap_or(cfg.checkpoint_time);
        }
        if let Some(model) = spec.model {
            if let Some(value) = model.l {
                cfg.l = value;
            }
            cfg.l_x = model.l_x;
            cfg.l_y = model.l_y;
            cfg.l_z = model.l_z;
            if let Some(value) = model.temperatures.or(model.t) {
                cfg.temperatures = value;
            }
            if let Some(value) = model.delta_j_z.or(model.djz) {
                cfg.delta_j_z = value;
            }
            if let Some(value) = model.samples {
                cfg.samples = value;
            }
            if let Some(value) = model.base_seed {
                cfg.base_seed = value;
            }
            if let Some(value) = model.j_xy {
                cfg.j_xy = value;
            }
            if let Some(value) = model.j_z_mean {
                cfg.j_z_mean = value;
            }
        }
        if let Some(run) = spec.run {
            if let Some(value) = run.sweeps {
                cfg.sweeps = value;
            }
            if let Some(value) = run.thermalization {
                cfg.thermalization = value;
            }
            if let Some(value) = run.binsize {
                cfg.binsize = value;
            }
            if let Some(value) = run.proposal_width {
                cfg.proposal_width = value;
            }
            if let Some(value) = run.wolff_steps {
                cfg.wolff_steps = value;
            }
        }
        if let Some(measure) = spec.measure {
            cfg.correlation_rmax = measure.corr_rmax;
            cfg.correlation_rmax_xy = measure.corr_rmax_xy.or(cfg.correlation_rmax);
            cfg.correlation_rmax_z = measure.corr_rmax_z.or(cfg.correlation_rmax);
        }
        let options = ThetaRunOptions {
            output_dir: spec.output_dir,
            output_file: spec.output_file,
            merged_output_file: spec.merged_output_file,
            measurement_dir: spec.measurement_dir,
            checkpoint_dir: spec.checkpoint_dir,
            scheduler_dir: spec.scheduler_dir,
            checkpoint: spec.checkpoint.unwrap_or(false),
            ..Default::default()
        };
        (cfg, options)
    }

    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();
        cfg.l = parse_env_list("XY_L", &cfg.l)?;
        cfg.l_x = parse_optional_env_list("XY_LX")?;
        cfg.l_y = parse_optional_env_list("XY_LY")?;
        cfg.l_z = parse_optional_env_list("XY_LZ")?;
        cfg.temperatures = parse_env_list("XY_T", &cfg.temperatures)?;
        cfg.delta_j_z = parse_env_list("XY_DJZ", &cfg.delta_j_z)?;
        cfg.samples = parse_env_value("XY_SAMPLES", cfg.samples)?;
        cfg.base_seed = parse_env_value("XY_BASE_SEED", cfg.base_seed)?;
        cfg.j_xy = parse_env_value("XY_JXY", cfg.j_xy)?;
        cfg.j_z_mean = parse_env_value("XY_JZ_MEAN", cfg.j_z_mean)?;
        cfg.sweeps = parse_env_value("XY_SWEEPS", cfg.sweeps)?;
        cfg.thermalization = parse_env_value("XY_THERMAL", cfg.thermalization)?;
        cfg.binsize = parse_env_value("XY_BINSIZE", cfg.binsize)?;
        cfg.proposal_width = parse_env_value("XY_PROPOSAL_WIDTH", cfg.proposal_width)?;
        cfg.wolff_steps = parse_env_value("XY_WOLFF_STEPS", cfg.wolff_steps)?;
        cfg.run_time = parse_env_duration("XY_RUN_TIME", cfg.run_time)?;
        cfg.checkpoint_time = parse_env_duration("XY_CHECKPOINT_TIME", cfg.checkpoint_time)?;
        cfg.correlation_rmax = parse_optional_env_value("XY_CORR_RMAX")?;
        cfg.correlation_rmax_xy =
            parse_optional_env_value("XY_CORR_RMAX_XY")?.or(cfg.correlation_rmax);
        cfg.correlation_rmax_z =
            parse_optional_env_value("XY_CORR_RMAX_Z")?.or(cfg.correlation_rmax);
        cfg.job_name = std::env::var("XY_JOB_NAME").unwrap_or_else(|_| {
            format!(
                "xy_carlo_L{}_dJz{}",
                join_display(&cfg.l),
                join_display(&cfg.delta_j_z)
            )
        });
        if parse_env_value("XY_RANKS_PER_RUN", 1usize)? != 1 {
            return Err(
                "XY_RANKS_PER_RUN must be 1; Rust theta job runner uses task-level parallelism"
                    .to_string(),
            );
        }
        Ok(cfg)
    }

    pub fn make_job(&self) -> Result<ThetaJob, String> {
        if self.samples == 0 {
            return Err("samples must be positive".to_string());
        }
        if self.binsize == 0 {
            return Err("binsize must be positive".to_string());
        }
        if self.sweeps < self.binsize {
            return Err("sweeps must be at least binsize".to_string());
        }
        let mut tasks = Vec::new();
        for (l_x, l_y, l_z, l) in self.lattice_specs() {
            for &delta_j_z in &self.delta_j_z {
                for &temperature in &self.temperatures {
                    for sample in 1..=self.samples {
                        let disorder_seed = self.base_seed + sample as u64 - 1;
                        let seed = self.base_seed
                            + 100_000 * l_z as u64
                            + 1_000 * sample as u64
                            + (100.0 * temperature).round() as u64
                            + (1_000.0 * delta_j_z).round() as u64;
                        let j_z_array = generate_layer_disorder_values(
                            l_z,
                            self.j_z_mean,
                            delta_j_z,
                            disorder_seed,
                        )?;
                        tasks.push(ThetaTask {
                            name: format!(
                                "L{}x{}x{}_T{:.6}_dJz{:.6}_sample{}",
                                l_x, l_y, l_z, temperature, delta_j_z, sample
                            ),
                            l,
                            l_x,
                            l_y,
                            l_z,
                            temperature,
                            j_xy: self.j_xy,
                            j_z_mean: self.j_z_mean,
                            delta_j_z,
                            disorder_seed,
                            seed,
                            sample,
                            sweeps: self.sweeps,
                            thermalization: self.thermalization,
                            binsize: self.binsize,
                            proposal_width: self.proposal_width,
                            wolff_steps: self.wolff_steps,
                            correlation_rmax: self.correlation_rmax.unwrap_or(l_z / 2),
                            correlation_rmax_xy: self
                                .correlation_rmax_xy
                                .unwrap_or_else(|| l_x.min(l_y) / 2),
                            correlation_rmax_z: self.correlation_rmax_z.unwrap_or(l_z / 2),
                            j_z_array: Some(j_z_array),
                        });
                    }
                }
            }
        }
        Ok(ThetaJob {
            name: self.job_name.clone(),
            tasks,
        })
    }

    fn lattice_specs(&self) -> Vec<(usize, usize, usize, usize)> {
        if self.l_x.is_none() && self.l_y.is_none() && self.l_z.is_none() {
            return self.l.iter().map(|&l| (l, l, l, l)).collect();
        }
        let l_x = self.l_x.as_deref().unwrap_or(&self.l);
        let l_y = self.l_y.as_deref().unwrap_or(&self.l);
        let l_z = self.l_z.as_deref().unwrap_or(&self.l);
        let mut specs = Vec::new();
        for &x in l_x {
            for &y in l_y {
                for &z in l_z {
                    specs.push((x, y, z, z));
                }
            }
        }
        specs
    }
}

#[derive(Debug, Default)]
struct ObservableSeries(BTreeMap<String, ScalarAccumulator>);

impl ObservableSeries {
    fn push(&mut self, name: impl Into<String>, value: f64) {
        self.0.entry(name.into()).or_default().push(value);
    }

    fn from_samples(samples: BTreeMap<String, Vec<f64>>) -> Self {
        Self(
            samples
                .into_iter()
                .map(|(name, samples)| (name, ScalarAccumulator { samples }))
                .collect(),
        )
    }

    fn samples(&self) -> BTreeMap<String, Vec<f64>> {
        self.0
            .iter()
            .map(|(name, acc)| (name.clone(), acc.samples().to_vec()))
            .collect()
    }

    fn estimates_and_measurement_bins(
        &self,
        binsize: usize,
    ) -> Result<
        (
            BTreeMap<String, ObservableEstimate>,
            BTreeMap<String, Vec<f64>>,
        ),
        String,
    > {
        let binned = self
            .0
            .iter()
            .map(|(name, acc)| Ok((name.clone(), acc.estimate(binsize)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let measurement_bins = binned
            .iter()
            .map(|(name, estimate)| (name.clone(), estimate.internal_bins.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut estimates = binned
            .iter()
            .map(|(name, estimate)| (name.clone(), ObservableEstimate::new(estimate, binsize)))
            .collect::<BTreeMap<_, _>>();
        if let (Some(rho_xy), Some(rho_z)) = (binned.get("RhoXY"), binned.get("RhoZ")) {
            let diff = BinnedEstimate::jackknife_difference(rho_xy, rho_z)?;
            estimates.insert(
                "RhoDifference".to_string(),
                ObservableEstimate::new(&diff, binsize),
            );
        }
        Ok((estimates, measurement_bins))
    }
}

pub fn generate_layer_disorder_values(
    l_z: usize,
    j_z_mean: f64,
    delta_j_z: f64,
    disorder_seed: u64,
) -> Result<Vec<f64>, String> {
    let mut lattice = ThetaLattice::new(1, 1, l_z)?;
    let mut rng = ChaCha8Rng::seed_from_u64(disorder_seed);
    initialize_disorder(
        &mut lattice,
        &Parameters::new(1.0, j_z_mean, delta_j_z, 1.0),
        &mut rng,
    )?;
    Ok(lattice.j_z)
}

pub fn run_theta_task(task: &ThetaTask) -> Result<ThetaTaskResult, String> {
    run_theta_task_with_checkpoint(task, 0, None)
}

fn run_theta_task_with_checkpoint(
    task: &ThetaTask,
    task_index: usize,
    checkpoint: Option<&ThetaCheckpointRuntime>,
) -> Result<ThetaTaskResult, String> {
    let params = task.params();
    let mut lattice = ThetaLattice::new(task.l_x, task.l_y, task.l_z)?;
    match &task.j_z_array {
        Some(j_z_array) => {
            if j_z_array.len() != task.l_z {
                return Err("J_z_array length must match Lz".to_string());
            }
            lattice.j_z.clone_from(j_z_array);
        }
        None => initialize_disorder(
            &mut lattice,
            &params,
            &mut ChaCha8Rng::seed_from_u64(task.disorder_seed),
        )?,
    }
    if lattice.j_z.iter().any(|&j| j < 0.0) {
        return Err(
            "theta simulation requires nonnegative J_z; got negative layer coupling".to_string(),
        );
    }

    let mut rng = ChaCha8Rng::seed_from_u64(task.seed);
    let mut thermalization_start = 0usize;
    let mut measurement_start = 0usize;
    let mut acceptance_sum = 0.0;
    let mut acceptance_count = 0usize;
    let mut series = ObservableSeries::default();

    if let Some(runtime) = checkpoint.filter(|runtime| runtime.resume && runtime.path.exists()) {
        let state = read_theta_task_checkpoint(&runtime.path)?;
        if state.task.l_x != task.l_x
            || state.task.l_y != task.l_y
            || state.task.l_z != task.l_z
            || (state.task.temperature - task.temperature).abs() > f64::EPSILON
        {
            return Err(format!(
                "checkpoint {} does not match requested theta task dimensions/temperature",
                runtime.path.display()
            ));
        }
        lattice.theta = state.theta;
        lattice.j_z = state.j_z;
        rng.set_word_pos(state.rng_word_pos);
        thermalization_start = state.thermalization_sweeps.min(task.thermalization);
        measurement_start = state.measurement_sweeps.min(task.sweeps);
        acceptance_sum = state.acceptance_sum;
        acceptance_count = state.acceptance_count;
        series = ObservableSeries::from_samples(state.measurement_samples);
    } else {
        initialize_angles(&mut lattice, InitMode::Random, &mut rng)?;
    }

    let mut theta_scratch = ThetaScratch::new(&lattice);
    let mut wolff_scratch = WolffScratch::new(&lattice);
    let mut last_checkpoint = Instant::now();

    for thermalization_sweeps in thermalization_start..task.thermalization {
        acceptance_sum += metropolis_sweep_with_scratch(
            &mut lattice,
            &params,
            &mut theta_scratch,
            task.proposal_width,
            &mut rng,
        )?;
        acceptance_count += 1;
        for _ in 0..task.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                &mut lattice,
                &params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                &mut rng,
            )?;
        }
        maybe_write_theta_checkpoint(
            checkpoint,
            &mut last_checkpoint,
            task,
            task_index,
            &lattice,
            &rng,
            thermalization_sweeps + 1,
            measurement_start,
            acceptance_sum,
            acceptance_count,
            &series,
        )?;
    }

    let volume = lattice.volume() as f64;
    let beta = 1.0 / task.temperature;
    let corr_rmax_xy = task.correlation_rmax_xy.min(task.l_x / 2).min(task.l_y / 2);
    let corr_rmax_z = task.correlation_rmax_z.min(task.l_z / 2);

    for measurement_sweeps in measurement_start..task.sweeps {
        let sweep_started = Instant::now();
        acceptance_sum += metropolis_sweep_with_scratch(
            &mut lattice,
            &params,
            &mut theta_scratch,
            task.proposal_width,
            &mut rng,
        )?;
        acceptance_count += 1;
        for _ in 0..task.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                &mut lattice,
                &params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                &mut rng,
            )?;
        }
        let sweep_seconds = sweep_started.elapsed().as_secs_f64();

        let measure_started = Instant::now();
        let obs = measure_theta_observables(&lattice, &params);
        let rho_x = obs.cos_x / volume - beta * obs.sin_x.powi(2) / volume;
        let rho_y = obs.cos_y / volume - beta * obs.sin_y.powi(2) / volume;
        let rho_z = obs.cos_z / volume - beta * obs.sin_z.powi(2) / volume;
        series.push("RhoXY", (rho_x + rho_y) / 2.0);
        series.push("RhoZ", rho_z);
        series.push("Energy", obs.energy);
        series.push("Magnetization", obs.magnetization);
        if corr_rmax_xy > 0 || corr_rmax_z > 0 {
            let corr =
                measure_theta_correlations(&lattice, None, Some(corr_rmax_xy), Some(corr_rmax_z));
            for (r, value) in corr.r_xy.iter().zip(corr.corr_x) {
                series.push(format!("CorrX_r{r}"), value);
            }
            for (r, value) in corr.r_xy.iter().zip(corr.corr_y) {
                series.push(format!("CorrY_r{r}"), value);
            }
            for (r, value) in corr.r_xy.iter().zip(corr.corr_xy) {
                series.push(format!("CorrXY_r{r}"), value);
            }
            for (r, value) in corr.r_z.iter().zip(corr.corr_z) {
                series.push(format!("CorrZ_r{r}"), value);
            }
        }
        let measure_seconds = measure_started.elapsed().as_secs_f64();
        series.push("_ll_sweep_time", sweep_seconds);
        series.push("_ll_measure_time", measure_seconds);
        maybe_write_theta_checkpoint(
            checkpoint,
            &mut last_checkpoint,
            task,
            task_index,
            &lattice,
            &rng,
            task.thermalization,
            measurement_sweeps + 1,
            acceptance_sum,
            acceptance_count,
            &series,
        )?;
    }

    if let Some(checkpoint) = checkpoint {
        let state = ThetaCheckpointState {
            task: task.clone(),
            task_index,
            theta: lattice.theta.clone(),
            j_z: lattice.j_z.clone(),
            rng_word_pos: rng.get_word_pos(),
            thermalization_sweeps: task.thermalization,
            measurement_sweeps: task.sweeps,
            acceptance_sum,
            acceptance_count,
            measurement_samples: series.samples(),
        };
        write_theta_checkpoint_state_to_path(&state, &checkpoint.path)?;
    }

    let measurement_samples = series.samples();
    let (observables, measurement_bins) = series.estimates_and_measurement_bins(task.binsize)?;
    Ok(ThetaTaskResult {
        task: task.clone(),
        task_index,
        observables,
        acceptance: acceptance_sum / acceptance_count.max(1) as f64,
        measurements: task.sweeps,
        measurement_bins,
        measurement_samples,
        final_theta: lattice.theta.clone(),
        final_j_z: lattice.j_z.clone(),
        rng_word_pos: rng.get_word_pos(),
        thermalization_sweeps: task.thermalization,
        measurement_sweeps: task.sweeps,
        acceptance_sum,
        acceptance_count,
    })
}

pub fn run_theta_job(job: &ThetaJob, assignment: JobAssignment) -> Result<ThetaJobResult, String> {
    let tasks = job
        .selected_tasks(assignment)
        .map(|(task_index, task)| run_theta_task_with_checkpoint(task, task_index, None))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ThetaJobResult {
        job_name: job.name.clone(),
        rank: assignment.rank,
        world_size: assignment.world_size,
        tasks,
    })
}

pub fn write_theta_job_result(
    result: &ThetaJobResult,
    output_dir: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    fs::create_dir_all(&output_dir).map_err(|err| err.to_string())?;
    let path = output_dir.as_ref().join(format!(
        "{}.rank{}.results.json",
        result.job_name, result.rank
    ));
    write_theta_job_result_to_path(result, path)
}

pub fn write_theta_job_result_to_path(
    result: &ThetaJobResult,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let file = File::create(&path).map_err(|err| err.to_string())?;
    serde_json::to_writer_pretty(BufWriter::new(file), result).map_err(|err| err.to_string())?;
    Ok(path.as_ref().to_path_buf())
}

pub fn write_theta_job_measurements(
    result: &ThetaJobResult,
    output_dir: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::with_capacity(result.tasks.len());
    for task_result in &result.tasks {
        let task_dir = output_dir
            .as_ref()
            .join(format!("task{:04}", task_result.task_index + 1));
        fs::create_dir_all(&task_dir).map_err(|err| err.to_string())?;
        let path = task_dir.join("run0001.meas.h5");
        write_theta_task_measurements_to_path(task_result, &path)?;
        paths.push(path);
    }
    Ok(paths)
}

pub fn write_theta_task_measurements_to_path(
    task_result: &ThetaTaskResult,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    if task_result.measurement_bins.is_empty() {
        return Err("theta task result does not contain measurement bins".to_string());
    }
    let path = path.as_ref();
    let tmp_path = path.with_extension("h5.tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let mut builder = FileBuilder::new();
    let mut observables_group = builder.create_group("observables");
    for (name, samples) in &task_result.measurement_bins {
        let Some(observable) = task_result.observables.get(name) else {
            return Err(format!("missing observable estimate for {name}"));
        };
        if observable.internal_bin_len == 0 {
            return Err(format!("{name}: internal bin length must be positive"));
        }
        if samples.is_empty() {
            return Err(format!("{name}: no complete measurement bins to write"));
        }
        let mut observable_group = observables_group.create_group(name);
        observable_group
            .create_dataset("bin_length")
            .with_i64_data(&[observable.internal_bin_len as i64])
            .with_shape(&[]);
        observable_group
            .create_dataset("samples")
            .with_f64_data(samples)
            .with_shape(&[samples.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1000]);
        observables_group.add_group(observable_group.finish());
    }
    builder.add_group(observables_group.finish());

    let mut version_group = builder.create_group("version");
    add_fixed_string_dataset(&mut version_group, "carlo_version", &carlo_version());
    add_fixed_string_dataset(&mut version_group, "mc_package", "SLSF.XYCarlo");
    add_fixed_string_dataset(&mut version_group, "mc_version", &mc_version());
    builder.add_group(version_group.finish());

    builder.write(&tmp_path).map_err(|err| err.to_string())?;
    fs::rename(&tmp_path, path).map_err(|err| err.to_string())?;
    Ok(path.to_path_buf())
}

fn add_fixed_string_dataset(group: &mut GroupBuilder, name: &str, value: &str) {
    let bytes = value.as_bytes().to_vec();
    group
        .create_dataset(name)
        .with_raw_data(
            Datatype::String {
                size: bytes.len() as u32,
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Ascii,
            },
            bytes,
            1,
        )
        .with_shape(&[]);
}

fn carlo_version() -> String {
    std::env::var("XY_CARLO_VERSION").unwrap_or_else(|_| "0.3.4".to_string())
}

fn mc_version() -> String {
    std::env::var("XY_MC_VERSION").unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string())
}

fn theta_task_checkpoint_path(output_dir: impl AsRef<Path>, task_index: usize) -> PathBuf {
    output_dir
        .as_ref()
        .join(format!("task{:04}", task_index + 1))
        .join("run0001.dump.h5")
}

fn checkpoint_dir_for_job(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| default_measurement_dir_with_options(job_name, options))
}

fn checkpoint_runtime_for_task(
    job_name: &str,
    options: &ThetaRunOptions,
    checkpoint_time: Duration,
    task_index: usize,
) -> Option<ThetaCheckpointRuntime> {
    if !(options.checkpoint || options.restart || checkpoint_enabled()) {
        return None;
    }
    let checkpoint_dir = checkpoint_dir_for_job(job_name, options);
    Some(ThetaCheckpointRuntime {
        path: theta_task_checkpoint_path(checkpoint_dir, task_index),
        interval: checkpoint_time,
        resume: options.restart,
    })
}

fn finish_theta_job_run(
    result: &ThetaJobResult,
    output_path: PathBuf,
    started: Instant,
    options: &ThetaRunOptions,
) -> Result<JobRunSummary, String> {
    let task_count = result.tasks.len();
    let measurement_dir = options
        .measurement_dir
        .clone()
        .unwrap_or_else(|| default_measurement_dir_with_options(&result.job_name, options));
    write_theta_job_measurements(result, measurement_dir)?;
    let checkpoint_paths = if options.checkpoint || checkpoint_enabled() {
        let checkpoint_dir = checkpoint_dir_for_job(&result.job_name, options);
        write_theta_job_checkpoints(result, checkpoint_dir)?
    } else {
        Vec::new()
    };
    let output_path = write_theta_job_result_to_path(result, output_path)?;
    Ok(JobRunSummary {
        output_path,
        task_count,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        checkpoint_paths,
    })
}

fn checkpoint_enabled() -> bool {
    std::env::var("XY_CHECKPOINT")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn write_theta_job_checkpoints(
    result: &ThetaJobResult,
    output_dir: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::with_capacity(result.tasks.len());
    for task_result in &result.tasks {
        let path = theta_task_checkpoint_path(output_dir.as_ref(), task_result.task_index);
        write_theta_task_checkpoint_to_path(task_result, &path)?;
        paths.push(path);
    }
    Ok(paths)
}

fn write_theta_checkpoint_state_to_path(
    state: &ThetaCheckpointState,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let path = path.as_ref();
    let tmp_path = path.with_extension("h5.tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let task_json = serde_json::to_string(&state.task).map_err(|err| err.to_string())?;
    let samples_json =
        serde_json::to_string(&state.measurement_samples).map_err(|err| err.to_string())?;

    let mut builder = FileBuilder::new();

    let mut parameters_group = builder.create_group("parameters");
    parameters_group
        .create_dataset("T")
        .with_f64_data(&[state.task.temperature])
        .with_shape(&[]);
    parameters_group
        .create_dataset("J_z")
        .with_f64_data(&state.j_z)
        .with_shape(&[state.j_z.len() as u64]);
    parameters_group
        .create_dataset("j_xy")
        .with_f64_data(&[state.task.j_xy])
        .with_shape(&[]);
    parameters_group
        .create_dataset("delta_j_z")
        .with_f64_data(&[state.task.delta_j_z])
        .with_shape(&[]);
    builder.add_group(parameters_group.finish());

    let mut state_group = builder.create_group("state");
    state_group
        .create_dataset("theta")
        .with_f64_data(&state.theta)
        .with_shape(&[
            state.task.l_x as u64,
            state.task.l_y as u64,
            state.task.l_z as u64,
        ]);
    state_group
        .create_dataset("rng_word_pos")
        .with_u64_data(&u128_to_u64_pair(state.rng_word_pos))
        .with_shape(&[2]);
    builder.add_group(state_group.finish());

    let mut progress_group = builder.create_group("progress");
    progress_group
        .create_dataset("task_index")
        .with_i64_data(&[state.task_index as i64])
        .with_shape(&[]);
    progress_group
        .create_dataset("thermalization_sweeps")
        .with_i64_data(&[state.thermalization_sweeps as i64])
        .with_shape(&[]);
    progress_group
        .create_dataset("measurement_sweeps")
        .with_i64_data(&[state.measurement_sweeps as i64])
        .with_shape(&[]);
    progress_group
        .create_dataset("target_thermalization")
        .with_i64_data(&[state.task.thermalization as i64])
        .with_shape(&[]);
    progress_group
        .create_dataset("target_sweeps")
        .with_i64_data(&[state.task.sweeps as i64])
        .with_shape(&[]);
    progress_group
        .create_dataset("acceptance_sum")
        .with_f64_data(&[state.acceptance_sum])
        .with_shape(&[]);
    progress_group
        .create_dataset("acceptance_count")
        .with_i64_data(&[state.acceptance_count as i64])
        .with_shape(&[]);
    builder.add_group(progress_group.finish());

    let mut measurements_group = builder.create_group("measurements");
    for (name, samples) in &state.measurement_samples {
        let mut observable_group = measurements_group.create_group(name);
        observable_group
            .create_dataset("samples")
            .with_f64_data(samples)
            .with_shape(&[samples.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1000]);
        measurements_group.add_group(observable_group.finish());
    }
    builder.add_group(measurements_group.finish());

    let mut metadata_group = builder.create_group("metadata");
    add_fixed_string_dataset(&mut metadata_group, "checkpoint_version", "1");
    add_fixed_string_dataset(&mut metadata_group, "model", "theta");
    add_fixed_string_dataset(&mut metadata_group, "task", &task_json);
    add_fixed_string_dataset(&mut metadata_group, "measurement_samples", &samples_json);
    builder.add_group(metadata_group.finish());

    let mut contexts_group = builder.create_group("contexts");
    let mut rank_group = contexts_group.create_group("rank0000");
    let mut simulation_group = rank_group.create_group("simulation");
    add_fixed_string_dataset(&mut simulation_group, "task", &task_json);
    simulation_group
        .create_dataset("task_index")
        .with_i64_data(&[state.task_index as i64])
        .with_shape(&[]);
    simulation_group
        .create_dataset("measurements")
        .with_i64_data(&[state.measurement_sweeps as i64])
        .with_shape(&[]);
    rank_group.add_group(simulation_group.finish());
    contexts_group.add_group(rank_group.finish());
    builder.add_group(contexts_group.finish());

    let mut version_group = builder.create_group("version");
    add_fixed_string_dataset(&mut version_group, "carlo_version", &carlo_version());
    add_fixed_string_dataset(&mut version_group, "mc_package", "SLSF.XYCarlo");
    add_fixed_string_dataset(&mut version_group, "mc_version", &mc_version());
    builder.add_group(version_group.finish());

    builder.write(&tmp_path).map_err(|err| err.to_string())?;
    fs::rename(&tmp_path, path).map_err(|err| err.to_string())?;
    Ok(path.to_path_buf())
}

fn theta_checkpoint_state_from_result(task_result: &ThetaTaskResult) -> ThetaCheckpointState {
    ThetaCheckpointState {
        task: task_result.task.clone(),
        task_index: task_result.task_index,
        theta: task_result.final_theta.clone(),
        j_z: task_result.final_j_z.clone(),
        rng_word_pos: task_result.rng_word_pos,
        thermalization_sweeps: task_result.thermalization_sweeps,
        measurement_sweeps: task_result.measurement_sweeps,
        acceptance_sum: task_result.acceptance_sum,
        acceptance_count: task_result.acceptance_count,
        measurement_samples: task_result.measurement_samples.clone(),
    }
}

fn maybe_write_theta_checkpoint(
    checkpoint: Option<&ThetaCheckpointRuntime>,
    last_checkpoint: &mut Instant,
    task: &ThetaTask,
    task_index: usize,
    lattice: &ThetaLattice,
    rng: &ChaCha8Rng,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
    acceptance_sum: f64,
    acceptance_count: usize,
    series: &ObservableSeries,
) -> Result<(), String> {
    let Some(checkpoint) = checkpoint else {
        return Ok(());
    };
    if checkpoint.interval > Duration::ZERO && last_checkpoint.elapsed() < checkpoint.interval {
        return Ok(());
    }
    let state = ThetaCheckpointState {
        task: task.clone(),
        task_index,
        theta: lattice.theta.clone(),
        j_z: lattice.j_z.clone(),
        rng_word_pos: rng.get_word_pos(),
        thermalization_sweeps,
        measurement_sweeps,
        acceptance_sum,
        acceptance_count,
        measurement_samples: series.samples(),
    };
    write_theta_checkpoint_state_to_path(&state, &checkpoint.path)?;
    *last_checkpoint = Instant::now();
    Ok(())
}

fn u128_to_u64_pair(value: u128) -> [u64; 2] {
    [(value >> 64) as u64, value as u64]
}

fn u64_pair_to_u128(values: &[u64]) -> Result<u128, String> {
    if values.len() != 2 {
        return Err("rng_word_pos dataset must contain two u64 values".to_string());
    }
    Ok(((values[0] as u128) << 64) | values[1] as u128)
}

fn hdf5_file(path: &Path) -> Result<Hdf5File, String> {
    Hdf5File::from_bytes(fs::read(path).map_err(|err| err.to_string())?)
        .map_err(|err| err.to_string())
}

fn read_scalar_i64(group: &Group<'_>, name: &str) -> Result<i64, String> {
    group
        .dataset(name)
        .map_err(|err| err.to_string())?
        .read_i64()
        .map_err(|err| err.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| format!("{name} dataset is empty"))
}

fn read_scalar_f64(group: &Group<'_>, name: &str) -> Result<f64, String> {
    group
        .dataset(name)
        .map_err(|err| err.to_string())?
        .read_f64()
        .map_err(|err| err.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| format!("{name} dataset is empty"))
}

fn read_scalar_string(group: &Group<'_>, name: &str) -> Result<String, String> {
    group
        .dataset(name)
        .map_err(|err| err.to_string())?
        .read_string()
        .map_err(|err| err.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| format!("{name} dataset is empty"))
}

fn read_theta_task_checkpoint(path: impl AsRef<Path>) -> Result<ThetaCheckpointState, String> {
    let path = path.as_ref();
    let file = hdf5_file(path)?;
    let parameters_group = file.group("parameters").map_err(|err| err.to_string())?;
    let state_group = file.group("state").map_err(|err| err.to_string())?;
    let progress_group = file.group("progress").map_err(|err| err.to_string())?;
    let metadata_group = file.group("metadata").map_err(|err| err.to_string())?;

    let task_json = read_scalar_string(&metadata_group, "task")?;
    let mut task: ThetaTask = serde_json::from_str(&task_json).map_err(|err| err.to_string())?;
    task.temperature = read_scalar_f64(&parameters_group, "T")?;
    let j_z = parameters_group
        .dataset("J_z")
        .map_err(|err| err.to_string())?
        .read_f64()
        .map_err(|err| err.to_string())?;
    let theta = state_group
        .dataset("theta")
        .map_err(|err| err.to_string())?
        .read_f64()
        .map_err(|err| err.to_string())?;
    if j_z.len() != task.l_z {
        return Err(format!(
            "checkpoint {} has J_z length {}, expected {}",
            path.display(),
            j_z.len(),
            task.l_z
        ));
    }
    if theta.len() != task.l_x * task.l_y * task.l_z {
        return Err(format!(
            "checkpoint {} has theta length {}, expected {}",
            path.display(),
            theta.len(),
            task.l_x * task.l_y * task.l_z
        ));
    }
    let rng_word_pos = u64_pair_to_u128(
        &state_group
            .dataset("rng_word_pos")
            .map_err(|err| err.to_string())?
            .read_u64()
            .map_err(|err| err.to_string())?,
    )?;
    let measurement_samples = read_scalar_string(&metadata_group, "measurement_samples")
        .and_then(|json| serde_json::from_str(&json).map_err(|err| err.to_string()))
        .unwrap_or_default();

    Ok(ThetaCheckpointState {
        task,
        task_index: read_scalar_i64(&progress_group, "task_index")? as usize,
        theta,
        j_z,
        rng_word_pos,
        thermalization_sweeps: read_scalar_i64(&progress_group, "thermalization_sweeps")? as usize,
        measurement_sweeps: read_scalar_i64(&progress_group, "measurement_sweeps")? as usize,
        acceptance_sum: read_scalar_f64(&progress_group, "acceptance_sum")?,
        acceptance_count: read_scalar_i64(&progress_group, "acceptance_count")? as usize,
        measurement_samples,
    })
}

pub fn write_theta_task_checkpoint_to_path(
    task_result: &ThetaTaskResult,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let state = theta_checkpoint_state_from_result(task_result);
    write_theta_checkpoint_state_to_path(&state, path)
}

pub fn run_theta_job_dynamic(
    job: &ThetaJob,
    scheduler_dir: impl AsRef<Path>,
    rank: usize,
    world_size: usize,
) -> Result<ThetaJobResult, String> {
    let scheduler_dir = scheduler_dir.as_ref();
    fs::create_dir_all(scheduler_dir).map_err(|err| err.to_string())?;

    let mut tasks = Vec::new();
    for (task_index, task) in job.tasks.iter().enumerate() {
        let done_path = scheduler_dir.join(format!("task{task_index:04}.done"));
        if done_path.exists() {
            continue;
        }
        let claim_path = scheduler_dir.join(format!("task{task_index:04}.claim"));
        let claim = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&claim_path);
        let Ok(mut claim) = claim else {
            continue;
        };
        writeln!(claim, "pid={} rank={rank}", std::process::id()).map_err(|err| err.to_string())?;
        let mut task_result = run_theta_task(task)?;
        task_result.task_index = task_index;
        File::create(&done_path).map_err(|err| err.to_string())?;
        tasks.push(task_result);
    }

    Ok(ThetaJobResult {
        job_name: job.name.clone(),
        rank,
        world_size,
        tasks,
    })
}

pub fn run_theta_job_dynamic_from_env() -> Result<JobRunSummary, String> {
    let cfg = ThetaJobConfig::from_env()?;
    let options = ThetaRunOptions::default();
    run_theta_job_dynamic_with_options(&cfg, options)
}
pub fn read_theta_job_result(path: impl AsRef<Path>) -> Result<ThetaJobResult, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    serde_json::from_reader(BufReader::new(file)).map_err(|err| err.to_string())
}

pub fn merge_theta_job_results(results: Vec<ThetaJobResult>) -> Result<ThetaJobResult, String> {
    let mut iter = results.into_iter();
    let first = iter
        .next()
        .ok_or_else(|| "at least one rank result is required for merge".to_string())?;
    let job_name = first.job_name.clone();
    let world_size = first.world_size;
    let mut tasks = first.tasks;

    for result in iter {
        if result.job_name != job_name {
            return Err("cannot merge results from different jobs".to_string());
        }
        if result.world_size != world_size {
            return Err("cannot merge results with different world sizes".to_string());
        }
        tasks.extend(result.tasks);
    }

    tasks.sort_by(|left, right| left.task.name.cmp(&right.task.name));
    for (task_index, task) in tasks.iter_mut().enumerate() {
        if task.task_index == 0 {
            task.task_index = task_index;
        }
    }
    Ok(ThetaJobResult {
        job_name,
        rank: 0,
        world_size,
        tasks,
    })
}

pub fn merge_theta_job_result_files(
    paths: impl IntoIterator<Item = impl AsRef<Path>>,
) -> Result<ThetaJobResult, String> {
    paths
        .into_iter()
        .map(read_theta_job_result)
        .collect::<Result<Vec<_>, _>>()
        .and_then(merge_theta_job_results)
}

fn mpi_merge_wait_timeout(cfg: &ThetaJobConfig) -> Duration {
    cfg.run_time + cfg.checkpoint_time + Duration::from_secs(60)
}

fn wait_for_rank_result_files(paths: &[PathBuf], timeout: Duration) -> Result<(), String> {
    let started = Instant::now();
    while started.elapsed() <= timeout {
        let missing = paths
            .iter()
            .filter(|path| !path.exists())
            .cloned()
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let missing = paths
        .iter()
        .filter(|path| !path.exists())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    Err(format!(
        "timed out waiting for MPI rank result file(s): {}",
        missing.join(", ")
    ))
}

fn restart_ready_path(cfg: &ThetaJobConfig, options: &ThetaRunOptions) -> PathBuf {
    options
        .scheduler_dir
        .clone()
        .unwrap_or_else(|| {
            default_measurement_dir_with_options(&cfg.job_name, options).join("scheduler")
        })
        .join(".restart-ready")
}

fn delete_theta_job_outputs(cfg: &ThetaJobConfig, options: &ThetaRunOptions) -> Result<(), String> {
    let merged_path = options
        .merged_output_file
        .clone()
        .unwrap_or_else(|| merged_result_path_with_options(&cfg.job_name, options));
    remove_path_if_exists(&merged_path)?;

    if let Some(path) = &options.output_file {
        remove_path_if_exists(path)?;
    }
    for rank in 0..options.world_size() {
        remove_path_if_exists(&rank_result_path_with_options(&cfg.job_name, rank, options))?;
    }

    let measurement_dir = options
        .measurement_dir
        .clone()
        .unwrap_or_else(|| default_measurement_dir_with_options(&cfg.job_name, options));
    remove_path_if_exists(&measurement_dir)?;

    if let Some(checkpoint_dir) = &options.checkpoint_dir {
        remove_path_if_exists(checkpoint_dir)?;
    }
    if let Some(scheduler_dir) = &options.scheduler_dir {
        remove_path_if_exists(scheduler_dir)?;
    }
    Ok(())
}

fn remove_path_if_exists(path: &Path) -> Result<(), String> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            fs::remove_dir_all(path).map_err(|err| err.to_string())
        }
        Ok(_) => fs::remove_file(path).map_err(|err| err.to_string()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.to_string()),
    }
}

fn theta_job_status(cfg: &ThetaJobConfig, options: &ThetaRunOptions) -> Result<String, String> {
    let job = cfg.make_job()?;
    let measurement_dir = default_measurement_dir_with_options(&job.name, options);
    let scheduler_dir = options
        .scheduler_dir
        .clone()
        .unwrap_or_else(|| measurement_dir.join("scheduler"));
    let completed_markers = job
        .tasks
        .iter()
        .enumerate()
        .filter(|(idx, _)| scheduler_dir.join(format!("task{idx:04}.done")).exists())
        .count();
    let completed_rank_tasks = (0..options.world_size())
        .filter_map(|rank| {
            read_theta_job_result(rank_result_path_with_options(&job.name, rank, options)).ok()
        })
        .map(|result| result.tasks.len())
        .sum::<usize>();
    let completed = completed_markers.max(completed_rank_tasks);
    let rank_files = (0..options.world_size())
        .filter(|&rank| rank_result_path_with_options(&job.name, rank, options).exists())
        .count();
    Ok(format!(
        "{} of {} theta task(s) marked done; {} of {} rank result file(s) present",
        completed,
        job.tasks.len(),
        rank_files,
        options.world_size()
    ))
}

pub fn run_theta_job_from_env() -> Result<JobRunSummary, String> {
    let cfg = ThetaJobConfig::from_env()?;
    let options = ThetaRunOptions::default();
    run_theta_job_with_options(&cfg, options)
}

pub fn run_theta_job_with_options(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobRunSummary, String> {
    let job = cfg.make_job()?;
    let assignment = options.assignment()?;
    let output_path = options
        .output_file
        .clone()
        .unwrap_or_else(|| rank_result_path_with_options(&job.name, assignment.rank, &options));

    let started = Instant::now();
    let tasks = job
        .selected_tasks(assignment)
        .map(|(task_index, task)| {
            let checkpoint =
                checkpoint_runtime_for_task(&job.name, &options, cfg.checkpoint_time, task_index);
            run_theta_task_with_checkpoint(task, task_index, checkpoint.as_ref())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = ThetaJobResult {
        job_name: job.name.clone(),
        rank: assignment.rank,
        world_size: assignment.world_size,
        tasks,
    };
    finish_theta_job_run(&result, output_path, started, &options)
}

pub fn run_theta_job_dynamic_with_options(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobRunSummary, String> {
    let job = cfg.make_job()?;
    let rank = options.rank();
    let world_size = options.world_size();
    let output_path = options
        .output_file
        .clone()
        .unwrap_or_else(|| rank_result_path_with_options(&job.name, rank, &options));
    let scheduler_dir = options.scheduler_dir.clone().unwrap_or_else(|| {
        default_measurement_dir_with_options(&job.name, &options).join("scheduler")
    });

    fs::create_dir_all(&scheduler_dir).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut tasks = Vec::new();
    for (task_index, task) in job.tasks.iter().enumerate() {
        let done_path = scheduler_dir.join(format!("task{task_index:04}.done"));
        if done_path.exists() {
            continue;
        }
        let claim_path = scheduler_dir.join(format!("task{task_index:04}.claim"));
        let claim = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&claim_path);
        let Ok(mut claim) = claim else {
            continue;
        };
        writeln!(claim, "pid={} rank={rank}", std::process::id()).map_err(|err| err.to_string())?;
        let checkpoint =
            checkpoint_runtime_for_task(&job.name, &options, cfg.checkpoint_time, task_index);
        let task_result = run_theta_task_with_checkpoint(task, task_index, checkpoint.as_ref())?;
        File::create(&done_path).map_err(|err| err.to_string())?;
        tasks.push(task_result);
    }
    let result = ThetaJobResult {
        job_name: job.name.clone(),
        rank,
        world_size,
        tasks,
    };
    finish_theta_job_run(&result, output_path, started, &options)
}

pub fn run_theta_job_mpi_with_options(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobMpiRunSummary, String> {
    let rank = options.rank();
    let world_size = options.world_size();
    if options.restart && rank == 0 {
        remove_path_if_exists(&restart_ready_path(cfg, &options))?;
    }

    let run = if options.single || world_size == 1 {
        let mut single_options = options.clone();
        single_options.single = true;
        single_options.rank = Some(0);
        single_options.world_size = Some(1);
        run_theta_job_with_options(cfg, single_options)?
    } else {
        run_theta_job_dynamic_with_options(cfg, options.clone())?
    };

    let merge = if !options.single && world_size > 1 && rank == 0 {
        let job = cfg.make_job()?;
        let input_paths = (0..world_size)
            .map(|rank| rank_result_path_with_options(&job.name, rank, &options))
            .collect::<Vec<_>>();
        wait_for_rank_result_files(&input_paths, mpi_merge_wait_timeout(cfg))?;
        Some(merge_theta_job_with_options(cfg, options.clone())?)
    } else {
        None
    };

    Ok(JobMpiRunSummary {
        run,
        merge,
        rank,
        world_size,
    })
}

pub fn merge_theta_job_from_env() -> Result<JobMergeSummary, String> {
    let cfg = ThetaJobConfig::from_env()?;
    let options = ThetaRunOptions::default();
    merge_theta_job_with_options(&cfg, options)
}

pub fn merge_theta_job_with_options(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobMergeSummary, String> {
    let job = cfg.make_job()?;
    let world_size = options.world_size();
    let input_paths = (0..world_size)
        .map(|rank| rank_result_path_with_options(&job.name, rank, &options))
        .collect::<Vec<_>>();
    let output_path = options
        .merged_output_file
        .clone()
        .unwrap_or_else(|| merged_result_path_with_options(&job.name, &options));
    let merged = merge_theta_job_result_files(&input_paths)?;
    let task_count = merged.tasks.len();
    let output_path = write_theta_job_result_to_path(&merged, output_path)?;
    Ok(JobMergeSummary {
        output_path,
        input_paths,
        task_count,
    })
}

pub fn run_theta_job_command_from_env(args: &[String]) -> Result<String, String> {
    run_theta_job_command(
        std::iter::once(OsString::from("slsf")).chain(args.iter().map(OsString::from)),
    )
}

pub fn run_theta_job_command<I, T>(args: I) -> Result<String, String>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match ThetaCli::try_parse_from(args) {
        Ok(cli) => {
            let command = cli.command.unwrap_or(ThetaCommand::Run(CommandArgs {
                from_env: true,
                ..Default::default()
            }));
            run_theta_command(command)
        }
        Err(err)
            if matches!(
                err.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            Ok(err.to_string())
        }
        Err(err) => Err(err.to_string()),
    }
}

fn run_theta_command(command: ThetaCommand) -> Result<String, String> {
    match command {
        ThetaCommand::Run(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display()
            ))
        }
        ThetaCommand::Merge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = merge_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "merged {} rank file(s), {} theta task(s); wrote {}",
                summary.input_paths.len(),
                summary.task_count,
                summary.output_path.display()
            ))
        }
        ThetaCommand::RunMerge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let run = run_theta_job_with_options(&cfg, options.clone())?;
            let merge = merge_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}; merged {} rank file(s) into {}",
                run.task_count,
                run.elapsed_seconds,
                run.output_path.display(),
                merge.input_paths.len(),
                merge.output_path.display()
            ))
        }
        ThetaCommand::RunDynamic(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_dynamic_with_options(&cfg, options)?;
            Ok(format!(
                "dynamically completed {} theta task(s) in {:.3}s; wrote {}; checkpoints {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display(),
                summary.checkpoint_paths.len()
            ))
        }
        ThetaCommand::MpiRun(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_mpi_with_options(&cfg, options)?;
            let merge_message = summary
                .merge
                .as_ref()
                .map(|merge| {
                    format!(
                        "; merged {} rank file(s) into {}",
                        merge.input_paths.len(),
                        merge.output_path.display()
                    )
                })
                .unwrap_or_default();
            Ok(format!(
                "MPI rank {}/{} completed {} theta task(s) in {:.3}s; wrote {}{}",
                summary.rank,
                summary.world_size,
                summary.run.task_count,
                summary.run.elapsed_seconds,
                summary.run.output_path.display(),
                merge_message
            ))
        }
        ThetaCommand::Checkpoint(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let job = cfg.make_job()?;
            let assignment = options.assignment()?;
            let result = run_theta_job(&job, assignment)?;
            let dir = options
                .checkpoint_dir
                .clone()
                .unwrap_or_else(|| default_measurement_dir_with_options(&job.name, &options));
            let paths = write_theta_job_checkpoints(&result, dir)?;
            Ok(format!("wrote {} checkpoint file(s)", paths.len()))
        }
        ThetaCommand::Status(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            theta_job_status(&cfg, &options)
        }
        ThetaCommand::Delete(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            delete_theta_job_outputs(&cfg, &options)?;
            Ok("deleted theta job outputs".to_string())
        }
    }
}

fn load_config_and_options(
    args: &CommandArgs,
) -> Result<(ThetaJobConfig, ThetaRunOptions), String> {
    let (cfg, options) = if let Some(path) = &args.config {
        ThetaJobConfig::from_toml_path(path)?
    } else {
        (ThetaJobConfig::from_env()?, ThetaRunOptions::default())
    };
    Ok((cfg, options.with_overrides(args)))
}

fn rank_result_path_with_options(
    job_name: &str,
    rank: usize,
    options: &ThetaRunOptions,
) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.rank{rank}.results.json", file_stem(job_name)))
}

fn merged_result_path_with_options(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.results.json", file_stem(job_name)))
}

fn default_measurement_dir_with_options(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.data", file_stem(job_name)))
}

fn default_output_dir(job_name: &str) -> PathBuf {
    Path::new(job_name)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("results"))
}

fn file_stem(job_name: &str) -> String {
    Path::new(job_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(job_name)
        .to_string()
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

fn env_usize_any(keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
}

fn mpi_env_rank() -> Option<usize> {
    env_usize_any(&[
        "XY_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    ])
}

fn mpi_env_world_size() -> Option<usize> {
    env_usize_any(&[
        "XY_WORLD_SIZE",
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ])
}

fn parse_env_duration(key: &str, default: Duration) -> Result<Duration, String> {
    std::env::var(key).map_or(Ok(default), |value| parse_duration(&value))
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let (date_part, time_part) = value
        .split_once('-')
        .map_or(("0", value), |(days, rest)| (days, rest));
    let days = date_part
        .parse::<u64>()
        .map_err(|err| format!("failed to parse duration days in {value}: {err}"))?;
    let parts = time_part.split(':').collect::<Vec<_>>();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("{value} does not match [[HH:]MM:]SS"));
    }
    let numbers = parts
        .iter()
        .map(|part| {
            part.parse::<u64>()
                .map_err(|err| format!("failed to parse duration component in {value}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (hours, minutes, seconds) = match numbers.as_slice() {
        [seconds] => (0, 0, *seconds),
        [minutes, seconds] => (0, *minutes, *seconds),
        [hours, minutes, seconds] => (*hours, *minutes, *seconds),
        _ => unreachable!(),
    };
    Ok(Duration::from_secs(
        days * 24 * 60 * 60 + hours * 60 * 60 + minutes * 60 + seconds,
    ))
}

fn parse_env_value<T>(key: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or(Ok(default), |value| {
        value
            .parse::<T>()
            .map_err(|err| format!("failed to parse {key}: {err}"))
    })
}

fn parse_optional_env_value<T>(key: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or(Ok(None), |value| {
        value
            .parse::<T>()
            .map(Some)
            .map_err(|err| format!("failed to parse {key}: {err}"))
    })
}

fn parse_env_list<T>(key: &str, default: &[T]) -> Result<Vec<T>, String>
where
    T: Clone + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or_else(|_| Ok(default.to_vec()), |value| parse_list(key, &value))
}

fn parse_optional_env_list<T>(key: &str) -> Result<Option<Vec<T>>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or(Ok(None), |value| parse_list(key, &value).map(Some))
}

fn parse_list<T>(key: &str, value: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let parsed = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<T>()
                .map_err(|err| format!("failed to parse {key}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(parsed)
}

fn join_display<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}
