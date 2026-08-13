use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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
use carlo_mc::*;


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaTask {
    pub name: String,
    pub l: usize,
    pub l_x: usize,
    pub l_y: usize,
    pub l_z: usize,
    pub temperature: f64,
    pub j_xy: f64,
    pub delta_j_xy: f64,
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
    #[serde(default = "one_usize")]
    pub correlation_interval: usize,
    pub j_xy_array: Option<Vec<f64>>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaTaskResult {
    pub task: ThetaTask,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub task_index: usize,
    pub observables: BTreeMap<String, carlo_mc::Estimate>,
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
    pub stopped_early: bool,
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
    pub delta_j_xy: Vec<f64>,
    pub samples: usize,
    pub base_seed: u64,
    pub j_xy: f64,
    pub j_z_mean: f64,
    pub sweeps: usize,
    pub thermalization: usize,
    pub binsize: usize,
    pub proposal_width: f64,
    pub wolff_steps: usize,
    pub correlation_rmax: Option<Vec<usize>>,
    pub correlation_rmax_xy: Option<Vec<usize>>,
    pub correlation_rmax_z: Option<Vec<usize>>,
    pub correlation_interval: usize,
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
    pub delta_j_xy: Option<Vec<f64>>,
    pub djxy: Option<Vec<f64>>,
    pub j_z_mean: Option<f64>,
    pub j_z: Option<f64>,
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
    pub corr_rmax: Option<Vec<usize>>,
    pub corr_rmax_xy: Option<Vec<usize>>,
    pub corr_rmax_z: Option<Vec<usize>>,
    pub corr_interval: Option<usize>,
    pub correlation_interval: Option<usize>,
}

fn one_usize() -> usize {
    1
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

#[cfg(feature = "carlo-mc")]
pub fn theta_tasks_from_config(cfg: &ThetaJobConfig) -> Result<Vec<Task<ThetaTask>>, String> {
    let theta_tasks = cfg.make_theta_tasks()?;
    Ok(theta_tasks
        .into_iter()
        .map(|task| {
            let name = task.name.clone();
            let sweeps = task.sweeps;
            let thermalization = task.thermalization;
            let binsize = task.binsize;
            let seed = task.seed;
            Task::new(name, task)
                .sweeps(sweeps)
                .thermalization(thermalization)
                .binsize(binsize)
                .seed(seed)
        })
        .collect())
}

#[cfg(feature = "carlo-mc")]
#[derive(Debug, Serialize, Deserialize)]
pub struct ThetaModel {
    lattice: ThetaLattice,
    params: Parameters,
    rng: FastRng,
    theta_scratch: ThetaScratch,
    wolff_scratch: WolffScratch,
    proposal_width: f64,
    wolff_steps: usize,
    corr_rmax_xy: usize,
    corr_rmax_z: usize,
    correlation_interval: usize,
    acceptance_sum: f64,
    acceptance_count: usize,
    measurement_count: usize,
    last_sweep_seconds: f64,
}

#[cfg(feature = "carlo-mc")]
#[derive(Debug)]
pub struct ThetaModelError(String);

#[cfg(feature = "carlo-mc")]
impl std::fmt::Display for ThetaModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "carlo-mc")]
impl std::error::Error for ThetaModelError {}

#[cfg(feature = "carlo-mc")]
impl From<String> for ThetaModelError {
    fn from(message: String) -> Self {
        Self(message)
    }
}

#[cfg(feature = "carlo-mc")]
impl From<&str> for ThetaModelError {
    fn from(message: &str) -> Self {
        Self(message.to_string())
    }
}

#[cfg(feature = "carlo-mc")]
impl From<GenericJobError> for ThetaModelError {
    fn from(error: GenericJobError) -> Self {
        Self(error.to_string())
    }
}

#[cfg(feature = "carlo-mc")]
impl MonteCarlo for ThetaModel {
    type Parameters = ThetaTask;
    type Error = ThetaModelError;
    type Estimate = carlo_mc::Estimate;

    fn new(task: &ThetaTask) -> Result<Self, Self::Error> {
        let params = task.params();
        let mut lattice = ThetaLattice::new(task.l_x, task.l_y, task.l_z)?;
        let mut disorder_rng = FastRng::seed_from_u64(task.disorder_seed);
        if let Some(j_xy_array) = &task.j_xy_array {
            if j_xy_array.len() != task.l_z {
                return Err("J_xy_array length must match Lz".into());
            }
            lattice.j_xy.clone_from(j_xy_array);
        } else {
            initialize_two_point_layer_disorder(
                &mut lattice.j_xy,
                task.j_xy,
                task.delta_j_xy,
                &mut disorder_rng,
                "J_xy",
            )?;
        }
        if let Some(j_z_array) = &task.j_z_array {
            if j_z_array.len() != task.l_z {
                return Err("J_z_array length must match Lz".into());
            }
            lattice.j_z.clone_from(j_z_array);
        } else {
            initialize_two_point_layer_disorder(
                &mut lattice.j_z,
                task.j_z_mean,
                task.delta_j_z,
                &mut disorder_rng,
                "J_z",
            )?;
        }
        if lattice.j_xy.iter().any(|&j| j < 0.0) || lattice.j_z.iter().any(|&j| j < 0.0) {
            return Err("theta simulation requires nonnegative layer couplings".into());
        }
        let rng = FastRng::seed_from_u64(task.seed);
        let theta_scratch = ThetaScratch::new(&lattice);
        let wolff_scratch = WolffScratch::new(&lattice);
        Ok(Self {
            lattice,
            params,
            rng,
            theta_scratch,
            wolff_scratch,
            proposal_width: task.proposal_width,
            wolff_steps: task.wolff_steps,
            corr_rmax_xy: task
                .correlation_rmax_xy
                .min(task.l_x / 2)
                .min(task.l_y / 2),
            corr_rmax_z: task.correlation_rmax_z.min(task.l_z / 2),
            correlation_interval: task.correlation_interval.max(1),
            acceptance_sum: 0.0,
            acceptance_count: 0,
            measurement_count: 0,
            last_sweep_seconds: 0.0,
        })
    }

    fn init(&mut self, _context: &mut Context) -> Result<(), Self::Error> {
        initialize_angles(&mut self.lattice, InitMode::Random, &mut self.rng)?;
        self.theta_scratch.refresh(&self.lattice)?;
        Ok(())
    }

    fn sweep(&mut self, _context: &mut Context) -> Result<(), Self::Error> {
        let started = Instant::now();
        self.acceptance_sum += metropolis_sweep_with_scratch(
            &mut self.lattice,
            &self.params,
            &mut self.theta_scratch,
            self.proposal_width,
            &mut self.rng,
        )?;
        self.acceptance_count += 1;
        for _ in 0..self.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                &mut self.lattice,
                &self.params,
                &mut self.wolff_scratch,
                Some(&mut self.theta_scratch),
                &mut self.rng,
            )?;
        }
        self.last_sweep_seconds = started.elapsed().as_secs_f64();
        Ok(())
    }

    fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error> {
        let measure_started = Instant::now();
        let obs =
            measure_theta_observables_with_scratch(&self.lattice, &self.params, &self.theta_scratch);
        measure_theta_helicity(context, &obs)?;
        context.measure("Energy", obs.energy)?;
        context.measure("EnergySquared", obs.energy.powi(2))?;
        context.measure("MagnetizationSquared", obs.magnetization_squared)?;
        if (self.corr_rmax_xy > 0 || self.corr_rmax_z > 0)
            && self.measurement_count.is_multiple_of(self.correlation_interval)
        {
            let corr = measure_theta_correlations_with_scratch(
                &self.lattice,
                &self.theta_scratch,
                None,
                Some(self.corr_rmax_xy),
                Some(self.corr_rmax_z),
            );
            for (r, value) in corr.r_xy.iter().zip(corr.corr_x) {
                context.measure(format!("CorrX_r{r}"), value)?;
            }
            for (r, value) in corr.r_xy.iter().zip(corr.corr_y) {
                context.measure(format!("CorrY_r{r}"), value)?;
            }
            for (r, value) in corr.r_xy.iter().zip(&corr.corr_xy) {
                context.measure(format!("CorrXY_r{r}"), *value)?;
            }
            for (z, layer_corr) in corr.corr_xy_by_z.iter().enumerate() {
                for (r, value) in corr.r_xy.iter().zip(layer_corr) {
                    context.measure(format!("CorrXY_z{z}_r{r}"), *value)?;
                }
            }
            for (r, value) in corr.r_z.iter().zip(corr.corr_z) {
                context.measure(format!("CorrZ_r{r}"), value)?;
            }
        }
        context.measure("_ll_sweep_time", self.last_sweep_seconds)?;
        context.measure("_ll_measure_time", measure_started.elapsed().as_secs_f64())?;
        self.measurement_count += 1;
        Ok(())
    }

    fn finalize_estimates(
        &self,
        parameters: &ThetaTask,
        raw_bins: &BTreeMap<String, Vec<f64>>,
        bin_lengths: &BTreeMap<String, usize>,
    ) -> Result<BTreeMap<String, carlo_mc::Estimate>, GenericJobError> {
        let volume = self.lattice.volume() as f64;
        let beta = 1.0 / parameters.temperature;
        finalize_theta_estimates(raw_bins, bin_lengths, volume, beta).map_err(|error| {
            GenericJobError::Model {
                task: parameters.name.clone(),
                source: Box::new(ThetaModelError(error)),
            }
        })
    }

    fn task_metadata(&self) -> BTreeMap<String, f64> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "acceptance".to_string(),
            self.acceptance_sum / self.acceptance_count.max(1) as f64,
        );
        metadata
    }

    fn build_tasks(config: &Params) -> Result<Vec<Task<ThetaTask>>, Self::Error> {
        let value = toml::Value::Table(config.clone().into_table());
        let spec: ThetaJobToml = value
            .try_into()
            .map_err(|error: toml::de::Error| ThetaModelError(error.to_string()))?;
        let (cfg, _options) = ThetaJobConfig::try_from_toml_spec(spec)?;
        theta_tasks_from_config(&cfg).map_err(ThetaModelError)
    }
}

#[cfg(all(test, feature = "carlo-mc"))]
mod theta_carlo_model_tests {
    use super::*;

    const TINY_CONFIG: &str = r#"
name = "tiny"
[model]
l = [2]
temperatures = [2.0]
samples = 1
j_xy = 1.0
j_z_mean = 0.1
delta_j_xy = [0.0]
delta_j_z = [0.0]
[run]
sweeps = 4
thermalization = 2
binsize = 2
wolff_steps = 1
"#;

    #[test]
    fn theta_model_runs_through_carlo_runner() {
        let params = Params::parse(TINY_CONFIG).unwrap();
        let tasks = ThetaModel::build_tasks(&params).unwrap();
        assert_eq!(tasks.len(), 1);

        let job = Job::<ThetaModel>::new("tiny", tasks);
        let run = Runner::new().run(&job, &RunOptions::default()).unwrap();
        assert!(!run.stopped_early);
        assert_eq!(run.result.tasks.len(), 1);

        let result = &run.result.tasks[0];
        assert!(result.completed);
        assert!(result.observables.contains_key("Energy"));
        assert!(result.observables.contains_key("MagnetizationSquared"));
    }
}

impl Default for ThetaJobConfig {
    fn default() -> Self {
        Self {
            l: (4..=20).collect(),
            l_x: None,
            l_y: None,
            l_z: None,
            temperatures: vec![2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
            delta_j_z: vec![0.0],
            delta_j_xy: vec![0.0],
            samples: 16,
            base_seed: 20260414,
            j_xy: 1.0,
            j_z_mean: 0.1,
            sweeps: 20_000,
            thermalization: 5_000,
            binsize: 50,
            proposal_width: std::f64::consts::PI,
            wolff_steps: 1,
            correlation_rmax: None,
            correlation_rmax_xy: None,
            correlation_rmax_z: None,
            correlation_interval: 1,
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
        Self::try_from_toml_spec(spec)
    }

    pub fn from_toml_spec(spec: ThetaJobToml) -> (Self, ThetaRunOptions) {
        Self::try_from_toml_spec(spec)
            .expect("ThetaJobToml contains an invalid duration")
    }

    pub fn try_from_toml_spec(
        spec: ThetaJobToml,
    ) -> Result<(Self, ThetaRunOptions), String> {
        let mut cfg = Self::default();
        if let Some(name) = spec.name {
            cfg.job_name = name;
        }
        if let Some(run_time) = spec.run_time {
            cfg.run_time = parse_duration(&run_time)
                .map_err(|err| format!("invalid run_time: {err}"))?;
        }
        if let Some(checkpoint_time) = spec.checkpoint_time {
            cfg.checkpoint_time = parse_duration(&checkpoint_time)
                .map_err(|err| format!("invalid checkpoint_time: {err}"))?;
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
            if let Some(value) = model.delta_j_xy.or(model.djxy) {
                cfg.delta_j_xy = value;
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
            if let Some(value) = model.j_z_mean.or(model.j_z) {
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
            cfg.correlation_rmax_xy = measure
                .corr_rmax_xy
                .or_else(|| cfg.correlation_rmax.clone());
            cfg.correlation_rmax_z = measure
                .corr_rmax_z
                .or_else(|| cfg.correlation_rmax.clone());
            if let Some(value) = measure.correlation_interval.or(measure.corr_interval) {
                cfg.correlation_interval = value;
            }
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
        Ok((cfg, options))
    }

    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();
        cfg.l = parse_env_list("XY_L", &cfg.l)?;
        cfg.l_x = parse_optional_env_list("XY_LX")?;
        cfg.l_y = parse_optional_env_list("XY_LY")?;
        cfg.l_z = parse_optional_env_list("XY_LZ")?;
        cfg.temperatures = parse_env_list("XY_T", &cfg.temperatures)?;
        cfg.delta_j_z = parse_env_list("XY_DJZ", &cfg.delta_j_z)?;
        cfg.delta_j_xy = parse_env_list("XY_DJXY", &cfg.delta_j_xy)?;
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
        cfg.correlation_rmax = parse_optional_env_list("XY_CORR_RMAX")?;
        cfg.correlation_rmax_xy =
            parse_optional_env_list("XY_CORR_RMAX_XY")?.or_else(|| cfg.correlation_rmax.clone());
        cfg.correlation_rmax_z =
            parse_optional_env_list("XY_CORR_RMAX_Z")?.or_else(|| cfg.correlation_rmax.clone());
        cfg.correlation_interval = parse_env_value("XY_CORR_INTERVAL", cfg.correlation_interval)?;
        cfg.job_name = std::env::var("XY_JOB_NAME").unwrap_or_else(|_| {
            format!(
                "xy_carlo_L{}_dJxy{}_dJz{}",
                join_display(&cfg.l),
                join_display(&cfg.delta_j_xy),
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
        let tasks = self.make_theta_tasks()?;
        Ok(ThetaJob {
            name: self.job_name.clone(),
            tasks,
        })
    }

    pub fn make_theta_tasks(&self) -> Result<Vec<ThetaTask>, String> {
        let lattice_specs = self.lattice_specs();
        if lattice_specs.is_empty() {
            return Err("at least one lattice size must be configured".to_string());
        }
        validate_correlation_rmax_count(
            "corr_rmax",
            self.correlation_rmax.as_deref(),
            lattice_specs.len(),
        )?;
        validate_correlation_rmax_count(
            "corr_rmax_xy",
            self.correlation_rmax_xy.as_deref(),
            lattice_specs.len(),
        )?;
        validate_correlation_rmax_count(
            "corr_rmax_z",
            self.correlation_rmax_z.as_deref(),
            lattice_specs.len(),
        )?;
        if self.temperatures.is_empty() {
            return Err("at least one temperature must be configured".to_string());
        }
        if self.delta_j_xy.is_empty() || self.delta_j_z.is_empty() {
            return Err("delta_j_xy and delta_j_z must not be empty".to_string());
        }
        if self.samples == 0 {
            return Err("samples must be positive".to_string());
        }
        if self.binsize == 0 {
            return Err("binsize must be positive".to_string());
        }
        if self.correlation_interval == 0 {
            return Err("correlation_interval must be positive".to_string());
        }
        if self.sweeps < self.binsize {
            return Err("sweeps must be at least binsize".to_string());
        }
        let mut tasks = Vec::new();
        for (lattice_index, &(l_x, l_y, l_z, l)) in lattice_specs.iter().enumerate() {
            for &delta_j_xy in &self.delta_j_xy {
                for &delta_j_z in &self.delta_j_z {
                    for &temperature in &self.temperatures {
                        for sample in 1..=self.samples {
                            let disorder_seed = self.base_seed + sample as u64 - 1;
                            let seed = self.base_seed
                                + 100_000 * l_z as u64
                                + 1_000 * sample as u64
                                + (100.0 * temperature).round() as u64
                                + (1_000.0 * delta_j_xy).round() as u64 * 10_000
                                + (1_000.0 * delta_j_z).round() as u64;
                            let mut disorder_rng = FastRng::seed_from_u64(disorder_seed);
                            let j_xy_array = generate_layer_disorder_values(
                                l_z,
                                self.j_xy,
                                delta_j_xy,
                                &mut disorder_rng,
                                "J_xy",
                            )?;
                            let j_z_array = generate_layer_disorder_values(
                                l_z,
                                self.j_z_mean,
                                delta_j_z,
                                &mut disorder_rng,
                                "J_z",
                            )?;
                            tasks.push(ThetaTask {
                                name: format!(
                                    "L{}x{}x{}_T{:.6}_dJxy{:.6}_dJz{:.6}_sample{}",
                                    l_x, l_y, l_z, temperature, delta_j_xy, delta_j_z, sample
                                ),
                                l,
                                l_x,
                                l_y,
                                l_z,
                                temperature,
                                j_xy: self.j_xy,
                                delta_j_xy,
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
                                correlation_rmax: self
                                    .correlation_rmax
                                    .as_ref()
                                    .map(|values| values[lattice_index])
                                    .unwrap_or(l_z / 2),
                                correlation_rmax_xy: self
                                    .correlation_rmax_xy
                                    .as_ref()
                                    .map(|values| values[lattice_index])
                                    .unwrap_or_else(|| l_x.min(l_y) / 2),
                                correlation_rmax_z: self
                                    .correlation_rmax_z
                                    .as_ref()
                                    .map(|values| values[lattice_index])
                                    .unwrap_or(l_z / 2),
                                correlation_interval: self.correlation_interval,
                                j_xy_array: Some(j_xy_array),
                                j_z_array: Some(j_z_array),
                            });
                        }
                    }
                }
            }
        }
        Ok(tasks)
    }

    pub(crate) fn lattice_specs(&self) -> Vec<(usize, usize, usize, usize)> {
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

fn validate_correlation_rmax_count(
    name: &str,
    values: Option<&[usize]>,
    lattice_count: usize,
) -> Result<(), String> {
    if let Some(values) = values {
        if values.len() != lattice_count {
            return Err(format!(
                "{name} must contain exactly one value per lattice size: expected {lattice_count}, got {}",
                values.len()
            ));
        }
    }
    Ok(())
}

const HELICITY_COS_X: &str = "_helicity_cos_x";
const HELICITY_COS_Y: &str = "_helicity_cos_y";
const HELICITY_COS_Z: &str = "_helicity_cos_z";
const HELICITY_SIN_X: &str = "_helicity_sin_x";
const HELICITY_SIN_Y: &str = "_helicity_sin_y";
const HELICITY_SIN_Z: &str = "_helicity_sin_z";
const HELICITY_SIN2_X: &str = "_helicity_sin2_x";
const HELICITY_SIN2_Y: &str = "_helicity_sin2_y";
const HELICITY_SIN2_Z: &str = "_helicity_sin2_z";

fn register_theta_evaluables(
    evaluator: &mut Evaluator<'_, carlo_mc::Estimate>,
    volume: f64,
    beta: f64,
) {
    evaluator.evaluate("Magnetization", ["MagnetizationSquared"], |[mag2]| mag2.sqrt());
    evaluator.evaluate("Chi", ["MagnetizationSquared"], |[mag2]| beta * volume * mag2);
    evaluator.evaluate("SpecificHeat", ["EnergySquared", "Energy"], |[energy2, energy]| {
        beta * beta * volume * (energy2 - energy * energy)
    });
    evaluator.evaluate(
        "RhoXY",
        [
            HELICITY_COS_X,
            HELICITY_SIN_X,
            HELICITY_SIN2_X,
            HELICITY_COS_Y,
            HELICITY_SIN_Y,
            HELICITY_SIN2_Y,
        ],
        |[cos_x, sin_x, sin2_x, cos_y, sin_y, sin2_y]| {
            let rho_x = cos_x / volume - beta * (sin2_x - sin_x.powi(2)) / volume;
            let rho_y = cos_y / volume - beta * (sin2_y - sin_y.powi(2)) / volume;
            (rho_x + rho_y) / 2.0
        },
    );
    evaluator.evaluate(
        "RhoZ",
        [HELICITY_COS_Z, HELICITY_SIN_Z, HELICITY_SIN2_Z],
        |[cos_z, sin_z, sin2_z]| cos_z / volume - beta * (sin2_z - sin_z.powi(2)) / volume,
    );
}

pub(crate) fn measure_theta_helicity(
    context: &mut Context,
    obs: &ThetaObservables,
) -> Result<(), GenericJobError> {
    context.measure(HELICITY_COS_X, obs.cos_x)?;
    context.measure(HELICITY_COS_Y, obs.cos_y)?;
    context.measure(HELICITY_COS_Z, obs.cos_z)?;
    context.measure(HELICITY_SIN_X, obs.sin_x)?;
    context.measure(HELICITY_SIN_Y, obs.sin_y)?;
    context.measure(HELICITY_SIN_Z, obs.sin_z)?;
    context.measure(HELICITY_SIN2_X, obs.sin_x.powi(2))?;
    context.measure(HELICITY_SIN2_Y, obs.sin_y.powi(2))?;
    context.measure(HELICITY_SIN2_Z, obs.sin_z.powi(2))?;
    Ok(())
}

pub(crate) fn finalize_theta_estimates(
    raw_bins: &BTreeMap<String, Vec<f64>>,
    bin_lengths: &BTreeMap<String, usize>,
    volume: f64,
    beta: f64,
) -> Result<BTreeMap<String, carlo_mc::Estimate>, String> {
    let binned = raw_bins
        .iter()
        .map(|(name, bins)| {
            let bin_length = *bin_lengths.get(name).unwrap_or(&1);
            Ok((
                name.clone(),
                BinnedEstimate::from_internal_bins(bins.clone(), bin_length)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut estimates = binned
        .iter()
        .map(|(name, estimate)| {
            (
                name.clone(),
                carlo_mc::Estimate::from_binned(estimate, estimate.internal_bin_length),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut evaluator = Evaluator::<carlo_mc::Estimate>::new(binned, &mut estimates);
    register_theta_evaluables(&mut evaluator, volume, beta);
    Ok(estimates)
}

pub fn generate_layer_disorder_values<R: Rng + ?Sized>(
    l_z: usize,
    mean: f64,
    delta: f64,
    rng: &mut R,
    coupling_name: &str,
) -> Result<Vec<f64>, String> {
    let mut values = vec![0.0; l_z];
    initialize_two_point_layer_disorder(&mut values, mean, delta, rng, coupling_name)?;
    Ok(values)
}

pub(crate) fn rank_result_path_with_options(
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

pub(crate) fn merged_result_path_with_options(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.results.json", file_stem(job_name)))
}

pub(crate) fn default_measurement_dir_with_options(
    job_name: &str,
    options: &ThetaRunOptions,
) -> PathBuf {
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

pub(crate) fn mpi_env_rank() -> Option<usize> {
    env_usize_any(&[
        "XY_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    ])
}

pub(crate) fn mpi_env_world_size() -> Option<usize> {
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
