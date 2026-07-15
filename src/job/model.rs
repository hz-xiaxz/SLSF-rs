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
        Self::from_internal_bins(internal_bins, internal_bin_length)
    }

    pub fn from_internal_bins(
        internal_bins: Vec<f64>,
        internal_bin_length: usize,
    ) -> Result<Self, String> {
        if internal_bin_length == 0 {
            return Err("binsize must be positive".to_string());
        }
        if internal_bins.is_empty() {
            return Err("binsize is larger than the sample series".to_string());
        }
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
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactObservableAccumulator {
    pub internal_bins: Vec<f64>,
    pub pending_sum: f64,
    pub pending_count: usize,
    pub total_count: usize,
    pub internal_bin_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarAccumulator {
    internal_bins: Vec<f64>,
    pending_sum: f64,
    pending_count: usize,
    total_count: usize,
    internal_bin_length: usize,
}

impl ScalarAccumulator {
    pub fn new(internal_bin_length: usize) -> Self {
        Self {
            internal_bins: Vec::new(),
            pending_sum: 0.0,
            pending_count: 0,
            total_count: 0,
            internal_bin_length: internal_bin_length.max(1),
        }
    }

    pub fn from_compact(mut compact: CompactObservableAccumulator, fallback_bin_length: usize) -> Self {
        if compact.internal_bin_length == 0 {
            compact.internal_bin_length = fallback_bin_length.max(1);
        }
        Self {
            internal_bins: compact.internal_bins,
            pending_sum: compact.pending_sum,
            pending_count: compact.pending_count,
            total_count: compact.total_count,
            internal_bin_length: compact.internal_bin_length,
        }
    }

    pub fn from_samples(samples: Vec<f64>, internal_bin_length: usize) -> Self {
        let mut acc = Self::new(internal_bin_length);
        for sample in samples {
            acc.push(sample);
        }
        acc
    }

    pub fn from_internal_bins(internal_bins: Vec<f64>, internal_bin_length: usize) -> Self {
        let internal_bin_length = internal_bin_length.max(1);
        let total_count = internal_bins.len() * internal_bin_length;
        Self {
            internal_bins,
            pending_sum: 0.0,
            pending_count: 0,
            total_count,
            internal_bin_length,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.pending_sum += value;
        self.pending_count += 1;
        self.total_count += 1;
        if self.pending_count == self.internal_bin_length {
            self.internal_bins
                .push(self.pending_sum / self.internal_bin_length as f64);
            self.pending_sum = 0.0;
            self.pending_count = 0;
        }
    }

    pub fn internal_bins(&self) -> &[f64] {
        &self.internal_bins
    }

    pub fn compact(&self) -> CompactObservableAccumulator {
        CompactObservableAccumulator {
            internal_bins: self.internal_bins.clone(),
            pending_sum: self.pending_sum,
            pending_count: self.pending_count,
            total_count: self.total_count,
            internal_bin_length: self.internal_bin_length,
        }
    }

    pub fn estimate(&self) -> Result<BinnedEstimate, String> {
        BinnedEstimate::from_internal_bins(self.internal_bins.clone(), self.internal_bin_length)
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ThetaCheckpointRuntime {
    pub(crate) path: PathBuf,
    pub(crate) interval: Duration,
    pub(crate) resume: bool,
    pub(crate) heartbeat_path: Option<PathBuf>,
    pub(crate) deadline: Option<Instant>,
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
    measurement_accumulators: BTreeMap<String, CompactObservableAccumulator>,
}
