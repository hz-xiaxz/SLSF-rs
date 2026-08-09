const HELICITY_COS_X: &str = "_helicity_cos_x";
const HELICITY_COS_Y: &str = "_helicity_cos_y";
const HELICITY_COS_Z: &str = "_helicity_cos_z";
const HELICITY_SIN_X: &str = "_helicity_sin_x";
const HELICITY_SIN_Y: &str = "_helicity_sin_y";
const HELICITY_SIN_Z: &str = "_helicity_sin_z";
const HELICITY_SIN2_X: &str = "_helicity_sin2_x";
const HELICITY_SIN2_Y: &str = "_helicity_sin2_y";
const HELICITY_SIN2_Z: &str = "_helicity_sin2_z";

#[derive(Debug)]
struct Evaluable {
    estimate: ObservableEstimate,
}

struct Evaluator<'a> {
    binned: BTreeMap<String, BinnedEstimate>,
    estimates: &'a mut BTreeMap<String, ObservableEstimate>,
}

type ObservableEstimates = BTreeMap<String, ObservableEstimate>;
type MeasurementBins = BTreeMap<String, Vec<f64>>;
type ObservableSummary = (ObservableEstimates, MeasurementBins);

#[derive(Debug)]
struct ObservableSeries {
    accumulators: BTreeMap<String, ScalarAccumulator>,
    binsize: usize,
}

impl ObservableSeries {
    fn new(binsize: usize) -> Self {
        Self {
            accumulators: BTreeMap::new(),
            binsize: binsize.max(1),
        }
    }

    fn push(&mut self, name: impl Into<String>, value: f64) {
        self.push_with_binsize(name, value, self.binsize);
    }

    fn push_with_binsize(&mut self, name: impl Into<String>, value: f64, binsize: usize) {
        self.accumulators
            .entry(name.into())
            .or_insert_with(|| ScalarAccumulator::new(binsize.max(1)))
            .push(value);
    }

    fn push_helicity(&mut self, obs: &ThetaObservables) {
        self.push(HELICITY_COS_X, obs.cos_x);
        self.push(HELICITY_COS_Y, obs.cos_y);
        self.push(HELICITY_COS_Z, obs.cos_z);
        self.push(HELICITY_SIN_X, obs.sin_x);
        self.push(HELICITY_SIN_Y, obs.sin_y);
        self.push(HELICITY_SIN_Z, obs.sin_z);
        self.push(HELICITY_SIN2_X, obs.sin_x.powi(2));
        self.push(HELICITY_SIN2_Y, obs.sin_y.powi(2));
        self.push(HELICITY_SIN2_Z, obs.sin_z.powi(2));
    }

    fn from_compact(
        accumulators: BTreeMap<String, CompactObservableAccumulator>,
        binsize: usize,
    ) -> Self {
        Self {
            accumulators: accumulators
                .into_iter()
                .map(|(name, acc)| (name, ScalarAccumulator::from_compact(acc, binsize)))
                .collect(),
            binsize: binsize.max(1),
        }
    }

    fn compact(&self) -> BTreeMap<String, CompactObservableAccumulator> {
        self.accumulators
            .iter()
            .map(|(name, acc)| (name.clone(), acc.compact()))
            .collect()
    }

    fn estimates_and_measurement_bins(
        &self,
        volume: f64,
        beta: f64,
    ) -> Result<ObservableSummary, String> {
        let binned = self
            .accumulators
            .iter()
            .map(|(name, acc)| Ok((name.clone(), acc.estimate()?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let measurement_bins = binned
            .iter()
            .map(|(name, estimate)| (name.clone(), estimate.internal_bins.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut estimates = binned
            .iter()
            .map(|(name, estimate)| {
                (
                    name.clone(),
                    ObservableEstimate::new(estimate, estimate.internal_bin_length),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut evaluator = Evaluator::new(binned, &mut estimates);
        register_theta_evaluables(&mut evaluator, volume, beta)?;
        Ok((estimates, measurement_bins))
    }
}

impl<'a> Evaluator<'a> {
    fn new(
        binned: BTreeMap<String, BinnedEstimate>,
        estimates: &'a mut BTreeMap<String, ObservableEstimate>,
    ) -> Self {
        Self { binned, estimates }
    }

    fn evaluate<const N: usize, F>(
        &mut self,
        name: &str,
        ingredients: [&str; N],
        evaluation: F,
    ) -> Result<(), String>
    where
        F: Fn([f64; N]) -> f64,
    {
        if let Some(evaluable) = evaluate(&self.binned, ingredients, evaluation)? {
            self.estimates.insert(name.to_string(), evaluable.estimate);
        }
        Ok(())
    }
}

fn register_theta_evaluables(
    evaluator: &mut Evaluator<'_>,
    volume: f64,
    beta: f64,
) -> Result<(), String> {
    evaluator.evaluate("Magnetization", ["MagnetizationSquared"], |[mag2]| mag2.sqrt())?;
    evaluator.evaluate("Chi", ["MagnetizationSquared"], |[mag2]| beta * volume * mag2)?;
    evaluator.evaluate("SpecificHeat", ["EnergySquared", "Energy"], |[energy2, energy]| {
        beta * beta * volume * (energy2 - energy * energy)
    })?;
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
    )?;
    evaluator.evaluate(
        "RhoZ",
        [HELICITY_COS_Z, HELICITY_SIN_Z, HELICITY_SIN2_Z],
        |[cos_z, sin_z, sin2_z]| cos_z / volume - beta * (sin2_z - sin_z.powi(2)) / volume,
    )?;
    Ok(())
}

fn evaluate<const N: usize, F>(
    binned: &BTreeMap<String, BinnedEstimate>,
    ingredients: [&str; N],
    evaluation: F,
) -> Result<Option<Evaluable>, String>
where
    F: Fn([f64; N]) -> f64,
{
    let Some(used) = ingredients
        .iter()
        .map(|name| binned.get(*name))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let internal_bin_length = used
        .iter()
        .map(|estimate| estimate.internal_bin_length)
        .min()
        .unwrap_or(1);
    let rebin_length = used
        .iter()
        .map(|estimate| estimate.rebin_length)
        .min()
        .unwrap_or(1);
    let rebin_count = used
        .iter()
        .map(|estimate| estimate.bins.len())
        .min()
        .unwrap_or(0);
    if rebin_count == 0 {
        return Ok(None);
    }
    let rebin_samples = jackknife_evaluate(&used, rebin_count, evaluation);
    let estimate = BinnedEstimate {
        mean: rebin_samples.mean,
        stderr: rebin_samples.stderr,
        bins: rebin_samples.jacked_evals,
        internal_bins: Vec::new(),
        internal_bin_length,
        rebin_length,
    };
    Ok(Some(Evaluable {
        estimate: ObservableEstimate::new(&estimate, internal_bin_length),
    }))
}

struct JackknifeEstimate {
    mean: f64,
    stderr: f64,
    jacked_evals: Vec<f64>,
}

fn jackknife_evaluate<const N: usize, F>(
    sample_set: &[&BinnedEstimate],
    sample_count: usize,
    evaluation: F,
) -> JackknifeEstimate
where
    F: Fn([f64; N]) -> f64,
{
    let sums = std::array::from_fn::<_, N, _>(|i| {
        sample_set[i].bins[..sample_count].iter().sum::<f64>()
    });
    let complete_eval = evaluation(std::array::from_fn(|i| sums[i] / sample_count as f64));
    if sample_count <= 1 {
        return JackknifeEstimate {
            mean: complete_eval,
            stderr: f64::NAN,
            jacked_evals: vec![complete_eval],
        };
    }

    let jacked_evals = (0..sample_count)
        .map(|sample_index| {
            evaluation(std::array::from_fn(|i| {
                (sums[i] - sample_set[i].bins[sample_index]) / (sample_count - 1) as f64
            }))
        })
        .collect::<Vec<_>>();
    let jacked_mean = mean(&jacked_evals);
    let bias_corrected_mean =
        sample_count as f64 * complete_eval - (sample_count - 1) as f64 * jacked_mean;
    let error = jacked_evals
        .iter()
        .map(|value| (value - jacked_mean).powi(2))
        .sum::<f64>();
    let stderr = (((sample_count - 1) as f64) * error / sample_count as f64).sqrt();
    JackknifeEstimate {
        mean: bias_corrected_mean,
        stderr,
        jacked_evals,
    }
}

#[cfg(test)]
mod job_simulation_tests {
    use super::*;

    #[test]
    fn helicity_modulus_uses_binned_current_variance() {
        let mut series = ObservableSeries::new(2);
        for sin in [1.0, 3.0] {
            series.push_helicity(&ThetaObservables {
                energy: 0.0,
                magnetization_squared: 0.0,
                cos_x: 2.0,
                cos_y: 2.0,
                cos_z: 1.0,
                sin_x: sin,
                sin_y: sin,
                sin_z: sin + 1.0,
            });
        }

        let (observables, measurement_bins) = series
            .estimates_and_measurement_bins(2.0, 1.0)
            .expect("helicity estimates");

        assert!(observables.contains_key(HELICITY_COS_X));
        assert!(observables.contains_key(HELICITY_SIN2_X));
        assert!(!observables.contains_key("MagnetizationSquared"));
        assert!((observables["RhoXY"].mean - 0.5).abs() < 1e-12);
        assert!((observables["RhoZ"].mean - 0.0).abs() < 1e-12);
        assert_eq!(measurement_bins[HELICITY_COS_X], vec![2.0]);
        assert!(!measurement_bins.contains_key("RhoXY"));
    }

    #[test]
    fn evaluator_builds_derived_observables_from_registered_ingredients() {
        let mut series = ObservableSeries::new(2);
        for (energy, mag2) in [(1.0, 0.25), (3.0, 0.49), (5.0, 0.81), (7.0, 1.0)] {
            series.push("Energy", energy);
            series.push("EnergySquared", energy * energy);
            series.push("MagnetizationSquared", mag2);
        }

        let (observables, measurement_bins) = series
            .estimates_and_measurement_bins(8.0, 2.0)
            .expect("derived observables");

        assert!(observables.contains_key("EnergySquared"));
        assert!(observables.contains_key("MagnetizationSquared"));
        assert_eq!(measurement_bins["Energy"], vec![2.0, 6.0]);
        assert!(!measurement_bins.contains_key("Magnetization"));
        assert!(!measurement_bins.contains_key("Chi"));
        assert!(!measurement_bins.contains_key("SpecificHeat"));
        assert!((observables["Magnetization"].mean - 0.817076375991209).abs() < 1e-12);
        assert!((observables["Chi"].mean - 10.2).abs() < 1e-12);
        assert!((observables["SpecificHeat"].mean - 288.0).abs() < 1e-12);
    }

    #[test]
    fn evaluator_jackknifes_nonlinear_observables_from_rebin_means() {
        let mut series = ObservableSeries::new(1);
        let samples = (0..64)
            .map(|index| if index < 32 { 2.0 } else { -1.0 })
            .collect::<Vec<_>>();
        for sin in samples {
            series.push_helicity(&ThetaObservables {
                energy: 0.0,
                magnetization_squared: 0.0,
                cos_x: 0.0,
                cos_y: 0.0,
                cos_z: 0.0,
                sin_x: sin,
                sin_y: sin,
                sin_z: 0.0,
            });
        }

        let (observables, _) = series
            .estimates_and_measurement_bins(1.0, 1.0)
            .expect("helicity estimates");

        let sin_bins = BinnedEstimate::from_internal_bins(
            series.accumulators[HELICITY_SIN_X].internal_bins().to_vec(),
            1,
        )
        .unwrap();
        let sin2_bins = BinnedEstimate::from_internal_bins(
            series.accumulators[HELICITY_SIN2_X].internal_bins().to_vec(),
            1,
        )
        .unwrap();
        let sample_count = sin_bins.bins.len().min(sin2_bins.bins.len());
        let sum_sin = sin_bins.bins[..sample_count].iter().sum::<f64>();
        let sum_sin2 = sin2_bins.bins[..sample_count].iter().sum::<f64>();
        let complete_eval = -(sum_sin2 / sample_count as f64
            - (sum_sin / sample_count as f64).powi(2));
        let jacked = (0..sample_count)
            .map(|index| {
                let mean_sin = (sum_sin - sin_bins.bins[index]) / (sample_count - 1) as f64;
                let mean_sin2 = (sum_sin2 - sin2_bins.bins[index]) / (sample_count - 1) as f64;
                -(mean_sin2 - mean_sin.powi(2))
            })
            .collect::<Vec<_>>();
        let jacked_mean = mean(&jacked);
        let expected = sample_count as f64 * complete_eval - (sample_count - 1) as f64 * jacked_mean;
        let wrong_rebin_average = sin_bins
            .bins
            .iter()
            .zip(&sin2_bins.bins)
            .take(sample_count)
            .map(|(sin, sin2)| -(sin2 - sin.powi(2)))
            .sum::<f64>()
            / sample_count as f64;

        assert!((observables["RhoXY"].mean - expected).abs() < 1e-12);
        assert!((observables["RhoXY"].mean - wrong_rebin_average).abs() > 1e-3);
    }
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

pub fn run_theta_task(task: &ThetaTask) -> Result<ThetaTaskResult, String> {
    run_theta_task_with_checkpoint(task, 0, None)
}

fn checkpoint_deadline_reached(checkpoint: Option<&ThetaCheckpointRuntime>) -> bool {
    checkpoint
        .and_then(|checkpoint| checkpoint.deadline)
        .map(|deadline| Instant::now() >= deadline)
        .unwrap_or(false)
}

pub(crate) fn run_theta_task_with_checkpoint(
    task: &ThetaTask,
    task_index: usize,
    checkpoint: Option<&ThetaCheckpointRuntime>,
) -> Result<ThetaTaskResult, String> {
    let params = task.params();
    let mut lattice = ThetaLattice::new(task.l_x, task.l_y, task.l_z)?;
    let mut disorder_rng = FastRng::seed_from_u64(task.disorder_seed);
    match &task.j_xy_array {
        Some(j_xy_array) => {
            if j_xy_array.len() != task.l_z {
                return Err("J_xy_array length must match Lz".to_string());
            }
            lattice.j_xy.clone_from(j_xy_array);
        }
        None => initialize_two_point_layer_disorder(
            &mut lattice.j_xy,
            task.j_xy,
            task.delta_j_xy,
            &mut disorder_rng,
            "J_xy",
        )?,
    }
    match &task.j_z_array {
        Some(j_z_array) => {
            if j_z_array.len() != task.l_z {
                return Err("J_z_array length must match Lz".to_string());
            }
            lattice.j_z.clone_from(j_z_array);
        }
        None => initialize_two_point_layer_disorder(
            &mut lattice.j_z,
            task.j_z_mean,
            task.delta_j_z,
            &mut disorder_rng,
            "J_z",
        )?,
    }
    if lattice.j_xy.iter().any(|&j| j < 0.0) || lattice.j_z.iter().any(|&j| j < 0.0) {
        return Err("theta simulation requires nonnegative layer couplings".to_string());
    }

    let mut rng = FastRng::seed_from_u64(task.seed);
    let mut thermalization_start = 0usize;
    let mut measurement_start = 0usize;
    let mut acceptance_sum = 0.0;
    let mut acceptance_count = 0usize;
    let mut series = ObservableSeries::new(task.binsize);

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
        rng.set_position(state.rng_word_pos);
        thermalization_start = state.thermalization_sweeps.min(task.thermalization);
        measurement_start = state.measurement_sweeps.min(task.sweeps);
        acceptance_sum = state.acceptance_sum;
        acceptance_count = state.acceptance_count;
        series = ObservableSeries::from_compact(state.measurement_accumulators, task.binsize);
    } else {
        initialize_angles(&mut lattice, InitMode::Random, &mut rng)?;
    }

    let mut theta_scratch = ThetaScratch::new(&lattice);
    let mut wolff_scratch = WolffScratch::new(&lattice);
    let mut last_checkpoint = Instant::now();
    #[cfg(feature = "profile-stats")]
    crate::updates::reset_update_profile_stats();

    let mut completed_thermalization_sweeps = thermalization_start;
    let mut completed_measurement_sweeps = measurement_start;

    for thermalization_sweeps in thermalization_start..task.thermalization {
        if checkpoint_deadline_reached(checkpoint) {
            break;
        }
        #[cfg(feature = "profile-stats")]
        let thermal_metropolis_started = Instant::now();
        acceptance_sum += metropolis_sweep_with_scratch(
            &mut lattice,
            &params,
            &mut theta_scratch,
            task.proposal_width,
            &mut rng,
        )?;
        acceptance_count += 1;
        #[cfg(feature = "profile-stats")]
        let thermal_metropolis_seconds = thermal_metropolis_started.elapsed().as_secs_f64();
        #[cfg(feature = "profile-stats")]
        let thermal_wolff_started = Instant::now();
        for _ in 0..task.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                &mut lattice,
                &params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                &mut rng,
            )?;
        }
        #[cfg(feature = "profile-stats")]
        crate::updates::record_profile_phase(
            thermal_metropolis_seconds,
            thermal_wolff_started.elapsed().as_secs_f64(),
            0.0,
        );
        completed_thermalization_sweeps = thermalization_sweeps + 1;
        maybe_write_theta_checkpoint(
            checkpoint,
            &mut last_checkpoint,
            task,
            task_index,
            &lattice,
            &rng,
            completed_thermalization_sweeps,
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
    let correlation_interval = task.correlation_interval.max(1);

    for measurement_sweeps in measurement_start..task.sweeps {
        if checkpoint_deadline_reached(checkpoint) {
            break;
        }
        let sweep_started = Instant::now();
        acceptance_sum += metropolis_sweep_with_scratch(
            &mut lattice,
            &params,
            &mut theta_scratch,
            task.proposal_width,
            &mut rng,
        )?;
        acceptance_count += 1;
        #[cfg(feature = "profile-stats")]
        let metropolis_seconds = sweep_started.elapsed().as_secs_f64();
        #[cfg(feature = "profile-stats")]
        let wolff_started = Instant::now();
        for _ in 0..task.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                &mut lattice,
                &params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                &mut rng,
            )?;
        }
        #[cfg(feature = "profile-stats")]
        let wolff_seconds = wolff_started.elapsed().as_secs_f64();
        let sweep_seconds = sweep_started.elapsed().as_secs_f64();

        let measure_started = Instant::now();
        let obs = measure_theta_observables_with_scratch(&lattice, &params, &theta_scratch);
        series.push_helicity(&obs);
        series.push("Energy", obs.energy);
        series.push("EnergySquared", obs.energy.powi(2));
        series.push("MagnetizationSquared", obs.magnetization_squared);
        if (corr_rmax_xy > 0 || corr_rmax_z > 0) && measurement_sweeps % correlation_interval == 0 {
            let corr = measure_theta_correlations_with_scratch(
                &lattice,
                &theta_scratch,
                None,
                Some(corr_rmax_xy),
                Some(corr_rmax_z),
            );
            for (r, value) in corr.r_xy.iter().zip(corr.corr_x) {
                series.push(format!("CorrX_r{r}"), value);
            }
            for (r, value) in corr.r_xy.iter().zip(corr.corr_y) {
                series.push(format!("CorrY_r{r}"), value);
            }
            for (r, value) in corr.r_xy.iter().zip(&corr.corr_xy) {
                series.push(format!("CorrXY_r{r}"), *value);
            }
            for (z, layer_corr) in corr.corr_xy_by_z.iter().enumerate() {
                for (r, value) in corr.r_xy.iter().zip(layer_corr) {
                    series.push(format!("CorrXY_z{z}_r{r}"), *value);
                }
            }
            for (r, value) in corr.r_z.iter().zip(corr.corr_z) {
                series.push(format!("CorrZ_r{r}"), value);
            }
        }
        let measure_seconds = measure_started.elapsed().as_secs_f64();
        #[cfg(feature = "profile-stats")]
        crate::updates::record_profile_phase(metropolis_seconds, wolff_seconds, measure_seconds);
        series.push("_ll_sweep_time", sweep_seconds);
        series.push("_ll_measure_time", measure_seconds);
        completed_measurement_sweeps = measurement_sweeps + 1;
        maybe_write_theta_checkpoint(
            checkpoint,
            &mut last_checkpoint,
            task,
            task_index,
            &lattice,
            &rng,
            completed_thermalization_sweeps,
            completed_measurement_sweeps,
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
            rng_word_pos: rng.position(),
            thermalization_sweeps: completed_thermalization_sweeps,
            measurement_sweeps: completed_measurement_sweeps,
            acceptance_sum,
            acceptance_count,
            measurement_accumulators: series.compact(),
        };
        write_theta_checkpoint_state_to_path(&state, &checkpoint.path)?;
    }

    let completed = completed_thermalization_sweeps >= task.thermalization
        && completed_measurement_sweeps >= task.sweeps;
    if !completed {
        return Ok(ThetaTaskResult {
            task: task.clone(),
            task_index,
            observables: BTreeMap::new(),
            acceptance: acceptance_sum / acceptance_count.max(1) as f64,
            measurements: completed_measurement_sweeps,
            measurement_bins: BTreeMap::new(),
            measurement_samples: BTreeMap::new(),
            final_theta: lattice.theta.clone(),
            final_j_z: lattice.j_z.clone(),
            rng_word_pos: rng.position(),
            thermalization_sweeps: completed_thermalization_sweeps,
            measurement_sweeps: completed_measurement_sweeps,
            acceptance_sum,
            acceptance_count,
        });
    }

    let (observables, measurement_bins) = series.estimates_and_measurement_bins(volume, beta)?;

    #[cfg(feature = "profile-stats")]
    {
        let stats = crate::updates::take_update_profile_stats();
        eprintln!(
            "profile-stats task={} metropolis_s={:.6} wolff_s={:.6} measurement_s={:.6} wolff_clusters={} wolff_sites={} examined_edges={} zero_probability_edges={} scalar_uphill={:?} x4_uphill={:?} x8_uphill={:?}",
            task.name,
            stats.metropolis_seconds,
            stats.wolff_seconds,
            stats.measurement_seconds,
            stats.wolff_clusters,
            stats.wolff_cluster_sites,
            stats.wolff_examined_edges,
            stats.wolff_zero_probability_edges,
            stats.metropolis_scalar_uphill,
            stats.metropolis_x4_uphill,
            stats.metropolis_x8_uphill,
        );
    }

    Ok(ThetaTaskResult {
        task: task.clone(),
        task_index,
        observables,
        acceptance: acceptance_sum / acceptance_count.max(1) as f64,
        measurements: completed_measurement_sweeps,
        measurement_bins,
        measurement_samples: BTreeMap::new(),
        final_theta: lattice.theta.clone(),
        final_j_z: lattice.j_z.clone(),
        rng_word_pos: rng.position(),
        thermalization_sweeps: task.thermalization,
        measurement_sweeps: task.sweeps,
        acceptance_sum,
        acceptance_count,
    })
}
