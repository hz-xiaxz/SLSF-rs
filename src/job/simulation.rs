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
    measurement_bins: Vec<f64>,
}

struct Evaluator<'a> {
    binned: BTreeMap<String, BinnedEstimate>,
    estimates: &'a mut BTreeMap<String, ObservableEstimate>,
    measurement_bins: &'a mut BTreeMap<String, Vec<f64>>,
}

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
    ) -> Result<
        (
            BTreeMap<String, ObservableEstimate>,
            BTreeMap<String, Vec<f64>>,
        ),
        String,
    > {
        let binned = self
            .accumulators
            .iter()
            .map(|(name, acc)| Ok((name.clone(), acc.estimate()?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let mut measurement_bins = binned
            .iter()
            .filter(|(name, _)| !is_evaluator_ingredient(name))
            .map(|(name, estimate)| (name.clone(), estimate.internal_bins.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut estimates = binned
            .iter()
            .filter(|(name, _)| !is_evaluator_ingredient(name))
            .map(|(name, estimate)| {
                (
                    name.clone(),
                    ObservableEstimate::new(estimate, estimate.internal_bin_length),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut evaluator = Evaluator::new(binned, &mut estimates, &mut measurement_bins);
        register_theta_evaluables(&mut evaluator, volume, beta)?;
        Ok((estimates, measurement_bins))
    }
}

impl<'a> Evaluator<'a> {
    fn new(
        binned: BTreeMap<String, BinnedEstimate>,
        estimates: &'a mut BTreeMap<String, ObservableEstimate>,
        measurement_bins: &'a mut BTreeMap<String, Vec<f64>>,
    ) -> Self {
        Self {
            binned,
            estimates,
            measurement_bins,
        }
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
            let internal_bin_len = evaluable.estimate.internal_bin_len;
            self.measurement_bins
                .insert(name.to_string(), evaluable.measurement_bins.clone());
            self.estimates.insert(name.to_string(), evaluable.estimate);
            self.binned.insert(
                name.to_string(),
                ScalarAccumulator::from_internal_bins(evaluable.measurement_bins, internal_bin_len)
                    .estimate()?,
            );
        }
        Ok(())
    }
}

fn is_evaluator_ingredient(name: &str) -> bool {
    name.starts_with("_helicity_")
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
    evaluator.evaluate("RhoDifference", ["RhoXY", "RhoZ"], |[rho_xy, rho_z]| {
        rho_xy - rho_z
    })?;
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
    let internal_bin_count = used
        .iter()
        .map(|estimate| estimate.internal_bins.len())
        .min()
        .unwrap_or(0);
    if internal_bin_count == 0 {
        return Ok(None);
    }
    let measurement_bins = (0..internal_bin_count)
        .map(|index| evaluation(std::array::from_fn(|i| used[i].internal_bins[index])))
        .collect::<Vec<_>>();
    let rebin_count = used
        .iter()
        .map(|estimate| estimate.bins.len())
        .min()
        .unwrap_or(0);
    if rebin_count == 0 {
        return Ok(None);
    }
    let rebin_samples = (0..rebin_count)
        .map(|index| evaluation(std::array::from_fn(|i| used[i].bins[index])))
        .collect::<Vec<_>>();
    let (mean, stderr) = jackknife_evaluate(&rebin_samples);
    let estimate = BinnedEstimate {
        mean,
        stderr,
        bins: rebin_samples,
        internal_bins: measurement_bins.clone(),
        internal_bin_length,
        rebin_length,
    };
    Ok(Some(Evaluable {
        estimate: ObservableEstimate::new(&estimate, internal_bin_length),
        measurement_bins,
    }))
}

fn jackknife_evaluate(rebin_samples: &[f64]) -> (f64, f64) {
    let sample_count = rebin_samples.len();
    let complete_mean = mean(rebin_samples);
    if sample_count <= 1 {
        return (complete_mean, f64::NAN);
    }
    let sum = rebin_samples.iter().sum::<f64>();
    let jacked = rebin_samples
        .iter()
        .map(|sample| (sum - sample) / (sample_count - 1) as f64)
        .collect::<Vec<_>>();
    let jacked_mean = mean(&jacked);
    let bias_corrected_mean = sample_count as f64 * complete_mean - (sample_count - 1) as f64 * jacked_mean;
    let error = jacked
        .iter()
        .map(|value| (value - jacked_mean).powi(2))
        .sum::<f64>();
    let error = (((sample_count - 1) as f64) * error / sample_count as f64).sqrt();
    (bias_corrected_mean, error)
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

        assert!(!observables.contains_key(HELICITY_COS_X));
        assert!(!observables.contains_key(HELICITY_SIN2_X));
        assert!(!observables.contains_key("MagnetizationSquared"));
        assert!((observables["RhoXY"].mean - 0.5).abs() < 1e-12);
        assert!((observables["RhoZ"].mean - 0.0).abs() < 1e-12);
        assert!((observables["RhoDifference"].mean - 0.5).abs() < 1e-12);
        assert_eq!(measurement_bins["RhoXY"], vec![0.5]);
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
        assert_eq!(measurement_bins["Magnetization"], vec![0.37_f64.sqrt(), 0.905_f64.sqrt()]);
        assert_eq!(measurement_bins["Chi"], vec![5.92, 14.48]);
        assert_eq!(measurement_bins["SpecificHeat"], vec![32.0, 32.0]);
        assert!((observables["Magnetization"].mean - ((0.37_f64.sqrt() + 0.905_f64.sqrt()) / 2.0)).abs() < 1e-12);
        assert!((observables["Chi"].mean - 10.2).abs() < 1e-12);
        assert!((observables["SpecificHeat"].mean - 32.0).abs() < 1e-12);
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

    let mut completed_thermalization_sweeps = thermalization_start;
    let mut completed_measurement_sweeps = measurement_start;

    for thermalization_sweeps in thermalization_start..task.thermalization {
        if checkpoint_deadline_reached(checkpoint) {
            break;
        }
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

    let (observables, measurement_bins) = series.estimates_and_measurement_bins(volume, beta)?;

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
