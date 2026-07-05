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
        binsize: usize,
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
            .filter(|(name, _)| !name.starts_with("_helicity_"))
            .map(|(name, estimate)| (name.clone(), estimate.internal_bins.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut estimates = binned
            .iter()
            .filter(|(name, _)| !name.starts_with("_helicity_"))
            .map(|(name, estimate)| {
                (
                    name.clone(),
                    ObservableEstimate::new(estimate, estimate.internal_bin_length),
                )
            })
            .collect::<BTreeMap<_, _>>();
        if let Some(rho_xy) = helicity_estimate(
            &binned,
            [(HELICITY_COS_X, HELICITY_SIN_X, HELICITY_SIN2_X),
             (HELICITY_COS_Y, HELICITY_SIN_Y, HELICITY_SIN2_Y)],
            binsize,
            volume,
            beta,
        )? {
            measurement_bins.insert("RhoXY".to_string(), rho_xy.internal_bins.clone());
            estimates.insert("RhoXY".to_string(), ObservableEstimate::new(&rho_xy, binsize));
        }
        if let Some(rho_z) = helicity_estimate(
            &binned,
            [(HELICITY_COS_Z, HELICITY_SIN_Z, HELICITY_SIN2_Z)],
            binsize,
            volume,
            beta,
        )? {
            measurement_bins.insert("RhoZ".to_string(), rho_z.internal_bins.clone());
            estimates.insert("RhoZ".to_string(), ObservableEstimate::new(&rho_z, binsize));
        }
        if let (Some(rho_xy), Some(rho_z)) = (
            estimates.get("RhoXY").and_then(|_| measurement_bins.get("RhoXY")),
            estimates.get("RhoZ").and_then(|_| measurement_bins.get("RhoZ")),
        ) {
            let rho_xy = BinnedEstimate::from_internal_bins(rho_xy.clone(), binsize)?;
            let rho_z = BinnedEstimate::from_internal_bins(rho_z.clone(), binsize)?;
            let diff = BinnedEstimate::jackknife_difference(&rho_xy, &rho_z)?;
            measurement_bins.insert("RhoDifference".to_string(), diff.internal_bins.clone());
            estimates.insert(
                "RhoDifference".to_string(),
                ObservableEstimate::new(&diff, diff.internal_bin_length),
            );
        }
        Ok((estimates, measurement_bins))
    }
}

fn helicity_estimate<const N: usize>(
    binned: &BTreeMap<String, BinnedEstimate>,
    components: [(&str, &str, &str); N],
    binsize: usize,
    volume: f64,
    beta: f64,
) -> Result<Option<BinnedEstimate>, String> {
    let mut component_bins = Vec::with_capacity(N);
    for (cos_name, sin_name, sin2_name) in components {
        let (Some(cos), Some(sin), Some(sin2)) = (
            binned.get(cos_name),
            binned.get(sin_name),
            binned.get(sin2_name),
        ) else {
            return Ok(None);
        };
        let bin_count = cos
            .internal_bins
            .len()
            .min(sin.internal_bins.len())
            .min(sin2.internal_bins.len());
        component_bins.push((cos, sin, sin2, bin_count));
    }
    let bin_count = component_bins
        .iter()
        .map(|(_, _, _, bin_count)| *bin_count)
        .min()
        .unwrap_or(0);
    if bin_count == 0 {
        return Ok(None);
    }
    let internal_bins = (0..bin_count)
        .map(|bin_index| {
            component_bins
                .iter()
                .map(|(cos, sin, sin2, _)| {
                    cos.internal_bins[bin_index] / volume
                        - beta
                            * (sin2.internal_bins[bin_index] - sin.internal_bins[bin_index].powi(2))
                            / volume
                })
                .sum::<f64>()
                / N as f64
        })
        .collect::<Vec<_>>();
    BinnedEstimate::from_internal_bins(internal_bins, binsize).map(Some)
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
            .estimates_and_measurement_bins(2, 2.0, 1.0)
            .expect("helicity estimates");

        assert!(!observables.contains_key(HELICITY_COS_X));
        assert!((observables["RhoXY"].mean - 0.5).abs() < 1e-12);
        assert!((observables["RhoZ"].mean - 0.0).abs() < 1e-12);
        assert!((observables["RhoDifference"].mean - 0.5).abs() < 1e-12);
        assert_eq!(measurement_bins["RhoXY"], vec![0.5]);
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
        series.push("Magnetization", obs.magnetization_squared.sqrt());
        series.push("MagnetizationSquared", obs.magnetization_squared);
        series.push("Chi", beta * volume * obs.magnetization_squared);
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

    let (observables, measurement_bins) =
        series.estimates_and_measurement_bins(task.binsize, volume, beta)?;
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
