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
        self.accumulators
            .entry(name.into())
            .or_insert_with(|| ScalarAccumulator::new(self.binsize))
            .push(value);
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

pub(crate) fn run_theta_task_with_checkpoint(
    task: &ThetaTask,
    task_index: usize,
    checkpoint: Option<&ThetaCheckpointRuntime>,
) -> Result<ThetaTaskResult, String> {
    let params = task.params();
    let mut lattice = ThetaLattice::new(task.l_x, task.l_y, task.l_z)?;
    let mut rng = ChaCha8Rng::seed_from_u64(task.seed);
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
            &mut rng,
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
            &mut rng,
            "J_z",
        )?,
    }
    if lattice.j_xy.iter().any(|&j| j < 0.0) || lattice.j_z.iter().any(|&j| j < 0.0) {
        return Err("theta simulation requires nonnegative layer couplings".to_string());
    }
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
        rng.set_word_pos(state.rng_word_pos);
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
            measurement_accumulators: series.compact(),
        };
        write_theta_checkpoint_state_to_path(&state, &checkpoint.path)?;
    }

    let (observables, measurement_bins) = series.estimates_and_measurement_bins(task.binsize)?;
    Ok(ThetaTaskResult {
        task: task.clone(),
        task_index,
        observables,
        acceptance: acceptance_sum / acceptance_count.max(1) as f64,
        measurements: task.sweeps,
        measurement_bins,
        measurement_samples: BTreeMap::new(),
        final_theta: lattice.theta.clone(),
        final_j_z: lattice.j_z.clone(),
        rng_word_pos: rng.get_word_pos(),
        thermalization_sweeps: task.thermalization,
        measurement_sweeps: task.sweeps,
        acceptance_sum,
        acceptance_count,
    })
}
