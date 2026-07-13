fn theta_task_checkpoint_path(output_dir: impl AsRef<Path>, task_index: usize) -> PathBuf {
    output_dir
        .as_ref()
        .join(format!("task{:04}", task_index + 1))
        .join("run0001.dump.h5")
}

fn theta_task_measurement_path_from_checkpoint(path: &Path) -> PathBuf {
    path.with_file_name("run0001.meas.h5")
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
    run_time: Duration,
    checkpoint_time: Duration,
    run_started: Instant,
    task_index: usize,
) -> Option<ThetaCheckpointRuntime> {
    if !(options.checkpoint || options.restart || checkpoint_enabled()) {
        return None;
    }
    let checkpoint_dir = checkpoint_dir_for_job(job_name, options);
    let deadline = Some(run_started + run_time.checked_sub(checkpoint_time).unwrap_or_default());
    Some(ThetaCheckpointRuntime {
        path: theta_task_checkpoint_path(checkpoint_dir, task_index),
        interval: checkpoint_time,
        resume: options.restart,
        heartbeat_path: None,
        deadline,
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
    let stopped_early = result.tasks.iter().any(|task_result| {
        task_result.thermalization_sweeps < task_result.task.thermalization
            || task_result.measurement_sweeps < task_result.task.sweeps
    });
    Ok(JobRunSummary {
        output_path,
        task_count,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        checkpoint_paths,
        stopped_early,
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
    write_theta_checkpoint_measurements(state, &theta_task_measurement_path_from_checkpoint(path))?;

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
    measurements_group
        .create_dataset("default_bin_size")
        .with_i64_data(&[state.task.binsize as i64])
        .with_shape(&[]);
    let mut measurement_observables_group = measurements_group.create_group("observables");
    for (name, accumulator) in &state.measurement_accumulators {
        let mut observable_group = measurement_observables_group.create_group(name);
        observable_group
            .create_dataset("bin_length")
            .with_i64_data(&[accumulator.internal_bin_length as i64])
            .with_shape(&[]);
        observable_group
            .create_dataset("current_bin_filling")
            .with_i64_data(&[accumulator.pending_count as i64])
            .with_shape(&[]);
        observable_group
            .create_dataset("samples")
            .with_f64_data(&[accumulator.pending_sum])
            .with_shape(&[1])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1000]);
        measurement_observables_group.add_group(observable_group.finish());
    }
    measurements_group.add_group(measurement_observables_group.finish());
    builder.add_group(measurements_group.finish());

    let mut metadata_group = builder.create_group("metadata");
    add_fixed_string_dataset(&mut metadata_group, "checkpoint_version", "2");
    add_fixed_string_dataset(&mut metadata_group, "model", "theta");
    add_fixed_string_dataset(&mut metadata_group, "task", &task_json);
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

fn write_theta_checkpoint_measurements(
    state: &ThetaCheckpointState,
    path: &Path,
) -> Result<Option<PathBuf>, String> {
    let measurement_bins = state
        .measurement_accumulators
        .iter()
        .filter(|(_, accumulator)| !accumulator.internal_bins.is_empty())
        .collect::<Vec<_>>();
    if measurement_bins.is_empty() {
        return Ok(None);
    }
    let tmp_path = path.with_extension("h5.tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let mut builder = FileBuilder::new();
    let mut observables_group = builder.create_group("observables");
    for (name, accumulator) in measurement_bins {
        let mut observable_group = observables_group.create_group(name);
        observable_group
            .create_dataset("bin_length")
            .with_i64_data(&[accumulator.internal_bin_length as i64])
            .with_shape(&[]);
        observable_group
            .create_dataset("samples")
            .with_f64_data(&accumulator.internal_bins)
            .with_shape(&[accumulator.internal_bins.len() as u64])
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
    Ok(Some(path.to_path_buf()))
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
        measurement_accumulators: task_result
            .measurement_bins
            .iter()
            .map(|(name, bins)| {
                (
                    name.clone(),
                    ScalarAccumulator::from_internal_bins(
                        bins.clone(),
                        task_result.observables[name].internal_bin_len,
                    )
                    .compact(),
                )
            })
            .collect(),
    }
}

pub(crate) fn write_scheduler_heartbeat(
    path: impl AsRef<Path>,
    task_index: usize,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
) -> Result<(), String> {
    let path = path.as_ref();
    let tmp_path = path.with_extension(format!("heartbeat.tmp.{}", std::process::id()));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    {
        let mut heartbeat = File::create(&tmp_path).map_err(|err| err.to_string())?;
        writeln!(heartbeat, "task_index={task_index}").map_err(|err| err.to_string())?;
        writeln!(heartbeat, "pid={}", std::process::id()).map_err(|err| err.to_string())?;
        writeln!(heartbeat, "updated_at_unix={}", current_unix_seconds())
            .map_err(|err| err.to_string())?;
        writeln!(heartbeat, "thermalization_sweeps={thermalization_sweeps}")
            .map_err(|err| err.to_string())?;
        writeln!(heartbeat, "measurement_sweeps={measurement_sweeps}")
            .map_err(|err| err.to_string())?;
        heartbeat.sync_all().map_err(|err| err.to_string())?;
    }
    fs::rename(&tmp_path, path).map_err(|err| err.to_string())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_write_theta_checkpoint(
    checkpoint: Option<&ThetaCheckpointRuntime>,
    last_checkpoint: &mut Instant,
    task: &ThetaTask,
    task_index: usize,
    lattice: &ThetaLattice,
    rng: &FastRng,
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
        rng_word_pos: rng.position(),
        thermalization_sweeps,
        measurement_sweeps,
        acceptance_sum,
        acceptance_count,
        measurement_accumulators: series.compact(),
    };
    write_theta_checkpoint_state_to_path(&state, &checkpoint.path)?;
    if let Some(heartbeat_path) = &checkpoint.heartbeat_path {
        write_scheduler_heartbeat(
            heartbeat_path,
            task_index,
            thermalization_sweeps,
            measurement_sweeps,
        )?;
    }
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

fn read_optional_scalar_i64(group: &Group<'_>, name: &str) -> Result<Option<i64>, String> {
    match group.dataset(name) {
        Ok(dataset) => dataset
            .read_i64()
            .map_err(|err| err.to_string())?
            .into_iter()
            .next()
            .map(Some)
            .ok_or_else(|| format!("{name} dataset is empty")),
        Err(_) => Ok(None),
    }
}

fn read_optional_scalar_f64(group: &Group<'_>, name: &str) -> Result<Option<f64>, String> {
    match group.dataset(name) {
        Ok(dataset) => dataset
            .read_f64()
            .map_err(|err| err.to_string())?
            .into_iter()
            .next()
            .map(Some)
            .ok_or_else(|| format!("{name} dataset is empty")),
        Err(_) => Ok(None),
    }
}

fn read_measurement_accumulators(
    file: &Hdf5File,
    measurement_path: &Path,
    fallback_bin_length: usize,
) -> Result<BTreeMap<String, CompactObservableAccumulator>, String> {
    let mut accumulators = read_measurement_file_accumulators(measurement_path, fallback_bin_length)?;
    let Ok(measurements_group) = file.group("measurements") else {
        return Ok(accumulators);
    };
    let observable_names = if let Ok(observables_group) = measurements_group.group("observables") {
        observables_group.groups().map_err(|err| err.to_string())?
    } else {
        measurements_group.groups().map_err(|err| err.to_string())?
    };
    for name in observable_names {
        let observable_group = if let Ok(observables_group) = measurements_group.group("observables") {
            observables_group.group(&name).map_err(|err| err.to_string())?
        } else {
            measurements_group
                .group(&name)
                .map_err(|err| err.to_string())?
        };
        let internal_bin_length = read_optional_scalar_i64(&observable_group, "bin_length")?
            .map(|value| value as usize)
            .unwrap_or(fallback_bin_length.max(1));
        let mut internal_bins = accumulators
            .remove(&name)
            .map(|accumulator| accumulator.internal_bins)
            .unwrap_or_default();
        let carlo_current_bin = observable_group
            .dataset("current_bin_filling")
            .is_ok()
            .then(|| {
                let samples = observable_group
                    .dataset("samples")
                    .map_err(|err| err.to_string())?
                    .read_f64()
                    .map_err(|err| err.to_string())?;
                let filling = read_scalar_i64(&observable_group, "current_bin_filling")? as usize;
                Ok::<_, String>((samples.into_iter().next().unwrap_or(0.0), filling))
            })
            .transpose()?;
        match observable_group.dataset("internal_bins") {
            Ok(dataset) => {
                internal_bins.extend(dataset.read_f64().map_err(|err| err.to_string())?);
            }
            Err(_) if carlo_current_bin.is_none() => {
                if let Ok(dataset) = observable_group.dataset("samples") {
                    internal_bins.extend(
                        ScalarAccumulator::from_samples(
                            dataset.read_f64().map_err(|err| err.to_string())?,
                            internal_bin_length,
                        )
                        .compact()
                        .internal_bins,
                    );
                }
            }
            Err(_) => {}
        }
        let pending_sum = if let Some((sum, _)) = carlo_current_bin {
            sum
        } else {
            read_optional_scalar_f64(&observable_group, "pending_sum")?.unwrap_or(0.0)
        };
        let pending_count = if let Some((_, count)) = carlo_current_bin {
            count
        } else {
            read_optional_scalar_i64(&observable_group, "pending_count")?
                .map(|value| value as usize)
                .unwrap_or(0)
        };
        let total_count = read_optional_scalar_i64(&observable_group, "total_count")?
            .map(|value| value as usize)
            .unwrap_or(internal_bins.len() * internal_bin_length + pending_count);
        accumulators.insert(
            name,
            CompactObservableAccumulator {
                internal_bins,
                pending_sum,
                pending_count,
                total_count,
                internal_bin_length,
            },
        );
    }
    Ok(accumulators)
}

fn read_measurement_file_accumulators(
    path: &Path,
    fallback_bin_length: usize,
) -> Result<BTreeMap<String, CompactObservableAccumulator>, String> {
    let Ok(file) = hdf5_file(path) else {
        return Ok(BTreeMap::new());
    };
    let Ok(observables_group) = file.group("observables") else {
        return Ok(BTreeMap::new());
    };
    let mut accumulators = BTreeMap::new();
    for name in observables_group.groups().map_err(|err| err.to_string())? {
        let observable_group = observables_group.group(&name).map_err(|err| err.to_string())?;
        let internal_bin_length = read_optional_scalar_i64(&observable_group, "bin_length")?
            .map(|value| value as usize)
            .unwrap_or(fallback_bin_length.max(1));
        let internal_bins = observable_group
            .dataset("samples")
            .map_err(|err| err.to_string())?
            .read_f64()
            .map_err(|err| err.to_string())?;
        accumulators.insert(
            name,
            CompactObservableAccumulator {
                total_count: internal_bins.len() * internal_bin_length,
                internal_bins,
                pending_sum: 0.0,
                pending_count: 0,
                internal_bin_length,
            },
        );
    }
    Ok(accumulators)
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
    let measurement_accumulators = read_measurement_accumulators(
        &file,
        &theta_task_measurement_path_from_checkpoint(path),
        task.binsize,
    )?;

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
        measurement_accumulators,
    })
}

pub fn write_theta_task_checkpoint_to_path(
    task_result: &ThetaTaskResult,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let state = theta_checkpoint_state_from_result(task_result);
    write_theta_checkpoint_state_to_path(&state, path)
}
