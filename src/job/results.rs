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
