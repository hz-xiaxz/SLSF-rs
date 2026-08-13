use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure::{CharacterSet, Datatype, FileBuilder, GroupBuilder, StringPadding};
use carlo_mc::*;

use crate::model::*;



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
    let output_path = write_theta_job_result_to_path(result, output_path)?;
    let stopped_early = result.tasks.iter().any(|task_result| {
        task_result.thermalization_sweeps < task_result.task.thermalization
            || task_result.measurement_sweeps < task_result.task.sweeps
    });
    Ok(JobRunSummary {
        output_path,
        task_count,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        checkpoint_paths: Vec::new(),
        stopped_early,
    })
}

fn carlo_run_to_theta_result(run: &RunResult<ThetaTask, carlo_mc::Estimate>) -> ThetaJobResult {
    let tasks = run
        .result
        .tasks
        .iter()
        .map(|task| ThetaTaskResult {
            task: task.task.parameters.clone(),
            task_index: task.task_index,
            observables: task.observables.clone(),
            acceptance: task.metadata.get("acceptance").copied().unwrap_or(0.0),
            measurements: task.measurement_sweeps,
            measurement_bins: task.measurement_bins.clone(),
            measurement_samples: BTreeMap::new(),
            final_theta: Vec::new(),
            final_j_z: Vec::new(),
            rng_word_pos: 0,
            thermalization_sweeps: task.thermalization_sweeps,
            measurement_sweeps: task.measurement_sweeps,
            acceptance_sum: 0.0,
            acceptance_count: 0,
        })
        .collect();
    ThetaJobResult {
        job_name: run.result.job_name.clone(),
        rank: run.result.rank,
        world_size: run.result.world_size,
        tasks,
    }
}

pub fn run_theta_job_with_carlo(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobRunSummary, String> {
    let job_name = cfg.job_name.clone();
    let assignment = options.assignment()?;
    let tasks = theta_tasks_from_config(cfg)?;
    let job = Job::<ThetaModel>::new(job_name.clone(), tasks);
    let run_options = RunOptions {
        assignment: Some(assignment),
        ..RunOptions::default()
    };
    let started = Instant::now();
    let run = Runner::<ThetaModel>::new()
        .run(&job, &run_options)
        .map_err(|error| error.to_string())?;
    let result = carlo_run_to_theta_result(&run);
    let output_path = options
        .output_file
        .clone()
        .unwrap_or_else(|| rank_result_path_with_options(&job_name, assignment.rank, &options));
    finish_theta_job_run(&result, output_path, started, &options)
}

pub fn run_theta_job_dynamic_with_carlo(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobRunSummary, String> {
    let job_name = cfg.job_name.clone();
    let assignment = options.assignment()?;
    let tasks = theta_tasks_from_config(cfg)?;
    let job = Job::<ThetaModel>::new(job_name.clone(), tasks);
    let checkpoint_dir = options
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| default_measurement_dir_with_options(&job_name, &options));
    let run_options = RunOptions {
        assignment: Some(assignment),
        checkpoint_dir: Some(checkpoint_dir),
        ..RunOptions::default()
    };
    let started = Instant::now();
    let run = Runner::<ThetaModel>::new()
        .dynamic()
        .run(&job, &run_options)
        .map_err(|error| error.to_string())?;
    let result = carlo_run_to_theta_result(&run);
    let output_path = options
        .output_file
        .clone()
        .unwrap_or_else(|| rank_result_path_with_options(&job_name, assignment.rank, &options));
    finish_theta_job_run(&result, output_path, started, &options)
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

pub fn merge_theta_job_with_carlo(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobMergeSummary, String> {
    let job_name = cfg.job_name.clone();
    let world_size = options.world_size();
    let input_paths = (0..world_size)
        .map(|rank| rank_result_path_with_options(&job_name, rank, &options))
        .collect::<Vec<_>>();
    let output_path = options
        .merged_output_file
        .clone()
        .unwrap_or_else(|| merged_result_path_with_options(&job_name, &options));
    let merged = merge_theta_job_result_files(&input_paths)?;
    let task_count = merged.tasks.len();
    let output_path = write_theta_job_result_to_path(&merged, output_path)?;
    Ok(JobMergeSummary {
        output_path,
        input_paths,
        task_count,
    })
}

pub fn run_theta_job_mpi_with_carlo(
    cfg: &ThetaJobConfig,
    options: ThetaRunOptions,
) -> Result<JobMpiRunSummary, String> {
    let rank = options.rank();
    let world_size = options.world_size();
    let single = options.single || world_size == 1;
    let run = if single {
        let mut single_options = options.clone();
        single_options.single = true;
        single_options.rank = Some(0);
        single_options.world_size = Some(1);
        run_theta_job_with_carlo(cfg, single_options)?
    } else {
        run_theta_job_dynamic_with_carlo(cfg, options.clone())?
    };
    let merge = if !single && world_size > 1 && rank == 0 {
        Some(merge_theta_job_with_carlo(cfg, options.clone())?)
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

fn remove_path_if_exists(path: &Path) -> Result<(), String> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path).map_err(|err| err.to_string()),
        Ok(_) => fs::remove_file(path).map_err(|err| err.to_string()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.to_string()),
    }
}

pub fn delete_theta_job_outputs(
    cfg: &ThetaJobConfig,
    options: &ThetaRunOptions,
) -> Result<(), String> {
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
    Ok(())
}

pub fn theta_job_status_with_carlo(
    cfg: &ThetaJobConfig,
    options: &ThetaRunOptions,
) -> Result<String, String> {
    let mut completed = 0usize;
    let mut rank_files = 0usize;
    for rank in 0..options.world_size() {
        let path = rank_result_path_with_options(&cfg.job_name, rank, options);
        if path.exists() {
            rank_files += 1;
            if let Ok(result) = read_theta_job_result(path) {
                completed += result.tasks.len();
            }
        }
    }
    Ok(format!(
        "{} theta task(s) completed; {} of {} rank result file(s) present",
        completed,
        rank_files,
        options.world_size()
    ))
}
