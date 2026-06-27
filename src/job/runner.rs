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
