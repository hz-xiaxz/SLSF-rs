const CLAIM_STALE_AFTER: Duration = Duration::from_secs(6 * 60 * 60);

fn task_marker_path(scheduler_dir: &Path, task_index: usize, extension: &str) -> PathBuf {
    scheduler_dir.join(format!("task{task_index:04}.{extension}"))
}

fn current_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn claim_age(path: &Path) -> Option<Duration> {
    fs::metadata(path)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
}

fn heartbeat_path_for_task(scheduler_dir: &Path, task_index: usize) -> PathBuf {
    task_marker_path(scheduler_dir, task_index, "heartbeat")
}

fn task_liveness_age(
    scheduler_dir: &Path,
    task_index: usize,
    claim_path: &Path,
) -> Option<Duration> {
    claim_age(&heartbeat_path_for_task(scheduler_dir, task_index)).or_else(|| claim_age(claim_path))
}

pub(crate) fn remove_stale_claim_if_needed(
    scheduler_dir: &Path,
    task_index: usize,
    claim_path: &Path,
    stale_after: Duration,
) -> Result<(), String> {
    if task_marker_path(scheduler_dir, task_index, "done").exists() {
        return Ok(());
    }
    let Some(age) = task_liveness_age(scheduler_dir, task_index, claim_path) else {
        return Ok(());
    };
    if age < stale_after {
        return Ok(());
    }

    let stale_path = scheduler_dir.join(format!(
        "task{task_index:04}.claim.stale.{}.{}",
        current_unix_seconds(),
        std::process::id()
    ));
    match fs::rename(claim_path, &stale_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!(
            "failed to retire stale claim {}: {err}",
            claim_path.display()
        )),
    }
}

fn try_claim_task(
    scheduler_dir: &Path,
    task_index: usize,
    rank: usize,
    world_size: usize,
) -> Result<Option<PathBuf>, String> {
    let done_path = task_marker_path(scheduler_dir, task_index, "done");
    if done_path.exists() {
        return Ok(None);
    }

    let claim_path = task_marker_path(scheduler_dir, task_index, "claim");
    remove_stale_claim_if_needed(scheduler_dir, task_index, &claim_path, CLAIM_STALE_AFTER)?;
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&claim_path)
    {
        Ok(mut claim) => {
            writeln!(claim, "task_index={task_index}").map_err(|err| err.to_string())?;
            writeln!(claim, "rank={rank}").map_err(|err| err.to_string())?;
            writeln!(claim, "world_size={world_size}").map_err(|err| err.to_string())?;
            writeln!(claim, "pid={}", std::process::id()).map_err(|err| err.to_string())?;
            writeln!(claim, "claimed_at_unix={}", current_unix_seconds())
                .map_err(|err| err.to_string())?;
            Ok(Some(claim_path))
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
        Err(err) => Err(format!(
            "failed to claim task {task_index} at {}: {err}",
            claim_path.display()
        )),
    }
}

fn mark_task_done(scheduler_dir: &Path, task_index: usize, claim_path: &Path) -> Result<(), String> {
    let done_path = task_marker_path(scheduler_dir, task_index, "done");
    let tmp_done_path = scheduler_dir.join(format!(
        "task{task_index:04}.done.tmp.{}.{}",
        current_unix_seconds(),
        std::process::id()
    ));
    {
        let mut done = File::create(&tmp_done_path).map_err(|err| err.to_string())?;
        writeln!(done, "task_index={task_index}").map_err(|err| err.to_string())?;
        writeln!(done, "completed_at_unix={}", current_unix_seconds())
            .map_err(|err| err.to_string())?;
        done.sync_all().map_err(|err| err.to_string())?;
    }
    fs::rename(&tmp_done_path, &done_path).map_err(|err| err.to_string())?;
    remove_path_if_exists(&heartbeat_path_for_task(scheduler_dir, task_index))?;
    match fs::remove_file(claim_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!(
            "failed to remove claim {} after completing task {task_index}: {err}",
            claim_path.display()
        )),
    }
}

fn dynamic_task_order(
    task_count: usize,
    rank: usize,
    world_size: usize,
) -> impl Iterator<Item = usize> {
    let offset = if task_count == 0 {
        0
    } else {
        rank.saturating_mul(task_count) / world_size.max(1)
    };
    (0..task_count).map(move |step| (offset + step) % task_count)
}

pub fn run_theta_job_dynamic(
    job: &ThetaJob,
    scheduler_dir: impl AsRef<Path>,
    rank: usize,
    world_size: usize,
) -> Result<ThetaJobResult, String> {
    let scheduler_dir = scheduler_dir.as_ref();
    fs::create_dir_all(scheduler_dir).map_err(|err| err.to_string())?;

    let mut tasks = Vec::new();
    for task_index in dynamic_task_order(job.tasks.len(), rank, world_size) {
        let task = &job.tasks[task_index];
        let Some(claim_path) = try_claim_task(scheduler_dir, task_index, rank, world_size)? else {
            continue;
        };
        let mut task_result = run_theta_task(task)?;
        task_result.task_index = task_index;
        mark_task_done(scheduler_dir, task_index, &claim_path)?;
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
    let mut completed_markers = 0usize;
    let mut claimed = 0usize;
    let mut stale = 0usize;
    let mut heartbeat = 0usize;
    for task_index in 0..job.tasks.len() {
        if task_marker_path(&scheduler_dir, task_index, "done").exists() {
            completed_markers += 1;
            continue;
        }
        let claim_path = task_marker_path(&scheduler_dir, task_index, "claim");
        if claim_path.exists() {
            claimed += 1;
            let heartbeat_path = heartbeat_path_for_task(&scheduler_dir, task_index);
            if heartbeat_path.exists() {
                heartbeat += 1;
            }
            if task_liveness_age(&scheduler_dir, task_index, &claim_path)
                .map(|age| age >= CLAIM_STALE_AFTER)
                .unwrap_or(false)
            {
                stale += 1;
            }
        }
    }
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
    let pending = job.tasks.len().saturating_sub(completed + claimed);
    Ok(format!(
        "{} of {} theta task(s) marked done; {} claimed/running ({} with heartbeat, {} stale); {} pending; {} of {} rank result file(s) present",
        completed,
        job.tasks.len(),
        claimed,
        heartbeat,
        stale,
        pending,
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
            let checkpoint = checkpoint_runtime_for_task(
                &job.name,
                &options,
                cfg.run_time,
                cfg.checkpoint_time,
                started,
                task_index,
            );
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
    for task_index in dynamic_task_order(job.tasks.len(), rank, world_size) {
        let task = &job.tasks[task_index];
        let Some(claim_path) = try_claim_task(&scheduler_dir, task_index, rank, world_size)? else {
            continue;
        };
        let mut checkpoint = checkpoint_runtime_for_task(
            &job.name,
            &options,
            cfg.run_time,
            cfg.checkpoint_time,
            started,
            task_index,
        );
        if let Some(checkpoint) = checkpoint.as_mut() {
            checkpoint.heartbeat_path = Some(heartbeat_path_for_task(&scheduler_dir, task_index));
        }
        let task_result = run_theta_task_with_checkpoint(task, task_index, checkpoint.as_ref())?;
        mark_task_done(&scheduler_dir, task_index, &claim_path)?;
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

fn cleanup_theta_job_checkpoints(
    cfg: &ThetaJobConfig,
    options: &ThetaRunOptions,
    task_count: usize,
) -> Result<(), String> {
    if !(options.checkpoint || checkpoint_enabled()) {
        return Ok(());
    }
    let checkpoint_dir = checkpoint_dir_for_job(&cfg.job_name, options);
    for task_index in 0..task_count {
        remove_path_if_exists(&theta_task_checkpoint_path(&checkpoint_dir, task_index))?;
    }
    Ok(())
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
    cleanup_theta_job_checkpoints(cfg, &options, task_count)?;
    Ok(JobMergeSummary {
        output_path,
        input_paths,
        task_count,
    })
}
