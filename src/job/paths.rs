fn rank_result_path_with_options(
    job_name: &str,
    rank: usize,
    options: &ThetaRunOptions,
) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.rank{rank}.results.json", file_stem(job_name)))
}

fn merged_result_path_with_options(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.results.json", file_stem(job_name)))
}

fn default_measurement_dir_with_options(job_name: &str, options: &ThetaRunOptions) -> PathBuf {
    options
        .output_dir
        .clone()
        .or_else(|| std::env::var("XY_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| default_output_dir(job_name))
        .join(format!("{}.data", file_stem(job_name)))
}

fn default_output_dir(job_name: &str) -> PathBuf {
    Path::new(job_name)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("results"))
}

fn file_stem(job_name: &str) -> String {
    Path::new(job_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(job_name)
        .to_string()
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

fn env_usize_any(keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
}

fn mpi_env_rank() -> Option<usize> {
    env_usize_any(&[
        "XY_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    ])
}

fn mpi_env_world_size() -> Option<usize> {
    env_usize_any(&[
        "XY_WORLD_SIZE",
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ])
}

fn parse_env_duration(key: &str, default: Duration) -> Result<Duration, String> {
    std::env::var(key).map_or(Ok(default), |value| parse_duration(&value))
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let (date_part, time_part) = value
        .split_once('-')
        .map_or(("0", value), |(days, rest)| (days, rest));
    let days = date_part
        .parse::<u64>()
        .map_err(|err| format!("failed to parse duration days in {value}: {err}"))?;
    let parts = time_part.split(':').collect::<Vec<_>>();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("{value} does not match [[HH:]MM:]SS"));
    }
    let numbers = parts
        .iter()
        .map(|part| {
            part.parse::<u64>()
                .map_err(|err| format!("failed to parse duration component in {value}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (hours, minutes, seconds) = match numbers.as_slice() {
        [seconds] => (0, 0, *seconds),
        [minutes, seconds] => (0, *minutes, *seconds),
        [hours, minutes, seconds] => (*hours, *minutes, *seconds),
        _ => unreachable!(),
    };
    Ok(Duration::from_secs(
        days * 24 * 60 * 60 + hours * 60 * 60 + minutes * 60 + seconds,
    ))
}

fn parse_env_value<T>(key: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or(Ok(default), |value| {
        value
            .parse::<T>()
            .map_err(|err| format!("failed to parse {key}: {err}"))
    })
}

fn parse_env_list<T>(key: &str, default: &[T]) -> Result<Vec<T>, String>
where
    T: Clone + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or_else(|_| Ok(default.to_vec()), |value| parse_list(key, &value))
}

fn parse_optional_env_list<T>(key: &str) -> Result<Option<Vec<T>>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(key).map_or(Ok(None), |value| parse_list(key, &value).map(Some))
}

fn parse_list<T>(key: &str, value: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let parsed = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<T>()
                .map_err(|err| format!("failed to parse {key}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(parsed)
}

fn join_display<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}
