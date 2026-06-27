pub fn run_theta_job_command_from_env(args: &[String]) -> Result<String, String> {
    run_theta_job_command(
        std::iter::once(OsString::from("slsf")).chain(args.iter().map(OsString::from)),
    )
}

pub fn run_theta_job_command<I, T>(args: I) -> Result<String, String>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match ThetaCli::try_parse_from(args) {
        Ok(cli) => {
            let command = cli.command.unwrap_or(ThetaCommand::Run(CommandArgs {
                from_env: true,
                ..Default::default()
            }));
            run_theta_command(command)
        }
        Err(err)
            if matches!(
                err.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            Ok(err.to_string())
        }
        Err(err) => Err(err.to_string()),
    }
}

fn run_theta_command(command: ThetaCommand) -> Result<String, String> {
    match command {
        ThetaCommand::Run(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display()
            ))
        }
        ThetaCommand::Merge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = merge_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "merged {} rank file(s), {} theta task(s); wrote {}",
                summary.input_paths.len(),
                summary.task_count,
                summary.output_path.display()
            ))
        }
        ThetaCommand::RunMerge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let run = run_theta_job_with_options(&cfg, options.clone())?;
            let merge = merge_theta_job_with_options(&cfg, options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}; merged {} rank file(s) into {}",
                run.task_count,
                run.elapsed_seconds,
                run.output_path.display(),
                merge.input_paths.len(),
                merge.output_path.display()
            ))
        }
        ThetaCommand::RunDynamic(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_dynamic_with_options(&cfg, options)?;
            Ok(format!(
                "dynamically completed {} theta task(s) in {:.3}s; wrote {}; checkpoints {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display(),
                summary.checkpoint_paths.len()
            ))
        }
        ThetaCommand::MpiRun(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_mpi_with_options(&cfg, options)?;
            let merge_message = summary
                .merge
                .as_ref()
                .map(|merge| {
                    format!(
                        "; merged {} rank file(s) into {}",
                        merge.input_paths.len(),
                        merge.output_path.display()
                    )
                })
                .unwrap_or_default();
            Ok(format!(
                "MPI rank {}/{} completed {} theta task(s) in {:.3}s; wrote {}{}",
                summary.rank,
                summary.world_size,
                summary.run.task_count,
                summary.run.elapsed_seconds,
                summary.run.output_path.display(),
                merge_message
            ))
        }
        ThetaCommand::Checkpoint(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let job = cfg.make_job()?;
            let assignment = options.assignment()?;
            let result = run_theta_job(&job, assignment)?;
            let dir = options
                .checkpoint_dir
                .clone()
                .unwrap_or_else(|| default_measurement_dir_with_options(&job.name, &options));
            let paths = write_theta_job_checkpoints(&result, dir)?;
            Ok(format!("wrote {} checkpoint file(s)", paths.len()))
        }
        ThetaCommand::Status(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            theta_job_status(&cfg, &options)
        }
        ThetaCommand::Delete(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            delete_theta_job_outputs(&cfg, &options)?;
            Ok("deleted theta job outputs".to_string())
        }
    }
}

fn load_config_and_options(
    args: &CommandArgs,
) -> Result<(ThetaJobConfig, ThetaRunOptions), String> {
    let (cfg, options) = if let Some(path) = &args.config {
        ThetaJobConfig::from_toml_path(path)?
    } else {
        (ThetaJobConfig::from_env()?, ThetaRunOptions::default())
    };
    Ok((cfg, options.with_overrides(args)))
}
