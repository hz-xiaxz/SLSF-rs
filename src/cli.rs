use std::ffi::OsString;
use std::path::PathBuf;

use clap::{error::ErrorKind, Args as ClapArgs, Parser, Subcommand};
use carlo_mc::*;

use crate::model::*;
use crate::results::*;



impl ThetaRunOptions {
    fn with_overrides(mut self, args: &CommandArgs) -> Self {
        if let Some(value) = &args.output_dir {
            self.output_dir = Some(value.clone());
        }
        if let Some(value) = &args.output_file {
            self.output_file = Some(value.clone());
        }
        if let Some(value) = &args.merged_output_file {
            self.merged_output_file = Some(value.clone());
        }
        if let Some(value) = &args.measurement_dir {
            self.measurement_dir = Some(value.clone());
        }
        if let Some(value) = &args.checkpoint_dir {
            self.checkpoint_dir = Some(value.clone());
        }
        if let Some(value) = &args.scheduler_dir {
            self.scheduler_dir = Some(value.clone());
        }
        if args.checkpoint {
            self.checkpoint = true;
        }
        if args.restart {
            self.restart = true;
        }
        if args.single {
            self.single = true;
            self.rank = Some(0);
            self.world_size = Some(1);
        }
        if let Some(value) = args.rank {
            self.rank = Some(value);
        }
        if let Some(value) = args.world_size {
            self.world_size = Some(value);
        }
        self
    }

    pub(crate) fn rank(&self) -> usize {
        if self.single {
            return 0;
        }
        self.rank.or_else(mpi_env_rank).unwrap_or(0)
    }

    pub(crate) fn world_size(&self) -> usize {
        if self.single {
            return 1;
        }
        self.world_size.or_else(mpi_env_world_size).unwrap_or(1)
    }

    pub(crate) fn assignment(&self) -> Result<JobAssignment, String> {
        JobAssignment::new(self.rank(), self.world_size()).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust XY/Theta job runner")]
pub struct ThetaCli {
    #[command(subcommand)]
    command: Option<ThetaCommand>,
}

#[derive(Debug, Subcommand)]
pub enum ThetaCommand {
    #[command(alias = "r")]
    Run(CommandArgs),
    #[command(alias = "m")]
    Merge(CommandArgs),
    Check(CommandArgs),
    RunMerge(CommandArgs),
    RunDynamic(CommandArgs),
    MpiRun(CommandArgs),
    Checkpoint(CommandArgs),
    #[command(alias = "df")]
    Dataframe(ResultToolArgs),
    #[command(alias = "s")]
    Status(CommandArgs),
    #[command(alias = "d")]
    Delete(CommandArgs),
}

#[derive(Debug, Clone, Default, ClapArgs)]
pub struct CommandArgs {
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub from_env: bool,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long)]
    pub output_file: Option<PathBuf>,
    #[arg(long)]
    pub merged_output_file: Option<PathBuf>,
    #[arg(long)]
    pub measurement_dir: Option<PathBuf>,
    #[arg(long)]
    pub checkpoint_dir: Option<PathBuf>,
    #[arg(long)]
    pub scheduler_dir: Option<PathBuf>,
    #[arg(long)]
    pub checkpoint: bool,
    #[arg(short, long)]
    pub single: bool,
    #[arg(short, long)]
    pub restart: bool,
    #[arg(long)]
    pub rank: Option<usize>,
    #[arg(long)]
    pub world_size: Option<usize>,
}

#[derive(Debug, Clone, ClapArgs)]
pub struct ResultToolArgs {
    pub result_json: PathBuf,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub plot: bool,
    #[arg(long, default_value = "Energy")]
    pub observable: String,
    #[arg(long)]
    pub plot_output: Option<PathBuf>,
    #[arg(long)]
    pub script_output: Option<PathBuf>,
}


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
            let summary = run_theta_job_with_carlo(&cfg, options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display()
            ))
        }
        ThetaCommand::Merge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = merge_theta_job_with_carlo(&cfg, options)?;
            Ok(format!(
                "merged {} rank file(s), {} theta task(s); wrote {}",
                summary.input_paths.len(),
                summary.task_count,
                summary.output_path.display()
            ))
        }
        ThetaCommand::Check(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let job = cfg.make_job()?;
            let output_path = merged_result_path_with_options(&job.name, &options);
            let lattice_count = cfg.lattice_specs().len();
            Ok(format!(
                "valid theta config: name={}; tasks={}; lattices={}; temperatures={}; samples={}; sweeps={}; thermalization={}; binsize={}; run_time={}s; checkpoint_time={}s; output={}",
                job.name,
                job.tasks.len(),
                lattice_count,
                cfg.temperatures.len(),
                cfg.samples,
                cfg.sweeps,
                cfg.thermalization,
                cfg.binsize,
                cfg.run_time.as_secs(),
                cfg.checkpoint_time.as_secs(),
                output_path.display()
            ))
        }
        ThetaCommand::RunMerge(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let run = run_theta_job_with_carlo(&cfg, options.clone())?;
            let merge = merge_theta_job_with_carlo(&cfg, options)?;
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
            let summary = run_theta_job_dynamic_with_carlo(&cfg, options)?;
            Ok(format!(
                "dynamically completed {} theta task(s) in {:.3}s; wrote {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display()
            ))
        }
        ThetaCommand::MpiRun(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            let summary = run_theta_job_mpi_with_carlo(&cfg, options)?;
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
            let mut checkpoint_options = options.clone();
            checkpoint_options.checkpoint = true;
            let summary = run_theta_job_with_carlo(&cfg, checkpoint_options)?;
            Ok(format!(
                "completed {} theta task(s) in {:.3}s; wrote {}",
                summary.task_count,
                summary.elapsed_seconds,
                summary.output_path.display()
            ))
        }
        ThetaCommand::Dataframe(args) => run_result_tool_command(args),
        ThetaCommand::Status(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            theta_job_status_with_carlo(&cfg, &options)
        }
        ThetaCommand::Delete(args) => {
            let (cfg, options) = load_config_and_options(&args)?;
            delete_theta_job_outputs(&cfg, &options)?;
            Ok("deleted theta job outputs".to_string())
        }
    }
}

fn run_result_tool_command(args: ResultToolArgs) -> Result<String, String> {
    let table_path = args.output.clone().unwrap_or_else(|| {
        args.result_json
            .with_file_name(format!(
                "{}.tsv",
                args.result_json
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("results")
            ))
    });
    crate::result_tools::write_dataframe(&args.result_json, &table_path)?;
    if args.plot {
        let plot_path = args
            .plot_output
            .clone()
            .unwrap_or_else(|| table_path.with_extension(format!("{}.png", args.observable)));
        let script_path = args
            .script_output
            .clone()
            .unwrap_or_else(|| table_path.with_extension(format!("{}.gnuplot", args.observable)));
        crate::result_tools::write_gnuplot_script(
            &table_path,
            &script_path,
            &plot_path,
            &args.observable,
        )?;
        crate::result_tools::plot_with_gnuplot(&script_path)?;
        Ok(format!(
            "wrote {}; plotted {} to {} using {}",
            table_path.display(),
            args.observable,
            plot_path.display(),
            script_path.display()
        ))
    } else {
        Ok(format!("wrote {}", table_path.display()))
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
