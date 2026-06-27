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

    fn rank(&self) -> usize {
        if self.single {
            return 0;
        }
        self.rank.or_else(mpi_env_rank).unwrap_or(0)
    }

    fn world_size(&self) -> usize {
        if self.single {
            return 1;
        }
        self.world_size.or_else(mpi_env_world_size).unwrap_or(1)
    }

    fn assignment(&self) -> Result<JobAssignment, String> {
        JobAssignment::new(self.rank(), self.world_size())
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
    RunMerge(CommandArgs),
    RunDynamic(CommandArgs),
    MpiRun(CommandArgs),
    Checkpoint(CommandArgs),
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
