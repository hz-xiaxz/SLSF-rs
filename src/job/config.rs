impl Default for ThetaJobConfig {
    fn default() -> Self {
        Self {
            l: (4..=20).collect(),
            l_x: None,
            l_y: None,
            l_z: None,
            temperatures: vec![2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
            delta_j_z: vec![0.0],
            delta_j_xy: vec![0.0],
            samples: 16,
            base_seed: 20260414,
            j_xy: 1.0,
            j_z_mean: 0.1,
            sweeps: 20_000,
            thermalization: 5_000,
            binsize: 50,
            proposal_width: std::f64::consts::PI,
            wolff_steps: 1,
            correlation_rmax: None,
            correlation_rmax_xy: None,
            correlation_rmax_z: None,
            run_time: Duration::from_secs(12 * 60 * 60),
            checkpoint_time: Duration::from_secs(30 * 60),
            job_name: "xy_theta_rust".to_string(),
        }
    }
}

impl ThetaJobConfig {
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<(Self, ThetaRunOptions), String> {
        let text = fs::read_to_string(path.as_ref()).map_err(|err| err.to_string())?;
        let spec = toml::from_str::<ThetaJobToml>(&text).map_err(|err| err.to_string())?;
        Ok(Self::from_toml_spec(spec))
    }

    pub fn from_toml_spec(spec: ThetaJobToml) -> (Self, ThetaRunOptions) {
        let mut cfg = Self::default();
        if let Some(name) = spec.name {
            cfg.job_name = name;
        }
        if let Some(run_time) = spec.run_time {
            cfg.run_time = parse_duration(&run_time).unwrap_or(cfg.run_time);
        }
        if let Some(checkpoint_time) = spec.checkpoint_time {
            cfg.checkpoint_time = parse_duration(&checkpoint_time).unwrap_or(cfg.checkpoint_time);
        }
        if let Some(model) = spec.model {
            if let Some(value) = model.l {
                cfg.l = value;
            }
            cfg.l_x = model.l_x;
            cfg.l_y = model.l_y;
            cfg.l_z = model.l_z;
            if let Some(value) = model.temperatures.or(model.t) {
                cfg.temperatures = value;
            }
            if let Some(value) = model.delta_j_z.or(model.djz) {
                cfg.delta_j_z = value;
            }
            if let Some(value) = model.delta_j_xy.or(model.djxy) {
                cfg.delta_j_xy = value;
            }
            if let Some(value) = model.samples {
                cfg.samples = value;
            }
            if let Some(value) = model.base_seed {
                cfg.base_seed = value;
            }
            if let Some(value) = model.j_xy {
                cfg.j_xy = value;
            }
            if let Some(value) = model.j_z_mean.or(model.j_z) {
                cfg.j_z_mean = value;
            }
        }
        if let Some(run) = spec.run {
            if let Some(value) = run.sweeps {
                cfg.sweeps = value;
            }
            if let Some(value) = run.thermalization {
                cfg.thermalization = value;
            }
            if let Some(value) = run.binsize {
                cfg.binsize = value;
            }
            if let Some(value) = run.proposal_width {
                cfg.proposal_width = value;
            }
            if let Some(value) = run.wolff_steps {
                cfg.wolff_steps = value;
            }
        }
        if let Some(measure) = spec.measure {
            cfg.correlation_rmax = measure.corr_rmax;
            cfg.correlation_rmax_xy = measure.corr_rmax_xy.or(cfg.correlation_rmax);
            cfg.correlation_rmax_z = measure.corr_rmax_z.or(cfg.correlation_rmax);
        }
        let options = ThetaRunOptions {
            output_dir: spec.output_dir,
            output_file: spec.output_file,
            merged_output_file: spec.merged_output_file,
            measurement_dir: spec.measurement_dir,
            checkpoint_dir: spec.checkpoint_dir,
            scheduler_dir: spec.scheduler_dir,
            checkpoint: spec.checkpoint.unwrap_or(false),
            ..Default::default()
        };
        (cfg, options)
    }

    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();
        cfg.l = parse_env_list("XY_L", &cfg.l)?;
        cfg.l_x = parse_optional_env_list("XY_LX")?;
        cfg.l_y = parse_optional_env_list("XY_LY")?;
        cfg.l_z = parse_optional_env_list("XY_LZ")?;
        cfg.temperatures = parse_env_list("XY_T", &cfg.temperatures)?;
        cfg.delta_j_z = parse_env_list("XY_DJZ", &cfg.delta_j_z)?;
        cfg.delta_j_xy = parse_env_list("XY_DJXY", &cfg.delta_j_xy)?;
        cfg.samples = parse_env_value("XY_SAMPLES", cfg.samples)?;
        cfg.base_seed = parse_env_value("XY_BASE_SEED", cfg.base_seed)?;
        cfg.j_xy = parse_env_value("XY_JXY", cfg.j_xy)?;
        cfg.j_z_mean = parse_env_value("XY_JZ_MEAN", cfg.j_z_mean)?;
        cfg.sweeps = parse_env_value("XY_SWEEPS", cfg.sweeps)?;
        cfg.thermalization = parse_env_value("XY_THERMAL", cfg.thermalization)?;
        cfg.binsize = parse_env_value("XY_BINSIZE", cfg.binsize)?;
        cfg.proposal_width = parse_env_value("XY_PROPOSAL_WIDTH", cfg.proposal_width)?;
        cfg.wolff_steps = parse_env_value("XY_WOLFF_STEPS", cfg.wolff_steps)?;
        cfg.run_time = parse_env_duration("XY_RUN_TIME", cfg.run_time)?;
        cfg.checkpoint_time = parse_env_duration("XY_CHECKPOINT_TIME", cfg.checkpoint_time)?;
        cfg.correlation_rmax = parse_optional_env_value("XY_CORR_RMAX")?;
        cfg.correlation_rmax_xy =
            parse_optional_env_value("XY_CORR_RMAX_XY")?.or(cfg.correlation_rmax);
        cfg.correlation_rmax_z =
            parse_optional_env_value("XY_CORR_RMAX_Z")?.or(cfg.correlation_rmax);
        cfg.job_name = std::env::var("XY_JOB_NAME").unwrap_or_else(|_| {
            format!(
                "xy_carlo_L{}_dJxy{}_dJz{}",
                join_display(&cfg.l),
                join_display(&cfg.delta_j_xy),
                join_display(&cfg.delta_j_z)
            )
        });
        if parse_env_value("XY_RANKS_PER_RUN", 1usize)? != 1 {
            return Err(
                "XY_RANKS_PER_RUN must be 1; Rust theta job runner uses task-level parallelism"
                    .to_string(),
            );
        }
        Ok(cfg)
    }

    pub fn make_job(&self) -> Result<ThetaJob, String> {
        if self.samples == 0 {
            return Err("samples must be positive".to_string());
        }
        if self.binsize == 0 {
            return Err("binsize must be positive".to_string());
        }
        if self.sweeps < self.binsize {
            return Err("sweeps must be at least binsize".to_string());
        }
        let mut tasks = Vec::new();
        for (l_x, l_y, l_z, l) in self.lattice_specs() {
            for &delta_j_xy in &self.delta_j_xy {
                for &delta_j_z in &self.delta_j_z {
                    for &temperature in &self.temperatures {
                        for sample in 1..=self.samples {
                            let disorder_seed = self.base_seed + sample as u64 - 1;
                            let seed = self.base_seed
                                + 100_000 * l_z as u64
                                + 1_000 * sample as u64
                                + (100.0 * temperature).round() as u64
                                + (1_000.0 * delta_j_xy).round() as u64 * 10_000
                                + (1_000.0 * delta_j_z).round() as u64;
                            let mut disorder_rng = ChaCha8Rng::seed_from_u64(disorder_seed);
                            let j_xy_array = generate_layer_disorder_values(
                                l_z,
                                self.j_xy,
                                delta_j_xy,
                                &mut disorder_rng,
                                "J_xy",
                            )?;
                            let j_z_array = generate_layer_disorder_values(
                                l_z,
                                self.j_z_mean,
                                delta_j_z,
                                &mut disorder_rng,
                                "J_z",
                            )?;
                            tasks.push(ThetaTask {
                                name: format!(
                                    "L{}x{}x{}_T{:.6}_dJxy{:.6}_dJz{:.6}_sample{}",
                                    l_x, l_y, l_z, temperature, delta_j_xy, delta_j_z, sample
                                ),
                                l,
                                l_x,
                                l_y,
                                l_z,
                                temperature,
                                j_xy: self.j_xy,
                                delta_j_xy,
                                j_z_mean: self.j_z_mean,
                                delta_j_z,
                                disorder_seed,
                                seed,
                                sample,
                                sweeps: self.sweeps,
                                thermalization: self.thermalization,
                                binsize: self.binsize,
                                proposal_width: self.proposal_width,
                                wolff_steps: self.wolff_steps,
                                correlation_rmax: self.correlation_rmax.unwrap_or(l_z / 2),
                                correlation_rmax_xy: self
                                    .correlation_rmax_xy
                                    .unwrap_or_else(|| l_x.min(l_y) / 2),
                                correlation_rmax_z: self.correlation_rmax_z.unwrap_or(l_z / 2),
                                j_xy_array: Some(j_xy_array),
                                j_z_array: Some(j_z_array),
                            });
                        }
                    }
                }
            }
        }
        Ok(ThetaJob {
            name: self.job_name.clone(),
            tasks,
        })
    }

    fn lattice_specs(&self) -> Vec<(usize, usize, usize, usize)> {
        if self.l_x.is_none() && self.l_y.is_none() && self.l_z.is_none() {
            return self.l.iter().map(|&l| (l, l, l, l)).collect();
        }
        let l_x = self.l_x.as_deref().unwrap_or(&self.l);
        let l_y = self.l_y.as_deref().unwrap_or(&self.l);
        let l_z = self.l_z.as_deref().unwrap_or(&self.l);
        let mut specs = Vec::new();
        for &x in l_x {
            for &y in l_y {
                for &z in l_z {
                    specs.push((x, y, z, z));
                }
            }
        }
        specs
    }
}
