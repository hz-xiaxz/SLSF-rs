use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs;
use std::time::Duration;

use crate::*;

fn assert_err_eq<T>(result: Result<T, String>, expected: &str) {
    assert_eq!(result.err().as_deref(), Some(expected));
}

fn sorted_strings(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values
}

#[test]
fn theta_lattice_initialization_and_disorder() {
    assert!(ThetaLattice::new(0, 2, 2).is_err());
    assert!(ThetaLattice::new(2, 0, 2).is_err());
    assert!(ThetaLattice::new(2, 2, 0).is_err());

    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let mut lat = ThetaLattice::new(3, 4, 5).unwrap();
    assert_eq!(lat.theta.len(), 3 * 4 * 5);
    assert_eq!(lat.j_z.len(), 5);
    assert!(lat.theta.iter().all(|&v| v == 0.0));
    assert!(lat.j_z.iter().all(|&v| v == 0.0));

    let params = Parameters::new(1.0, 0.7, 0.0, 2.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    assert!(lat.j_z.iter().all(|&v| v == 0.7));

    let mut uniform_lat = ThetaLattice::new(1, 1, 200_000).unwrap();
    let uniform_params = Parameters::new(1.0, 1.0, 0.5, 2.0);
    initialize_disorder(&mut uniform_lat, &uniform_params, &mut rng).unwrap();
    assert!(uniform_lat.j_z.iter().all(|&v| (0.5..=1.5).contains(&v)));
    let disorder_mean = uniform_lat.j_z.iter().sum::<f64>() / uniform_lat.j_z.len() as f64;
    assert_abs_diff_eq!(disorder_mean, 1.0, epsilon = 5e-3);
    let variance = uniform_lat
        .j_z
        .iter()
        .map(|value| (value - disorder_mean).powi(2))
        .sum::<f64>()
        / (uniform_lat.j_z.len() - 1) as f64;
    assert_abs_diff_eq!(variance.sqrt(), 0.5 / 3.0_f64.sqrt(), epsilon = 5e-3);

    assert_err_eq(
        initialize_disorder(&mut lat, &Parameters::new(1.0, 1.0, -0.1, 2.0), &mut rng),
        "δJ_z must be nonnegative",
    );
    assert_err_eq(
        initialize_disorder(&mut lat, &Parameters::new(1.0, 0.05, 0.1, 2.0), &mut rng),
        "uniform layer disorder requires J_z_mean - δJ_z >= 0",
    );

    initialize_angles(&mut lat, InitMode::Cold, &mut rng).unwrap();
    assert!(lat.theta.iter().all(|&v| v == 0.0));
    initialize_angles(&mut lat, InitMode::Random, &mut rng).unwrap();
    assert!(lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));
}

#[test]
fn theta_energy_magnetization_and_correlations() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mut lat = ThetaLattice::new(2, 2, 2).unwrap();
    let params = Parameters::new(1.5, 0.5, 0.0, 1.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    initialize_angles(&mut lat, InitMode::Cold, &mut rng).unwrap();

    let expected_energy_density = -(2.0 * params.j_xy + params.j_z_mean);
    assert_abs_diff_eq!(
        measure_theta_energy(&lat, &params),
        expected_energy_density,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(measure_magnetization(&lat), 1.0, epsilon = 1e-12);
    let obs = measure_theta_observables(&lat, &params);
    let (cos_x, cos_y, cos_z, sin_x, sin_y, sin_z) = helicity_sums(&lat, &params);
    assert_abs_diff_eq!(
        obs.energy,
        measure_theta_energy(&lat, &params),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        obs.magnetization,
        measure_magnetization(&lat),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(obs.cos_x, cos_x, epsilon = 1e-12);
    assert_abs_diff_eq!(obs.cos_y, cos_y, epsilon = 1e-12);
    assert_abs_diff_eq!(obs.cos_z, cos_z, epsilon = 1e-12);
    assert_abs_diff_eq!(obs.sin_x, sin_x, epsilon = 1e-12);
    assert_abs_diff_eq!(obs.sin_y, sin_y, epsilon = 1e-12);
    assert_abs_diff_eq!(obs.sin_z, sin_z, epsilon = 1e-12);

    let corr = measure_theta_correlations(&lat, Some(1), None, None);
    assert_eq!(corr.r, vec![1]);
    assert_abs_diff_eq!(corr.corr_x[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(corr.corr_y[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(corr.corr_xy[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(corr.corr_z[0], 1.0, epsilon = 1e-12);

    lat.set(0, 0, 0, std::f64::consts::PI);
    let magnetization = measure_magnetization(&lat);
    assert!((0.0..=1.0).contains(&magnetization));
}

#[test]
fn theta_metropolis_updates_validate_and_keep_angles_wrapped() {
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let mut lat = ThetaLattice::new(3, 3, 3).unwrap();
    let params = Parameters::new(1.0, 0.8, 0.0, 2.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    initialize_angles(&mut lat, InitMode::Random, &mut rng).unwrap();

    let old_theta = lat.theta.clone();
    assert!(local_metropolis_step(&mut lat, &params, 0.0, &mut rng).unwrap());
    assert_eq!(lat.theta, old_theta);

    let acceptance = metropolis_sweep(&mut lat, &params, 0.5, &mut rng).unwrap();
    assert!((0.0..=1.0).contains(&acceptance));
    assert!(lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));
    assert_err_eq(
        local_metropolis_step(&mut lat, &params, -0.1, &mut rng),
        "proposal_width must be nonnegative and finite",
    );
    assert_err_eq(
        metropolis_sweep(&mut lat, &params, f64::INFINITY, &mut rng),
        "proposal_width must be nonnegative and finite",
    );

    let mut theta_scratch = ThetaScratch::new(&lat);
    let acceptance =
        metropolis_sweep_with_scratch(&mut lat, &params, &mut theta_scratch, 0.5, &mut rng)
            .unwrap();
    assert!((0.0..=1.0).contains(&acceptance));
}

#[test]
fn theta_temperature_validation_and_helicity() {
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let mut lat = ThetaLattice::new(3, 3, 3).unwrap();
    let params = Parameters::new(1.2, 0.6, 0.0, 0.5);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    initialize_angles(&mut lat, InitMode::Cold, &mut rng).unwrap();

    let (rho_x, rho_y, rho_z) = measure_helicity_modulus(&lat, &params).unwrap();
    assert_abs_diff_eq!(rho_x, params.j_xy, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_y, params.j_xy, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_z, params.j_z_mean, epsilon = 1e-12);

    let mut scratch = ThetaScratch::new(&lat);
    let mut wolff_scratch = WolffScratch::new(&lat);
    for bad_t in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let bad_params = Parameters::new(params.j_xy, params.j_z_mean, params.delta_j_z, bad_t);
        assert_err_eq(
            local_metropolis_step(&mut lat, &bad_params, std::f64::consts::PI, &mut rng),
            "temperature T must be positive and finite",
        );
        assert_err_eq(
            metropolis_sweep(&mut lat, &bad_params, std::f64::consts::PI, &mut rng),
            "temperature T must be positive and finite",
        );
        assert_err_eq(
            metropolis_sweep_with_scratch(
                &mut lat,
                &bad_params,
                &mut scratch,
                std::f64::consts::PI,
                &mut rng,
            ),
            "temperature T must be positive and finite",
        );
        assert_err_eq(
            wolff_cluster_step(&mut lat, &bad_params, &mut wolff_scratch, &mut rng),
            "temperature T must be positive and finite",
        );
        assert_err_eq(
            measure_helicity_modulus(&lat, &bad_params),
            "temperature T must be positive and finite",
        );
    }
}

#[test]
fn theta_simulation_driver() {
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    let mut lat = ThetaLattice::new(3, 3, 3).unwrap();
    let params = Parameters::new(1.0, 1.0, 0.0, 0.5);
    let options = ThetaSimulationOptions {
        thermal_sweeps: 5,
        measure_sweeps: 10,
        measure_interval: 2,
        proposal_width: 0.4,
        correlation_rmax: Some(1),
        init_mode: InitMode::Cold,
        ..Default::default()
    };
    let res = run_theta_simulation(&mut lat, &params, &options, &mut rng).unwrap();
    assert!(res.energy.is_finite());
    assert!(res.rho_sx.is_finite());
    assert!(res.rho_sy.is_finite());
    assert!(res.rho_sz.is_finite());
    assert!((0.0..=1.0).contains(&res.magnetization));
    assert_eq!(res.corr_r.len(), 1);
    assert_eq!(res.corr_z.len(), 1);
    assert_eq!(res.num_correlation_measurements, res.num_measurements);
    assert!(res.corr_z[0].is_finite());
    assert!((0.0..=1.0).contains(&res.acceptance));

    let mut default_lat = ThetaLattice::new(2, 2, 2).unwrap();
    let default_options = ThetaSimulationOptions {
        thermal_sweeps: 0,
        measure_sweeps: 1,
        measure_interval: 1,
        init_mode: InitMode::Cold,
        ..Default::default()
    };
    assert!(
        run_theta_simulation(&mut default_lat, &params, &default_options, &mut rng)
            .unwrap()
            .energy
            .is_finite()
    );

    let bad_interval = ThetaSimulationOptions {
        measure_interval: 0,
        ..Default::default()
    };
    assert_err_eq(
        run_theta_simulation(
            &mut ThetaLattice::new(2, 2, 2).unwrap(),
            &params,
            &bad_interval,
            &mut rng,
        ),
        "measure_interval must be positive",
    );
    let no_measurements = ThetaSimulationOptions {
        thermal_sweeps: 0,
        measure_sweeps: 1,
        measure_interval: 2,
        ..Default::default()
    };
    assert_err_eq(run_theta_simulation(&mut ThetaLattice::new(2, 2, 2).unwrap(), &params, &no_measurements, &mut rng), "measure_sweeps must include at least one measurement; require measure_sweeps >= measure_interval");
    let bad_corr_interval = ThetaSimulationOptions {
        correlation_interval: 0,
        ..Default::default()
    };
    assert_err_eq(
        run_theta_simulation(
            &mut ThetaLattice::new(2, 2, 2).unwrap(),
            &params,
            &bad_corr_interval,
            &mut rng,
        ),
        "correlation_interval must be positive",
    );
    let bad_width = ThetaSimulationOptions {
        thermal_sweeps: 0,
        measure_sweeps: 1,
        measure_interval: 1,
        proposal_width: f64::NAN,
        ..Default::default()
    };
    assert_err_eq(
        run_theta_simulation(
            &mut ThetaLattice::new(2, 2, 2).unwrap(),
            &params,
            &bad_width,
            &mut rng,
        ),
        "proposal_width must be nonnegative and finite",
    );

    let throttled_options = ThetaSimulationOptions {
        thermal_sweeps: 0,
        measure_sweeps: 5,
        measure_interval: 1,
        proposal_width: 0.0,
        wolff_steps: 0,
        correlation_rmax: Some(1),
        correlation_interval: 2,
        init_mode: InitMode::Cold,
        ..Default::default()
    };
    let throttled = run_theta_simulation(
        &mut ThetaLattice::new(2, 2, 2).unwrap(),
        &params,
        &throttled_options,
        &mut rng,
    )
    .unwrap();
    assert_eq!(throttled.num_measurements, 5);
    assert_eq!(throttled.num_correlation_measurements, 3);
    assert_abs_diff_eq!(throttled.corr_x[0], 1.0, epsilon = 1e-12);

    let bad_params = Parameters::new(1.0, -0.1, 0.0, 1.0);
    assert_err_eq(
        run_theta_simulation(
            &mut ThetaLattice::new(2, 2, 2).unwrap(),
            &bad_params,
            &ThetaSimulationOptions {
                thermal_sweeps: 0,
                measure_sweeps: 1,
                measure_interval: 1,
                ..Default::default()
            },
            &mut rng,
        ),
        "uniform layer disorder requires J_z_mean - δJ_z >= 0",
    );
}

#[test]
fn theta_wolff_cluster_update() {
    assert_abs_diff_eq!(
        wolff_add_probability(1.0, 1.0, 1.0, 1.0),
        1.0 - (-2.0_f64).exp(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        wolff_add_probability(1.0, 1.0, 1.0, -1.0),
        0.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        wolff_add_probability(1.0, 1.0, -1.0, 1.0),
        0.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        wolff_add_probability(1.0, 1.0, 0.0, 1.0),
        0.0,
        epsilon = 1e-12
    );
    let theta = 0.37;
    let phi = 1.20;
    let reflected = wolff_reflect_angle(theta, phi);
    let reflected_twice = wolff_reflect_angle(reflected, phi);
    assert_abs_diff_eq!(reflected_twice, theta, epsilon = 1e-12);
    assert!((0.0..TWO_PI).contains(&reflected));

    let mut rng = ChaCha8Rng::seed_from_u64(15);
    let mut lat = ThetaLattice::new(4, 4, 4).unwrap();
    let params = Parameters::new(1.0, 1.0, 0.0, 1.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    initialize_angles(&mut lat, InitMode::Random, &mut rng).unwrap();
    let mut scratch = WolffScratch::new(&lat);
    let mut theta_scratch = ThetaScratch::new(&lat);
    let cluster_size = wolff_cluster_step_with_theta_scratch(
        &mut lat,
        &params,
        &mut scratch,
        Some(&mut theta_scratch),
        &mut rng,
    )
    .unwrap();
    assert!((1..=4 * 4 * 4).contains(&cluster_size));
    assert!(lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));
}

#[test]
fn theta_carlo_entrypoint_job_config_and_binning() {
    let estimate = BinnedEstimate::from_samples(&[1.0, 3.0, 5.0, 7.0], 2).unwrap();
    assert_abs_diff_eq!(estimate.mean, 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(estimate.stderr, 2.0, epsilon = 1e-12);
    assert_eq!(estimate.bins, vec![2.0, 6.0]);
    assert_eq!(estimate.internal_bins, vec![2.0, 6.0]);
    assert_eq!(estimate.internal_bin_length, 2);
    assert_eq!(estimate.rebin_length, 1);
    assert_err_eq(
        BinnedEstimate::from_samples(&[1.0], 2),
        "binsize is larger than the sample series",
    );

    let cfg = ThetaJobConfig {
        l: vec![2],
        temperatures: vec![1.0, 1.5],
        delta_j_z: vec![0.0],
        samples: 2,
        base_seed: 17,
        sweeps: 4,
        thermalization: 1,
        binsize: 2,
        wolff_steps: 0,
        correlation_rmax: Some(0),
        job_name: "unit_theta_job".to_string(),
        ..Default::default()
    };
    let job = cfg.make_job().unwrap();
    assert_eq!(job.name, "unit_theta_job");
    assert_eq!(job.tasks.len(), 4);
    assert_eq!(job.tasks[0].l, 2);
    assert_eq!(job.tasks[0].l_x, 2);
    assert_eq!(job.tasks[0].l_y, 2);
    assert_eq!(job.tasks[0].l_z, 2);
    assert_eq!(job.tasks[0].binsize, 2);
    assert_eq!(job.tasks[0].j_z_array.as_ref().unwrap().len(), 2);

    let selected = job
        .selected_tasks(JobAssignment::new(1, 2).unwrap())
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    assert_eq!(selected, vec![1, 3]);

    let explicit_dims = ThetaJobConfig {
        l: vec![4],
        l_x: Some(vec![2]),
        l_y: Some(vec![3]),
        l_z: Some(vec![5]),
        temperatures: vec![1.0],
        delta_j_z: vec![0.0],
        samples: 1,
        sweeps: 2,
        binsize: 1,
        ..Default::default()
    }
    .make_job()
    .unwrap();
    assert_eq!(explicit_dims.tasks[0].l, 5);
    assert_eq!(explicit_dims.tasks[0].l_x, 2);
    assert_eq!(explicit_dims.tasks[0].l_y, 3);
    assert_eq!(explicit_dims.tasks[0].l_z, 5);

    let (toml_cfg, toml_options) = ThetaJobConfig::from_toml_spec(ThetaJobToml {
        name: Some("carlo_mpi_params".to_string()),
        run_time: Some("1-02:03:04".to_string()),
        checkpoint_time: Some("05:06".to_string()),
        checkpoint: Some(true),
        ..Default::default()
    });
    assert_eq!(toml_cfg.job_name, "carlo_mpi_params");
    assert_eq!(toml_cfg.run_time, std::time::Duration::from_secs(93_784));
    assert_eq!(
        toml_cfg.checkpoint_time,
        std::time::Duration::from_secs(306)
    );
    assert!(toml_options.checkpoint);
}

#[test]
fn theta_dynamic_scheduler_skips_completed_claims() {
    let task_a = ThetaTask {
        name: "dyn_a".to_string(),
        l: 2,
        l_x: 2,
        l_y: 2,
        l_z: 2,
        temperature: 1.0,
        j_xy: 1.0,
        j_z_mean: 1.0,
        delta_j_z: 0.0,
        disorder_seed: 123,
        seed: 456,
        sample: 1,
        sweeps: 4,
        thermalization: 1,
        binsize: 2,
        proposal_width: 0.0,
        wolff_steps: 0,
        correlation_rmax: 0,
        correlation_rmax_xy: 0,
        correlation_rmax_z: 0,
        j_z_array: Some(vec![1.0, 1.0]),
    };
    let mut task_b = task_a.clone();
    task_b.name = "dyn_b".to_string();
    task_b.temperature = 1.2;
    task_b.seed = 789;
    let job = ThetaJob {
        name: "dynamic_unit".to_string(),
        tasks: vec![task_a, task_b],
    };
    let dir = std::env::temp_dir().join(format!("slsf_theta_dynamic_test_{}", std::process::id()));
    let scheduler_dir = dir.join("scheduler");
    fs::create_dir_all(&scheduler_dir).unwrap();
    fs::File::create(scheduler_dir.join("task0000.done")).unwrap();

    let result = run_theta_job_dynamic(&job, &scheduler_dir, 0, 1).unwrap();
    assert_eq!(result.tasks.len(), 1);
    assert_eq!(result.tasks[0].task_index, 1);
    assert_eq!(result.tasks[0].task.name, "dyn_b");
    assert!(scheduler_dir.join("task0001.claim").exists());
    assert!(scheduler_dir.join("task0001.done").exists());

    fs::remove_file(scheduler_dir.join("task0000.done")).unwrap();
    fs::remove_file(scheduler_dir.join("task0001.claim")).unwrap();
    fs::remove_file(scheduler_dir.join("task0001.done")).unwrap();
    fs::remove_dir(scheduler_dir).unwrap();
    fs::remove_dir(dir).unwrap();
}

#[test]
fn theta_dynamic_scheduler_heartbeat_prevents_stale_reclaim() {
    let dir =
        std::env::temp_dir().join(format!("slsf_theta_heartbeat_test_{}", std::process::id()));
    let scheduler_dir = dir.join("scheduler");
    fs::create_dir_all(&scheduler_dir).unwrap();
    let claim_path = scheduler_dir.join("task0000.claim");
    let heartbeat_path = scheduler_dir.join("task0000.heartbeat");
    fs::write(&claim_path, "rank=999\n").unwrap();
    write_scheduler_heartbeat(&heartbeat_path, 0, 10, 20).unwrap();

    remove_stale_claim_if_needed(&scheduler_dir, 0, &claim_path, Duration::from_secs(60)).unwrap();

    assert!(claim_path.exists());
    assert!(heartbeat_path.exists());
    assert!(!fs::read_dir(&scheduler_dir).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains("claim.stale")));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn theta_checkpoint_writes_scheduler_heartbeat() {
    let dir = std::env::temp_dir().join(format!(
        "slsf_theta_checkpoint_heartbeat_test_{}",
        std::process::id()
    ));
    let checkpoint_path = dir.join("task0001/run0001.dump.h5");
    let heartbeat_path = dir.join("scheduler/task0000.heartbeat");
    let task = ThetaTask {
        name: "checkpoint_heartbeat".to_string(),
        l: 2,
        l_x: 2,
        l_y: 2,
        l_z: 2,
        temperature: 1.0,
        j_xy: 1.0,
        j_z_mean: 1.0,
        delta_j_z: 0.0,
        disorder_seed: 123,
        seed: 456,
        sample: 1,
        sweeps: 4,
        thermalization: 1,
        binsize: 2,
        proposal_width: 0.0,
        wolff_steps: 0,
        correlation_rmax: 0,
        correlation_rmax_xy: 0,
        correlation_rmax_z: 0,
        j_z_array: Some(vec![1.0, 1.0]),
    };
    let checkpoint = ThetaCheckpointRuntime {
        path: checkpoint_path.clone(),
        interval: Duration::ZERO,
        resume: false,
        heartbeat_path: Some(heartbeat_path.clone()),
    };

    let result = run_theta_task_with_checkpoint(&task, 0, Some(&checkpoint)).unwrap();
    assert_eq!(result.measurements, 4);
    assert!(checkpoint_path.exists());
    assert!(heartbeat_path.exists());
    let heartbeat = fs::read_to_string(&heartbeat_path).unwrap();
    assert!(heartbeat.contains("task_index=0"));
    assert!(heartbeat.contains("thermalization_sweeps=1"));
    assert!(heartbeat.contains("measurement_sweeps=4"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn theta_toml_config_expands_job_and_cli_runs() {
    let root =
        std::env::temp_dir().join(format!("slsf_theta_toml_cli_test_{}", std::process::id()));
    let output_dir = root.join("out");
    fs::create_dir_all(&root).unwrap();
    let config_path = root.join("theta.toml");
    let config = format!(
        r#"
name = "toml_unit"
output_dir = {:?}
checkpoint = true

[model]
L = [2]
T = [1.0, 1.2]
delta_j_z = [0.0]
samples = 1
base_seed = 41
j_xy = 1.0
j_z_mean = 1.0

[run]
sweeps = 4
thermalization = 1
binsize = 2
proposal_width = 0.0
wolff_steps = 0

[measure]
corr_rmax = 1
"#,
        output_dir.to_string_lossy()
    );
    fs::write(&config_path, config).unwrap();

    let (cfg, options) = ThetaJobConfig::from_toml_path(&config_path).unwrap();
    assert_eq!(cfg.job_name, "toml_unit");
    assert_eq!(cfg.l, vec![2]);
    assert_eq!(cfg.temperatures, vec![1.0, 1.2]);
    assert_eq!(cfg.delta_j_z, vec![0.0]);
    assert_eq!(cfg.samples, 1);
    assert_eq!(cfg.sweeps, 4);
    assert_eq!(cfg.thermalization, 1);
    assert_eq!(cfg.binsize, 2);
    assert_eq!(cfg.correlation_rmax, Some(1));
    assert_eq!(options.output_dir.as_deref(), Some(output_dir.as_path()));
    assert!(options.checkpoint);
    assert_eq!(cfg.make_job().unwrap().tasks.len(), 2);

    let message = run_theta_job_command([
        "slsf",
        "run-dynamic",
        "--config",
        config_path.to_str().unwrap(),
    ])
    .unwrap();
    assert!(message.contains("dynamically completed 2 theta task(s)"));
    assert!(output_dir.join("toml_unit.rank0.results.json").exists());
    assert!(output_dir
        .join("toml_unit.data/task0001/run0001.meas.h5")
        .exists());
    assert!(output_dir
        .join("toml_unit.data/task0001/run0001.dump.h5")
        .exists());
    assert!(output_dir
        .join("toml_unit.data/scheduler/task0000.done")
        .exists());
    assert!(output_dir
        .join("toml_unit.data/scheduler/task0001.done")
        .exists());

    let status =
        run_theta_job_command(["slsf", "status", "--config", config_path.to_str().unwrap()])
            .unwrap();
    assert!(status.contains("2 of 2 theta task(s) marked done"));

    let delete_message =
        run_theta_job_command(["slsf", "delete", "--config", config_path.to_str().unwrap()])
            .unwrap();
    assert_eq!(delete_message, "deleted theta job outputs");
    assert!(!output_dir.join("toml_unit.data").exists());
    assert!(!output_dir.join("toml_unit.results.json").exists());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn theta_carlo_entrypoint_runs_task_and_roundtrips_result_json() {
    let task = ThetaTask {
        name: "tiny".to_string(),
        l: 2,
        l_x: 2,
        l_y: 2,
        l_z: 2,
        temperature: 1.0,
        j_xy: 1.0,
        j_z_mean: 1.0,
        delta_j_z: 0.0,
        disorder_seed: 23,
        seed: 24,
        sample: 1,
        sweeps: 4,
        thermalization: 1,
        binsize: 2,
        proposal_width: 0.0,
        wolff_steps: 0,
        correlation_rmax: 1,
        correlation_rmax_xy: 1,
        correlation_rmax_z: 1,
        j_z_array: Some(vec![1.0, 1.0]),
    };
    let job = ThetaJob {
        name: "tiny_job".to_string(),
        tasks: vec![task],
    };
    let result = run_theta_job(&job, JobAssignment::single()).unwrap();
    assert_eq!(result.job_name, "tiny_job");
    assert_eq!(result.rank, 0);
    assert_eq!(result.world_size, 1);
    assert_eq!(result.tasks.len(), 1);
    let task_result = &result.tasks[0];
    assert_eq!(task_result.measurements, 4);
    assert!(task_result.observables["Energy"].mean.is_finite());
    assert!(task_result.observables["RhoXY"].mean.is_finite());
    assert!(task_result.observables["RhoZ"].mean.is_finite());
    assert!(task_result.observables["_ll_sweep_time"].mean.is_finite());
    assert!(task_result.observables["_ll_measure_time"].mean.is_finite());
    assert_eq!(task_result.observables["Energy"].bins, 2);
    assert_eq!(task_result.observables["Energy"].bin_length, 2);
    assert!(task_result.observables["CorrXY_r1"].mean.is_finite());

    let out_dir = std::env::temp_dir().join(format!("slsf_theta_job_test_{}", std::process::id()));
    let measurement_paths = write_theta_job_measurements(&result, &out_dir).unwrap();
    assert_eq!(measurement_paths.len(), 1);
    assert_eq!(
        measurement_paths[0]
            .file_name()
            .and_then(|name| name.to_str()),
        Some("run0001.meas.h5")
    );
    let h5 = hdf5_pure::File::from_bytes(std::fs::read(&measurement_paths[0]).unwrap()).unwrap();
    let root_group = h5.group("/").unwrap();
    assert_eq!(
        sorted_strings(root_group.groups().unwrap()),
        vec!["observables".to_string(), "version".to_string()]
    );
    assert_eq!(root_group.datasets().unwrap(), Vec::<String>::new());
    let observables_group = h5.group("observables").unwrap();
    assert_eq!(
        sorted_strings(observables_group.groups().unwrap()),
        result.tasks[0]
            .measurement_bins
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    );
    assert_eq!(observables_group.datasets().unwrap(), Vec::<String>::new());
    for (name, samples) in &result.tasks[0].measurement_bins {
        let observable_group = observables_group.group(name).unwrap();
        assert_eq!(observable_group.groups().unwrap(), Vec::<String>::new());
        assert_eq!(
            sorted_strings(observable_group.datasets().unwrap()),
            vec!["bin_length".to_string(), "samples".to_string()]
        );

        let bin_length = observable_group.dataset("bin_length").unwrap();
        assert_eq!(bin_length.shape().unwrap(), Vec::<u64>::new());
        assert_eq!(bin_length.dtype().unwrap().to_string(), "i64");
        assert_eq!(
            bin_length.read_i64().unwrap(),
            vec![result.tasks[0].observables[name].internal_bin_len as i64]
        );

        let sample_dataset = observable_group.dataset("samples").unwrap();
        assert_eq!(sample_dataset.shape().unwrap(), vec![samples.len() as u64]);
        assert_eq!(sample_dataset.dtype().unwrap().to_string(), "f64");
        assert_eq!(sample_dataset.read_f64().unwrap(), *samples);
    }

    let version_group = h5.group("version").unwrap();
    assert_eq!(version_group.groups().unwrap(), Vec::<String>::new());
    assert_eq!(
        sorted_strings(version_group.datasets().unwrap()),
        vec![
            "carlo_version".to_string(),
            "mc_package".to_string(),
            "mc_version".to_string()
        ]
    );
    for name in ["carlo_version", "mc_package", "mc_version"] {
        let dataset = version_group.dataset(name).unwrap();
        assert_eq!(dataset.shape().unwrap(), Vec::<u64>::new());
        assert_eq!(dataset.dtype().unwrap().to_string(), "string");
    }
    assert_eq!(
        version_group
            .dataset("carlo_version")
            .unwrap()
            .read_string()
            .unwrap(),
        vec!["0.3.4".to_string()]
    );
    assert_eq!(
        version_group
            .dataset("mc_package")
            .unwrap()
            .read_string()
            .unwrap(),
        vec!["SLSF.XYCarlo".to_string()]
    );
    assert_eq!(
        version_group
            .dataset("mc_version")
            .unwrap()
            .read_string()
            .unwrap(),
        vec![env!("CARGO_PKG_VERSION").to_string()]
    );

    let checkpoint_paths = write_theta_job_checkpoints(&result, &out_dir).unwrap();
    assert_eq!(checkpoint_paths.len(), 1);
    assert_eq!(
        checkpoint_paths[0]
            .file_name()
            .and_then(|name| name.to_str()),
        Some("run0001.dump.h5")
    );
    let checkpoint =
        hdf5_pure::File::from_bytes(std::fs::read(&checkpoint_paths[0]).unwrap()).unwrap();
    let checkpoint_root = checkpoint.group("/").unwrap();
    assert_eq!(
        sorted_strings(checkpoint_root.groups().unwrap()),
        vec![
            "contexts".to_string(),
            "measurements".to_string(),
            "metadata".to_string(),
            "parameters".to_string(),
            "progress".to_string(),
            "state".to_string(),
            "version".to_string()
        ]
    );
    let simulation_group = checkpoint
        .group("contexts/rank0000/simulation")
        .expect("checkpoint simulation group");
    assert_eq!(
        sorted_strings(simulation_group.datasets().unwrap()),
        vec![
            "measurements".to_string(),
            "task".to_string(),
            "task_index".to_string()
        ]
    );
    assert_eq!(
        simulation_group
            .dataset("measurements")
            .unwrap()
            .read_i64()
            .unwrap(),
        vec![4]
    );
    assert_eq!(
        checkpoint
            .group("parameters")
            .unwrap()
            .dataset("T")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![result.tasks[0].task.temperature]
    );
    assert_eq!(
        checkpoint
            .group("parameters")
            .unwrap()
            .dataset("J_z")
            .unwrap()
            .read_f64()
            .unwrap()
            .len(),
        result.tasks[0].task.l_z
    );
    assert_eq!(
        checkpoint
            .group("state")
            .unwrap()
            .dataset("theta")
            .unwrap()
            .read_f64()
            .unwrap()
            .len(),
        result.tasks[0].task.l_x * result.tasks[0].task.l_y * result.tasks[0].task.l_z
    );
    assert_eq!(
        checkpoint
            .group("progress")
            .unwrap()
            .dataset("measurement_sweeps")
            .unwrap()
            .read_i64()
            .unwrap(),
        vec![4]
    );

    let path = write_theta_job_result(&result, &out_dir).unwrap();
    let restored = read_theta_job_result(&path).unwrap();
    assert_eq!(restored.job_name, result.job_name);
    assert_eq!(restored.rank, result.rank);
    assert_eq!(restored.world_size, result.world_size);
    assert_eq!(restored.tasks.len(), result.tasks.len());
    assert_eq!(restored.tasks[0].task, result.tasks[0].task);
    assert_abs_diff_eq!(
        restored.tasks[0].observables["Energy"].mean,
        result.tasks[0].observables["Energy"].mean,
        epsilon = 1e-12
    );
    assert_eq!(
        restored.tasks[0].observables["Energy"].bins,
        result.tasks[0].observables["Energy"].bins
    );
    assert_eq!(
        restored.tasks[0].observables["Energy"].bin_length,
        result.tasks[0].observables["Energy"].bin_length
    );

    let mut rank1 = restored.clone();
    rank1.rank = 1;
    rank1.tasks[0].task.name = "tiny_rank1".to_string();
    let rank1_path = out_dir.join("tiny_job.rank1.results.json");
    write_theta_job_result_to_path(&rank1, &rank1_path).unwrap();
    let merged = merge_theta_job_result_files([&path, &rank1_path]).unwrap();
    assert_eq!(merged.rank, 0);
    assert_eq!(merged.world_size, 1);
    assert_eq!(merged.tasks.len(), 2);
    assert_eq!(merged.tasks[0].task.name, "tiny");
    assert_eq!(merged.tasks[1].task.name, "tiny_rank1");

    let table_path = out_dir.join("tiny_job.tsv");
    let dataframe_message = run_theta_job_command([
        "slsf",
        "dataframe",
        path.to_str().unwrap(),
        "--output",
        table_path.to_str().unwrap(),
    ])
    .unwrap();
    assert!(dataframe_message.contains("wrote"));
    let table = std::fs::read_to_string(&table_path).unwrap();
    assert!(table.contains("job_name\trank\tworld_size\ttask_index\ttask_name\tL\tLx\tLy\tLz\tT"));
    assert!(table.contains("Energy\tEnergy_error\tEnergy_measurement"));
    assert!(table.contains("tiny_job\t0\t1\t0\ttiny\t2\t2\t2\t2\t1.0000000000000000"));

    let script_path = out_dir.join("tiny_job.gnuplot");
    let plot_path = out_dir.join("tiny_job.png");
    write_gnuplot_script(&table_path, &script_path, &plot_path, "Energy").unwrap();
    let script = std::fs::read_to_string(&script_path).unwrap();
    assert!(script.contains("set datafile separator '\\t'"));
    assert!(script.contains("using 'T':'Energy':'Lx'"));

    let mpi_root =
        std::env::temp_dir().join(format!("slsf_theta_mpi_cli_test_{}", std::process::id()));
    let mpi_out = mpi_root.join("out");
    fs::create_dir_all(&mpi_root).unwrap();
    let mpi_config_path = mpi_root.join("theta.toml");
    let mpi_config = format!(
        r#"
name = "mpi_unit"
output_dir = {:?}
run_time = "00:01"
checkpoint_time = "00:01"

[model]
L = [2]
T = [1.0, 1.2]
delta_j_z = [0.0]
samples = 1
base_seed = 43

[run]
sweeps = 4
thermalization = 1
binsize = 2
proposal_width = 0.0
wolff_steps = 0

[measure]
corr_rmax = 0
"#,
        mpi_out.to_string_lossy()
    );
    fs::write(&mpi_config_path, mpi_config).unwrap();

    let mpi_rank0 = run_theta_job_command([
        "slsf",
        "mpi-run",
        "--config",
        mpi_config_path.to_str().unwrap(),
        "--single",
        "--restart",
    ])
    .unwrap();
    assert!(mpi_rank0.contains("MPI rank 0/1 completed 2 theta task(s)"));
    assert!(mpi_out.join("mpi_unit.rank0.results.json").exists());

    let short_alias = run_theta_job_command([
        "slsf",
        "s",
        "--config",
        mpi_config_path.to_str().unwrap(),
        "--single",
    ])
    .unwrap();
    assert!(short_alias.contains("2 of 2 theta task(s) marked done"));

    fs::remove_dir_all(mpi_root).unwrap();

    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(rank1_path).unwrap();
    std::fs::remove_file(table_path).unwrap();
    std::fs::remove_file(script_path).unwrap();
    std::fs::remove_file(&measurement_paths[0]).unwrap();
    std::fs::remove_file(&checkpoint_paths[0]).unwrap();
    std::fs::remove_dir(out_dir.join("task0001")).unwrap();
    std::fs::remove_dir(out_dir).unwrap();
}
