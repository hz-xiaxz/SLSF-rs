use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use std::fs;

use crate::*;

fn assert_err_eq<T>(result: Result<T, String>, expected: &str) {
    assert_eq!(result.err().as_deref(), Some(expected));
}

#[test]
fn fast_math_x8_matches_scalar_regression() {
    let input = [-2.0, -1.25, -0.5, -0.125, 0.0, 0.125, 0.5, 1.75];

    let (simd_sin, simd_cos) = crate::fast_math::sin_cos_x8(input);
    let simd_exp = crate::fast_math::exp_x8(input);

    for (lane, &value) in input.iter().enumerate() {
        assert_abs_diff_eq!(simd_sin[lane], value.sin(), epsilon = 1e-14);
        assert_abs_diff_eq!(simd_cos[lane], value.cos(), epsilon = 1e-14);
        assert_abs_diff_eq!(simd_exp[lane], value.exp(), epsilon = 1e-14);
    }
}

#[test]
fn metropolis_batch_acceptance_probabilities_match_scalar() {
    let delta = [-2.0, 0.0, 0.125, 0.5, 1.0, 1.5, 2.0, 3.0];
    let beta = 0.75;
    let probability =
        crate::updates::metropolis_accept_probabilities(delta, beta, crate::fast_math::exp_x8);

    for (lane, &energy_delta) in delta.iter().enumerate() {
        let expected = if energy_delta <= 0.0 {
            1.0
        } else {
            (-energy_delta * beta).exp()
        };
        assert_abs_diff_eq!(probability[lane], expected, epsilon = 1e-14);
    }

    assert_eq!(crate::updates::vector_exp_min_uphill::<4>(), 2);
    assert_eq!(crate::updates::vector_exp_min_uphill::<8>(), 3);
}

#[test]
fn theta_lattice_initialization_and_disorder() {
    assert!(ThetaLattice::new(0, 2, 2).is_err());
    assert!(ThetaLattice::new(2, 0, 2).is_err());
    assert!(ThetaLattice::new(2, 2, 0).is_err());

    let mut rng = FastRng::seed_from_u64(12345);
    let mut lat = ThetaLattice::new(3, 4, 5).unwrap();
    assert_eq!(lat.theta.len(), 3 * 4 * 5);
    assert_eq!(lat.j_xy.len(), 5);
    assert_eq!(lat.j_z.len(), 5);
    assert!(lat.theta.iter().all(|&v| v == 0.0));
    assert!(lat.j_xy.iter().all(|&v| v == 0.0));
    assert!(lat.j_z.iter().all(|&v| v == 0.0));

    let params = Parameters::new(1.0, 0.7, 0.0, 2.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    assert!(lat.j_z.iter().all(|&v| v == 0.7));

    let mut uniform_lat = ThetaLattice::new(1, 1, 200_000).unwrap();
    let uniform_params = Parameters::new(1.0, 1.0, 0.5, 2.0);
    initialize_disorder(&mut uniform_lat, &uniform_params, &mut rng).unwrap();
    assert!(uniform_lat.j_z.iter().all(|&v| v == 0.5 || v == 1.5));
    let disorder_mean = uniform_lat.j_z.iter().sum::<f64>() / uniform_lat.j_z.len() as f64;
    assert_abs_diff_eq!(disorder_mean, 1.0, epsilon = 5e-3);
    let variance = uniform_lat
        .j_z
        .iter()
        .map(|value| (value - disorder_mean).powi(2))
        .sum::<f64>()
        / (uniform_lat.j_z.len() - 1) as f64;
    assert_abs_diff_eq!(variance.sqrt(), 0.5, epsilon = 5e-3);

    assert_err_eq(
        initialize_disorder(&mut lat, &Parameters::new(1.0, 1.0, -0.1, 2.0), &mut rng),
        "δJ_z must be nonnegative",
    );
    assert_err_eq(
        initialize_disorder(&mut lat, &Parameters::new(1.0, 0.05, 0.1, 2.0), &mut rng),
        "two-point layer disorder requires J_z_mean - δJ_z >= 0",
    );

    initialize_angles(&mut lat, InitMode::Cold, &mut rng).unwrap();
    assert!(lat.theta.iter().all(|&v| v == 0.0));
    initialize_angles(&mut lat, InitMode::Random, &mut rng).unwrap();
    assert!(lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));
}

#[test]
fn theta_energy_magnetization_and_correlations() {
    let mut rng = FastRng::seed_from_u64(11);
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
    assert_abs_diff_eq!(obs.magnetization_squared, 1.0, epsilon = 1e-12);
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
    assert_eq!(corr.corr_xy_by_z.len(), lat.l_z);
    assert!(corr
        .corr_xy_by_z
        .iter()
        .all(|layer| layer.len() == 1 && (layer[0] - 1.0).abs() < 1e-12));
    assert_abs_diff_eq!(corr.corr_z[0], 1.0, epsilon = 1e-12);

    for y in 0..lat.l_y {
        for x in 0..lat.l_x {
            lat.set(x, y, 1, if (x + y) % 2 == 0 { -1.0 } else { 1.0 });
        }
    }
    let layer_corr = measure_theta_correlations(&lat, Some(1), None, None);
    assert!((layer_corr.corr_xy_by_z[0][0] - 1.0).abs() < 1e-12);
    assert!(layer_corr.corr_xy_by_z[1][0] < 0.0);
    assert!(
        (layer_corr.corr_xy[0]
            - (layer_corr.corr_xy_by_z[0][0] + layer_corr.corr_xy_by_z[1][0]) / 2.0)
            .abs()
            < 1e-12
    );

    lat.set(0, 0, 0, std::f64::consts::PI);
    let magnetization = measure_magnetization(&lat);
    assert!((0.0..=1.0).contains(&magnetization));
}

#[test]
fn theta_metropolis_updates_validate_and_keep_angles_wrapped() {
    let mut rng = FastRng::seed_from_u64(12);
    let mut lat = ThetaLattice::new(3, 3, 3).unwrap();
    let params = Parameters::new(1.0, 0.8, 0.0, 2.0);
    initialize_disorder(&mut lat, &params, &mut rng).unwrap();
    initialize_angles(&mut lat, InitMode::Random, &mut rng).unwrap();

    let old_theta = lat.theta.clone();
    assert!(local_metropolis_step(&mut lat, &params, 0.0, &mut rng).unwrap());
    assert_eq!(lat.theta, old_theta);

    assert_err_eq(
        metropolis_sweep(&mut lat, &params, 0.5, &mut rng),
        "red-black Metropolis sweep requires even lattice dimensions",
    );
    assert!(lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));
    assert_err_eq(
        local_metropolis_step(&mut lat, &params, -0.1, &mut rng),
        "proposal_width must be nonnegative and finite",
    );
    assert_err_eq(
        metropolis_sweep(&mut lat, &params, f64::INFINITY, &mut rng),
        "proposal_width must be nonnegative and finite",
    );

    let mut even_lat = ThetaLattice::new(4, 4, 4).unwrap();
    initialize_disorder(&mut even_lat, &params, &mut rng).unwrap();
    initialize_angles(&mut even_lat, InitMode::Random, &mut rng).unwrap();
    let acceptance = metropolis_sweep(&mut even_lat, &params, 0.5, &mut rng).unwrap();
    assert!((0.0..=1.0).contains(&acceptance));
    assert!(even_lat.theta.iter().all(|&v| (0.0..TWO_PI).contains(&v)));

    let mut theta_scratch = ThetaScratch::new(&even_lat);
    let acceptance =
        metropolis_sweep_with_scratch(&mut even_lat, &params, &mut theta_scratch, 0.5, &mut rng)
            .unwrap();
    assert!((0.0..=1.0).contains(&acceptance));
}

#[test]
fn theta_temperature_validation_and_helicity() {
    let mut rng = FastRng::seed_from_u64(13);
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
    let mut rng = FastRng::seed_from_u64(14);
    let mut lat = ThetaLattice::new(4, 4, 4).unwrap();
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
    assert!((0.0..=1.0).contains(&res.magnetization_squared));
    assert!(res.chi.is_finite());
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
        "two-point layer disorder requires J_z_mean - δJ_z >= 0",
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

    let mut rng = FastRng::seed_from_u64(15);
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
    assert!(scratch.cluster_sites.is_empty());
    assert!(scratch.in_cluster.iter().all(|&in_cluster| !in_cluster));

    let second_cluster_size = wolff_cluster_step_with_theta_scratch(
        &mut lat,
        &params,
        &mut scratch,
        Some(&mut theta_scratch),
        &mut rng,
    )
    .unwrap();
    assert!((1..=4 * 4 * 4).contains(&second_cluster_size));
    assert!(scratch.cluster_sites.is_empty());
    assert!(scratch.in_cluster.iter().all(|&in_cluster| !in_cluster));
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
        correlation_rmax: Some(vec![0]),
        correlation_interval: 2,
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
    assert_eq!(job.tasks[0].correlation_interval, 2);
    assert_eq!(job.tasks[0].j_xy_array.as_ref().unwrap().len(), 2);
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

    let (toml_measure_cfg, _) = ThetaJobConfig::from_toml_spec(ThetaJobToml {
        measure: Some(ThetaMeasureToml {
            corr_rmax: Some(vec![1, 2]),
            corr_interval: Some(3),
            ..Default::default()
        }),
        ..Default::default()
    });
    assert_eq!(toml_measure_cfg.correlation_rmax, Some(vec![1, 2]));
    assert_eq!(toml_measure_cfg.correlation_interval, 3);

    let correlated_job = ThetaJobConfig {
        l: vec![16, 32],
        temperatures: vec![1.0],
        delta_j_z: vec![0.0],
        samples: 1,
        sweeps: 2,
        binsize: 1,
        correlation_rmax: Some(vec![8, 16]),
        correlation_rmax_xy: Some(vec![7, 15]),
        correlation_rmax_z: Some(vec![6, 14]),
        ..Default::default()
    }
    .make_job()
    .unwrap();
    assert_eq!(correlated_job.tasks.len(), 2);
    assert_eq!(correlated_job.tasks[0].l, 16);
    assert_eq!(correlated_job.tasks[0].correlation_rmax, 8);
    assert_eq!(correlated_job.tasks[0].correlation_rmax_xy, 7);
    assert_eq!(correlated_job.tasks[0].correlation_rmax_z, 6);
    assert_eq!(correlated_job.tasks[1].l, 32);
    assert_eq!(correlated_job.tasks[1].correlation_rmax, 16);
    assert_eq!(correlated_job.tasks[1].correlation_rmax_xy, 15);
    assert_eq!(correlated_job.tasks[1].correlation_rmax_z, 14);

    assert_err_eq(
        ThetaJobConfig {
            l: vec![16, 32],
            correlation_rmax_xy: Some(vec![16]),
            ..Default::default()
        }
        .make_job(),
        "corr_rmax_xy must contain exactly one value per lattice size: expected 2, got 1",
    );

    let invalid_duration = ThetaJobConfig::try_from_toml_spec(ThetaJobToml {
        run_time: Some("48:invalid:00".to_string()),
        ..Default::default()
    })
    .unwrap_err();
    assert!(invalid_duration.contains("invalid run_time"));
}

#[test]
fn theta_carlo_runner_writes_master_compatible_result() {
    let root = std::env::temp_dir().join(format!("slsf_carlo_test_{}", std::process::id()));
    let output_dir = root.join("out");
    fs::create_dir_all(&root).unwrap();
    let cfg = ThetaJobConfig {
        l: vec![2],
        temperatures: vec![2.0],
        delta_j_z: vec![0.0],
        samples: 1,
        sweeps: 4,
        thermalization: 2,
        binsize: 2,
        wolff_steps: 1,
        correlation_rmax: Some(vec![1]),
        job_name: "carlo_unit".to_string(),
        ..Default::default()
    };
    let options = ThetaRunOptions {
        output_dir: Some(output_dir.clone()),
        ..Default::default()
    };
    let summary = run_theta_job_with_carlo(&cfg, options).unwrap();
    assert_eq!(summary.task_count, 1);
    assert!(!summary.stopped_early);

    let result = read_theta_job_result(&summary.output_path).unwrap();
    assert_eq!(result.job_name, "carlo_unit");
    assert_eq!(result.tasks.len(), 1);
    let task = &result.tasks[0];
    assert_eq!(task.measurements, 4);
    assert!(task.acceptance.is_finite());
    assert!(task.observables.contains_key("Energy"));
    assert!(task.observables.contains_key("RhoXY"));
    assert!(task.observables.contains_key("RhoZ"));
    assert!(task.observables.contains_key("Magnetization"));
    assert!(task.observables["Energy"].mean.is_finite());
    assert!(task.observables["RhoXY"].mean.is_finite());
    assert!(output_dir.join("carlo_unit.data/task0001/run0001.meas.h5").exists());

    fs::remove_dir_all(root).unwrap();
}
