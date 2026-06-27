use rand::Rng;

use crate::autocorrelation::{blocking_stderr, mean};
use crate::initialization::{initialize_angles, initialize_disorder};
use crate::observables::{measure_theta_correlations, measure_theta_observables};
use crate::types::{
    validate_proposal_width, validate_temperature, Parameters, ThetaLattice, ThetaScratch,
    ThetaSimulationOptions, ThetaSimulationResult, WolffScratch,
};
use crate::updates::{metropolis_sweep_with_scratch, wolff_cluster_step_with_theta_scratch};

pub fn run_theta_simulation<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    options: &ThetaSimulationOptions,
    rng: &mut R,
) -> Result<ThetaSimulationResult, String> {
    if options.measure_interval == 0 {
        return Err("measure_interval must be positive".to_string());
    }
    if options.correlation_interval == 0 {
        return Err("correlation_interval must be positive".to_string());
    }
    let num_measurements = options.measure_sweeps / options.measure_interval;
    if num_measurements == 0 {
        return Err("measure_sweeps must include at least one measurement; require measure_sweeps >= measure_interval".to_string());
    }
    validate_temperature(params)?;
    let width = validate_proposal_width(options.proposal_width)?;
    let corr_xy_requested = options.correlation_rmax_xy.unwrap_or_else(|| {
        options
            .correlation_rmax
            .unwrap_or_else(|| (lattice.l_x / 2).min(lattice.l_y / 2))
    });
    let corr_z_requested = options
        .correlation_rmax_z
        .unwrap_or_else(|| options.correlation_rmax.unwrap_or(lattice.l_z / 2));
    let corr_rmax_xy_eff = corr_xy_requested.min(lattice.l_x / 2).min(lattice.l_y / 2);
    let corr_rmax_z_eff = corr_z_requested.min(lattice.l_z / 2);

    initialize_disorder(lattice, params, rng)?;
    if lattice.j_z.iter().any(|&j| j < 0.0) {
        return Err(
            "theta simulation requires nonnegative J_z; got negative layer coupling".to_string(),
        );
    }
    initialize_angles(lattice, options.init_mode, rng)?;

    let mut acceptance_sum = 0.0;
    let mut acceptance_count = 0usize;
    let mut theta_scratch = ThetaScratch::new(lattice);
    let mut wolff_scratch = WolffScratch::new(lattice);

    for _ in 0..options.thermal_sweeps {
        acceptance_sum +=
            metropolis_sweep_with_scratch(lattice, params, &mut theta_scratch, width, rng)?;
        acceptance_count += 1;
        for _ in 0..options.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                lattice,
                params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                rng,
            )?;
        }
    }

    let mut energy = Vec::with_capacity(num_measurements);
    let mut cos_x = Vec::with_capacity(num_measurements);
    let mut cos_y = Vec::with_capacity(num_measurements);
    let mut cos_z = Vec::with_capacity(num_measurements);
    let mut sin_x = Vec::with_capacity(num_measurements);
    let mut sin_y = Vec::with_capacity(num_measurements);
    let mut sin_z = Vec::with_capacity(num_measurements);
    let mut magnetization = Vec::with_capacity(num_measurements);
    let mut corr_x_sum = vec![0.0; corr_rmax_xy_eff];
    let mut corr_y_sum = vec![0.0; corr_rmax_xy_eff];
    let mut corr_xy_sum = vec![0.0; corr_rmax_xy_eff];
    let mut corr_z_sum = vec![0.0; corr_rmax_z_eff];
    let mut corr_measurements = 0usize;

    for step in 1..=options.measure_sweeps {
        acceptance_sum +=
            metropolis_sweep_with_scratch(lattice, params, &mut theta_scratch, width, rng)?;
        acceptance_count += 1;
        for _ in 0..options.wolff_steps {
            wolff_cluster_step_with_theta_scratch(
                lattice,
                params,
                &mut wolff_scratch,
                Some(&mut theta_scratch),
                rng,
            )?;
        }

        if step % options.measure_interval == 0 {
            let obs = measure_theta_observables(lattice, params);
            energy.push(obs.energy);
            cos_x.push(obs.cos_x);
            cos_y.push(obs.cos_y);
            cos_z.push(obs.cos_z);
            sin_x.push(obs.sin_x);
            sin_y.push(obs.sin_y);
            sin_z.push(obs.sin_z);
            magnetization.push(obs.magnetization);
            let idx = energy.len();
            if (corr_rmax_xy_eff > 0 || corr_rmax_z_eff > 0)
                && (idx - 1) % options.correlation_interval == 0
            {
                let corr = measure_theta_correlations(
                    lattice,
                    None,
                    Some(corr_rmax_xy_eff),
                    Some(corr_rmax_z_eff),
                );
                for i in 0..corr_rmax_xy_eff {
                    corr_x_sum[i] += corr.corr_x[i];
                    corr_y_sum[i] += corr.corr_y[i];
                    corr_xy_sum[i] += corr.corr_xy[i];
                }
                for i in 0..corr_rmax_z_eff {
                    corr_z_sum[i] += corr.corr_z[i];
                }
                corr_measurements += 1;
            }
        }
    }

    let volume = lattice.volume() as f64;
    let beta = 1.0 / params.temperature;
    let mean_sin_x = mean(&sin_x);
    let mean_sin_y = mean(&sin_y);
    let mean_sin_z = mean(&sin_z);
    let rho_sx = mean(&cos_x) / volume - beta * (mean_square(&sin_x) - mean_sin_x.powi(2)) / volume;
    let rho_sy = mean(&cos_y) / volume - beta * (mean_square(&sin_y) - mean_sin_y.powi(2)) / volume;
    let rho_sz = mean(&cos_z) / volume - beta * (mean_square(&sin_z) - mean_sin_z.powi(2)) / volume;

    let eff_rho_sx = cos_x
        .iter()
        .zip(&sin_x)
        .map(|(c, s)| c / volume - beta * (s.powi(2) - mean_sin_x * s) / volume)
        .collect::<Vec<_>>();
    let eff_rho_sy = cos_y
        .iter()
        .zip(&sin_y)
        .map(|(c, s)| c / volume - beta * (s.powi(2) - mean_sin_y * s) / volume)
        .collect::<Vec<_>>();
    let eff_rho_sz = cos_z
        .iter()
        .zip(&sin_z)
        .map(|(c, s)| c / volume - beta * (s.powi(2) - mean_sin_z * s) / volume)
        .collect::<Vec<_>>();
    let corr_norm = if corr_measurements > 0 {
        corr_measurements as f64
    } else {
        1.0
    };

    Ok(ThetaSimulationResult {
        energy: mean(&energy),
        rho_sx,
        rho_sy,
        rho_sz,
        std_rho_sx: blocking_stderr(&eff_rho_sx),
        std_rho_sy: blocking_stderr(&eff_rho_sy),
        std_rho_sz: blocking_stderr(&eff_rho_sz),
        magnetization: mean(&magnetization),
        corr_r: (1..=corr_rmax_xy_eff).collect(),
        corr_r_xy: (1..=corr_rmax_xy_eff).collect(),
        corr_r_z: (1..=corr_rmax_z_eff).collect(),
        corr_x: corr_x_sum.into_iter().map(|v| v / corr_norm).collect(),
        corr_y: corr_y_sum.into_iter().map(|v| v / corr_norm).collect(),
        corr_xy: corr_xy_sum.into_iter().map(|v| v / corr_norm).collect(),
        corr_z: corr_z_sum.into_iter().map(|v| v / corr_norm).collect(),
        acceptance: if acceptance_count == 0 {
            0.0
        } else {
            acceptance_sum / acceptance_count as f64
        },
        num_measurements,
        num_correlation_measurements: corr_measurements,
    })
}

fn mean_square(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64
}
