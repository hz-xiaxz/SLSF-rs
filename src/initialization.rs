use rand::{Rng, RngExt};

use crate::types::{InitMode, Parameters, ThetaLattice, TWO_PI};

pub fn initialize_two_point_layer_disorder<R: Rng + ?Sized>(
    values: &mut [f64],
    mean: f64,
    delta: f64,
    rng: &mut R,
    coupling_name: &str,
) -> Result<(), String> {
    if delta < 0.0 {
        return Err(format!("δ{coupling_name} must be nonnegative"));
    }
    let lower = mean - delta;
    let upper = mean + delta;
    if lower < 0.0 {
        return Err(format!(
            "two-point layer disorder requires {coupling_name}_mean - δ{coupling_name} >= 0"
        ));
    }

    if delta == 0.0 {
        values.fill(mean);
    } else {
        for value in values {
            *value = if rng.random::<bool>() { upper } else { lower };
        }
    }
    Ok(())
}

pub fn initialize_disorder<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    rng: &mut R,
) -> Result<(), String> {
    lattice.j_xy.fill(params.j_xy);
    initialize_two_point_layer_disorder(
        &mut lattice.j_z,
        params.j_z_mean,
        params.delta_j_z,
        rng,
        "J_z",
    )
}

pub fn initialize_angles<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    mode: InitMode,
    rng: &mut R,
) -> Result<(), String> {
    match mode {
        InitMode::Random => {
            for theta in &mut lattice.theta {
                *theta = TWO_PI * rng.random::<f64>();
            }
        }
        InitMode::Cold => lattice.theta.fill(0.0),
    }
    Ok(())
}
