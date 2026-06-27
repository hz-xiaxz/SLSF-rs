use rand::Rng;

use crate::types::{InitMode, Parameters, ThetaLattice, TWO_PI};

pub fn initialize_disorder<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    rng: &mut R,
) -> Result<(), String> {
    if params.delta_j_z < 0.0 {
        return Err("δJ_z must be nonnegative".to_string());
    }
    let lower = params.j_z_mean - params.delta_j_z;
    let upper = params.j_z_mean + params.delta_j_z;
    if lower < 0.0 {
        return Err("uniform layer disorder requires J_z_mean - δJ_z >= 0".to_string());
    }

    if params.delta_j_z == 0.0 {
        lattice.j_z.fill(params.j_z_mean);
    } else {
        let width = upper - lower;
        for jz in &mut lattice.j_z {
            *jz = lower + width * rng.gen::<f64>();
        }
    }
    Ok(())
}

pub fn initialize_angles<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    mode: InitMode,
    rng: &mut R,
) -> Result<(), String> {
    match mode {
        InitMode::Random => {
            for theta in &mut lattice.theta {
                *theta = TWO_PI * rng.gen::<f64>();
            }
        }
        InitMode::Cold => lattice.theta.fill(0.0),
    }
    Ok(())
}
