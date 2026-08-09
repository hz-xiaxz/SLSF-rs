use std::simd::{num::SimdFloat, Simd};

use crate::types::{
    angle_diff, plus, validate_temperature, Parameters, ThetaCorrelations, ThetaLattice,
    ThetaObservables, ThetaScratch,
};

#[inline]
fn diff_cos_sin(cos_a: f64, sin_a: f64, cos_b: f64, sin_b: f64) -> (f64, f64) {
    (cos_a * cos_b + sin_a * sin_b, sin_a * cos_b - cos_a * sin_b)
}

#[inline]
fn cached_correlation_dot_x8(
    cos_theta: &[f64],
    sin_theta: &[f64],
    a_start: usize,
    b_start: usize,
    len: usize,
) -> f64 {
    const LANES: usize = 8;
    let mut i = 0;
    let mut sum = Simd::<f64, LANES>::splat(0.0);
    while i + LANES <= len {
        let a = a_start + i;
        let b = b_start + i;
        let cos_a = Simd::<f64, LANES>::from_slice(&cos_theta[a..a + LANES]);
        let sin_a = Simd::<f64, LANES>::from_slice(&sin_theta[a..a + LANES]);
        let cos_b = Simd::<f64, LANES>::from_slice(&cos_theta[b..b + LANES]);
        let sin_b = Simd::<f64, LANES>::from_slice(&sin_theta[b..b + LANES]);
        sum += cos_a * cos_b + sin_a * sin_b;
        i += LANES;
    }

    let mut scalar_sum = sum.reduce_sum();
    while i < len {
        let a = a_start + i;
        let b = b_start + i;
        scalar_sum += cos_theta[a] * cos_theta[b] + sin_theta[a] * sin_theta[b];
        i += 1;
    }
    scalar_sum
}

pub fn measure_theta_energy(lattice: &ThetaLattice, _params: &Parameters) -> f64 {
    let mut energy = 0.0;
    for z in 0..lattice.l_z {
        for y in 0..lattice.l_y {
            for x in 0..lattice.l_x {
                let theta = lattice.get(x, y, z);
                energy -= lattice.j_xy[z]
                    * angle_diff(theta, lattice.get(plus(x, lattice.l_x), y, z)).cos();
                energy -= lattice.j_xy[z]
                    * angle_diff(theta, lattice.get(x, plus(y, lattice.l_y), z)).cos();
                energy -= lattice.j_z[z]
                    * angle_diff(theta, lattice.get(x, y, plus(z, lattice.l_z))).cos();
            }
        }
    }
    energy / lattice.volume() as f64
}

pub fn measure_magnetization_squared(lattice: &ThetaLattice) -> f64 {
    let (mx, my) = lattice.theta.iter().fold((0.0, 0.0), |(mx, my), &theta| {
        (mx + theta.cos(), my + theta.sin())
    });
    let volume = lattice.volume() as f64;
    (mx * mx + my * my) / (volume * volume)
}

pub fn measure_magnetization(lattice: &ThetaLattice) -> f64 {
    measure_magnetization_squared(lattice).sqrt()
}

pub fn measure_theta_observables(lattice: &ThetaLattice, params: &Parameters) -> ThetaObservables {
    let scratch = ThetaScratch::new(lattice);
    measure_theta_observables_with_scratch(lattice, params, &scratch)
}

pub fn measure_theta_observables_with_scratch(
    lattice: &ThetaLattice,
    _params: &Parameters,
    scratch: &ThetaScratch,
) -> ThetaObservables {
    scratch
        .validate(lattice)
        .expect("theta scratch dimensions must match lattice dimensions");

    let mut energy_sum = 0.0;
    let mut mx_sum = 0.0;
    let mut my_sum = 0.0;
    let mut cos_x = 0.0;
    let mut cos_y = 0.0;
    let mut cos_z = 0.0;
    let mut sin_x = 0.0;
    let mut sin_y = 0.0;
    let mut sin_z = 0.0;

    for z in 0..lattice.l_z {
        for y in 0..lattice.l_y {
            for x in 0..lattice.l_x {
                let idx = lattice.idx(x, y, z);
                let cos_theta = scratch.cos_theta[idx];
                let sin_theta = scratch.sin_theta[idx];
                mx_sum += cos_theta;
                my_sum += sin_theta;

                let x_idx = lattice.idx(plus(x, lattice.l_x), y, z);
                let y_idx = lattice.idx(x, plus(y, lattice.l_y), z);
                let z_idx = lattice.idx(x, y, plus(z, lattice.l_z));
                let (dx_cos, dx_sin) = diff_cos_sin(
                    cos_theta,
                    sin_theta,
                    scratch.cos_theta[x_idx],
                    scratch.sin_theta[x_idx],
                );
                let (dy_cos, dy_sin) = diff_cos_sin(
                    cos_theta,
                    sin_theta,
                    scratch.cos_theta[y_idx],
                    scratch.sin_theta[y_idx],
                );
                let (dz_cos, dz_sin) = diff_cos_sin(
                    cos_theta,
                    sin_theta,
                    scratch.cos_theta[z_idx],
                    scratch.sin_theta[z_idx],
                );
                let jcx = lattice.j_xy[z] * dx_cos;
                let jcy = lattice.j_xy[z] * dy_cos;
                let jcz = lattice.j_z[z] * dz_cos;
                cos_x += jcx;
                cos_y += jcy;
                cos_z += jcz;
                sin_x += lattice.j_xy[z] * dx_sin;
                sin_y += lattice.j_xy[z] * dy_sin;
                sin_z += lattice.j_z[z] * dz_sin;
                energy_sum -= jcx + jcy + jcz;
            }
        }
    }

    let volume = lattice.volume() as f64;
    ThetaObservables {
        energy: energy_sum / volume,
        magnetization_squared: (mx_sum * mx_sum + my_sum * my_sum) / (volume * volume),
        cos_x,
        cos_y,
        cos_z,
        sin_x,
        sin_y,
        sin_z,
    }
}

pub fn measure_theta_correlations(
    lattice: &ThetaLattice,
    rmax: Option<usize>,
    rmax_xy: Option<usize>,
    rmax_z: Option<usize>,
) -> ThetaCorrelations {
    let scratch = ThetaScratch::new(lattice);
    measure_theta_correlations_with_scratch(lattice, &scratch, rmax, rmax_xy, rmax_z)
}

pub fn measure_theta_correlations_with_scratch(
    lattice: &ThetaLattice,
    scratch: &ThetaScratch,
    rmax: Option<usize>,
    rmax_xy: Option<usize>,
    rmax_z: Option<usize>,
) -> ThetaCorrelations {
    scratch
        .validate(lattice)
        .expect("theta scratch dimensions must match lattice dimensions");

    let requested_xy =
        rmax_xy.unwrap_or_else(|| rmax.unwrap_or_else(|| (lattice.l_x / 2).min(lattice.l_y / 2)));
    let requested_z = rmax_z.unwrap_or_else(|| rmax.unwrap_or(lattice.l_z / 2));
    let rmax_xy_eff = requested_xy.min(lattice.l_x / 2).min(lattice.l_y / 2);
    let rmax_z_eff = requested_z.min(lattice.l_z / 2);
    let mut corr_x = vec![0.0; rmax_xy_eff];
    let mut corr_y = vec![0.0; rmax_xy_eff];
    let mut corr_xy_by_z = vec![vec![0.0; rmax_xy_eff]; lattice.l_z];
    let mut corr_z = vec![0.0; rmax_z_eff];
    let volume = lattice.volume() as f64;
    let layer_area = (lattice.l_x * lattice.l_y) as f64;

    for r in 1..=rmax_xy_eff {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for (z, layer_corr) in corr_xy_by_z.iter_mut().enumerate() {
            let z_base = lattice.l_x * lattice.l_y * z;
            let mut layer_sx = 0.0;
            let mut layer_sy = 0.0;
            for y in 0..lattice.l_y {
                let row_start = z_base + lattice.l_x * y;
                layer_sx += cached_correlation_dot_x8(
                    &scratch.cos_theta,
                    &scratch.sin_theta,
                    row_start,
                    row_start + r,
                    lattice.l_x - r,
                );
                layer_sx += cached_correlation_dot_x8(
                    &scratch.cos_theta,
                    &scratch.sin_theta,
                    row_start + lattice.l_x - r,
                    row_start,
                    r,
                );

                let y_shift = (y + r) % lattice.l_y;
                layer_sy += cached_correlation_dot_x8(
                    &scratch.cos_theta,
                    &scratch.sin_theta,
                    row_start,
                    z_base + lattice.l_x * y_shift,
                    lattice.l_x,
                );
            }
            sx += layer_sx;
            sy += layer_sy;
            layer_corr[r - 1] = (layer_sx + layer_sy) / (2.0 * layer_area);
        }
        corr_x[r - 1] = sx / volume;
        corr_y[r - 1] = sy / volume;
    }

    for r in 1..=rmax_z_eff {
        let mut sz = 0.0;
        for z in 0..lattice.l_z {
            let z_base = lattice.l_x * lattice.l_y * z;
            let z_shift_base = lattice.l_x * lattice.l_y * ((z + r) % lattice.l_z);
            for y in 0..lattice.l_y {
                sz += cached_correlation_dot_x8(
                    &scratch.cos_theta,
                    &scratch.sin_theta,
                    z_base + lattice.l_x * y,
                    z_shift_base + lattice.l_x * y,
                    lattice.l_x,
                );
            }
        }
        corr_z[r - 1] = sz / volume;
    }

    let corr_xy = corr_x
        .iter()
        .zip(&corr_y)
        .map(|(x, y)| (x + y) / 2.0)
        .collect::<Vec<_>>();
    let r_xy = (1..=rmax_xy_eff).collect::<Vec<_>>();
    let r_z = (1..=rmax_z_eff).collect::<Vec<_>>();
    ThetaCorrelations {
        r: r_xy.clone(),
        r_xy,
        r_z,
        corr_x,
        corr_y,
        corr_xy,
        corr_xy_by_z,
        corr_z,
    }
}

pub fn helicity_sums(
    lattice: &ThetaLattice,
    _params: &Parameters,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut cos_x = 0.0;
    let mut cos_y = 0.0;
    let mut cos_z = 0.0;
    let mut sin_x = 0.0;
    let mut sin_y = 0.0;
    let mut sin_z = 0.0;

    for z in 0..lattice.l_z {
        for y in 0..lattice.l_y {
            for x in 0..lattice.l_x {
                let theta = lattice.get(x, y, z);
                let dx = angle_diff(theta, lattice.get(plus(x, lattice.l_x), y, z));
                let dy = angle_diff(theta, lattice.get(x, plus(y, lattice.l_y), z));
                let dz = angle_diff(theta, lattice.get(x, y, plus(z, lattice.l_z)));
                cos_x += lattice.j_xy[z] * dx.cos();
                cos_y += lattice.j_xy[z] * dy.cos();
                cos_z += lattice.j_z[z] * dz.cos();
                sin_x += lattice.j_xy[z] * dx.sin();
                sin_y += lattice.j_xy[z] * dy.sin();
                sin_z += lattice.j_z[z] * dz.sin();
            }
        }
    }
    (cos_x, cos_y, cos_z, sin_x, sin_y, sin_z)
}

pub fn measure_helicity_modulus(
    lattice: &ThetaLattice,
    params: &Parameters,
) -> Result<(f64, f64, f64), String> {
    validate_temperature(params)?;
    let volume = lattice.volume() as f64;
    let beta = 1.0 / params.temperature;
    let (cos_x, cos_y, cos_z, sin_x, sin_y, sin_z) = helicity_sums(lattice, params);
    Ok((
        cos_x / volume - beta * sin_x.powi(2) / volume,
        cos_y / volume - beta * sin_y.powi(2) / volume,
        cos_z / volume - beta * sin_z.powi(2) / volume,
    ))
}
