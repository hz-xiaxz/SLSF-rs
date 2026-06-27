use crate::types::{
    angle_diff, plus, validate_temperature, Parameters, ThetaCorrelations, ThetaLattice,
    ThetaObservables,
};

pub fn measure_theta_energy(lattice: &ThetaLattice, params: &Parameters) -> f64 {
    let mut energy = 0.0;
    for z in 0..lattice.l_z {
        for y in 0..lattice.l_y {
            for x in 0..lattice.l_x {
                let theta = lattice.get(x, y, z);
                energy -=
                    params.j_xy * angle_diff(theta, lattice.get(plus(x, lattice.l_x), y, z)).cos();
                energy -=
                    params.j_xy * angle_diff(theta, lattice.get(x, plus(y, lattice.l_y), z)).cos();
                energy -= lattice.j_z[z]
                    * angle_diff(theta, lattice.get(x, y, plus(z, lattice.l_z))).cos();
            }
        }
    }
    energy / lattice.volume() as f64
}

pub fn measure_magnetization(lattice: &ThetaLattice) -> f64 {
    let (mx, my) = lattice.theta.iter().fold((0.0, 0.0), |(mx, my), &theta| {
        (mx + theta.cos(), my + theta.sin())
    });
    (mx * mx + my * my).sqrt() / lattice.volume() as f64
}

pub fn measure_theta_observables(lattice: &ThetaLattice, params: &Parameters) -> ThetaObservables {
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
                let theta = lattice.get(x, y, z);
                mx_sum += theta.cos();
                my_sum += theta.sin();
                let dx = angle_diff(theta, lattice.get(plus(x, lattice.l_x), y, z));
                let dy = angle_diff(theta, lattice.get(x, plus(y, lattice.l_y), z));
                let dz = angle_diff(theta, lattice.get(x, y, plus(z, lattice.l_z)));
                let jcx = params.j_xy * dx.cos();
                let jcy = params.j_xy * dy.cos();
                let jcz = lattice.j_z[z] * dz.cos();
                cos_x += jcx;
                cos_y += jcy;
                cos_z += jcz;
                sin_x += params.j_xy * dx.sin();
                sin_y += params.j_xy * dy.sin();
                sin_z += lattice.j_z[z] * dz.sin();
                energy_sum -= jcx + jcy + jcz;
            }
        }
    }

    let volume = lattice.volume() as f64;
    ThetaObservables {
        energy: energy_sum / volume,
        magnetization: (mx_sum * mx_sum + my_sum * my_sum).sqrt() / volume,
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
    let requested_xy =
        rmax_xy.unwrap_or_else(|| rmax.unwrap_or_else(|| (lattice.l_x / 2).min(lattice.l_y / 2)));
    let requested_z = rmax_z.unwrap_or_else(|| rmax.unwrap_or(lattice.l_z / 2));
    let rmax_xy_eff = requested_xy.min(lattice.l_x / 2).min(lattice.l_y / 2);
    let rmax_z_eff = requested_z.min(lattice.l_z / 2);
    let mut corr_x = vec![0.0; rmax_xy_eff];
    let mut corr_y = vec![0.0; rmax_xy_eff];
    let mut corr_z = vec![0.0; rmax_z_eff];
    let volume = lattice.volume() as f64;

    for r in 1..=rmax_xy_eff {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for z in 0..lattice.l_z {
            for y in 0..lattice.l_y {
                for x in 0..lattice.l_x {
                    let theta = lattice.get(x, y, z);
                    sx += angle_diff(theta, lattice.get((x + r) % lattice.l_x, y, z)).cos();
                    sy += angle_diff(theta, lattice.get(x, (y + r) % lattice.l_y, z)).cos();
                }
            }
        }
        corr_x[r - 1] = sx / volume;
        corr_y[r - 1] = sy / volume;
    }

    for r in 1..=rmax_z_eff {
        let mut sz = 0.0;
        for z in 0..lattice.l_z {
            for y in 0..lattice.l_y {
                for x in 0..lattice.l_x {
                    let theta = lattice.get(x, y, z);
                    sz += angle_diff(theta, lattice.get(x, y, (z + r) % lattice.l_z)).cos();
                }
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
        corr_z,
    }
}

pub fn helicity_sums(
    lattice: &ThetaLattice,
    params: &Parameters,
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
                cos_x += params.j_xy * dx.cos();
                cos_y += params.j_xy * dy.cos();
                cos_z += lattice.j_z[z] * dz.cos();
                sin_x += params.j_xy * dx.sin();
                sin_y += params.j_xy * dy.sin();
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
