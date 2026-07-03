use rand::{Rng, RngExt};

use crate::types::{
    minus, plus, validate_proposal_width, validate_temperature, Parameters, ThetaLattice,
    ThetaScratch, WolffScratch, TWO_PI,
};

type Site = (usize, usize, usize);

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    site: Site,
    coupling: f64,
}

macro_rules! neighbors {
    ($lattice:expr, $params:expr, $x:expr, $y:expr, $z:expr) => {{
        let x_p = plus($x, $lattice.l_x);
        let x_m = minus($x, $lattice.l_x);
        let y_p = plus($y, $lattice.l_y);
        let y_m = minus($y, $lattice.l_y);
        let z_p = plus($z, $lattice.l_z);
        let z_m = minus($z, $lattice.l_z);

        [
            Neighbor {
                site: (x_p, $y, $z),
                coupling: $lattice.j_xy[$z],
            },
            Neighbor {
                site: (x_m, $y, $z),
                coupling: $lattice.j_xy[$z],
            },
            Neighbor {
                site: ($x, y_p, $z),
                coupling: $lattice.j_xy[$z],
            },
            Neighbor {
                site: ($x, y_m, $z),
                coupling: $lattice.j_xy[$z],
            },
            Neighbor {
                site: ($x, $y, z_p),
                coupling: $lattice.j_z[$z],
            },
            Neighbor {
                site: ($x, $y, z_m),
                coupling: $lattice.j_z[z_m],
            },
        ]
    }};
}

#[inline]
fn local_field(
    lattice: &ThetaLattice,
    scratch: &ThetaScratch,
    idx: usize,
    x: usize,
    y: usize,
    z: usize,
) -> (f64, f64) {
    let plane = lattice.l_x * lattice.l_y;
    let x_p_idx = if x + 1 == lattice.l_x {
        idx + 1 - lattice.l_x
    } else {
        idx + 1
    };
    let x_m_idx = if x == 0 {
        idx + lattice.l_x - 1
    } else {
        idx - 1
    };
    let y_p_idx = if y + 1 == lattice.l_y {
        idx + lattice.l_x - plane
    } else {
        idx + lattice.l_x
    };
    let y_m_idx = if y == 0 {
        idx + plane - lattice.l_x
    } else {
        idx - lattice.l_x
    };
    let z_p_idx = if z + 1 == lattice.l_z {
        idx + plane - lattice.volume()
    } else {
        idx + plane
    };
    let z_m = if z == 0 { lattice.l_z - 1 } else { z - 1 };
    let z_m_idx = if z == 0 {
        idx + lattice.volume() - plane
    } else {
        idx - plane
    };

    let j_xy = lattice.j_xy[z];
    let j_z_p = lattice.j_z[z];
    let j_z_m = lattice.j_z[z_m];

    let hx = j_xy
        * (scratch.cos_theta[x_p_idx]
            + scratch.cos_theta[x_m_idx]
            + scratch.cos_theta[y_p_idx]
            + scratch.cos_theta[y_m_idx])
        + j_z_p * scratch.cos_theta[z_p_idx]
        + j_z_m * scratch.cos_theta[z_m_idx];
    let hy = j_xy
        * (scratch.sin_theta[x_p_idx]
            + scratch.sin_theta[x_m_idx]
            + scratch.sin_theta[y_p_idx]
            + scratch.sin_theta[y_m_idx])
        + j_z_p * scratch.sin_theta[z_p_idx]
        + j_z_m * scratch.sin_theta[z_m_idx];

    (hx, hy)
}

#[inline]
fn local_theta_energy_from_trig(sin_theta: f64, cos_theta: f64, hx: f64, hy: f64) -> f64 {
    -(cos_theta * hx + sin_theta * hy)
}

#[inline]
fn local_metropolis_step_at_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut ThetaScratch,
    width: f64,
    idx: usize,
    x: usize,
    y: usize,
    z: usize,
    rng: &mut R,
) -> bool {
    let theta_old = lattice.theta[idx];
    let theta_new = crate::types::wrap_angle(theta_old + width * (2.0 * rng.random::<f64>() - 1.0));
    let (new_sin, new_cos) = theta_new.sin_cos();
    let (hx, hy) = local_field(lattice, scratch, idx, x, y, z);
    let old_energy =
        local_theta_energy_from_trig(scratch.sin_theta[idx], scratch.cos_theta[idx], hx, hy);
    let delta_energy = local_theta_energy_from_trig(new_sin, new_cos, hx, hy) - old_energy;

    if delta_energy <= 0.0 || rng.random::<f64>() < (-delta_energy / params.temperature).exp() {
        lattice.theta[idx] = theta_new;
        scratch.set_site_trig(idx, new_sin, new_cos);
        true
    } else {
        false
    }
}

pub(crate) fn local_metropolis_step_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut ThetaScratch,
    width: f64,
    volume: usize,
    rng: &mut R,
) -> bool {
    let idx = rng.random_range(0..volume);
    let plane = lattice.l_x * lattice.l_y;
    let z = idx / plane;
    let xy = idx - z * plane;
    let y = xy / lattice.l_x;
    let x = xy - y * lattice.l_x;
    local_metropolis_step_at_unchecked(lattice, params, scratch, width, idx, x, y, z, rng)
}

pub fn local_metropolis_step<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    proposal_width: f64,
    rng: &mut R,
) -> Result<bool, String> {
    validate_temperature(params)?;
    let width = validate_proposal_width(proposal_width)?;
    let mut scratch = ThetaScratch::new(lattice);
    Ok(local_metropolis_step_unchecked(
        lattice,
        params,
        &mut scratch,
        width,
        lattice.volume(),
        rng,
    ))
}

#[inline]
fn validate_red_black_lattice(lattice: &ThetaLattice) -> Result<(), String> {
    if lattice.l_x % 2 != 0 || lattice.l_y % 2 != 0 || lattice.l_z % 2 != 0 {
        return Err("red-black Metropolis sweep requires even lattice dimensions".to_string());
    }
    Ok(())
}

pub fn metropolis_sweep_with_scratch<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut ThetaScratch,
    proposal_width: f64,
    rng: &mut R,
) -> Result<f64, String> {
    validate_temperature(params)?;
    let width = validate_proposal_width(proposal_width)?;
    scratch.validate(lattice)?;
    validate_red_black_lattice(lattice)?;

    let mut accepted = 0usize;
    for parity in 0..2usize {
        for z in 0..lattice.l_z {
            for y in 0..lattice.l_y {
                let x_start = (parity + y + z) & 1;
                for x in (x_start..lattice.l_x).step_by(2) {
                    let idx = lattice.idx(x, y, z);
                    accepted += local_metropolis_step_at_unchecked(
                        lattice, params, scratch, width, idx, x, y, z, rng,
                    ) as usize;
                }
            }
        }
    }

    Ok(accepted as f64 / lattice.volume() as f64)
}

pub fn metropolis_sweep<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    proposal_width: f64,
    rng: &mut R,
) -> Result<f64, String> {
    let mut scratch = ThetaScratch::new(lattice);
    metropolis_sweep_with_scratch(lattice, params, &mut scratch, proposal_width, rng)
}

#[inline]
pub fn wolff_add_probability(beta: f64, coupling: f64, ri: f64, rj: f64) -> f64 {
    1.0 - (-2.0 * beta * coupling * ri * rj).min(0.0).exp()
}

#[inline]
pub fn wolff_reflect_angle(theta: f64, phi: f64) -> f64 {
    crate::types::wrap_angle(2.0 * phi + std::f64::consts::PI - theta)
}

fn try_add_wolff_neighbor<R: Rng + ?Sized>(
    lattice: &ThetaLattice,
    scratch: &mut WolffScratch,
    theta_scratch: Option<&ThetaScratch>,
    beta: f64,
    phi: f64,
    sin_phi: f64,
    cos_phi: f64,
    site: Site,
    neighbor: Neighbor,
    rng: &mut R,
) {
    let (x, y, z) = site;
    let (xn, yn, zn) = neighbor.site;
    let nidx = lattice.idx(xn, yn, zn);
    if scratch.in_cluster[nidx] {
        return;
    }

    let idx = lattice.idx(x, y, z);
    let (ri, rj) = match theta_scratch {
        Some(theta_scratch) => (
            theta_scratch.cos_theta[idx] * cos_phi + theta_scratch.sin_theta[idx] * sin_phi,
            theta_scratch.cos_theta[nidx] * cos_phi + theta_scratch.sin_theta[nidx] * sin_phi,
        ),
        None => (
            (lattice.theta[idx] - phi).cos(),
            (lattice.theta[nidx] - phi).cos(),
        ),
    };
    if rng.random::<f64>() < wolff_add_probability(beta, neighbor.coupling, ri, rj) {
        scratch.in_cluster[nidx] = true;
        scratch.stack.push((xn, yn, zn));
    }
}

pub fn wolff_cluster_step_with_theta_scratch<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut WolffScratch,
    theta_scratch: Option<&mut ThetaScratch>,
    rng: &mut R,
) -> Result<usize, String> {
    validate_temperature(params)?;
    scratch.validate(lattice)?;
    if let Some(ts) = theta_scratch.as_ref() {
        ts.validate(lattice)?;
    }
    let beta = 1.0 / params.temperature;
    let phi = TWO_PI * rng.random::<f64>();
    let (sin_phi, cos_phi) = phi.sin_cos();
    let x0 = rng.random_range(0..lattice.l_x);
    let y0 = rng.random_range(0..lattice.l_y);
    let z0 = rng.random_range(0..lattice.l_z);

    scratch.in_cluster.fill(false);
    scratch.stack.clear();
    let seed_idx = lattice.idx(x0, y0, z0);
    scratch.in_cluster[seed_idx] = true;
    scratch.stack.push((x0, y0, z0));

    while let Some(site @ (x, y, z)) = scratch.stack.pop() {
        for neighbor in neighbors!(lattice, params, x, y, z) {
            try_add_wolff_neighbor(
                lattice,
                scratch,
                theta_scratch.as_deref(),
                beta,
                phi,
                sin_phi,
                cos_phi,
                site,
                neighbor,
                rng,
            );
        }
    }

    let mut cluster_size = 0usize;
    match theta_scratch {
        Some(ts) => {
            for idx in 0..scratch.in_cluster.len() {
                if scratch.in_cluster[idx] {
                    let new_theta = wolff_reflect_angle(lattice.theta[idx], phi);
                    lattice.theta[idx] = new_theta;
                    ts.update_site(idx, new_theta);
                    cluster_size += 1;
                }
            }
        }
        None => {
            for idx in 0..scratch.in_cluster.len() {
                if scratch.in_cluster[idx] {
                    lattice.theta[idx] = wolff_reflect_angle(lattice.theta[idx], phi);
                    cluster_size += 1;
                }
            }
        }
    }
    Ok(cluster_size)
}

pub fn wolff_cluster_step<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut WolffScratch,
    rng: &mut R,
) -> Result<usize, String> {
    wolff_cluster_step_with_theta_scratch(lattice, params, scratch, None, rng)
}
