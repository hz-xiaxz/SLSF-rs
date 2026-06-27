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
    _params: &Parameters,
    scratch: &ThetaScratch,
    x: usize,
    y: usize,
    z: usize,
) -> (f64, f64) {
    neighbors!(lattice, _params, x, y, z)
        .into_iter()
        .fold((0.0, 0.0), |(hx, hy), neighbor| {
            let (nx, ny, nz) = neighbor.site;
            let idx = lattice.idx(nx, ny, nz);
            (
                hx + neighbor.coupling * scratch.cos_theta[idx],
                hy + neighbor.coupling * scratch.sin_theta[idx],
            )
        })
}

#[inline]
fn local_theta_energy(theta: f64, hx: f64, hy: f64) -> f64 {
    -(theta.cos() * hx + theta.sin() * hy)
}

pub(crate) fn local_metropolis_step_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    params: &Parameters,
    scratch: &mut ThetaScratch,
    width: f64,
    rng: &mut R,
) -> bool {
    let x = rng.random_range(0..lattice.l_x);
    let y = rng.random_range(0..lattice.l_y);
    let z = rng.random_range(0..lattice.l_z);
    let idx = lattice.idx(x, y, z);
    let theta_old = lattice.theta[idx];
    let theta_new = crate::types::wrap_angle(theta_old + width * (2.0 * rng.random::<f64>() - 1.0));
    let (hx, hy) = local_field(lattice, params, scratch, x, y, z);
    let old_energy = -(scratch.cos_theta[idx] * hx + scratch.sin_theta[idx] * hy);
    let delta_energy = local_theta_energy(theta_new, hx, hy) - old_energy;

    if delta_energy <= 0.0 || rng.random::<f64>() < (-delta_energy / params.temperature).exp() {
        lattice.theta[idx] = theta_new;
        scratch.update_site(idx, theta_new);
        true
    } else {
        false
    }
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
        rng,
    ))
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
    let volume = lattice.volume();
    let mut accepted = 0usize;
    for _ in 0..volume {
        accepted += local_metropolis_step_unchecked(lattice, params, scratch, width, rng) as usize;
    }
    Ok(accepted as f64 / volume as f64)
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
    beta: f64,
    phi: f64,
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
    let ri = (lattice.get(x, y, z) - phi).cos();
    let rj = (lattice.get(xn, yn, zn) - phi).cos();
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
            try_add_wolff_neighbor(lattice, scratch, beta, phi, site, neighbor, rng);
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
