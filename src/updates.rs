use rand::{Rng, RngExt};

use crate::types::{
    minus, plus, validate_proposal_width, validate_temperature, Parameters, ThetaLattice,
    ThetaScratch, WolffScratch, TWO_PI,
};

type Site = (usize, usize, usize);
type SinCosBatch<const LANES: usize> = fn([f64; LANES]) -> ([f64; LANES], [f64; LANES]);

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UpdateProfileStats {
    pub wolff_clusters: usize,
    pub wolff_cluster_sites: usize,
    pub wolff_examined_edges: usize,
    pub wolff_zero_probability_edges: usize,
    pub metropolis_scalar_uphill: [usize; 2],
    pub metropolis_x4_uphill: [usize; 5],
    pub metropolis_x8_uphill: [usize; 9],
    pub metropolis_seconds: f64,
    pub wolff_seconds: f64,
    pub measurement_seconds: f64,
}

#[cfg(feature = "profile-stats")]
thread_local! {
    static UPDATE_PROFILE_STATS: std::cell::RefCell<UpdateProfileStats> =
        std::cell::RefCell::new(UpdateProfileStats::default());
}

#[cfg(feature = "profile-stats")]
pub fn reset_update_profile_stats() {
    UPDATE_PROFILE_STATS.with(|stats| *stats.borrow_mut() = UpdateProfileStats::default());
}

#[cfg(feature = "profile-stats")]
pub fn take_update_profile_stats() -> UpdateProfileStats {
    UPDATE_PROFILE_STATS.with(|stats| std::mem::take(&mut *stats.borrow_mut()))
}

#[cfg(feature = "profile-stats")]
pub(crate) fn record_profile_phase(metropolis: f64, wolff: f64, measurement: f64) {
    UPDATE_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        stats.metropolis_seconds += metropolis;
        stats.wolff_seconds += wolff;
        stats.measurement_seconds += measurement;
    });
}

#[cfg(feature = "profile-stats")]
fn record_scalar_uphill(uphill: usize) {
    UPDATE_PROFILE_STATS.with(|stats| stats.borrow_mut().metropolis_scalar_uphill[uphill] += 1);
}

#[cfg(feature = "profile-stats")]
fn record_batch_uphill<const LANES: usize>(uphill: usize) {
    UPDATE_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        match LANES {
            4 => stats.metropolis_x4_uphill[uphill] += 1,
            8 => stats.metropolis_x8_uphill[uphill] += 1,
            _ => unreachable!("unsupported Metropolis SIMD lane count"),
        }
    });
}

#[cfg(feature = "profile-stats")]
fn record_wolff_cluster(scratch: &mut WolffScratch, cluster_size: usize) {
    let examined_edges = std::mem::take(&mut scratch.examined_edges);
    let zero_probability_edges = std::mem::take(&mut scratch.zero_probability_edges);
    UPDATE_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        stats.wolff_clusters += 1;
        stats.wolff_cluster_sites += cluster_size;
        stats.wolff_examined_edges += examined_edges;
        stats.wolff_zero_probability_edges += zero_probability_edges;
    });
}

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
#[allow(clippy::too_many_arguments)]
fn local_metropolis_step_at_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    scratch: &mut ThetaScratch,
    width: f64,
    beta: f64,
    idx: usize,
    x: usize,
    y: usize,
    z: usize,
    rng: &mut R,
) -> bool {
    let theta_old = lattice.theta[idx];
    let theta_new = crate::types::wrap_angle(theta_old + width * (2.0 * rng.random::<f64>() - 1.0));
    let (new_sin, new_cos) = crate::fast_math::sin_cos(theta_new);
    let (hx, hy) = local_field(lattice, scratch, idx, x, y, z);
    let old_energy =
        local_theta_energy_from_trig(scratch.sin_theta[idx], scratch.cos_theta[idx], hx, hy);
    let delta_energy = local_theta_energy_from_trig(new_sin, new_cos, hx, hy) - old_energy;

    #[cfg(feature = "profile-stats")]
    {
        crate::updates::record_scalar_uphill((delta_energy > 0.0) as usize);
    }

    if delta_energy <= 0.0 || rng.random::<f64>() < crate::fast_math::exp(-delta_energy * beta) {
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
    local_metropolis_step_at_unchecked(
        lattice,
        scratch,
        width,
        1.0 / params.temperature,
        idx,
        x,
        y,
        z,
        rng,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn local_metropolis_step_x_batch<const LANES: usize, R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    scratch: &mut ThetaScratch,
    width: f64,
    beta: f64,
    x0: usize,
    y: usize,
    z: usize,
    rng: &mut R,
    sin_cos_batch: SinCosBatch<LANES>,
    exp_batch: fn([f64; LANES]) -> [f64; LANES],
) -> usize {
    let mut idxs = [0usize; LANES];
    let mut theta_new = [0.0; LANES];
    let mut hx = [0.0; LANES];
    let mut hy = [0.0; LANES];
    let mut old_energy = [0.0; LANES];

    for lane in 0..LANES {
        let x = x0 + 2 * lane;
        let idx = lattice.idx(x, y, z);
        idxs[lane] = idx;
        theta_new[lane] = crate::types::wrap_angle(
            lattice.theta[idx] + width * (2.0 * rng.random::<f64>() - 1.0),
        );
        let (site_hx, site_hy) = local_field(lattice, scratch, idx, x, y, z);
        hx[lane] = site_hx;
        hy[lane] = site_hy;
        old_energy[lane] = local_theta_energy_from_trig(
            scratch.sin_theta[idx],
            scratch.cos_theta[idx],
            site_hx,
            site_hy,
        );
    }

    let (new_sin, new_cos) = sin_cos_batch(theta_new);
    let mut delta_energy = [0.0; LANES];
    let mut exp_arg = [0.0; LANES];
    let mut uphill_lane = [usize::MAX; LANES];
    let mut uphill_count = 0usize;
    for lane in 0..LANES {
        delta_energy[lane] =
            local_theta_energy_from_trig(new_sin[lane], new_cos[lane], hx[lane], hy[lane])
                - old_energy[lane];
        if delta_energy[lane] > 0.0 {
            exp_arg[uphill_count] = -delta_energy[lane] * beta;
            uphill_lane[uphill_count] = lane;
            uphill_count += 1;
        }
    }

    let mut accept_prob = [1.0; LANES];
    #[cfg(feature = "profile-stats")]
    record_batch_uphill::<LANES>(uphill_count);
    if uphill_count == LANES {
        accept_prob = exp_batch(exp_arg);
    } else {
        for uphill_idx in 0..uphill_count {
            let lane = uphill_lane[uphill_idx];
            accept_prob[lane] = crate::fast_math::exp(exp_arg[uphill_idx]);
        }
    }

    let mut accepted = 0usize;
    for lane in 0..LANES {
        if delta_energy[lane] <= 0.0 || rng.random::<f64>() < accept_prob[lane] {
            lattice.theta[idxs[lane]] = theta_new[lane];
            scratch.set_site_trig(idxs[lane], new_sin[lane], new_cos[lane]);
            accepted += 1;
        }
    }
    accepted
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn local_metropolis_step_x_batch4_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    scratch: &mut ThetaScratch,
    width: f64,
    beta: f64,
    x0: usize,
    y: usize,
    z: usize,
    rng: &mut R,
) -> usize {
    local_metropolis_step_x_batch(
        lattice,
        scratch,
        width,
        beta,
        x0,
        y,
        z,
        rng,
        crate::fast_math::sin_cos_x4,
        crate::fast_math::exp_x4,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn local_metropolis_step_x_batch8_unchecked<R: Rng + ?Sized>(
    lattice: &mut ThetaLattice,
    scratch: &mut ThetaScratch,
    width: f64,
    beta: f64,
    x0: usize,
    y: usize,
    z: usize,
    rng: &mut R,
) -> usize {
    local_metropolis_step_x_batch(
        lattice,
        scratch,
        width,
        beta,
        x0,
        y,
        z,
        rng,
        crate::fast_math::sin_cos_x8,
        crate::fast_math::exp_x8,
    )
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
    if !lattice.l_x.is_multiple_of(2)
        || !lattice.l_y.is_multiple_of(2)
        || !lattice.l_z.is_multiple_of(2)
    {
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
    let beta = 1.0 / params.temperature;
    let use_batch8 = crate::fast_math::has_avx512_f64_lanes();
    for parity in 0..2usize {
        for z in 0..lattice.l_z {
            for y in 0..lattice.l_y {
                let x_start = (parity + y + z) & 1;
                let mut x = x_start;
                if use_batch8 {
                    while x + 14 < lattice.l_x {
                        accepted += local_metropolis_step_x_batch8_unchecked(
                            lattice, scratch, width, beta, x, y, z, rng,
                        );
                        x += 16;
                    }
                }
                while x + 6 < lattice.l_x {
                    accepted += local_metropolis_step_x_batch4_unchecked(
                        lattice, scratch, width, beta, x, y, z, rng,
                    );
                    x += 8;
                }
                while x < lattice.l_x {
                    let idx = lattice.idx(x, y, z);
                    accepted += local_metropolis_step_at_unchecked(
                        lattice, scratch, width, beta, idx, x, y, z, rng,
                    ) as usize;
                    x += 2;
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
    1.0 - crate::fast_math::exp((-2.0 * beta * coupling * ri * rj).min(0.0))
}

#[inline]
pub fn wolff_reflect_angle(theta: f64, phi: f64) -> f64 {
    crate::types::wrap_angle(2.0 * phi + std::f64::consts::PI - theta)
}

#[allow(clippy::too_many_arguments)]
fn try_add_wolff_neighbor<R: Rng + ?Sized>(
    lattice: &ThetaLattice,
    scratch: &mut WolffScratch,
    theta_scratch: Option<&ThetaScratch>,
    beta: f64,
    phi: f64,
    sin_phi: f64,
    cos_phi: f64,
    ri: f64,
    neighbor: Neighbor,
    rng: &mut R,
) {
    let (xn, yn, zn) = neighbor.site;
    let nidx = lattice.idx(xn, yn, zn);
    if scratch.in_cluster[nidx] {
        return;
    }
    #[cfg(feature = "profile-stats")]
    {
        scratch.examined_edges += 1;
    }

    let rj = match theta_scratch {
        Some(theta_scratch) => {
            theta_scratch.cos_theta[nidx] * cos_phi + theta_scratch.sin_theta[nidx] * sin_phi
        }
        None => crate::fast_math::cos(lattice.theta[nidx] - phi),
    };
    if neighbor.coupling * ri * rj <= 0.0 {
        #[cfg(feature = "profile-stats")]
        {
            scratch.zero_probability_edges += 1;
        }
        return;
    }
    if rng.random::<f64>() < wolff_add_probability(beta, neighbor.coupling, ri, rj) {
        scratch.in_cluster[nidx] = true;
        scratch.stack.push((xn, yn, zn));
        scratch.cluster_sites.push(nidx);
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
    let (sin_phi, cos_phi) = crate::fast_math::sin_cos(phi);
    let x0 = rng.random_range(0..lattice.l_x);
    let y0 = rng.random_range(0..lattice.l_y);
    let z0 = rng.random_range(0..lattice.l_z);

    debug_assert!(scratch.cluster_sites.is_empty());
    scratch.stack.clear();
    let seed_idx = lattice.idx(x0, y0, z0);
    scratch.in_cluster[seed_idx] = true;
    scratch.stack.push((x0, y0, z0));
    scratch.cluster_sites.push(seed_idx);

    while let Some((x, y, z)) = scratch.stack.pop() {
        let idx = lattice.idx(x, y, z);
        let ri = match theta_scratch.as_deref() {
            Some(ts) => ts.cos_theta[idx] * cos_phi + ts.sin_theta[idx] * sin_phi,
            None => crate::fast_math::cos(lattice.theta[idx] - phi),
        };
        for neighbor in neighbors!(lattice, params, x, y, z) {
            try_add_wolff_neighbor(
                lattice,
                scratch,
                theta_scratch.as_deref(),
                beta,
                phi,
                sin_phi,
                cos_phi,
                ri,
                neighbor,
                rng,
            );
        }
    }

    let cluster_size = scratch.cluster_sites.len();
    #[cfg(feature = "profile-stats")]
    record_wolff_cluster(scratch, cluster_size);
    match theta_scratch {
        Some(ts) => {
            for &idx in &scratch.cluster_sites {
                let new_theta = wolff_reflect_angle(lattice.theta[idx], phi);
                lattice.theta[idx] = new_theta;
                ts.update_site(idx, new_theta);
                scratch.in_cluster[idx] = false;
            }
        }
        None => {
            for &idx in &scratch.cluster_sites {
                lattice.theta[idx] = wolff_reflect_angle(lattice.theta[idx], phi);
                scratch.in_cluster[idx] = false;
            }
        }
    }
    scratch.cluster_sites.clear();
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
