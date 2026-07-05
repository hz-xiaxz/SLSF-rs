use std::convert::Infallible;

use rand::{SeedableRng, TryRng};

pub const TWO_PI: f64 = std::f64::consts::PI * 2.0;
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Debug, Clone)]
pub struct FastRng {
    seed: u64,
    state: u64,
    draws: u128,
}

impl FastRng {
    #[inline]
    fn splitmix64_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(SPLITMIX64_GAMMA);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn position(&self) -> u128 {
        self.draws
    }

    pub fn set_position(&mut self, position: u128) {
        self.state = self
            .seed
            .wrapping_add(SPLITMIX64_GAMMA.wrapping_mul(position as u64));
        self.draws = position;
    }
}

impl SeedableRng for FastRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        let seed = u64::from_le_bytes(seed);
        Self::seed_from_u64(seed)
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self {
            seed,
            state: seed,
            draws: 0,
        }
    }
}

impl TryRng for FastRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let word = Self::splitmix64_next(&mut self.state);
        self.draws += 1;
        Ok(word)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let word = self.try_next_u64()?.to_le_bytes();
            remainder.copy_from_slice(&word[..remainder.len()]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    pub j_xy: f64,
    pub j_z_mean: f64,
    pub delta_j_z: f64,
    pub temperature: f64,
}

impl Parameters {
    pub fn new(j_xy: f64, j_z_mean: f64, delta_j_z: f64, temperature: f64) -> Self {
        Self {
            j_xy,
            j_z_mean,
            delta_j_z,
            temperature,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMode {
    Random,
    Cold,
}

#[derive(Debug, Clone)]
pub struct ThetaLattice {
    pub l_x: usize,
    pub l_y: usize,
    pub l_z: usize,
    pub theta: Vec<f64>,
    pub j_xy: Vec<f64>,
    pub j_z: Vec<f64>,
}

impl ThetaLattice {
    pub fn new(l_x: usize, l_y: usize, l_z: usize) -> Result<Self, String> {
        if l_x == 0 {
            return Err("L_x must be positive".to_string());
        }
        if l_y == 0 {
            return Err("L_y must be positive".to_string());
        }
        if l_z == 0 {
            return Err("L_z must be positive".to_string());
        }
        Ok(Self {
            l_x,
            l_y,
            l_z,
            theta: vec![0.0; l_x * l_y * l_z],
            j_xy: vec![0.0; l_z],
            j_z: vec![0.0; l_z],
        })
    }

    #[inline]
    pub fn volume(&self) -> usize {
        self.l_x * self.l_y * self.l_z
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < self.l_x && y < self.l_y && z < self.l_z);
        x + self.l_x * (y + self.l_y * z)
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f64 {
        self.theta[self.idx(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f64) {
        let idx = self.idx(x, y, z);
        self.theta[idx] = value;
    }
}

#[derive(Debug, Clone)]
pub struct ThetaScratch {
    pub(crate) sin_theta: Vec<f64>,
    pub(crate) cos_theta: Vec<f64>,
    dims: (usize, usize, usize),
}

impl ThetaScratch {
    pub fn new(lattice: &ThetaLattice) -> Self {
        let mut scratch = Self {
            sin_theta: vec![0.0; lattice.volume()],
            cos_theta: vec![0.0; lattice.volume()],
            dims: (lattice.l_x, lattice.l_y, lattice.l_z),
        };
        scratch
            .refresh(lattice)
            .expect("fresh theta scratch has matching dimensions");
        scratch
    }

    pub fn refresh(&mut self, lattice: &ThetaLattice) -> Result<(), String> {
        self.validate(lattice)?;
        for (i, &theta) in lattice.theta.iter().enumerate() {
            let (sin_theta, cos_theta) = crate::fast_math::sin_cos(theta);
            self.sin_theta[i] = sin_theta;
            self.cos_theta[i] = cos_theta;
        }
        Ok(())
    }

    pub fn validate(&self, lattice: &ThetaLattice) -> Result<(), String> {
        if self.dims != (lattice.l_x, lattice.l_y, lattice.l_z)
            || self.sin_theta.len() != lattice.volume()
            || self.cos_theta.len() != lattice.volume()
        {
            return Err("theta scratch dimensions do not match lattice dimensions".to_string());
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn update_site(&mut self, idx: usize, theta: f64) {
        let (sin_theta, cos_theta) = crate::fast_math::sin_cos(theta);
        self.set_site_trig(idx, sin_theta, cos_theta);
    }

    #[inline]
    pub(crate) fn set_site_trig(&mut self, idx: usize, sin_theta: f64, cos_theta: f64) {
        self.sin_theta[idx] = sin_theta;
        self.cos_theta[idx] = cos_theta;
    }
}

#[derive(Debug, Clone)]
pub struct WolffScratch {
    pub(crate) in_cluster: Vec<bool>,
    pub(crate) stack: Vec<(usize, usize, usize)>,
    dims: (usize, usize, usize),
}

impl WolffScratch {
    pub fn new(lattice: &ThetaLattice) -> Self {
        let volume = lattice.volume();
        Self {
            in_cluster: vec![false; volume],
            stack: Vec::with_capacity(volume),
            dims: (lattice.l_x, lattice.l_y, lattice.l_z),
        }
    }

    pub(crate) fn validate(&self, lattice: &ThetaLattice) -> Result<(), String> {
        if self.dims != (lattice.l_x, lattice.l_y, lattice.l_z)
            || self.in_cluster.len() != lattice.volume()
        {
            return Err("Wolff scratch dimensions do not match lattice dimensions".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaObservables {
    pub energy: f64,
    pub magnetization_squared: f64,
    pub cos_x: f64,
    pub cos_y: f64,
    pub cos_z: f64,
    pub sin_x: f64,
    pub sin_y: f64,
    pub sin_z: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaCorrelations {
    pub r: Vec<usize>,
    pub r_xy: Vec<usize>,
    pub r_z: Vec<usize>,
    pub corr_x: Vec<f64>,
    pub corr_y: Vec<f64>,
    pub corr_xy: Vec<f64>,
    pub corr_z: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaSimulationOptions {
    pub thermal_sweeps: usize,
    pub measure_sweeps: usize,
    pub measure_interval: usize,
    pub proposal_width: f64,
    pub wolff_steps: usize,
    pub correlation_interval: usize,
    pub correlation_rmax: Option<usize>,
    pub correlation_rmax_xy: Option<usize>,
    pub correlation_rmax_z: Option<usize>,
    pub init_mode: InitMode,
}

impl Default for ThetaSimulationOptions {
    fn default() -> Self {
        Self {
            thermal_sweeps: 1000,
            measure_sweeps: 5000,
            measure_interval: 10,
            proposal_width: std::f64::consts::PI,
            wolff_steps: 1,
            correlation_interval: 1,
            correlation_rmax: None,
            correlation_rmax_xy: None,
            correlation_rmax_z: None,
            init_mode: InitMode::Random,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaSimulationResult {
    pub energy: f64,
    pub rho_sx: f64,
    pub rho_sy: f64,
    pub rho_sz: f64,
    pub std_rho_sx: f64,
    pub std_rho_sy: f64,
    pub std_rho_sz: f64,
    pub magnetization_squared: f64,
    pub chi: f64,
    pub corr_r: Vec<usize>,
    pub corr_r_xy: Vec<usize>,
    pub corr_r_z: Vec<usize>,
    pub corr_x: Vec<f64>,
    pub corr_y: Vec<f64>,
    pub corr_xy: Vec<f64>,
    pub corr_z: Vec<f64>,
    pub acceptance: f64,
    pub num_measurements: usize,
    pub num_correlation_measurements: usize,
}

#[inline]
pub fn wrap_angle(theta: f64) -> f64 {
    if theta >= TWO_PI {
        theta - TWO_PI
    } else if theta < 0.0 {
        theta + TWO_PI
    } else {
        theta
    }
}

#[inline]
pub fn wrap_angle_full(theta: f64) -> f64 {
    theta.rem_euclid(TWO_PI)
}

#[inline]
pub(crate) fn plus(i: usize, l: usize) -> usize {
    if i + 1 < l {
        i + 1
    } else {
        0
    }
}

#[inline]
pub(crate) fn minus(i: usize, l: usize) -> usize {
    if i > 0 {
        i - 1
    } else {
        l - 1
    }
}

#[inline]
pub(crate) fn angle_diff(a: f64, b: f64) -> f64 {
    a - b
}

pub(crate) fn validate_temperature(params: &Parameters) -> Result<(), String> {
    if !(params.temperature > 0.0 && params.temperature.is_finite()) {
        return Err("temperature T must be positive and finite".to_string());
    }
    Ok(())
}

pub(crate) fn validate_proposal_width(proposal_width: f64) -> Result<f64, String> {
    if !(proposal_width >= 0.0 && proposal_width.is_finite()) {
        return Err("proposal_width must be nonnegative and finite".to_string());
    }
    Ok(proposal_width)
}
