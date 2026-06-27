pub const TWO_PI: f64 = std::f64::consts::PI * 2.0;

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
            self.sin_theta[i] = theta.sin();
            self.cos_theta[i] = theta.cos();
        }
        Ok(())
    }

    pub(crate) fn validate(&self, lattice: &ThetaLattice) -> Result<(), String> {
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
        self.sin_theta[idx] = theta.sin();
        self.cos_theta[idx] = theta.cos();
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
    pub magnetization: f64,
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
    pub magnetization: f64,
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
