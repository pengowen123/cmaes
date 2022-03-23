//! Initialization of constant parameters of the algorithm.

mod weights;

use nalgebra::DVector;

pub use weights::Weights;

use weights::{FinalWeights, InitialWeights};

/// Stores constant parameters for the algorithm. Obtained by calling
/// [`CMAES::parameters`][crate::CMAES::parameters].
#[derive(Clone, Debug)]
pub struct Parameters {
    /// Number of dimensions to search
    dim: usize,
    /// Population size,
    lambda: usize,
    /// Number of individuals to select each generation
    mu: usize,
    /// Initial value for sigma
    initial_sigma: f64,
    /// Variance-effective selection mass
    mu_eff: f64,
    /// Individual weights
    weights: FinalWeights,
    /// Learning rate for rank-one update cumulation
    cc: f64,
    /// Learning rate for rank-one update
    c1: f64,
    /// Learning rate for step size update
    cs: f64,
    /// Learning rate for rank-mu update
    cmu: f64,
    /// Learning rate for the mean
    cm: f64,
    /// Damping parameter for step size update
    damp_s: f64,
    /// Value for the TolFun termination criterion
    tol_fun: f64,
    /// Value for the TolX termination criterion
    tol_x: f64,
    /// Seed for the RNG
    seed: u64,
}

impl Parameters {
    /// Calculates and returns a new set of `Parameters`
    pub(crate) fn new(
        dim: usize,
        lambda: usize,
        weights: Weights,
        seed: u64,
        initial_sigma: f64,
        cm: f64,
        tol_fun: f64,
        tol_x: f64,
    ) -> Self {
        let initial_weights = InitialWeights::new(lambda, weights);
        let mu = initial_weights.mu();
        let mu_eff = initial_weights.mu_eff();

        // Covariance matrix adaptation
        let a_cov = 2.0;
        let cc = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
        let c1 = a_cov / ((dim as f64 + 1.3).powi(2) + mu_eff);
        let cmu = (1.0 - c1).min(
            a_cov * (mu_eff - 2.0 + 1.0 / mu_eff)
                / ((dim as f64 + 2.0).powi(2) + a_cov * mu_eff / 2.0),
        );

        let final_weights = initial_weights.finalize(dim, c1, cmu);

        // Step size adaptation
        let cs = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let damp_s = 1.0 + cs + 2.0 * (((mu_eff - 1.0) / (dim as f64 + 1.0)).sqrt() - 1.0).max(0.0);

        Parameters {
            dim,
            lambda,
            mu,
            initial_sigma,
            mu_eff,
            weights: final_weights,
            cc,
            c1,
            cs,
            cmu,
            cm,
            damp_s,
            tol_fun,
            tol_x,
            seed,
        }
    }

    /// Returns the problem dimension `N`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the population size `lambda`.
    pub fn lambda(&self) -> usize {
        self.lambda
    }

    /// Returns the selected population size `mu`.
    pub fn mu(&self) -> usize {
        self.mu
    }

    /// Returns the initial step size `sigma0`.
    pub fn initial_sigma(&self) -> f64 {
        self.initial_sigma
    }

    /// Returns the variance-effective selection mass `mu_eff`.
    pub fn mu_eff(&self) -> f64 {
        self.mu_eff
    }

    /// Returns the weights `w`.
    pub fn weights(&self) -> &DVector<f64> {
        &self.weights
    }

    /// Returns the setting used for calculating the weights.
    pub fn weights_setting(&self) -> Weights {
        self.weights.setting()
    }

    /// Returns the learning rate for rank-one update cumulation `cc`.
    pub fn cc(&self) -> f64 {
        self.cc
    }

    /// Returns the learning rate for the rank-one update `c1`.
    pub fn c1(&self) -> f64 {
        self.c1
    }

    /// Returns the learning rate for the step size update `cs`.
    pub fn cs(&self) -> f64 {
        self.cs
    }

    /// Returns the learning rate for the rank-mu update `cmu`.
    pub fn cmu(&self) -> f64 {
        self.cmu
    }

    /// Returns the learning rate for the mean update `cm`.
    pub fn cm(&self) -> f64 {
        self.cm
    }

    /// Returns the damping factor for the step size update `damp_s`.
    pub fn damp_s(&self) -> f64 {
        self.damp_s
    }

    /// Returns the value for the [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]
    /// termination criterion.
    pub fn tol_fun(&self) -> f64 {
        self.tol_fun
    }

    /// Returns the value for the [`TerminationReason::TolX`][crate::TerminationReason::TolX]
    /// termination criterion.
    pub fn tol_x(&self) -> f64 {
        self.tol_x
    }

    /// Returns the seed for the RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}