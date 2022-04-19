//! Initialization of constant parameters of the algorithm.

mod weights;

use nalgebra::DVector;

use std::time::Duration;

pub use weights::Weights;

use crate::mode::Mode;
use crate::{termination, CMAESOptions};
use weights::{FinalWeights, InitialWeights};

/// Parameters of the termination criteria
#[derive(Clone, Debug)]
pub(crate) struct TerminationParameters {
    /// Value for the MaxFunctionEvals termination criterion (disabled if `None`)
    pub max_function_evals: Option<usize>,
    /// Value for the MaxGenerations termination criterion (disabled if `None`)
    pub max_generations: Option<usize>,
    /// Value for the MaxTime termination criterion (disabled if `None`)
    pub max_time: Option<Duration>,
    /// Value for the FunTarget termination criterion (disabled if `None`)
    pub fun_target: Option<f64>,
    /// Value for the TolFun termination criterion
    pub tol_fun: f64,
    /// Value for the TolFunRel termination criterion
    pub tol_fun_rel: f64,
    /// Value for the TolFunHist termination criterion
    pub tol_fun_hist: f64,
    /// Value for the TolX termination criterion
    pub tol_x: f64,
    /// Minimum value for the TolStagnation termination criterion
    pub tol_stagnation: usize,
    /// Value for the TolXUp termination criterion
    pub tol_x_up: f64,
    /// Value for the TolConditionCov termination criterion
    pub tol_condition_cov: f64,
}

impl TerminationParameters {
    /// Initializes the `TerminationParameters` with the parameters set in `options`
    pub(crate) fn from_options(options: &CMAESOptions) -> Self {
        let dimensions = options.initial_mean.len();
        let tol_x = options.tol_x.unwrap_or(1e-12 * options.initial_step_size);
        let default_tol_stagnation =
            termination::get_default_tol_stagnation_option(dimensions, options.population_size);
        let tol_stagnation = options.tol_stagnation.unwrap_or(default_tol_stagnation);

        Self {
            max_function_evals: options.max_function_evals,
            max_generations: options.max_generations,
            max_time: options.max_time,
            fun_target: options.fun_target,
            tol_fun: options.tol_fun,
            tol_fun_rel: options.tol_fun_rel,
            tol_fun_hist: options.tol_fun_hist,
            tol_x,
            tol_stagnation,
            tol_x_up: options.tol_x_up,
            tol_condition_cov: options.tol_condition_cov,
        }
    }
}

/// Stores constant parameters for the algorithm. Obtained by calling
/// [`CMAES::parameters`][crate::CMAES::parameters].
#[derive(Clone, Debug)]
pub struct Parameters {
    /// Optimization mode
    mode: Mode,
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
    /// Parameters of the termination criteria
    termination: TerminationParameters,
    /// Seed for the RNG
    seed: u64,
}

impl Parameters {
    /// Calculates and returns a new set of `Parameters`
    pub(crate) fn new(
        options: &CMAESOptions,
        seed: u64,
        termination: TerminationParameters,
    ) -> Self {
        let dim = options.initial_mean.len();
        let initial_weights = InitialWeights::new(options.population_size, options.weights);
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
            mode: options.mode,
            dim,
            lambda: options.population_size,
            mu,
            initial_sigma: options.initial_step_size,
            mu_eff,
            weights: final_weights,
            cc,
            c1,
            cs,
            cmu,
            cm: options.cm,
            damp_s,
            termination,
            seed,
        }
    }

    /// Calculates and returns a new set of `Parameters` from the provided options
    pub(crate) fn from_options(options: &CMAESOptions, seed: u64) -> Self {
        Self::new(options, seed, TerminationParameters::from_options(options))
    }

    /// Returns the optimization mode.
    pub fn mode(&self) -> Mode {
        self.mode
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

    /// Returns the value for the
    /// [`TerminationReason::MaxFunctionEvals`][crate::TerminationReason::MaxFunctionEvals]
    /// termination criterion.
    pub fn max_function_evals(&self) -> Option<usize> {
        self.termination.max_function_evals
    }

    /// Returns the value for the
    /// [`TerminationReason::MaxGenerations`][crate::TerminationReason::MaxGenerations]
    /// termination criterion.
    pub fn max_generations(&self) -> Option<usize> {
        self.termination.max_generations
    }

    /// Returns the value for the [`TerminationReason::MaxTime`][crate::TerminationReason::MaxTime]
    /// termination criterion.
    pub fn max_time(&self) -> Option<Duration> {
        self.termination.max_time
    }

    /// Returns the value for the
    /// [`TerminationReason::FunTarget`][crate::TerminationReason::FunTarget] termination criterion.
    pub fn fun_target(&self) -> Option<f64> {
        self.termination.fun_target
    }

    /// Returns the value for the [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]
    /// termination criterion.
    pub fn tol_fun(&self) -> f64 {
        self.termination.tol_fun
    }

    /// Returns the value for the
    /// [`TerminationReason::TolFunRel`][crate::TerminationReason::TolFunRel] termination criterion.
    pub fn tol_fun_rel(&self) -> f64 {
        self.termination.tol_fun_rel
    }

    /// Returns the value for the
    /// [`TerminationReason::TolFunHist`][crate::TerminationReason::TolFunHist] termination
    /// criterion.
    pub fn tol_fun_hist(&self) -> f64 {
        self.termination.tol_fun_hist
    }

    /// Returns the value for the [`TerminationReason::TolX`][crate::TerminationReason::TolX]
    /// termination criterion.
    pub fn tol_x(&self) -> f64 {
        self.termination.tol_x
    }

    /// Returns the minimum value for the
    /// [`TerminationReason::TolStagnation`][crate::TerminationReason::TolStagnation] termination
    /// criterion.
    pub fn tol_stagnation(&self) -> usize {
        self.termination.tol_stagnation
    }

    /// Returns the value for the [`TerminationReason::TolXUp`][crate::TerminationReason::TolXUp]
    /// termination criterion.
    pub fn tol_x_up(&self) -> f64 {
        self.termination.tol_x_up
    }

    /// Returns the value for the
    /// [`TerminationReason::TolConditionCov`][crate::TerminationReason::TolConditionCov]
    /// termination criterion.
    pub fn tol_condition_cov(&self) -> f64 {
        self.termination.tol_condition_cov
    }

    /// Returns the seed for the RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}
