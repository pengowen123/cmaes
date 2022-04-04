//! Types related to initializing a [`CMAES`]. See [`CMAESOptions`] for full documentation.

use nalgebra::DVector;

use std::time::Duration;

use crate::parameters::Weights;
use crate::{PlotOptions, CMAES};

/// A builder for [`CMAES`]. Used to adjust parameters of the algorithm to each particular
/// problem and to change other options. See the fields and methods for a full list of options.
///
/// # Examples
///
/// ```
/// use cmaes::{CMAESOptions, DVector, PlotOptions, Weights};
///
/// let function = |x: &DVector<f64>| x.magnitude();
/// let dim = 3;
/// let mut cmaes_state = CMAESOptions::new(vec![2.0; dim], 5.0)
///     .weights(Weights::Positive)
///     .population_size(100)
///     .enable_plot(PlotOptions::new(0, false))
///     .enable_printing(200)
///     .build(function)
///     .unwrap();
/// ```
#[derive(Clone)]
pub struct CMAESOptions {
    /// Initial mean of the search distribution, also used to determine the problem dimension (`N`).
    /// This should be set to a first guess at the solution.
    pub initial_mean: DVector<f64>,
    /// Initial step size of the search distribution (`sigma0`). This should be set to a first guess
    /// at how far the solution is from the initial mean.
    pub initial_step_size: f64,
    /// Number of points to generate each generation (`lambda`). Default value is
    /// `4 + floor(3 * ln(dimensions))`.
    ///
    /// A larger population size will increase the robustness of the algorithm and help avoid
    /// converging to local optima, but will lead to a slower convergence rate. Conversely, a lower
    /// value will reduce the robustness of the algorithm but will lead to a higher convergence
    /// rate. Generally, the population size should only be left at the default value or increased.
    /// It is generally useful to restart the algorithm repeatedly with increasing population sizes.
    pub population_size: usize,
    /// The distribution to use when assigning weights to individuals. Default value is
    /// [`Weights::Negative`].
    pub weights: Weights,
    /// The learning rate for adapting the mean. Can be reduced for noisy functions. Default value
    /// is `1.0`.
    pub cm: f64,
    /// The value to use for the
    /// [`TerminationReason::MaxFunctionEvals`][crate::TerminationReason::MaxFunctionEvals]
    /// termination criterion. Default value is `None`.
    pub max_function_evals: Option<usize>,
    /// The value to use for the
    /// [`TerminationReason::MaxGenerations`][crate::TerminationReason::MaxGenerations]
    /// termination criterion. Default value is `None`.
    pub max_generations: Option<usize>,
    /// The value to use for the [`TerminationReason::MaxTime`][crate::TerminationReason::MaxTime]
    /// termination criterion. Default value is `None`.
    pub max_time: Option<Duration>,
    /// The value to use for the
    /// [`TerminationReason::FunTarget`][crate::TerminationReason::FunTarget] termination criterion.
    /// Default value is `1e-12`.
    pub fun_target: f64,
    /// The value to use for the [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]
    /// termination criterion. Default value is `1e-12`.
    pub tol_fun: f64,
    /// The value to use for the
    /// [`TerminationReason::TolFunRel`][crate::TerminationReason::TolFunRel] termination criterion.
    /// Default value is `0` (disabled).
    pub tol_fun_rel: f64,
    /// The value to use for the
    /// [`TerminationReason::TolFunHist`][crate::TerminationReason::TolFunHist] termination
    /// criterion. Default value is `1e-12`.
    pub tol_fun_hist: f64,
    /// The value to use for the [`TerminationReason::TolX`][crate::TerminationReason::TolX]
    /// termination criterion. Default value is `1e-12 * initial_step_size`, used if this field is
    /// `None`.
    pub tol_x: Option<f64>,
    /// The minimum number of generations over which to measure the
    /// [`TerminationReason::TolStagnation`][crate::TerminationReason::TolStagnation] termination
    /// criterion. Default value is `100 + 100 * dimensions^1.5 / lambda`, used if this field is
    /// `None`.
    pub tol_stagnation: Option<usize>,
    /// The value to use for the [`TerminationReason::TolXUp`][crate::TerminationReason::TolXUp]
    /// termination criterion. Default value is `1e+8`.
    pub tol_x_up: f64,
    /// The value to use for the
    /// [`TerminationReason::TolConditionCov`][crate::TerminationReason::TolConditionCov]
    /// termination criterion. Default value is `1e+14`.
    pub tol_condition_cov: f64,
    /// The seed for the RNG used in the algorithm. Can be set manually for deterministic runs. By
    /// default a random seed is used if this field is `None`.
    pub seed: Option<u64>,
    /// Options for the data plot. Default value is `None`, meaning no plot will be generated. See
    /// [`Plot`][crate::plotting::Plot].
    pub plot_options: Option<PlotOptions>,
    /// How many function evaluations to wait for in between each automatic
    /// [`CMAES::print_info`] call. Default value is `None`, meaning no info will be
    /// automatically printed.
    pub print_gap_evals: Option<usize>,
}

impl CMAESOptions {
    /// Creates a new `CMAESOptions` with default values. Set individual options using the provided
    /// methods.
    ///
    /// - `initial_mean` should be set to a first guess at the solution and is used to determine the
    /// problem dimension.
    /// - `initial_step_size` should be set to a first guess at how far the solution is in each
    /// dimension from the initial mean (`solution[i] ~= [initial_mean[i] - initial_step_size,
    /// initial_mean[i] + initial_step_size]`). Must be positive.
    pub fn new<V: Into<DVector<f64>>>(initial_mean: V, initial_step_size: f64) -> Self {
        let initial_mean = initial_mean.into();
        let dimensions = initial_mean.len();
        Self {
            initial_mean,
            initial_step_size,
            population_size: 4 + (3.0 * (dimensions as f64).ln()).floor() as usize,
            weights: Weights::Negative,
            cm: 1.0,
            max_function_evals: None,
            max_generations: None,
            max_time: None,
            fun_target: 1e-12,
            tol_fun: 1e-12,
            tol_fun_rel: 0.0,
            tol_fun_hist: 1e-12,
            tol_x: None,
            tol_stagnation: None,
            tol_x_up: 1e8,
            tol_condition_cov: 1e14,
            seed: None,
            plot_options: None,
            print_gap_evals: None,
        }
    }

    /// Changes the initial mean.
    pub fn initial_mean<V: Into<DVector<f64>>>(mut self, initial_mean: V) -> Self {
        self.initial_mean = initial_mean.into();
        self
    }

    /// Changes the initial step size. Must be positive.
    pub fn initial_step_size(mut self, initial_step_size: f64) -> Self {
        self.initial_step_size = initial_step_size;
        self
    }

    /// Changes the population size from the default value. Must be at least 4.
    pub fn population_size(mut self, population_size: usize) -> Self {
        self.population_size = population_size;
        self
    }

    /// Changes the weight distribution from the default value. See [`Weights`] for
    /// possible distributions.
    pub fn weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    /// Changes the learning rate for the mean from the default value. Must be between `0.0` and
    /// `1.0`.
    pub fn cm(mut self, cm: f64) -> Self {
        self.cm = cm;
        self
    }

    /// Changes the value for the `MaxFunctionEvals` termination criterion from the default value
    /// (see [`TerminationReason::MaxFunctionEvals`][crate::TerminationReason::MaxFunctionEvals]).
    pub fn max_function_evals(mut self, max_function_evals: usize) -> Self {
        self.max_function_evals = Some(max_function_evals);
        self
    }

    /// Changes the value for the `MaxGenerations` termination criterion from the default value
    /// (see [`TerminationReason::MaxGenerations`][crate::TerminationReason::MaxGenerations]).
    pub fn max_generations(mut self, max_generations: usize) -> Self {
        self.max_generations = Some(max_generations);
        self
    }

    /// Changes the value for the `MaxTime` termination criterion from the default value
    /// (see [`TerminationReason::MaxTime`][crate::TerminationReason::MaxTime]).
    pub fn max_time(mut self, max_time: Duration) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Changes the value for the `FunTarget` termination criterion from the default value (see
    /// [`TerminationReason::FunTarget`][crate::TerminationReason::FunTarget]).
    pub fn fun_target(mut self, fun_target: f64) -> Self {
        self.fun_target = fun_target;
        self
    }

    /// Changes the value for the `TolFun` termination criterion from the default value (see
    /// [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]).
    pub fn tol_fun(mut self, tol_fun: f64) -> Self {
        self.tol_fun = tol_fun;
        self
    }

    /// Changes the value for the `TolFunRel` termination criterion from the default value (see
    /// [`TerminationReason::TolFunRel`][crate::TerminationReason::TolFunRel]).
    pub fn tol_fun_rel(mut self, tol_fun_rel: f64) -> Self {
        self.tol_fun_rel = tol_fun_rel;
        self
    }

    /// Changes the value for the `TolFunHist` termination criterion from the default value (see
    /// [`TerminationReason::TolFunHist`][crate::TerminationReason::TolFunHist]).
    pub fn tol_fun_hist(mut self, tol_fun_hist: f64) -> Self {
        self.tol_fun_hist = tol_fun_hist;
        self
    }

    /// Changes the value for the `TolX` termination criterion from the default value (see
    /// [`TerminationReason::TolX`][crate::TerminationReason::TolX]).
    pub fn tol_x(mut self, tol_x: f64) -> Self {
        self.tol_x = Some(tol_x);
        self
    }

    /// Changes the minimum value for the `TolStagnation` termination criterion from the default
    /// value (see [`TerminationReason::TolStagnation`][crate::TerminationReason::TolStagnation]).
    pub fn tol_stagnation(mut self, tol_stagnation: usize) -> Self {
        self.tol_stagnation = Some(tol_stagnation);
        self
    }

    /// Changes the value for the `TolXUp` termination criterion from the default value (see
    /// [`TerminationReason::TolXUp`][crate::TerminationReason::TolXUp]).
    pub fn tol_x_up(mut self, tol_x_up: f64) -> Self {
        self.tol_x_up = tol_x_up;
        self
    }

    /// Changes the value for the `TolConditionCov` termination criterion from the default value
    /// (see [`TerminationReason::TolConditionCov`][crate::TerminationReason::TolConditionCov]).
    pub fn tol_condition_cov(mut self, tol_condition_cov: f64) -> Self {
        self.tol_condition_cov = tol_condition_cov;
        self
    }

    /// Sets the seed for the RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enables recording of a data plot for various state variables of the algorithm. See
    /// [`Plot`][crate::plotting::Plot].
    pub fn enable_plot(mut self, plot_options: PlotOptions) -> Self {
        self.plot_options = Some(plot_options);
        self
    }

    /// Enables automatic printing of various state variables of the algorithm. A minimum of
    /// `min_gap_evals` function evaluations will be waited for between each
    /// [`CMAES::print_info`] call, but it will always be called for the first few generations.
    pub fn enable_printing(mut self, min_gap_evals: usize) -> Self {
        self.print_gap_evals = Some(min_gap_evals);
        self
    }

    /// Attempts to build the [`CMAES`] using the chosen options.
    pub fn build<F>(self, objective_function: F) -> Result<CMAES<F>, InvalidOptionsError> {
        CMAES::new(objective_function, self)
    }
}

/// Represents invalid options for CMA-ES.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvalidOptionsError {
    /// The number of dimensions is set to zero.
    Dimensions,
    /// The population size is too small (must be at least 4).
    PopulationSize,
    /// The initial step size is negative or non-normal.
    InitialStepSize,
    /// The learning rate is outside the valid range (`0.0` to `1.0`).
    Cm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build() {
        let dummy_function = |_: &DVector<f64>| 0.0;
        assert!(CMAESOptions::new(vec![1.0; 5], 1.0)
            .build(dummy_function)
            .is_ok());
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], 1.0)
                .population_size(3)
                .build(dummy_function),
            Err(InvalidOptionsError::PopulationSize),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], -1.0).build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], f64::NAN).build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], f64::INFINITY).build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![], 1.0).build(dummy_function),
            Err(InvalidOptionsError::Dimensions),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], 1.0)
                .cm(2.0)
                .build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], 1.0)
                .cm(-1.0)
                .build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], 1.0)
                .cm(f64::NAN)
                .build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(vec![1.0; 5], 1.0)
                .cm(f64::INFINITY)
                .build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
    }
}
