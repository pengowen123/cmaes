//! Types related to initializing a [`CMAES`]. See [`CMAESOptions`] for full documentation.

use nalgebra::DVector;

use crate::parameters::Weights;
use crate::{ObjectiveFunction, PlotOptions, CMAES};

/// A builder for [`CMAES`]. Used to adjust parameters of the algorithm to each particular
/// problem and to change other options. See the fields and methods for a full list of options.
///
/// # Examples
///
/// ```
/// # use cmaes::{CMAESOptions, DVector, PlotOptions};
/// let function = |x: &DVector<f64>| x.magnitude();
/// let dim = 3;
/// let mut cmaes_state = CMAESOptions::new(dim)
///     .initial_mean(vec![2.0; dim])
///     .initial_step_size(5.0)
///     .enable_plot(PlotOptions::new(0, false))
///     .enable_printing(200)
///     .build(function)
///     .unwrap();
/// ```
#[derive(Clone)]
pub struct CMAESOptions {
    /// Number of dimensions to search (`N`).
    pub dimensions: usize,
    /// Initial mean of the search distribution. This should be set to a first guess at the
    /// solution. Default value is the origin.
    pub initial_mean: DVector<f64>,
    /// Initial step size of the search distribution (`sigma0`). This should be set to a first guess
    /// at how far the solution is from the initial mean. Default value is `0.5`.
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
    /// The value to use for the
    /// [`TerminationReason::FunTarget`][crate::TerminationReason::FunTarget] termination criterion.
    /// Default value is `1e-12`.
    pub fun_target: f64,
    /// The value to use for the [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]
    /// termination criterion. Default value is `1e-12`.
    pub tol_fun: f64,
    /// The value to use for the [`TerminationReason::TolX`][crate::TerminationReason::TolX]
    /// termination criterion. Default value is `1e-12 * initial_step_size`, used if this field is
    /// `None`.
    pub tol_x: Option<f64>,
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
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            initial_mean: DVector::zeros(dimensions),
            initial_step_size: 0.5,
            population_size: 4 + (3.0 * (dimensions as f64).ln()).floor() as usize,
            weights: Weights::Negative,
            cm: 1.0,
            max_function_evals: None,
            max_generations: None,
            fun_target: 1e-12,
            tol_fun: 1e-12,
            tol_x: None,
            seed: None,
            plot_options: None,
            print_gap_evals: None,
        }
    }

    /// Changes the initial mean from the origin.
    pub fn initial_mean<V: Into<DVector<f64>>>(mut self, initial_mean: V) -> Self {
        self.initial_mean = initial_mean.into();
        self
    }

    /// Changes the initial step size from the default value. Must be positive.
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

    /// Changes the value for the `TolFun` termination criterion from the default value (see
    /// [`TerminationReason::TolFun`][crate::TerminationReason::TolFun]).
    pub fn tol_fun(mut self, tol_fun: f64) -> Self {
        self.tol_fun = tol_fun;
        self
    }

    /// Changes the value for the `FunTarget` termination criterion from the default value (see
    /// [`TerminationReason::FunTarget`][crate::TerminationReason::FunTarget]).
    pub fn fun_target(mut self, fun_target: f64) -> Self {
        self.fun_target = fun_target;
        self
    }

    /// Changes the value for the `TolX` termination criterion from the default value (see
    /// [`TerminationReason::TolX`][crate::TerminationReason::TolX]).
    pub fn tol_x(mut self, tol_x: f64) -> Self {
        self.tol_x = Some(tol_x);
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

    /// Attempts to build the [`CMAES`] using the chosen options. See [`CMAES`] for
    /// information about the lifetime parameter.
    pub fn build<'a, F: ObjectiveFunction + 'a>(
        self,
        objective_function: F,
    ) -> Result<CMAES<'a>, InvalidOptionsError> {
        CMAES::new(Box::new(objective_function), self)
    }
}

/// Represents invalid options for CMA-ES.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvalidOptionsError {
    /// The number of dimensions is set to zero.
    Dimensions,
    /// The dimension of the initial mean does not match the chosen dimension.
    MeanDimensionMismatch,
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
        assert!(CMAESOptions::new(5).build(dummy_function).is_ok());
        assert!(matches!(
            CMAESOptions::new(5)
                .population_size(3)
                .build(dummy_function),
            Err(InvalidOptionsError::PopulationSize),
        ));
        assert!(matches!(
            CMAESOptions::new(5)
                .initial_step_size(-1.0)
                .build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(5)
                .initial_step_size(f64::NAN)
                .build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(5)
                .initial_step_size(f64::INFINITY)
                .build(dummy_function),
            Err(InvalidOptionsError::InitialStepSize),
        ));
        assert!(matches!(
            CMAESOptions::new(5)
                .initial_mean(vec![1.0; 2])
                .build(dummy_function),
            Err(InvalidOptionsError::MeanDimensionMismatch),
        ));
        assert!(matches!(
            CMAESOptions::new(0).build(dummy_function),
            Err(InvalidOptionsError::Dimensions),
        ));
        assert!(matches!(
            CMAESOptions::new(5).cm(2.0).build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(5).cm(-1.0).build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(5).cm(f64::NAN).build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
        assert!(matches!(
            CMAESOptions::new(5).cm(f64::INFINITY).build(dummy_function),
            Err(InvalidOptionsError::Cm),
        ));
    }
}
