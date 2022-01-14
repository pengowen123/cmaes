//! Types related to initializing a [CMAESState]. See [CMAESOptions] for full documentation.

use nalgebra::DVector;

use crate::CMAESState;

/// A builder for [`CMAESState`]. Used to adjust parameters of the algorithm to each particular
/// problem.
///
/// # Examples
///
/// ```
/// # use nalgebra::DVector;
/// # use cmaes::CMAESOptions;
/// let function = |x: &DVector<f64>| x.magnitude();
/// let dim = 3;
/// let cmaes_state = CMAESOptions::new(function, dim)
///     .initial_mean(vec![2.0; dim])
///     .initial_step_size(5.0)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone)]
pub struct CMAESOptions<F> {
    /// The objective function to minimize.
    pub function: F,
    /// Number of dimensions to search.
    pub dimensions: usize,
    /// Initial mean of the search distribution. This should be set to a first guess at the
    /// solution. Default value is the origin.
    pub initial_mean: DVector<f64>,
    /// Initial step size of the search distribution. This should be set to a first guess at how far
    /// the solution is from the initial mean. Default value is `0.5`.
    pub initial_step_size: f64,
    /// Number of points to generate each generation. Default value is `4 + 3 * floor(ln(dimensions))`.
    ///
    /// A larger population size will increase the robustness of the algorithm and help avoid local optima,
    /// but will lead to a slower convergence rate. Conversely, a lower value will reduce the robustness of
    /// the algorithm but will lead to a higher convergence rate. Generally, the population size should only
    /// be increased from the default value. It may be useful to restart the algorithm repeatedly
    /// with increasing population sizes.
    pub population_size: usize,
    /// The distribution to use when assigning weights to individuals. Default value is
    /// `Weights::Negative`.
    pub weights: Weights,
    /// The value to use for the `TolFun` termination criterion (see
    /// [`TerminationReason`][crate::TerminationReason]). Default value is `1e-10`.
    pub tol_fun: f64,
    /// The value to use for the `TolX` termination criterion (see
    /// [`TerminationReason`][crate::TerminationReason]). Default value is `1e-10 *
    /// initial_step_size`, used if this field is `None`.
    pub tol_x: Option<f64>,
}

impl<F: Fn(&DVector<f64>) -> f64> CMAESOptions<F> {
    /// Creates a new `CMAESOptions` with default values. Set individual options using the provided
    /// methods.
    pub fn new(function: F, dimensions: usize) -> Self {
        Self {
            function,
            dimensions,
            initial_mean: DVector::zeros(dimensions),
            initial_step_size: 0.5,
            population_size: 4 + 3 * (dimensions as f64).ln().floor() as usize,
            weights: Weights::Negative,
            tol_fun: 1e-10,
            tol_x: None,
        }
    }

    /// Changes the initial mean from the origin.
    pub fn initial_mean<V: Into<DVector<f64>>>(mut self, initial_mean: V) -> Self {
        self.initial_mean = initial_mean.into();
        self
    }

    /// Changes the initial step size from the default value.
    pub fn initial_step_size(mut self, initial_step_size: f64) -> Self {
        self.initial_step_size = initial_step_size;
        self
    }

    /// Changes the population size from the default value (must be at least 4).
    pub fn population_size(mut self, population_size: usize) -> Self {
        self.population_size = population_size;
        self
    }

    /// Changes the weight distribution from the default of `Weights::Negative`. See [`Weights`] for
    /// possible distributions.
    pub fn weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    /// Changes the value for the `TolFun` termination criterion from the default value (see
    /// [`TerminationReason`][crate::TerminationReason]).
    pub fn tol_fun(mut self, tol_fun: f64) -> Self {
        self.tol_fun = tol_fun;
        self
    }

    /// Changes the value for the `TolX` termination criterion from the default value (see
    /// [`TerminationReason`][crate::TerminationReason]).
    pub fn tol_x(mut self, tol_x: f64) -> Self {
        self.tol_x = Some(tol_x);
        self
    }

    /// Attempts to build the [`CMAESState`] using the chosen options.
    pub fn build(self) -> Result<CMAESState<F>, InvalidOptionsError> {
        CMAESState::new(self)
    }
}

/// The distribution of weights for the population. The default value is `Negative`.
#[derive(Clone)]
pub enum Weights {
    /// Weights are higher for higher-ranked selected individuals and are zero for the rest of the
    /// population.
    Positive,
    /// Similar to `Positive`, but non-selected individuals have negative weights. With this
    /// setting, the algorithm is known as active CMA-ES or aCMA-ES.
    Negative,
    /// Weights for selected individuals are equal and are zero for the rest of the population. This
    /// setting will likely perform much worse than the others.
    Uniform,
}

impl Default for Weights {
    fn default() -> Self {
        Self::Negative
    }
}

/// Represents invalid options for CMA-ES.
#[derive(Clone, Debug)]
pub enum InvalidOptionsError {
    /// The number of dimensions is set to zero.
    ZeroDimensions,
    /// The dimension of the initial mean does not match the chosen dimension.
    MeanDimensionMismatch,
    /// The population size is too small (must be at least 4).
    SmallPopulationSize,
}

