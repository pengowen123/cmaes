//! Configuration of automatic restarts at the top-level (not configuration of specific restart
//! strategies themselves). See [`RestartOptions`] for full documentation.

use std::ops::RangeInclusive;
use std::time::Duration;

use super::{RestartStrategy, Restarter};
use crate::Mode;

/// Represents invalid options for a `Restarter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidRestartOptionsError {
    /// The number of dimensions is set to zero.
    Dimensions,
    /// The search range size is zero.
    SearchRange,
}

/// Represents invalid options for an individual restart strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidRestartStrategyOptionsError {
    /// The initial step size would be set to an invalid value with the given options.
    InitialStepSize,
    /// The population size would be set to an invalid value with the given options.
    PopulationSize,
}

/// A builder for [`Restarter`][Restarter]. Configuration of individual strategies is done when
/// creating their respective types.
///
/// # Examples
///
/// ```no_run
/// use cmaes::restart::{RestartOptions, RestartStrategy};
///
/// let strategy = RestartStrategy::BIPOP(Default::default());
/// let restarter = RestartOptions::new(10, -1.0..=1.0, strategy).build().unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct RestartOptions {
    /// The strategy to use in performing the restarts.
    pub strategy: RestartStrategy,
    /// The number of dimensions to search.
    pub dimensions: usize,
    /// The optimization mode. Default value is [`Minimize`][Mode::Minimize].
    pub mode: Mode,
    /// The range in which to generate the initial mean for each run. The same range is used in each
    /// dimension (i.e., `[A, B]^N`). To scale the search range separately in each dimension, the
    /// appropriate transformation should be made to the objective function itself using
    /// [`Scale`][crate::objective_function::Scale].
    pub search_range: RangeInclusive<f64>,
    /// The target objective function value. The restart strategy will terminate if/when this value
    /// is reached. Default value is `None`.
    pub fun_target: Option<f64>,
    /// The maximum number of objective function evaluations allowed across all runs. Default value
    /// is `None`
    pub max_function_evals: Option<usize>,
    /// The time limit across all runs. Default value is `None`.
    pub max_time: Option<Duration>,
    /// The maximum number of objective function evaluations allowed for each run. Default value
    /// is `None`.
    pub max_function_evals_per_run: Option<usize>,
    /// The maximum number of generations allowed for each run. Default value is `None`.
    pub max_generations_per_run: Option<usize>,
    /// Whether to print info about each run. Default value is `false`.
    pub enable_printing: bool,
    /// The seed for the [`Restarter`] RNG. This is not the seed for the CMA-ES runs themselves, but
    /// it is used to generate them. Can be set manually for deterministic runs. By default a random
    /// seed is used if this field is `None`.
    pub seed: Option<u64>,
}

impl RestartOptions {
    /// Returns the default set of `RestartOptions` with the chosen restart strategy. Set individual
    /// options using the provided methods.
    pub fn new(
        dimensions: usize,
        mut search_range: RangeInclusive<f64>,
        strategy: RestartStrategy,
    ) -> Self {
        // Correct flipped ranges
        if search_range.is_empty() {
            search_range = *search_range.end()..=*search_range.start();
        }

        Self {
            strategy,
            dimensions,
            mode: Mode::Minimize,
            search_range,
            fun_target: None,
            max_function_evals: None,
            max_generations_per_run: None,
            max_time: None,
            max_function_evals_per_run: None,
            enable_printing: false,
            seed: None,
        }
    }

    /// Sets the optimization mode.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the target objective function value.
    pub fn fun_target(mut self, fun_target: f64) -> Self {
        self.fun_target = Some(fun_target);
        self
    }

    /// Sets the maximum number of objective function evaluations allowed across all runs.
    pub fn max_function_evals(mut self, function_evals: usize) -> Self {
        self.max_function_evals = Some(function_evals);
        self
    }

    /// Sets the time limit.
    pub fn max_time(mut self, max_time: Duration) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Sets the maximum number of objective function evaluations allowed for each run.
    pub fn max_function_evals_per_run(mut self, function_evals: usize) -> Self {
        self.max_function_evals_per_run = Some(function_evals);
        self
    }

    /// Sets the maximum number of generations allowed for each run.
    pub fn max_generations_per_run(mut self, generations: usize) -> Self {
        self.max_generations_per_run = Some(generations);
        self
    }

    /// Sets whether to print info about each run.
    pub fn enable_printing(mut self, enable_printing: bool) -> Self {
        self.enable_printing = enable_printing;
        self
    }

    /// Sets the seed for the [`Restarter`] RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Attempts to build the [`Restarter`] from the chosen options.
    pub fn build(self) -> Result<Restarter, InvalidRestartOptionsError> {
        Restarter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::restart::Local;

    #[test]
    fn test_build() {
        assert!(RestartOptions::new(
            2,
            0.0..=1.0,
            RestartStrategy::Local(Local::new(10, None).unwrap())
        )
        .build()
        .is_ok());
        assert!(matches!(
            RestartOptions::new(
                0,
                0.0..=1.0,
                RestartStrategy::Local(Local::new(10, None).unwrap())
            )
            .build(),
            Err(InvalidRestartOptionsError::Dimensions)
        ));
        assert!(matches!(
            RestartOptions::new(
                2,
                2.0..=2.0,
                RestartStrategy::Local(Local::new(10, None).unwrap())
            )
            .build(),
            Err(InvalidRestartOptionsError::SearchRange)
        ));
    }
}
