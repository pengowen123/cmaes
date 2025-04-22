//! An implementation of the CMA-ES optimization algorithm. It is used to minimize or maximize the
//! value of an objective function and performs well on high-dimension, non-linear, non-convex,
//! ill-conditioned, and/or noisy problems.
//!
//! # Overview
//!
//! There are three main ways to use `cmaes`:
//!
//! The quickest and easiest way is to use a convenience function provided in the [`functions`]
//! module, such as [`fmin`][crate::functions::fmin] or [`fmax`][crate::functions::fmax]:
//!
//! ```no_run
//! use cmaes::DVector;
//!
//! let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
//! let dim = 10;
//! let solution = cmaes::fmin(sphere, vec![5.0; dim], 1.0);
//! ```
//!
//! Further configuration of a run can be performed by creating and building a [`CMAESOptions`] and
//! calling [`CMAES::run`] or [`CMAES::run_parallel`]. This option provides the most flexibility and
//! customization and allows for visualizing runs through data plots (see [`Plot`]; requires the
//! `plotters` feature).
//!
//! ```no_run
//! use cmaes::{CMAESOptions, DVector};
//!
//! let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
//!
//! let dim = 10;
//! let mut cmaes_state = CMAESOptions::new(vec![1.0; dim], 1.0)
//!     .fun_target(1e-8)
//!     .enable_printing(200)
//!     .max_generations(20000)
//!     .build(sphere)
//!     .unwrap();
//!
//! let results = cmaes_state.run();
//! ```
//!
//! For many problems, it is useful to perform multiple independent restarts of the algorithm with
//! varying initial parameters. The [`restart`] module implements several automatic restart
//! strategies to this end. To use them, create and build a
//! [`RestartOptions`][crate::restart::RestartOptions] and call
//! [`Restarter::run`][crate::restart::Restarter::run],
//! [`Restarter::run_parallel`][crate::restart::Restarter::run_parallel`], or another `run_*`
//! method. This option provides greater robustness in solving more complex
//! problems but provides less control over individual runs and cannot produce data plots.
//!
//! ```no_run
//! use cmaes::restart::{RestartOptions, RestartStrategy};
//! use cmaes::DVector;
//!
//! let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
//!
//! let dim = 10;
//! let strategy = RestartStrategy::BIPOP(Default::default());
//! let restarter = RestartOptions::new(dim, -5.0..=5.0, strategy)
//!     .fun_target(1e-8)
//!     .enable_printing(true)
//!     .build()
//!     .unwrap();
//!
//! let results = restarter.run_parallel(|| sphere);
//! ```
//!
//! # Further configuration
//!
//! [`CMAESOptions`] provides many configurable parameters and can be used to tune the algorithm to
//! each particular problem (though this is usually unnecessary beyond changing the initial mean and
//! step size).
//!
//! The [`objective_function`] module provides traits that allow for custom objective function types
//! that store state and parameters.
//!
//! The [`CMAES::next`] method provides finer control over iteration if needed.
//!
//! # Citations
//!
//! The following contain more detailed information on the algorithms implemented by this library
//! or were referenced in its implementation.
//!
//! Auger, Anne and Hansen, Nikolaus. “A Restart CMA Evolution Strategy with Increasing Population Size.” 2005 IEEE Congress on Evolutionary Computation, vol. 2, 2005, pp. 1769-1776 Vol. 2, <https://doi.org/10.1109/CEC.2005.1554902>.
//!
//! Hansen, Nikolaus. “Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed.” GECCO (Companion), July 2009, <https://doi.org/10.1145/1570256.1570333>.
//!
//! Auger, Anne, and Nikolaus Hansen. Tutorial CMA-ES. 2013, <https://doi.org/10.1145/2464576.2483910>.
//!
//! Hansen, Nikolaus, Akimoto, Youhei, and Baudis, Petr. CMA-ES/Pycma on Github. Feb. 2019, <https://doi.org/10.5281/zenodo.2559634>.

// lib.rs contains the top-level `CMAES` type that implements the algorithm and interface as well as
// some user-facing types for termination data.
//
// Configuration is handled in the `options` module.
//
// Initialization happens in `CMAES::new`, but initialization of individual components can be found
// in their respective modules.
//
// Iteration happens in `CMAES::next`, but the interesting parts are handled in the `sampling` and
// `state` modules.
//
// Termination criteria are handled in the `termination` module.
//
// Automatic restart algorithms are contained in the `restart` module.

pub mod functions;
mod history;
mod matrix;
mod mode;
pub mod objective_function;
pub mod options;
pub mod parameters;
#[cfg(feature = "plotters")]
pub mod plotting;
pub mod restart;
mod sampling;
mod state;
pub mod termination;
mod utils;

pub use nalgebra::DVector;

pub use crate::functions::*;
pub use crate::history::MAX_HISTORY_LENGTH;
pub use crate::mode::Mode;
pub use crate::objective_function::{ObjectiveFunction, ParallelObjectiveFunction};
pub use crate::options::CMAESOptions;
pub use crate::parameters::Weights;
#[cfg(feature = "plotters")]
pub use crate::plotting::PlotOptions;
pub use crate::sampling::Bounds;
pub use crate::sampling::Constraints;
pub use crate::termination::TerminationReason;

use std::f64;
use std::time::{Duration, Instant};

use crate::history::History;
use crate::matrix::SquareMatrix;
use crate::options::InvalidOptionsError;
use crate::parameters::Parameters;
#[cfg(feature = "plotters")]
use crate::plotting::Plot;
use crate::sampling::{EvaluatedPoint, InvalidFunctionValueError, Sampler};
use crate::state::State;
use crate::termination::TerminationCheck;

/// An individual point with its corresponding objective function value.
#[derive(Clone, Debug)]
pub struct Individual {
    pub point: DVector<f64>,
    pub value: f64,
}

impl Individual {
    fn new(point: DVector<f64>, value: f64) -> Self {
        Self { point, value }
    }
}

/// Data returned when the algorithm terminates.
///
/// Contains the:
///
/// - Best individual of the latest generation
/// - Best individual overall
/// - Final mean, which may be better than either individual
/// - Reasons for termination, which can be used to decide how to interpret the result and
/// whether and how to restart the algorithm
#[derive(Clone, Debug)]
pub struct TerminationData {
    /// Always `Some` unless the algorithm terminated in the first generation with
    /// [`TerminationReason::InvalidFunctionValue`][crate::TerminationReason::InvalidFunctionValue].
    pub current_best: Option<Individual>,
    /// Always `Some` unless the algorithm terminated in the first generation with
    /// [`TerminationReason::InvalidFunctionValue`][crate::TerminationReason::InvalidFunctionValue].
    pub overall_best: Option<Individual>,
    pub final_mean: DVector<f64>,
    pub reasons: Vec<TerminationReason>,
}

/// A type that handles algorithm iteration and printing/plotting of results. Use [`CMAESOptions`]
/// to create a `CMAES`.
///
/// # Type parameters
///
/// Static dispatch for the objective function type is used by default, but if this is not
/// desirable then dynamic dispatch can be used instead:
///
/// ```
/// use cmaes::{CMAES, CMAESOptions, DVector, ObjectiveFunction};
///
/// struct Container(CMAES<Box<dyn ObjectiveFunction>>);
///
/// let mut x = 0.0;
/// let function = move |_: &DVector<f64>| {
///     x += 1.0;
///     x
/// };
/// let cmaes_state = CMAESOptions::new(vec![0.0; 2], 1.0)
///     .build(Box::new(function) as _)
///     .unwrap();
/// let container = Container(cmaes_state);
/// ```
pub struct CMAES<F> {
    /// Point sampler/evaluator
    sampler: Sampler<F>,
    /// Constant parameters
    parameters: Parameters,
    /// Variable state
    state: State,
    /// Objective function value history
    history: History,
    /// Data plot if enabled
    #[cfg(feature = "plotters")]
    plot: Option<Plot>,
    /// The minimum number of function evaluations to wait for in between each automatic
    /// [`CMAES::print_info`] call
    print_gap_evals: Option<usize>,
    /// The last time [`CMAES::print_info`] was called, in function evaluations
    last_print_evals: usize,
    /// The time at which the `CMAES` was created
    time_created: Instant,
}

impl<F> CMAES<F> {
    /// Initializes a `CMAES` from a set of [`CMAESOptions`]. [`CMAESOptions::build`] should
    /// generally be used instead.
    pub fn new(objective_function: F, options: CMAESOptions) -> Result<Self, InvalidOptionsError> {
        let dimensions = options.initial_mean.len();
        // Check for invalid options
        if dimensions == 0 {
            return Err(InvalidOptionsError::Dimensions);
        }

        if options.population_size < 2 {
            return Err(InvalidOptionsError::PopulationSize);
        }

        if !options::is_initial_step_size_valid(options.initial_step_size) {
            return Err(InvalidOptionsError::InitialStepSize);
        }

        if !options.cm.is_normal() || options.cm <= 0.0 || options.cm > 1.0 {
            return Err(InvalidOptionsError::Cm);
        }

        let seed = options.seed.unwrap_or_else(rand::random);

        // Initialize constant parameters according to the options
        let parameters = Parameters::from_options(&options, seed);

        // Initialize point sampler
        let sampler = Sampler::new(
            dimensions,
            options.constraints,
            options.max_resamples,
            options.population_size,
            objective_function,
            seed,
        );

        // Initialize variable parameters
        let state = State::new(options.initial_mean, options.initial_step_size);

        // Initialize function value history
        let history = History::new();

        // Initialize plot if enabled
        #[cfg(feature = "plotters")]
        let plot = options
            .plot_options
            .map(|o| Plot::new(dimensions, o, options.mode));

        let cmaes = Self {
            sampler,
            parameters,
            state,
            history,
            #[cfg(feature = "plotters")]
            plot,
            print_gap_evals: options.print_gap_evals,
            last_print_evals: 0,
            time_created: Instant::now(),
        };

        // Plot initial state
        #[cfg(feature = "plotters")]
        let mut cmaes = cmaes;
        #[cfg(feature = "plotters")]
        cmaes.add_plot_point();

        // Print initial info
        if cmaes.print_gap_evals.is_some() {
            cmaes.print_initial_info();
        }

        Ok(cmaes)
    }

    /// Shared logic between `run` and `run_parallel`
    fn run_internal(&mut self, result: &TerminationData) {
        // Plot/print the final state
        #[cfg(feature = "plotters")]
        self.add_plot_point();

        if self.print_gap_evals.is_some() {
            self.print_final_info(&result.reasons);
        }
    }

    /// Shared logic between `sample` and `sample_parallel`
    fn sample_internal(&mut self, individuals: &[EvaluatedPoint]) {
        // Update histories
        self.history.update(self.parameters.mode(), individuals);
    }

    /// Shared logic between `next` and `next_parallel`
    fn next_internal(&mut self, individuals: &[EvaluatedPoint]) -> Option<TerminationData> {
        // Update state
        if self
            .state
            .update(self.sampler.function_evals(), &self.parameters, individuals)
            .is_err()
        {
            return Some(self.get_termination_data(vec![TerminationReason::PosDefCov]));
        }

        // Plot latest state
        #[cfg(feature = "plotters")]
        if let Some(ref plot) = self.plot {
            // Always plot the first generation
            if self.state.generation() <= 1
                || self.sampler.function_evals() >= plot.get_next_data_point_evals()
            {
                self.add_plot_point();
            }
        }

        // Print latest state
        if let Some(gap_evals) = self.print_gap_evals {
            // The first few generations are always printed, then print_gap_evals is respected
            if self.sampler.function_evals() >= self.last_print_evals + gap_evals {
                self.print_info();
                self.last_print_evals = self.sampler.function_evals();
            } else if self.state.generation() < 4 {
                // Don't update last_print_evals so the printed generation numbers can remain
                // multiples of 10
                self.print_info();
            }
        }

        // Terminate with the current best individual if any termination criteria are met
        let termination_reasons = TerminationCheck {
            current_function_evals: self.sampler.function_evals(),
            time_created: self.time_created,
            parameters: &self.parameters,
            state: &self.state,
            history: &self.history,
            individuals,
        }
        .check_termination_criteria();

        if !termination_reasons.is_empty() {
            Some(self.get_termination_data(termination_reasons))
        } else {
            None
        }
    }

    /// Consumes `self` and returns the objective function. Useful for retrieving state stored in
    /// custom objective function types.
    pub fn into_objective_function(self) -> F {
        self.sampler.into_objective_function()
    }

    /// Returns the constant parameters of the algorithm.
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Returns the number of generations that have been completed.
    pub fn generation(&self) -> usize {
        self.state.generation()
    }

    /// Returns the number of times the objective function has been evaluated.
    pub fn function_evals(&self) -> usize {
        self.sampler.function_evals()
    }

    /// Returns the current mean of the distribution.
    pub fn mean(&self) -> &DVector<f64> {
        self.state.mean()
    }

    /// Returns the current covariance matrix of the distribution.
    pub fn covariance_matrix(&self) -> &SquareMatrix<f64> {
        self.state.cov()
    }

    /// Returns the current eigenvalues of the distribution.
    pub fn eigenvalues(&self) -> DVector<f64> {
        self.state
            .cov_sqrt_eigenvalues()
            .diagonal()
            .map(|x| x.powi(2))
    }

    /// Returns the current step size of the distribution.
    pub fn sigma(&self) -> f64 {
        self.state.sigma()
    }

    /// Returns the current axis ratio of the distribution.
    pub fn axis_ratio(&self) -> f64 {
        self.state.axis_ratio()
    }

    /// Returns the best individual of the latest generation and its function value. Will always
    /// return `Some` as long as [`next`][Self::next] has been called at least once and the
    /// algorithm did not terminate in the first generation with
    /// [`TerminationReason::InvalidFunctionValue`][crate::TerminationReason::InvalidFunctionValue].
    pub fn current_best_individual(&self) -> Option<&Individual> {
        self.history.current_best_individual()
    }

    /// Returns the best individual of any generation and its function value. Will always
    /// return `Some` as long as [`next`][Self::next] has been called at least once and the
    /// algorithm did not terminate in the first generation with
    /// [`TerminationReason::InvalidFunctionValue`][crate::TerminationReason::InvalidFunctionValue].
    pub fn overall_best_individual(&self) -> Option<&Individual> {
        self.history.overall_best_individual()
    }

    /// Returns the time at which the `CMAES` was created.
    pub fn time_created(&self) -> Instant {
        self.time_created
    }

    /// Returns the time elapsed since the `CMAES` was created.
    pub fn elapsed(&self) -> Duration {
        self.time_created.elapsed()
    }

    /// Returns a reference to the data plot if enabled.
    #[cfg(feature = "plotters")]
    pub fn get_plot(&self) -> Option<&Plot> {
        self.plot.as_ref()
    }

    /// Returns a mutable reference to the data plot if enabled.
    #[cfg(feature = "plotters")]
    pub fn get_mut_plot(&mut self) -> Option<&mut Plot> {
        self.plot.as_mut()
    }

    /// Returns how many generations take place for each update of the eigendecomposition.
    ///
    /// For example, if this value is `3`, an eigen update will take place once for every `3`
    /// `next` calls.
    // Only used for benchmarks; probably not useful otherwise
    #[doc(hidden)]
    pub fn generations_per_eigen_update(&self) -> usize {
        // NOTE: will have to be updated if/when fevals per generation != lambda
        (self.state.evals_per_eigen_update(&self.parameters) as f64
            / self.parameters.lambda() as f64)
            .ceil() as usize
    }

    /// Returns a `TerminationData` with the current best individual/value and the given reasons.
    fn get_termination_data(&self, reasons: Vec<TerminationReason>) -> TerminationData {
        return TerminationData {
            current_best: self.current_best_individual().cloned(),
            overall_best: self.overall_best_individual().cloned(),
            final_mean: self.state.mean().clone(),
            reasons,
        };
    }

    /// Adds a data point to the data plot if enabled and not already called this generation. Can be
    /// called manually after termination to plot the final state if [`run`][Self::run] isn't used.
    #[cfg(feature = "plotters")]
    pub fn add_plot_point(&mut self) {
        if let Some(ref mut plot) = self.plot {
            plot.add_data_point(self.sampler.function_evals(), &self.state, &self.history);
        }
    }

    /// Prints various initial parameters of the algorithm as well as the headers for the columns
    /// printed by [`print_info`][Self::print_info]. The parameters that are printed are the:
    ///
    /// - Algorithm variant (based on the [`Weights`] setting)
    /// - Dimension (N)
    /// - Population size (lambda)
    /// - Seed
    ///
    /// This function is called automatically if [`CMAESOptions::enable_printing`] is set.
    pub fn print_initial_info(&self) {
        let params = &self.parameters;
        let variant = match params.weights_setting() {
            Weights::Positive | Weights::Uniform => "CMA-ES",
            Weights::Negative => "aCMA-ES",
        };
        println!(
            "{} with dimension={}, lambda={}, seed={}",
            variant,
            params.dim(),
            params.lambda(),
            params.seed()
        );

        let title_string = format!(
            "{:^7} | {:^7} | {:^19} | {:^10} | {:^10} | {:^10} | {:^10}",
            "Gen #", "f evals", "Best function value", "Axis Ratio", "Sigma", "Min std", "Max std",
        );

        println!("{}", title_string);
        println!("{}", "-".repeat(title_string.chars().count()));
    }

    /// Prints various state variables of the algorithm. The variables that are printed are the:
    ///
    /// - Generations completed
    /// - Function evaluations made
    /// - Best function value of the latest generation
    /// - Distribution axis ratio
    /// - Overall standard deviation (sigma)
    /// - Minimum and maximum standard deviations in the coordinate axes
    ///
    /// This function is called automatically if [`CMAESOptions::enable_printing`] is set.
    pub fn print_info(&self) {
        let generations = format!("{:7}", self.state.generation());
        let evals = format!("{:7}", self.sampler.function_evals());
        let best_function_value = self
            .current_best_individual()
            .map(|x| utils::format_num(x.value, 19))
            .unwrap_or(format!("{:19}", ""));
        let axis_ratio = utils::format_num(self.axis_ratio(), 11);
        let sigma = utils::format_num(self.state.sigma(), 11);
        let cov_diag = self.state.cov().diagonal();
        let min_std = utils::format_num(self.state.sigma() * cov_diag.min().sqrt(), 11);
        let max_std = utils::format_num(self.state.sigma() * cov_diag.max().sqrt(), 11);

        // The preceding space for values that can't have a negative sign is removed (an extra
        // digit takes its place)
        println!(
            "{} | {} | {} |{} |{} |{} |{}",
            generations, evals, best_function_value, axis_ratio, sigma, min_std, max_std
        );
    }

    /// Calls [`print_info`][Self::print_info] if not already called automatically this generation
    /// and prints the results. The values that are printed are the:
    ///
    /// - Termination reasons if given
    /// - Best function value of the latest generation
    /// - Best function value of any generation
    /// - Final distribution mean
    ///
    /// This function is called automatically if [`CMAESOptions::enable_printing`] is set. Must be
    /// called manually after termination to print the final state if [`run`][`Self::run`] isn't
    /// used.
    pub fn print_final_info(&self, termination_reasons: &[TerminationReason]) {
        if self.sampler.function_evals() != self.last_print_evals {
            self.print_info();
        }

        let reasons_str = termination_reasons
            .iter()
            .map(|r| format!("`{}`", r))
            .collect::<Vec<_>>()
            .join(", ");
        println!("Terminated with reason(s): {}", reasons_str);

        let current_best = self.current_best_individual();
        let overall_best = self.overall_best_individual();

        if let (Some(current), Some(overall)) = (current_best, overall_best) {
            println!("Current best function value: {:e}", current.value);
            println!("Overall best function value: {:e}", overall.value);
        }

        println!("Final mean: {}", self.state.mean());
    }
}

impl<F: ObjectiveFunction> CMAES<F> {
    /// Iterates the algorithm until termination. [`next`][Self::next] can be called manually if
    /// more control over termination is needed (plotting/printing the final state must be done
    /// manually as well in this case).
    pub fn run(&mut self) -> TerminationData {
        let result = loop {
            if let Some(data) = self.next() {
                break data;
            }
        };

        self.run_internal(&result);

        result
    }

    /// Samples `lambda` points from the distribution and returns the points sorted by their
    /// objective function values. Also updates the histories of the best and median function
    /// values.
    ///
    /// Returns `Err` if an invalid function value was encountered.
    fn sample(&mut self) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        // Sample points
        let individuals = self.sampler.sample(
            &self.state,
            self.parameters.mode(),
            self.parameters.parallel_update(),
        )?;

        self.sample_internal(&individuals);

        Ok(individuals)
    }

    /// Advances to the next generation. Returns `Some` if a termination condition has been reached
    /// and the algorithm should be stopped. [`run`][Self::run] is generally easier to use, but
    /// iteration can be performed manually if finer control is needed (plotting/printing the final
    /// state must be done manually as well in this case).
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn next(&mut self) -> Option<TerminationData> {
        // Sample individuals
        let individuals = match self.sample() {
            Ok(x) => x,
            Err(_) => {
                return Some(
                    self.get_termination_data(vec![TerminationReason::InvalidFunctionValue]),
                );
            }
        };

        self.next_internal(&individuals)
    }
}

impl<F: ParallelObjectiveFunction> CMAES<F> {
    /// Like [`run`][Self::run], but executes the objective function in parallel using multiple
    /// threads. Requires that `F` implements
    /// [`ParallelObjectiveFunction`][crate::objective_function::ParallelObjectiveFunction].
    ///
    /// Uses [rayon][rayon] internally.
    pub fn run_parallel(&mut self) -> TerminationData {
        let result = loop {
            if let Some(data) = self.next_parallel() {
                break data;
            }
        };

        self.run_internal(&result);

        result
    }

    /// Like `sample`, but evaluates the sampled points using multiple threads
    fn sample_parallel(&mut self) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        let individuals = self.sampler.sample_parallel(
            &self.state,
            self.parameters.mode(),
            self.parameters.parallel_update(),
        )?;

        self.sample_internal(&individuals);

        Ok(individuals)
    }

    /// Like [`next`][Self::next], but executes the objective function in parallel using multiple
    /// threads. Requires that `F` implements
    /// [`ParallelObjectiveFunction`][crate::objective_function::ParallelObjectiveFunction].
    ///
    /// Uses [rayon][rayon] internally.
    pub fn next_parallel(&mut self) -> Option<TerminationData> {
        // Sample individuals
        let individuals = match self.sample_parallel() {
            Ok(x) => x,
            Err(_) => {
                return Some(
                    self.get_termination_data(vec![TerminationReason::InvalidFunctionValue]),
                );
            }
        };

        self.next_internal(&individuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_function(_: &DVector<f64>) -> f64 {
        0.0
    }

    #[test]
    fn test_get_best_individuals() {
        let mut cmaes = CMAESOptions::new(vec![0.0; 10], 1.0)
            .build(dummy_function)
            .unwrap();

        assert!(cmaes.current_best_individual().is_none());
        assert!(cmaes.overall_best_individual().is_none());

        let _ = cmaes.next();

        assert!(cmaes.current_best_individual().is_some());
        assert!(cmaes.overall_best_individual().is_some());
    }

    #[test]
    fn test_immediate_termination() {
        let function = |_: &DVector<f64>| f64::NAN;
        let mut cmaes = CMAESOptions::new(vec![0.0; 10], 1.0)
            .build(function)
            .unwrap();

        let result = cmaes.run();

        assert_eq!(
            vec![TerminationReason::InvalidFunctionValue],
            result.reasons
        );
    }

    #[test]
    fn test_run_final_plot() {
        let evals_per_plot_point = 100;
        let mut cmaes = CMAESOptions::new(vec![0.0; 10], 1.0)
            .enable_plot(PlotOptions::new(evals_per_plot_point, false))
            .max_generations(1)
            .build(dummy_function)
            .unwrap();

        // The initial state is always plotted
        assert_eq!(cmaes.get_plot().unwrap().len(), 1);

        let _ = cmaes.run();

        // The final state is always plotted when using CMAES::run, regardless of
        // evals_per_plot_point
        assert_eq!(cmaes.get_plot().unwrap().len(), 2);
    }

    #[test]
    fn test_generations_per_eigen_update() {
        let cmaes_3 = CMAESOptions::new(vec![0.0; 3], 1.0)
            .build(dummy_function)
            .unwrap();
        let cmaes_10 = CMAESOptions::new(vec![0.0; 10], 1.0)
            .build(dummy_function)
            .unwrap();
        let cmaes_30 = CMAESOptions::new(vec![0.0; 30], 1.0)
            .build(dummy_function)
            .unwrap();

        assert_eq!(2, cmaes_3.generations_per_eigen_update());
        assert_eq!(2, cmaes_10.generations_per_eigen_update());
        assert_eq!(3, cmaes_30.generations_per_eigen_update());
    }

    #[test]
    fn test_evals_per_eigen_update() {
        let cmaes_3 = CMAESOptions::new(vec![0.0; 3], 1.0)
            .build(dummy_function)
            .unwrap();
        let cmaes_10 = CMAESOptions::new(vec![0.0; 10], 1.0)
            .build(dummy_function)
            .unwrap();
        let cmaes_30 = CMAESOptions::new(vec![0.0; 30], 1.0)
            .build(dummy_function)
            .unwrap();

        assert_eq!(
            8,
            cmaes_3.state.evals_per_eigen_update(cmaes_3.parameters())
        );
        assert_eq!(
            15,
            cmaes_10.state.evals_per_eigen_update(cmaes_10.parameters())
        );
        assert_eq!(
            34,
            cmaes_30.state.evals_per_eigen_update(cmaes_30.parameters())
        );
    }
}
