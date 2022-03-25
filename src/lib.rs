//! An implementation of the CMA-ES optimization algorithm. It is used to minimize the value of an
//! objective function and performs well on high-dimension, non-linear, non-convex, ill-conditioned,
//! and/or noisy problems.
//!
//! # Quick Start
//!
//! To optimize a function, simply create and build a [`CMAESOptions`] and call
//! [`CMAES::run`]. Customization of algorithm parameters should be done on a per-problem basis
//! using [`CMAESOptions`]. See [`Plot`] for generation of data plots.
//!
//! ```no_run
//! use cmaes::{CMAESOptions, DVector};
//!
//! let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
//!
//! let dim = 10;
//! let mut cmaes_state = CMAESOptions::new(dim)
//!     .initial_mean(vec![1.0; dim])
//!     .enable_printing(200)
//!     .max_generations(20000)
//!     .build(sphere)
//!     .unwrap();
//!
//! let result = cmaes_state.run();
//! ```
//!
//! The [`objective_function`] module provides a trait that allows for custom objective function
//! types that store state and parameters, and the [`CMAES::next`] method provides finer control
//! over iteration if needed.
//!
//! See [this paper][0] for details on the algorithm itself. This library is based on the linked
//! paper and the [pycma][1] implementation.
//!
//! [0]: https://arxiv.org/pdf/1604.00772.pdf
//! [1]: https://github.com/CMA-ES/pycma

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

mod matrix;
pub mod objective_function;
pub mod options;
pub mod parameters;
pub mod plotting;
mod sampling;
mod state;
pub mod termination;
mod utils;

pub use nalgebra::DVector;

pub use crate::objective_function::ObjectiveFunction;
pub use crate::options::CMAESOptions;
pub use crate::parameters::Weights;
pub use crate::plotting::PlotOptions;
pub use crate::termination::TerminationReason;

use std::collections::VecDeque;
use std::f64;

use crate::matrix::SquareMatrix;
use crate::options::InvalidOptionsError;
use crate::parameters::{Parameters, TerminationParameters};
use crate::plotting::Plot;
use crate::sampling::{EvaluatedPoint, InvalidFunctionValueError, Sampler};
use crate::state::State;

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
/// - Reason for termination, which can be used to decide how to interpret the result and
/// whether and how to restart the algorithm
#[derive(Clone, Debug)]
pub struct TerminationData {
    pub current_best: Individual,
    pub overall_best: Individual,
    pub final_mean: DVector<f64>,
    pub reason: TerminationReason,
}

/// A type that handles algorithm iteration and printing/plotting of results. Use [`CMAESOptions`]
/// to create a `CMAES`.
///
/// # Lifetimes
///
/// The objective function may be non-`'static` (i.e., it borrows something), so there is a lifetime
/// parameter. If this functionality is not needed and the `CMAES` type must be specified
/// somewhere, the lifetime can simply be set to `'static`:
///
/// ```
/// # use cmaes::CMAES;
/// struct Container(CMAES<'static>);
/// ```
///
/// In the case of a closure that references variables from its scope, the `move` keyword can be
/// used to force a static lifetime:
///
/// ```
/// # use cmaes::{CMAESOptions, CMAES, DVector};
/// # struct Container(CMAES<'static>);
/// let mut x = 0.0;
/// let function = move |_: &DVector<f64>| {
///     x += 1.0;
///     x
/// };
/// let cmaes_state = CMAESOptions::new(2).build(function).unwrap();
/// let container = Container(cmaes_state);
/// ```
pub struct CMAES<'a> {
    /// Point sampler/evaluator
    sampler: Sampler<'a>,
    /// Constant parameters
    parameters: Parameters,
    /// Variable state
    state: State,
    /// A history of the best function values of the past k generations (see [`CMAES::next`])
    /// (values at the front are from more recent generations)
    best_function_value_history: VecDeque<f64>,
    /// A history of the median function values of the past k generations (see [`CMAES::next]`)
    /// (values at the front are from more recent generations)
    median_function_value_history: VecDeque<f64>,
    /// The best individual of the latest generation
    current_best_individual: Option<Individual>,
    /// The best individual of any generation
    overall_best_individual: Option<Individual>,
    /// Data plot if enabled
    plot: Option<Plot>,
    /// The minimum number of function evaluations to wait for in between each automatic
    /// [`CMAES::print_info`] call
    print_gap_evals: Option<usize>,
    /// The last time [`CMAES::print_info`] was called, in function evaluations
    last_print_evals: usize,
}

impl<'a> CMAES<'a> {
    /// Initializes a `CMAES` from a set of [`CMAESOptions`]. [`CMAESOptions::build`] should
    /// generally be used instead.
    pub fn new(
        objective_function: Box<dyn ObjectiveFunction + 'a>,
        options: CMAESOptions,
    ) -> Result<Self, InvalidOptionsError> {
        // Check for invalid options
        if options.dimensions == 0 {
            return Err(InvalidOptionsError::Dimensions);
        }

        if options.dimensions != options.initial_mean.len() {
            return Err(InvalidOptionsError::MeanDimensionMismatch);
        }

        if options.population_size < 4 {
            return Err(InvalidOptionsError::PopulationSize);
        }

        if !options.initial_step_size.is_normal() || options.initial_step_size <= 0.0 {
            return Err(InvalidOptionsError::InitialStepSize);
        }

        if !options.cm.is_normal() || options.cm <= 0.0 || options.cm > 1.0 {
            return Err(InvalidOptionsError::Cm);
        }

        // Initialize point sampler
        let seed = options.seed.unwrap_or_else(rand::random);
        let sampler = Sampler::new(
            options.dimensions,
            options.population_size,
            objective_function,
            seed,
        );

        // Initialize constant parameters according to the options
        let tol_x = options.tol_x.unwrap_or(1e-12 * options.initial_step_size);
        let termination_parameters = TerminationParameters {
            max_function_evals: options.max_function_evals,
            max_generations: options.max_generations,
            tol_fun: options.tol_fun,
            tol_x,
        };
        let parameters = Parameters::new(
            options.dimensions,
            options.population_size,
            options.weights,
            seed,
            options.initial_step_size,
            options.cm,
            termination_parameters,
        );

        // Initialize variable parameters
        let state = State::new(options.initial_mean, options.initial_step_size);

        // Initialize plot if enabled
        let plot = options
            .plot_options
            .map(|o| Plot::new(options.dimensions, o));

        let mut cmaes = Self {
            sampler,
            parameters,
            state,
            best_function_value_history: VecDeque::new(),
            median_function_value_history: VecDeque::new(),
            current_best_individual: None,
            overall_best_individual: None,
            plot,
            print_gap_evals: options.print_gap_evals,
            last_print_evals: 0,
        };

        // Plot initial state
        cmaes.add_plot_point();

        // Print initial info
        if cmaes.print_gap_evals.is_some() {
            cmaes.print_initial_info();
        }

        Ok(cmaes)
    }

    /// Iterates the algorithm until termination. [`next`][Self::next] can be called manually if
    /// more control over termination is needed (plotting/printing the final state must be done
    /// manually as well in this case).
    pub fn run(&mut self) -> TerminationData {
        let result = loop {
            if let Some(data) = self.next() {
                break data;
            }
        };

        // Plot/print the final state
        self.add_plot_point();

        if self.print_gap_evals.is_some() {
            self.print_final_info(result.reason);
        }

        result
    }

    /// Samples `lambda` points from the distribution and returns the points sorted by their
    /// objective function values. Also updates the histories of the best and median function
    /// values.
    ///
    /// Returns `Err` if an invalid function value was encountered.
    fn sample(
        &mut self,
        max_history_size: usize,
    ) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        // Sample points
        let individuals = self.sampler.sample(&self.state)?;

        // Update histories of best and median values
        let best = &individuals[0];

        self.best_function_value_history.push_front(best.value());
        if self.best_function_value_history.len() > max_history_size {
            self.best_function_value_history.pop_back();
        }
        // Not perfectly accurate but it shouldn't make a difference
        let median = &individuals[individuals.len() / 2];
        self.median_function_value_history
            .push_front(median.value());
        if self.median_function_value_history.len() > max_history_size {
            self.median_function_value_history.pop_back();
        }

        self.update_best_individuals(Individual::new(best.point().clone(), best.value()));

        Ok(individuals)
    }

    /// Advances to the next generation. Returns `Some` if a termination condition has been reached
    /// and the algorithm should be stopped. [`run`][Self::run] is generally easier to use, but
    /// iteration can be performed manually if finer control is needed (plotting/printing the final
    /// state must be done manually as well in this case).
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn next(&mut self) -> Option<TerminationData> {
        let dim = self.parameters.dim();
        let lambda = self.parameters.lambda();

        // How many generations to store in self.best_function_value_history and
        // self.median_function_value
        // This is the largest history size required by any termination criterion
        let max_history_size = ((0.2 * self.state.generation() as f64).ceil() as usize)
            .max(120 + (30.0 * dim as f64 / lambda as f64).ceil() as usize)
            .min(20000);

        // Sample individuals
        let individuals = match self.sample(max_history_size) {
            Ok(x) => x,
            Err(_) => {
                return Some(self.get_termination_data(TerminationReason::InvalidFunctionValue));
            }
        };

        // Update state
        if self
            .state
            .update(
                self.sampler.function_evals(),
                &self.parameters,
                &individuals,
            )
            .is_err()
        {
            return Some(self.get_termination_data(TerminationReason::PosDefCov));
        }

        // Plot latest state
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

        // Terminate with the current best individual if any termination criterion is met
        let termination_reason = termination::check_termination_criteria(
            self.sampler.function_evals(),
            &self.parameters,
            &self.state,
            &self.best_function_value_history,
            &self.median_function_value_history,
            max_history_size,
            &individuals,
        );

        termination_reason.map(|reason| self.get_termination_data(reason))
    }

    /// Updates the current and overall best individuals.
    fn update_best_individuals(&mut self, current_best: Individual) {
        self.current_best_individual = Some(current_best.clone());

        match &mut self.overall_best_individual {
            Some(ref mut overall) => {
                if current_best.value < overall.value {
                    *overall = current_best;
                }
            }
            None => self.overall_best_individual = Some(current_best),
        }
    }

    /// Consumes `self` and returns the objective function.
    pub fn into_objective_function(self) -> Box<dyn ObjectiveFunction + 'a> {
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
    /// return `Some` as long as [`next`][Self::next] has been called at least once.
    pub fn current_best_individual(&self) -> Option<&Individual> {
        self.current_best_individual.as_ref()
    }

    /// Returns the best individual of any generation and its function value. Will always return
    /// `Some` as long as [`next`][Self::next] has been called at least once.
    pub fn overall_best_individual(&self) -> Option<&Individual> {
        self.overall_best_individual.as_ref()
    }

    /// Returns a reference to the data plot if enabled.
    pub fn get_plot(&self) -> Option<&Plot> {
        self.plot.as_ref()
    }

    /// Returns a mutable reference to the data plot if enabled.
    pub fn get_mut_plot(&mut self) -> Option<&mut Plot> {
        self.plot.as_mut()
    }

    /// Returns a `TerminationData` with the current best individual/value and the given reason.
    fn get_termination_data(&self, reason: TerminationReason) -> TerminationData {
        return TerminationData {
            current_best: self.current_best_individual().unwrap().clone(),
            overall_best: self.overall_best_individual().unwrap().clone(),
            final_mean: self.state.mean().clone(),
            reason,
        };
    }

    /// Adds a data point to the data plot if enabled and not already called this generation. Can be
    /// called manually after termination to plot the final state if [`run`][Self::run] isn't used.
    pub fn add_plot_point(&mut self) {
        if let Some(ref mut plot) = self.plot {
            plot.add_data_point(
                self.sampler.function_evals(),
                &self.state,
                self.current_best_individual.as_ref(),
            );
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
    /// - Termination reason if given
    /// - Best function value of the latest generation
    /// - Best function value of any generation
    /// - Final distribution mean
    ///
    /// This function is called automatically if [`CMAESOptions::enable_printing`] is set. Must be
    /// called manually after termination to print the final state if [`run`][`Self::run`] isn't
    /// used.
    pub fn print_final_info(&self, termination_reason: TerminationReason) {
        if self.sampler.function_evals() != self.last_print_evals {
            self.print_info();
        }

        println!("Terminated with reason `{}`", termination_reason);

        let current_best = self.current_best_individual();
        let overall_best = self.overall_best_individual();

        if let (Some(current), Some(overall)) = (current_best, overall_best) {
            println!("Current best function value: {:e}", current.value);
            println!("Overall best function value: {:e}", overall.value);
        }

        println!("Final mean: {}", self.state.mean());
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
        let mut cmaes = CMAESOptions::new(10).build(dummy_function).unwrap();

        assert!(cmaes.current_best_individual().is_none());
        assert!(cmaes.overall_best_individual().is_none());

        let _ = cmaes.next();

        assert!(cmaes.current_best_individual().is_some());
        assert!(cmaes.overall_best_individual().is_some());
    }

    #[test]
    fn test_update_best_individuals() {
        let dim = 10;
        let mut cmaes = CMAESOptions::new(dim).build(dummy_function).unwrap();
        let origin = DVector::from(vec![0.0; dim]);

        cmaes.update_best_individuals(Individual::new(origin.clone(), 1.0));
        assert_eq!(cmaes.current_best_individual().unwrap().value, 1.0);
        assert_eq!(cmaes.overall_best_individual().unwrap().value, 1.0);

        cmaes.update_best_individuals(Individual::new(origin.clone(), 2.0));
        assert_eq!(cmaes.current_best_individual().unwrap().value, 2.0);
        assert_eq!(cmaes.overall_best_individual().unwrap().value, 1.0);

        cmaes.update_best_individuals(Individual::new(origin.clone(), 1.5));
        assert_eq!(cmaes.current_best_individual().unwrap().value, 1.5);
        assert_eq!(cmaes.overall_best_individual().unwrap().value, 1.0);

        cmaes.update_best_individuals(Individual::new(origin.clone(), 0.5));
        assert_eq!(cmaes.current_best_individual().unwrap().value, 0.5);
        assert_eq!(cmaes.overall_best_individual().unwrap().value, 0.5);
    }

    #[test]
    fn test_run_final_plot() {
        let evals_per_plot_point = 100;
        let mut cmaes = CMAESOptions::new(10)
            .enable_plot(PlotOptions::new(evals_per_plot_point, false))
            .max_generations(1)
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();

        // The initial state is always plotted
        assert_eq!(cmaes.get_plot().unwrap().len(), 1);

        let _ = cmaes.run();

        // The final state is always plotted when using CMAES::run, regardless of
        // evals_per_plot_point
        assert_eq!(cmaes.get_plot().unwrap().len(), 2);
    }
}
