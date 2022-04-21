//! Convenience functions for easier use of the library.

use nalgebra::DVector;

use crate::{CMAESOptions, Individual, Mode, ObjectiveFunction, ParallelObjectiveFunction};

const PRINT_GAP_EVALS: usize = 200;
const INVALID_OPTIONS: &str = "Invalid options";
const INVALID_VALUE: &str = "Objective function returned an invalid value";

/// Minimizes the value of `f` and returns the best solution found.
///
/// Equivalent to simply using the default [`CMAESOptions`][crate::CMAESOptions] with printing
/// enabled. [`CMAESOptions`][crate::CMAESOptions] should be used instead if further configuration
/// is desired.
///
/// # Examples
///
/// ```
/// use cmaes::DVector;
///
/// let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
/// let dim = 10;
/// let solution = cmaes::fmin(sphere, vec![5.0; dim], 1.0);
/// ```
pub fn fmin<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where
    F: ObjectiveFunction,
    V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .enable_printing(PRINT_GAP_EVALS)
        .build(f)
        .expect(INVALID_OPTIONS)
        .run()
        .overall_best
        .expect(INVALID_VALUE)
}

/// Like [`fmin`], but executes the objective function in parallel using multiple threads.
///
/// Requires that `F` implements [`ParallelObjectiveFunction`][crate::ParallelObjectiveFunction].
///
/// # Examples
///
/// ```
/// use cmaes::DVector;
///
/// let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
/// let dim = 10;
/// let solution = cmaes::fmin_parallel(sphere, vec![5.0; dim], 1.0);
/// ```
pub fn fmin_parallel<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where
    F: ParallelObjectiveFunction,
    V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .enable_printing(PRINT_GAP_EVALS)
        .build(f)
        .expect(INVALID_OPTIONS)
        .run_parallel()
        .overall_best
        .expect(INVALID_VALUE)
}

/// Maximizes the value of `f` and returns the best solution found.
///
/// Equivalent to simply using the default [`CMAESOptions`][crate::CMAESOptions] with the mode set
/// to [`Mode::Maximize`][crate::Mode::Maximize] and printing enabled.
/// [`CMAESOptions`][crate::CMAESOptions] should be used instead if further configuration is
/// desired.
///
/// # Examples
///
/// ```
/// use cmaes::DVector;
///
/// let function = |x: &DVector<f64>| 1.0 / x.magnitude();
/// let dim = 10;
/// let solution = cmaes::fmax(function, vec![5.0; dim], 1.0);
/// ```
pub fn fmax<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where
    F: ObjectiveFunction,
    V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .mode(Mode::Maximize)
        .enable_printing(PRINT_GAP_EVALS)
        .build(f)
        .expect(INVALID_OPTIONS)
        .run()
        .overall_best
        .expect(INVALID_VALUE)
}

/// Like [`fmax`], but executes the objective function in parallel using multiple threads.
///
/// Requires that `F` implements [`ParallelObjectiveFunction`][crate::ParallelObjectiveFunction].
///
/// # Examples
///
/// ```
/// use cmaes::DVector;
///
/// let function = |x: &DVector<f64>| 1.0 / x.magnitude();
/// let dim = 10;
/// let solution = cmaes::fmax_parallel(function, vec![5.0; dim], 1.0);
/// ```
pub fn fmax_parallel<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where
    F: ParallelObjectiveFunction,
    V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .mode(Mode::Maximize)
        .enable_printing(PRINT_GAP_EVALS)
        .build(f)
        .expect(INVALID_OPTIONS)
        .run_parallel()
        .overall_best
        .expect(INVALID_VALUE)
}
