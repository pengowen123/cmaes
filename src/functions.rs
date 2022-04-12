//! Convenience functions for easier use of the library.

use nalgebra::DVector;

use crate::{CMAESOptions, Individual, ObjectiveFunction, ParallelObjectiveFunction};

/// Optimizes the value of `f` and returns the best solution found.
///
/// Equivalent to simply using the default [`CMAESOptions`][crate::CMAESOptions] with printing
/// enabled. [`CMAESOptions`][crate::CMAESOptions] should be used instead if further configuration
/// is desired.
///
/// # Examples
///
/// ```no_run
/// use cmaes::DVector;
///
/// let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
/// let dim = 10;
/// let solution = cmaes::fmin(sphere, vec![5.0; dim], 1.0);
/// ```
pub fn fmin<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where F: ObjectiveFunction,
      V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .enable_printing(200)
        .build(f)
        .unwrap()
        .run()
        .overall_best
        .unwrap()
}

/// Like [`fmin`], but executes the objective function in parallel using multiple threads.
///
/// Requires that `F` implements [`ParallelObjectiveFunction`][crate::ParallelObjectiveFunction].
///
/// # Examples
///
/// ```no_run
/// use cmaes::DVector;
///
/// let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
/// let dim = 10;
/// let solution = cmaes::fmin_parallel(sphere, vec![5.0; dim], 1.0);
/// ```
pub fn fmin_parallel<F, V>(f: F, initial_mean: V, initial_step_size: f64) -> Individual
where F: ParallelObjectiveFunction,
      V: Into<DVector<f64>>,
{
    CMAESOptions::new(initial_mean, initial_step_size)
        .enable_printing(200)
        .build(f)
        .unwrap()
        .run_parallel()
        .overall_best
        .unwrap()
}
