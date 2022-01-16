//! An implementation of the CMA-ES optimization algorithm. It is used to minimize the value of an
//! objective function, and performs well on high-dimension, non-linear, non-convex,
//! ill-conditioned, and/or noisy problems.
//!
//! // TODO: example
//!
//! See [this paper][0] for details on the algorithm itself.
//!
//! [0]: https://arxiv.org/pdf/1604.00772.pdf

pub mod options;

pub use options::{CMAESOptions, Weights};

use nalgebra::{DMatrix, DVector};
use statistical;

use std::cmp::Ordering;
use std::collections::VecDeque;

use options::InvalidOptionsError;

/// Data returned when the algorithm terminates.
///
/// Contains the best individual and its corresponding function value as of the latest generation.
/// Also contains the reason for termination, which can be matched on to decide whether and how to
/// restart the algorithm.
#[derive(Clone, Debug)]
pub struct TerminationData {
    pub best_individual: DVector<f64>,
    pub best_function_value: f64,
    pub reason: TerminationReason,
}

/// Represents the reason for the algorithm terminating. Most of these are for preventing numerical
/// instability, while `TolFun` and `TolX` are problem-dependent parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    /// All function values of the latest generation and the range of the best function values of
    /// many consecutive generations lie below `tol_fun`.
    TolFun,
    /// The standard deviation of the distribution is smaller than `tol_x` in every coordinate and
    /// `step_size * p_c` is smaller than `tol_x` in every coordinate. Indicates that the algorithm
    /// has converged.
    TolX,
    /// The range of best function values in many consecutive generations is approximately zero
    /// (i.e.no improvement is occurring).
    EqualFunValues,
    /// The best and median function values have not improved significantly over many generations.
    Stagnation,
    /// The maximum step size across all dimensions increased by a factor of more than `10^4` in a
    /// single generation. This is likely due to the function diverging or the initial step size
    /// being set far too small, in which case a restart with a larger step size may be useful.
    TolXUp,
    /// The standard deviation in any principal axis of the covariance matrix is extremely small.
    NoEffectAxis,
    /// The standard deviation in any coordinate axis is extremely small.
    NoEffectCoord,
    /// The condition number of the covariance matrix exceeds `10^14`.
    ConditionCov,
}

/// Stores constant parameters for the algorithm.
#[derive(Clone, Debug)]
struct Parameters {
    /// Number of dimensions to search
    dim: usize,
    /// Population size,
    lambda: usize,
    /// Number of individuals to select each generation
    mu: usize,
    /// Variance-effective selection mass
    mu_eff: f64,
    /// Individual weights
    weights: DVector<f64>,
    /// Learning rate for rank-one update cumulation
    cc: f64,
    /// Learning rate for rank-one update
    c1: f64,
    /// Learning rate for step size update
    cs: f64,
    /// Learning rate for rank-mu update
    cmu: f64,
    /// Damping parameter for step size update
    damp_s: f64,
    /// Value for the TolFun termination criterion
    tol_fun: f64,
    /// Value for the TolX termination criterion
    tol_x: f64,
}

/// Stores the iteration state of and runs the algorithm. Use [`CMAESOptions`] to create a
/// `CMAESState`.
pub struct CMAESState<F> {
    /// The objective function to minimize
    function: F,
    /// Constant parameters, see [Parameters]
    parameters: Parameters,
    /// Generation number
    generation: usize,
    /// The distribution mean
    mean: DVector<f64>,
    /// The distribution covariance matrix
    cov: DMatrix<f64>,
    /// Normalized eigenvectors of `cov`, forming an orthonormal basis of the matrix
    cov_eigenvectors: DMatrix<f64>,
    /// Diagonal matrix containing the square roots of the eigenvalues of `cov`, which are the
    /// scales of the basis axes
    cov_sqrt_eigenvalues: DMatrix<f64>,
    /// The distribution step size
    sigma: f64,
    /// Evolution path of the mean used to update the covariance matrix
    path_c: DVector<f64>,
    /// Evolution path of the mean used to update the step size
    path_sigma: DVector<f64>,
    /// A history of the best function values of the past k generations (see [`CMAESState::next`])
    /// (values at the front are from more recent generations)
    best_function_values: VecDeque<f64>,
    /// A history of the median function values of the past k generations (see [`CMAESState::next]`)
    /// (Values at the front are from more recent generations)
    median_function_values: VecDeque<f64>,
    /// The current best individual
    current_best_individual: Option<DVector<f64>>,
}

impl<F: Fn(&DVector<f64>) -> f64> CMAESState<F> {
    /// Initializes a `CMAESState` from a set of [`CMAESOptions`]. [`CMAESOptions::build`] should
    /// generally be used instead.
    pub fn new(options: CMAESOptions<F>) -> Result<Self, InvalidOptionsError> {
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

        if options.initial_step_size <= 0.0 {
            return Err(InvalidOptionsError::InitialStepSize);
        }

        // Initialize constant parameters according to the options
        let dim = options.dimensions;
        let lambda = options.population_size;
        let mu = lambda / 2;

        // Set initial weights
        // They will be normalized later
        let mut weights: DVector<f64> = match options.weights {
            // weights.len() == mu
            Weights::Uniform => vec![1.0; mu],
            // weights.len() == mu
            Weights::Positive => (1..=mu)
                .map(|i| (mu as f64 + 0.5).log10() - (i as f64).log10())
                .collect::<Vec<_>>(),
            // weights.len() == lambda
            Weights::Negative => (1..=lambda)
                .map(|i| (mu as f64 + 0.5).log10() - (i as f64).log10())
                .collect::<Vec<_>>(),
        }
        .into();

        // Square of sum divided by sum of squares of the first mu weights (all positive weights)
        let mu_eff = weights.iter().take(mu).sum::<f64>().powi(2)
            / weights.iter().take(mu).map(|w| w.powi(2)).sum::<f64>();

        // Covariance matrix adaptation
        let a_cov = 2.0;
        let cc = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
        let c1 = a_cov / ((dim as f64 + 1.3).powi(2) + mu_eff);
        let cmu = (1.0 - c1).min(
            a_cov * (mu_eff - 2.0 + 1.0 / mu_eff)
                / ((dim as f64 + 2.0).powi(2) + a_cov * mu_eff / 2.0),
        );

        // Normalize the sum of the weights
        // All positive weights will sum to 1, while the sum of the negative weights is chosen below

        // Like mu_eff but for all negative weights (those past the first mu)
        let mu_eff_minus = weights.iter().skip(mu).sum::<f64>().powi(2)
            / weights.iter().skip(mu).map(|w| w.powi(2)).sum::<f64>();

        // Possible sums of negative weights
        // The smallest of these values will be used
        let a_mu = 1.0 + c1 / cmu;
        let a_mu_eff = 1.0 + (2.0 * mu_eff_minus) / (mu_eff + 2.0);
        let a_pos_def = (1.0 - c1 - cmu) / (dim as f64 * cmu);
        let a = a_mu.min(a_mu_eff.min(a_pos_def));

        // Sums for normalization
        let sum_positive_weights = weights.iter().filter(|w| **w > 0.0).sum::<f64>();
        let sum_negative_weights = weights.iter().filter(|w| **w < 0.0).sum::<f64>().abs();

        for w in &mut weights {
            if *w > 0.0 {
                *w /= sum_positive_weights;
            } else {
                *w = *w * a / sum_negative_weights;
            }
        }

        // Step size adaptation
        let cs = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let damp_s = 1.0 + cs + 2.0 * (((mu_eff - 1.0) / (dim as f64 + 1.0)).sqrt() - 1.0).max(0.0);

        let parameters = Parameters {
            dim,
            lambda,
            mu,
            mu_eff,
            weights,
            cc,
            c1,
            cs,
            cmu,
            damp_s,
            tol_fun: options.tol_fun,
            tol_x: options.tol_x.unwrap_or(1e-10 * options.initial_step_size),
        };

        // Initialize variable parameters
        let mean = options.initial_mean;
        let cov = DMatrix::identity(options.dimensions, options.dimensions);
        let eigen = decompose_cov(cov.clone());
        let cov_eigenvectors = eigen.eigenvectors;
        let cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        let sigma = options.initial_step_size;
        let path_c = DVector::zeros(options.dimensions);
        let path_sigma = DVector::zeros(options.dimensions);

        Ok(Self {
            function: options.function,
            parameters,
            generation: 0,
            mean,
            cov,
            cov_eigenvectors,
            cov_sqrt_eigenvalues,
            sigma,
            path_c,
            path_sigma,
            best_function_values: VecDeque::new(),
            median_function_values: VecDeque::new(),
            current_best_individual: None,
        })
    }

    /// Iterates the algorithm until termination or until `max_generations` is reached. If no
    /// termination criteria are met before `max_generations` is reached, `None` will be returned.
    /// [`Self::next`] can be called manually if more control over termination is needed.
    pub fn run(&mut self, max_generations: usize) -> Option<TerminationData> {
        for _ in 0..max_generations {
            if let Some(data) = self.next() {
                return Some(data);
            }
        }

        None
    }

    /// Advances to the next generation. Returns `Some` if a termination condition has been reached
    /// and the algorithm should be stopped.
    pub fn next(&mut self) -> Option<TerminationData> {
        let Parameters {
            dim,
            lambda,
            mu,
            mu_eff,
            ref weights,
            cc,
            c1,
            cs,
            cmu,
            damp_s,
            ..
        } = self.parameters;

        let Self {
            ref mut mean,
            ref mut cov,
            ref mut cov_eigenvectors,
            ref mut cov_sqrt_eigenvalues,
            ref mut sigma,
            ref mut path_c,
            ref mut path_sigma,
            ..
        } = self;

        // Used when checking termination criteria
        let old_max_standard_deviation = *sigma
            * cov_sqrt_eigenvalues
                .diagonal()
                .iter()
                .max_by(|a, b| partial_cmp(*a, *b))
                .unwrap();

        let individual_values = Vec::<f64>::new();

        // How many generations to store in self.best_function_values and
        // self.median_function_values
        let past_generations_to_store = ((0.2 * self.generation as f64).ceil() as usize)
            .max(120 + (30.0 * dim as f64 / lambda as f64).ceil() as usize)
            .min(20000);

        // Terminate with the current best individual if any termination criterion is met
        let termination_reason = self.check_termination_criteria(
            &individual_values,
            past_generations_to_store,
            old_max_standard_deviation,
        );
        if let Some(reason) = termination_reason {
            let (best_individual, best_function_value) =
                self.get_current_best_individual().unwrap();

            return Some(TerminationData {
                best_individual: best_individual.clone(),
                best_function_value,
                reason,
            });
        }

        self.generation += 1;

        None
    }

    /// Returns `Some` if any termination criterion is met.
    fn check_termination_criteria(
        &self,
        individual_values: &[f64],
        past_generations_to_store: usize,
        old_max_standard_deviation: f64,
    ) -> Option<TerminationReason> {
        let Parameters {
            dim,
            lambda,
            tol_fun,
            tol_x,
            ..
        } = self.parameters;

        let Self {
            ref generation,
            ref cov,
            ref cov_eigenvectors,
            ref cov_sqrt_eigenvalues,
            ref sigma,
            ref path_c,
            ..
        } = self;

        // See TerminationReason for what properties these are checking
        const EPSILON: f64 = 1e-10;

        // Check TerminationReason::TolFun
        let past_generations_a = 10 + (30.0 * dim as f64 / lambda as f64).ceil() as usize;
        let mut range_past_generations_a = None;

        if self.best_function_values.len() >= past_generations_a {
            let max = self
                .best_function_values
                .iter()
                .take(past_generations_a)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap();

            let min = self
                .best_function_values
                .iter()
                .take(past_generations_a)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap();

            let range = (max - min).abs();
            range_past_generations_a = Some(range);

            if range < tol_fun && individual_values.iter().all(|y| *y < tol_fun) {
                return Some(TerminationReason::TolFun);
            }
        }

        // Check TerminationReason::TolX
        if (0..dim).all(|i| *sigma * cov[(i, i)] < tol_x)
            && path_c.iter().all(|x| (*sigma * *x).abs() < tol_x)
        {
            return Some(TerminationReason::TolX);
        }

        // Check TerminationReason::ConditionCov
        let diag = cov_sqrt_eigenvalues.diagonal();
        let max_eigenvalue = diag
            .iter()
            .max_by(|a, b| partial_cmp(*a, *b))
            .unwrap()
            .powi(2);
        let min_eigenvalue = diag
            .iter()
            .min_by(|a, b| partial_cmp(*a, *b))
            .unwrap()
            .powi(2);
        let cond = (max_eigenvalue / min_eigenvalue).abs();

        if !cond.is_normal() || cond > 1e14 {
            return Some(TerminationReason::ConditionCov);
        }

        // Check TerminationReason::NoEffectAxis
        // Cycles from 0 to n-1 to avoid checking every column every iteration
        let index_to_check = *generation % dim;

        let no_effect_axis_check = 0.1
            * *sigma
            * cov_sqrt_eigenvalues[(index_to_check, index_to_check)]
            * cov_eigenvectors.column(index_to_check);

        if no_effect_axis_check.magnitude() < EPSILON {
            return Some(TerminationReason::NoEffectAxis);
        }

        // Check TerminationReason::NoEffectCoord
        if (0..dim).any(|i| 0.2 * *sigma * cov[(i, i)].abs() < EPSILON) {
            return Some(TerminationReason::NoEffectCoord);
        }

        // Check TerminationReason::EqualFunValues
        if let Some(range) = range_past_generations_a {
            if range < EPSILON {
                return Some(TerminationReason::EqualFunValues);
            }
        }

        // Check TerminationReason::Stagnation
        let past_generations_b = past_generations_to_store;

        if self.best_function_values.len() >= past_generations_b
            && self.median_function_values.len() >= past_generations_b
        {
            // Checks whether the median of the values has improved over the past
            // `past_generations_b` generations
            // Returns false if the values either did not improve significantly or became worse
            let did_values_improve = |values: &VecDeque<f64>| {
                let subrange_length = (past_generations_b as f64 * 0.3) as usize;

                // Most recent `subrange_length `values within the past `past_generations_b`
                // generations
                let first_values = values
                    .iter()
                    .take(past_generations_b)
                    .take(subrange_length)
                    .cloned()
                    .collect::<Vec<_>>();

                // Least recent `subrange_length` values within the past `past_generations_b`
                // generations
                let last_values = values
                    .iter()
                    .take(past_generations_b)
                    .skip(past_generations_b - subrange_length)
                    .cloned()
                    .collect::<Vec<_>>();

                statistical::median(&first_values) <= statistical::median(&last_values) - EPSILON
            };

            if !did_values_improve(&self.best_function_values)
                && !did_values_improve(&self.median_function_values)
            {
                return Some(TerminationReason::Stagnation);
            }
        }

        // Check TerminationReason::TolXUp
        let new_max_standard_deviation = *sigma
            * cov_sqrt_eigenvalues
                .diagonal()
                .iter()
                .max_by(|a, b| partial_cmp(*a, *b))
                .unwrap();

        if (new_max_standard_deviation / old_max_standard_deviation) > 1e4 {
            return Some(TerminationReason::TolXUp);
        }

        None
    }

    /// Returns the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the current best individual and its function value. Will always return `Some` as
    /// long as [`Self::next`] has been called at least once.
    pub fn get_current_best_individual(&self) -> Option<(&DVector<f64>, f64)> {
        self.best_function_values
            .front()
            .map(|x| (self.current_best_individual.as_ref().unwrap(), *x))
    }
}

/// Used for finding max/min values
fn partial_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

// Decomposition of a covariance matrix
struct CovDecomposition {
    // Columns are eigenvectors
    eigenvectors: DMatrix<f64>,
    // Diagonal matrix with square roots of eigenvalues
    sqrt_eigenvalues: DMatrix<f64>,
}

// Decomposes a covariance matrix into a set of normalized eigenvectors and a diagonal matrix
// containing the square roots of the corresponding eigenvalues
fn decompose_cov(matrix: DMatrix<f64>) -> CovDecomposition {
    let mut eigen = matrix.symmetric_eigen();

    for mut col in eigen.eigenvectors.column_iter_mut() {
        col.normalize_mut();
    }

    CovDecomposition {
        eigenvectors: eigen.eigenvectors,
        sqrt_eigenvalues: DMatrix::from_diagonal(&eigen.eigenvalues.map(|x| x.sqrt())),
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::{DMatrix, DVector};

    use super::decompose_cov;
    use crate::{CMAESOptions, TerminationReason, Weights};

    fn dummy_function(_: &DVector<f64>) -> f64 {
        0.0
    }

    #[test]
    fn test_decompose_cov() {
        let matrix = DMatrix::from_iterator(2, 2, [3.0, 1.5, 1.5, 2.0]);

        let eigen = decompose_cov(matrix.clone());

        let reconstructed = eigen.eigenvectors.clone()
            * eigen.sqrt_eigenvalues.pow(2)
            * eigen.eigenvectors.transpose();

        for x in (reconstructed - matrix).iter() {
            assert_approx_eq!(x, 0.0);
        }
    }

    // Tests that Weights::Positive produces only positive weight values and that they are
    // normalized properly
    #[test]
    fn test_weights_positive() {
        for n in 1..10 {
            let cmaes_state = CMAESOptions::new(dummy_function, 3)
                .weights(Weights::Positive)
                .population_size(n * 4)
                .build()
                .unwrap();

            assert!(cmaes_state.parameters.weights.iter().all(|w| *w > 0.0));
            assert_approx_eq!(cmaes_state.parameters.weights.iter().sum::<f64>(), 1.0);
        }
    }

    // Tests that Weights::Negative produces only positive values for the first mu weights and only
    // negative values for the rest
    #[test]
    fn test_weights_negative() {
        for n in 1..10 {
            let cmaes_state = CMAESOptions::new(dummy_function, 3)
                .weights(Weights::Negative)
                .population_size(n * 4)
                .build()
                .unwrap();

            assert!(cmaes_state
                .parameters
                .weights
                .iter()
                .take(cmaes_state.parameters.mu)
                .all(|w| *w > 0.0));

            assert!(cmaes_state
                .parameters
                .weights
                .iter()
                .skip(cmaes_state.parameters.mu)
                .all(|w| *w < 0.0));
        }
    }

    #[test]
    fn test_check_termination_criteria_none() {
        let cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5),
            None,
        );

        // A large standard deviation in one axis should not meet any termination criteria on its
        // own
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        cmaes_state.cov = DMatrix::from_iterator(2, 2, [0.01, 0.0, 0.0, 1e5]);
        let eigen = decompose_cov(cmaes_state.cov.clone());
        cmaes_state.cov_eigenvectors = eigen.eigenvectors;
        cmaes_state.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, 50000.0),
            None,
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun() {
        // A population of below-tolerance function values and a small range of historical values
        // produces TolFun
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        cmaes_state.best_function_values.extend(vec![1.0; 100]);
        cmaes_state.best_function_values.push_front(1.0 + 1e-12);
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5),
            Some(TerminationReason::TolFun),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x() {
        // A small step size and evolution path produces TolX
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        cmaes_state.sigma = 1e-13;
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5),
            Some(TerminationReason::TolX),
        );
    }

    #[test]
    fn test_check_termination_criteria_equal_fun_values() {
        // A small range of historical values produces EqualFunValues
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        cmaes_state.best_function_values.extend(vec![1.0; 100]);
        assert_eq!(
            cmaes_state.check_termination_criteria(&[1.0; 100], 100, 0.5),
            Some(TerminationReason::EqualFunValues),
        );
    }

    #[test]
    fn test_check_termination_criteria_stagnation() {
        // Median/best function values that change but hover around the same point for many
        // generations produces Stagnation
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        let mut values = Vec::new();
        values.extend(vec![1.0; 20]);
        values.extend(vec![2.0; 20]);
        values.extend(vec![2.0; 20]);
        values.extend(vec![1.0; 20]);
        cmaes_state.best_function_values.extend(values.clone());
        cmaes_state.median_function_values.extend(values.clone());
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 80, 0.5),
            Some(TerminationReason::Stagnation),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x_up() {
        // A large increase in maximum standard deviation produces TolXUp
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        let old_max_standard_deviation = 0.1;
        cmaes_state.sigma = 1e5;
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, old_max_standard_deviation),
            Some(TerminationReason::TolXUp),
        );
    }

    #[test]
    fn test_check_termination_criteria_no_effect_axis() {
        // A small standard deviation (eigenvalue) in a principal axis of the covariance matrix
        // (eigenvector) produces NoEffectAxis
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        let eigenvectors = [
            DVector::from(vec![3.0, 2.0]).normalize(),
            DVector::from(vec![-2.0, 3.0]).normalize(),
        ];
        cmaes_state.cov_eigenvectors = DMatrix::from_columns(&eigenvectors);
        cmaes_state.cov_sqrt_eigenvalues = DMatrix::from_diagonal(&vec![1e-4, 1e-10].into());
        cmaes_state.cov = &cmaes_state.cov_eigenvectors
            * &cmaes_state.cov_sqrt_eigenvalues.pow(2)
            * &cmaes_state.cov_eigenvectors.transpose();

        let mut terminated = false;
        for g in 0..dim {
            cmaes_state.generation = g;
            if let Some(reason) = cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5) {
                assert_eq!(reason, TerminationReason::NoEffectAxis);
                terminated = true;
            }
        }
        assert!(terminated);
    }

    #[test]
    fn test_check_termination_criteria_no_effect_coord() {
        // A small standard deviation in a coordinate axis produces NoEffectCoord (only if
        // NoEffectCoord is not met)
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.cov_eigenvectors = DMatrix::identity(2, 2);
        cmaes_state.cov_sqrt_eigenvalues = DMatrix::from_diagonal(&vec![1e-2, 1e-6].into());
        cmaes_state.cov = &cmaes_state.cov_eigenvectors
            * &cmaes_state.cov_sqrt_eigenvalues.pow(2)
            * &cmaes_state.cov_eigenvectors.transpose();

        let mut terminated = false;
        for g in 0..dim {
            cmaes_state.generation = g;
            if let Some(reason) = cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5) {
                assert_eq!(reason, TerminationReason::NoEffectCoord);
                terminated = true;
            }
        }
        assert!(terminated);
    }

    #[test]
    fn test_check_termination_criteria_condition_cov() {
        // A large difference between the maximum and minimum standard deviations producesc
        // ConditionCov
        let mut cmaes_state = CMAESOptions::new(dummy_function, 2).build().unwrap();
        cmaes_state.cov = DMatrix::from_iterator(2, 2, [0.01, 0.0, 0.0, 1e15]);
        let eigen = decompose_cov(cmaes_state.cov.clone());
        cmaes_state.cov_eigenvectors = eigen.eigenvectors;
        cmaes_state.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        assert_eq!(
            cmaes_state.check_termination_criteria(&[0.0; 100], 100, 0.5),
            Some(TerminationReason::ConditionCov),
        );
    }
}
