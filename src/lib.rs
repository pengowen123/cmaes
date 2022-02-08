//! An implementation of the CMA-ES optimization algorithm. It is used to minimize the value of an
//! objective function, and performs well on high-dimension, non-linear, non-convex,
//! ill-conditioned, and/or noisy problems.
//!
//! // TODO: example
//!
//! See [this paper][0] for details on the algorithm itself. Based on the paper and the [pycma][1]
//! implementation.
//!
//! [0]: https://arxiv.org/pdf/1604.00772.pdf
//! [1]: https://github.com/CMA-ES/pycma

pub mod objective_function;
pub mod options;

pub use crate::objective_function::ObjectiveFunction;
pub use crate::options::{CMAESOptions, Weights};

use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rayon::prelude::*;
use statrs::distribution::Normal;
use statrs::statistics::{Data, Median};

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f64;

use options::InvalidOptionsError;

/// Data returned when the algorithm terminates.
///
/// Contains the best individual and its corresponding function value as of the latest generation.
/// Also contains the reason for termination, which can be matched on to decide how to interpret
/// the result and whether and how to restart the algorithm.
#[derive(Clone, Debug)]
pub struct TerminationData {
    pub best_individual: DVector<f64>,
    pub best_function_value: f64,
    pub reason: TerminationReason,
}

#[derive(Clone, Debug)]
struct InvalidFunctionValueError;
#[derive(Clone, Debug)]
struct PosDefCovError;

/// Represents the reason for the algorithm terminating. Most of these are for preventing numerical
/// instability, while `TolFun` and `TolX` are problem-dependent parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    /// All function values of the latest generation and the range of the best function values of
    /// many consecutive generations lie below `tol_fun`.
    TolFun,
    /// The standard deviation of the distribution is smaller than `tol_x` in every coordinate and
    /// the mean has not moved much recently. Indicates that the algorithm has converged.
    TolX,
    /// The range of best function values in many consecutive generations is zero (i.e. no
    /// improvement is occurring).
    EqualFunValues,
    /// The best and median function values have not improved significantly over many generations.
    Stagnation,
    /// The maximum standard deviation across all dimensions increased by a factor of more than
    /// `10^8`. This is likely due to the function diverging or the initial step size being set far
    /// too small. In the latter case a restart with a larger step size may be useful.
    TolXUp,
    /// The standard deviation in any principal axis in the distribution is too small to perform any
    /// meaningful calculations.
    NoEffectAxis,
    /// The standard deviation in any coordinate axis in the distribution is too small to perform
    /// any meaningful calculations.
    NoEffectCoord,
    /// The condition number of the covariance matrix exceeds `10^14` or is non-normal.
    ConditionCov,
    /// The objective function has returned an invalid value (`NAN` or `-NAN`).
    InvalidFunctionValue,
    /// The covariance matrix is not positive definite. If this is returned frequently, it probably
    /// indicates a bug in the library and can be reported [here][0]. Using `Weights::Positive`
    /// should prevent this entirely in the meantime.
    ///
    /// [0]: https://github.com/pengowen123/cmaes/issues/
    PosDefCov,
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
    /// Learning rate for the mean
    cm: f64,
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
    objective_function: F,
    /// Constant parameters, see [Parameters]
    parameters: Parameters,
    /// The number of generations that have been fully completed
    generation: usize,
    /// The number of times the objective function has been evaluated
    function_evals: usize,
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
    /// (values at the front are from more recent generations)
    median_function_values: VecDeque<f64>,
    /// The current best individual
    current_best_individual: Option<DVector<f64>>,
    /// The initial max standard deviation along any principal axis of the distribution
    initial_max_standard_deviation: f64,
    /// The last time the eigendecomposition was updated, in function evals
    last_eigen_update_evals: usize,
}

impl<F: ObjectiveFunction> CMAESState<F> {
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

        if options.cm <= 0.0 || options.cm > 1.0 {
            return Err(InvalidOptionsError::Cm);
        }

        // Initialize constant parameters according to the options
        let dim = options.dimensions;
        let lambda = options.population_size;
        let mu = lambda / 2;
        let cm = options.cm;

        // Set initial weights
        // They will be normalized later
        let mut weights: DVector<f64> = match options.weights {
            // weights.len() == mu
            Weights::Uniform => vec![1.0; mu],
            // weights.len() == mu
            Weights::Positive => (1..=mu)
                .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
                .collect::<Vec<_>>(),
            // weights.len() == lambda
            Weights::Negative => (1..=lambda)
                .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
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
            cm,
            damp_s,
            tol_fun: options.tol_fun,
            tol_x: options.tol_x.unwrap_or(1e-12 * options.initial_step_size),
        };

        // Initialize variable parameters
        let mean = options.initial_mean;
        let cov = DMatrix::identity(options.dimensions, options.dimensions);
        let eigen = decompose_cov(cov.clone()).unwrap();
        let cov_eigenvectors = eigen.eigenvectors;
        let cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        let sigma = options.initial_step_size;
        let path_c = DVector::zeros(options.dimensions);
        let path_sigma = DVector::zeros(options.dimensions);

        Ok(Self {
            objective_function: options.objective_function,
            parameters,
            generation: 0,
            function_evals: 0,
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
            initial_max_standard_deviation: sigma,
            last_eigen_update_evals: 0,
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

    /// Samples `lambda` points from the distribution and returns the points and their
    /// corresponding objective function values sorted by their objective function values. Also
    /// updates the histories of the best and median function values.
    ///
    /// Returns `Err` if an invalid function value was encountered.
    fn sample(
        &mut self,
        past_generations_to_store: usize,
    ) -> Result<Vec<(DVector<f64>, f64)>, InvalidFunctionValueError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let params = &self.parameters;

        // Random steps in the distribution N(0, cov)
        let y = (0..params.lambda)
            .map(|_| {
                DVector::from_iterator(params.dim, (0..params.dim).map(|_| normal.sample(&mut rng)))
            })
            .map(|zk| &self.cov_eigenvectors * &self.cov_sqrt_eigenvalues * zk)
            .collect::<Vec<_>>();
        // Transform steps in y to the final distribution of N(mean, sigma^2 * cov)
        let z = y
            .iter()
            .map(|yk| &self.mean + self.sigma * yk)
            .collect::<Vec<_>>();

        // Evaluate and rank steps by evaluating their transformed counterparts
        let mut individuals = y
            .par_iter()
            .cloned()
            .zip(z.par_iter().map(|zk| self.objective_function.evaluate(zk)))
            .collect::<Vec<_>>();

        self.function_evals += individuals.len();

        individuals.sort_by(|a, b| partial_cmp(&a.1, &b.1));

        // Update histories of best and median values
        let (best_step, best_value) = individuals[0].clone();
        let best_individual = &self.mean + self.sigma * best_step;
        self.current_best_individual = Some(best_individual);

        self.best_function_values.push_front(best_value);
        if self.best_function_values.len() > past_generations_to_store {
            self.best_function_values.pop_back();
        }
        // Not perfectly accurate but it shouldn't make a difference
        let (_, median_value) = individuals[individuals.len() / 2];
        self.median_function_values.push_front(median_value);
        if self.median_function_values.len() > past_generations_to_store {
            self.median_function_values.pop_back();
        }

        // Check for invalid function values
        if individuals.iter().any(|(_, x)| x.is_nan()) {
            Err(InvalidFunctionValueError)
        } else {
            Ok(individuals)
        }
    }

    /// Advances to the next generation. Returns `Some` if a termination condition has been reached
    /// and the algorithm should be stopped.
    pub fn next(&mut self) -> Option<TerminationData> {
        // How many generations to store in self.best_function_values and
        // self.median_function_value8
        let past_generations_to_store = ((0.2 * self.generation as f64).ceil() as usize)
            .max(
                120 + (30.0 * self.parameters.dim as f64 / self.parameters.lambda as f64).ceil()
                    as usize,
            )
            .min(20000);

        // Sample individuals
        let individuals = match self.sample(past_generations_to_store) {
            Ok(x) => x,
            Err(_) => {
                return Some(self.get_termination_data(TerminationReason::InvalidFunctionValue));
            }
        };

        let params = &self.parameters;

        // Calculate new mean through weighted recombination
        // Only the mu best individuals are used even if there are lambda weights
        let yw = individuals
            .iter()
            .take(params.mu)
            .enumerate()
            .map(|(i, (y, _))| y * params.weights[i])
            .sum::<DVector<f64>>();
        self.mean = &self.mean + &(params.cm * self.sigma * &yw);

        // Update evolution paths
        let sqrt_inv_c = &self.cov_eigenvectors
            * DMatrix::from_diagonal(&self.cov_sqrt_eigenvalues.map_diagonal(|d| 1.0 / d))
            * self.cov_eigenvectors.transpose();

        self.path_sigma = (1.0 - params.cs) * &self.path_sigma
            + (params.cs * (2.0 - params.cs) * params.mu_eff).sqrt() * &sqrt_inv_c * &yw;

        // Expectation of N(0, I)
        let chi_n = (params.dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * params.dim as f64) + 1.0 / (21.0 * params.dim.pow(2) as f64));

        let hs = if (self.path_sigma.magnitude()
            / (1.0 - (1.0 - params.cs).powi(2 * (self.generation as i32 + 1))).sqrt())
            < (1.4 + 2.0 / (params.dim as f64 + 1.0)) * chi_n
        {
            1.0
        } else {
            0.0
        };

        self.path_c = (1.0 - params.cc) * &self.path_c
            + hs * (params.cc * (2.0 - params.cc) * params.mu_eff).sqrt() * &yw;

        // Update step size
        self.sigma *=
            ((params.cs / params.damp_s) * ((self.path_sigma.magnitude() / chi_n) - 1.0)).exp();

        // Update covariance matrix
        let weights_cov = params
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                *w * if *w >= 0.0 {
                    1.0
                } else {
                    params.dim as f64 / (&sqrt_inv_c * &individuals[i].0).magnitude().powi(2)
                }
            })
            .collect::<Vec<_>>();

        let delta_hs = (1.0 - hs) * params.cc * (2.0 - params.cc);
        self.cov = (1.0 + params.c1 * delta_hs
            - params.c1
            - params.cmu * params.weights.iter().sum::<f64>())
            * &self.cov
            + params.c1 * &self.path_c * self.path_c.transpose()
            + params.cmu
                * weights_cov
                    .into_iter()
                    .enumerate()
                    .map(|(i, wc)| wc * &individuals[i].0 * individuals[i].0.transpose())
                    .sum::<DMatrix<f64>>();

        // Ensure symmetry
        self.cov.fill_lower_triangle_with_upper_triangle();

        // Update eigendecomposition occasionally (updating every generation is unnecessary and
        // inefficient for high dim)
        let evals_per_eigen = (0.5 * params.dim as f64 * params.lambda as f64
            / ((params.c1 + params.cmu) * params.dim.pow(2) as f64))
            as usize;

        if self.function_evals > self.last_eigen_update_evals + evals_per_eigen {
            self.last_eigen_update_evals = self.function_evals;

            match decompose_cov(self.cov.clone()) {
                Ok(eigen) => {
                    self.cov_eigenvectors = eigen.eigenvectors;
                    self.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
                }
                Err(_) => return Some(self.get_termination_data(TerminationReason::PosDefCov)),
            };
        }

        self.generation += 1;

        // println!("(evals {}) best: {}", self.function_evals, self.get_current_best_individual().unwrap().1);

        // Terminate with the current best individual if any termination criterion is met
        let termination_reason =
            self.check_termination_criteria(&individuals, past_generations_to_store);
        if let Some(reason) = termination_reason {
            Some(self.get_termination_data(reason))
        } else {
            None
        }
    }

    /// Returns `Some` if any termination criterion is met.
    fn check_termination_criteria(
        &self,
        individuals: &[(DVector<f64>, f64)],
        past_generations_to_store: usize,
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
            ref mean,
            ref cov,
            ref cov_eigenvectors,
            ref cov_sqrt_eigenvalues,
            ref sigma,
            ref path_c,
            ref initial_max_standard_deviation,
            ..
        } = self;

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

            if range < tol_fun && individuals.iter().all(|(_, y)| *y < tol_fun) {
                return Some(TerminationReason::TolFun);
            }
        }

        // Check TerminationReason::TolX
        if (0..dim).all(|i| (*sigma * cov[(i, i)]).abs() < tol_x)
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

        if mean == &(mean + no_effect_axis_check) {
            return Some(TerminationReason::NoEffectAxis);
        }

        // Check TerminationReason::NoEffectCoord
        if (0..dim).any(|i| mean[i] == mean[i] + 0.2 * *sigma * cov[(i, i)]) {
            return Some(TerminationReason::NoEffectCoord);
        }

        // Check TerminationReason::EqualFunValues
        if let Some(range) = range_past_generations_a {
            if range == 0.0 {
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

                Data::new(first_values).median() < Data::new(last_values).median()
            };

            if !did_values_improve(&self.best_function_values)
                && !did_values_improve(&self.median_function_values)
            {
                return Some(TerminationReason::Stagnation);
            }
        }

        // Check TerminationReason::TolXUp
        let max_standard_deviation = *sigma
            * cov_sqrt_eigenvalues
                .diagonal()
                .iter()
                .max_by(|a, b| partial_cmp(*a, *b))
                .unwrap();

        if max_standard_deviation / initial_max_standard_deviation > 1e8 {
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

    /// Returns a `TerminationData` with the current best individual/value and the given reason.
    fn get_termination_data(&self, reason: TerminationReason) -> TerminationData {
        let (best_individual, best_function_value) = self.get_current_best_individual().unwrap();

        return TerminationData {
            best_individual: best_individual.clone(),
            best_function_value,
            reason,
        };
    }
}

/// Used for finding max/min values
fn partial_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Less)
}

/// Decomposition of a covariance matrix
struct CovDecomposition {
    /// Columns are eigenvectors
    eigenvectors: DMatrix<f64>,
    /// Diagonal matrix with square roots of eigenvalues
    sqrt_eigenvalues: DMatrix<f64>,
}

/// Decomposes a covariance matrix into a set of normalized eigenvectors and a diagonal matrix
/// containing the square roots of the corresponding eigenvalues
///
/// Returns `Err` if the matrix is not positive-definite
fn decompose_cov(matrix: DMatrix<f64>) -> Result<CovDecomposition, PosDefCovError> {
    let mut eigen = nalgebra_lapack::SymmetricEigen::new(matrix);

    for mut col in eigen.eigenvectors.column_iter_mut() {
        col.normalize_mut();
    }

    if eigen.eigenvalues.iter().any(|x| *x <= 0.0) {
        Err(PosDefCovError)
    } else {
        Ok(CovDecomposition {
            eigenvectors: eigen.eigenvectors,
            sqrt_eigenvalues: DMatrix::from_diagonal(&eigen.eigenvalues.map(|x| x.sqrt())),
        })
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

        let eigen = decompose_cov(matrix.clone()).unwrap();

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
        for d in 2..30 {
            for n in 1..20 {
                let mut options = CMAESOptions::new(dummy_function, d).weights(Weights::Positive);

                options.population_size *= n;

                let cmaes_state = options.build().unwrap();

                assert!(cmaes_state.parameters.weights.iter().all(|w| *w > 0.0));
                assert_approx_eq!(
                    cmaes_state.parameters.weights.iter().sum::<f64>(),
                    1.0,
                    1e-12
                );
            }
        }
    }

    // Tests that Weights::Negative produces only positive values for the first mu weights and only
    // negative values for the rest
    #[test]
    fn test_weights_negative() {
        for d in 2..30 {
            for n in 1..20 {
                let mut options = CMAESOptions::new(dummy_function, d).weights(Weights::Negative);
                options.population_size *= n;
                let cmaes_state = options.build().unwrap();

                assert!(cmaes_state
                    .parameters
                    .weights
                    .iter()
                    .take(cmaes_state.parameters.mu)
                    .all(|w| *w > 0.0));

                assert_approx_eq!(
                    cmaes_state
                        .parameters
                        .weights
                        .iter()
                        .take(cmaes_state.parameters.mu)
                        .sum::<f64>(),
                    1.0,
                    1e-12
                );

                assert!(cmaes_state
                    .parameters
                    .weights
                    .iter()
                    .skip(cmaes_state.parameters.mu)
                    .all(|w| *w <= 0.0));
            }
        }
    }

    #[test]
    fn test_check_termination_criteria_none() {
        let dim = 2;
        let cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            None,
        );

        // A large standard deviation in one axis should not meet any termination criteria if the
        // initial step size was also large
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim)
            .initial_step_size(1e3)
            .build()
            .unwrap();
        cmaes_state.sigma = 1e4;
        cmaes_state.cov = DMatrix::from_iterator(2, 2, [0.01, 0.0, 0.0, 1e4]);
        let eigen = decompose_cov(cmaes_state.cov.clone()).unwrap();
        cmaes_state.cov_eigenvectors = eigen.eigenvectors;
        cmaes_state.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            None,
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun() {
        // A population of below-tolerance function values and a small range of historical values
        // produces TolFun
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.best_function_values.extend(vec![1.0; 100]);
        cmaes_state.best_function_values.push_front(1.0 + 1e-13);
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            Some(TerminationReason::TolFun),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x() {
        // A small step size and evolution path produces TolX
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.sigma = 1e-13;
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            Some(TerminationReason::TolX),
        );
    }

    #[test]
    fn test_check_termination_criteria_equal_fun_values() {
        // A zero range of historical values produces EqualFunValues
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.best_function_values.extend(vec![1.0; 100]);
        // Equal to 1.0 due to lack of precision
        cmaes_state.best_function_values.push_front(1.0 + 1e-17);
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 1.0); 100], 100),
            Some(TerminationReason::EqualFunValues),
        );
    }

    #[test]
    fn test_check_termination_criteria_stagnation() {
        // Median/best function values that change but don't improve for many generations produces
        // Stagnation
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        let mut values = Vec::new();
        values.extend(vec![1.0; 10]);
        values.extend(vec![2.0; 10]);
        values.extend(vec![2.0; 10]);
        values.extend(vec![1.0; 10]);
        cmaes_state.best_function_values.extend(values.clone());
        cmaes_state.median_function_values.extend(values.clone());
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 1.0); 100], 40),
            Some(TerminationReason::Stagnation),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x_up() {
        // A large increase in maximum standard deviation produces TolXUp
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.sigma = 1e8;
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            Some(TerminationReason::TolXUp),
        );
    }

    #[test]
    fn test_check_termination_criteria_no_effect_axis() {
        // A lack of available precision along a principal axis in the distribution produces
        // NoEffectAxis
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        let eigenvectors = [
            DVector::from(vec![3.0, 2.0]).normalize(),
            DVector::from(vec![-2.0, 3.0]).normalize(),
        ];
        cmaes_state.mean = vec![100.0; 2].into();
        cmaes_state.sigma = 1e-10;
        cmaes_state.cov_eigenvectors = DMatrix::from_columns(&eigenvectors);
        cmaes_state.cov_sqrt_eigenvalues = DMatrix::from_diagonal(&vec![1e-1, 1e-6].into());
        cmaes_state.cov = &cmaes_state.cov_eigenvectors
            * &cmaes_state.cov_sqrt_eigenvalues.pow(2)
            * &cmaes_state.cov_eigenvectors.transpose();

        let mut terminated = false;
        for g in 0..dim {
            cmaes_state.generation = g;
            let termination_reason =
                cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100);
            if let Some(reason) = termination_reason {
                assert_eq!(reason, TerminationReason::NoEffectAxis);
                terminated = true;
            }
        }
        assert!(terminated);
    }

    #[test]
    fn test_check_termination_criteria_no_effect_coord() {
        // A lack of available precision along a coordinate axis in the distribution produces
        // NoEffectCoord
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.mean = vec![100.0; 2].into();
        cmaes_state.cov_eigenvectors = DMatrix::identity(2, 2);
        cmaes_state.cov_sqrt_eigenvalues = DMatrix::from_diagonal(&vec![1e-4, 1e-10].into());
        cmaes_state.cov = &cmaes_state.cov_eigenvectors
            * &cmaes_state.cov_sqrt_eigenvalues.pow(2)
            * &cmaes_state.cov_eigenvectors.transpose();

        let mut terminated = false;
        for g in 0..dim {
            cmaes_state.generation = g;
            let termination_reason =
                cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100);
            if let Some(reason) = termination_reason {
                assert_eq!(reason, TerminationReason::NoEffectCoord);
                terminated = true;
            }
        }
        assert!(terminated);
    }

    #[test]
    fn test_check_termination_criteria_condition_cov() {
        // A large difference between the maximum and minimum standard deviations produces
        // ConditionCov
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(dummy_function, dim).build().unwrap();
        cmaes_state.cov = DMatrix::from_iterator(2, 2, [0.01, 0.0, 0.0, 1e15]);
        let eigen = decompose_cov(cmaes_state.cov.clone()).unwrap();
        cmaes_state.cov_eigenvectors = eigen.eigenvectors;
        cmaes_state.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        assert_eq!(
            cmaes_state.check_termination_criteria(&vec![(DVector::zeros(dim), 0.0); 100], 100),
            Some(TerminationReason::ConditionCov),
        );
    }
}
