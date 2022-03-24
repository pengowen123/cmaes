//! Algorithm termination handling. See [`TerminationReason`] for full documentation.

use statrs::statistics::{Data, Median};

use std::collections::VecDeque;
use std::fmt::{self, Debug};

use crate::parameters::Parameters;
use crate::sampling::EvaluatedPoint;
use crate::state::State;
use crate::utils;

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
    /// indicates a bug in the library and can be reported [here][0]. Using
    /// [`Weights::Positive`][crate::parameters::Weights::Positive] should prevent this entirely in
    /// the meantime.
    ///
    /// [0]: https://github.com/pengowen123/cmaes/issues/
    PosDefCov,
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, fmt)
    }
}

/// Checks the state, objective function value history, and current generation, and returns
/// `Some` if any termination criterion is met
pub(crate) fn check_termination_criteria(
    parameters: &Parameters,
    state: &State,
    best_function_value_history: &VecDeque<f64>,
    median_function_value_history: &VecDeque<f64>,
    max_history_size: usize,
    individuals: &[EvaluatedPoint],
) -> Option<TerminationReason> {
    let dim = parameters.dim();
    let lambda = parameters.lambda();
    let initial_sigma = parameters.initial_sigma();
    let tol_fun = parameters.tol_fun();
    let tol_x = parameters.tol_x();

    let mean = state.mean();
    let cov = state.cov();
    let cov_eigenvectors = state.cov_eigenvectors();
    let cov_sqrt_eigenvalues = state.cov_sqrt_eigenvalues();
    let sigma = state.sigma();
    let path_c = state.path_c();

    // Check TerminationReason::TolFun
    let past_generations_a = 10 + (30.0 * dim as f64 / lambda as f64).ceil() as usize;
    let mut range_past_generations_a = None;

    if best_function_value_history.len() >= past_generations_a {
        let max = best_function_value_history
            .iter()
            .take(past_generations_a)
            .max_by(|a, b| utils::partial_cmp(*a, *b))
            .unwrap();

        let min = best_function_value_history
            .iter()
            .take(past_generations_a)
            .min_by(|a, b| utils::partial_cmp(*a, *b))
            .unwrap();

        let range = (max - min).abs();
        range_past_generations_a = Some(range);

        if range < tol_fun && individuals.iter().all(|p| p.value() < tol_fun) {
            return Some(TerminationReason::TolFun);
        }
    }

    // Check TerminationReason::TolX
    if (0..dim).all(|i| (sigma * cov[(i, i)]).abs() < tol_x)
        && path_c.iter().all(|x| (sigma * *x).abs() < tol_x)
    {
        return Some(TerminationReason::TolX);
    }

    // Check TerminationReason::ConditionCov
    let cond = state.axis_ratio().powi(2);

    if !cond.is_normal() || cond > 1e14 {
        return Some(TerminationReason::ConditionCov);
    }

    // Check TerminationReason::NoEffectAxis
    // Cycles from 0 to n-1 to avoid checking every column every iteration
    let index_to_check = state.generation() % dim;

    let no_effect_axis_check = 0.1
        * sigma
        * cov_sqrt_eigenvalues[(index_to_check, index_to_check)]
        * cov_eigenvectors.column(index_to_check);

    if mean == &(mean + no_effect_axis_check) {
        return Some(TerminationReason::NoEffectAxis);
    }

    // Check TerminationReason::NoEffectCoord
    if (0..dim).any(|i| mean[i] == mean[i] + 0.2 * sigma * cov[(i, i)]) {
        return Some(TerminationReason::NoEffectCoord);
    }

    // Check TerminationReason::EqualFunValues
    if let Some(range) = range_past_generations_a {
        if range == 0.0 {
            return Some(TerminationReason::EqualFunValues);
        }
    }

    // Check TerminationReason::Stagnation
    let past_generations_b = max_history_size;

    if best_function_value_history.len() >= past_generations_b
        && median_function_value_history.len() >= past_generations_b
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

        if !did_values_improve(best_function_value_history)
            && !did_values_improve(median_function_value_history)
        {
            return Some(TerminationReason::Stagnation);
        }
    }

    // Check TerminationReason::TolXUp
    let max_standard_deviation = sigma
        * cov_sqrt_eigenvalues
            .diagonal()
            .iter()
            .max_by(|a, b| utils::partial_cmp(*a, *b))
            .unwrap();

    if max_standard_deviation / initial_sigma > 1e8 {
        return Some(TerminationReason::TolXUp);
    }

    None
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;
    use crate::matrix::SquareMatrix;
    use crate::parameters::{TerminationParameters, Weights};
    use crate::state::State;

    const DEFAULT_INITIAL_SIGMA: f64 = 0.5;
    const DIM: usize = 2;
    const MAX_HISTORY_LENGTH: usize = 100;

    fn get_parameters(initial_sigma: Option<f64>) -> Parameters {
        let initial_sigma = initial_sigma.unwrap_or(DEFAULT_INITIAL_SIGMA);
        let lambda = 6;
        let cm = 1.0;
        let termination_parameters = TerminationParameters {
            tol_fun: 1e-12,
            tol_x: 1e-12 * initial_sigma,
        };
        Parameters::new(
            DIM,
            lambda,
            Weights::Negative,
            0,
            initial_sigma,
            cm,
            termination_parameters,
        )
    }

    fn get_state(initial_sigma: Option<f64>) -> State {
        State::new(
            vec![0.0; DIM].into(),
            initial_sigma.unwrap_or(DEFAULT_INITIAL_SIGMA),
        )
    }

    fn get_dummy_generation(function_value: f64) -> Vec<EvaluatedPoint> {
        (0..100)
            .map(|_| {
                EvaluatedPoint::new(
                    DVector::zeros(DIM),
                    &DVector::zeros(DIM),
                    1.0,
                    &mut |_: &DVector<f64>| function_value,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn test_check_termination_criteria_none() {
        // A fresh state should not meet any termination criteria
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            None,
        );
    }

    #[test]
    fn test_check_termination_criteria_none_large_std_dev() {
        // A large standard deviation in one axis should not meet any termination criteria if the
        // initial step size was also large
        let initial_sigma = Some(1e3);
        let mut state = get_state(initial_sigma);

        *state.mut_sigma() = 1e4;
        state
            .mut_cov()
            .set_cov(
                SquareMatrix::from_iterator(2, 2, [0.01, 0.0, 0.0, 1e4]),
                true,
            )
            .unwrap();

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            None,
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun() {
        // A population of below-tolerance function values and a small range of historical values
        // produces TolFun
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let mut best_function_value_history = VecDeque::new();
        best_function_value_history.extend(vec![1.0; 100]);
        best_function_value_history.push_front(1.0 + 1e-13);

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &best_function_value_history,
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            Some(TerminationReason::TolFun),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x() {
        // A small step size and evolution path produces TolX
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        *state.mut_sigma() = 1e-13;

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            Some(TerminationReason::TolX),
        );
    }

    #[test]
    fn test_check_termination_criteria_equal_fun_values() {
        // A zero range of historical values (that are above tol_fun) produces EqualFunValues
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let mut best_function_value_history = VecDeque::new();
        best_function_value_history.extend(vec![1.0; 100]);
        // Equal to 1.0 due to lack of precision
        best_function_value_history.push_front(1.0 + 1e-17);

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &best_function_value_history,
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            Some(TerminationReason::EqualFunValues),
        );
    }

    #[test]
    fn test_check_termination_criteria_stagnation() {
        // Median/best function values that change but don't improve for many generations produces
        // Stagnation
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let mut best_function_value_history = VecDeque::new();
        best_function_value_history.extend(vec![1.0; 100]);
        // Equal to 1.0 due to lack of precision
        best_function_value_history.push_front(1.0 + 1e-17);

        let mut values = Vec::new();
        values.extend(vec![1.0; 10]);
        values.extend(vec![2.0; 10]);
        values.extend(vec![2.0; 10]);
        values.extend(vec![1.0; 10]);
        let best_function_value_history = values.clone().into();
        let median_function_value_history = values.clone().into();

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &best_function_value_history,
                &median_function_value_history,
                values.len(),
                &get_dummy_generation(1.0),
            ),
            Some(TerminationReason::Stagnation),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x_up() {
        // A large increase in maximum standard deviation produces TolXUp
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        *state.mut_sigma() = 1e8;

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            Some(TerminationReason::TolXUp),
        );
    }

    #[test]
    fn test_check_termination_criteria_no_effect_axis() {
        // A lack of available precision along a principal axis in the distribution produces
        // NoEffectAxis
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        *state.mut_mean() = vec![100.0; 2].into();
        *state.mut_sigma() = 1e-10;

        let eigenvectors = SquareMatrix::from_columns(&[
            DVector::from(vec![3.0, 2.0]).normalize(),
            DVector::from(vec![-2.0, 3.0]).normalize(),
        ]);
        let sqrt_eigenvalues = SquareMatrix::from_diagonal(&vec![1e-1, 1e-6].into());
        let cov = &eigenvectors * sqrt_eigenvalues.pow(2) * eigenvectors.transpose();
        state.mut_cov().set_cov(cov, true).unwrap();

        let mut terminated = false;
        for g in 0..DIM {
            *state.mut_generation() = g;

            let termination_reason = check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            );

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
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        *state.mut_mean() = vec![100.0; 2].into();

        let eigenvectors = SquareMatrix::<f64>::identity(2, 2);
        let sqrt_eigenvalues = SquareMatrix::from_diagonal(&vec![1e-4, 1e-10].into());
        let cov = &eigenvectors * sqrt_eigenvalues.pow(2) * eigenvectors.transpose();
        state.mut_cov().set_cov(cov, true).unwrap();

        let mut terminated = false;
        for g in 0..DIM {
            *state.mut_generation() = g;

            let termination_reason = check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            );

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
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        state
            .mut_cov()
            .set_cov(
                SquareMatrix::from_iterator(2, 2, [0.99, 0.0, 0.0, 1e14]),
                true,
            )
            .unwrap();

        assert_eq!(
            check_termination_criteria(
                &get_parameters(initial_sigma),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(0.0),
            ),
            Some(TerminationReason::ConditionCov),
        );
    }
}
