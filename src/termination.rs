//! Algorithm termination handling. See [`TerminationReason`] for full documentation.

use statrs::statistics::{Data, Median};

use std::collections::VecDeque;
use std::fmt::{self, Debug};

use crate::parameters::Parameters;
use crate::sampling::EvaluatedPoint;
use crate::state::State;
use crate::utils;

/// Represents a reason for the algorithm terminating. Most of these are for preventing numerical
/// instability, while `Tol*` are problem-dependent parameters and `Max*` are for bounding
/// iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    /// The maximum number of objective function evaluations has been reached.
    MaxFunctionEvals,
    /// The maximum number of generations has been reached.
    MaxGenerations,
    /// The target objective function value has been reached.
    FunTarget,
    /// The range of function values of the latest generation and the range of the best function
    /// values of many consecutive generations lie below `tol_fun`. Indicates that the function
    /// value has stopped changing significantly.
    TolFun,
    /// The standard deviation of the distribution is smaller than `tol_x` in every coordinate and
    /// the mean has not moved much recently. Indicates that the algorithm has converged.
    TolX,
    /// The range of best function values in many consecutive generations is zero (i.e. no
    /// improvement is occurring).
    EqualFunValues,
    /// The best and median function values have not improved significantly over many generations.
    Stagnation,
    /// The maximum standard deviation across all distribution axes increased by a factor of more
    /// than `tol_x_up`. This is likely due to the function diverging or the initial step size being
    /// set far too small. In the latter case a restart with a larger step size may be useful.
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
/// a list of any termination criteria that are met
pub(crate) fn check_termination_criteria(
    current_function_evals: usize,
    parameters: &Parameters,
    state: &State,
    best_function_value_history: &VecDeque<f64>,
    median_function_value_history: &VecDeque<f64>,
    max_history_size: usize,
    individuals: &[EvaluatedPoint],
) -> Vec<TerminationReason> {
    let mut result = Vec::new();

    let dim = parameters.dim();
    let lambda = parameters.lambda();
    let initial_sigma = parameters.initial_sigma();
    let fun_target = parameters.fun_target();
    let tol_fun = parameters.tol_fun();
    let tol_x = parameters.tol_x();
    let tol_x_up = parameters.tol_x_up();

    let mean = state.mean();
    let cov = state.cov();
    let cov_eigenvectors = state.cov_eigenvectors();
    let cov_sqrt_eigenvalues = state.cov_sqrt_eigenvalues();
    let sigma = state.sigma();
    let path_c = state.path_c();

    // Check TerminationReason::MaxFunctionEvals
    if let Some(max_function_evals) = parameters.max_function_evals() {
        if current_function_evals >= max_function_evals {
            result.push(TerminationReason::MaxFunctionEvals);
        }
    }

    // Check TerminationReason::MaxGenerations
    if let Some(max_generations) = parameters.max_generations() {
        if state.generation() >= max_generations {
            result.push(TerminationReason::MaxGenerations);
        }
    }

    // Check TerminationReason::FunTarget
    if individuals.iter().any(|ind| ind.value() <= fun_target) {
        result.push(TerminationReason::FunTarget);
    }

    // Check TerminationReason::TolFun
    let past_generations_a = 10 + (30.0 * dim as f64 / lambda as f64).ceil() as usize;
    let mut range_past_generations_a = None;

    if best_function_value_history.len() >= past_generations_a {
        let range_history = utils::range(
            best_function_value_history
                .iter()
                .take(past_generations_a)
                .cloned(),
        )
        .unwrap();
        range_past_generations_a = Some(range_history);

        let range_current = utils::range(individuals.iter().map(|p| p.value())).unwrap();

        if range_history < tol_fun && range_current < tol_fun {
            result.push(TerminationReason::TolFun);
        }
    }

    // Check TerminationReason::TolX
    if (0..dim).all(|i| (sigma * cov[(i, i)]).abs() < tol_x)
        && path_c.iter().all(|x| (sigma * *x).abs() < tol_x)
    {
        result.push(TerminationReason::TolX);
    }

    // Check TerminationReason::ConditionCov
    let cond = state.axis_ratio().powi(2);

    if !cond.is_normal() || cond > 1e14 {
        result.push(TerminationReason::ConditionCov);
    }

    // Check TerminationReason::NoEffectAxis
    // Cycles from 0 to n-1 to avoid checking every column every iteration
    let index_to_check = state.generation() % dim;

    let no_effect_axis_check = 0.1
        * sigma
        * cov_sqrt_eigenvalues[(index_to_check, index_to_check)]
        * cov_eigenvectors.column(index_to_check);

    if mean == &(mean + no_effect_axis_check) {
        result.push(TerminationReason::NoEffectAxis);
    }

    // Check TerminationReason::NoEffectCoord
    if (0..dim).any(|i| mean[i] == mean[i] + 0.2 * sigma * cov[(i, i)]) {
        result.push(TerminationReason::NoEffectCoord);
    }

    // Check TerminationReason::EqualFunValues
    if let Some(range) = range_past_generations_a {
        if range == 0.0 {
            result.push(TerminationReason::EqualFunValues);
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
            result.push(TerminationReason::Stagnation);
        }
    }

    // Check TerminationReason::TolXUp
    let max_standard_deviation = sigma
        * cov_sqrt_eigenvalues
            .diagonal()
            .iter()
            .max_by(|a, b| utils::partial_cmp(**a, **b))
            .unwrap();

    if max_standard_deviation / initial_sigma > tol_x_up {
        result.push(TerminationReason::TolXUp);
    }

    result
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

    fn get_parameters(
        initial_sigma: Option<f64>,
        max_function_evals: Option<usize>,
        max_generations: Option<usize>,
    ) -> Parameters {
        let initial_sigma = initial_sigma.unwrap_or(DEFAULT_INITIAL_SIGMA);
        let lambda = 6;
        let cm = 1.0;
        let termination_parameters = TerminationParameters {
            max_function_evals: max_function_evals,
            max_generations: max_generations,
            fun_target: 1e-12,
            tol_fun: 1e-12,
            tol_x: 1e-12 * initial_sigma,
            tol_x_up: 1e8,
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
    fn test_check_termination_criteria_max_function_evals() {
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let current_function_evals = 100;

        assert_eq!(
            check_termination_criteria(
                current_function_evals,
                &get_parameters(initial_sigma, Some(100), None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::MaxFunctionEvals],
        );
    }

    #[test]
    fn test_check_termination_criteria_max_generations() {
        let initial_sigma = None;
        let mut state = get_state(initial_sigma);

        *state.mut_generation() = 100;

        assert_eq!(
            check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, Some(100)),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::MaxGenerations],
        );
    }

    #[test]
    fn test_check_termination_criteria_none() {
        // A fresh state should not meet any termination criteria
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        assert!(check_termination_criteria(
            0,
            &get_parameters(initial_sigma, None, None),
            &state,
            &VecDeque::new(),
            &VecDeque::new(),
            MAX_HISTORY_LENGTH,
            &get_dummy_generation(1.0),
        )
        .is_empty());
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

        assert!(check_termination_criteria(
            0,
            &get_parameters(initial_sigma, None, None),
            &state,
            &VecDeque::new(),
            &VecDeque::new(),
            MAX_HISTORY_LENGTH,
            &get_dummy_generation(1.0),
        )
        .is_empty());
    }

    #[test]
    fn test_check_termination_criteria_fun_target() {
        // A best function value below a threshold produces FunTarget
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        assert_eq!(
            check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1e-16),
            ),
            vec![TerminationReason::FunTarget],
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun() {
        // Small ranges of current and historical function values produces TolFun
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let mut best_function_value_history = VecDeque::new();
        best_function_value_history.extend(vec![1.0; 100]);
        best_function_value_history.push_front(1.0 + 1e-13);

        assert_eq!(
            check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &best_function_value_history,
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(best_function_value_history[0]),
            ),
            vec![TerminationReason::TolFun],
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
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::TolX],
        );
    }

    #[test]
    fn test_check_termination_criteria_equal_fun_values() {
        // A zero range of historical best values with a non-small range of current function
        // values produces EqualFunValues
        let initial_sigma = None;
        let state = get_state(initial_sigma);

        let mut best_function_value_history = VecDeque::new();
        best_function_value_history.extend(vec![1.0; 100]);
        // Equal to 1.0 due to lack of precision
        best_function_value_history.push_front(1.0 + 1e-17);

        let mut individuals = get_dummy_generation(1.5);
        individuals[0] = EvaluatedPoint::new(
            DVector::zeros(DIM),
            &DVector::zeros(DIM),
            0.0,
            &mut |_: &DVector<f64>| best_function_value_history[0],
        )
        .unwrap();

        assert_eq!(
            check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &best_function_value_history,
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &individuals,
            ),
            vec![TerminationReason::EqualFunValues],
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
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &best_function_value_history,
                &median_function_value_history,
                values.len(),
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::Stagnation],
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
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::TolXUp],
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

            let termination_reasons = check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            );

            if !termination_reasons.is_empty() {
                assert_eq!(termination_reasons, vec![TerminationReason::NoEffectAxis]);
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

            let termination_reasons = check_termination_criteria(
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            );

            if !termination_reasons.is_empty() {
                assert_eq!(termination_reasons, vec![TerminationReason::NoEffectCoord]);
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
                0,
                &get_parameters(initial_sigma, None, None),
                &state,
                &VecDeque::new(),
                &VecDeque::new(),
                MAX_HISTORY_LENGTH,
                &get_dummy_generation(1.0),
            ),
            vec![TerminationReason::ConditionCov],
        );
    }
}
