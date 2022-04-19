//! Algorithm termination handling. See [`TerminationReason`] for full documentation.

use statrs::statistics::{Data, Median};

use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::time::Instant;

use crate::history::History;
use crate::parameters::Parameters;
use crate::sampling::EvaluatedPoint;
use crate::state::State;
use crate::{utils, MAX_HISTORY_LENGTH};

/// Represents a reason for the algorithm terminating. Most of these are for preventing numerical
/// instability, while `Tol*` are problem-dependent parameters and `Max*` are for bounding
/// iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    /// The maximum number of objective function evaluations has been reached.
    MaxFunctionEvals,
    /// The maximum number of generations has been reached.
    MaxGenerations,
    /// The algorithm has been running for longer than the time limit.
    MaxTime,
    /// The target objective function value has been reached.
    FunTarget,
    /// The range of function values of the latest generation and the range of the best function
    /// values of many consecutive generations lie below `tol_fun`. Indicates that the function
    /// value has stopped changing significantly and that the function value spread of each
    /// generation is equally insignificant.
    TolFun,
    /// Like `TolFun`, but the range is `tol_fun_rel * (first_median - best_median)` (i.e. it is
    /// relative to the overall improvement in the median objective function value).
    TolFunRel,
    /// The range of best function values in many consecutive generations is lower than
    /// `tol_fun_hist` (i.e. little to no improvement or change is occurring).
    TolFunHist,
    /// The standard deviation of the distribution is smaller than `tol_x` in every coordinate and
    /// the mean has not moved much recently. Indicates that the algorithm has converged.
    TolX,
    /// The best and median function values have not improved over the past 20% of all generations,
    /// clamped to the range `[tol_stagnation, MAX_HISTORY_LENGTH]`. Setting `tol_stagnation` to be
    /// greater than `MAX_HISTORY_LENGTH` effectively disables this termination criterion.
    TolStagnation,
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
    /// The condition number of the covariance matrix exceeds `tol_condition_cov` or is non-normal.
    TolConditionCov,
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

/// Stores parameters of the termination check
#[cfg_attr(test, derive(Clone))]
pub(crate) struct TerminationCheck<'a> {
    pub current_function_evals: usize,
    /// The time at which the `CMAES` was created
    pub time_created: Instant,
    pub parameters: &'a Parameters,
    pub state: &'a State,
    pub history: &'a History,
    /// The current generation of individuals
    pub individuals: &'a [EvaluatedPoint],
}

impl<'a> TerminationCheck<'a> {
    /// Checks whether any termination criteria are met based on the stored parameters
    pub(crate) fn check_termination_criteria(self) -> Vec<TerminationReason> {
        let mut result = Vec::new();

        let mode = self.parameters.mode();
        let dim = self.parameters.dim();
        let lambda = self.parameters.lambda();
        let initial_sigma = self.parameters.initial_sigma();
        let tol_fun = self.parameters.tol_fun();
        let tol_fun_rel_option = self.parameters.tol_fun_rel();
        let tol_fun_hist = self.parameters.tol_fun_hist();
        let tol_x = self.parameters.tol_x();
        let tol_stagnation_option = self.parameters.tol_stagnation();
        let tol_x_up = self.parameters.tol_x_up();
        let tol_condition_cov = self.parameters.tol_condition_cov();

        let mean = self.state.mean();
        let cov = self.state.cov();
        let cov_eigenvectors = self.state.cov_eigenvectors();
        let cov_sqrt_eigenvalues = self.state.cov_sqrt_eigenvalues();
        let sigma = self.state.sigma();
        let path_c = self.state.path_c();

        // Check TerminationReason::MaxFunctionEvals
        if let Some(max_function_evals) = self.parameters.max_function_evals() {
            if self.current_function_evals >= max_function_evals {
                result.push(TerminationReason::MaxFunctionEvals);
            }
        }

        // Check TerminationReason::MaxGenerations
        if let Some(max_generations) = self.parameters.max_generations() {
            if self.state.generation() >= max_generations {
                result.push(TerminationReason::MaxGenerations);
            }
        }

        // Check TerminationReason::MaxTime
        if let Some(max_time) = self.parameters.max_time() {
            if self.time_created.elapsed() >= max_time {
                result.push(TerminationReason::MaxTime);
            }
        }

        // Check TerminationReason::FunTarget
        if let Some(fun_target) = self.parameters.fun_target() {
            if self
                .individuals
                .iter()
                .any(|ind| mode.is_better(ind.value(), fun_target))
            {
                result.push(TerminationReason::FunTarget);
            }
        }

        // Check TerminationReason::TolFun*
        let past_generations_a = 10 + (30.0 * dim as f64 / lambda as f64).ceil() as usize;

        if self.history.best_function_values().len() >= past_generations_a {
            let range_history = utils::range(
                self.history
                    .best_function_values()
                    .iter()
                    .take(past_generations_a)
                    .cloned(),
            )
            .unwrap();

            let range_current = utils::range(self.individuals.iter().map(|p| p.value())).unwrap();

            if range_history < tol_fun_hist {
                result.push(TerminationReason::TolFunHist);
            }

            if range_history < tol_fun && range_current < tol_fun {
                result.push(TerminationReason::TolFun);
            }

            if let (Some(first_median_value), Some(best_median_value)) = (
                self.history.first_median_function_value(),
                self.history.best_median_function_value(),
            ) {
                let tol_fun_rel_range =
                    tol_fun_rel_option * (first_median_value - best_median_value).abs();

                if range_history < tol_fun_rel_range && range_current < tol_fun_rel_range {
                    result.push(TerminationReason::TolFunRel);
                }
            }
        }

        // Check TerminationReason::TolX
        if (0..dim).all(|i| (sigma * cov[(i, i)]).abs() < tol_x)
            && path_c.iter().all(|x| (sigma * *x).abs() < tol_x)
        {
            result.push(TerminationReason::TolX);
        }

        // Check TerminationReason::TolConditionCov
        let cond = self.state.axis_ratio().powi(2);

        if !cond.is_normal() || cond > tol_condition_cov {
            result.push(TerminationReason::TolConditionCov);
        }

        // Check TerminationReason::NoEffectAxis
        // Cycles from 0 to n-1 to avoid checking every column every iteration
        let index_to_check = self.state.generation() % dim;

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

        // Check TerminationReason::TolStagnation
        let tol_stagnation_generations =
            get_tol_stagnation_generations(tol_stagnation_option, self.state.generation());

        if let Some(tol_stagnation_generations) = tol_stagnation_generations {
            if self.history.best_function_values().len() >= tol_stagnation_generations
                && self.history.median_function_values().len() >= tol_stagnation_generations
            {
                // Checks whether the median of the values has regressed over the past
                // `tol_stagnation_generations` generations
                // Returns true if the values became worse
                let did_values_regress = |values: &VecDeque<f64>| {
                    // Note that TolStagnation is effectively disabled if tol_stagnation_generations
                    // is < 4, enforcing an effective minimum bound on tol_stagnation
                    let subrange_length = (tol_stagnation_generations as f64 * 0.3) as usize;

                    // Most recent `subrange_length `values within the past
                    // `tol_stagnation_generations` generations
                    let first_values = values
                        .iter()
                        .take(tol_stagnation_generations)
                        .take(subrange_length)
                        .cloned()
                        .collect::<Vec<_>>();

                    // Least recent `subrange_length` values within the past
                    // tol_stagnation_generations` generations
                    let last_values = values
                        .iter()
                        .take(tol_stagnation_generations)
                        .skip(tol_stagnation_generations - subrange_length)
                        .cloned()
                        .collect::<Vec<_>>();

                    mode.is_better(
                        Data::new(last_values).median(),
                        Data::new(first_values).median(),
                    )
                };

                if did_values_regress(self.history.best_function_values())
                    && did_values_regress(self.history.median_function_values())
                {
                    result.push(TerminationReason::TolStagnation);
                }
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
}

/// Returns the default value for the `tol_stagnation` option (which is the lower bound for
/// `TolStagnation`)
pub(crate) fn get_default_tol_stagnation_option(dim: usize, lambda: usize) -> usize {
    100 + (100.0 * (dim as f64).powf(1.5) / lambda as f64).ceil() as usize
}

/// Returns the number of generations over which to check `TolStagnation`
///
/// Returns `None` if the history isn't long enough to perform the check
fn get_tol_stagnation_generations(
    tol_stagnation_option: usize,
    current_generation: usize,
) -> Option<usize> {
    // 20% of past generations
    let generations = (current_generation / 5)
        // At most the max history length
        .min(MAX_HISTORY_LENGTH);

    // Don't check TolStagnation if `generations` is below the lower bound
    if generations < tol_stagnation_option {
        None
    } else {
        Some(generations)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use std::time::Duration;

    use super::*;
    use crate::matrix::SquareMatrix;
    use crate::mode::Mode;
    use crate::parameters::{TerminationParameters, Weights};
    use crate::state::State;
    use crate::CMAESOptions;

    #[test]
    fn test_get_default_tol_stagnation_option() {
        assert_eq!(180, get_default_tol_stagnation_option(4, 10));
        assert_eq!(600, get_default_tol_stagnation_option(100, 200));
    }

    #[test]
    fn test_get_tol_stagnation_generations() {
        assert_eq!(Some(0), get_tol_stagnation_generations(0, 0));
        assert_eq!(None, get_tol_stagnation_generations(100, 0));
        assert_eq!(None, get_tol_stagnation_generations(100, 200));
        assert_eq!(Some(100), get_tol_stagnation_generations(100, 500));
        assert_eq!(Some(400), get_tol_stagnation_generations(100, 2000));
        assert_eq!(Some(20_000), get_tol_stagnation_generations(100, 1_000_000));
        assert_eq!(None, get_tol_stagnation_generations(30_000, 1_000_000));
    }

    const DEFAULT_INITIAL_SIGMA: f64 = 0.5;
    const DIM: usize = 2;
    const TOL_STAGNATION: usize = 40;

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

    /// Sets up the state and parameters using the map_* functions and checks the termination
    /// criteria
    fn run_termination_test<S, H, P, R>(
        mode: Mode,
        time_created: Option<Instant>,
        initial_sigma: Option<f64>,
        current_function_evals: usize,
        current_generation_function_value: f64,
        map_state: S,
        map_history: H,
        map_parameters: P,
        map_results: R,
    ) where
        S: FnOnce(&mut State),
        H: FnOnce(&mut History),
        P: FnOnce(&mut TerminationParameters),
        R: FnOnce(Vec<TerminationReason>),
    {
        let initial_sigma = initial_sigma.unwrap_or(DEFAULT_INITIAL_SIGMA);
        let initial_mean = DVector::from(vec![0.0; DIM]);

        let mut state = State::new(initial_mean.clone(), initial_sigma);
        map_state(&mut state);

        let mut history = History::new();
        map_history(&mut history);

        // Take most default parameters from default CMAESOptions
        let options = CMAESOptions::new(initial_mean, initial_sigma).tol_stagnation(TOL_STAGNATION);
        let mut termination_parameters = TerminationParameters::from_options(&options);
        map_parameters(&mut termination_parameters);

        let lambda = 6;
        let cm = 1.0;
        let parameters = Parameters::new(
            mode,
            DIM,
            lambda,
            Weights::Negative,
            0,
            initial_sigma,
            cm,
            termination_parameters,
        );

        let results = TerminationCheck {
            current_function_evals,
            time_created: time_created.unwrap_or_else(Instant::now),
            parameters: &parameters,
            state: &state,
            history: &history,
            individuals: &get_dummy_generation(current_generation_function_value),
        }
        .check_termination_criteria();

        map_results(results);
    }

    #[test]
    fn test_check_termination_criteria_none() {
        // A fresh state should not meet any termination criteria
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |_| {},
            |_| {},
            |_| {},
            |results| assert!(results.is_empty()),
        );
    }

    #[test]
    fn test_check_termination_criteria_none_large_std_dev() {
        // A large standard deviation in one axis should not meet any termination criteria if the
        // initial step size was also large
        let initial_sigma = Some(1e3);

        run_termination_test(
            Mode::Minimize,
            None,
            initial_sigma,
            400,
            1.0,
            |state| {
                *state.mut_sigma() = 1e6;
                state
                    .mut_cov()
                    .set_cov(
                        SquareMatrix::from_iterator(DIM, DIM, [0.01, 0.0, 0.0, 1e6]),
                        true,
                    )
                    .unwrap();
            },
            |_| {},
            |_| {},
            |results| assert!(results.is_empty()),
        );
    }

    #[test]
    fn test_check_termination_criteria_max_function_evals() {
        let current_function_evals = 100;

        run_termination_test(
            Mode::Minimize,
            None,
            None,
            current_function_evals,
            1.0,
            |_| {},
            |_| {},
            |params| params.max_function_evals = Some(current_function_evals),
            |results| assert_eq!(results, &[TerminationReason::MaxFunctionEvals]),
        );
    }

    #[test]
    fn test_check_termination_criteria_max_generations() {
        let current_generation = 100;

        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |state| *state.mut_generation() = current_generation,
            |_| {},
            |params| params.max_generations = Some(current_generation),
            |results| assert_eq!(results, &[TerminationReason::MaxGenerations]),
        );
    }

    #[test]
    fn test_check_termination_criteria_max_time() {
        let max_time = Duration::from_secs(4);
        let time_created = Instant::now() - Duration::from_secs(5);

        run_termination_test(
            Mode::Minimize,
            Some(time_created),
            None,
            400,
            1.0,
            |_| {},
            |_| {},
            |params| params.max_time = Some(max_time),
            |results| assert_eq!(results, &[TerminationReason::MaxTime]),
        );
    }

    #[test]
    fn test_check_termination_criteria_fun_target() {
        // A best function value better than the threshold produces FunTarget
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1e-15,
            |_| {},
            |_| {},
            |params| params.fun_target = Some(1e-12),
            |results| assert_eq!(results, &[TerminationReason::FunTarget]),
        );

        run_termination_test(
            Mode::Maximize,
            None,
            None,
            400,
            25.0,
            |_| {},
            |_| {},
            |params| params.fun_target = Some(10.0),
            |results| assert_eq!(results, &[TerminationReason::FunTarget]),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun() {
        // Small ranges of current and historical function values produces TolFun
        let historical_best = 1.0;
        let most_recent_best = historical_best - 1e-13;

        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            most_recent_best,
            |_| {},
            |history| {
                history
                    .mut_best_function_values()
                    .extend(vec![historical_best; 100]);
                history
                    .mut_best_function_values()
                    .push_front(most_recent_best);
            },
            // TolFunHist must be disabled because it overlaps
            |params| params.tol_fun_hist = 0.0,
            |results| assert_eq!(results, &[TerminationReason::TolFun]),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun_rel() {
        // Small ranges of current and historical function values relative to the overall
        // improvement in median function value produces TolFunRel
        let historical_best = 1.0;
        // Outside the range of TolFun/TolFunHist
        let most_recent_best = historical_best - 0.01;

        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            most_recent_best,
            |_| {},
            |history| {
                // A very large improvement increases the range of TolFunRel
                *history.mut_first_median_function_value() = Some(1e12);
                *history.mut_best_median_function_value() = Some(1.0);

                history
                    .mut_best_function_values()
                    .extend(vec![historical_best; 100]);
                history
                    .mut_best_function_values()
                    .push_front(most_recent_best);
            },
            |params| params.tol_fun_rel = 1e-12,
            |results| assert_eq!(results, &[TerminationReason::TolFunRel]),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_fun_hist() {
        // A small range of historical best values produces TolFunHist
        let historical_best = 1.0;
        let most_recent_best = historical_best - 1e-13;

        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            historical_best,
            |_| {},
            |history| {
                history
                    .mut_best_function_values()
                    .extend(vec![historical_best; 100]);
                history
                    .mut_best_function_values()
                    .push_front(most_recent_best);
            },
            // TolFun must be disabled because it overlaps
            |params| params.tol_fun = 0.0,
            |results| assert_eq!(results, &[TerminationReason::TolFunHist]),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_x() {
        // A small step size and evolution path (zero length in this case) produces TolX
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |state| *state.mut_sigma() = 1e-13,
            |_| {},
            |_| {},
            |results| assert_eq!(results, &[TerminationReason::TolX]),
        );
    }

    fn check_termination_criteria_tol_stagnation(mode: Mode, historical_values: [f64; 4]) {
        // Median/best function values that worsen or don't improve over many generations produces
        // TolStagnation
        // TolStagnation is disabled until the generation is high enough
        let map_state = |state: &mut State| *state.mut_generation() = TOL_STAGNATION * 5;
        let map_history = |history: &mut History| {
            let mut values = Vec::new();
            values.extend(vec![historical_values[0]; TOL_STAGNATION / 4]);
            values.extend(vec![historical_values[1]; TOL_STAGNATION / 4]);
            values.extend(vec![historical_values[2]; TOL_STAGNATION / 4]);
            values.extend(vec![historical_values[3]; TOL_STAGNATION / 4]);
            *history.mut_best_function_values() = values.clone().into();
            *history.mut_median_function_values() = values.clone().into();
        };

        let run = |tol_stagnation, expected: &[TerminationReason]| {
            run_termination_test(
                mode,
                None,
                None,
                400,
                1.0,
                map_state,
                map_history,
                |params| params.tol_stagnation = tol_stagnation,
                |results| assert_eq!(results, expected),
            );
        };

        run(TOL_STAGNATION, &[TerminationReason::TolStagnation]);
        // Check that no panic occurs if tol_stagnation is 0 (this is only the lower bound, so it
        // is still checked over TOL_STAGNATION generations in this case)
        run(0, &[TerminationReason::TolStagnation]);
    }

    #[test]
    fn test_check_termination_criteria_tol_stagnation_minimize() {
        check_termination_criteria_tol_stagnation(Mode::Minimize, [3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_check_termination_criteria_tol_stagnation_maximize() {
        check_termination_criteria_tol_stagnation(Mode::Maximize, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_check_termination_criteria_tol_x_up() {
        // A large increase in maximum standard deviation produces TolXUp
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |state| *state.mut_sigma() = 1e8,
            |_| {},
            |_| {},
            |results| assert_eq!(results, &[TerminationReason::TolXUp]),
        );
    }

    #[test]
    fn test_check_termination_criteria_no_effect_axis() {
        // A lack of available precision along a principal axis in the distribution produces
        // NoEffectAxis
        let mut terminated_count = 0;
        for g in 0..DIM {
            run_termination_test(
                Mode::Minimize,
                None,
                None,
                400,
                1.0,
                |state| {
                    // Change the axis being checked
                    *state.mut_generation() = g;

                    // Shrink the distribution and reduce available precision by moving the mean
                    *state.mut_mean() = vec![100.0; 2].into();
                    *state.mut_sigma() = 1e-10;

                    // Adjust the shape of the distribution to not be coordinate-axis-aligned
                    let eigenvectors = SquareMatrix::from_columns(&[
                        DVector::from(vec![3.0, 2.0]).normalize(),
                        DVector::from(vec![-2.0, 3.0]).normalize(),
                    ]);
                    // Adjust the distribution axis scales
                    let sqrt_eigenvalues = SquareMatrix::from_diagonal(&vec![1e-1, 1e-6].into());
                    let cov = &eigenvectors * sqrt_eigenvalues.pow(2) * eigenvectors.transpose();
                    state.mut_cov().set_cov(cov, true).unwrap();
                },
                |_| {},
                |_| {},
                |results| {
                    if !results.is_empty() {
                        assert_eq!(results, &[TerminationReason::NoEffectAxis]);
                        terminated_count += 1;
                    }
                },
            );
        }

        // Only one axis should match
        assert_eq!(terminated_count, 1);
    }

    #[test]
    fn test_check_termination_criteria_no_effect_coord() {
        // A lack of available precision along a coordinate axis in the distribution produces
        // NoEffectCoord
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |state| {
                // Reduce available precision by moving the mean
                *state.mut_mean() = vec![100.0; 2].into();

                // Adjust the coordinate axis scales (the eigenvectors are
                // coordinate-axis-aligned)
                let eigenvectors = SquareMatrix::<f64>::identity(2, 2);
                let sqrt_eigenvalues = SquareMatrix::from_diagonal(&vec![1e-4, 1e-10].into());
                let cov = &eigenvectors * sqrt_eigenvalues.pow(2) * eigenvectors.transpose();
                state.mut_cov().set_cov(cov, true).unwrap();
            },
            |_| {},
            |_| {},
            |results| assert_eq!(results, &[TerminationReason::NoEffectCoord]),
        );
    }

    #[test]
    fn test_check_termination_criteria_tol_condition_cov() {
        // A large difference between the maximum and minimum standard deviations produces
        // TolConditionCov
        run_termination_test(
            Mode::Minimize,
            None,
            None,
            400,
            1.0,
            |state| {
                state
                    .mut_cov()
                    .set_cov(
                        SquareMatrix::from_iterator(2, 2, [0.99, 0.0, 0.0, 1e14]),
                        true,
                    )
                    .unwrap();
            },
            |_| {},
            |_| {},
            |results| assert_eq!(results, &[TerminationReason::TolConditionCov]),
        );
    }
}
