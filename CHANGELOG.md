# Changelog

## Unreleased

- Renamed `CMAESState` to `CMAES`.
- Added `Plot::len`, `Plot::is_empty`, and `Plot::capacity` methods to retrieve the number of data points stored and allocated for.
- Added `Scale` wrapper type for objective functions that scales the search space.
- Added `MaxFunctionEvals` and `MaxGenerations` termination criteria.
- Adjusted the signatures of `CMAES::run` and `CMAES::print_final_info` to reflect the new termination criteria.
- Renamed `reason` field of `TerminationData` to `reasons` and changed its type from `TerminationReason` to `Vec<TerminationReason>`.
- Changed signature of `CMAES::print_final_info` to take `&[TerminationReason]` instead of `TerminationReason`.
- Fixed `TolFun` termination criterion checking whether the current generation's function values are all below `tol_fun` instead of whether the range of them is.
- Added `FunTarget` termination criterion that checks whether a function value threshold has been reached.
- Added `tol_x_up` option to configure `TolXUp` termination criterion.
- Renamed `ConditionCov` termination criterion to `TolConditionCov` and added `tol_condition_cov` option to configure it.
- Added `MaxTime` termination criterion to limit the running time of the algorithm.
- Renamed `EqualFunValues` termination criterion to `TolFunHist` and added `tol_fun_hist` option to configure its range.
- Added `TolFunRel` termination criterion to allow for a relative tolerance in function.
- Renamed `Stagnation` to `TolStagnation` and added `tol_stagnation` option to configure its lower bound.
- Fixed panic occurring when the algorithm immediately terminates with `InvalidFunctionValue`.
- Changed the types of the `current_best` and `overall_best` fields of `TerminationData` from `Individual` to `Option<Individual>`.
- Made `initial_mean` and `initial_step_size` the required options for `CMAESOptions`. `dimensions` is now taken from `initial_mean`. Also removed `InvalidOptionsError::MeanDimension` to reflect these changes.

## 0.1.1 (March 4th, 2022)

- Exposed `nalgebra-lapack` features to allow choosing the LAPACK provider ([#5](https://github.com/pengowen123/cmaes/pull/5)).

## 0.1.0 (February 16th, 2022)

- Initial release
