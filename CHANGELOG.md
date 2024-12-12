# Changelog

## 0.2.2 (December 13th, 2024)

### Miscellaneous

- Bumped dependencies ([#9](https://github.com/pengowen123/cmaes/pull/9)).

## 0.2.1 (May 18th, 2022)

### Features

- Made LAPACK dependency optional, defaulting to using `nalgebra`. This allows the library to be built on Windows
([#8](https://github.com/pengowen123/cmaes/pull/8)).

## 0.2.0 (April 21st, 2022)

### Features

- Added `restart` module that implements the `LR`, `IPOP`, and `BIPOP` automatic restart algorithms.
- Added `ParallelObjectiveFunction` trait and `CMAES::run_parallel` and `CMAES::next_parallel` methods to allow parallel execution of objective functions.
- Added `mode` option to choose whether to minimize or maximize the objective function.
- Added `fmin`, `fmax`, `fmin_parallel`, and `fmax_parallel` functions for convenience in cases where configuration is not required.
- Added `Scale` wrapper type for objective functions that scales the search space.
- Added `parallel_update` option to choose whether to perform the state update in parallel, which can improve performance for large population sizes.
- Added the following termination criteria: `MaxFunctionEvals`, `MaxGenerations`, `MaxTime`, `FunTarget`, `TolFunRel`.
- Added options to configure the following termination criteria: `TolXUp`, `TolConditionCov`, `TolFunHist` (formerly `EqualFunValues`), `TolStagnation`.
- Added median objective function value to data plots.
- Added `Plot::len`, `Plot::is_empty`, and `Plot::capacity` methods to retrieve the number of data points stored and allocated for.

### Changes

- Made `initial_mean` and `initial_step_size` the required options for `CMAESOptions`. `dimensions` is now taken from `initial_mean`. Also removed `InvalidOptionsError::MeanDimension` to reflect these changes.
- Switched to static dispatch for objective function types.
- Renamed `CMAESState` to `CMAES`.
- Removed the `max_generations` argument of `CMAES::run` (covered by a termination criterion now).
- Renamed `reason` field of `TerminationData` to `reasons` and changed its type from `TerminationReason` to `Vec<TerminationReason>`.
- Changed the types of the `current_best` and `overall_best` fields of `TerminationData` from `Individual` to `Option<Individual>`.
- Renamed `ConditionCov` termination criterion to `TolConditionCov`.
- Renamed `Stagnation` termination criterion to `TolStagnation`.
- Replaced `EqualFunValues` termination criterion with `TolFunHist`.
- Changed signature of `CMAES::print_final_info` to take `&[TerminationReason]` instead of `Option<TerminationReason>`.

### Fixes

- Fixed `TolFun` termination criterion checking whether the current generation's function values are all below `tol_fun` instead of whether the range of them is.
- Fixed panic occurring when the algorithm immediately terminates with `InvalidFunctionValue`.

### Performance

- Improved overall performance. Iteration times are reduced by between ~10% and ~65% depending on the problem dimension and population size.

## 0.1.1 (March 4th, 2022)

- Exposed `nalgebra-lapack` features to allow choosing the LAPACK provider ([#5](https://github.com/pengowen123/cmaes/pull/5)).

## 0.1.0 (February 16th, 2022)

- Initial release
