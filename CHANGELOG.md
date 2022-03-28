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

## 0.1.1 (March 4th, 2022)

- Exposed `nalgebra-lapack` features to allow choosing the LAPACK provider ([#5](https://github.com/pengowen123/cmaes/pull/5)).

## 0.1.0 (February 16th, 2022)

- Initial release
