//! Abstractions over different restart strategies

use rand_chacha::ChaChaRng;

use super::{Local, BIPOP, IPOP};
use crate::{CMAESOptions, TerminationData, TerminationReason, CMAES};

/// A type returned by restart strategies to allow them to control restart execution
#[derive(Clone, Copy, Debug)]
pub enum RestartControl {
    /// Continue executing more restarts (if no termination criteria are met)
    Continue,
    /// Stop executing restarts (the run that returns this will be the final one)
    MaxRunsReached,
}

impl RestartControl {
    /// Returns whether this `RestartControl` value represents an instruction to terminate the
    /// restart strategy
    #[cfg(test)]
    pub fn should_terminate(&self) -> bool {
        match *self {
            RestartControl::Continue => false,
            RestartControl::MaxRunsReached => true,
        }
    }
}

/// The strategy to use in performing automatic restarts. A good default choice is `BIPOP`.
#[derive(Clone, Debug)]
pub enum RestartStrategy {
    /// See [`Local`].
    Local(Local),
    /// See [`IPOP`].
    IPOP(IPOP),
    /// See [`BIPOP`].
    BIPOP(BIPOP),
}

/// A trait implemented by all restart strategy types
pub trait Strategy {
    /// Returns whether the restart strategy has zero max runs and therefore should not be run at
    /// all. Normal termination is done with the `RestartControl` return value of
    /// `Strategy::next_run`.
    fn has_zero_max_runs(&self) -> bool;
    /// Configures and executes the next run (using `run`). Returns the final state of the run, the
    /// termination reasons of the run, and an instruction for how/whether to proceed with future
    /// restarts.
    ///
    /// - `default_options` is the default configuration for all restarts, including a random
    /// initial mean and seed. Individual options such as the initial step size should be overridden
    /// if necessary, although `max_function_evals` and `max_generations` should only ever be
    /// decreased from their default values in order to respect their corresponding `Restarter`
    /// options.
    /// - `search_range_size` is the size of the search range of the restart strategy and can be used
    /// to calculate other options.
    /// - `rng` should be used to generate any random numbers used.
    fn next_run<F, R: FnOnce(&mut CMAES<F>) -> TerminationData>(
        &mut self,
        default_options: CMAESOptions,
        search_range_size: f64,
        objective_function: F,
        run: R,
        rng: &mut ChaChaRng,
    ) -> (CMAES<F>, Vec<TerminationReason>, RestartControl);
    /// Returns the name of the algorithm with this restart strategy, e.g. BIPOP-aCMA-ES.
    fn get_algorithm_name(&self) -> &'static str;
    /// Returns the parameters of the restart strategy in a name to value map.
    fn get_parameters_as_strings(&self) -> Vec<(String, String)>;
}

impl Strategy for RestartStrategy {
    fn has_zero_max_runs(&self) -> bool {
        match *self {
            RestartStrategy::Local(ref s) => s.has_zero_max_runs(),
            RestartStrategy::IPOP(ref s) => s.has_zero_max_runs(),
            RestartStrategy::BIPOP(ref s) => s.has_zero_max_runs(),
        }
    }

    fn next_run<F, R: FnOnce(&mut CMAES<F>) -> TerminationData>(
        &mut self,
        default_options: CMAESOptions,
        search_range_size: f64,
        objective_function: F,
        run: R,
        rng: &mut ChaChaRng,
    ) -> (CMAES<F>, Vec<TerminationReason>, RestartControl) {
        match *self {
            RestartStrategy::Local(ref mut s) => s.next_run(
                default_options,
                search_range_size,
                objective_function,
                run,
                rng,
            ),
            RestartStrategy::IPOP(ref mut s) => s.next_run(
                default_options,
                search_range_size,
                objective_function,
                run,
                rng,
            ),
            RestartStrategy::BIPOP(ref mut s) => s.next_run(
                default_options,
                search_range_size,
                objective_function,
                run,
                rng,
            ),
        }
    }

    fn get_algorithm_name(&self) -> &'static str {
        match *self {
            RestartStrategy::Local(ref s) => s.get_algorithm_name(),
            RestartStrategy::IPOP(ref s) => s.get_algorithm_name(),
            RestartStrategy::BIPOP(ref s) => s.get_algorithm_name(),
        }
    }

    fn get_parameters_as_strings(&self) -> Vec<(String, String)> {
        match *self {
            RestartStrategy::Local(ref s) => s.get_parameters_as_strings(),
            RestartStrategy::IPOP(ref s) => s.get_parameters_as_strings(),
            RestartStrategy::BIPOP(ref s) => s.get_parameters_as_strings(),
        }
    }
}
