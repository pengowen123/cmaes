//! An IPOP restart strategy, increasing the population size each restart

use rand_chacha::ChaChaRng;

use super::strategy::{RestartControl, Strategy};
use super::InvalidRestartStrategyOptionsError;
use crate::{CMAESOptions, TerminationData, TerminationReason, CMAES};

/// The default population size increase factor
const DEFAULT_INCREASE_FACTOR: usize = 2;
/// The maximum number of runs allowed
const MAX_RUNS: usize = 10;

/// An IPOP restart strategy, increasing the population size each restart. Also called IPOP-aCMA-ES.
///
/// Useful for most problems, but especially useful for multimodal functions with exploitable
/// global structures. For multimodal functions without global structures, the
/// [`Local`][crate::restart::Local] or [`BIPOP`][crate::restart::BIPOP] strategies may perform
/// better.
#[derive(Clone, Debug)]
pub struct IPOP {
    /// The number of runs performed so far
    runs: usize,
    /// The current multiplier for the population size
    /// The first run is always with the default population size
    current_multiplier: usize,
    /// The factor by which to increase the population size each restart
    increase_factor: usize,
}

impl IPOP {
    /// Returns a new `IPOP` that increases the population size by a factor of `increase_factor`
    /// each restart. The first run uses the default population size.
    ///
    /// [`Default::default`][Self::default] uses a factor of `2`, which should perform well on most
    /// problems.
    pub fn new(increase_factor: usize) -> Result<Self, InvalidRestartStrategyOptionsError> {
        if increase_factor == 0 {
            Err(InvalidRestartStrategyOptionsError::PopulationSize)
        } else {
            Ok(Self {
                runs: 0,
                current_multiplier: 1,
                increase_factor,
            })
        }
    }

    /// Returns the initial step size to use for the given search range size
    fn get_initial_step_size(&self, search_range_size: f64) -> f64 {
        search_range_size / 2.0
    }
}

impl Default for IPOP {
    fn default() -> Self {
        Self::new(DEFAULT_INCREASE_FACTOR).unwrap()
    }
}

impl Strategy for IPOP {
    fn has_zero_max_runs(&self) -> bool {
        false
    }

    fn next_run<F, R: FnOnce(&mut CMAES<F>) -> TerminationData>(
        &mut self,
        mut options: CMAESOptions,
        search_range_size: f64,
        objective_function: F,
        run: R,
        _: &mut ChaChaRng,
    ) -> (CMAES<F>, Vec<TerminationReason>, RestartControl) {
        // Configure the run
        options.initial_step_size = self.get_initial_step_size(search_range_size);
        options.population_size *= self.current_multiplier;

        // Execute the run
        let mut cmaes_state = options.build(objective_function).unwrap();
        let results = run(&mut cmaes_state);

        // Update internal state
        self.runs += 1;
        self.current_multiplier *= self.increase_factor;

        let control = if self.runs >= MAX_RUNS {
            RestartControl::MaxRunsReached
        } else {
            RestartControl::Continue
        };

        (cmaes_state, results.reasons, control)
    }

    fn get_algorithm_name(&self) -> &'static str {
        "IPOP-aCMA-ES"
    }

    fn get_parameters_as_strings(&self) -> Vec<(String, String)> {
        [(
            "increase_factor".to_string(),
            format!("{}", self.increase_factor),
        )]
        .into()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_new() {
        assert!(IPOP::new(1).is_ok());
        assert!(matches!(
            IPOP::new(0),
            Err(InvalidRestartStrategyOptionsError::PopulationSize)
        ));
    }

    #[test]
    fn test_get_initial_step_size() {
        let ipop = IPOP::default();

        assert_eq!(50.0, ipop.get_initial_step_size(100.0));
    }

    #[test]
    fn test_ipop() {
        let mut ipop = IPOP::default();
        let function = |x: &DVector<f64>| x.magnitude();

        for i in 0..10 {
            let (_, _, control) = ipop.next_run(
                CMAESOptions::new(vec![1.0; 2], 0.5),
                1.0,
                function,
                |state| state.run(),
                &mut ChaChaRng::seed_from_u64(rand::random()),
            );

            assert_eq!(i + 1 == 10, control.should_terminate());
        }

        assert_eq!(10, ipop.runs);
        // This is the value that would be used for the next run
        assert_eq!(2usize.pow(10), ipop.current_multiplier);
        assert_eq!(2, ipop.increase_factor);
    }
}
