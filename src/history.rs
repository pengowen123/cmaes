//! Objective function value history tracking.

use std::collections::VecDeque;

use crate::mode::Mode;
use crate::sampling::EvaluatedPoint;
use crate::Individual;

/// The maximum number of elements to store in the objective function value histories.
pub const MAX_HISTORY_LENGTH: usize = 20_000;

/// A type that tracks various histories of the objective function value, as well as the current
/// and overall best points.
pub struct History {
    /// A history of the best function values (values at the front are from more recent generations)
    best_function_values: VecDeque<f64>,
    /// A history of the median function values (values at the front are from more recent
    /// generations)
    median_function_values: VecDeque<f64>,
    /// The best individual of the latest generation
    current_best_individual: Option<Individual>,
    /// The best individual of any generation
    overall_best_individual: Option<Individual>,
    /// The median function value of the first generation
    first_median_function_value: Option<f64>,
    /// The best median function value of any generation
    best_median_function_value: Option<f64>,
}

impl History {
    pub fn new() -> Self {
        Self {
            best_function_values: VecDeque::new(),
            median_function_values: VecDeque::new(),
            current_best_individual: None,
            overall_best_individual: None,
            first_median_function_value: None,
            best_median_function_value: None,
        }
    }

    pub fn best_function_values(&self) -> &VecDeque<f64> {
        &self.best_function_values
    }

    pub fn median_function_values(&self) -> &VecDeque<f64> {
        &self.median_function_values
    }

    /// Always `Some` if `Self::update` has been called at least once
    pub fn current_best_individual(&self) -> Option<&Individual> {
        self.current_best_individual.as_ref()
    }

    /// Always `Some` if `Self::update` has been called at least once
    pub fn overall_best_individual(&self) -> Option<&Individual> {
        self.overall_best_individual.as_ref()
    }

    /// Always `Some` if `Self::update` has been called at least once
    #[cfg(feature = "plotters")]
    pub fn current_median_function_value(&self) -> Option<f64> {
        self.median_function_values.get(0).cloned()
    }

    /// Always `Some` if `Self::update` has been called at least once
    pub fn first_median_function_value(&self) -> Option<f64> {
        self.first_median_function_value
    }

    /// Always `Some` if `Self::update` has been called at least once
    pub fn best_median_function_value(&self) -> Option<f64> {
        self.best_median_function_value
    }

    /// Updates the histories based on the current generation of individuals. Assumes that
    /// `current_generation` is already sorted by objective function value.
    pub fn update(&mut self, mode: Mode, current_generation: &[EvaluatedPoint]) {
        let best = &current_generation[0];

        self.best_function_values.push_front(best.value());
        if self.best_function_values.len() > MAX_HISTORY_LENGTH {
            self.best_function_values.pop_back();
        }

        let median_value = get_median_value(current_generation);
        self.median_function_values.push_front(median_value);
        if self.median_function_values.len() > MAX_HISTORY_LENGTH {
            self.median_function_values.pop_back();
        }

        self.first_median_function_value = self.first_median_function_value.or(Some(median_value));

        match self.best_median_function_value {
            Some(ref mut value) => *value = mode.choose_best(*value, median_value),
            None => self.best_median_function_value = Some(median_value),
        }

        self.update_best_individuals(mode, Individual::new(best.point().clone(), best.value()));
    }

    /// Updates the current and overall best individuals
    fn update_best_individuals(&mut self, mode: Mode, current_best: Individual) {
        self.current_best_individual = Some(current_best.clone());

        match &mut self.overall_best_individual {
            Some(ref mut overall) => {
                if mode.is_better(current_best.value, overall.value) {
                    *overall = current_best;
                }
            }
            None => self.overall_best_individual = Some(current_best),
        }
    }

    #[cfg(test)]
    pub fn mut_best_function_values(&mut self) -> &mut VecDeque<f64> {
        &mut self.best_function_values
    }

    #[cfg(test)]
    pub fn mut_median_function_values(&mut self) -> &mut VecDeque<f64> {
        &mut self.median_function_values
    }

    #[cfg(test)]
    pub fn mut_first_median_function_value(&mut self) -> &mut Option<f64> {
        &mut self.first_median_function_value
    }

    #[cfg(test)]
    pub fn mut_best_median_function_value(&mut self) -> &mut Option<f64> {
        &mut self.best_median_function_value
    }
}

/// Returns the median function value of the generation
/// Assumes that `individuals` is already sorted by function value
fn get_median_value(individuals: &[EvaluatedPoint]) -> f64 {
    if individuals.len() % 2 == 0 {
        (individuals[individuals.len() / 2 - 1].value()
            + individuals[individuals.len() / 2].value())
            / 2.0
    } else {
        individuals[individuals.len() / 2].value()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;

    #[test]
    fn test_get_median_value() {
        let get_point = |value| {
            let zeros = DVector::zeros(2);
            EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 1.0, &mut |_: &_| {
                value
            })
            .unwrap()
        };
        let get_generation =
            |values: &[f64]| values.iter().map(|v| get_point(*v)).collect::<Vec<_>>();

        assert_eq!(
            2.0,
            get_median_value(&get_generation(&[0.0, 1.0, 2.0, 3.0, 4.0]))
        );
        assert_eq!(
            1.5,
            get_median_value(&get_generation(&[0.0, 1.0, 2.0, 3.0]))
        );
    }

    fn update_shared(mode: Mode, expected_best: f64, expected_median: f64) {
        let mut history = History::new();

        assert!(history.best_function_values().is_empty());
        assert!(history.median_function_values().is_empty());

        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            match mode {
                Mode::Minimize => counter -= 1.0,
                Mode::Maximize => counter += 1.0,
            }
            counter
        };

        let mut update = |h: &mut History| {
            let zeros = DVector::zeros(4);
            let mut generation = [
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
            ];

            generation.sort_by(|a, b| mode.sort_cmp(a.value(), b.value()));

            h.update(mode, &generation);
        };

        update(&mut history);

        assert_eq!(expected_best, history.best_function_values()[0]);
        assert_eq!(expected_median, history.median_function_values()[0]);

        for _ in 0..MAX_HISTORY_LENGTH + 5 {
            update(&mut history);
        }

        assert_eq!(MAX_HISTORY_LENGTH, history.best_function_values().len());
        assert_eq!(MAX_HISTORY_LENGTH, history.median_function_values().len());
    }

    #[test]
    fn test_update_minimize() {
        update_shared(Mode::Minimize, -2.0, -1.5);
    }

    #[test]
    fn test_update_maximize() {
        update_shared(Mode::Maximize, 2.0, 1.5);
    }

    fn update_and_test(
        mode: Mode,
        history: &mut History,
        current_best: f64,
        expected_current_best: f64,
        expected_overall_best: f64,
    ) {
        history.update_best_individuals(mode, Individual::new(DVector::zeros(4), current_best));
        assert_eq!(
            history.current_best_individual().unwrap().value,
            expected_current_best
        );
        assert_eq!(
            history.overall_best_individual().unwrap().value,
            expected_overall_best
        );
    }

    #[test]
    fn test_update_best_individuals_minimize() {
        let mut history = History::new();
        let mode = Mode::Minimize;

        assert!(history.current_best_individual().is_none());
        assert!(history.overall_best_individual().is_none());

        update_and_test(mode, &mut history, 1.0, 1.0, 1.0);
        update_and_test(mode, &mut history, 2.0, 2.0, 1.0);
        update_and_test(mode, &mut history, 1.5, 1.5, 1.0);
        update_and_test(mode, &mut history, 0.5, 0.5, 0.5);
    }

    #[test]
    fn test_update_best_individuals_maximize() {
        let mut history = History::new();
        let mode = Mode::Maximize;

        assert!(history.current_best_individual().is_none());
        assert!(history.overall_best_individual().is_none());

        update_and_test(mode, &mut history, 3.0, 3.0, 3.0);
        update_and_test(mode, &mut history, 2.0, 2.0, 3.0);
        update_and_test(mode, &mut history, 2.5, 2.5, 3.0);
        update_and_test(mode, &mut history, 4.0, 4.0, 4.0);
    }

    fn median_value_improving(mode: Mode) {
        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            match mode {
                // The value improves with each call
                Mode::Minimize => counter -= 1.0,
                Mode::Maximize => counter += 1.0,
            }
            counter
        };
        let mut history = History::new();

        assert!(history.first_median_function_value().is_none());
        assert!(history.best_median_function_value().is_none());

        let zeros = DVector::zeros(4);
        history.update(
            mode,
            &[
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
            ],
        );

        // For the first generation both values are equal
        let first = history.first_median_function_value().unwrap();
        let best = history.best_median_function_value().unwrap();

        assert_eq!(first, best);

        history.update(
            mode,
            &[
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
            ],
        );

        // For subsequent generations only the best value improves
        assert_eq!(first, history.first_median_function_value().unwrap());
        assert!(mode.is_better(history.best_median_function_value().unwrap(), best));
    }

    #[test]
    fn test_median_value_improving_minimize() {
        median_value_improving(Mode::Minimize);
    }

    #[test]
    fn test_median_value_improving_maximize() {
        median_value_improving(Mode::Maximize);
    }

    fn median_value_worsening(mode: Mode) {
        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            match mode {
                // The value worsens with each call
                Mode::Minimize => counter += 1.0,
                Mode::Maximize => counter -= 1.0,
            }
            counter
        };
        let mut history = History::new();

        let zeros = DVector::zeros(4);
        history.update(
            mode,
            &[
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
            ],
        );

        // For the first generation both values are equal
        let first = history.first_median_function_value().unwrap();
        let best = history.best_median_function_value().unwrap();

        history.update(
            mode,
            &[
                EvaluatedPoint::new(zeros.clone(), zeros.clone(), &zeros, 0.0, &mut function)
                    .unwrap(),
            ],
        );

        // For subsequent generations neither value changes because the median is increasing
        assert_eq!(first, history.first_median_function_value().unwrap());
        assert_eq!(best, history.best_median_function_value().unwrap());
    }

    #[test]
    fn test_median_value_worsening_minimize() {
        median_value_worsening(Mode::Minimize);
    }

    #[test]
    fn test_median_value_worsening_maximize() {
        median_value_worsening(Mode::Maximize);
    }
}
