//! Objective function value history tracking.

use std::collections::VecDeque;

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
    pub fn update(&mut self, current_generation: &[EvaluatedPoint]) {
        let best = &current_generation[0];

        self.best_function_values.push_front(best.value());
        if self.best_function_values.len() > MAX_HISTORY_LENGTH {
            self.best_function_values.pop_back();
        }

        let median_value = if current_generation.len() % 2 == 0 {
            (current_generation[current_generation.len() / 2 - 1].value()
                + current_generation[current_generation.len() / 2].value())
                / 2.0
        } else {
            current_generation[current_generation.len() / 2].value()
        };
        self.median_function_values.push_front(median_value);
        if self.median_function_values.len() > MAX_HISTORY_LENGTH {
            self.median_function_values.pop_back();
        }

        self.first_median_function_value = self.first_median_function_value.or(Some(median_value));

        match self.best_median_function_value {
            Some(ref mut value) => *value = value.min(median_value),
            None => self.best_median_function_value = Some(median_value),
        }

        self.update_best_individuals(Individual::new(best.point().clone(), best.value()));
    }

    /// Updates the current and overall best individuals
    fn update_best_individuals(&mut self, current_best: Individual) {
        self.current_best_individual = Some(current_best.clone());

        match &mut self.overall_best_individual {
            Some(ref mut overall) => {
                if current_best.value < overall.value {
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

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;

    #[test]
    fn test_update() {
        let mut history = History::new();

        assert!(history.best_function_values().is_empty());
        assert!(history.median_function_values().is_empty());

        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            counter -= 1.0;
            counter
        };

        let mut update = |h: &mut History| {
            h.update(&[
                EvaluatedPoint::new(DVector::zeros(4), &DVector::zeros(4), 0.0, &mut function)
                    .unwrap(),
                EvaluatedPoint::new(DVector::zeros(4), &DVector::zeros(4), 0.0, &mut function)
                    .unwrap(),
            ]);
        };

        update(&mut history);

        assert_eq!(-1.0, history.best_function_values()[0]);
        assert_eq!(-1.5, history.median_function_values()[0]);

        for _ in 0..MAX_HISTORY_LENGTH + 5 {
            update(&mut history);
        }

        assert_eq!(MAX_HISTORY_LENGTH, history.best_function_values().len());
        assert_eq!(MAX_HISTORY_LENGTH, history.median_function_values().len());
    }

    #[test]
    fn test_update_best_individuals() {
        let mut history = History::new();
        let origin = DVector::from(vec![0.0; 10]);

        assert!(history.current_best_individual().is_none());
        assert!(history.overall_best_individual().is_none());

        history.update_best_individuals(Individual::new(origin.clone(), 7.0));
        assert_eq!(history.current_best_individual().unwrap().value, 7.0);
        assert_eq!(history.overall_best_individual().unwrap().value, 7.0);

        history.update_best_individuals(Individual::new(origin.clone(), 1.0));
        assert_eq!(history.current_best_individual().unwrap().value, 1.0);
        assert_eq!(history.overall_best_individual().unwrap().value, 1.0);

        history.update_best_individuals(Individual::new(origin.clone(), 2.0));
        assert_eq!(history.current_best_individual().unwrap().value, 2.0);
        assert_eq!(history.overall_best_individual().unwrap().value, 1.0);

        history.update_best_individuals(Individual::new(origin.clone(), 1.5));
        assert_eq!(history.current_best_individual().unwrap().value, 1.5);
        assert_eq!(history.overall_best_individual().unwrap().value, 1.0);

        history.update_best_individuals(Individual::new(origin.clone(), 0.5));
        assert_eq!(history.current_best_individual().unwrap().value, 0.5);
        assert_eq!(history.overall_best_individual().unwrap().value, 0.5);
    }

    #[test]
    fn test_median_value_decreasing() {
        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            counter -= 1.0;
            counter
        };
        let mut history = History::new();

        assert!(history.first_median_function_value().is_none());
        assert!(history.best_median_function_value().is_none());

        history.update(&[EvaluatedPoint::new(
            DVector::zeros(4),
            &DVector::zeros(4),
            0.0,
            &mut function,
        )
        .unwrap()]);

        // For the first generation both values are equal
        let first = history.first_median_function_value().unwrap();
        let best = history.best_median_function_value().unwrap();

        assert_eq!(first, best);

        history.update(&[EvaluatedPoint::new(
            DVector::zeros(4),
            &DVector::zeros(4),
            0.0,
            &mut function,
        )
        .unwrap()]);

        // For subsequent generations only the best value improves
        assert_eq!(first, history.first_median_function_value().unwrap());
        assert!(best > history.best_median_function_value().unwrap());
    }

    #[test]
    fn test_median_value_increasing() {
        let mut counter = 0.0;
        let mut function = |_: &DVector<f64>| {
            counter += 1.0;
            counter
        };
        let mut history = History::new();

        history.update(&[EvaluatedPoint::new(
            DVector::zeros(4),
            &DVector::zeros(4),
            0.0,
            &mut function,
        )
        .unwrap()]);

        // For the first generation both values are equal
        let first = history.first_median_function_value().unwrap();
        let best = history.best_median_function_value().unwrap();

        history.update(&[EvaluatedPoint::new(
            DVector::zeros(4),
            &DVector::zeros(4),
            0.0,
            &mut function,
        )
        .unwrap()]);

        // For subsequent generations neither value changes because the median is increasing
        assert_eq!(first, history.first_median_function_value().unwrap());
        assert_eq!(best, history.best_median_function_value().unwrap());
    }
}
