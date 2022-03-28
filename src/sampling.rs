//! Types related to sampling points from the distribution

use nalgebra::DVector;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use statrs::distribution::Normal;

use crate::state::State;
use crate::{utils, ObjectiveFunction};

/// A type for sampling and evaluating points from the distribution for each generation
pub struct Sampler<'a> {
    /// Number of dimensions to sample from
    dim: usize,
    /// Number of points to sample each generation
    population_size: usize,
    /// RNG from which all random numbers are sourced
    rng: ChaCha12Rng,
    /// The objective function to minimize, used to evaluate points
    objective_function: Box<dyn ObjectiveFunction + 'a>,
    /// The number of times the objective function has been evaluated
    function_evals: usize,
}

impl<'a> Sampler<'a> {
    pub fn new(
        dim: usize,
        population_size: usize,
        objective_function: Box<dyn ObjectiveFunction + 'a>,
        rng_seed: u64,
    ) -> Self {
        Self {
            dim,
            population_size,
            rng: ChaCha12Rng::seed_from_u64(rng_seed),
            objective_function,
            function_evals: 0,
        }
    }

    /// Samples and returns a new generation of points, sorted in ascending order by their
    /// corresponding objective function values
    ///
    /// Returns Err if the objective function returned an invalid value
    pub fn sample(
        &mut self,
        state: &State,
    ) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Random steps in the distribution N(0, cov)
        let y = (0..self.population_size)
            .map(|_| {
                DVector::from_iterator(
                    self.dim,
                    (0..self.dim).map(|_| normal.sample(&mut self.rng)),
                )
            })
            .map(|zk| state.cov_eigenvectors() * state.cov_sqrt_eigenvalues() * zk);

        // Evaluate and rank points
        let mut points = y
            .map(|yk| {
                EvaluatedPoint::new(
                    yk,
                    state.mean(),
                    state.sigma(),
                    &mut *self.objective_function,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.function_evals += points.len();

        points.sort_by(|a, b| utils::partial_cmp(a.value, b.value));
        Ok(points)
    }

    pub fn function_evals(&self) -> usize {
        self.function_evals
    }

    /// Consumes `self` and returns the objective function
    pub fn into_objective_function(self) -> Box<dyn ObjectiveFunction + 'a> {
        self.objective_function
    }
}

/// A point from the distribution that has been evaluated by the objective function
#[derive(Clone, Debug)]
pub struct EvaluatedPoint {
    /// The evaluated point
    point: DVector<f64>,
    /// The point before scaling/translation
    /// In the distribution N(0, cov)
    step: DVector<f64>,
    /// The objective value at the point
    value: f64,
}

impl EvaluatedPoint {
    /// Returns a new `EvaluatedPoint` from the untransformed step from the mean, the mean, and
    /// the step size
    ///
    /// Returns `Err` if the objective function returned an invalid value
    pub fn new<'a, F: ObjectiveFunction + ?Sized + 'a>(
        step: DVector<f64>,
        mean: &DVector<f64>,
        sigma: f64,
        objective_function: &mut F,
    ) -> Result<Self, InvalidFunctionValueError> {
        let point = mean + sigma * &step;
        let value = objective_function.evaluate(&point);

        if value.is_nan() {
            Err(InvalidFunctionValueError)
        } else {
            Ok(Self { point, step, value })
        }
    }

    pub fn point(&self) -> &DVector<f64> {
        &self.point
    }

    pub fn step(&self) -> &DVector<f64> {
        &self.step
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

/// The objective function returned an invalid value
#[derive(Clone, Debug)]
pub struct InvalidFunctionValueError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluated_point() {
        let dim = 5;
        let mean = DVector::from(vec![2.0; dim]);
        let step = DVector::from(vec![1.0; dim]);
        let sigma = 3.0;
        let mut function = |x: &DVector<f64>| x.iter().sum();

        let point = EvaluatedPoint::new(step.clone(), &mean, sigma, &mut function).unwrap();

        assert_eq!(point.step, DVector::from(vec![1.0; dim]));
        assert_eq!(point.point, DVector::from(vec![5.0; dim]));
        assert_eq!(point.value, 5.0 * dim as f64);

        let mut function_nan = |_: &DVector<f64>| f64::NAN;
        assert!(EvaluatedPoint::new(step, &mean, sigma, &mut function_nan).is_err());
    }

    #[test]
    fn test_sample() {
        let dim = 10;
        let population_size = 12;
        let mut sampler = Sampler::new(dim, population_size, Box::new(|_: &DVector<f64>| 0.0), 1);
        let state = State::new(vec![0.0; dim].into(), 2.0);

        let n = 5;
        for _ in 0..n {
            let individuals = sampler.sample(&state).unwrap();

            assert_eq!(individuals.len(), population_size);
        }

        assert_eq!(sampler.function_evals(), n * population_size);

        let mut sampler_nan = Sampler::new(
            dim,
            population_size,
            Box::new(|_: &DVector<f64>| f64::NAN),
            1,
        );

        assert!(sampler_nan.sample(&state).is_err());
    }
}
