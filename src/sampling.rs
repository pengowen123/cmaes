//! Types related to sampling points from the distribution

use nalgebra::DVector;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use statrs::distribution::Normal;

use crate::mode::Mode;
use crate::state::State;
use crate::{ObjectiveFunction, ParallelObjectiveFunction};

pub trait Constraints: Sync + std::fmt::Debug {
    fn meets_constraints(&self, x: &DVector<f64>) -> bool;
    fn clone_box(&self) -> Box<dyn Constraints>;
}

impl Clone for Box<dyn Constraints> {
    fn clone(&self) -> Box<dyn Constraints> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl Constraints for Bounds {
    fn meets_constraints(&self, x: &DVector<f64>) -> bool {
        (0..x.len()).all(|i| x[i] >= self.lower[i] && x[i] <= self.upper[i])
    }

    fn clone_box(&self) -> Box<dyn Constraints> {
        Box::new(self.clone())
    }
}

/// A type for sampling and evaluating points from the distribution for each generation
pub struct Sampler<F> {
    /// Number of dimensions to sample from
    dim: usize,
    /// If set, resamples until all points satisfy the constraints
    constraints: Option<Box<dyn Constraints>>,
    /// The maximum number of resamples.
    /// If this limit is hit, uses points even if they violate the constraints
    max_resamples: Option<usize>,
    /// Number of points to sample each generation
    population_size: usize,
    /// RNG from which all random numbers are sourced
    rng: ChaCha12Rng,
    /// The objective function to optimize, used to evaluate points
    objective_function: F,
    /// The number of times the objective function has been evaluated
    function_evals: usize,
}

impl<F> Sampler<F> {
    pub fn new(
        dim: usize,
        constraints: Option<Box<dyn Constraints>>,
        max_resamples: Option<usize>,
        population_size: usize,
        objective_function: F,
        rng_seed: u64,
    ) -> Self {
        Self {
            dim,
            constraints,
            max_resamples,
            population_size,
            rng: ChaCha12Rng::seed_from_u64(rng_seed),
            objective_function,
            function_evals: 0,
        }
    }

    /// Shared logic between `sample` and `sample_parallel`
    fn sample_internal<
        P: Fn(Vec<DVector<f64>>, &mut F) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError>,
    >(
        &mut self,
        state: &State,
        mode: Mode,
        parallel_update: bool,
        evaluate_points: P,
    ) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Random steps in the distribution N(0, I)
        let mut sample = |n: usize, constraints: Option<&dyn Constraints>| {
            let z = (0..n)
                .map(|_| {
                    DVector::from_iterator(
                        self.dim,
                        (0..self.dim).map(|_| normal.sample(&mut self.rng)),
                    )
                })
                .collect::<Vec<_>>();
            let transform = |zk| state.cov_transform() * zk;

            let ok_constraints = |yk: &DVector<f64>| match constraints {
                Some(constraints) => {
                    constraints.meets_constraints(&to_point(&yk, state.mean(), state.sigma()))
                }
                None => true,
            };

            if parallel_update {
                z.into_par_iter()
                    .map(transform)
                    .filter(ok_constraints)
                    .collect()
            } else {
                z.into_iter()
                    .map(transform)
                    .filter(ok_constraints)
                    .collect()
            }
        };

        let mut y: Vec<DVector<f64>> = vec![];

        let mut i: usize = 0;
        loop {
            let remain = self.population_size - y.len();
            if remain == 0 {
                break;
            }

            let mut constraints = self.constraints.as_ref().map(|x| x.as_ref());
            if let Some(max) = self.max_resamples {
                if i >= max {
                    constraints = None;
                }
            }

            let mut new_samps: Vec<_> = sample(remain, constraints);
            y.append(&mut new_samps);

            i += 1;
        }

        let mut points = evaluate_points(y, &mut self.objective_function)?;

        self.function_evals += points.len();

        points.sort_by(|a, b| mode.sort_cmp(a.value, b.value));
        Ok(points)
    }

    pub fn function_evals(&self) -> usize {
        self.function_evals
    }

    /// Consumes `self` and returns the objective function
    pub fn into_objective_function(self) -> F {
        self.objective_function
    }
}

impl<F: ObjectiveFunction> Sampler<F> {
    /// Samples and returns a new generation of points, sorted in ascending order by their
    /// corresponding objective function values
    ///
    /// Returns Err if the objective function returned an invalid value
    pub fn sample(
        &mut self,
        state: &State,
        mode: Mode,
        parallel_update: bool,
    ) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        self.sample_internal(state, mode, parallel_update, |y, objective_function| {
            y.into_iter()
                .map(|yk| {
                    EvaluatedPoint::new(yk, state.mean(), state.sigma(), |x| {
                        objective_function.evaluate(x)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

impl<F: ParallelObjectiveFunction> Sampler<F> {
    /// Like `sample`, but evaluates the sampled points using multiple threads
    pub fn sample_parallel(
        &mut self,
        state: &State,
        mode: Mode,
        parallel_update: bool,
    ) -> Result<Vec<EvaluatedPoint>, InvalidFunctionValueError> {
        self.sample_internal(state, mode, parallel_update, |y, objective_function| {
            y.into_par_iter()
                .map(|yk| {
                    EvaluatedPoint::new(yk, state.mean(), state.sigma(), |x| {
                        objective_function.evaluate_parallel(x)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

/// A point from the distribution that has been evaluated by the objective function
#[derive(Clone, Debug)]
pub struct EvaluatedPoint {
    /// The evaluated point
    point: DVector<f64>,
    /// The step from the mean of the point before scaling by sigma
    /// In the distribution N(0, cov)
    unscaled_step: DVector<f64>,
    /// The objective value at the point
    value: f64,
}

fn to_point(unscaled_step: &DVector<f64>, mean: &DVector<f64>, sigma: f64) -> DVector<f64> {
    mean + sigma * unscaled_step
}

impl EvaluatedPoint {
    /// Returns a new `EvaluatedPoint` from the unscaled step from the mean, the mean, and the step
    /// size
    ///
    /// Returns `Err` if the objective function returned an invalid value
    pub fn new<F: FnMut(&DVector<f64>) -> f64>(
        unscaled_step: DVector<f64>,
        mean: &DVector<f64>,
        sigma: f64,
        mut objective_function: F,
    ) -> Result<Self, InvalidFunctionValueError> {
        let point = to_point(&unscaled_step, &mean, sigma);
        let value = objective_function(&point);

        if value.is_nan() {
            Err(InvalidFunctionValueError)
        } else {
            Ok(Self {
                point,
                unscaled_step,
                value,
            })
        }
    }

    pub fn point(&self) -> &DVector<f64> {
        &self.point
    }

    pub fn unscaled_step(&self) -> &DVector<f64> {
        &self.unscaled_step
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

        assert_eq!(point.unscaled_step, DVector::from(vec![1.0; dim]));
        assert_eq!(point.point, DVector::from(vec![5.0; dim]));
        assert_eq!(point.value, 5.0 * dim as f64);

        let mut function_nan = |_: &DVector<f64>| f64::NAN;
        assert!(EvaluatedPoint::new(step, &mean, sigma, &mut function_nan).is_err());
    }

    #[test]
    fn test_sample() {
        let dim = 10;
        let population_size = 12;
        let mut sampler = Sampler::new(
            dim,
            None,
            None,
            population_size,
            Box::new(|_: &DVector<f64>| 0.0),
            1,
        );
        let state = State::new(vec![0.0; dim].into(), 2.0);

        let n = 5;
        for _ in 0..n {
            let individuals = sampler.sample(&state, Mode::Minimize, false).unwrap();

            assert_eq!(individuals.len(), population_size);
        }

        assert_eq!(sampler.function_evals(), n * population_size);

        let mut sampler_nan = Sampler::new(
            dim,
            None,
            None,
            population_size,
            Box::new(|_: &DVector<f64>| f64::NAN),
            1,
        );

        assert!(sampler_nan.sample(&state, Mode::Minimize, false).is_err());
    }

    #[test]
    fn test_resample() {
        let dim = 1;
        let population_size = 1;

        let bounds = Bounds {
            lower: vec![0.0],
            upper: vec![1.0],
        };

        let objective_function = |_: &DVector<f64>| 0.0;

        // No resampling: Value should be out-of-bounds
        {
            let mut sampler = Sampler::new(
                dim,
                Some(Box::new(bounds.clone())),
                Some(0),
                population_size,
                objective_function,
                1,
            );
            let state = State::new(vec![0.0; dim].into(), 2.0);
            let individuals = sampler.sample(&state, Mode::Minimize, false).unwrap();

            assert!(
                individuals[0].point[0] < bounds.lower[0]
                    || individuals[0].point[0] > bounds.upper[0]
            );
        }

        // With limited resampling: Value should be in bounds
        {
            let mut sampler = Sampler::new(
                dim,
                Some(Box::new(bounds.clone())),
                Some(10),
                population_size,
                objective_function,
                1,
            );
            let state = State::new(vec![0.0; dim].into(), 2.0);
            let individuals = sampler.sample(&state, Mode::Minimize, false).unwrap();

            assert!(individuals[0].point[0] >= bounds.lower[0]);
            assert!(individuals[0].point[0] <= bounds.upper[0]);
        }

        // With unlimited resampling: Value should be in bounds
        {
            let mut sampler = Sampler::new(
                dim,
                Some(Box::new(bounds.clone())),
                None,
                population_size,
                objective_function,
                1,
            );
            let state = State::new(vec![0.0; dim].into(), 2.0);
            let individuals = sampler.sample(&state, Mode::Minimize, false).unwrap();

            assert!(individuals[0].point[0] >= bounds.lower[0]);
            assert!(individuals[0].point[0] <= bounds.upper[0]);
        }
    }

    fn sample_sort(mode: Mode, expected: [f64; 5]) {
        let mut counter = 0.0;
        let function = |_: &DVector<f64>| {
            match mode {
                // Ensure the individuals are unsorted initially by making the function values
                // improve for later calls
                Mode::Minimize => counter -= 1.0,
                Mode::Maximize => counter += 1.0,
            }
            counter
        };

        let dim = 10;
        let population_size = expected.len();

        let mut sampler = Sampler::new(dim, None, None, population_size, function, 1);
        let state = State::new(vec![0.0; dim].into(), 2.0);

        let individuals = sampler.sample(&state, mode, false).unwrap();
        let values = individuals
            .into_iter()
            .map(|ind| ind.value)
            .collect::<Vec<_>>();

        assert_eq!(expected, values.as_slice());
    }

    #[test]
    fn test_sample_sort_minimize() {
        sample_sort(Mode::Minimize, [-5.0, -4.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn test_sample_sort_maximize() {
        sample_sort(Mode::Maximize, [5.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
