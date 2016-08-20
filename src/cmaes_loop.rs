extern crate rand;
extern crate la;

use std::thread;
use std::sync::Arc;
use std::cmp::Ordering;

use la::{Matrix, EigenDecomposition};
use rand::distributions::{Normal, IndependentSample};

use super::utils::Parameters;
use super::fitness::FitnessFunction;
use super::vector::*;
use super::options::{CMAESOptions, CMAESEndConditions};

const BAD_FITNESS: f64 = 10e10;

pub fn cmaes_loop<T>(object: &T, options: CMAESOptions) -> Option<(Vec<f64>, f64)>
    where T: 'static + FitnessFunction + Clone + Send + Sync
{
    //! Minimizes a function. Takes as an argument a type that implements the
    //! FitnessFunction trait and an instance of the CMAESOptions struct.
    //! Returns a solution with as small a fitness as possible, along with the fitness value of the
    //! solution.
    //!
    //! # Panics
    //!
    //! Panics if the fitness function panics or returns NaN or infinite, or if threads is 0. There
    //! are some checks to make sure invalid values do not show up anywhere. If the function
    //! returns None, please open an issue on the repository and include reproduction steps.

    // Take options out of the struct
    let conditions = options.end_conditions;
    let d = options.dimension;
    let threads = options.threads;
    let deviations = options.initial_standard_deviations;
    let mean = options.initial_mean;

    if deviations.len() != d {
        panic!("Length of initial deviation vector must be equal to the number of dimensions");
    }

    if mean.len() != d {
        panic!("Length of initial mean vector must be equal to the number of dimensions");
    }

    // Various numbers; mutable variables are only used as a starting point and
    // are adapted by the algorithm
    let n = d as f64;
    let sample_size = 4.0 + (3.0 * n.ln()).floor();
    let parents = (sample_size / 2.0).floor() as usize;
    let sample_size = sample_size as usize;

    let mut generation: Vec<Parameters> = Vec::new();
    let mut covariance_matrix: Matrix<f64> = Matrix::diag(deviations);
    let mut eigenvectors: Matrix<f64> = Matrix::id(d, d);
    let mut eigenvalues: Matrix<f64> = Matrix::vector(vec![1.0; d]);
    let mut mean_vector = mean;
    let mut step_size = options.initial_step_size;
    let mut path_s: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut path_c: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut inv_sqrt_cov: Matrix<f64> = Matrix::id(d, d);
    let mut g = 0;
    let mut eigeneval = 0;
    let mut hs;

    let weights = (0..parents)
                      .map(|i| (parents as f64 + 1.0 / 2.0).ln() - ((i as f64 + 1.0).ln()))
                      .collect::<Vec<f64>>();

    let sum = sum_vec(&weights);
    let weights = weights.iter().map(|i| i / sum).collect::<Vec<f64>>();
    let variance_eff = sum_vec(&weights) / sum_vec(&mul_vec_2(&weights, &weights));
    let cc = (4.0 + variance_eff / n as f64) / (n + 4.0 + 2.0 * variance_eff / n);
    let cs = (variance_eff + 2.0) / (n + variance_eff + 5.0);
    let c1 = 2.0 / ((n + 1.3).powi(2) + variance_eff);
    let cmu = {
        let a = 1.0 - c1;
        let b = 2.0 * (variance_eff - 2.0 + 1.0 / variance_eff) /
                ((n + 2.0).powi(2) + variance_eff);
        if a <= b {
            a
        } else {
            b
        }
    };
    let damps = 1.0 + cs +
                2.0 *
                {
        let result = ((variance_eff - 1.0) / (n + 1.0)).sqrt() - 1.0;
        if 0.0 > result {
            0.0
        } else {
            result
        }
    };
    let expectation = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n.powi(2)));

    // Thread stuff
    let mut per_thread = vec![0; sample_size as usize];

    let mut t = 0;
    for _ in 0..sample_size {
        per_thread[t] += 1;
        t = if threads as usize > t {
            0
        } else {
            t + 1
        };
    }

    // Normal distribution
    let dist = Normal::new(0.0, 1.0);

    // End condition variables
    let mut stable = 0;
    let mut best = 0.0;
    let mut end = false;

    loop {
        // Protect against invalid values
        if step_size.is_nan() || step_size.is_infinite() {
            break;
        }

        let mut invalid_values = false;

        for n in eigenvalues.get_data() {
            if n.is_nan() || n.is_infinite() {
                invalid_values = true;
            }
        }

        for n in eigenvectors.get_data() {
            if n.is_nan() || n.is_infinite() {
                invalid_values = true;
            }
        }

        if invalid_values {
            break;
        }

        // More thread stuff
        generation = Vec::new();

        let object = Arc::new(object.clone());
        let vectors = Arc::new(eigenvectors.clone());
        let values = Arc::new(eigenvalues.clone());
        let mean = Arc::new(mean_vector.clone());

        let do_work = Arc::new(move |thread_values: Arc<Matrix<f64>>,
                            thread_vectors: Arc<Matrix<f64>>,
                            thread_mean: Arc<Vec<f64>>,
                            thread_object: Arc<T>,
                            t: usize| {
            let mut individuals = Vec::new();

            for _ in 0..t as usize {
                let random_values;

                random_values = vec![dist.ind_sample(&mut rand::thread_rng()); d];

                // Sample multivariate normal
                let parameters = mul_vec_2(&*thread_values.get_data(), &random_values);
                let parameters = matrix_by_vector(&*thread_vectors, &parameters);
                let parameters = add_vec(&*thread_mean, &mul_vec(&parameters, step_size));

                // Get fitness of parameters
                let mut individual = Parameters::new(&parameters);
                individual.fitness = thread_object.get_fitness(&parameters);

                // Protect from invalid values
                if individual.fitness.is_nan() || individual.fitness.is_infinite() {
                    individual.fitness = BAD_FITNESS;
                }

                individuals.push(individual);
            }

            individuals
        });

        // Create new individuals
        
        if threads > 0 {
            for t in per_thread.clone() {
                let thread_object = object.clone();
                let thread_mean = mean.clone();
                let thread_vectors = vectors.clone();
                let thread_values = values.clone();
                let do_work = do_work.clone();

                let handle = thread::spawn(move || {
                    do_work(thread_values, thread_vectors, thread_mean, thread_object, t)
                });

                // User-defined function might panic
                let individuals = match handle.join() {
                    Ok(v) => v,
                    Err(..) => {
                        println!("Warning: Early return due to panic");
                        return match generation.first() {
                            Some(i) => Some((i.parameters.clone(), i.fitness)),
                            None => return None
                        }
                    }
                };

                for item in individuals {
                    generation.push(item);
                }
            }
        } else {
            generation = do_work(values, vectors, mean, object, sample_size);
        }

        // Increment function evaluations counter
        g += sample_size as usize;

        // Detect bad things
        let mut bad_things = false;

        // Sort generation by fitness; smallest fitness will be first
        generation.sort_by(|a, b| match a.fitness.partial_cmp(&b.fitness) {
            Some(v) => v,
            None => {
                bad_things = true;
                Ordering::Equal
            }
        });

        if bad_things {
            println!("Warning: Early return due to non-normal parameter values");
            return Some((generation[0].parameters.clone(), generation[0].fitness));
        }

        // Update mean vector
        // New mean vector is the average of the parents
        let mut mean = vec![0.0; d];
        for (i, parent) in generation[0..parents as usize].iter().enumerate() {
            mean = add_vec(&mean, &mul_vec(&parent.parameters, weights[i])).to_vec();
        }

        let mean_vector_old = mean_vector.clone();
        mean_vector = mean;

        // To prevent duplicate code
        let diff = sub_vec(&mean_vector, &mean_vector_old).to_vec();

        // Update the evolution path for the step size (sigma)
        // Measures how many steps the step size has taken in the same direction
        let a_ = mul_vec(path_s.get_data(), 1.0 - cs);
        let b_ = (cs * (2.0 - cs) * variance_eff).sqrt();
        let c_ = inv_sqrt_cov.scale(b_);
        let e_ = (&c_ * Matrix::vector(diff.clone())).scale(1.0 / step_size);
        let f_ = add_vec(&a_, e_.get_data());
        path_s = Matrix::vector(f_.to_vec());

        // hs determines whether to do an additional update to the covariance matrix
        let a_ = magnitude(path_s.get_data());
        let b_ = (1.0 - (1.0 - cs).powf(2.0 * g as f64 / sample_size as f64)).sqrt();
        hs = (a_ / b_ / expectation < 1.4 + 2.0 / (n + 1.0)) as i32 as f64;

        // Update the evolution path for the covariance matrix (capital sigma)
        // Measures how many steps the step size has taken in the same direction
        let a_ = mul_vec(path_c.get_data(), 1.0 - cc);
        let b_ = hs * (cc * (2.0 - cc) * variance_eff).sqrt();
        let d_ = mul_vec(&diff, b_);
        let e_ = div_vec(&d_, step_size);
        let f_ = add_vec(&a_, &e_);
        path_c = Matrix::vector(f_.to_vec());

        // Factor in the values of the individuals
        let mut artmp = transpose(&Matrix::new(parents as usize,
                                               d,
                                               concat(generation[0..parents as usize]
                                                          .iter()
                                                          .map(|p| p.parameters.clone())
                                                          .collect())));

        let artmp2 = transpose(&Matrix::new(parents as usize,
                                            d,
                                            concat(vec![mean_vector_old; parents])));

        artmp = (artmp - artmp2).scale(1.0 / step_size);

        // Update the covariance matrix
        // Determines the shape of the search area
        let a_ = covariance_matrix.scale(1.0 - c1 - cmu);
        let b_ = (&path_c * transpose(&path_c) +
                  covariance_matrix.scale(((1.0 - hs) * cc * (2.0 - cc))))
                     .scale(c1);
        let c_ = &artmp.scale(cmu) * Matrix::diag(weights.clone()) * transpose(&artmp);
        covariance_matrix = a_ + b_ + c_;

        // Update the step size
        // Determines the size of the search area
        // Increased if the length of its evolution path is greater than the expectation of N(0, I)
        step_size = step_size *
                    ((cs / damps) * (magnitude(path_s.get_data()) / expectation - 1.0)).exp();

        // Update the eigenvectors and eigenvalues every so often
        if (g - eigeneval) as f64 > sample_size as f64 / (c1 + cmu) / n / 10.0 {
            eigeneval = g;

            for y in 0..d {
                for x in y + 1..d {
                    let cell = covariance_matrix.get(y, x);
                    covariance_matrix.set(x, y, cell);
                }
            }

            let e = EigenDecomposition::new(&covariance_matrix);
            eigenvectors = e.get_v().clone();

            let mut new = Vec::new();

            {
                let data = eigenvectors.get_data();

                for i in 0..d {
                    let i = i * d;
                    new.extend_from_slice(&reverse(&data[i..i + d]));
                }
            }

            eigenvectors = Matrix::new(d, d, new);

            eigenvalues = e.get_d().clone();

            eigenvalues = Matrix::vector(reverse(&(0..d)
                                                      .map(|i| eigenvalues[(i, i)].sqrt())
                                                      .collect::<Vec<f64>>())
                                             .to_vec());

            let inverse = Matrix::diag(eigenvalues.get_data()
                                                  .iter()
                                                  .map(|n| n.powi(-1))
                                                  .collect::<Vec<f64>>());

            inv_sqrt_cov = &eigenvectors * inverse * transpose(&eigenvectors);
        }

        // Test the end conditions
        for condition in &conditions {
            match *condition {
                CMAESEndConditions::StableGenerations(t, g_) => {
                    stable += ((best - generation[0].fitness).abs() < t) as usize;
                    end = stable >= g_;
                }

                CMAESEndConditions::FitnessThreshold(f) => {
                    end = generation[0].fitness <= f;
                }

                CMAESEndConditions::MaxGenerations(g_) => {
                    end = g / sample_size >= g_;
                }

                CMAESEndConditions::MaxEvaluations(e) => {
                    end = g >= e;
                }
            }
        }

        if end {
            break;
        }

        best = generation[0].fitness;
    }

    Some((generation[0].parameters.to_vec(), best))
}
