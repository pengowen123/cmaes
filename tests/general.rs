//! General tests

use assert_approx_eq::assert_approx_eq;
use cmaes::{CMAESOptions, ObjectiveFunction, Weights, CMAES};
use nalgebra::DVector;

use std::collections::HashMap;
use std::f64::consts::PI;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 30;
// Maximum generations per test
const MAX_GENERATIONS: usize = 5000;

fn run_test<F: ObjectiveFunction + Clone + 'static>(
    objective_function: F,
    options: CMAESOptions,
    max_avg_evals: usize,
    max_failures: usize,
) {
    let mut total_evals = 0;
    let mut highest_evals = 0;
    let mut failures = Vec::new();
    let mut reasons = HashMap::new();

    for _ in 0..TEST_REPETITIONS {
        let mut cmaes_state = options
            .clone()
            .max_generations(MAX_GENERATIONS)
            .build(objective_function.clone())
            .unwrap();

        let result = cmaes_state.run();
        let overall_best = result.overall_best.unwrap();
        let evals = cmaes_state.function_evals();

        for reason in &result.reasons {
            *reasons.entry(*reason).or_insert(0) += 1;
        }

        if !(overall_best.value < options.tol_fun) {
            failures.push((result.reasons, overall_best.value));
        }

        total_evals += evals;

        if evals > highest_evals {
            highest_evals = evals;
        }
    }
    let avg_evals = (total_evals as f64 / TEST_REPETITIONS as f64) as usize;

    println!("avg evals: {}", avg_evals);
    println!("failures: {:?}", failures);
    println!("reasons: {:?}", reasons);
    assert!(
        failures.len() <= max_failures,
        "max failures exceeded: {}",
        failures.len(),
    );
    assert!(avg_evals < max_avg_evals);
    assert!(highest_evals < max_avg_evals * 2);
}

#[test]
fn test_sphere() {
    let run_test_sphere = |dim, pop_size_mult, max_avg_evals, max_failures, weights| {
        let mut options = CMAESOptions::new(vec![0.1; dim], 0.1).weights(weights);
        options.population_size *= pop_size_mult;

        run_test(sphere, options, max_avg_evals, max_failures);
    };

    run_test_sphere(3, 1, 700, 0, Weights::Negative);
    run_test_sphere(10, 1, 2140, 0, Weights::Negative);
    run_test_sphere(10, 10, 10000, 0, Weights::Negative);
    run_test_sphere(10, 10, 10200, 0, Weights::Positive);
    run_test_sphere(30, 1, 5730, 0, Weights::Negative);
}

#[test]
fn test_ellipsoid() {
    let run_test_ellipsoid = |dim, pop_size_mult, max_avg_evals, max_failures, weights| {
        let mut options = CMAESOptions::new(vec![0.1; dim], 0.1).weights(weights);
        options.population_size *= pop_size_mult;

        run_test(ellipsoid, options, max_avg_evals, max_failures);
    };

    run_test_ellipsoid(3, 1, 940, 0, Weights::Negative);
    run_test_ellipsoid(10, 1, 4500, 0, Weights::Negative);
    run_test_ellipsoid(10, 10, 13610, 0, Weights::Negative);
    run_test_ellipsoid(10, 10, 14350, 0, Weights::Positive);
    run_test_ellipsoid(30, 1, 29000, 0, Weights::Negative);
}

fn run_test_rosenbrock(
    dim: usize,
    pop_size_mult: usize,
    max_avg_evals: usize,
    max_failures: usize,
    weights: Weights,
) {
    let mut options = CMAESOptions::new(vec![0.1; dim], 0.1).weights(weights);
    options.population_size *= pop_size_mult;

    run_test(rosenbrock, options, max_avg_evals, max_failures);
}

#[test]
fn test_rosenbrock() {
    run_test_rosenbrock(3, 1, 1320, 0, Weights::Negative);
    run_test_rosenbrock(10, 1, 6060, 1, Weights::Negative);
    // Finds local minimum sometimes with larger population size
    run_test_rosenbrock(10, 10, 21600, TEST_REPETITIONS / 3, Weights::Negative);
    run_test_rosenbrock(10, 10, 22500, TEST_REPETITIONS / 2, Weights::Positive);
    run_test_rosenbrock(30, 10, 132000, 1, Weights::Negative);
}

#[test]
fn test_rastrigin() {
    let run_test_rastrigin = |dim, pop_size_mult, max_avg_evals, max_failures, weights| {
        let mut options = CMAESOptions::new(vec![5.0; dim], 5.0).weights(weights);
        options.population_size *= pop_size_mult;

        run_test(rastrigin, options, max_avg_evals, max_failures);
    };

    run_test_rastrigin(3, 10, 4410, TEST_REPETITIONS / 2, Weights::Negative);
    run_test_rastrigin(10, 20, 30500, TEST_REPETITIONS * 3 / 5, Weights::Negative);
    run_test_rastrigin(10, 20, 30500, TEST_REPETITIONS * 3 / 4, Weights::Positive);
}

#[test]
fn test_cigar() {
    let run_test_cigar = |dim, pop_size_mult, max_avg_evals, max_failures, weights| {
        let mut options = CMAESOptions::new(vec![0.1; dim], 0.1).weights(weights);
        options.population_size *= pop_size_mult;

        run_test(cigar, options.clone(), max_avg_evals, max_failures);
    };

    run_test_cigar(3, 1, 1210, 0, Weights::Negative);
    run_test_cigar(10, 1, 4810, 0, Weights::Negative);
    run_test_cigar(10, 10, 15800, 0, Weights::Negative);
    run_test_cigar(10, 10, 17000, 0, Weights::Positive);
    run_test_cigar(30, 1, 14500, 1, Weights::Negative);
}

// Must be updated after every change to the algorithm (after thorough testing)
#[test]
fn test_fixed_seed() {
    let function = rosenbrock;
    let seed = 76561199230847669;
    let dim = 4;
    let population_size = 12;
    let mut cmaes_state = CMAESOptions::new(vec![0.0; dim], 5.0)
        .population_size(population_size)
        .seed(seed)
        .build(function)
        .unwrap();

    let params = cmaes_state.parameters();

    let eps = 1e-12;
    assert_eq!(params.dim(), dim);
    assert_eq!(params.lambda(), population_size);
    assert_eq!(params.mu(), 6);
    assert_approx_eq!(params.initial_sigma(), 5.0, eps);
    assert_approx_eq!(params.mu_eff(), 3.729458934303068, eps);
    let weights_expected = [
        0.4024029428187127,
        0.25338908403288657,
        0.16622156455542053,
        0.10437522524706053,
        0.0564034775763251,
        0.017207705769594468,
        -0.05531493617379553,
        -0.1549841113726818,
        -0.24289855445276476,
        -0.321540703596474,
        -0.3926811809212799,
        -0.45762734831325463,
    ];
    for (w, expected) in params.weights().iter().zip(weights_expected) {
        assert_approx_eq!(w, expected, eps);
    }

    assert_approx_eq!(params.cc(), 0.5, eps);
    assert_approx_eq!(params.c1(), 0.06285462000247571, eps);
    assert_approx_eq!(params.cs(), 0.4500944591496695, eps);
    assert_approx_eq!(params.cmu(), 0.10055985647786812, eps);
    assert_approx_eq!(params.cm(), 1.0, eps);
    assert_approx_eq!(params.damp_s(), 1.4500944591496694, eps);
    assert_approx_eq!(params.fun_target(), 0.000000000001, eps);
    assert_approx_eq!(params.tol_fun(), 0.000000000001, eps);
    assert_approx_eq!(params.tol_fun_rel(), 0.0, eps);
    assert_approx_eq!(params.tol_fun_hist(), 0.000000000001, eps);
    assert_approx_eq!(params.tol_x(), 0.000000000005, eps);
    assert_eq!(params.tol_stagnation(), 167);
    assert_approx_eq!(params.tol_x_up(), 1e8, eps);
    assert_approx_eq!(params.tol_condition_cov(), 1e14, eps);
    assert_eq!(params.seed(), seed);

    let generations = 10;
    for _ in 0..generations {
        let _ = cmaes_state.next();
    }

    assert_eq!(cmaes_state.generation(), generations);
    assert_eq!(cmaes_state.function_evals(), population_size * generations);

    let mean_expected = [
        -0.7410175385751399,
        0.4584000754445906,
        0.07271743391612415,
        -0.21426659534075343,
    ];
    for (x, expected) in cmaes_state.mean().iter().zip(mean_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    let eigenvalues_expected = [
        0.16552675641022155,
        0.20638999168067776,
        0.245971498782978,
        0.7533179588300154,
    ];
    for (x, expected) in cmaes_state.eigenvalues().iter().zip(eigenvalues_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(cmaes_state.axis_ratio(), 2.1333153488330385, eps);
    assert_approx_eq!(cmaes_state.sigma(), 1.248640716345183, eps);

    let current_best = cmaes_state.current_best_individual().unwrap();
    let current_best_expected = [
        -1.13647778742836,
        0.6467966530116105,
        0.3239978388983541,
        -0.41993207745293704,
    ];
    for (x, expected) in current_best.point.iter().zip(current_best_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(current_best.value, 75.16391027290686, eps);

    let overall_best = cmaes_state.overall_best_individual().unwrap();
    let overall_best_expected = [
        -0.7449503227893762,
        0.1848895283976989,
        -0.2416606558228348,
        -0.19268839630032886,
    ];
    for (x, expected) in overall_best.point.iter().zip(overall_best_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(overall_best.value, 32.859092832300654, eps);
}

/// For tests with consistent results
mod consistent {
    use super::*;

    // Checks that certain usage patterns work
    #[test]
    fn test_api_usage() {
        // Non-static objective function (references something or is a reference)
        let mut x = 0.0;
        let mut non_static_function = |_: &DVector<f64>| {
            x += 1.0;
            x
        };

        let _ = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(&mut non_static_function)
            .unwrap();
        let non_static_state = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(non_static_function)
            .unwrap();

        // Storing a CMAES with a static objective function without dealing with lifetimes
        struct StaticContainer(CMAES<'static>);

        let static_function = |x: &DVector<f64>| x.magnitude();

        let static_state = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(static_function)
            .unwrap();
        StaticContainer(static_state);

        // Storing a CMAES with any lifetime
        struct NonStaticContainer<'a>(CMAES<'a>);
        NonStaticContainer(non_static_state);

        let static_state = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(static_function)
            .unwrap();
        NonStaticContainer(static_state);
    }

    #[test]
    fn test_rosenbrock_small_dim() {
        run_test_rosenbrock(3, 1, 1320, 0, Weights::Negative);
        run_test_rosenbrock(3, 1, 1600, 0, Weights::Positive);
    }
}

// N-dimensional sphere function
fn sphere(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 1);
    x.iter().map(|xi| xi.powi(2)).sum::<f64>()
}

// N-dimensional ellipsoid function
fn ellipsoid(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 1);
    (0..x.len())
        .map(|i| 1e6f64.powf(i as f64 / x.len() as f64) * x[i].powi(2))
        .sum::<f64>()
}

// N-dimensional Rosenbrock function
fn rosenbrock(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
        .sum::<f64>()
}

// N-dimensional Rastrigin function
fn rastrigin(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

// N-dimensional cigar function
fn cigar(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 1);
    x[0].powi(2) + 1e6 * x.iter().skip(1).map(|xi| xi.powi(2)).sum::<f64>()
}
