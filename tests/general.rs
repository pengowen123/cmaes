//! General tests

use assert_approx_eq::assert_approx_eq;
use cmaes::{CMAESOptions, Mode, ObjectiveFunction, TerminationReason, Weights, CMAES};
use nalgebra::DVector;

use std::collections::HashMap;
use std::f64::consts::PI;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 30;
// Maximum generations per test
const MAX_GENERATIONS: usize = 5000;

// Checks that FunTarget is reached within the specified average function evals per run and number
// of failures to reach fun_target
fn run_test<F: ObjectiveFunction + Clone + 'static>(
    objective_function: F,
    // Negates the objective function value and fun_target if this is Mode::Maximize
    mode: Mode,
    options: CMAESOptions,
    max_avg_evals: usize,
    max_failures: usize,
) {
    let mut total_evals = 0;
    let mut highest_evals = 0;
    let mut failures = Vec::new();
    let mut reasons = HashMap::new();

    for _ in 0..TEST_REPETITIONS {
        let modified_function: Box<dyn ObjectiveFunction>;

        let mut options = options.clone().max_generations(MAX_GENERATIONS).mode(mode);

        match mode {
            Mode::Maximize => {
                let mut objective_function = objective_function.clone();
                modified_function =
                    Box::new(move |x: &DVector<f64>| -objective_function.evaluate(x));
                options.fun_target = Some(-1e-12);
            }
            Mode::Minimize => {
                modified_function = Box::new(objective_function.clone());
                options.fun_target = Some(1e-12)
            }
        }

        let mut cmaes_state = options.build(modified_function).unwrap();

        let result = cmaes_state.run();
        let overall_best = result.overall_best.unwrap();
        let evals = cmaes_state.function_evals();

        for reason in &result.reasons {
            *reasons.entry(*reason).or_insert(0) += 1;
        }

        // Check that the target function value was reached
        if !result
            .reasons
            .iter()
            .any(|r| *r == TerminationReason::FunTarget)
        {
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

        run_test(sphere, Mode::Minimize, options, max_avg_evals, max_failures);
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

        run_test(
            ellipsoid,
            Mode::Minimize,
            options,
            max_avg_evals,
            max_failures,
        );
    };

    run_test_ellipsoid(3, 1, 940, 0, Weights::Negative);
    run_test_ellipsoid(10, 1, 4500, 0, Weights::Negative);
    run_test_ellipsoid(10, 10, 13610, 0, Weights::Negative);
    run_test_ellipsoid(10, 10, 14350, 0, Weights::Positive);
    run_test_ellipsoid(30, 1, 29000, 0, Weights::Negative);
}

fn run_test_rosenbrock(
    mode: Mode,
    dim: usize,
    pop_size_mult: usize,
    max_avg_evals: usize,
    max_failures: usize,
    weights: Weights,
) {
    let mut options = CMAESOptions::new(vec![0.1; dim], 0.1).weights(weights);
    options.population_size *= pop_size_mult;

    run_test(rosenbrock, mode, options, max_avg_evals, max_failures);
}

fn rosenbrock_shared(mode: Mode) {
    run_test_rosenbrock(mode, 3, 1, 1320, 0, Weights::Negative);
    run_test_rosenbrock(mode, 10, 1, 6060, 1, Weights::Negative);
    // Finds local optima sometimes with larger population size
    run_test_rosenbrock(mode, 10, 10, 21600, TEST_REPETITIONS / 3, Weights::Negative);
    run_test_rosenbrock(mode, 10, 10, 22500, TEST_REPETITIONS / 2, Weights::Positive);
    run_test_rosenbrock(mode, 30, 10, 132000, 1, Weights::Negative);
}

#[test]
fn test_rosenbrock_minimize() {
    rosenbrock_shared(Mode::Minimize);
}

#[test]
fn test_rosenbrock_maximize() {
    rosenbrock_shared(Mode::Maximize);
}

#[test]
fn test_rastrigin() {
    let run_test_rastrigin = |dim, pop_size_mult, max_avg_evals, max_failures, weights| {
        let mut options = CMAESOptions::new(vec![5.0; dim], 5.0).weights(weights);
        options.population_size *= pop_size_mult;

        run_test(
            rastrigin,
            Mode::Minimize,
            options,
            max_avg_evals,
            max_failures,
        );
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

        run_test(
            cigar,
            Mode::Minimize,
            options.clone(),
            max_avg_evals,
            max_failures,
        );
    };

    run_test_cigar(3, 1, 1210, 0, Weights::Negative);
    run_test_cigar(10, 1, 4810, 0, Weights::Negative);
    run_test_cigar(10, 10, 15800, 0, Weights::Negative);
    run_test_cigar(10, 10, 17000, 0, Weights::Positive);
    run_test_cigar(30, 1, 14500, 1, Weights::Negative);
}

/// For tests with consistent results
mod consistent {
    use super::*;

    // Must be updated after every change to the algorithm (after thorough testing)
    fn fixed_seed(mode: Mode, parallel_function: bool, parallel_update: bool) {
        let function = match mode {
            Mode::Minimize => rosenbrock,
            // Flip the function so maximizing it makes sense
            Mode::Maximize => (|x| -rosenbrock(x)) as fn(&DVector<f64>) -> f64,
        };
        let seed = 26001788261131793;
        let dim = 4;
        let mut cmaes_state = CMAESOptions::new(vec![0.0; dim], 5.0)
            .mode(mode)
            .seed(seed)
            .parallel_update(parallel_update)
            .build(function)
            .unwrap();

        let params = cmaes_state.parameters();
        let lambda = params.lambda();

        let eps = 1e-12;
        assert_eq!(params.dim(), dim);
        assert_eq!(lambda, 8);
        assert_eq!(params.mu(), 4);
        assert_approx_eq!(params.initial_sigma(), 5.0, eps);
        assert_approx_eq!(params.mu_eff(), 2.6001788261131793, eps);
        let weights_expected = [
            0.5299301844787792,
            0.2857142857142857,
            0.14285714285714282,
            0.041498386949792215,
            -0.17013144983238102,
            -0.4645361478293982,
            -0.7134517732694924,
            -0.9290722956587965,
        ];
        for (w, expected) in params.weights().iter().zip(weights_expected) {
            assert_approx_eq!(w, expected, eps);
        }

        assert_approx_eq!(params.cc(), 0.5, eps);
        assert_approx_eq!(params.c1(), 0.06516742738228268, eps);
        assert_approx_eq!(params.cs(), 0.39656102677983807, eps);
        assert_approx_eq!(params.cmu(), 0.05102399983259446, eps);
        assert_approx_eq!(params.cm(), 1.0, eps);
        assert_approx_eq!(params.damp_s(), 1.396561026779838, eps);
        assert!(params.fun_target().is_none());
        assert_approx_eq!(params.tol_fun(), 0.000000000001, eps);
        assert_approx_eq!(params.tol_fun_rel(), 0.0, eps);
        assert_approx_eq!(params.tol_fun_hist(), 0.000000000001, eps);
        assert_approx_eq!(params.tol_x(), 0.000000000005, eps);
        assert_eq!(params.tol_stagnation(), 200);
        assert_approx_eq!(params.tol_x_up(), 1e8, eps);
        assert_approx_eq!(params.tol_condition_cov(), 1e14, eps);
        assert_eq!(params.seed(), seed);

        let generations = 10;
        for _ in 0..generations {
            let _ = if parallel_function {
                cmaes_state.next_parallel()
            } else {
                cmaes_state.next()
            };
        }

        assert_eq!(cmaes_state.generation(), generations);
        assert_eq!(cmaes_state.function_evals(), lambda * generations);

        let mean_expected = [
            0.12571237293855164,
            0.2280643277648201,
            0.17627386121151933,
            0.6616196164462642,
        ];
        for (x, expected) in cmaes_state.mean().iter().zip(mean_expected) {
            assert_approx_eq!(x, expected, eps);
        }

        let eigenvalues_expected = [
            1.3068929025500882,
            0.1358452979163828,
            0.6455285582368571,
            0.4569850734591793,
        ];
        for (x, expected) in cmaes_state.eigenvalues().iter().zip(eigenvalues_expected) {
            assert_approx_eq!(x, expected, eps);
        }

        assert_approx_eq!(cmaes_state.axis_ratio(), 3.1016850332870485, eps);
        assert_approx_eq!(cmaes_state.sigma(), 1.6845266959237821, eps);

        let current_best = cmaes_state.current_best_individual().unwrap();
        let current_best_expected = [
            0.14583793998288663,
            0.6158716510808776,
            0.24480184155861673,
            0.33848727834833875,
        ];
        for (x, expected) in current_best.point.iter().zip(current_best_expected) {
            assert_approx_eq!(x, expected, eps);
        }

        let expected_sign = match mode {
            Mode::Minimize => 1.0,
            Mode::Maximize => -1.0,
        };
        assert_approx_eq!(current_best.value, 46.37118717981843 * expected_sign, eps);

        let overall_best = cmaes_state.overall_best_individual().unwrap();
        let overall_best_expected = [
            0.14583793998288724,
            0.6158716510808762,
            0.24480184155861373,
            0.3384872783483405,
        ];
        for (x, expected) in overall_best.point.iter().zip(overall_best_expected) {
            assert_approx_eq!(x, expected, eps);
        }

        assert_approx_eq!(overall_best.value, 46.371187179818456 * expected_sign, eps);
    }

    #[test]
    fn test_fixed_seed_minimize() {
        fixed_seed(Mode::Minimize, false, false);
    }

    #[test]
    fn test_fixed_seed_maximize() {
        fixed_seed(Mode::Maximize, false, false);
    }

    // Check that parallel objective function execution doesn't affect the results
    #[test]
    fn test_fixed_seed_minimize_parallel_objective_function() {
        fixed_seed(Mode::Minimize, true, false);
    }

    // Check that parallel state updates don't affect the results too much (they do after some
    // number of iterations, but the results shouldn't differ much at first)
    #[test]
    fn test_fixed_seed_minimize_parallel_update() {
        fixed_seed(Mode::Minimize, true, true);
    }

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
        let _ = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(non_static_function)
            .unwrap();

        // Storing a CMAES with a static objective function without dealing with type parameters
        struct StaticContainer(CMAES<Box<dyn ObjectiveFunction>>);

        let static_function = |x: &DVector<f64>| x.magnitude();

        let static_cmaes = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(Box::new(static_function) as _)
            .unwrap();
        StaticContainer(static_cmaes);

        // Storing a CMAES with any lifetime
        struct NonStaticContainer<F: ObjectiveFunction>(CMAES<F>);

        let mut x = 0.0;
        let mut non_static_function = |_: &DVector<f64>| {
            x += 1.0;
            x
        };
        let non_static_cmaes = CMAESOptions::new(vec![0.0; 5], 1.0)
            .build(&mut non_static_function)
            .unwrap();
        NonStaticContainer(non_static_cmaes);
    }

    #[test]
    fn test_rosenbrock_small_dim() {
        run_test_rosenbrock(Mode::Minimize, 3, 1, 1320, 0, Weights::Negative);
        run_test_rosenbrock(Mode::Minimize, 3, 1, 1600, 0, Weights::Positive);

        run_test_rosenbrock(Mode::Maximize, 3, 1, 1320, 0, Weights::Negative);
        run_test_rosenbrock(Mode::Maximize, 3, 1, 1600, 0, Weights::Positive);
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
