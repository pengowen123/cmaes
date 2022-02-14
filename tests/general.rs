//! General tests

use assert_approx_eq::assert_approx_eq;
use cmaes::{CMAESOptions, ObjectiveFunction, Weights};
use nalgebra::DVector;

use std::collections::HashMap;
use std::f64::consts::PI;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 100;
// Maximum generations per test
const MAX_GENERATIONS: usize = 5000;

fn run_test<F: ObjectiveFunction + Clone + 'static>(
    objective_function: F,
    options: CMAESOptions,
    max_avg_generations: usize,
    max_failures: usize,
) {
    let mut total_generations = 0;
    let mut highest_generations = 0;
    let mut failures = Vec::new();
    let mut reasons = HashMap::new();

    for _ in 0..TEST_REPETITIONS {
        let mut cmaes_state = options.clone().build(objective_function.clone()).unwrap();

        let result = cmaes_state.run(MAX_GENERATIONS).expect("did not terminate");
        let generations = cmaes_state.generation();

        if !(result.overall_best.value < options.tol_fun) {
            failures.push((result.reason, result.overall_best.value));
        }

        *reasons.entry(result.reason).or_insert(0) += 1;

        total_generations += generations;

        if generations > highest_generations {
            highest_generations = generations;
        }
    }
    let avg_generations = (total_generations as f64 / TEST_REPETITIONS as f64) as usize;

    println!("avg generations: {}", avg_generations);
    println!("failures: {:?}", failures);
    println!("reasons: {:?}", reasons);
    assert!(
        failures.len() <= max_failures,
        "max failures exceeded: {:?}",
        failures,
    );
    assert!(avg_generations < max_avg_generations);
    assert!(highest_generations < max_avg_generations * 3);
}

#[test]
fn test_build() {
    let dummy_function = |_: &DVector<f64>| 0.0;
    assert!(CMAESOptions::new(5).build(dummy_function).is_ok());
    assert!(CMAESOptions::new(5)
        .population_size(3)
        .build(dummy_function)
        .is_err());
    assert!(CMAESOptions::new(5)
        .initial_step_size(-1.0)
        .build(dummy_function)
        .is_err());
    assert!(CMAESOptions::new(5)
        .initial_mean(vec![1.0; 2])
        .build(dummy_function)
        .is_err());
    assert!(CMAESOptions::new(0).build(dummy_function).is_err());
    assert!(CMAESOptions::new(0).cm(2.0).build(dummy_function).is_err());
    assert!(CMAESOptions::new(0).cm(-1.0).build(dummy_function).is_err());
}

#[test]
fn test_rosenbrock() {
    let function = |x: &DVector<f64>| {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    };

    for d in 2..5 {
        println!("{} dimensions:", d);
        let max_avg_generations = 1000;
        let max_failures = TEST_REPETITIONS / 50;
        for n in 2..4 {
            let mut options = CMAESOptions::new(d);
            options.population_size *= n;
            println!("pop size: {}", options.population_size);
            run_test(
                function,
                options.clone().weights(Weights::Negative),
                max_avg_generations,
                max_failures,
            );
            run_test(
                function,
                options.weights(Weights::Positive),
                max_avg_generations,
                max_failures,
            );
        }
    }
}

#[test]
fn test_rastrigin() {
    let function = |x: &DVector<f64>| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos())
                .sum::<f64>()
    };

    for d in 2..5 {
        println!("{} dimensions:", d);
        let max_avg_generations = 300;
        let max_failures = TEST_REPETITIONS / 5;
        let options = CMAESOptions::new(d)
            .initial_mean(vec![5.0; d])
            .initial_step_size(1.0)
            .population_size(200);
        run_test(
            function,
            options.clone().weights(Weights::Negative),
            max_avg_generations,
            max_failures,
        );
        run_test(
            function,
            options.weights(Weights::Positive),
            max_avg_generations,
            max_failures,
        );
    }
}

#[test]
fn test_eggholder() {
    return;
    let function = |x: &DVector<f64>| {
        let fitness = 962.102316 + -(x[1] + 47.0) * (0.5 * x[0] + x[1] + 47.0).abs().sqrt().sin()
            - x[0] * (x[0] - (x[1] + 47.0)).abs().sqrt().sin();
        // Add boundary
        let dx = (x[0] - 512.0).max(-512.0 - x[0]).max(0.0);
        let dy = (x[1] - 512.0).max(-512.0 - x[1]).max(0.0);
        fitness + (dx + dy).powi(2)
    };

    let max_avg_generations = 300;
    for n in 1..4 {
        let max_failures = 1;
        let mut options = CMAESOptions::new(2).initial_step_size(128.0).tol_x(1e-12);
        options.population_size *= n;
        println!("pop size: {}", options.population_size);
        run_test(
            function,
            options.clone().weights(Weights::Negative),
            max_avg_generations,
            max_failures,
        );
        run_test(
            function,
            options.weights(Weights::Positive),
            max_avg_generations,
            max_failures,
        );
    }
}

// Must be updated after every change to the algorithm (after thorough testing)
#[test]
fn test_fixed_seed() {
    let function = |x: &DVector<f64>| {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    };

    let seed = 76561199230847669;
    let dimension = 4;
    let population_size = 12;
    let mut cmaes_state = CMAESOptions::new(dimension)
        .population_size(population_size)
        .seed(seed)
        .build(function)
        .unwrap();

    let params = cmaes_state.parameters();

    let eps = 1e-15;
    assert_eq!(params.dim(), dimension);
    assert_eq!(params.lambda(), population_size);
    assert_eq!(params.mu(), 6);
    assert_approx_eq!(params.initial_sigma(), 0.5, eps);
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
    assert_approx_eq!(params.tol_fun(), 0.000000000001, eps);
    assert_approx_eq!(params.tol_x(), 0.0000000000005, eps);
    assert_eq!(params.seed(), seed);

    let generations = 10;
    for _ in 0..generations {
        let _ = cmaes_state.next();
    }

    assert_eq!(cmaes_state.generation(), generations);
    assert_eq!(cmaes_state.function_evals(), population_size * generations);

    let mean_expected = [
        0.3905013183630414,
        0.143349143153743,
        -0.02861857391815352,
        0.030451572191736394,
    ];
    for (x, expected) in cmaes_state.mean().iter().zip(mean_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    let eigenvalues_expected = [
        0.21631966578264544,
        0.24993048179002433,
        0.2659429224395665,
        0.7871437989294525,
    ];
    for (x, expected) in cmaes_state.eigenvalues().iter().zip(eigenvalues_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(cmaes_state.axis_ratio(), 1.907563648004273, eps);
    assert_approx_eq!(cmaes_state.sigma(), 0.1708927201378601, eps);

    let current_best = cmaes_state.current_best_individual().unwrap();
    let current_best_expected = [
        0.45043390884929013,
        0.12148408546978581,
        -0.024154735384168395,
        -0.050013317101592646,
    ];
    for (x, expected) in current_best.point.iter().zip(current_best_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(current_best.value, 3.1928361882670977, eps);

    let overall_best = cmaes_state.overall_best_individual().unwrap();
    let overall_best_expected = [
        0.4001605505779719,
        0.16306101682314394,
        -0.006695725176343273,
        0.01598022024773967,
    ];
    for (x, expected) in overall_best.point.iter().zip(overall_best_expected) {
        assert_approx_eq!(x, expected, eps);
    }

    assert_approx_eq!(overall_best.value, 2.210750747950352, eps);
}
