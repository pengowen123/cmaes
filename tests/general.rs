//! General tests

use cmaes::{CMAESOptions, ObjectiveFunction, Weights};
use nalgebra::DVector;

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
    use std::collections::HashMap;
    let mut reasons = HashMap::new();
    for _ in 0..TEST_REPETITIONS {
        let mut cmaes_state = options.clone().build(objective_function.clone()).unwrap();

        let result = cmaes_state.run(MAX_GENERATIONS).expect("did not terminate");
        let generations = cmaes_state.generation();

        if !(result.best_function_value < options.tol_fun) {
            failures.push((result.reason, result.best_function_value));
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
    assert!(CMAESOptions::new(0)
        .cm(2.0)
        .build(dummy_function)
        .is_err());
    assert!(CMAESOptions::new(0)
        .cm(-1.0)
        .build(dummy_function)
        .is_err());
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
            + (0..x.len())
                .map(|i| x[i].powi(2) - 10.0 * (2.0 * PI * x[i]).cos())
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
        let mut options = CMAESOptions::new(2)
            .initial_step_size(128.0)
            .tol_x(1e-12);
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
