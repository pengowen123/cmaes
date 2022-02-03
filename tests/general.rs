//! General tests

use cmaes::{CMAESOptions, ObjectiveFunction, Weights};
use nalgebra::DVector;

use std::f64::consts::PI;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 100;
// Maximum generations per test
const MAX_GENERATIONS: usize = 5000;

fn run_test<F: ObjectiveFunction + Clone>(
    options: CMAESOptions<F>,
    max_avg_generations: usize,
    max_failures: usize,
) {
    let mut total_generations = 0;
    let mut highest_generations = 0;
    let mut failures = Vec::new();
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for _ in 0..TEST_REPETITIONS {
        let mut cmaes_state = options.clone().build().unwrap();

        let result = cmaes_state.run(MAX_GENERATIONS).expect("did not terminate");
        let generations = cmaes_state.generation();

        if !(result.best_function_value < options.tol_fun) {
            failures.push((result.reason, result.best_function_value));
        }

        *map.entry(result.reason).or_insert(0) += 1;

        total_generations += generations;

        if generations > highest_generations {
            highest_generations = generations;
        }
    }
    let avg_generations = (total_generations as f64 / TEST_REPETITIONS as f64) as usize;

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
    assert!(CMAESOptions::new(dummy_function, 5).build().is_ok());
    assert!(CMAESOptions::new(dummy_function, 5)
        .population_size(3)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 5)
        .initial_step_size(-1.0)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 5)
        .initial_mean(vec![1.0; 2])
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 0).build().is_err());
    assert!(CMAESOptions::new(dummy_function, 0)
        .cm(2.0)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 0)
        .cm(-1.0)
        .build()
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
        let max_avg_generations = 210 * d / 2;
        for n in 1..4 {
            let max_failures = 0;
            let mut options = CMAESOptions::new(function, d);
            options.population_size *= n;
            println!("pop size: {}", options.population_size);
            run_test(
                options.clone().weights(Weights::Negative),
                max_avg_generations,
                max_failures,
            );
            run_test(
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
        10.0 + (0..x.len())
            .map(|i| x[i].powi(2) - 10.0 * (2.0 * PI * x[i]).cos())
            .sum::<f64>()
    };

    for d in 2..5 {
        println!("{} dimensions:", d);
        let max_avg_generations = 300;
        for n in 1..4 {
            let max_failures = 8 / n;
            let mut options = CMAESOptions::new(function, d).initial_mean(vec![2.0; d]);
            options.population_size *= n;
            println!("pop size: {}", options.population_size);
            run_test(
                options.clone().weights(Weights::Negative),
                max_avg_generations,
                max_failures,
            );
            run_test(
                options.weights(Weights::Positive),
                max_avg_generations,
                max_failures,
            );
        }
    }
}

#[test]
fn test_eggholder() {
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
        let mut options = CMAESOptions::new(function, 2).initial_step_size(128.0).tol_x(1e-12);
        options.population_size *= n;
        println!("pop size: {}", options.population_size);
        run_test(
            options.clone().weights(Weights::Negative),
            max_avg_generations,
            max_failures,
        );
        run_test(
            options.weights(Weights::Positive),
            max_avg_generations,
            max_failures,
        );
    }
}
