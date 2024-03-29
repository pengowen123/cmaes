//! An example of using `cmaes` to minimize various simple functions.
//!
//! Requires the `plotters` feature.

#![allow(dead_code)]

use cmaes::objective_function::{ObjectiveFunction, Scale};
use cmaes::{CMAESOptions, DVector, PlotOptions, Weights};
use rand;

use std::f64::consts::PI;
use std::time::Duration;

fn main() {
    let function = rosenbrock;

    // Customize parameters for the problem
    let dim = 10;
    // The search space can be scaled in each dimension
    let mut function = Scale::new(function, vec![2.0; dim]);

    let mut cmaes_state = CMAESOptions::new(vec![0.1; dim], 0.1)
        .weights(Weights::Positive)
        .fun_target(1e-10)
        .tol_x(1e-13)
        .max_generations(10_000)
        .max_function_evals(100_000)
        .max_time(Duration::from_secs(5))
        // Enable recording the plot and printing info
        .enable_plot(PlotOptions::new(0, false))
        .enable_printing(200)
        .build(function.clone())
        .unwrap();

    // Find a solution
    let solution = cmaes_state.run();
    let overall_best = solution.overall_best.unwrap();

    println!(
        "Final mean has value {:e}",
        function.evaluate(cmaes_state.mean())
    );
    println!(
        "Solution individual has value {:e} and point {}",
        overall_best.value, overall_best.point,
    );

    // Save the plot
    let plot = cmaes_state.get_plot().unwrap();
    plot.save_to_file(
        format!("{}/test_output/plot.png", env!("CARGO_MANIFEST_DIR")),
        true,
    )
    .unwrap();
}

// A random function (completely independent of input)
fn random(_: &DVector<f64>) -> f64 {
    rand::random()
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
