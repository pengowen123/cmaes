//! An example of using `cmaes` to minimize various simple functions.

#![allow(dead_code)]

use cmaes::{CMAESOptions, PlotOptions, Weights, DVector};
use rand;

use std::f64::consts::PI;

fn main() {
    let function = rosenbrock;

    // Customize parameters for the problem
    let dim = 10;
    let mut cmaes_state = CMAESOptions::new(dim)
        .weights(Weights::Positive)
        .tol_fun(1e-6)
        .tol_x(1e-13)
        .initial_step_size(0.1)
        .initial_mean(vec![0.1; dim])
        // Enable recording the plot and printing info
        .enable_plot(PlotOptions::new(0, false))
        .enable_printing(200)
        .build(function)
        .unwrap();

    // Find a solution
    let max_generations = 20000;
    let solution = cmaes_state.run(max_generations);

    println!("Final mean has value {:e}", function(cmaes_state.mean()));
    if let Some(s) = solution {
        println!(
            "Solution individual has value {:e} and point {}",
            s.overall_best.value, s.overall_best.point,
        );
    }

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
