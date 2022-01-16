//! An example of using CMA-ES to minimize the Rosenbrock function.

use cmaes::{CMAESOptions, Weights};
use nalgebra::DVector;

fn rosenbrock(point: &DVector<f64>) -> f64 {
    let a = 1.0;
    let b = 100.0;

    return (a - point[0]).powi(2) + b * (point[1] - point[0].powi(2)).powi(2);
}

fn main() {
    // Customize parameters for the problem
    let dim = 2;
    let mut cmaes_state = CMAESOptions::new(rosenbrock, dim)
        .weights(Weights::Positive)
        .tol_fun(1e-10)
        .tol_x(1e-10)
        .build()
        .unwrap();

    let max_generations = 20000;
    let solution = cmaes_state.run(max_generations);

    if let Some(data) = solution {
        println!(
            "Solution found with value {} at point {}\nTermination reason: {:?}",
            data.best_function_value, data.best_individual, data.reason,
        );
    } else {
        let current_best = cmaes_state.get_current_best_individual().unwrap();

        println!(
            "No solution found in {} generations, best point has value {} at point {}",
            max_generations, current_best.1, current_best.0,
        );
    }
}
