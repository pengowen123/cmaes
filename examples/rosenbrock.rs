//! An example of using CMA-ES to minimize the Rosenbrock function.

use cmaes::{CMAESOptions, Weights, TerminationReason};
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

    let mut found_solution = false;

    loop {
        // Iterate the algorithm until a termination criterion has been met
        if let Some(data) = cmaes_state.next() {
            // A good solution was found
            if let TerminationReason::TolFun = data.reason {
                found_solution = true;
                println!("solution point: {}", data.best_individual);
                println!("solution value: {}", data.best_function_value);
            }

            break;
        }

        // Stop if no solution is found after 20000 generations
        if cmaes_state.generation() > 20000 {
            break;
        }
    }

    // If a solution wasn't found, get the current best point instead
    if !found_solution {
        let current_best = cmaes_state.get_current_best_individual().unwrap();

        println!("best solution point after 20k generations: {}", current_best.0);
        println!("best solution value after 20k generations: {}", current_best.1);
    }
}
