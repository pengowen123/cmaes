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
        .tol_fun(1e-13)
        .tol_x(1e-13)
        .build()
        .unwrap();

    // Find a solution
    let max_generations = 20000;
    let solution = cmaes_state.run(max_generations);

    if let Some(data) = solution {
        println!(
            "Terminated after {} generations with termination reason `{:?}`",
            cmaes_state.generation(),
            data.reason,
        );
        println!(
            "Best point has value {:e} and coordinates {:?}",
            data.best_function_value,
            data.best_individual.as_slice(),
        );
    } else {
        let current_best = cmaes_state.get_current_best_individual().unwrap();

        println!(
            "Did not terminate after {} generations\nBest point has value {:e} and coordinates {:?}",
            cmaes_state.generation(), current_best.1, current_best.0.as_slice(),
        );
    }
}
