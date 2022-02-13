//! An example of using CMA-ES to minimize the Rosenbrock function.

use cmaes::{CMAESOptions, PlotOptions, Weights};
use nalgebra::DVector;

fn rosenbrock(x: &DVector<f64>) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
        .sum::<f64>()
}

fn main() {
    // Customize parameters for the problem
    let dim = 10;
    let mut cmaes_state = CMAESOptions::new(dim)
        .weights(Weights::Positive)
        .tol_fun(1e-13)
        .tol_x(1e-13)
        .initial_step_size(0.1)
        .initial_mean(vec![0.1; dim])
        .enable_plot(PlotOptions::new(0, false))
        .build(rosenbrock)
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
        let current_best = cmaes_state.current_best_individual().unwrap();

        println!(
            "Did not terminate after {} generations\nBest point has value {:e} and coordinates {:?}",
            cmaes_state.generation(), current_best.1, current_best.0.as_slice(),
        );
    }

    // Save the plot
    let plot = cmaes_state.get_plot().unwrap();
    plot.save_to_file("plot.png").unwrap();
}
