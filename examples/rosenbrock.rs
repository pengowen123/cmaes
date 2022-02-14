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
        .enable_printing(200)
        .build(rosenbrock)
        .unwrap();

    // Find a solution
    let max_generations = 20000;
    let solution = cmaes_state.run(max_generations);

    println!("Final mean has value {:e}", rosenbrock(cmaes_state.mean()));
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
