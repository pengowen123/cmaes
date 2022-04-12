//! An example of using the single-line function interface of `cmaes`.

use cmaes::DVector;

fn main() {
    let dim = 10;
    let _solution = cmaes::fmin(rosenbrock, vec![0.1; dim], 0.5);
}

// N-dimensional Rosenbrock function
fn rosenbrock(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
        .sum::<f64>()
}
