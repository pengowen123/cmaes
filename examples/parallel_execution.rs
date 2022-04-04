//! An example of using `cmaes` with parallel objective function execution.

use cmaes::{CMAESOptions, DVector};

use std::thread;
use std::time::Duration;

fn main() {
    let function = |x: &DVector<f64>| {
        // Expensive computations
        thread::sleep(Duration::from_millis(100));
        x.magnitude()
    };

    // Customize parameters for the problem
    let dim = 10;

    let mut cmaes_state = CMAESOptions::new(vec![0.1; dim], 0.1)
        .max_time(Duration::from_secs(5))
        .enable_printing(200)
        .build(&function)
        .unwrap();

    // Find a solution
    let use_threads = true;
    let solution = if use_threads {
        cmaes_state.run_parallel()
    } else {
        cmaes_state.run()
    };
    let overall_best = solution.overall_best.unwrap();

    println!(
        "Solution individual has value {:e} and point {}",
        overall_best.value, overall_best.point,
    );
}
