//! An example of using `cmaes` with a custom objective function type.

use cmaes::{CMAESOptions, DVector, ObjectiveFunction};

// Custom objective function types can be used to store parameters and state
struct Rosenbrock {
    a: f64,
    b: f64,
    counter: f64,
}

impl Rosenbrock {
    fn new(a: f64, b: f64) -> Self {
        Self { a, b, counter: 0.0 }
    }
}

impl ObjectiveFunction for Rosenbrock {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        assert!(x.len() == 2);
        let y = (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2);
        // Track the sum of values outputted by the function
        self.counter += y;
        y
    }
}

// `ObjectiveFunction` must be implemented for references separately
impl<'a> ObjectiveFunction for &'a mut Rosenbrock {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        ObjectiveFunction::evaluate(*self, x)
    }
}

fn main() {
    // Initialize the objective function
    let mut function = Rosenbrock::new(5.0, 30.0);

    // A scope must be used to prevent overlapping borrows
    {
        // Customize parameters for the problem
        let dim = 2;
        let mut cmaes_state = CMAESOptions::new(vec![0.1; dim], 0.1)
            .enable_printing(200)
            // Use a mutable reference to the function so its state can be retrieved easily later
            .build(&mut function)
            .unwrap();

        // Find a solution
        let solution = cmaes_state.run();
        let overall_best = solution.overall_best.unwrap();

        println!(
            "Solution individual has value {:e} and point {}",
            overall_best.value, overall_best.point,
        );
    }

    // Retrieve the state stored in the objective function
    println!(
        "Sum of objective function values outputted: {}",
        function.counter
    );
}
