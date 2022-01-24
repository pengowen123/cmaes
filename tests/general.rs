//! General tests

use cmaes::CMAESOptions;
use nalgebra::DVector;

fn dummy_function(_: &DVector<f64>) -> f64 {
    0.0
}

#[test]
fn test_build() {
    assert!(CMAESOptions::new(dummy_function, 5).build().is_ok());
    assert!(CMAESOptions::new(dummy_function, 5)
        .population_size(3)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 5)
        .initial_step_size(-1.0)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 5)
        .initial_mean(vec![1.0; 2])
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 0).build().is_err());
    assert!(CMAESOptions::new(dummy_function, 0)
        .cm(2.0)
        .build()
        .is_err());
    assert!(CMAESOptions::new(dummy_function, 0)
        .cm(-1.0)
        .build()
        .is_err());
}

#[test]
fn test_optimization() {
    // TODO: some test problems
}
