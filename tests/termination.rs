//! Tests for the termination criteria

use cmaes::{CMAESOptions, TerminationReason};
use nalgebra::DVector;

#[test]
fn test_tol_fun() {
    let function = |x: &DVector<f64>| x.magnitude().powi(2);
    let mut cmaes_state = CMAESOptions::new(function, 2)
        .initial_mean(vec![5.0; 2])
        .build()
        .unwrap();

    let result = cmaes_state.run(250).expect("no solution found");

    assert_eq!(result.reason, TerminationReason::TolFun);
}

#[test]
fn test_equal_fun_values() {
    // The function bottoms out before convergence
    let function = |x: &DVector<f64>| x.magnitude().max(1e-8);
    let mut cmaes_state = CMAESOptions::new(function, 5).build().unwrap();

    let result = cmaes_state.run(1000).expect("no solution found");

    assert_eq!(result.reason, TerminationReason::EqualFunValues);
}

// TODO: more of these
