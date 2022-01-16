//! Tests for the termination criteria

use cmaes::{CMAESOptions, TerminationReason};
use nalgebra::DVector;

#[test]
fn test_tol_fun() {
    let function = |x: &DVector<f64>| x.magnitude();
    let mut cmaes_state = CMAESOptions::new(function, 5)
        .initial_mean(vec![5.0; 5])
        .tol_fun(1e-8)
        .build()
        .unwrap();

    let result = cmaes_state.run(250).expect("no solution found");

    assert_eq!(result.reason, TerminationReason::TolFun);
}

#[test]
fn test_equal_fun_values() {
    let function = |x: &DVector<f64>| x.magnitude() + 2.0;
    let mut cmaes_state = CMAESOptions::new(function, 5)
        .tol_fun(1e-8)
        .build()
        .unwrap();

    let result = cmaes_state.run(250).expect("no solution found");

    assert_eq!(result.reason, TerminationReason::EqualFunValues);
}

// TODO: more of these
