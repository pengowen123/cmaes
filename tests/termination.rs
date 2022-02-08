//! Tests for certain termination criteria being reached

use cmaes::{CMAESOptions, ObjectiveFunction, TerminationReason};
use nalgebra::DVector;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 100;
// Maximum generations per test
const MAX_GENERATIONS: usize = 1000;

fn run_test<F: ObjectiveFunction + Clone, R: Fn(TerminationReason) -> bool>(
    options: CMAESOptions<F>,
    check_reason: R,
    max_mismatches: usize,
) {
    let mut mismatches = Vec::new();
    for _ in 0..TEST_REPETITIONS {
        let mut cmaes_state = options.clone().build().unwrap();

        let result = cmaes_state.run(MAX_GENERATIONS).expect("did not terminate");

        if !check_reason(result.reason) {
            mismatches.push(result.reason);
        }
    }
    if mismatches.len() > max_mismatches {
        panic!("exceeded {} mismatches: {:?}", max_mismatches, mismatches);
    }
}

#[test]
fn test_tol_fun() {
    // The function reaches `tol_fun` more quickly than the algorithm converges
    let function = |x: &DVector<f64>| x.magnitude().powi(2);
    run_test(
        CMAESOptions::new(function, 2).initial_mean(vec![5.0; 2]),
        |r| matches!(r, TerminationReason::TolFun),
        0,
    );
}

#[test]
fn test_tol_x() {
    // The algorithm converges more quickly than `tol_fun is reached`
    let function = |x: &DVector<f64>| x.magnitude().sqrt();
    run_test(
        CMAESOptions::new(function, 2).initial_mean(vec![5.0; 2]),
        |r| matches!(r, TerminationReason::TolX),
        0,
    );
}

#[test]
fn test_equal_fun_values() {
    // The function bottoms out before convergence
    let function = |x: &DVector<f64>| x.magnitude().max(1e-6);
    run_test(
        CMAESOptions::new(function, 2).initial_mean(vec![5.0; 2]),
        |r| matches!(r, TerminationReason::EqualFunValues),
        0,
    );
}

#[test]
fn test_stagnation() {
    // The function is noisy, so it will continue changing despite never improving overall
    let function = |x: &DVector<f64>| 1.0 + x.magnitude().powi(2) + rand::random::<f64>() * 1e-2;
    run_test(
        CMAESOptions::new(function, 2).initial_mean(vec![5.0; 2]),
        |r| matches!(r, TerminationReason::Stagnation),
        0,
    );
}

#[test]
fn test_tol_x_up() {
    // The initial step size is far too small
    let function = |x: &DVector<f64>| x[0].powi(2) + x[1].powi(2);
    run_test(
        CMAESOptions::new(function, 2)
            .initial_mean(vec![1e3; 2])
            .initial_step_size(1e-9),
        |r| matches!(r, TerminationReason::TolXUp),
        0,
    );
}

fn run_test_no_effect<F: Fn(TerminationReason) -> bool + Clone>(check_reason: F) {
    // Neither `tol_fun` nor `tol_x` can be reached
    let function =
        |x: &DVector<f64>| 1e-8 + (2.0 * x[0] - x[1]).abs().powf(1.5) + (2.0 - x[1]).powi(2);
    run_test(
        CMAESOptions::new(function, 2)
            .tol_x(1e-16)
            .initial_step_size(4.0),
        check_reason,
        1,
    );
}

#[test]
fn test_no_effect() {
    run_test_no_effect(|r| {
        matches!(
            r,
            TerminationReason::NoEffectAxis | TerminationReason::NoEffectCoord,
        )
    });
}

#[test]
#[should_panic(expected = "NoEffectCoord")]
fn test_no_effect_axis() {
    run_test_no_effect(|r| matches!(r, TerminationReason::NoEffectAxis));
}

#[test]
#[should_panic(expected = "NoEffectAxis")]
fn test_no_effect_coord() {
    run_test_no_effect(|r| matches!(r, TerminationReason::NoEffectCoord));
}

#[test]
fn test_condition_cov() {
    // The function diverges in one axis while converging in another, causing the distribution to
    // become extremely thin and long
    let function = |x: &DVector<f64>| 0.1 + x[0].abs().powi(2) - (x[1] * 1e-14).abs().sqrt();
    run_test(
        CMAESOptions::new(function, 2)
            .initial_step_size(1e3)
            .tol_x(1e-12),
        |r| matches!(r, TerminationReason::ConditionCov),
        1,
    );
}

#[test]
fn test_invalid_function_value() {
    // The function produces NAN
    let function = |x: &DVector<f64>| x[0].sqrt() + x[1].sqrt();
    run_test(
        CMAESOptions::new(function, 2).initial_mean(vec![5.0; 2]),
        |r| matches!(r, TerminationReason::InvalidFunctionValue),
        0,
    );
}
