//! Tests for certain termination criteria being reached

use cmaes::{CMAESOptions, Mode, ObjectiveFunction, TerminationReason};
use nalgebra::DVector;

use std::thread;
use std::time::Duration;

// Number of times to repeat each test
// Necessary to account for the inherent randomness of the algorithm
const TEST_REPETITIONS: usize = 100;
// Maximum generations per test
const MAX_GENERATIONS: usize = 1000;

fn run_test<F: ObjectiveFunction + Clone + 'static, R: Fn(TerminationReason) -> bool>(
    objective_function: F,
    options: CMAESOptions,
    check_reason: R,
    max_mismatches: usize,
) {
    let mut mismatches = Vec::new();
    for _ in 0..TEST_REPETITIONS {
        let mut options = options.clone();
        options.max_generations = options.max_generations.or(Some(MAX_GENERATIONS));
        let mut cmaes_state = options.build(objective_function.clone()).unwrap();

        let result = cmaes_state.run();

        for reason in result.reasons {
            if !check_reason(reason) {
                mismatches.push(reason);
            }
        }
    }
    if mismatches.len() > max_mismatches {
        panic!("exceeded {} mismatches: {:?}", max_mismatches, mismatches);
    }
}

#[test]
fn test_max_function_evals() {
    // `max_function_evals` is reached more quickly than the algorithm converges
    let function = |x: &DVector<f64>| x.magnitude().powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).max_function_evals(100),
        |r| matches!(r, TerminationReason::MaxFunctionEvals),
        0,
    );
}

#[test]
fn test_max_generations() {
    // `max_generations` is reached more quickly than the algorithm converges
    let function = |x: &DVector<f64>| x.magnitude().powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).max_generations(20),
        |r| matches!(r, TerminationReason::MaxGenerations),
        0,
    );
}

#[test]
fn test_max_time() {
    // `max_time` is reached more quickly than the algorithm converges
    let function = |x: &DVector<f64>| {
        thread::sleep(Duration::from_millis(1));
        x.magnitude().powi(2)
    };
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).max_time(Duration::from_millis(1)),
        |r| matches!(r, TerminationReason::MaxTime),
        0,
    );
}

#[test]
fn test_fun_target() {
    // The function reaches `fun_target` more quickly than the algorithm converges
    fn function(x: &DVector<f64>) -> f64 {
        x.magnitude().powi(2)
    }
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).fun_target(1e-12),
        |r| matches!(r, TerminationReason::FunTarget),
        0,
    );

    // Test Mode::Maximize with a flipped function and fun_target
    run_test(
        (|x| -function(x)) as fn(&DVector<f64>) -> _,
        CMAESOptions::new(vec![5.0; 2], 1.0)
            .mode(Mode::Maximize)
            .fun_target(-1e-12),
        |r| matches!(r, TerminationReason::FunTarget),
        0,
    );
}

#[test]
fn test_tol_fun() {
    // The function reaches `tol_fun` more quickly than the algorithm converges
    let function = |x: &DVector<f64>| 1.0 + x.magnitude().powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).tol_fun_hist(0.0),
        |r| matches!(r, TerminationReason::TolFun),
        0,
    );
}

#[test]
fn test_tol_fun_rel() {
    // The function reaches `tol_fun_rel` more quickly than `tol_fun` due to the large improvement
    // in function value
    let function = |x: &DVector<f64>| 1.0 + x.magnitude().powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![1e6; 2], 1.0).tol_fun_rel(1e-12),
        |r| matches!(r, TerminationReason::TolFunRel),
        0,
    );
}

#[test]
fn test_tol_x() {
    // The algorithm converges more quickly than `tol_fun is reached`
    let function = |x: &DVector<f64>| x.magnitude().sqrt();
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0),
        |r| matches!(r, TerminationReason::TolX),
        0,
    );
}

#[test]
fn test_tol_fun_hist() {
    // The function bottoms out before convergence
    let function = |x: &DVector<f64>| x.magnitude().max(1e-6);
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).tol_fun(0.0),
        |r| matches!(r, TerminationReason::TolFunHist),
        0,
    );
}

#[test]
fn test_tol_stagnation() {
    // The function is noisy, so it will get worse occasionally
    fn function(x: &DVector<f64>) -> f64 {
        1.0 + x.magnitude() + rand::random::<f64>() * 1e1
    }
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0).tol_stagnation(20),
        |r| matches!(r, TerminationReason::TolStagnation),
        0,
    );

    // Test Mode::Maximize with a flipped function
    run_test(
        (|x| -function(x)) as fn(&DVector<f64>) -> _,
        CMAESOptions::new(vec![5.0; 2], 1.0)
            .mode(Mode::Maximize)
            .tol_stagnation(20),
        |r| matches!(r, TerminationReason::TolStagnation),
        0,
    );
}

#[test]
fn test_tol_x_up() {
    // The initial step size is far too small
    let function = |x: &DVector<f64>| x[0].powi(2) + x[1].powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![1e3; 2], 1e-9),
        |r| matches!(r, TerminationReason::TolXUp),
        0,
    );
}

fn run_test_no_effect<F: Fn(TerminationReason) -> bool + Clone>(check_reason: F) {
    // Neither `tol_fun` nor `tol_x` can be reached
    let function =
        |x: &DVector<f64>| 1e-8 + (2.0 * x[0] - x[1]).abs().powf(1.5) + (2.0 - x[1]).powi(2);
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 4.0)
            .tol_fun(0.0)
            .tol_fun_hist(0.0)
            .tol_x(1e-16),
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
fn test_tol_condition_cov() {
    // The function diverges in one axis while converging in another, causing the distribution to
    // become extremely thin and long
    let function = |x: &DVector<f64>| 0.1 + x[0].abs().powi(2) - (x[1] * 1e-14).abs().sqrt();
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1e3).tol_x(1e-12),
        |r| matches!(r, TerminationReason::TolConditionCov),
        1,
    );
}

#[test]
fn test_invalid_function_value() {
    // The function produces NAN
    let function = |x: &DVector<f64>| x[0].sqrt() + x[1].sqrt();
    run_test(
        function,
        CMAESOptions::new(vec![5.0; 2], 1.0),
        |r| matches!(r, TerminationReason::InvalidFunctionValue),
        0,
    );
}
