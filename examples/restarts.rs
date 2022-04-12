//! An example of using automatic restarts with `cmaes` to optimize multimodal functions.

#![allow(dead_code)]
#![allow(unused_variables)]

#[allow(unused_imports)]
use cmaes::restart::{Local, RestartOptions, RestartStrategy};
use cmaes::DVector;

use std::f64::consts::PI;
use std::ops::RangeInclusive;

fn main() {
    // Customize parameters for the problem
    let rastrigin = (rastrigin, -5.0..=5.0, 30);
    let weierstrass = (weierstrass, -0.5..=0.5, 30);
    let eggholder = (eggholder, -512.0..=512.0, 2);

    let (function, search_range, dim) = weierstrass;

    // Different strategies are useful for different problems, although BIPOP and IPOP are usually
    // better than LR
    // let strategy = RestartStrategy::IPOP(Default::default());
    // let strategy = RestartStrategy::Local(Local::new(60, None).unwrap());
    let strategy = RestartStrategy::BIPOP(Default::default());

    let restarter = RestartOptions::new(dim, search_range, strategy)
        .fun_target(1e-10)
        .max_time(std::time::Duration::from_secs(120))
        .enable_printing(true)
        .build()
        .unwrap();

    // Find a solution
    let results = restarter.run_parallel(|| function);
}

// N-dimensional Rastrigin function
//
// Search range: [-5, 5]^N
/// Global minimum: 0 at [0]^N
fn rastrigin(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

/// N-dimensional Weierstrass function
///
/// Search range: [-0.5, 0.5]^N
/// Global minimum: 0 at [0]^N
fn weierstrass(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    let k_max = 20;
    let a = 0.5f64;
    let b = 3.0f64;
    let y = x
        .iter()
        .map(|xi| {
            (0..=k_max)
                .map(|k| a.powi(k) * (2.0 * PI * b.powi(k) * (xi + 0.5)).cos())
                .sum::<f64>()
        })
        .sum::<f64>()
        - x.len() as f64
            * (0..=k_max)
                .map(|k| a.powi(k) * (2.0 * PI * b.powi(k) * 0.5).cos())
                .sum::<f64>();
    // Apply bounds with penalty
    y + penalty(x, -0.5..=0.5, 1e6)
}

// Eggholder function
//
// Search range: [-512.0, 512.0]^2
/// Global minimum: ~0 at ~[512, 404.23]
fn eggholder(x: &DVector<f64>) -> f64 {
    assert!(x.len() == 2);
    959.640636
        - (x[1] + 47.0) * (x[1] + x[0] / 2.0 + 47.0).abs().sqrt().sin()
        - x[0] * (x[0] - (x[1] + 47.0)).abs().sqrt().sin()
        // Apply bounds with penalty
        + penalty(x, -512.0..=512.0, 1e6)
}

// Calculates a quadratic penalty for a point if it lies outside the bounds in any dimension
fn penalty(x: &DVector<f64>, bounds: RangeInclusive<f64>, penalty_factor: f64) -> f64 {
    penalty_factor
        * x.iter()
            .map(|xi| (xi - bounds.end()).max(0.0).powi(2) + (bounds.start() - xi).max(0.0).powi(2))
            .sum::<f64>()
}
