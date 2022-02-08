//! A trait for types that can be used as an objective function. See [`ObjectiveFunction`] for full
//! documentation.

use nalgebra::DVector;

// TODO: docs
pub trait ObjectiveFunction {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64;
}

impl<F: FnMut(&DVector<f64>) -> f64> ObjectiveFunction for F {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        (self)(x)
    }
}
