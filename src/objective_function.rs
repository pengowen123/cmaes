//! A trait for types that can be used as an objective function. See [`ObjectiveFunction`] for full
//! documentation.

use nalgebra::DVector;

/// A trait for types that can be used as an objective function. This trait is implemented for
/// functions and closures with the correct signature, so the following can all be used in a
/// [`CMAESState`][crate::CMAESState]:
///
/// ```
/// use cmaes::DVector;
///
/// fn foo(x: &DVector<f64>) -> f64 { x.magnitude()  }
/// let bar = |x: &DVector<f64>| x.magnitude();
/// ```
///
/// Objective functions do not necessarily have to be `'static`, so they may borrow other values:
///
/// ```
/// # use cmaes::{CMAESOptions, DVector};
/// let mut n = 0.0;
/// let baz = |_: &DVector<f64>| {
///     n += 1.0;
///     n
/// };
///
/// let mut state = CMAESOptions::new(2).build(baz).unwrap();
/// ```
///
/// Or be references themselves:
///
/// ```
/// # use cmaes::{CMAESOptions, DVector};
/// # let mut n = 0.0;
/// # let mut baz = |_: &DVector<f64>| {
/// #     n += 1.0;
/// #     n
/// # };
/// let mut state = CMAESOptions::new(2).build(&mut baz).unwrap();
/// ```
///
/// The trait may also be implemented for custom types:
///
/// ```
/// use cmaes::{CMAESOptions, DVector, ObjectiveFunction};
///
/// struct Custom {
///     counter: f64,
/// }
///
/// impl ObjectiveFunction for Custom {
///     fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
///         self.counter += 1.0;
///         x.magnitude() + self.counter
///     }
/// }
///
/// impl<'a> ObjectiveFunction for &'a mut Custom {
///     fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
///         Custom::evaluate(*self, x)
///     }
/// }
///
/// fn main() {
///     // The objective function's state can be retrieved after optimization
///     let mut custom = Custom { counter: 0.0 };
///
///     {
///         let mut cmaes_state = CMAESOptions::new(2).build(&mut custom).unwrap();
///         cmaes_state.run(20000);
///     }
///
///     println!("{}", custom.counter);
/// }
/// ```
pub trait ObjectiveFunction {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64;
}

impl<F: FnMut(&DVector<f64>) -> f64> ObjectiveFunction for F {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        (self)(x)
    }
}
