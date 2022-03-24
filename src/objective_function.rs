//! A trait for types that can be used as an objective function, as well as wrapper types that
//! modify the behavior of objective functions.

use nalgebra::DVector;

/// A trait for types that can be used as an objective function. This trait is implemented for
/// functions and closures with the correct signature, so the following can all be used in a
/// [`CMAES`][crate::CMAES]:
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
/// // The objective function's state can be retrieved after optimization
/// let mut custom = Custom { counter: 0.0 };
///
/// {
///     let mut cmaes_state = CMAESOptions::new(2).build(&mut custom).unwrap();
///     cmaes_state.run(20000);
/// }
///
/// println!("{}", custom.counter);
/// ```
///
/// Wrapper types for modifying objective functions are provided in the
/// [`objective_function`][crate::objective_function] module, and more can be implemented manually.
pub trait ObjectiveFunction {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64;
}

impl<F: FnMut(&DVector<f64>) -> f64> ObjectiveFunction for F {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        (self)(x)
    }
}

/// A type that wraps any [`ObjectiveFunction`] and scales the input vectors before passing them to
/// the wrapped function.
///
/// Can be used to restrict or widen the the search space in one or more
/// dimensions in case it is known that the solution likely lies within a narrower or wider range in
/// them.
///
/// # Examples
///
/// ```
/// # use cmaes::{CMAESOptions, DVector};
/// use cmaes::objective_function::Scale;
///
/// let mut function = |x: &DVector<f64>| x.iter().sum();
/// let scale = Scale::new(function, vec![0.2, 0.2, 1.0]);
///
/// let mut state = CMAESOptions::new(2).build(scale).unwrap();
/// ```
#[derive(Clone)]
pub struct Scale<F> {
    function: F,
    scales: DVector<f64>,
}

impl<F: ObjectiveFunction> Scale<F> {
    /// Returns a new `Scale`, wrapping `function` and multiplying each dimension of its inputs by
    /// the respective element of `scales`.
    pub fn new<S: Into<DVector<f64>>>(function: F, scales: S) -> Self {
        Self {
            function,
            scales: scales.into(),
        }
    }

    /// Applies the `Scale` to a vector and returns it.
    pub fn scale(&self, vector: &DVector<f64>) -> DVector<f64> {
        vector.component_mul(&self.scales)
    }

    /// Returns the scales.
    pub fn scales(&self) -> &DVector<f64> {
        &self.scales
    }

    /// Consumes `self` and returns the wrapped function.
    pub fn into_wrapped_function(self) -> F {
        self.function
    }
}

impl<F: ObjectiveFunction> ObjectiveFunction for Scale<F> {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        let scaled = self.scale(x);
        self.function.evaluate(&scaled)
    }
}

impl<'a, F: ObjectiveFunction> ObjectiveFunction for &'a mut Scale<F> {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        ObjectiveFunction::evaluate(*self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        let mut function = |x: &DVector<f64>| x.iter().sum();
        let scales = vec![2.0, 4.0, 8.0];
        let mut scale = Scale::new(&mut function, scales);

        assert_eq!(
            DVector::from(vec![2.0, 4.0, 8.0]),
            scale.scale(&vec![1.0; 3].into())
        );
        assert_eq!(14.0, scale.evaluate(&vec![1.0; 3].into()));
    }
}
