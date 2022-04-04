//! Traits for types that can be used as an objective function, as well as wrapper types that
//! modify the behavior of objective functions.

use nalgebra::DVector;

/// A trait for types that can be used as an objective function.
///
/// This trait is implemented for functions and closures with the correct signature, so the
/// following can all be used in a [`CMAES`][crate::CMAES]:
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
/// let mut state = CMAESOptions::new(vec![0.0; 2], 1.0).build(baz).unwrap();
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
/// let mut state = CMAESOptions::new(vec![0.0; 2], 1.0).build(&mut baz).unwrap();
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
/// let mut cmaes_state = CMAESOptions::new(vec![0.0; 2], 1.0).build(&mut custom).unwrap();
/// let solution = cmaes_state.run();
///
/// println!("{}", custom.counter);
/// ```
///
/// Wrapper types for modifying objective functions are provided in the
/// [`objective_function`][crate::objective_function] module, and more can be implemented manually.
///
/// For objective functions that can be executed in parallel, see
/// [`ParallelObjectiveFunction`][ParallelObjectiveFunction].
pub trait ObjectiveFunction {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64;
}

impl<F: FnMut(&DVector<f64>) -> f64> ObjectiveFunction for F {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        (self)(x)
    }
}

impl ObjectiveFunction for Box<dyn ObjectiveFunction> {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        self.as_mut().evaluate(x)
    }
}

/// Like [`ObjectiveFunction`][ObjectiveFunction], but for objective functions that can be executed
/// in parallel.
///
/// The key difference is that it requires [`Sync`][std::marker::Sync] and has a `&self` parameter
/// instead of `&mut self`, meaning that mutable state cannot be used except for with interior
/// mutability, such as through [`Mutex`][std::sync::Mutex] or [atomic types][std::sync::atomic].
///
/// The trait is implemented for functions and closures with the correct signature, so the following
/// can all be used in a [`CMAES`][crate::CMAES]:
///
/// ```
/// use cmaes::DVector;
///
/// fn foo(x: &DVector<f64>) -> f64 { x.magnitude()  }
/// let bar = |x: &DVector<f64>| x.magnitude();
/// ```
///
/// It can also be implemented for custom types:
///
/// ```
/// use cmaes::{CMAESOptions, DVector, ObjectiveFunction, ParallelObjectiveFunction};
///
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// struct Custom {
///     counter: AtomicUsize,
/// }
///
/// impl ParallelObjectiveFunction for Custom {
///     fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
///         // Interior mutability must be used to modify any state
///         self.counter.fetch_add(1, Ordering::SeqCst);
///         x.magnitude()
///     }
/// }
///
/// impl<'a> ParallelObjectiveFunction for &'a Custom {
///     fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
///         Custom::evaluate_parallel(*self, x)
///     }
/// }
///
/// // `ObjectiveFunction` can be implemented as well for use with `CMAES::run`
/// impl ObjectiveFunction for Custom {
///     fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
///         self.evaluate_parallel(x)
///     }
/// }
///
/// impl<'a> ObjectiveFunction for &'a mut Custom {
///     fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
///         self.evaluate_parallel(x)
///     }
/// }
///
/// // The objective function's state can be retrieved after optimization
/// let custom = Custom { counter: 0.into() };
///
/// let mut cmaes_state = CMAESOptions::new(vec![0.0; 2], 1.0).build(&custom).unwrap();
/// let solution = cmaes_state.run_parallel();
///
/// println!("{}", custom.counter.load(Ordering::SeqCst));
/// ```
pub trait ParallelObjectiveFunction: Sync {
    fn evaluate_parallel(&self, x: &DVector<f64>) -> f64;
}

impl<F: Sync + Fn(&DVector<f64>) -> f64> ParallelObjectiveFunction for F {
    fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
        (self)(x)
    }
}

impl ParallelObjectiveFunction for Box<dyn ParallelObjectiveFunction> {
    fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
        self.as_ref().evaluate_parallel(x)
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
/// let mut function = |x: &DVector<f64>| x.magnitude();
/// let scale = Scale::new(function, vec![0.2, 0.2, 1.0]);
///
/// let mut state = CMAESOptions::new(vec![0.0; 2], 1.0).build(scale).unwrap();
/// ```
#[derive(Clone)]
pub struct Scale<F> {
    function: F,
    scales: DVector<f64>,
}

impl<F> Scale<F> {
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

impl<F: ParallelObjectiveFunction> ParallelObjectiveFunction for Scale<F> {
    fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
        let scaled = self.scale(x);
        self.function.evaluate_parallel(&scaled)
    }
}

impl<'a, F: ParallelObjectiveFunction> ParallelObjectiveFunction for &'a Scale<F> {
    fn evaluate_parallel(&self, x: &DVector<f64>) -> f64 {
        ParallelObjectiveFunction::evaluate_parallel(*self, x)
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
