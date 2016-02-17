/// The cmaes_loop function requires a type that implements this trait. Use the self argument to
/// get additional data to factor into the fitness calculation from other struct fields.
/// Implementing it for a dummy type also works.
///
/// # Examples
///
/// ```rust
/// use cmaes::FitnessFunction;
///
/// #[derive(Clone)]
/// struct FitnessDummy;
///
/// impl FitnessFunction for FitnessDummy {
///     fn get_fitness(&self, parameters: &[f64]) -> f64 {
///         // Calculate fitness of the parameters
///         parameters[0] + parameters[1]
///     }
/// }
/// ```
pub trait FitnessFunction {
    fn get_fitness(&self, parameters: &[f64]) -> f64;
}
