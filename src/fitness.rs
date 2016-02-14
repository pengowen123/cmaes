/// The cmaes_loop function requires a type that implements this trait. Implementing it for a dummy
/// type works fine.
///
/// # Examples
///
/// ```rust
/// use cmaes::FitnessFunction;
///
/// struct FitnessDummy;
///
/// impl FitnessFunction for FitnessDummy {
///     fn get_fitness(parameters: &[f64]) -> f64 {
///         // Calculate fitness of the parameters
///         parameters[0] + parameters[1]
///     }
/// }
/// ```
pub trait FitnessFunction {
    fn get_fitness(parameters: &[f64]) -> f64;
}
