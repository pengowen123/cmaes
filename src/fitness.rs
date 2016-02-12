pub trait FitnessFunction {
    fn get_fitness(parameters: &[f64]) -> f64;
}
