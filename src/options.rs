//! Option types for the CMA-ES algorithm

const DEFAULT_THREADS: usize = 1;
const DEFAULT_END_CONDITION: CMAESEndConditions = CMAESEndConditions::MaxGenerations(500);
const DEFAULT_STEP_SIZE: f64 = 0.3;
const DEFAULT_STANDARD_DEVIATION: f64 = 1.0;

#[derive(Clone)]
/// An enum representing a condition under which to terminate the CMA-ES algorithm.
pub enum CMAESEndConditions {
    // Maybe add a few more here
    /// Terminate if best fitness changes by less than some amount for some amount of generations.
    /// Usage: StableGenerations(/* fitness */, /* generations */)
    StableGenerations(f64, usize),

    /// Terminate if best fitness is under some amount.
    /// Usage: FitnessThreshold(/* fitness */)
    FitnessThreshold(f64),

    /// Terminate after the generation count reaches a number.
    /// Usage: MaxGenerations(/* generations */)
    MaxGenerations(usize),

    /// Terminate after calling the fitness function some amount of times.
    /// Usage: MaxEvaluations(/* calls */)
    MaxEvaluations(usize)
}

#[derive(Clone)]
/// A container for end conditions, problem dimension, and thread count.
///
/// # Examples
///
/// ```
/// use cmaes::CMAESOptions;
///
/// // A set of options with 2 variables to optimize, and a default of
/// // 1 thread and 500 max generations.
/// let default = CMAESOptions::default(2);
///
/// // A set of options with 2 variables to optimize, 2000 max evaluations,
/// // 1 thread, and 10 stable generations with 0.01 change in fitness.
/// let custom = CMAESOptions::custom(2)
///     .max_evaluations(2000)
///     .stable_generations(0.01, 10);
pub struct CMAESOptions {
    pub end_conditions: Vec<CMAESEndConditions>,
    pub dimension: usize,
    pub initial_step_size: f64,
    pub initial_standard_deviations: Vec<f64>,
    pub threads: usize
}

impl CMAESOptions {
    /// Returns a set of default options with the specified dimension (number of variables to
    /// optimize).
    pub fn default(dimension: usize) -> CMAESOptions {
        CMAESOptions {
            end_conditions: vec![DEFAULT_END_CONDITION],
            dimension: dimension,
            initial_step_size: DEFAULT_STEP_SIZE,
            initial_standard_deviations: vec![DEFAULT_STANDARD_DEVIATION; dimension],
            threads: DEFAULT_THREADS
        }
    }

    /// Returns a set of options with no end conditions.
    pub fn custom(dimension: usize) -> CMAESOptions {
        CMAESOptions {
            end_conditions: Vec::new(),
            dimension: dimension,
            initial_step_size: DEFAULT_STEP_SIZE,
            initial_standard_deviations: vec![DEFAULT_STANDARD_DEVIATION; dimension],
            threads: DEFAULT_THREADS
        }
    }

    /// Sets the number of threads to use in the algorithm.
    pub fn threads(mut self, threads: usize) -> CMAESOptions {
        self.threads = threads;
        self
    }

    /// Sets the initial step size (search radius). This is only a starting point and is adapted by
    /// the algorithm.
    pub fn initial_step_size(mut self, step_size: f64) -> CMAESOptions {
        if !step_size.is_normal() {
            panic!("Initial step size cannot be NaN or infinite");
        }

        self.initial_step_size = step_size;
        self
    }

    /// Sets the initial standard deviations of each variable (individual search radii). These are
    /// only used as starting points and are adapted by the algorithm.
    pub fn initial_standard_deviations(mut self, deviations: Vec<f64>) -> CMAESOptions {
        if deviations.len() != self.dimension {
            panic!("Length of initial deviation vector must be equal to the number of dimensions");
        }

        self.initial_standard_deviations = deviations;
        self
    }

    /// Sets the stable generation count. The algorithm terminates if the specified number of
    /// generations pass where the change in best fitness is under the specified amount.
    pub fn stable_generations(mut self, fitness: f64, generations: usize) -> CMAESOptions {
        self.add_condition(CMAESEndConditions::StableGenerations(fitness, generations));
        self
    }

    /// Sets the minimum fitness. The algorithm terminates if the best fitness is under the
    /// threshold.
    pub fn fitness_threshold(mut self, fitness: f64) -> CMAESOptions {
        self.add_condition(CMAESEndConditions::FitnessThreshold(fitness));
        self
    }

    /// Sets the maximum generation count. The algorithm terminates after the specified number of
    /// generations.
    pub fn max_generations(mut self, generations: usize) -> CMAESOptions {
        self.add_condition(CMAESEndConditions::MaxGenerations(generations));
        self
    }

    /// Sets the maximum evaluation count. The algorithm terminates after the specified number of
    /// fitness function calls.
    pub fn max_evaluations(mut self, evaluations: usize) -> CMAESOptions {
        self.add_condition(CMAESEndConditions::MaxEvaluations(evaluations));
        self
    }

    fn add_condition(&mut self, condition: CMAESEndConditions) {
        let mut duplicate = false;
        let mut duplicates = Vec::new();

        for (i, c) in self.end_conditions.iter().enumerate() {
            match (c.clone(), condition.clone()) {
                (CMAESEndConditions::StableGenerations(..),
                 CMAESEndConditions::StableGenerations(..)) => duplicate = true,
                (CMAESEndConditions::FitnessThreshold(..),
                 CMAESEndConditions::FitnessThreshold(..)) => duplicate = true,
                (CMAESEndConditions::MaxGenerations(..),
                 CMAESEndConditions::MaxGenerations(..)) => duplicate = true,
                (CMAESEndConditions::MaxEvaluations(..),
                 CMAESEndConditions::MaxEvaluations(..)) => duplicate = true,
                _ => duplicate = false,
            }

            if duplicate {
                duplicates.push((i, condition.clone()));
            }
        }

        for d in duplicates {
            self.end_conditions[d.0] = d.1;
        }

        if !duplicate {
            self.end_conditions.push(condition);
        }
    }
}
