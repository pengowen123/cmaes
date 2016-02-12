//! A standard CMA-ES implementation with multithreaded support.
//! Implemented based on example code [here](en.wikipedia.org/wiki/CMA-ES).
//!
//! The algorithm minimizes the fitness function, so a lower fitness
//! represents a better individual.
//!
//! # Examples
//!
//! ```
//! use cmaes::*;
//!
//! struct FitnessDummy;
//!
//! impl FitnessFunction for FitnessDummy {
//!     fn get_fitness(parameters: &[f64]) -> f64 {
//!         // Calculate fitness here
//!
//!         0.0
//!     }
//! }
//!
//! // See the documentation for CMAESOptions for a list of all options
//! let options = CMAESOptions::default(2);
//!
//! let solution = cmaes_loop(FitnessDummy, options);
//! ```

extern crate la;
extern crate rand;

mod utils;
mod vector;
pub mod fitness;
pub mod cmaes;
pub mod options;

pub use self::cmaes::cmaes_loop;
pub use self::fitness::FitnessFunction;
pub use self::options::CMAESOptions;
