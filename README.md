# cmaes
[![Build Status](https://travis-ci.org/pengowen123/cmaes.svg?branch=master)](https://travis-ci.org/pengowen123/cmaes)

An easy to use, multithreaded optimization library.
[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) is an optimization algorithm designed for complex problems, with little knowledge of the problem. Currently compiles on Rust version 1.6.0 stable. Future versions should be okay.

## Usage

Add this to your Cargo.toml:
```
[dependencies]
cmaes = { git = "https://github.com/pengowen123/cmaes" }
```

Here is a simple example:
```rust
extern crate cmaes;

use cmaes::*;

struct FitnessDummy;

impl FitnessFunction for FitnessDummy {
  fn get_fitness(parameters: &[f64]) -> f64 {
    // Calculate fitness of the parameters
    parameters[0] + parameters[1]
  }
}

fn main() {
  // 2 is the problem dimension (number of variables to optimize)
  let options = CMAESOptions::default(2);
  
  // solution will be a vector with optimized numbers
  let solution = cmaes_loop(FitnessDummy, options);
}
```

See the documentation for complete instructions.
