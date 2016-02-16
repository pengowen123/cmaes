# cmaes
[![Build Status](https://travis-ci.org/pengowen123/cmaes.svg?branch=master)](https://travis-ci.org/pengowen123/cmaes)

An easy to use, multithreaded optimization library.
[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) is an optimization algorithm designed for non-linear, noisy, high dimension problems. Currently compiles on Rust version 1.6.0 stablebut future versions should be okay. The main function returns an Option; if you get a None value consider it a bug and please report it.

## Usage

Add this to your Cargo.toml:
```
[dependencies]
cmaes = { git = "https://github.com/pengowen123/cmaes" }
```

And this to your crate root:
```rust
extern crate cmaes;
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
  let solution = cmaes_loop(FitnessDummy, options).unwrap();
}
```

See the [documentation](http://pengowen123.github.io/cmaes/cmaes/index.html) for complete instructions.

# Contributing

If you encounter a bug, please open an issue explaining what happened, and include reproduction steps. If you have a suggestion for a feature, either open an issue or add it yourself and open a pull request.
