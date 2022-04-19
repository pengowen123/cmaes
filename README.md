# cmaes

[![Crates.io](https://img.shields.io/crates/v/cmaes)](https://crates.io/crates/cmaes)
[![Rust](https://github.com/pengowen123/cmaes/actions/workflows/rust.yml/badge.svg?branch=master)](https://github.com/pengowen123/cmaes/actions/workflows/rust.yml)

A Rust implementation of the CMA-ES optimization algorithm. It is used to minimize the value of an objective function and performs well on high-dimension, non-linear, non-convex, ill-conditioned, and/or noisy problems. See [this paper][5] for details on the algorithm itself.

## Dependencies

`cmaes` uses some external libraries, so the following dependencies are required:

- Make
- CMake
- A C compiler
- A Fortran compiler (GCC's gfortran works)
- Rust (tested with rustc 1.57, earlier versions may work)
- FreeType

Dependencies may differ depending on the selected LAPACK implementation. Building is currently only supported on Linux (see [issue #4][2]).

## Quick Start

Add this to your Cargo.toml:

```
[dependencies]
cmaes = "0.1"
```

The LAPACK implementation used may be selected through Cargo features (see `Cargo.toml`). `netlib` is built from source by default.

Then, to optimize a function:
```rust
use cmaes::DVector;

let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
let dim = 10;
let solution = cmaes::fmin(sphere, vec![5.0; dim], 1.0);
```

More options can be accessed through the `CMAESOptions` type, including data plots (requires the `plotters` feature):
```rust
use cmaes::{CMAESOptions, DVector, PlotOptions};

let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();

let dim = 10;
let mut cmaes_state = CMAESOptions::new(vec![1.0; dim], 1.0)
    .fun_target(1e-8)
    .max_generations(20000)
    .enable_printing(200)
    .enable_plot(PlotOptions::new(0, false))
    .build(sphere)
    .unwrap();

let results = cmaes_state.run();

cmaes_state.get_plot().unwrap().save_to_file("plot.png", true).unwrap();
```

The produced plot will look like this:

<a href="https://github.com/pengowen123/cmaes/tree/master/images/plot_sphere.png">
    <img src="https://pengowen123.github.io/cmaes/images/plot_sphere.png" />
</a>

For more complex problems, automatic restarts are also provided:
```rust
use cmaes::restart::{RestartOptions, RestartStrategy};
use cmaes::DVector;

let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();

let strategy = RestartStrategy::BIPOP(Default::default());
let dim = 10;
let search_range = -5.0..=5.0;
let restarter = RestartOptions::new(dim, search_range, strategy)
    .fun_target(1e-10)
    .enable_printing(true)
    .build()
    .unwrap();

let results = restarter.run(|| sphere);
```

For more information, see the [documentation][0] and [examples][1].

## Testing

The library's tests can be run with `cargo test --release`. Note that some tests may fail occasionally due to the random nature of the algorithm, but as long as no tests fail consistently then they can be considered to have passed.

Benchmarks can be run with `cargo bench`.

## Contributing

Contributions are welcome! You can contribute by reporting any bugs or issues you have with the library, adding documentation, or opening pull requests.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without any additional terms or conditions.

## License

Licensed under either of

    Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
    MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Citations

The following contain more detailed information on the algorithms implemented by this library or were referenced in its implementation.

Auger, Anne and Hansen, Nikolaus. “A Restart CMA Evolution Strategy with Increasing Population Size.” 2005 IEEE Congress on Evolutionary Computation, vol. 2, 2005, pp. 1769-1776 Vol. 2, [https://doi.org/10.1109/CEC.2005.1554902][3].

Hansen, Nikolaus. “Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed.” GECCO (Companion), July 2009, [https://doi.org/10.1145/1570256.1570333][4].

Auger, Anne, and Nikolaus Hansen. Tutorial CMA-ES. 2013, [https://doi.org/10.1145/2464576.2483910][5].

Hansen, Nikolaus, Akimoto, Youhei, and Baudis, Petr. CMA-ES/Pycma on Github. Feb. 2019, [https://doi.org/10.5281/zenodo.2559634][6].

[0]: https://docs.rs/cmaes/latest/cmaes
[1]: https://github.com/pengowen123/cmaes/tree/master/examples
[2]: https://github.com/pengowen123/cmaes/issues/4
[3]: https://doi.org/10.1109/CEC.2005.1554902
[4]: https://doi.org/10.1145/1570256.1570333
[5]: https://doi.org/10.1145/2464576.2483910
[6]: https://doi.org/10.5281/zenodo.2559634
