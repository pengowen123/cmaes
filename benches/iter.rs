use cmaes::{
    CMAESOptions, DVector, ObjectiveFunction, PlotOptions, TerminationData, Weights, CMAES,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand;

use std::time::Duration;

// N-dimensional Rosenbrock function
fn rosenbrock(x: &DVector<f64>) -> f64 {
    assert!(x.len() >= 2);
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
        .sum::<f64>()
}

// Builds a `CMAES` with a random initial mean and advances it `100 + extra_setup_iters` iterations
// to avoid benchmarking based on the trivial initial state
fn get_cmaes_state(
    dim: usize,
    lambda_mult: usize,
    weights: Weights,
    plot: bool,
    extra_setup_iters: usize,
) -> CMAES<Box<dyn ObjectiveFunction>> {
    let initial_mean = (0..dim)
        .map(|_| 3.0 * rand::random::<f64>())
        .collect::<Vec<_>>();
    let mut options = CMAESOptions::new(initial_mean, 1.0).weights(weights);
    options.population_size *= lambda_mult;

    if plot {
        options = options.enable_plot(PlotOptions::new(0, false))
    }

    let mut cmaes_state = options.build(Box::new(rosenbrock) as _).unwrap();

    for _ in 0..100 + extra_setup_iters {
        let _ = cmaes_state.next();
    }

    cmaes_state
}

fn single_iter(state: &mut CMAES<Box<dyn ObjectiveFunction>>) -> Option<TerminationData> {
    state.next()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut extra_setup_iters = 0;
    let mut run_bench = |name, dim, lambda_mult, weights, plot| {
        c.bench_function(name, |b| {
            b.iter_batched_ref(
                || {
                    let state = get_cmaes_state(dim, lambda_mult, weights, plot, extra_setup_iters);

                    // Vary the number of initial setup iters to include the proper distribution of
                    // iters with eigen updates
                    extra_setup_iters =
                        (extra_setup_iters + 1) % state.generations_per_eigen_update();

                    state
                },
                |mut state| single_iter(&mut state),
                BatchSize::SmallInput,
            )
        });
    };

    run_bench(
        "single iter aCMA-ES n=3 lambda=default plot=false",
        3,
        1,
        Weights::Negative,
        false,
    );
    run_bench(
        "single iter aCMA-ES n=10 lambda=default plot=false",
        10,
        1,
        Weights::Negative,
        false,
    );
    run_bench(
        "single iter aCMA-ES n=10 lambda=default plot=false",
        10,
        1,
        Weights::Negative,
        false,
    );
    run_bench(
        "single iter aCMA-ES n=10 lambda=10*default plot=false",
        10,
        10,
        Weights::Negative,
        false,
    );
    run_bench(
        "single iter CMA-ES n=10 lambda=default plot=false",
        10,
        1,
        Weights::Positive,
        false,
    );
    run_bench(
        "single iter aCMA-ES n=10 lambda=default plot=true",
        10,
        1,
        Weights::Negative,
        true,
    );
    run_bench(
        "single iter aCMA-ES n=30 lambda=default plot=false",
        30,
        1,
        Weights::Negative,
        false,
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(15)).with_plots();
    targets = criterion_benchmark,
);
criterion_main!(benches);
