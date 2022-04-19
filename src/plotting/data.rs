//! Handling of adding and storing data points for the plot

use super::utils::apply_offset;
use crate::history::History;
use crate::state::State;
use crate::utils::partial_cmp;

/// Data points for the plot.
#[derive(Clone, Debug)]
pub struct PlotData {
    /// Function evals at which other data points were recorded
    function_evals: Vec<usize>,
    best_function_value: Vec<f64>,
    median_function_value: Vec<f64>,
    sigma: Vec<f64>,
    axis_ratio: Vec<f64>,
    // Each element of the following contains the histories of an individual dimension
    mean_dimensions: Vec<Vec<f64>>,
    sqrt_eigenvalues: Vec<Vec<f64>>,
    // Standard deviation in each coordinate axis (without sigma)
    coord_axis_scales: Vec<Vec<f64>>,
}

impl PlotData {
    /// Creates an empty `PlotData`
    pub fn new(dimensions: usize) -> Self {
        Self {
            function_evals: Vec::new(),
            best_function_value: Vec::new(),
            median_function_value: Vec::new(),
            sigma: Vec::new(),
            axis_ratio: Vec::new(),
            mean_dimensions: (0..dimensions).map(|_| Vec::new()).collect(),
            sqrt_eigenvalues: (0..dimensions).map(|_| Vec::new()).collect(),
            coord_axis_scales: (0..dimensions).map(|_| Vec::new()).collect(),
        }
    }

    /// Returns the number of data points currently being stored
    pub fn len(&self) -> usize {
        self.function_evals.len()
    }

    /// Returns the number of data points for which space has been allocated
    pub fn capacity(&self) -> usize {
        self.function_evals.capacity()
    }

    /// Returns whether there are no data points in the plot
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn function_evals(&self) -> &[usize] {
        &self.function_evals
    }

    pub fn best_function_value(&self) -> &[f64] {
        &self.best_function_value
    }

    pub fn median_function_value(&self) -> &[f64] {
        &self.median_function_value
    }

    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    pub fn axis_ratio(&self) -> &[f64] {
        &self.axis_ratio
    }

    pub fn mean_dimensions(&self) -> &[Vec<f64>] {
        &self.mean_dimensions
    }

    pub fn sqrt_eigenvalues(&self) -> &[Vec<f64>] {
        &self.sqrt_eigenvalues
    }

    pub fn coord_axis_scales(&self) -> &[Vec<f64>] {
        &self.coord_axis_scales
    }

    /// Adds a data point to the plot from the current state
    pub fn add_data_point(
        &mut self,
        current_function_evals: usize,
        state: &State,
        history: &History,
    ) {
        let best_function_value = history
            .current_best_individual()
            .map(|x| x.value)
            // At 0 function evals there isn't a best individual yet, so assign it NAN and filter it
            // later
            .unwrap_or(f64::NAN);

        let median_function_value = history
            .current_median_function_value()
            // At 0 function evals there isn't a median function value yet, so use NAN and filter it
            // later
            .unwrap_or(f64::NAN);

        self.function_evals.push(current_function_evals);
        self.best_function_value
            .push(apply_offset(best_function_value));
        self.median_function_value
            .push(apply_offset(median_function_value));
        self.sigma.push(apply_offset(state.sigma()));

        self.axis_ratio.push(apply_offset(state.axis_ratio()));

        let mean = state.mean();
        for (i, x) in mean.iter().enumerate() {
            self.mean_dimensions[i].push(*x);
        }

        let mut sqrt_eigenvalues = state.cov_sqrt_eigenvalues().diagonal();
        let sorted_sqrt_eigenvalues = sqrt_eigenvalues.as_mut_slice();
        sorted_sqrt_eigenvalues.sort_by(|a, b| partial_cmp(*a, *b));
        for (i, x) in sorted_sqrt_eigenvalues.iter().enumerate() {
            self.sqrt_eigenvalues[i].push(apply_offset(*x));
        }

        let cov_diagonal = state.cov().diagonal();
        let coord_axis_scales = cov_diagonal.iter().map(|x| x.sqrt());
        for (i, x) in coord_axis_scales.enumerate() {
            self.coord_axis_scales[i].push(apply_offset(x));
        }
    }

    /// Clears the plot except for the most recent data point in each history (note that the
    /// memory is not actually freed; it is only cleared for reuse).
    pub fn clear(&mut self) {
        let clear = |data: &mut Vec<_>| {
            let len = data.len();
            data.swap(0, len - 1);
            data.truncate(1);
        };

        let len = self.function_evals.len();
        self.function_evals.swap(0, len - 1);
        self.function_evals.truncate(1);

        clear(&mut self.best_function_value);
        clear(&mut self.median_function_value);
        clear(&mut self.sigma);
        clear(&mut self.axis_ratio);

        for x in &mut self.mean_dimensions {
            clear(x);
        }

        for x in &mut self.coord_axis_scales {
            clear(x);
        }

        for x in &mut self.sqrt_eigenvalues {
            clear(x);
        }
    }
}
