//! Configuration of data plot generation

/// Configuration of the data plot.
#[derive(Clone, Debug)]
pub struct PlotOptions {
    /// Minimum function evaluations between each data point. Can be used to adjust the granularity
    /// of the recorded data points, with `0` recording a data point every generation.
    pub min_gap_evals: usize,
    /// Whether to use scientific notation for non-log scale axis labels.
    pub scientific_notation: bool,
}

impl PlotOptions {
    /// Creates a new `PlotOptions` with the provided values.
    pub fn new(min_gap_evals: usize, scientific_notation: bool) -> Self {
        Self {
            min_gap_evals,
            scientific_notation,
        }
    }
}
