//! Types for plotting support. See [`Plot`] for usage and what is plotted.

mod data;
mod draw;
mod options;
mod utils;

pub use options::PlotOptions;

use plotters::coord;
use plotters::drawing::{DrawingArea, DrawingAreaErrorKind, IntoDrawingArea};
use plotters::prelude::{BitMapBackend, DrawingBackend};
use plotters::style::colors;

use std::error::Error;
use std::fmt::{self, Debug};
use std::fs::DirBuilder;
use std::io;
use std::path::Path;

use crate::history::History;
use crate::state::State;
use data::PlotData;

/// The drawing backend to use for rendering the plot.
pub type Backend<'a> = BitMapBackend<'a>;
/// The error type returned by drawing functions.
pub type DrawingError<'a> = DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>;

/// The height of plot images in pixels.
pub const PLOT_HEIGHT: u32 = 1200;
/// The width of plot images in pixels.
pub const PLOT_WIDTH: u32 = 1200;

/// Data plot for the algorithm. Can be obtained by calling
/// [`CMAES::get_plot`][crate::CMAES::get_plot] or
/// [`CMAES::get_mut_plot`][crate::CMAES::get_mut_plot] and should be saved with
/// [`save_to_file`][`Self::save_to_file`]. Configuration is done using [`PlotOptions`]. To enable
/// the plot, use [`CMAESOptions::enable_plot`][crate::CMAESOptions::enable_plot].
///
/// Plots for each iteration the:
/// - Distance of best value from the minimum objective function value
/// - Absolute best objective function value
/// - Absolute median objective function value
/// - Distribution axis ratio
/// - Distribution mean
/// - Scaling of each distribution axis.
/// - Standard deviation in each coordinate axis (without sigma)
///
/// # Examples
///
/// ```no_run
/// use cmaes::{CMAESOptions, DVector, PlotOptions};
///
/// let sphere = |x: &DVector<f64>| x.iter().map(|xi| xi.powi(2)).sum();
/// let mut state = CMAESOptions::new(vec![1.0; 10], 1.0)
///     .enable_plot(PlotOptions::new(0, false))
///     .build(sphere)
///     .unwrap();
///
/// let result = state.run();
///
/// state.get_plot().unwrap().save_to_file("plot.png", true).unwrap();
/// ```
///
/// The produced plot wil look like this:
///
/// <a href="https://pengowen123.github.io/cmaes/images/plot_sphere.png">
///     <img src="https://pengowen123.github.io/cmaes/images/plot_sphere.png"
///         width=750 height=750 />
/// </a>
#[derive(Clone, Debug)]
pub struct Plot {
    data: PlotData,
    options: PlotOptions,
    /// The last time a data point was recorded, in function evals
    /// Is None if no data points have been recorded yet
    last_data_point_evals: Option<usize>,
    /// Like last_data_point_evals, but tracks the generation number of the last data point
    /// recorded
    last_data_point_generation: Option<usize>,
}

impl Plot {
    /// Initializes an empty `Plot` with the provided options.
    pub(crate) fn new(dimensions: usize, options: PlotOptions) -> Self {
        Self {
            data: PlotData::new(dimensions),
            options,
            last_data_point_evals: None,
            last_data_point_generation: None,
        }
    }

    /// Returns the next time a data point should be recorded, in function evals.
    pub(crate) fn get_next_data_point_evals(&self) -> usize {
        match self.last_data_point_evals {
            Some(evals) => evals + self.options.min_gap_evals,
            None => 0,
        }
    }

    /// Adds a data point to the plot from the current state if not already called this generation.
    pub(crate) fn add_data_point(
        &mut self,
        current_function_evals: usize,
        state: &State,
        history: &History,
    ) {
        let already_added = match self.last_data_point_generation {
            Some(generation) => generation == state.generation(),
            None => false,
        };
        if !already_added {
            self.data
                .add_data_point(current_function_evals, state, history);
            self.last_data_point_evals = Some(current_function_evals);
            self.last_data_point_generation = Some(state.generation());
        }
    }

    /// Saves the data plot to a bitmap image file. Recursively creates the necessary directories if
    /// `create_dirs` is `true`.
    pub fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        create_dirs: bool,
    ) -> Result<(), PlotError> {
        let path = path.as_ref();
        if create_dirs {
            if let Some(parent) = path.parent() {
                DirBuilder::new().recursive(true).create(parent)?;
            }
        }

        let plot = self.build_plot(&path)?;
        plot.present().map_err(Into::into)
    }

    /// Builds the data plot and returns it (does not save to a file)
    fn build_plot<'a, P: AsRef<Path> + 'a>(
        &self,
        path: &'a P,
    ) -> Result<DrawingArea<Backend<'a>, coord::Shift>, DrawingError> {
        let root_area = Backend::new(path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();

        root_area.fill(&colors::WHITE)?;

        let mut child_drawing_areas = root_area.split_evenly((2, 2)).into_iter();
        let top_left = child_drawing_areas.next().unwrap();
        let top_right = child_drawing_areas.next().unwrap();
        let bottom_left = child_drawing_areas.next().unwrap();
        let bottom_right = child_drawing_areas.next().unwrap();

        draw::draw_single_dimensioned(&self.data, &top_left)?;
        draw::draw_mean(&self.data, &self.options, &top_right)?;
        draw::draw_sqrt_eigenvalues(&self.data, &bottom_left)?;
        draw::draw_coord_axis_scales(&self.data, &bottom_right)?;

        Ok(root_area)
    }

    /// Returns the number of data points currently stored in the plot.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of data points for which space has been allocated.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns whether there are no data points in the plot.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clears the plot data except for the most recent data point for each variable. Can be called
    /// after using [`save_to_file`][`Plot::save_to_file`] (or not) to avoid endlessly growing
    /// allocations (note that the memory is not actually freed; it is simply cleared for reuse).
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// An error produced while creating or saving a plot.
#[derive(Debug)]
pub enum PlotError<'a> {
    DrawingError(DrawingError<'a>),
    IoError(io::Error),
}

impl<'a> fmt::Display for PlotError<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PlotError::DrawingError(ref e) => write!(fmt, "DrawingError({})", e),
            PlotError::IoError(ref e) => write!(fmt, "IoError({})", e),
        }
    }
}

impl<'a> Error for PlotError<'a> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            PlotError::DrawingError(ref e) => Some(e),
            PlotError::IoError(ref e) => Some(e),
        }
    }
}

impl<'a> From<DrawingError<'a>> for PlotError<'a> {
    fn from(error: DrawingError<'a>) -> Self {
        PlotError::DrawingError(error)
    }
}

impl<'a> From<io::Error> for PlotError<'a> {
    fn from(error: io::Error) -> Self {
        PlotError::IoError(error)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;
    use crate::CMAESOptions;

    fn get_plot_path(name: &str) -> String {
        format!("{}/test_output/{}.png", env!("CARGO_MANIFEST_DIR"), name)
    }

    #[test]
    fn test_plot_not_enabled() {
        let state = CMAESOptions::new(vec![1.0; 10], 1.0)
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();

        assert!(state.get_plot().is_none());
    }

    #[test]
    fn test_plot_empty() {
        let state = CMAESOptions::new(vec![1.0; 10], 1.0)
            .enable_plot(PlotOptions::new(0, false))
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();
        let plot = state.get_plot().unwrap();
        assert!(plot
            .save_to_file(get_plot_path("test_plot_empty"), true)
            .is_ok())
    }

    #[test]
    fn test_plot() {
        let mut state = CMAESOptions::new(vec![1.0; 10], 1.0)
            .enable_plot(PlotOptions::new(0, false))
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();

        for _ in 0..10 {
            let _ = state.next();
        }

        let plot = state.get_plot().unwrap();
        assert!(plot.save_to_file(get_plot_path("test_plot"), true).is_ok());
    }

    #[test]
    fn test_redundant_plot() {
        let mut state = CMAESOptions::new(vec![1.0; 10], 1.0)
            .enable_plot(PlotOptions::new(0, false))
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();

        for _ in 0..10 {
            let _ = state.add_plot_point();
        }

        // Redundant add_plot_point calls are ignored
        assert_eq!(state.get_plot().unwrap().len(), 1);
    }

    #[test]
    fn test_plot_clear() {
        let mut state = CMAESOptions::new(vec![0.0; 10], 1.0)
            .enable_plot(PlotOptions::new(0, false))
            .build(|_: &DVector<f64>| 0.0)
            .unwrap();

        // Fresh plots contain one element
        assert_eq!(state.get_plot().unwrap().len(), 1);
        assert_eq!(state.get_plot().unwrap().capacity(), 4);

        // Clear a fresh plot
        state.get_mut_plot().unwrap().clear();

        // Clearing leaves one element behind
        assert_eq!(state.get_plot().unwrap().len(), 1);
        assert_eq!(state.get_plot().unwrap().capacity(), 4);

        for _ in 0..10 {
            let _ = state.next();
        }

        assert_eq!(state.get_plot().unwrap().len(), 11);
        assert_eq!(state.get_plot().unwrap().capacity(), 16);

        // Clear a plot after adding some data
        state.get_mut_plot().unwrap().clear();

        assert_eq!(state.get_plot().unwrap().len(), 1);
        assert_eq!(state.get_plot().unwrap().capacity(), 16);

        let plot = state.get_plot().unwrap();
        assert!(plot
            .save_to_file(get_plot_path("test_plot_clear"), true)
            .is_ok());
    }
}
