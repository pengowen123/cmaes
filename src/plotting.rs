//! Types for plotting support. See [`Plot`] for usage and what is plotted.

use plotters::chart::{ChartBuilder, ChartContext, SeriesAnno, SeriesLabelPosition};
use plotters::coord;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::combinators::IntoLogRange;
use plotters::coord::ranged1d::{AsRangedCoord, DefaultFormatting, Ranged, ValueFormatter};
use plotters::coord::types::RangedCoordusize;
use plotters::drawing::{DrawingArea, DrawingAreaErrorKind, IntoDrawingArea};
use plotters::element::{Cross, PathElement};
use plotters::prelude::{BitMapBackend, DrawingBackend};
use plotters::series::LineSeries;
use plotters::style::{colors, Color, Palette, Palette99};

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{self, Debug};
use std::fs::DirBuilder;
use std::io;
use std::ops::Range;
use std::path::Path;

use crate::state::State;
use crate::{utils, Individual};

/// The drawing backend to use for rendering the plot.
pub type Backend<'a> = BitMapBackend<'a>;
/// The error type returned by drawing functions.
pub type DrawingError<'a> = DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>;

/// The height of plot images in pixels.
pub const PLOT_HEIGHT: u32 = 1200;
/// The width of plot images in pixels.
pub const PLOT_WIDTH: u32 = 1200;

/// The font to use for text in the plot
const FONT: &str = "sans-serif";

/// Data points for the plot.
#[derive(Clone, Debug)]
struct PlotData {
    /// Function evals at which other data points were recorded
    function_evals: Vec<usize>,
    best_function_value: Vec<f64>,
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
    fn new(dimensions: usize) -> Self {
        Self {
            function_evals: Vec::new(),
            best_function_value: Vec::new(),
            sigma: Vec::new(),
            axis_ratio: Vec::new(),
            mean_dimensions: (0..dimensions).map(|_| Vec::new()).collect(),
            sqrt_eigenvalues: (0..dimensions).map(|_| Vec::new()).collect(),
            coord_axis_scales: (0..dimensions).map(|_| Vec::new()).collect(),
        }
    }

    /// Returns the number of data points currently being stored
    fn len(&self) -> usize {
        self.function_evals.len()
    }

    /// Returns the number of data points for which space has been allocated
    fn capacity(&self) -> usize {
        self.function_evals.capacity()
    }

    /// Returns whether there are no data points in the plot
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a data point to the plot from the current state
    fn add_data_point(
        &mut self,
        current_function_evals: usize,
        state: &State,
        current_best_individual: Option<&Individual>,
    ) {
        self.function_evals.push(current_function_evals);
        let best_function_value = current_best_individual
            .map(|x| x.value)
            // At 0 function evals there isn't a best individual yet, so assign it NAN and filter it
            // later
            .unwrap_or(f64::NAN);
        self.best_function_value
            .push(apply_offset(best_function_value));
        self.sigma.push(apply_offset(state.sigma()));

        let mut sqrt_eigenvalues = state.cov_sqrt_eigenvalues().diagonal();
        self.axis_ratio.push(apply_offset(state.axis_ratio()));

        let mean = state.mean();
        for (i, x) in mean.iter().enumerate() {
            self.mean_dimensions[i].push(*x);
        }

        let sqrt_eigenvalues = sqrt_eigenvalues.as_mut_slice();
        sqrt_eigenvalues.sort_by(|a, b| utils::partial_cmp(*a, *b));
        for (i, x) in sqrt_eigenvalues.iter().enumerate() {
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
    fn clear(&mut self) {
        let clear = |data: &mut Vec<_>| {
            let len = data.len();
            data.swap(0, len - 1);
            data.truncate(1);
        };

        let len = self.function_evals.len();
        self.function_evals.swap(0, len - 1);
        self.function_evals.truncate(1);

        clear(&mut self.best_function_value);
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

/// Data plot for the algorithm. Can be obtained by calling
/// [`CMAES::get_plot`][crate::CMAES::get_plot] or
/// [`CMAES::get_mut_plot`][crate::CMAES::get_mut_plot] and should be saved with
/// [`save_to_file`][`Self::save_to_file`]. Configuration is done using [`PlotOptions`]. To enable
/// the plot, use [`CMAESOptions::enable_plot`][crate::CMAESOptions::enable_plot].
///
/// Plots for each iteration the:
/// - Distance from the minimum objective function value
/// - Absolute objective function value
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
/// let mut state = CMAESOptions::new(10)
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
        current_best_individual: Option<&Individual>,
    ) {
        let already_added = match self.last_data_point_generation {
            Some(generation) => generation == state.generation(),
            None => false,
        };
        if !already_added {
            self.data
                .add_data_point(current_function_evals, state, current_best_individual);
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

        self.draw_single_dimensioned(&top_left)?;
        self.draw_mean(&top_right)?;
        self.draw_sqrt_eigenvalues(&bottom_left)?;
        self.draw_coord_axis_scales(&bottom_right)?;

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

    /// Draws all single-dimensioned data to the drawing area (f - min(f), abs(f), sigma, axis
    /// ratio)
    fn draw_single_dimensioned(
        &self,
        area: &DrawingArea<Backend, coord::Shift>,
    ) -> Result<(), DrawingAreaErrorKind<<Backend as DrawingBackend>::ErrorType>> {
        let (min_index, min_function_value) = self
            .data
            .best_function_value
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or((0, 0.0));

        // The number of times the minimum function value appears, used to decide whether to include
        // it in the plot
        let min_count = self
            .data
            .best_function_value
            .iter()
            .filter(|y| **y == min_function_value)
            .count();

        // Transform from f to f - min(f)
        let dist_to_min = self
            .data
            .best_function_value
            .iter()
            .map(|y| apply_offset(y - min_function_value));

        let abs_best_value = self.data.best_function_value.iter().map(|y| y.abs());

        // Excludes a few values to not break the range
        let all_y_values = dist_to_min
            .clone()
            .enumerate()
            // The minimum value will be drawn if it is reached more than once, so include it in the
            // range only in that case
            .filter(|&(i, y)| !y.is_nan() && (min_count > 1 || i != min_index))
            .map(|(_, y)| y)
            .chain(abs_best_value.clone().filter(|y| !y.is_nan()))
            .chain(self.data.sigma.iter().cloned())
            .chain(self.data.axis_ratio.iter().cloned());
        let (y_range, num_y_labels) = get_log_range(all_y_values);

        let draw = |context: &mut ChartContext<_, _>| {
            let function_evals = self.data.function_evals.iter().cloned();

            // All points to the left of the minimum value
            // Include the minimum value if it is reached more than once (to avoid ugly discontinuities)
            let num_left = if min_count > 1 {
                min_index + 1
            } else {
                min_index
            };
            let points_dist_left = get_points(
                function_evals.clone().take(num_left),
                dist_to_min.clone().take(num_left),
            );
            add_to_legend(
                context.draw_series(LineSeries::new(points_dist_left, &colors::CYAN))?,
                "f - min(f)",
                colors::CYAN,
            );

            // All points to the right of the minimum value
            let num_skip = min_index + 1;
            let points_dist_right = get_points(
                function_evals.clone().skip(num_skip),
                dist_to_min.clone().skip(num_skip),
            );
            context.draw_series(LineSeries::new(points_dist_right, &colors::CYAN))?;

            // Best function values
            let points_abs_best_value = get_points(function_evals.clone(), abs_best_value);
            add_to_legend(
                context.draw_series(LineSeries::new(points_abs_best_value, &colors::BLUE))?,
                "abs(f)",
                colors::BLUE,
            );

            // Marker for overall best function value
            if !min_function_value.is_nan() {
                let abs_overall_best = (
                    self.data.function_evals[min_index],
                    min_function_value.abs(),
                );
                context
                    .plotting_area()
                    .draw(&Cross::new(abs_overall_best, 10, colors::RED))?;
            }

            // Sigma
            let points_sigma = get_points(function_evals.clone(), self.data.sigma.iter().cloned());
            add_to_legend(
                context.draw_series(LineSeries::new(points_sigma, &colors::GREEN))?,
                "Sigma",
                colors::GREEN,
            );

            // Axis ratio
            let points_axis_ratio =
                get_points(function_evals.clone(), self.data.axis_ratio.iter().cloned());
            add_to_legend(
                context.draw_series(LineSeries::new(points_axis_ratio, &colors::RED))?,
                "Axis Ratio",
                colors::RED,
            );
            Ok(())
        };

        self.configure_area(
            area,
            "f - min(f), abs(f), Sigma, Axis Ratio",
            Some(SeriesLabelPosition::LowerLeft),
            y_range,
            true,
            num_y_labels,
            self.options.scientific_notation,
            draw,
        )
    }

    /// Draws the mean to the drawing area
    fn draw_mean(
        &self,
        area: &DrawingArea<Backend, coord::Shift>,
    ) -> Result<(), DrawingAreaErrorKind<<Backend as DrawingBackend>::ErrorType>> {
        let all_y_values = self
            .data
            .mean_dimensions
            .iter()
            .flat_map(|d| d.iter().cloned());
        let (y_range, num_y_labels) = get_range(all_y_values);

        let draw = |context: &mut ChartContext<_, _>| {
            for (i, x) in self.data.mean_dimensions.iter().enumerate() {
                let points =
                    get_points(self.data.function_evals.iter().cloned(), x.iter().cloned());
                let color = Palette99::pick(i);
                add_to_legend(
                    context.draw_series(LineSeries::new(points, &color))?,
                    &format!("x[{}]", i),
                    color,
                );
            }

            Ok(())
        };

        self.configure_area(
            area,
            "Mean",
            Some(SeriesLabelPosition::LowerRight),
            y_range,
            false,
            num_y_labels,
            self.options.scientific_notation,
            draw,
        )
    }

    /// Draws the distribution axis scales to the drawing area
    fn draw_sqrt_eigenvalues(
        &self,
        area: &DrawingArea<Backend, coord::Shift>,
    ) -> Result<(), DrawingAreaErrorKind<<Backend as DrawingBackend>::ErrorType>> {
        let all_y_values = self
            .data
            .sqrt_eigenvalues
            .iter()
            .flat_map(|d| d.iter().cloned());
        let (y_range, num_y_labels) = get_log_range(all_y_values);

        let draw = |context: &mut ChartContext<_, _>| {
            for (i, x) in self.data.sqrt_eigenvalues.iter().enumerate() {
                let points =
                    get_points(self.data.function_evals.iter().cloned(), x.iter().cloned());
                context.draw_series(LineSeries::new(points, &Palette99::pick(i)))?;
            }

            Ok(())
        };

        self.configure_area(
            area,
            "Distribution Axis Scales",
            None,
            y_range,
            true,
            num_y_labels,
            self.options.scientific_notation,
            draw,
        )
    }

    /// Draws the coordinate axis standard deviations (without sigma) to the drawing area
    fn draw_coord_axis_scales(
        &self,
        area: &DrawingArea<Backend, coord::Shift>,
    ) -> Result<(), DrawingAreaErrorKind<<Backend as DrawingBackend>::ErrorType>> {
        let all_y_values = self
            .data
            .coord_axis_scales
            .iter()
            .flat_map(|d| d.iter().cloned());
        let (y_range, num_y_labels) = get_log_range(all_y_values);

        let draw = |context: &mut ChartContext<_, _>| {
            for (i, x) in self.data.coord_axis_scales.iter().enumerate() {
                let points =
                    get_points(self.data.function_evals.iter().cloned(), x.iter().cloned());
                let color = Palette99::pick(i);
                add_to_legend(
                    context.draw_series(LineSeries::new(points, &color))?,
                    &format!("{}", i),
                    color,
                );
            }

            Ok(())
        };

        self.configure_area(
            area,
            "Coord. Axis Standard Deviations (without sigma)",
            Some(SeriesLabelPosition::LowerLeft),
            y_range,
            true,
            num_y_labels,
            self.options.scientific_notation,
            draw,
        )
    }

    /// Creates a `ChartContext` with a common style and calls `map` with it
    #[allow(clippy::too_many_arguments)]
    fn configure_area<'a, 'b, Y, F>(
        &self,
        area: &'a DrawingArea<Backend<'b>, coord::Shift>,
        caption: &str,
        legend: Option<SeriesLabelPosition>,
        y_range: Y,
        log_y: bool,
        num_y_labels: usize,
        scientific_notation: bool,
        map: F,
    ) -> Result<(), DrawingError<'a>>
    where
        Y: AsRangedCoord<Value = f64>,
        Y::CoordDescType: ValueFormatter<f64>,
        F: FnOnce(
            &mut ChartContext<'a, Backend<'b>, Cartesian2d<RangedCoordusize, Y::CoordDescType>>,
        ) -> Result<(), DrawingError<'a>>,
    {
        let x_start = *self.data.function_evals.first().unwrap();
        let x_end = *self.data.function_evals.last().unwrap();
        let x_range = x_start..(x_end as f64 + (x_end - x_start) as f64 * 0.05) as usize;

        let y_label_formatter = |v: &f64| {
            if log_y {
                format!("1e{}", v.log10().round())
            } else if scientific_notation {
                format!("{:e}", v)
            } else {
                format!("{}", v)
            }
        };

        let mut context = ChartBuilder::on(area)
            .margin(30)
            .x_label_area_size(50)
            .y_label_area_size(40)
            .caption(caption, (FONT, 28))
            .build_cartesian_2d(x_range, y_range)?;

        context
            .configure_mesh()
            // Hide the fine mesh lines
            .light_line_style(&colors::WHITE)
            .x_labels(8)
            .x_label_formatter(&|v: &usize| format!("{}", v))
            .x_label_style((FONT, 22))
            .x_desc("Function Evaluations")
            .y_labels(num_y_labels)
            .y_label_formatter(&y_label_formatter)
            .y_label_style((FONT, 22))
            .axis_desc_style((FONT, 22))
            .draw()?;

        map(&mut context)?;

        if let Some(position) = legend {
            context
                .configure_series_labels()
                .label_font((FONT, 20))
                .border_style(&colors::BLACK)
                .position(position)
                .draw()?;
        }

        Ok(())
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

fn add_to_legend<C: Color + 'static>(annotation: &mut SeriesAnno<Backend>, label: &str, color: C) {
    annotation
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
}

/// Applies a small offset to the value to prevent taking the log of zero
fn apply_offset(x: f64) -> f64 {
    let offset = 1e-20;
    if x >= 0.0 {
        x + offset
    } else {
        x - offset
    }
}

/// Returns an iterator of (x, y) points with NAN y points filtered out
fn get_points<'a, X, Y>(x: X, y: Y) -> impl Iterator<Item = (usize, f64)> + 'a
where
    X: IntoIterator<Item = usize> + 'a,
    Y: IntoIterator<Item = f64> + 'a,
{
    x.into_iter()
        .zip(y)
        .filter(|&(_, y)| !y.is_nan())
        .map(|(x, y)| (x, y))
}

/// Returns a log range encompassing all values in the iterator and the number of y-labels to use
/// for the range. The range has a small margin added to either end.
fn get_log_range<I: Iterator<Item = f64> + Clone>(
    iter: I,
) -> (
    // A hack to work around LogRangeExt being private
    impl AsRangedCoord<
        CoordDescType = impl Ranged<FormatOption = DefaultFormatting, ValueType = f64>,
        Value = f64,
    >,
    usize,
) {
    // Margin to be added to the top and bottom of the range
    let margin = 0.4;
    let log_min = iter
        .clone()
        .min_by(|a, b| utils::partial_cmp(*a, *b))
        .unwrap()
        .log10()
        - margin;
    let log_max = iter
        .max_by(|a, b| utils::partial_cmp(*a, *b))
        .unwrap()
        .log10()
        + margin;

    let num_labels = ((log_max - log_min).round() as usize).min(26);
    (
        (10f64.powf(log_min)..10f64.powf(log_max)).log_scale(),
        num_labels,
    )
}

/// Returns a range encompassing all values in the iterator and the number of y-labels to use for
/// the range. The range has a small margin added to either end.
fn get_range<I: Iterator<Item = f64> + Clone>(iter: I) -> (Range<f64>, usize) {
    let mut min = iter
        .clone()
        .min_by(|a, b| utils::partial_cmp(*a, *b))
        .unwrap();
    let mut max = iter.max_by(|a, b| utils::partial_cmp(*a, *b)).unwrap();
    let mut margin = (max - min) * 0.15;

    if margin == 0.0 {
        if max == 0.0 {
            margin = 0.15;
        } else {
            margin = max * 0.15;
        }
    }

    min -= margin;
    max += margin;

    let num_labels = 26;
    (min..max, num_labels)
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
        let state = CMAESOptions::new(10).build(|_: &DVector<f64>| 0.0).unwrap();
        assert!(state.get_plot().is_none());
    }

    #[test]
    fn test_plot_empty() {
        let state = CMAESOptions::new(10)
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
        let mut state = CMAESOptions::new(10)
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
        let mut state = CMAESOptions::new(10)
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
        let mut state = CMAESOptions::new(10)
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
