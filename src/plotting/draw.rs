//! Drawing of the recorded data points to the plot

use plotters::chart::{ChartBuilder, ChartContext, SeriesAnno, SeriesLabelPosition};
use plotters::coord;
use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::combinators::{IntoLogRange, LogCoord};
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::coord::types::RangedCoordusize;
use plotters::drawing::{DrawingArea, DrawingAreaErrorKind};
use plotters::element::{Cross, PathElement};
use plotters::prelude::DrawingBackend;
use plotters::series::LineSeries;
use plotters::style::{colors, Color, Palette, Palette99};

use std::cmp::Ordering;
use std::ops::Range;

use super::data::PlotData;
use super::options::PlotOptions;
use super::utils::apply_offset;
use super::{Backend, DrawingError};
use crate::utils::partial_cmp;

/// The font to use for text in the plot
const FONT: &str = "sans-serif";

/// Parameters for a y-axis
struct YAxis<Y> {
    /// The range of the y values to be drawn
    range: Y,
    /// The preferred number of axis labels to use (may not be used exactly)
    num_labels: usize,
    /// The kind of axis `range` represents
    kind: YAxisKind,
}

impl<Y> YAxis<Y> {
    fn new(range: Y, num_labels: usize, kind: YAxisKind) -> Self {
        Self {
            range,
            num_labels,
            kind,
        }
    }
}

/// Represents which kind of scale and labels to use for the y-axis
enum YAxisKind {
    /// A log scale
    Log,
    /// A linear scale
    Linear {
        /// Whether to use scientific notation for the linear axis labels
        scientific_notation: bool,
    },
}

/// Draws all single-dimensioned data to the drawing area (f - min(f), abs(f), abs(median),
/// sigma, axis ratio)
pub fn draw_single_dimensioned<'a>(
    data: &PlotData,
    area: &DrawingArea<Backend, coord::Shift>,
) -> Result<(), DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>> {
    let (min_index, min_function_value) = data
        .best_function_value()
        .iter()
        .cloned()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
        .unwrap_or((0, 0.0));

    // The number of times the minimum function value appears, used to decide whether to include
    // it in the plot
    let min_count = data
        .best_function_value()
        .iter()
        .filter(|y| **y == min_function_value)
        .count();

    // Transform from f to f - min(f)
    let dist_to_min = data
        .best_function_value()
        .iter()
        .map(|y| apply_offset(y - min_function_value));

    let abs_best_value = data.best_function_value().iter().map(|y| y.abs());
    let abs_median_value = data.median_function_value().iter().map(|y| y.abs());

    // Excludes a few values to not break the range
    let all_y_values = dist_to_min
        .clone()
        .enumerate()
        // The minimum value will be drawn if it is reached more than once, so include it in the
        // range only in that case
        .filter(|&(i, _)| min_count > 1 || i != min_index)
        .map(|(_, y)| y)
        .chain(abs_best_value.clone())
        .chain(abs_median_value.clone())
        .chain(data.sigma().iter().cloned())
        .chain(data.axis_ratio().iter().cloned())
        // Filter out dummy values added at initialization
        .filter(|y| !y.is_nan());
    let y_axis = get_log_y_axis(all_y_values);

    let draw = |context: &mut ChartContext<_, _>| {
        let function_evals = data.function_evals().iter().cloned();

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

        // Median function values
        let points_abs_median_value = get_points(function_evals.clone(), abs_median_value);
        add_to_legend(
            context.draw_series(LineSeries::new(points_abs_median_value, &colors::MAGENTA))?,
            "abs(median)",
            colors::MAGENTA,
        );

        // Marker for overall best function value
        if !min_function_value.is_nan() {
            let abs_overall_best = (data.function_evals()[min_index], min_function_value.abs());
            context
                .plotting_area()
                .draw(&Cross::new(abs_overall_best, 10, colors::RED))?;
        }

        // Sigma
        let points_sigma = get_points(function_evals.clone(), data.sigma().iter().cloned());
        add_to_legend(
            context.draw_series(LineSeries::new(points_sigma, &colors::GREEN))?,
            "Sigma",
            colors::GREEN,
        );

        // Axis ratio
        let points_axis_ratio =
            get_points(function_evals.clone(), data.axis_ratio().iter().cloned());
        add_to_legend(
            context.draw_series(LineSeries::new(points_axis_ratio, &colors::RED))?,
            "Axis Ratio",
            colors::RED,
        );
        Ok(())
    };

    DrawingAreaSetup {
        area,
        function_evals_history: data.function_evals(),
        caption: "f - min(f), abs(f), abs(median) Sigma, Axis Ratio",
        legend_position: Some(SeriesLabelPosition::LowerLeft),
        y_axis,
        draw,
    }
    .configure_area()
}

/// Draws the mean to the drawing area
pub fn draw_mean<'a>(
    data: &PlotData,
    options: &PlotOptions,
    area: &DrawingArea<Backend, coord::Shift>,
) -> Result<(), DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>> {
    let all_y_values = data
        .mean_dimensions()
        .iter()
        .flat_map(|d| d.iter().cloned());
    let y_axis = get_linear_y_axis(all_y_values, options.scientific_notation);

    let draw = |context: &mut ChartContext<_, _>| {
        for (i, x) in data.mean_dimensions().iter().enumerate() {
            let points = get_points(data.function_evals().iter().cloned(), x.iter().cloned());
            let color = Palette99::pick(i);
            add_to_legend(
                context.draw_series(LineSeries::new(points, &color))?,
                &format!("x[{}]", i),
                color,
            );
        }

        Ok(())
    };

    DrawingAreaSetup {
        area,
        function_evals_history: data.function_evals(),
        caption: "Mean",
        legend_position: Some(SeriesLabelPosition::LowerRight),
        y_axis,
        draw,
    }
    .configure_area()
}

/// Draws the distribution axis scales to the drawing area
pub fn draw_sqrt_eigenvalues<'a>(
    data: &PlotData,
    area: &DrawingArea<Backend, coord::Shift>,
) -> Result<(), DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>> {
    let all_y_values = data
        .sqrt_eigenvalues()
        .iter()
        .flat_map(|d| d.iter().cloned());
    let y_axis = get_log_y_axis(all_y_values);

    let draw = |context: &mut ChartContext<_, _>| {
        for (i, x) in data.sqrt_eigenvalues().iter().enumerate() {
            let points = get_points(data.function_evals().iter().cloned(), x.iter().cloned());
            context.draw_series(LineSeries::new(points, &Palette99::pick(i)))?;
        }

        Ok(())
    };

    DrawingAreaSetup {
        area,
        function_evals_history: data.function_evals(),
        caption: "Distribution Axis Scales",
        legend_position: None,
        y_axis,
        draw,
    }
    .configure_area()
}

/// Draws the coordinate axis standard deviations (without sigma) to the drawing area
pub fn draw_coord_axis_scales<'a>(
    data: &PlotData,
    area: &DrawingArea<Backend, coord::Shift>,
) -> Result<(), DrawingAreaErrorKind<<Backend<'a> as DrawingBackend>::ErrorType>> {
    let all_y_values = data
        .coord_axis_scales()
        .iter()
        .flat_map(|d| d.iter().cloned());
    let y_axis = get_log_y_axis(all_y_values);

    let draw = |context: &mut ChartContext<_, _>| {
        for (i, x) in data.coord_axis_scales().iter().enumerate() {
            let points = get_points(data.function_evals().iter().cloned(), x.iter().cloned());
            let color = Palette99::pick(i);
            add_to_legend(
                context.draw_series(LineSeries::new(points, &color))?,
                &format!("{}", i),
                color,
            );
        }

        Ok(())
    };

    DrawingAreaSetup {
        area,
        function_evals_history: data.function_evals(),
        caption: "Coord. Axis Standard Deviations (without sigma)",
        legend_position: Some(SeriesLabelPosition::LowerLeft),
        y_axis,
        draw,
    }
    .configure_area()
}

/// Stores parameters for configuring a drawing area
struct DrawingAreaSetup<'a, 'b, Y, F> {
    // The drawing area to configure
    area: &'a DrawingArea<Backend<'b>, coord::Shift>,
    // The function evals history
    function_evals_history: &'a [usize],
    // The caption for the drawing area
    caption: &'static str,
    // The position of the legend (disabled if `None`)
    legend_position: Option<SeriesLabelPosition>,
    // Parameters for the y-axis
    y_axis: YAxis<Y>,
    // Called on the `ChartContext` after setup
    // Should be used to draw elements onto the drawing area
    draw: F,
}

impl<'a, 'b, Y, F> DrawingAreaSetup<'a, 'b, Y, F>
where
    Y: AsRangedCoord<Value = f64>,
    Y::CoordDescType: ValueFormatter<f64>,
    F: FnOnce(
        &mut ChartContext<'a, Backend<'b>, Cartesian2d<RangedCoordusize, Y::CoordDescType>>,
    ) -> Result<(), DrawingError<'a>>,
{
    /// Creates a `ChartContext` with a common style, configures it according to the stored options,
    /// and calls `draw` to draw to it
    fn configure_area(self) -> Result<(), DrawingError<'a>> {
        let x_start = *self.function_evals_history.first().unwrap();
        let x_end = *self.function_evals_history.last().unwrap();
        let x_range = x_start..(x_end as f64 * 1.05) as usize;

        let y_label_formatter = |v: &f64| match self.y_axis.kind {
            YAxisKind::Log => format!("1e{}", v.log10().round()),
            YAxisKind::Linear {
                scientific_notation: true,
            } => format!("{:e}", v),
            YAxisKind::Linear {
                scientific_notation: false,
            } => format!("{}", v),
        };

        let mut context = ChartBuilder::on(self.area)
            .margin(30)
            .x_label_area_size(50)
            .y_label_area_size(40)
            .caption(self.caption, (FONT, 28))
            .build_cartesian_2d(x_range, self.y_axis.range)?;

        context
            .configure_mesh()
            // Hide the fine mesh lines
            .light_line_style(&colors::WHITE)
            .x_labels(8)
            .x_label_formatter(&|v: &usize| format!("{}", v))
            .x_label_style((FONT, 22))
            .x_desc("Function Evaluations")
            .y_labels(self.y_axis.num_labels)
            .y_label_formatter(&y_label_formatter)
            .y_label_style((FONT, 22))
            .axis_desc_style((FONT, 22))
            .draw()?;

        (self.draw)(&mut context)?;

        if let Some(position) = self.legend_position {
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

/// Adds the series to the legend with the provided label and color
fn add_to_legend<C: Color + 'static>(annotation: &mut SeriesAnno<Backend>, label: &str, color: C) {
    annotation
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
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

/// Returns a log axis encompassing all values in the iterator. The range has a small margin added
/// to either end.
fn get_log_y_axis<I: Iterator<Item = f64> + Clone>(iter: I) -> YAxis<LogCoord<f64>> {
    // Margin to be added to the top and bottom of the range
    let margin = 0.4;
    let log_min = iter
        .clone()
        .min_by(|a, b| partial_cmp(*a, *b))
        .unwrap()
        .log10()
        - margin;
    let log_max = iter.max_by(|a, b| partial_cmp(*a, *b)).unwrap().log10() + margin;

    let num_labels = ((log_max - log_min).round() as usize).min(26);
    let y_range = (10f64.powf(log_min)..10f64.powf(log_max))
        .log_scale()
        .into();

    YAxis::new(y_range, num_labels, YAxisKind::Log)
}

/// Returns a linear axis encompassing all values in the iterator. The range has a small margin
/// added to either end.
///
/// Uses scientific notation for axis labels if `scientific_notation` is true
fn get_linear_y_axis<I: Iterator<Item = f64> + Clone>(
    iter: I,
    scientific_notation: bool,
) -> YAxis<Range<f64>> {
    let mut min = iter.clone().min_by(|a, b| partial_cmp(*a, *b)).unwrap();
    let mut max = iter.max_by(|a, b| partial_cmp(*a, *b)).unwrap();
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
    let y_range = min..max;

    YAxis::new(
        y_range,
        num_labels,
        YAxisKind::Linear {
            scientific_notation,
        },
    )
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use plotters::coord::ranged1d::Ranged;

    use super::*;

    #[test]
    fn test_get_points() {
        let points = get_points(0..4, [0.0, 1.0, f64::NAN, 3.0]).collect::<Vec<_>>();

        assert_eq!(vec![(0usize, 0f64), (1, 1.0), (3, 3.0)], points);
    }

    #[test]
    fn test_get_linear_y_axis() {
        let range = (0..=100).map(|x| x as f64);
        let y_axis = get_linear_y_axis(range, true);

        assert_eq!(-15.0, y_axis.range.start);
        assert_eq!(115.0, y_axis.range.end);
        assert!(matches!(
            y_axis.kind,
            YAxisKind::Linear {
                scientific_notation: true
            }
        ));
    }

    #[test]
    fn test_get_log_y_axis() {
        let range = (1..=100).map(|x| x as f64 * 100.0);
        let y_axis = get_log_y_axis(range);

        let margin_factor = 10f64.powf(0.4);
        assert_approx_eq!(100.0 / margin_factor, y_axis.range.range().start);
        assert_approx_eq!(10000.0 * margin_factor, y_axis.range.range().end);
        assert_eq!(3, y_axis.num_labels);
        assert!(matches!(y_axis.kind, YAxisKind::Log));
    }
}
