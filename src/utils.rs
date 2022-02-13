//! Various utilities

use std::cmp::Ordering;

/// Used for finding max/min values
pub fn partial_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Less)
}
