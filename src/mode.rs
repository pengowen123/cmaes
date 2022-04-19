//! Handling of different optimization modes (minimize and maximize)

use std::cmp::Ordering;

use crate::utils;

/// The mode to use when optimizing a function.
///
/// The default value is `Minimize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Minimize the value of the function.
    Minimize,
    /// Maximize the value of the function.
    Maximize,
}

impl Default for Mode {
    fn default() -> Self {
        Self::Minimize
    }
}

impl Mode {
    /// Compares the two values
    /// For use in sorting
    pub(crate) fn sort_cmp(&self, a: f64, b: f64) -> Ordering {
        match self {
            Mode::Minimize => utils::partial_cmp(a, b),
            Mode::Maximize => utils::partial_cmp(b, a),
        }
    }

    /// Returns whether `a` is a better function value than `b`
    pub(crate) fn is_better(&self, a: f64, b: f64) -> bool {
        match self {
            Mode::Minimize => a < b,
            Mode::Maximize => a > b,
        }
    }

    pub(crate) fn choose_best(&self, a: f64, b: f64) -> f64 {
        match self {
            Mode::Minimize => a.min(b),
            Mode::Maximize => a.max(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_cmp() {
        assert_eq!(Ordering::Less, Mode::Minimize.sort_cmp(1.0, 2.0));
        assert_eq!(Ordering::Equal, Mode::Minimize.sort_cmp(1.0, 1.0));
        assert_eq!(Ordering::Greater, Mode::Minimize.sort_cmp(2.0, 1.0));

        assert_eq!(Ordering::Greater, Mode::Maximize.sort_cmp(1.0, 2.0));
        assert_eq!(Ordering::Equal, Mode::Maximize.sort_cmp(1.0, 1.0));
        assert_eq!(Ordering::Less, Mode::Maximize.sort_cmp(2.0, 1.0));
    }

    #[test]
    fn test_is_better() {
        assert_eq!(true, Mode::Minimize.is_better(1.0, 2.0));
        assert_eq!(false, Mode::Minimize.is_better(2.0, 2.0));
        assert_eq!(false, Mode::Minimize.is_better(2.0, 1.0));

        assert_eq!(false, Mode::Maximize.is_better(1.0, 2.0));
        assert_eq!(false, Mode::Maximize.is_better(2.0, 2.0));
        assert_eq!(true, Mode::Maximize.is_better(2.0, 1.0));
    }

    #[test]
    fn test_choose_best() {
        assert_eq!(1.0, Mode::Minimize.choose_best(1.0, 2.0));
        assert_eq!(1.0, Mode::Minimize.choose_best(2.0, 1.0));

        assert_eq!(2.0, Mode::Maximize.choose_best(1.0, 2.0));
        assert_eq!(2.0, Mode::Maximize.choose_best(2.0, 1.0));
    }
}
