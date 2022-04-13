//! Utilities for plotting

/// Applies a small offset to the value to prevent taking the log of zero
pub fn apply_offset(x: f64) -> f64 {
    let offset = 1e-20;
    if x >= 0.0 {
        x + offset
    } else {
        x - offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_offset() {
        assert_eq!(1.0, apply_offset(1.0));
        assert_eq!(1e-8 + 1e-20, apply_offset(1e-8));
        assert_eq!(1e-20, apply_offset(0.0));
        assert_eq!(-1e-8 - 1e-20, apply_offset(-1e-8));
        assert_eq!(-1.0, apply_offset(-1.0));
    }
}
