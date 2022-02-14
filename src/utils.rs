//! Various utilities

use std::cmp::Ordering;

/// Used for finding max/min values
pub fn partial_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Less)
}

/// Formats an `f64` for printing and tries to give it a fixed width
/// Minimum width is 8
pub fn format_num(num: f64, target_width: usize) -> String {
    if num.is_nan() || num.is_infinite() {
        return format!("{:1$}", num, target_width);
    }
    let sign = if num >= 0.0 { ' ' } else { '-' };
    // Account for non-digit characters that are added to every number
    let pad_width = target_width.max(8) - 7;
    let num_string = format!("{:.1$e}", num.abs(), pad_width);
    let mut num_parts = num_string.split('e');
    let num_digits = num_parts.next().unwrap();
    let num_exponent = num_parts.next().unwrap().parse::<i32>().unwrap();

    format!("{}{}e{:+03}", sign, num_digits, num_exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_num() {
        assert_eq!(8, format_num(1.0, 1).chars().count());
        assert_eq!(8, format_num(1.0, 8).chars().count());
        assert_eq!(16, format_num(1.0, 16).chars().count());
        assert_eq!(" 1.2e-02", &format_num(0.01234567891235, 8));
        assert_eq!("-1.2e-02", &format_num(-0.01234567891235, 8));
        assert_eq!(" 1.234560000e-02", &format_num(0.0123456, 16));
        assert_eq!("-1.234560000e-02", &format_num(-0.0123456, 16));
        assert_eq!("-1.234567891e+08", &format_num(-123456789.06, 16));
        assert_eq!(" 0.00000e+00", &format_num(0.0, 12));
        assert_eq!("         NaN", &format_num(f64::NAN, 12));
        assert_eq!("         inf", &format_num(f64::INFINITY, 12));
        assert_eq!("        -inf", &format_num(-f64::INFINITY, 12));
    }
}
