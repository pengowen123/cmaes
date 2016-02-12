// Vector and matrix math functions

use la::Matrix;

pub fn magnitude(vec: &[f64]) -> f64 {
    let mut magnitude = 0.0;

    for n in vec {
        magnitude += n.powi(2);
    }

    if magnitude == 0.0 {
        0.0
    } else {
        magnitude.sqrt()
    }
}

pub fn concat(vecs: Vec<Vec<f64>>) -> Vec<f64> {
    let mut new = Vec::new();

    for v in vecs {
        new.extend_from_slice(&v);
    }

    new
}

pub fn reverse<'a, T: Clone>(vec: &[T]) -> Vec<T> {
    let mut new = Vec::new();
    let mut index = vec.len();

    while index <= vec.len() {
        if index == 0 {
            break;
        }
        new.push(vec[index - 1].clone());
        index -= 1;
    }

    new
}

pub fn mul_vec(vec: &[f64], val: f64) -> Vec<f64> {
    let mut new = Vec::new();

    for n in vec {
        new.push(n * val);
    }

    new
}

pub fn sum_vec(vec: &[f64]) -> f64 {
    let mut total = 0.0;

    for i in vec {
        total += *i;
    }

    total
}

pub fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] + b[i]);
    }

    new
}

pub fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] - b[i]);
    }

    new
}

pub fn div_vec(vec: &[f64], val: f64) -> Vec<f64> {
    let mut new = Vec::new();

    for n in vec {
        new.push(n / val);
    }

    new
}

pub fn matrix_by_vector(mat: &Matrix<f64>, vec: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    let mut rows = Vec::new();
    let n = vec.len();
    let w = mat.rows();
    let h = mat.cols();

    for y in 0..h {
        let mut row = Vec::new();

        for x in 0..w {
            row.push(mat[(x, y)]);
        }

        rows.push(row);
    }

    for row in rows {
        let mut sum = 0.0;

        for i in 0..n {
            sum += vec[i] * row[i];
        }

        result.push(sum);
    }

    result
}

pub fn mul_vec_2(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] * b[i]);
    }

    new
}

pub fn transpose(mat: &Matrix<f64>) -> Matrix<f64> {
    let mut new = Matrix::zero(mat.cols(), mat.rows());

    for y in 0..mat.rows() {
        for x in 0..mat.cols() {
            new.set(x, y, mat.get(y, x));
        }
    }

    new
}
