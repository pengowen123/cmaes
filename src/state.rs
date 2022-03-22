//! Variable state of the algorithm and updating of that state.

use nalgebra::{DVector, DMatrix};

use crate::parameters::Parameters;

/// Stores the variable state of the algorithm and handles updating it
pub struct State {
    /// The number of generations that have been fully completed
    generation: usize,
    /// The number of times the objective function has been evaluated
    function_evals: usize,
    /// The distribution mean
    mean: DVector<f64>,
    /// The distribution covariance matrix
    cov: DMatrix<f64>,
    /// Normalized eigenvectors of `cov`, forming an orthonormal basis of the matrix
    cov_eigenvectors: DMatrix<f64>,
    /// Diagonal matrix containing the square roots of the eigenvalues of `cov`, which are the
    /// scales of the basis axes
    cov_sqrt_eigenvalues: DMatrix<f64>,
    /// The distribution step size
    sigma: f64,
    /// Evolution path of the mean used to update the covariance matrix
    path_c: DVector<f64>,
    /// Evolution path of the mean used to update the step size
    path_sigma: DVector<f64>,
    /// The last time the eigendecomposition was updated, in function evals
    last_eigen_update_evals: usize,
}

impl State {
    /// Initializes the variable state of the algorithm
    pub fn new(initial_mean: DVector<f64>, initial_sigma: f64) -> Self {
        let dim = initial_mean.len();
        let mean = initial_mean;
        let cov = DMatrix::identity(dim, dim);
        let eigen = decompose_cov(cov.clone()).unwrap();
        let cov_eigenvectors = eigen.eigenvectors;
        let cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        let sigma = initial_sigma;
        let path_c = DVector::zeros(dim);
        let path_sigma = DVector::zeros(dim);

        Self {
            generation: 0,
            function_evals: 0,
            mean,
            cov,
            cov_eigenvectors,
            cov_sqrt_eigenvalues,
            sigma,
            path_c,
            path_sigma,
            last_eigen_update_evals: 0,
        }
    }

    /// Updates the variable state using the provided sampled individuals
    pub fn update(
        &mut self,
        params: &Parameters,
        individuals: &[(DVector<f64>, f64)],
    ) -> Result<(), PosDefCovError> {
        let dim = params.dim();
        let mu = params.mu();
        let mu_eff = params.mu_eff();
        let cc = params.cc();
        let c1 = params.c1();
        let cs = params.cs();
        let cmu = params.cmu();
        let cm = params.cm();
        let damp_s = params.damp_s();

        self.function_evals += individuals.len();

        // Calculate new mean through weighted recombination
        // Only the mu best individuals are used even if there are lambda weights
        let yw = individuals
            .iter()
            .take(mu)
            .enumerate()
            .map(|(i, (y, _))| y * params.weights()[i])
            .sum::<DVector<f64>>();
        self.mean = &self.mean + &(cm * self.sigma * &yw);

        // Update evolution paths
        let sqrt_inv_c = &self.cov_eigenvectors
            * DMatrix::from_diagonal(&self.cov_sqrt_eigenvalues.map_diagonal(|d| 1.0 / d))
            * self.cov_eigenvectors.transpose();

        self.path_sigma = (1.0 - cs) * &self.path_sigma
            + (cs * (2.0 - cs) * mu_eff).sqrt() * &sqrt_inv_c * &yw;

        // Expectation of N(0, I)
        let chi_n = (dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dim as f64) + 1.0 / (21.0 * dim.pow(2) as f64));

        let hs = if (self.path_sigma.magnitude()
            / (1.0 - (1.0 - cs).powi(2 * (self.generation as i32 + 1))).sqrt())
            < (1.4 + 2.0 / (dim as f64 + 1.0)) * chi_n
        {
            1.0
        } else {
            0.0
        };

        self.path_c = (1.0 - cc) * &self.path_c
            + hs * (cc * (2.0 - cc) * mu_eff).sqrt() * &yw;

        // Update step size
        self.sigma *=
            ((cs / damp_s) * ((self.path_sigma.magnitude() / chi_n) - 1.0)).exp();

        // Update covariance matrix
        let weights_cov = params
            .weights()
            .iter()
            .enumerate()
            .map(|(i, w)| {
                *w * if *w >= 0.0 {
                    1.0
                } else {
                    dim as f64 / (&sqrt_inv_c * &individuals[i].0).magnitude().powi(2)
                }
            })
            .collect::<Vec<_>>();

        let delta_hs = (1.0 - hs) * cc * (2.0 - cc);
        self.cov = (1.0 + c1 * delta_hs
            - c1
            - cmu * params.weights().iter().sum::<f64>())
            * &self.cov
            + c1 * &self.path_c * self.path_c.transpose()
            + cmu
                * weights_cov
                    .into_iter()
                    .enumerate()
                    .map(|(i, wc)| wc * &individuals[i].0 * individuals[i].0.transpose())
                    .sum::<DMatrix<f64>>();

        // Ensure symmetry
        self.cov.fill_lower_triangle_with_upper_triangle();

        // Update eigendecomposition occasionally (updating every generation is unnecessary and
        // inefficient for high dim)
        let evals_per_eigen = (0.5 * dim as f64 * params.lambda() as f64
            / ((c1 + cmu) * dim.pow(2) as f64))
            as usize;

        if self.function_evals > self.last_eigen_update_evals + evals_per_eigen {
            self.last_eigen_update_evals = self.function_evals;

            let eigen = decompose_cov(self.cov.clone())?;
            self.cov_eigenvectors = eigen.eigenvectors;
            self.cov_sqrt_eigenvalues = eigen.sqrt_eigenvalues;
        }

        self.generation += 1;

        Ok(())
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn function_evals(&self) -> usize {
        self.function_evals
    }

    pub fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    pub fn cov(&self) -> &DMatrix<f64> {
        &self.cov
    }

    pub fn cov_eigenvectors(&self) -> &DMatrix<f64> {
        &self.cov_eigenvectors
    }

    pub fn cov_sqrt_eigenvalues(&self) -> &DMatrix<f64> {
        &self.cov_sqrt_eigenvalues
    }

    /// Returns the current axis ratio of the distribution
    pub fn axis_ratio(&self) -> f64 {
        let diag = self.cov_sqrt_eigenvalues.diagonal();
        diag.max() / diag.min()
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn path_c(&self) -> &DVector<f64> {
        &self.path_c
    }

    // These methods are only used for setting up termination tests
    #[cfg(test)]
    pub fn mut_generation(&mut self) -> &mut usize {
        &mut self.generation
    }

    #[cfg(test)]
    pub fn mut_cov(&mut self) -> &mut DMatrix<f64> {
        &mut self.cov
    }

    #[cfg(test)]
    pub fn mut_mean(&mut self) -> &mut DVector<f64> {
        &mut self.mean
    }

    #[cfg(test)]
    pub fn mut_cov_eigenvectors(&mut self) -> &mut DMatrix<f64> {
        &mut self.cov_eigenvectors
    }

    #[cfg(test)]
    pub fn mut_cov_sqrt_eigenvalues(&mut self) -> &mut DMatrix<f64> {
        &mut self.cov_sqrt_eigenvalues
    }

    #[cfg(test)]
    pub fn mut_sigma(&mut self) -> &mut f64 {
        &mut self.sigma
    }
}

/// Decomposition of a covariance matrix
struct CovDecomposition {
    /// Columns are eigenvectors
    eigenvectors: DMatrix<f64>,
    /// Diagonal matrix with square roots of eigenvalues
    sqrt_eigenvalues: DMatrix<f64>,
}

#[derive(Clone, Debug)]
pub struct PosDefCovError;

/// Decomposes a covariance matrix into a set of normalized eigenvectors and a diagonal matrix
/// containing the square roots of the corresponding eigenvalues
///
/// Returns `Err` if the matrix is not positive-definite
fn decompose_cov(matrix: DMatrix<f64>) -> Result<CovDecomposition, PosDefCovError> {
    let mut eigen = nalgebra_lapack::SymmetricEigen::new(matrix);

    for mut col in eigen.eigenvectors.column_iter_mut() {
        col.normalize_mut();
    }

    if eigen.eigenvalues.iter().any(|x| *x <= 0.0) {
        Err(PosDefCovError)
    } else {
        Ok(CovDecomposition {
            eigenvectors: eigen.eigenvectors,
            sqrt_eigenvalues: DMatrix::from_diagonal(&eigen.eigenvalues.map(|x| x.sqrt())),
        })
    }
}

// Only used in termination tests
//
// Returns (eigenvectors, eigenvalues)
#[cfg(test)]
pub fn decompose_cov_pub(matrix: DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>), PosDefCovError> {
    decompose_cov(matrix).map(|eigen| (eigen.eigenvectors, eigen.sqrt_eigenvalues))
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn test_decompose_cov() {
        let matrix = DMatrix::from_iterator(2, 2, [3.0, 1.5, 1.5, 2.0]);

        let eigen = decompose_cov(matrix.clone()).unwrap();

        let reconstructed = eigen.eigenvectors.clone()
            * eigen.sqrt_eigenvalues.pow(2)
            * eigen.eigenvectors.transpose();

        for x in (reconstructed - matrix).iter() {
            assert_approx_eq!(x, 0.0);
        }
    }
}
