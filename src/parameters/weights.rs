//! Initialization of weights.

use nalgebra::DVector;

use std::ops::Deref;

/// The distribution of weights for the population. The default value is `Negative`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    /// Weights are higher for higher-ranked selected individuals and are zero for the rest of the
    /// population. Usually performs slightly worse than `Negative`.
    Positive,
    /// Similar to `Positive`, but non-selected individuals have negative weights. With this
    /// setting, the algorithm is known as active CMA-ES or aCMA-ES.
    Negative,
    /// Weights for selected individuals are equal and are zero for the rest of the population. This
    /// setting will likely perform much worse than the others.
    Uniform,
}

impl Default for Weights {
    fn default() -> Self {
        Self::Negative
    }
}

/// Initial distribution of weights, before normalization
#[derive(Clone, Debug)]
pub(super) struct InitialWeights {
    weights: DVector<f64>,
    setting: Weights,
    mu: usize,
}

impl InitialWeights {
    /// Calculates the initial weight distribution based on the population size and weights setting
    pub fn new(lambda: usize, setting: Weights) -> Self {
        let mu = lambda / 2;
        let weights: DVector<f64> = match setting {
            // weights.len() == mu
            Weights::Uniform => vec![1.0; mu],
            // weights.len() == mu
            Weights::Positive => (1..=mu)
                .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
                .collect::<Vec<_>>(),
            // weights.len() == lambda
            Weights::Negative => (1..=lambda)
                .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
                .collect::<Vec<_>>(),
        }
        .into();

        Self {
            weights,
            setting,
            mu,
        }
    }

    /// Returns the number of individuals to select each generation
    pub fn mu(&self) -> usize {
        self.mu
    }

    /// Returns the variance-effective selection mass
    pub fn mu_eff(&self) -> f64 {
        // Square of sum divided by sum of squares of the first mu weights (all positive weights)
        self.weights.iter().take(self.mu).sum::<f64>().powi(2)
            / self
                .weights
                .iter()
                .take(self.mu)
                .map(|w| w.powi(2))
                .sum::<f64>()
    }

    /// Returns the variance-effective mass for the negative weights if there are any negative
    /// weights
    pub fn mu_eff_minus(&self) -> Option<f64> {
        if self.weights.len() > self.mu {
            Some(
                self.weights.iter().skip(self.mu).sum::<f64>().powi(2)
                    / self
                        .weights
                        .iter()
                        .skip(self.mu)
                        .map(|w| w.powi(2))
                        .sum::<f64>(),
            )
        } else {
            None
        }
    }

    /// Calculates the final weight distribution based on the problem dimension and the learning
    /// rates for the rank-one and rank-mu updates
    pub fn finalize(self, dim: usize, c1: f64, cmu: f64) -> FinalWeights {
        let mu_eff = self.mu_eff();
        let mu_eff_minus = self.mu_eff_minus();
        let mut weights = self.weights;

        // Normalize the positive weights to sum to 1
        let sum_positive_weights = weights.iter().filter(|w| **w > 0.0).sum::<f64>();

        for w in &mut weights {
            if *w > 0.0 {
                *w /= sum_positive_weights;
            }
        }

        // Normalize the negative weights to sum to a value chosen below
        if let Some(mu_eff_minus) = mu_eff_minus {
            // Possible sums of negative weights
            // The smallest of these values will be used
            let a_mu = 1.0 + c1 / cmu;
            let a_mu_eff = 1.0 + (2.0 * mu_eff_minus) / (mu_eff + 2.0);
            let a_pos_def = (1.0 - c1 - cmu) / (dim as f64 * cmu);

            let a = a_mu.min(a_mu_eff.min(a_pos_def));

            let sum_negative_weights = weights.iter().filter(|w| **w < 0.0).sum::<f64>().abs();

            for w in &mut weights {
                if *w < 0.0 {
                    *w = *w * a / sum_negative_weights;
                }
            }
        }

        FinalWeights {
            weights,
            setting: self.setting,
        }
    }
}

/// Final distribution of weights, after normalization
#[derive(Clone, Debug)]
pub struct FinalWeights {
    weights: DVector<f64>,
    setting: Weights,
}

impl FinalWeights {
    /// Returns the setting used for calculating these `FinalWeights`
    pub fn setting(&self) -> Weights {
        self.setting
    }
}

impl Deref for FinalWeights {
    type Target = DVector<f64>;

    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    // Tests that Weights::Positive produces only positive weight values and that they are
    // normalized properly
    #[test]
    fn test_weights_positive() {
        for lambda in 4..200 {
            let initial_weights = InitialWeights::new(lambda, Weights::Positive);

            assert!(initial_weights.weights.iter().all(|w| *w > 0.0));

            let final_weights = initial_weights.finalize(6, 0.2, 0.8);

            assert!(final_weights.weights.iter().all(|w| *w > 0.0));
            assert_approx_eq!(final_weights.iter().sum::<f64>(), 1.0, 1e-12);
        }
    }

    // Tests that Weights::Negative produces only positive values for the first mu weights and only
    // negative values for the rest
    #[test]
    fn test_weights_negative() {
        for lambda in 4..200 {
            let initial_weights = InitialWeights::new(lambda, Weights::Negative);
            let mu = initial_weights.mu();

            assert!(initial_weights.weights.iter().take(mu).all(|w| *w > 0.0));

            assert!(initial_weights.weights.iter().skip(mu).all(|w| *w <= 0.0));

            let final_weights = initial_weights.finalize(4, 0.5, 0.5);

            assert_approx_eq!(final_weights.iter().take(mu).sum::<f64>(), 1.0, 1e-12);

            assert!(final_weights.iter().take(mu).all(|w| *w > 0.0));

            assert!(final_weights.iter().skip(mu).all(|w| *w <= 0.0));
        }
    }
}
