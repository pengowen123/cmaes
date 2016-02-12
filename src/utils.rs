#[derive(Clone, Debug)]
pub struct Parameters {
    pub parameters: Vec<f64>,
    pub fitness: f64,
}

impl Parameters {
    pub fn new(params: &[f64]) -> Parameters {
        Parameters {
            parameters: params.to_vec(),
            fitness: 0.0,
        }
    }
}
