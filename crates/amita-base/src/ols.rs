use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Ix1};
use linfa_linalg::qr::QRInto;

#[derive(Debug, Clone)]
pub enum StandardErrorType {
    NonRobust,
    Robust,
    Clustered { by: Array1<f64> },
}

#[derive(Debug, Clone)]
pub struct OLS {
    y: Array1<f64>,
    x: Array2<f64>,
    standard_error_type: StandardErrorType,

    coefficients: Option<Array1<f64>>,
    standard_errors: Option<Array1<f64>>,

    y_pred: Option<Array1<f64>>,
    resid: Option<Array1<f64>>,
}

// initializers
impl OLS {
    pub fn new<'a>(
        y: ArrayView1<'a, f64>, 
        x: ArrayView2<'a, f64>
    ) -> Self {
        assert_eq!(
            y.len(), x.nrows(), 
            "y and x must have same number of observations"
        );

        Self {
            y: y.to_owned(),
            x: x.to_owned(),
            standard_error_type: StandardErrorType::NonRobust,

            coefficients: None,
            standard_errors: None,

            y_pred: None,
            resid: None,
        }
    }

    pub fn with_se_type(mut self, se_type: StandardErrorType) -> Self {
        self.standard_error_type = se_type;
        self
    }
}

// fit model
impl OLS {
    pub fn fit(self) -> Self {
        self
        .coefficients()
        .standard_errors()
    }

    fn coefficients(mut self) -> Self {
        let xtx = self.x.t().dot(&self.x);
        let xtx_inv = xtx
            .qr_into().expect("Invertible X'X")
            .inverse().expect("Invertible X'X");
        let xty = self.x.t().dot(&self.y);

        let coefficients = xtx_inv.dot(&xty);
        self.coefficients = Some(
            coefficients
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap()
        );

        let y_pred = self.x.dot(&coefficients);
        let resid = self.y.clone() - y_pred.clone();

        self.y_pred = Some(y_pred);
        self.resid = Some(resid);

        self
    }

    fn standard_errors(mut self) -> Self {
        let standard_errors = match self.standard_error_type {
            StandardErrorType::Robust => self.nonrobust_standard_errors(),
            _ => todo!()
        };

        self.standard_errors = Some(standard_errors);
        self
    }

    fn nonrobust_standard_errors(&self) -> Array1<f64> {
        let resid = self.resid.clone().expect("Model not fitted");
        let sigma = resid.std(1.0);

        let xtx = self.x.t().dot(&self.x);
        let cov_mat = xtx
            .qr_into().expect("Invertible X'X")
            .inverse().expect("Invertible X'X")
            .map(|x| *x * sigma);
        let se = cov_mat
            .diag()
            .map(|x| x.sqrt() / self.y.len() as f64);
        println!("{:#?}", se);
        se
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn ols_toy_test() {
        let y = array![1., 4., 5.];
        let x = array![
            [3., 4.],
            [3.2, 4.1],
            [9.2, 1.1],
        ];

        let ols = OLS::new(y.view(), x.view());
        let ols = ols.coefficients();
        let se = ols.nonrobust_standard_errors();

        println!("{:#?}", se);
    }
}