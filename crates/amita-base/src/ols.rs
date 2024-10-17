use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Ix1};
use linfa_linalg::qr::QRInto;

#[derive(Debug, Clone)]
pub enum StandardErrorType {
    NonRobust,
    Robust, // equivalent to HC3
    HC1,
    HC2,
    HC3,
    Clustered { by: Array1<f64> },
}

#[derive(Debug, Clone)]
pub struct OLS {
    y: Array1<f64>,
    x: Array2<f64>,
    standard_error_type: StandardErrorType,

    // process outcomes
    n_obs: u64,
    n_regressors: u64,
    xtx: Option<Array2<f64>>,
    xtx_inv: Option<Array2<f64>>,
    xty: Option<Array1<f64>>,
    hat_mat: Option<Array2<f64>>, // the hat matrix X(X'X)^(-1)X'

    // key results
    coefficients: Option<Array1<f64>>,
    standard_errors: Option<Array1<f64>>,
    y_pred: Option<Array1<f64>>,
    resid: Option<Array1<f64>>,

    // goodness of fit
    // r_sq: Option<f64>,
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

        assert!(
            x.nrows() > x.ncols(),
            "Unidentifiable because the number of observations is less than regressos"
        );

        Self {
            y: y.to_owned(),
            x: x.to_owned(),
            standard_error_type: StandardErrorType::NonRobust,

            n_obs: y.len() as u64,
            n_regressors: x.ncols() as u64,
            xtx: None,
            xtx_inv: None,
            xty: None,
            hat_mat: None,

            coefficients: None,
            standard_errors: None,
            y_pred: None,
            resid: None,

            // r_sq: None,
        }
    }

    pub fn with_se_type(mut self, se_type: StandardErrorType) -> Self {
        self.standard_error_type = se_type;
        self
    }
}

impl OLS {
    fn calculate_process_outcomes(self) -> Self {
        self
        .calculate_xtx()
        .calculate_xtx_inv()
        .calculate_xty()
        .calculate_hat_mat()
    }

    fn calculate_xtx(mut self) -> Self {
        let xtx = self.x.t().dot(&self.x);
        self.xtx = Some(xtx);
        self
    }

    fn calculate_xtx_inv(mut self) -> Self {
        let xtx = self.xtx.as_ref()
            .expect("Calculation of X'X is required");
        let xtx_inv = xtx
            .clone()
            .qr_into().expect("Invertible X'X")
            .inverse().expect("Invertible X'X");
        self.xtx_inv = Some(xtx_inv);
        self
    }

    fn calculate_xty(mut self) -> Self {
        let xty = self.x.t().dot(&self.y);
        self.xty = Some(xty);
        self
    }

    fn calculate_hat_mat(mut self) -> Self {
        let xtx_inv = self.xtx_inv.clone().expect("Calculation of X'T^(-1) is needed");
        let hat_mat = self.x.dot(&xtx_inv).dot(&self.x.t());
        self.hat_mat = Some(hat_mat);
        self
    }
}

// fit model
impl OLS {
    pub fn fit(self) -> Self {
        self
        .calculate_process_outcomes()
        .calculate_coefficients()
        .calculate_standard_errors()
    }

    fn calculate_coefficients(mut self) -> Self {
        //TODO: error handling needed here
        let xtx_inv = self.xtx_inv.clone().expect("");
        let xty = self.xty.clone().expect("");

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

    fn calculate_standard_errors(mut self) -> Self {
        let standard_errors = match &self.standard_error_type {
            StandardErrorType::NonRobust => self.calclulate_nonrobust_standard_errors(),

            StandardErrorType::Robust => 
                self.calculate_robust_standard_errors(StandardErrorType::HC3),

            StandardErrorType::HC1 => 
                self.calculate_robust_standard_errors(StandardErrorType::HC1),

            StandardErrorType::HC2 => 
                self.calculate_robust_standard_errors(StandardErrorType::HC2),

            StandardErrorType::HC3 => 
                self.calculate_robust_standard_errors(StandardErrorType::HC3),

            StandardErrorType::Clustered { by } => 
                self.calculate_clustered_standard_errors(by.view()),
        };

        self.standard_errors = Some(standard_errors);
        self
    }

    fn calclulate_nonrobust_standard_errors(&self) -> Array1<f64> {
        let resid = self.resid.clone().expect("Model not fitted");
        let sigma = resid.std(self.n_regressors as f64);

        let xtx_inv = self.xtx_inv.clone().expect("X'X");
        let cov_mat = xtx_inv
            .map(|x| *x * sigma.powi(2));
        let se = cov_mat
            .diag()
            .map(|x| x.sqrt());

        se
    }

    fn calculate_robust_standard_errors(&self, standard_error_type: StandardErrorType) -> Array1<f64> {
        // TODO: check if computations are correct
        let xtx_inv = self.xtx_inv.clone().expect("Calculation of X'X is needed");
        let hat_diag = self.hat_mat.clone().expect("Calculation of Hat Matrix is needed").into_diag();
        let resid = self.resid.clone().expect("Model not fitted");
        let sandwich = Array::from_shape_fn((resid.len(), resid.len()), |(i, j)| {
            match i==j {
                true => match standard_error_type {
                    StandardErrorType::HC1 => resid[i].powi(2),
                    StandardErrorType::HC2 => resid[i].powi(2) / hat_diag[i],
                    StandardErrorType::HC3 => resid[i].powi(2) / hat_diag[i].powi(2),
                    _ => panic!("error"),
                },
                false => 0.0,
            }
        });

        let cov_mat = xtx_inv
            .dot(&self.x.t())
            .dot(&sandwich)
            .dot(&self.x)
            .dot(&xtx_inv);

        let se = cov_mat.diag().map(|x| x.sqrt());
        
        se
    }

    fn calculate_clustered_standard_errors(&self, _by: ArrayView1<f64>) -> Array1<f64> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array};

    use super::*;

    #[test]
    fn ols_toy_test() {
        let y = array![
            1.21480677, 1.75359491, 0.95045338, 0.46518267, 1.62639676,
            1.30519337, 2.2405838 , 1.17628327, 0.44609791, 1.13850936,
            1.54890333, 0.31711068, 2.0041781 , 1.60994092, 1.98669331,
            1.07566677, 0.79548689, 1.33459304, 1.74290122, 1.87712994,
            0.87538709, 3.48007729, 0.66694578, 1.18014668, 1.4914189 ,
            3.36273079, 0.77818997, 1.18842165, 0.28014109, 0.5146091 ];
        let x = array![
            [1.50389588, 1.08146505],
            [0.5792709 , 1.60501128],
            [0.22645338, 2.24800586],
            [0.63913737, 2.29803776],
            [1.53149857, 2.28381663],
            [1.62449029, 1.5287771 ],
            [4.31380024, 2.9520281 ],
            [2.21073239, 1.36139682],
            [1.58739252, 0.73909928],
            [2.31296468, 1.4415048 ],
            [0.946444  , 2.17274346],
            [1.70391658, 1.94405239],
            [5.26801233, 0.5677717 ],
            [0.95175788, 1.69732792],
            [3.89911516, 1.4154462 ],
            [0.31688258, 2.79693791],
            [2.06618706, 1.21376076],
            [2.60475865, 0.63882837],
            [2.57884221, 1.95513352],
            [0.69594403, 2.47892145],
            [1.0535839 , 1.31276684],
            [6.82965844, 3.93642541],
            [0.79309495, 2.12675983],
            [3.65696557, 0.00970589],
            [1.57778657, 2.39584546],
            [6.13640896, 3.97622262],
            [2.2205239 , 1.76285718],
            [1.64263539, 1.00365205],
            [1.64235011, 1.27979749],
            [1.16282132, 1.53101396]];

        let ols = OLS::new(y.view(), x.view())
            .with_se_type(StandardErrorType::HC1)
            .fit();

        let coef = ols.coefficients;
        let se = ols.standard_errors;

        println!("{:#?}, {:#?}", coef, se);
    }

    #[test]
    fn test_diag_matrix() {
        let arr = array![1., 2., 3., 4.];
        let diag_matrix = Array::from_shape_fn((4,4), |(i, j)| {
            match i == j {
                true => arr[i],
                false => 0.0,
            }
        });

        println!("{:#?}", diag_matrix);
    }
}