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
        .coefficients()
        .standard_errors()
    }

    fn coefficients(mut self) -> Self {
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

    fn standard_errors(mut self) -> Self {
        let standard_errors = match &self.standard_error_type {
            StandardErrorType::NonRobust => self.nonrobust_standard_errors(),
            StandardErrorType::Robust => self.robust_standard_errors(StandardErrorType::HC3),
            StandardErrorType::HC1 => self.robust_standard_errors(StandardErrorType::HC1),
            StandardErrorType::HC2 => self.robust_standard_errors(StandardErrorType::HC2),
            StandardErrorType::HC3 => self.robust_standard_errors(StandardErrorType::HC3),
            StandardErrorType::Clustered { by } => 
                self.clustered_standard_errors(by.view()),
        };

        self.standard_errors = Some(standard_errors);
        self
    }

    fn nonrobust_standard_errors(&self) -> Array1<f64> {
        let resid = self.resid.clone().expect("Model not fitted");
        let sigma = resid.std(1.0);

        let xtx_inv = self.xtx_inv.clone().expect("");
        let cov_mat = xtx_inv
            .map(|x| *x * sigma);
        let se = cov_mat
            .diag()
            .map(|x| x.sqrt());

        se
    }

    fn robust_standard_errors(&self, standard_error_type: StandardErrorType) -> Array1<f64> {
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

    fn clustered_standard_errors(&self, _by: ArrayView1<f64>) -> Array1<f64> {
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
        0.09184247, 0.62550257, 0.74293058, 0.07347042, 0.61313713,
        0.32103812, 0.69467948, 0.28029683, 0.43068552, 0.5422824 ,
        0.44097417, 0.75832185, 0.3563132 , 0.46074985, 0.24582634,
        0.69433977, 0.67706361, 0.7216024 , 0.01432223, 0.2093061 ,
        0.1712295 , 0.16820138, 0.09317267, 0.79562067, 0.89561129,
        0.53049739, 0.95144704, 0.1682391 , 0.60949497, 0.68278071,
        0.85771881, 0.01447203, 0.50908602, 0.66133125, 0.10508593,
        0.14901226, 0.38936361, 0.05824232, 0.33843664, 0.49621872,
        0.29156797, 0.5045094 , 0.73430844, 0.59400911, 0.58004687,
        0.42484166, 0.67932485, 0.80259561, 0.24967627, 0.56792099,
        0.16099413, 0.41191145, 0.07070767, 0.17013061, 0.74232379,
        0.2049244 , 0.35268963, 0.48600805, 0.40294928, 0.8195262 ,
        0.01098976, 0.74142847, 0.64865605, 0.65412679, 0.40619481,
        0.38382031, 0.37745239, 0.56315007, 0.00506225, 0.07140096,
        0.21159548, 0.86848231, 0.38983175, 0.56544448, 0.51843078,
        0.9611881 , 0.59803824, 0.00856724, 0.66751255, 0.2530761 ,
        0.44272496, 0.78398393, 0.96100371, 0.14748761, 0.7001803 ,
        0.46323023, 0.22127803, 0.76211335, 0.83679943, 0.204558  ,
        0.71031714, 0.94155111, 0.94381439, 0.65603307, 0.18831478,
        0.16357139, 0.56755902, 0.69073189, 0.12682648, 0.01388001];
        let x = array![
            [0.86673797, 0.93086729],
            [0.1190607 , 0.36908845],
            [0.55666895, 0.06251238],
            [0.44676691, 0.19860877],
            [0.57969673, 0.39192   ],
            [0.3130378 , 0.35209726],
            [0.64340903, 0.82509963],
            [0.32898253, 0.27461323],
            [0.76228651, 0.71387142],
            [0.41199767, 0.47654773],
            [0.42491462, 0.93719422],
            [0.81947416, 0.75433963],
            [0.39128161, 0.23997311],
            [0.14475847, 0.95932699],
            [0.06449407, 0.93391539],
            [0.88956944, 0.98410685],
            [0.08863503, 0.99865378],
            [0.61139922, 0.35413021],
            [0.74085165, 0.9935106 ],
            [0.64631772, 0.70578474],
            [0.07043252, 0.78552613],
            [0.66205076, 0.43495169],
            [0.68395226, 0.39912127],
            [0.45628479, 0.16639679],
            [0.85563841, 0.89248511],
            [0.85615874, 0.89885454],
            [0.5792661 , 0.66002446],
            [0.88648471, 0.15715184],
            [0.87975974, 0.16158428],
            [0.35355687, 0.95471047],
            [0.89339528, 0.41427853],
            [0.28975578, 0.92276691],
            [0.02487319, 0.07430861],
            [0.98573715, 0.25108361],
            [0.52422735, 0.54798119],
            [0.18308239, 0.84409261],
            [0.93414098, 0.93008706],
            [0.99089782, 0.56372447],
            [0.67581227, 0.56588385],
            [0.66990073, 0.53365427],
            [0.19684744, 0.12294267],
            [0.86456852, 0.52660878],
            [0.80068634, 0.41953367],
            [0.94067756, 0.18882413],
            [0.87060533, 0.15675899],
            [0.41076985, 0.86868165],
            [0.7886197 , 0.8536721 ],
            [0.73196484, 0.32674044],
            [0.03033948, 0.29294923],
            [0.5933108 , 0.4155952 ],
            [0.98117095, 0.83820288],
            [0.14982211, 0.38155742],
            [0.46825049, 0.83039121],
            [0.79710997, 0.31201954],
            [0.05930528, 0.76975787],
            [0.40699646, 0.29632116],
            [0.13055931, 0.89063306],
            [0.66853799, 0.06271364],
            [0.25101991, 0.5811766 ],
            [0.65284635, 0.21892755],
            [0.69594645, 0.19797174],
            [0.74349647, 0.61786879],
            [0.49690618, 0.35612848],
            [0.38211033, 0.07353456],
            [0.18259862, 0.6886489 ],
            [0.0499158 , 0.99923265],
            [0.11471028, 0.09411391],
            [0.40773201, 0.19753423],
            [0.95331789, 0.45002515],
            [0.65959801, 0.49869956],
            [0.95372386, 0.18017258],
            [0.50606284, 0.40880407],
            [0.86018644, 0.38140205],
            [0.19752314, 0.21658255],
            [0.76635226, 0.89034146],
            [0.82796362, 0.25989237],
            [0.97388754, 0.88007566],
            [0.38003472, 0.5078994 ],
            [0.69603631, 0.81642386],
            [0.57998003, 0.17083225],
            [0.07771019, 0.39147617],
            [0.28144354, 0.71197461],
            [0.08263469, 0.2774787 ],
            [0.4104159 , 0.04411456],
            [0.41162779, 0.91944661],
            [0.21849074, 0.84157271],
            [0.78363743, 0.70905252],
            [0.75398378, 0.19000958],
            [0.3074009 , 0.70926388],
            [0.38051081, 0.5188472 ],
            [0.19188396, 0.69324294],
            [0.59850967, 0.67998976],
            [0.00996697, 0.49205278],
            [0.54489617, 0.71758433],
            [0.68394244, 0.162534  ],
            [0.344249  , 0.21569585],
            [0.47252894, 0.7665989 ],
            [0.13981822, 0.7128977 ],
            [0.810617  , 0.60248122],
            [0.75686099, 0.04127328]];

        let ols = OLS::new(y.view(), x.view())
            .with_se_type(StandardErrorType::NonRobust)
            .fit();

        let se = ols.standard_errors;

        println!("{:#?}", se);
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