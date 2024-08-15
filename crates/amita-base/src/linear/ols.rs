use amita_error::AmitaError;
use amita_utils::traits::{Solver, SolverResults};
use linfa_linalg::qr::QR;
use ndarray::{Array1, Array2, Ix1};
use statrs::distribution::{ContinuousCDF, StudentsT};

#[derive(Debug, Clone)]
pub enum SEType {
    NonRobust,
    Robust,
}

#[derive(Debug, Clone)]
pub struct OLSResults {
    n_obs: usize,
    n_regressors: usize,
    se_type: SEType,

    // estimates
    coef: Option<Array1<f64>>, // beta
    se: Option<Array1<f64>>, // standard error of beta
    y_pred: Option<Array1<f64>>, // predicted y, or fitted y
    resid: Option<Array1<f64>>, // residuals, or the error term
    t: Option<Array1<f64>>, // t statistics
    p_vals: Option<Array1<f64>>, // p values of H_0: \beta = 0

    // goodness of fit
    r_sq: Option<f64>,
    r_sq_adj: Option<f64>,
}


impl SolverResults for OLSResults {
    fn coef(&self) -> Result<Array1<f64>, AmitaError> {
        let coef = self.coef.clone()
            .ok_or(AmitaError::NotSolved)?;
        Ok(coef)
        
    }

    fn se(&self) -> Result<Array1<f64>, AmitaError> {
        self.se.clone().ok_or(AmitaError::NotSolved)
    }

    fn t(&self) -> Result<Array1<f64>, AmitaError> {
        self.t.clone().ok_or(AmitaError::NotSolved)
    }

    fn p_vals(&self) -> Result<Array1<f64>, AmitaError> {
        self.p_vals.clone().ok_or(AmitaError::NotSolved)
    }
    
    fn summary(&self) -> Result<String, AmitaError> {
        todo!()
    }
}


#[derive(Debug, Clone)]
pub struct OLSSolver {
    y: Array1<f64>,
    x: Array2<f64>,

    q: Array2<f64>,
    r: Array2<f64>,

    results: OLSResults
}

impl Solver<OLSResults> for OLSSolver {
    fn results(&self) -> OLSResults {
        self.results.clone()
    }

    fn solve(self) -> Result<Self, AmitaError> {
        self
        .solve_coef()?
        .solve_se()?
        .solve_r_sq()

    }
}

// initializers
impl OLSSolver {
    pub fn new(
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<Self, AmitaError> {
        let _res = OLSSolver::validate_data(&y, &x)?;

        let y = y.to_owned();
        let x = x.to_owned();

        let qr_decomp = x.clone().qr().map_err(|_| {AmitaError::NotQRDecomposable { matrix_name: "Regressors x".to_string() }})?;
        let q = qr_decomp.generate_q();
        let r = qr_decomp.into_r();

        let n_obs = x.shape()[0];
        let n_regressors = x.shape()[1];

        let results = OLSResults {
            n_obs,
            n_regressors,
            se_type: SEType::NonRobust,

            coef: None,
            se: None,
            y_pred: None,
            resid: None,
            t: None,
            p_vals: None,

            r_sq: None,
            r_sq_adj: None,
        };

        Ok( OLSSolver { y, x, q, r, results } )
    }

    pub fn with_robust_se(mut self) -> Self {
        self.results.se_type = SEType::Robust;
        self
    }

    pub fn with_nonrobust_se(mut self) -> Self {
        self.results.se_type = SEType::NonRobust;
        self
    }

    fn validate_data(
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<(), AmitaError> {

        if x.shape()[0] != y.shape()[0] {
            return Err(AmitaError::NotSameObservations)
        }

        Ok(())
    }
}


impl OLSSolver {
    /// Estimate OLS' coefficients using QR-decomposed X:
    /// \beta = R^{-1} Q^{\transpose} y
    fn solve_coef(mut self) -> Result<Self, AmitaError> {
        let y = self.y.clone();
        let x = self.x.clone();
        let r_inverse = 
            self.r.clone()
            .qr().map_err(|_| AmitaError::NotQRDecomposable { 
                matrix_name: "R matrix from QR-decomposed X".to_string() 
            })?
            .inverse().map_err(|_| AmitaError::NotInvertible {
                matrix_name: "R matrix from QR-decomposed X".to_string() 
            })?;
        let q = self.q.clone();

        let coef = r_inverse.dot(&q.t()).dot(&y);
        let y_pred = x.dot(&coef);
        let resid = (&y - &y_pred).to_owned();

        self.results.coef = Some(coef);
        self.results.y_pred = Some(y_pred);
        self.results.resid = Some(resid);

        Ok(self)
    }

    fn solve_se(self) -> Result<Self, AmitaError> {
        match self.results.se_type {
            SEType::NonRobust => {
                self
                .solve_non_robust_se()?
                .solve_t_pvals()
            },
            SEType::Robust => {
                self
                .solve_robust_se()?
                .solve_t_pvals()
            }
        }
    }

    fn solve_non_robust_se(mut self) -> Result<Self, AmitaError> {
        let n_obs = self.results.n_obs;
        let n_regressors = self.results.n_regressors;
        
        let r = self.r.clone();
        let resid = self.results.clone().resid.ok_or(AmitaError::NotSolved)?;

        let sigma = resid.t().dot(&resid) / (n_obs - n_regressors) as f64;
        let variance = r.t().dot(&r)
            .qr().map_err(|_| AmitaError::NotQRDecomposable { 
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?
            .inverse().map_err(|_| AmitaError::NotInvertible {
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?;
        let se = variance.map(|x| (x * sigma).sqrt()).diag().to_owned();

        self.results.se = Some(se);

        Ok(self)
    }

    /// HC1 Robust Standard Error
    fn solve_robust_se(mut self) -> Result<Self, AmitaError> {
        let n_obs = self.results.n_obs as f64;
        let n_regressors = self.results.n_regressors as f64;

        let x = self.x.clone();
        let r = self.r.clone();
        let resid = self.results.clone().resid
            .ok_or(AmitaError::NotSolved)?
            .into_shape((self.results.n_regressors, 1))
            .unwrap(); //TODO: error handling needed here

        // X^{\transpose} X. Utilizing QR decomposition for calculation
        let x_gramian_inverse = r.t().dot(&r)
            .qr().map_err(|_| AmitaError::NotQRDecomposable { 
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?
            .inverse().map_err(|_| AmitaError::NotInvertible {
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?;
        println!("{:#?}", x_gramian_inverse.shape());

        // diag(e^{\transpose}e)
        let resid_gramian_diag = Array2::from_diag(&resid.dot(&resid.t()).diag());
        println!("{:#?}", resid_gramian_diag.shape());

        let variance = x_gramian_inverse.dot(&x.t()).dot(&resid_gramian_diag).dot(&x).dot(&x_gramian_inverse);
        let se = variance.diag().map(|x| (x * n_obs / (n_obs - n_regressors - 1.)).sqrt() );

        self.results.se = Some(se);
        
        Ok(self)
    }

    fn solve_t_pvals(mut self) -> Result<Self, AmitaError> {
        let df = (self.results.n_obs - self.results.n_regressors - 1) as f64;
        let coef = self.results.coef.clone().ok_or(AmitaError::NotSolved)?.into_shape((self.results.n_regressors,)).unwrap().into_dimensionality::<Ix1>().unwrap();
        let se = self.results.se.clone().ok_or(AmitaError::NotSolved)?;

        let t = coef / se.view();

        let t_dist = StudentsT::new(0., 1., df).unwrap();
        let p_vals = t.map(|x| 2. * ( 1. - t_dist.cdf(x.abs()) ) );

        self.results.t = Some(t);
        self.results.p_vals = Some(p_vals);

        Ok(self)
    }
}

// Goodness of fit
impl OLSSolver {
    fn solve_r_sq(mut self) -> Result<Self, AmitaError> {
        let resid = self.results.resid.clone().ok_or(AmitaError::NotSolved)?;
        let y = self.y.clone();
        let y_mean = y.mean().unwrap();

        let df_rss = (self.results.n_obs - self.results.n_regressors - 1) as f64;
        let df_tss = (self.results.n_obs - 1) as f64;

        let rss = resid.map(|x| x.powi(2)).sum();
        let tss = y.map(|x| (x - y_mean).powi(2)).sum();

        let r_sq = 1 as f64 - rss / tss;
        let r_sq_adj = 1 as f64 - ( rss / df_rss ) / ( tss / df_tss );

        self.results.r_sq = Some(r_sq);
        self.results.r_sq_adj = Some(r_sq_adj);

        Ok(self)
    }
}

