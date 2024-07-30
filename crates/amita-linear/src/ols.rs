use amita_error::AmitaError;
use linfa_linalg::qr::QR;
use ndarray::{ArrayBase, Dim, Ix1, Ix2, OwnedRepr, ViewRepr};
use statrs::distribution::{ContinuousCDF, StudentsT};


pub type ArrayFloat64Dim1 = ArrayBase<OwnedRepr<f64>, Ix1>;
pub type ArrayFloat64Dim2 = ArrayBase<OwnedRepr<f64>, Ix2>;

pub trait Results {
    fn coef(&self) -> Result<ArrayFloat64Dim2, AmitaError>;

    fn se(&self) -> Result<ArrayFloat64Dim1, AmitaError>;

    fn t(&self) -> Result<ArrayFloat64Dim2, AmitaError>;

    // fn p_vals(&self) -> Result<ArrayFloat64Dim2, AmitaError>;

    fn summary(&self) -> Result<String, AmitaError>;
}


#[derive(Debug, Clone)]
pub struct OLSResults {
    n_obs: usize,
    n_regressors: usize,

    // estimates
    coef: Option<ArrayFloat64Dim2>, // beta
    se: Option<ArrayFloat64Dim1>, // standard error of beta
    y_pred: Option<ArrayFloat64Dim2>, // predicted y, or fitted y
    resid: Option<ArrayFloat64Dim2>, // residuals, or the error term
    t: Option<ArrayFloat64Dim2>, // t statistics
    // p_vals: Option<ArrayFloat64Dim2>, // p values of H_0: \beta = 0

    // goodness of fit
    r_sq: Option<f64>,
    r_sq_adj: Option<f64>,
}

impl Results for OLSResults {
    fn coef(&self) -> Result<ArrayFloat64Dim2, AmitaError> {
        let coef = self.coef.clone().ok_or(AmitaError::NotSolved)?;
        // let coef = coef.into_dimensionality::<Ix1>().unwrap();
        Ok(coef)
        
    }

    fn se(&self) -> Result<ArrayFloat64Dim1, AmitaError> {
        self.se.clone().ok_or(AmitaError::NotSolved)
    }

    fn t(&self) -> Result<ArrayFloat64Dim2, AmitaError> {
        self.t.clone().ok_or(AmitaError::NotSolved)
    }

    // fn p_vals(&self) -> Result<ArrayFloat64Dim2, AmitaError> {
    //     self.p_vals.clone().ok_or(AmitaError::NotSolved)
    // }
    
    fn summary(&self) -> Result<String, AmitaError> {
        todo!()
    }
}

pub trait Solver<R>
where
    Self: Sized,
    R: Results
{
    fn results(&self) -> R;

    fn solve(self) -> Result<Self, AmitaError>;
}

#[derive(Debug, Clone)]
pub struct OLSSolver {
    y: ArrayFloat64Dim2,
    x: ArrayFloat64Dim2,

    q: ArrayFloat64Dim2,
    r: ArrayFloat64Dim2,

    results: OLSResults
}

impl Solver<OLSResults> for OLSSolver {
    fn results(&self) -> OLSResults {
        self.results.clone()
    }

    fn solve(self) -> Result<Self, AmitaError> {
        self
        .solve_coef()?
        .solve_se()
        // .solve_r_sq()
    }
}

// initializers
impl OLSSolver {
    pub fn new(
        y: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>>,
        x: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>>,
    ) -> Result<Self, AmitaError> {
        let _res = OLSSolver::validate_x_y(y, x)?;

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

            coef: None,
            se: None,
            y_pred: None,
            resid: None,
            t: None,
            // p_vals: None,

            r_sq: None,
            r_sq_adj: None,
        };

        Ok( OLSSolver { y, x, q, r, results } )
    }

    fn validate_x_y(
        y: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>>,
        x: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>>,
    ) -> Result<(), AmitaError> {
        if y.shape()[1] > 1 {
            return Err(AmitaError::ExpectedVector { matrix_name: "Target `y`".to_string() })
        }

        if x.shape()[0] != y.shape()[0] {
            return Err(AmitaError::NotSameObservations)
        }

        Ok(())
    }
}

// Estimate coefficients
impl OLSSolver {
    /// Estimate OLS' coefficients using QR-decomposed X
    /// \beta = R^{-1} Q^{\transpose} y
    fn solve_coef(mut self) -> Result<Self, AmitaError> {
        let y = self.y.clone();
        let x = self.x.clone();
        let r_inverse = 
            self.r.clone()
            .qr().map_err(|_| AmitaError::NotQRDecomposable { 
                matrix_name: "R matrix from QR-decomposed X".to_string() 
            })?
            .inverse().map_err(|_| AmitaError::NotInversable {
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
        self
        .solve_non_robust_se()
        // .solve_t_p()
    }

    fn solve_non_robust_se(mut self) -> Result<Self, AmitaError> {
        let n_obs = self.results.n_obs;
        let n_regressors = self.results.n_regressors;
        
        let r = self.r.clone();
        let resid = self.results.clone().resid.ok_or(AmitaError::NotSolved)?;

        let sigma = resid.t().dot(&resid)[[0,0]] / (n_obs - n_regressors) as f64;
        let variance = r.t().dot(&r)
            .qr().map_err(|_| AmitaError::NotQRDecomposable { 
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?
            .inverse().map_err(|_| AmitaError::NotInversable {
                matrix_name: "R^{\transpose}R matrix from QR-decomposed X".to_string() 
            })?;
        let se = variance.map(|x| (x * sigma).sqrt()).diag().to_owned();

        self.results.se = Some(se);

        Ok(self)
    }

    // fn solve_t_p(mut self) -> Result<Self, AmitaError> {
    //     // let df = (self.results.n_obs - self.results.n_regressors - 1) as f64;
    //     // let coef = self.results.coef.clone().ok_or(AmitaError::NotSolved)?;
    //     // let se = self.results.se.clone().ok_or(AmitaError::NotSolved)?;

    //     // let t = coef / se.view();

    //     // let t_dist = StudentsT::new(0., 1., df).unwrap();
    //     // let p_vals = t.map(|x| t_dist.cdf(x / 2 as f64));

    //     // self.results.t = Some(t);
    //     // self.results.p_vals = Some(p_vals);

    //     // Ok(self)
    //     todo!()
    // }
}

// Goodness of fit
impl OLSSolver {
    fn solve_r_sq(mut self) -> Result<Self, AmitaError> {
        let resid = self.results.resid.clone().ok_or(AmitaError::NotSolved)?;
        let y = self.y.clone();
        let y_mean = y.mean().unwrap();

        let df_rss = (self.results.n_regressors - self.results.n_obs - 1) as f64;
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;


    #[test]
    fn test_dimension() {
        let mut arr = array![[1., 2., 3.], [4., 5., 6.,]];
        let res = arr.view_mut().into_shape((6,1)).unwrap().into_dimensionality::<Ix1>();
        println!("{:#?}", res);
    }
}