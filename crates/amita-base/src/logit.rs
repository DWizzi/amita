use std::f64::consts::E;

use argmin::core::{CostFunction, Gradient, Hessian, Jacobian};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};


#[derive(Debug, Clone)]
struct LogitProblem {}

// impl CostFunction for LogitProblem {
//     type Param = Array1<f64>;

//     type Output = f64;

//     fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
//         todo!()
//     }
// }

// impl Gradient for LogitProblem {
//     type Param;

//     type Gradient;

//     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
//         todo!()
//     }
// }

// impl Jacobian for LogitProblem {
//     type Param;

//     type Jacobian;

//     fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, argmin::core::Error> {
//         todo!()
//     }
// }

// impl Hessian for LogitProblem {
//     type Param;

//     type Hessian;

//     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
//         todo!()
//     }
// }


#[derive(Debug, Clone)]
pub struct Logit {
    y: Array1<i32>,
    x: Array2<f64>,

    coefficients: Option<Array1<f64>>,
    standard_errors: Option<Array1<f64>>,
}

/// Auxiliary functions
/// warning: the following function assumes valid y and x's
impl Logit {
    fn logit(x: ArrayView1<f64>, coef: ArrayView1<f64>) -> f64 {
        1. / (1. + E.powf(-x.dot(&coef)))
    }

    fn logit_log_likelihood(
        y: ArrayView1<i32>,
        x: ArrayView2<f64>,
        coefficients: ArrayView1<f64>
    ) -> f64 {
        let mut ll = 0.;
        for idx in 0..y.len() {
            let y_row = y[idx] as f64;
            let x_row = x.index_axis(Axis(0), idx);
            let p_row = Self::logit(x_row, coefficients);
            
            let ll_row = y_row * p_row.log(E) + (1.-y_row) * (1.-p_row).log(E);
            ll += ll_row;
        }
        ll
    }

    fn logit_loss(
        y: ArrayView1<i32>,
        x: ArrayView2<f64>,
        coefficients: ArrayView1<f64>
    ) -> f64 {
        let ll = Self::logit_log_likelihood(y, x, coefficients);
        let loss = -ll / y.len() as f64;
        loss
    }

    fn logit_gradient(
        y: ArrayView1<i32>,
        x: ArrayView2<f64>,
        coefficients: ArrayView1<f64>
    ) -> Array1<f64> {
        let mut grad = Array1::<f64>::zeros(y.len());
        for idx in 0..y.len() {
            let y_row = y[idx];
            let x_row = x.index_axis(Axis(0), idx);
            let p_row = Self::logit(x_row, coefficients);
            
            let grad_row = (p_row - y_row as f64) / (y.len() as f64) * x_row.to_owned();
            grad = grad + grad_row;
        }
        grad
    }


}

impl Logit {
    pub fn new(y: ArrayView1<i32>, x: ArrayView2<f64>) -> Self {
        assert!(
            y.iter().all(|&x| x==0 || x==1),
            "y must be binary"
        );

        assert_eq!(
            y.len(), x.nrows(), 
            "y and x must have same number of observations"
        );

        assert!(
            x.nrows() > x.ncols(),
            "Unidentifiable because the number of observations is less than regressos"
        );

        Logit {
            y: y.to_owned(),
            x: x.to_owned(),

            coefficients: None,
            standard_errors: None,
        }
    }
}

impl Logit {
    // pub fn fit(mut self) -> Self {

    // }

    // fn calculate_coefficients(mut self) -> Self {

    // }
}



#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn test_hashset() {
        let arr = array![1, 1, 0 ,1, 1];
        let is_valid = arr.iter().all(|&x| x == 0 || x==1);

        println!("{:#?}", is_valid);
    }
}