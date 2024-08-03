use std::f64::consts::E;

use amita_error::AmitaError;
use ndarray::{Axis};
use ndarray::prelude::*;

use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, Operator, State};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

use ndarray::{Array1, Array2};

struct Logit {
    y: Array2<f64>,
    x: Array2<f64>,

    n_obs: usize,
    n_regressors: usize,
} 

impl Logit {
    pub fn new(
        y: Array2<f64>,
        x: Array2<f64>,
    ) -> Result<Self, AmitaError> {
        let n_obs = x.shape()[0] as usize;
        let n_regressors = x.shape()[1] as usize;

        Ok( Self {
            y, x, n_obs, n_regressors,
        } )
    }

    fn sigmoid(x: f64) -> f64 {
        1. / (1. + E.powf(-x))
    }
}

impl CostFunction for Logit {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let p = self.x.map_axis(Axis(1), |row| {
            Self::sigmoid(row.dot(param))
        }).into_shape((self.n_obs, 1)).unwrap();

        let pos = &self.y * p.map(|x| x.ln());
        let neg = &self.y.map(|x| 1. - x) * p.map(|x| (1. - x).ln());
        let cost = - (pos + neg).mean().unwrap();

        Ok( cost )
    }
}

impl Gradient for Logit {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let p = self.x.map_axis(Axis(1), |row| {
            Self::sigmoid(row.dot(param))
        }).into_shape((self.n_obs, 1)).unwrap();

        

        let px = (&p * &self.x).t()
            .map_axis(Axis(1), |row| row.sum());

        let yx = (&self.y * &self.x).t()
            .map_axis(Axis(1), |row| row.sum());

        Ok(yx - px)
    }
}

impl Operator for Logit {
    type Param = Array1<f64>;
    type Output = Array1<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok( self.x.dot(param) )
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_argmin() {
        let y = array![
            [1.],
            [1.],
            [0.],
            [1.],
            [1.],
        ];

        let x = array![
            [1., 6.],
            [1., 3.1],
            [1., 2.1],
            [1., 5.1],
            [1., 1.1],
        ];

        // let init_param = array![
        //     [1.],
        //     [1.],
        // ];

        let init_param = array![-0.5, 0.5];

        let logit = Logit::new(y, x).unwrap();
        let cost = logit.cost(&init_param);
        let gradient = logit.gradient(&init_param);

        println!("{:#?}", cost);
        println!("{:#?}", gradient);

        let linesearch = MoreThuenteLineSearch::new();
        let solver = SteepestDescent::new(linesearch);

        let res = Executor::new(logit, solver)
            .configure(|state| 
                state
                    .param(init_param)
                    .max_iters(100)
                    // .target_cost(-1000.)
            )
            .run()
            .unwrap();

        println!("{res}");

        
    }
}