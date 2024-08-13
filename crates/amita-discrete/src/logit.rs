use std::collections::HashSet;

use amita_error::AmitaError;
use amita_universal::math::sigmoid;
use amita_universal::traits::{Solver, SolverResults};
use linfa_linalg::qr::QR;
use ndarray::prelude::*;

use argmin::core::{CostFunction, Error, Executor, Gradient, Operator, IterState, Hessian, State};
use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};

use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct LogitResults {
    n_obs: usize,
    n_regressors: usize,

    coef: Option<Array1<f64>>,
    se: Option<Array1<f64>>,
    t: Option<Array1<f64>>,
}

impl SolverResults for LogitResults {
    fn coef(&self) -> Result<Array1<f64>, AmitaError> {
        self.coef.clone().ok_or(AmitaError::NotSolved)
    }

    fn se(&self) -> Result<Array1<f64>, AmitaError> {
        todo!()
    }

    fn t(&self) -> Result<Array1<f64>, AmitaError> {
        todo!()
    }

    fn p_vals(&self) -> Result<Array1<f64>, AmitaError> {
        todo!()
    }

    fn summary(&self) -> Result<String, AmitaError> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct LogitSolver {
    y: Array1<f64>,
    x: Array2<f64>,
    hessian: Option<Array2<f64>>,
    iter_state: Option<IterState<Array1<f64>, Array1<f64>, (), (), (), f64, >>,

    max_iter: u64,
    max_tolerance: f64,

    results: LogitResults,
}

impl LogitSolver {
    pub fn new(
        y: &Array1<i32>,
        x: &Array2<f64>,
    ) -> Result<Self, AmitaError> {
        let _res = LogitSolver::validate_data(y, x)?;

        let n_obs = x.shape()[0];
        let n_regressors = x.shape()[1];

        let results = LogitResults {
            n_obs,
            n_regressors,

            coef: None,
            se: None,
            t: None,
        };

        let y = y.clone().map(|x| *x as f64);

        Ok( Self {
            y: y,
            x: x.clone(),
            hessian: None,
            iter_state: None,

            max_iter: 1_000,
            max_tolerance: 0.0001,

            results: results,
        })
    }

    pub fn with_max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_max_tolerance(mut self, max_tolerance: f64) -> Self {
        self.max_tolerance = max_tolerance;
        self
    }

    fn validate_data(
        y: &Array1<i32>,
        x: &Array2<f64>,
    ) -> Result<(), AmitaError> {
        if x.shape()[0] != y.shape()[0] {
            return Err(AmitaError::NotSameObservations);
        }

        let mut y_allowed = HashSet::new();
        y_allowed.insert(0 as i32);
        y_allowed.insert(1 as i32);

        let y_unique = y.into_iter()
            .map(|x| *x)
            .collect::<HashSet<i32>>();

        if y_unique != y_allowed {
            return Err(AmitaError::NonBinary { matrix_name: "`y` of Logit model".to_string() });
        }

        Ok(())
    }
}

impl Solver<LogitResults> for LogitSolver {
    fn results(&self) -> LogitResults {
        todo!()
    }

    fn solve(self) -> Result<Self, AmitaError> {
        self
        .run_solver()?
        .solve_coef()?
        .solve_hessian()?
        .solve_se()?
        .solve_t()
    }
}

impl LogitSolver {
    fn run_solver(mut self) ->Result<Self, AmitaError> {
        let init_param = Array1::zeros((self.results.n_regressors, ));

        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 10)
            .with_tolerance_grad(self.max_tolerance)
            .unwrap();

        let res = Executor::new(self.clone(), solver)
            .configure(|state| 
                state
                    .param(init_param)
                    .max_iters(self.max_iter)
                    .target_cost(0.0)
            )
            .run()
            .unwrap();

        self.iter_state = Some(res.state.clone());

        Ok(self)   
    }

    fn solve_coef(mut self) -> Result<Self, AmitaError> {
        let iter_state = self.iter_state.clone().ok_or(AmitaError::NotSolved)?;
        self.results.coef = iter_state.get_param().map(|x| x.clone());

        Ok(self)
    }

    fn solve_hessian(mut self) -> Result<Self, AmitaError> {
        let coef = self.results.coef.clone().ok_or(AmitaError::NotSolved)?;
        let n_regressors = self.results.n_regressors;
        let mut hessian_elements = vec![];
        for i in 0..n_regressors {
            for j in 0..n_regressors {
                hessian_elements.push(self.second_order_derivative(&coef, i, j)?);
            }
        }

        let hessian = Array2::from_shape_vec((n_regressors, n_regressors), hessian_elements).unwrap();
        self.hessian = Some(hessian);
        Ok( self )
    }

    fn solve_se(self) -> Result<Self, AmitaError> {
        self.solve_non_robust_se()
    }

    fn solve_t(mut self) -> Result<Self, AmitaError> {
        let mut coef = self.results.coef.clone().ok_or(AmitaError::NotSolved)?;
        let se = self.results.se.clone().ok_or(AmitaError::NotSolved)?;

        coef.zip_mut_with(&se, |beta, se| *beta = *beta / *se);

        self.results.t = Some(coef);
        Ok( self )
    }

    fn solve_non_robust_se(mut self) -> Result<Self, AmitaError> {
        let n_obs = self.results.n_obs as f64;
        let hessian = self.hessian.clone().ok_or(AmitaError::NotSolved)?;

        let information_matrix = hessian;
        let var_cov = information_matrix.qr().unwrap().inverse().unwrap();

        let se = var_cov.diag().map(|x| x.sqrt() / n_obs.sqrt());
        self.results.se = Some(se);

        Ok(self)
    }
}

impl CostFunction for LogitSolver {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let p = self.x.map_axis(Axis(1), 
            |row| sigmoid(row.dot(param))
        );

        let pos = &self.y * p.map(|x| x.ln());
        let neg = &self.y.map(|x| 1. - x) * p.map(|x| (1. - x).ln());
        let cost = - (pos + neg).mean().unwrap();

        Ok( cost )
    }
}

impl Gradient for LogitSolver {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let p = self.x.map_axis(Axis(1), |row| {
            sigmoid(row.dot(param))
        }).into_shape((self.results.n_obs, )).unwrap();

        let px = (&p * &self.x.t())
            .map_axis(Axis(1), |row| row.mean().unwrap());

        let yx = (&self.y * &self.x.t())
            .map_axis(Axis(1), |row| row.mean().unwrap());

        Ok(px - yx)
    }
}

impl Hessian for LogitSolver {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let n_regressors = self.results.n_regressors;
        let mut hessian_elements = vec![];
        for i in 0..n_regressors {
            for j in 0..n_regressors {
                hessian_elements.push(self.second_order_derivative(param, i, j)?);
            }
        }
        Ok(
            Array2::from_shape_vec((n_regressors, n_regressors), hessian_elements).unwrap()
        )
    }
}

impl Operator for LogitSolver {
    type Param = Array1<f64>;
    type Output = Array1<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok( self.x.dot(param) )
    }
}

impl LogitSolver {
    /// i: the i-th regressor, or the i-th row of the hessian matrix
    /// j: the j-th regressor, or the j-th row of the hessian matrix
    fn second_order_derivative(
        &self, param: &Array1<f64>, i: usize, j: usize
    ) -> Result<f64, AmitaError>  {
        let p = self.x.map_axis(Axis(1), 
            |row| sigmoid(row.dot(param))
        );
        let x_i = self.x.slice(s![.., i]).clone();
        let x_j = self.x.slice(s![.., j]).clone();

        Ok( (&p * p.map(|x| (1. - x)) * x_i * x_j).mean().unwrap() )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray::prelude::*;

    use super::*;

    #[test]
    fn miscellaneous() {
        let x = array![
            [1.0, 3.0 ,],
            [4.0, 11.0 ,],
        ];

        println!("{:#?}", x.slice(s![.., 1]));

        let k = vec![
            1., 2., 3., 4.,
        ];
        let k = Array2::from_shape_vec((2,2), k);
        println!("{:#?}", k);
    }

    #[test]
    fn test_logit_solver() -> Result<(), AmitaError> {
        let x = array![
            [1., 1., 1., 1., 1.,],
            [3.1, 13.2, -23.5, -4.4, 9.4],
        ].t().to_owned();

        let y = array![0, 1, 1, 0, 1];

        let logit_solver = LogitSolver::new(&y, &x)?;
        let logit_solver = logit_solver.solve_coef()?;
        println!("{:#?}", logit_solver.results);
        println!("{:#?}", logit_solver.second_order_derivative(&array![1., 2.], 1, 1));

        Ok(())
    }

    #[test]
    fn test_hessian() -> Result<(), AmitaError> {
        let x = array![
            [1., 1., 1., 1., 1.,],
            [3.1, 13.2, -23.5, -4.4, 9.4],
        ].t().to_owned();

        let y = array![0, 1, 1, 0, 1];

        let logit_solver = LogitSolver::new(&y, &x).unwrap();
        let logit_solver = logit_solver.solve().unwrap();
        println!("{:#?}", logit_solver);

        let _coef = logit_solver.results.coef;
        let _hessian = logit_solver.hessian;

        // println!("{:#?}", coef);
        // println!("{:#?}", hessian);

        Ok(())
    }
}


