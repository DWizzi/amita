use std::collections::HashSet;

use amita_error::AmitaError;
use amita_universal::math::sigmoid;
use amita_universal::traits::Solver;
use amita_universal::traits::SolverResults;
// use argmin::core::Hessian;
use argmin::core::State;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Axis;

use argmin::core::{CostFunction, Error, Executor, Gradient, Operator};
use argmin::solver::linesearch::MoreThuenteLineSearch;

use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct LogitResults {
    n_obs: usize,
    n_regressors: usize,

    coef: Option<Array1<f64>>,
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
        };

        let y = y.clone().map(|x| *x as f64);

        Ok( Self {
            y: y,
            x: x.clone(),

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
        self.solve_coef()
    }
}

impl LogitSolver {
    fn solve_coef(mut self) -> Result<Self, AmitaError> {
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

        let coef = res.state()
            .get_param()
            .unwrap(); // error handling needed here

        self.results.coef = Some( coef.clone() );

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

// impl Hessian for LogitSolver {
//     type Param = Array1<f64>;
//     type Hessian = Array2<f64>;

//     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
//         todo!()
//     }
// }

impl Operator for LogitSolver {
    type Param = Array1<f64>;
    type Output = Array1<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok( self.x.dot(param) )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn miscellaneous() {
        let x = array![
            [1.0, 3.0 ,],
            [4.0, 11.0 ,],
        ];
        let y = array![3.0, 4.0, ];
        let p = array![0.1, -0.1, ];

        println!("{:#?}", y.clone() + p);
        println!("{:#?}", y.clone() * x);
    }

    #[test]
    fn test_logit_solver() -> Result<(), AmitaError> {
        let x = array![
            [1., 1., 1., 1., 1.,],
            [3.1, 13.2, -23.5, -4.4, 9.4],
        ].t().to_owned();

        let y = array![0, 1, 1, 0, 1];

        let logit_solver = LogitSolver::new(&y, &x)?;
        let res = logit_solver.solve_coef();
        println!("{:#?}", res);

        Ok(())
    }
}


