use std::collections::HashSet;

use amita_error::AmitaError;
use amita_universal::math::sigmoid;
use amita_universal::traits::Solver;
use amita_universal::traits::SolverResults;
use ndarray::Axis;

use argmin::core::{CostFunction, Error, Executor, Gradient, Operator};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
struct LogitResults {
    n_obs: usize,
    n_regressors: usize,
}

impl SolverResults for LogitResults {
    fn coef(&self) -> Result<Array1<f64>, AmitaError> {
        todo!()
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
struct LogitSolver {
    y: Array1<f64>,
    x: Array2<f64>,

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
        };

        let y = y.clone().map(|x| *x as f64);

        Ok( Self {
            y: y,
            x: x.clone(),
            results: results,
        })
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
            return Err(AmitaError::NotSolved); // TODO: new error definition needed here
        }

        Ok(())
    }
}

impl Solver<LogitResults> for LogitSolver {
    fn results(&self) -> LogitResults {
        todo!()
    }

    fn solve(self) -> Result<Self, AmitaError> {
        todo!()
    }
}

impl LogitSolver {
    fn solve_coef(mut self) -> Result<Self, AmitaError> {
        let init_param = Array1::zeros((self.results.n_obs, ));

        let linesearch = MoreThuenteLineSearch::new();
        let solver = SteepestDescent::new(linesearch);

        let res = Executor::new(self, solver)
            .configure(|state| 
                state
                    .param(init_param)
                    .max_iters(100)
                    // .target_cost(-1000.)
            )
            .run()
            .unwrap();

        todo!()
    }
}

impl CostFunction for LogitSolver {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let p = self.x.map_axis(Axis(1), |row| {
            sigmoid(row.dot(param))
        }).into_shape((self.results.n_obs, 1)).unwrap();

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
        }).into_shape((self.results.n_obs, 1)).unwrap();

        let px = (&p * &self.x).t()
            .map_axis(Axis(1), |row| row.sum());

        let yx = (&self.y * &self.x).t()
            .map_axis(Axis(1), |row| row.sum());

        Ok(yx - px)
    }
}

impl Operator for LogitSolver {
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
    fn miscellaneous() {
        let y_unique = vec![1 as i32, 0, 0, 1, 1, 1, 1, 0];
        let y_unique = y_unique.into_iter().collect::<HashSet<i32>>();

        let mut y_allowed = HashSet::new();
        y_allowed.insert(0 as i32);
        y_allowed.insert(1);

        assert!(y_unique == y_allowed)
    }
}


