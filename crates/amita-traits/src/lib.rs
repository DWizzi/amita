use ndarray::Array1;

use amita_error::AmitaError;

pub trait SolverResults {
    fn coef(&self) -> Result<Array1<f64>, AmitaError>;

    fn se(&self) -> Result<Array1<f64>, AmitaError>;

    fn t(&self) -> Result<Array1<f64>, AmitaError>;

    fn p_vals(&self) -> Result<Array1<f64>, AmitaError>;

    fn summary(&self) -> Result<String, AmitaError>;
}

pub trait Solver<R> 
where
    Self: Sized,
    R: SolverResults,
{
    fn results(&self) -> R;

    fn solve(self) -> Result<Self, AmitaError>;
}