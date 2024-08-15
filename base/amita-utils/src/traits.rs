use ndarray::Array1;

use amita_error::AmitaError;

pub trait AmitaModel<R>
where
    Self: Sized,
    R: AmitaModelResults,
{
    fn fit(&self) -> Result<R, AmitaError>;
}

pub trait AmitaModelResults {
    fn summary(&self) -> String;
}


pub trait BaseSolver<R> 
where
    Self: Sized,
    R: BaseResults,
{
    fn results(&self) -> R;

    fn solve(self) -> Result<Self, AmitaError>;
}

pub trait BaseResults {
    fn coef(&self) -> Result<Array1<f64>, AmitaError>;

    fn se(&self) -> Result<Array1<f64>, AmitaError>;

    fn t(&self) -> Result<Array1<f64>, AmitaError>;

    fn p_vals(&self) -> Result<Array1<f64>, AmitaError>;

    fn summary(&self) -> Result<String, AmitaError>;
}