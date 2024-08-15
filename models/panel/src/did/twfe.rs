use amita_base::linear::ols::{OLSResults, OLSSolver};
use amita_error::AmitaError;
use amita_utils::traits::BaseSolver;
use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct TWFE {
    data: DataFrame,
    outcome: String,
    treat: String,
    post: String,
    covariates: Option<Vec<String>>,
}

impl TWFE {
    pub fn new(
        data: &DataFrame,
        outcome: &'static str,
        treat: &'static str,
        post: &'static str,
        covarites: Option<Vec<String>>,
    ) -> TWFE {
        TWFE {
            data: data.clone(),
            outcome: outcome.to_string(),
            treat: treat.to_string(),
            post: post.to_string(),
            covariates: covarites,
        }
    }

    pub fn fit(&self) -> Result<OLSResults, AmitaError> {
        let solver = self.get_solver()?;
        let solver = solver.solve()?;
        Ok( solver.results() )
    }

    fn get_solver(&self) -> Result<OLSSolver, AmitaError> {
        let df = self.data.clone();
        let mut c = vec![];
        for _ in 0..df.height() {
            c.push(1.0);
        }
        let c = Series::new("_const", c);

        let mut columns = self.covariates.clone().unwrap_or(vec![]);
        columns.push(self.treat.clone());
        columns.push(self.post.clone());
        columns.push("treat*post".to_string());
        columns.push("_const".to_string());
        let columns = columns.iter().map(|x| {col(x)}).collect::<Vec<_>>();


        let regressors = df
            .clone()
            .lazy()
            .with_column(
                (col(&self.treat) * col(&self.post)).alias("treat*post")
            )
            .with_column(c.lit())
            .select(&columns)
            .collect()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::C)
            .unwrap();

        let outcome = df
            .clone()
            .select([&self.outcome])
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::C)
            .unwrap()
            .into_shape((df.height(),))
            .unwrap();

        let solver = OLSSolver::new(
            &outcome,
            &regressors,
        )?;

        // println!("{:#?}", solver);

        Ok(solver)

    }
}