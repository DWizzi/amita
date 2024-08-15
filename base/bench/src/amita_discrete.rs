#[cfg(test)]
mod tests {
    use amita_error::AmitaError;
    use amita_utils::traits::Solver;
    use amita_discrete::logit::LogitSolver;
    use polars::prelude::*;

    use crate::datasets;

    #[test]
    fn test_solver_coef() -> Result<(), AmitaError> {
        let df = datasets::wine_quality();
        let n_obs = df.height();

        let mut c = vec![];
        for _ in 0..n_obs {
            c.push(1.);
        }
        let c = Series::new("_const", c);
        let df = df.lazy().with_column(c.lit()).collect().unwrap();


        let x = df
            .select(["_const", "fixed_acidity", "alcohol"])
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::C)
            .unwrap();
        let y = df
            .select(["is_good"])
            .unwrap()
            .to_ndarray::<Int32Type>(IndexOrder::C)
            .unwrap()
            .into_shape((n_obs, ))
            .unwrap();

        let logit = LogitSolver::new(&y, &x)?;
        let logit = logit.solve()?;

        println!("{:#?}", logit);

        Ok(())
    }
}