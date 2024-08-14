
#[cfg(test)]
mod tests {
    use amita_error::AmitaError;
    use amita_linear::ols::OLSSolver;
    use amita_utils::traits::{Solver, SolverResults};
    use polars::prelude::*;

    use crate::datasets;

    #[test]
    fn test_solver_coef() -> Result<(), AmitaError> {
        let df = datasets::iris();
        let y = df
            .select(["petal.length"])
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::C)
            .unwrap()
            .into_shape((df.height(),))
            .unwrap();

        let mut c = vec![];
        for _ in 0..df.shape().0 {
            c.push(1.0);
        }
        let c = Series::new("_const", c);

        let x = df
            .lazy()
            .with_column(c.lit())
            .collect()
            .unwrap()
            .select(["_const", "sepal.length", "sepal.width"])
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::C)
            .unwrap();

        let solver = OLSSolver::new(&y, &x)?;
        let results = solver.solve()?.results();

        println!("{:#?}", results.se());
        println!("{:#?}", results);

        Ok(())
    }
}