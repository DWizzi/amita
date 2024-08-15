//! Provides utilities for statistical inference

use amita_error::AmitaError;
use ndarray::Array1;
use polars::frame::DataFrame;
use polars::prelude::DataFrameJoinOps;
use polars::prelude::DataType;
use polars::prelude::IndexOrder;
use polars::prelude::Int32Type;
use polars::prelude::IntoLazy;
// use polars::prelude::*;
use polars::prelude::col;
use polars::prelude::JoinArgs;
use polars::prelude::JoinType;
use polars::prelude::Literal;
use polars::series::Series;

#[derive(Debug)]
pub enum SolverSEType {
    Homoscedastic,
    HC1,
    HC2,
    HC3,
    Clustered { by: Array1<i32> },

    NonRobust, // alias for SolverSEType::Homoscedastic
    Robust, // alias for SolverSEType::HC3
}

#[derive(Debug)]
pub enum ModelSEType {
    Homoscedastic,
    HC1,
    HC2,
    HC3,
    Clustered { by: String },

    NonRobust, // alias for ModelSEType::Homoscedastic
    Robust, // alias for ModelSEType::HC3
}

impl ModelSEType {
    pub fn to_solver_se_type(
        self, 
        data: &DataFrame
    ) -> Result<SolverSEType, AmitaError> {
        match self {
            ModelSEType::Homoscedastic => Ok( SolverSEType::Homoscedastic ),
            ModelSEType::HC1 => Ok( SolverSEType::HC1 ),
            ModelSEType::HC2 => Ok( SolverSEType::HC2 ),
            ModelSEType::HC3 => Ok( SolverSEType::HC3 ),
            ModelSEType::NonRobust => Ok( SolverSEType::NonRobust ),
            ModelSEType::Robust => Ok( SolverSEType::Robust ),

            ModelSEType::Clustered { by } => {
                let arr = 
                    Self::cluster_col_name_into_array(&by, data)?;
                Ok( SolverSEType::Clustered { by: arr } )
            },
        }
    }

    fn cluster_col_name_into_array(
        column: &str, 
        data: &DataFrame
    ) -> Result<Array1<i32>, AmitaError> {
        Self::validate_cluster_col_type(column, data)?;

        let uniques = 
            data
            .clone()
            .lazy()
            .group_by([col(column)])
            .agg([
                col(column).count().alias("count")
            ])
            .collect()
            .unwrap();

        let is_single_obs_within_cluster = 
            uniques
            .clone()
            .lazy()
            .filter(col("count").lt_eq(1))
            .collect()
            .unwrap()
            .shape().0 > 0;

        if is_single_obs_within_cluster {
            return Err( AmitaError::SingleObservationWithinCluster { 
                cluster: column.to_string() 
            } )
        }

        let tag = (0..uniques.height()).into_iter().map(|x| x as i32).collect::<Series>();
        let tag = 
            uniques
            .clone()
            .lazy()
            .with_column(tag.lit().alias("tag"))
            .collect()
            .unwrap();

        let data = 
            data
            .join(
                &tag,
                [column],
                [column],
                JoinArgs::new(JoinType::Left)
            )
            .unwrap();

        let uniques = 
            data
            .select(["tag"])
            .unwrap()
            .to_ndarray::<Int32Type>(IndexOrder::C)
            .unwrap()
            .into_shape((data.shape().0, ))
            .unwrap();

        Ok( uniques )
    }

    fn validate_cluster_col_type(
        column: &str, 
        data: &DataFrame,
    ) -> Result<(), AmitaError> {
        let schema = data.schema();

        let dtype = schema
            .get(column)
            .ok_or(AmitaError::ColumnNotFound { 
                column: column.to_string() 
            })?;

        match dtype.clone() {
            DataType::Boolean => (),
            DataType::UInt8=> (),
            DataType::UInt16=> (),
            DataType::UInt32=> (),
            DataType::UInt64=> (),
            DataType::Int8=> (),
            DataType::Int16=> (),
            DataType::Int32=> (),
            DataType::Int64=> (),
            DataType::String=> (),
            _ => return Err( AmitaError::ColumnDataTypeError { 
                column: column.to_string(), 
                expected: "bool, int, or string".to_string(), 
                found: dtype.to_string(),
            } )
        }

        Ok(())
    }
}

