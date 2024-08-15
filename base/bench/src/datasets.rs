use polars::prelude::*;


pub fn iris() -> DataFrame {
    LazyCsvReader::new("iris.csv")
        .finish()
        .unwrap()
        .collect()
        .unwrap()
}

pub fn wine_quality() -> DataFrame {
    LazyCsvReader::new("wine_quality.csv")
        .finish()
        .unwrap()
        .with_column(
            when(col("quality").gt_eq(6))
            .then(1)
            .otherwise(0)
            .alias("is_good")
        )
        .collect()
        .unwrap()
}

pub fn banks() -> DataFrame {
    LazyCsvReader::new("banks.csv")
        .finish()
        .unwrap()
        .collect()
        .unwrap()
}