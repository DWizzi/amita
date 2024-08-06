use polars::prelude::*;


pub fn iris() -> DataFrame {
    LazyCsvReader::new("iris.csv")
        .finish()
        .unwrap()
        .collect()
        .unwrap()
}