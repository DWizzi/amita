use polars::prelude::*;

mod amita_linear;

pub fn load_iris_dataset() -> DataFrame {
    LazyCsvReader::new("iris.csv")
        .finish()
        .unwrap()
        .collect()
        .unwrap()
}

fn main() {
    println!("Tests");
}
