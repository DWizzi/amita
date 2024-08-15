
#[cfg(test)]
mod tests {
    use crate::datasets;
    use polars::prelude::*;


    #[test]
    fn test_df() {
        let data = datasets::iris();

        let schema = data.schema();
        println!("{:#?}", schema.get("variety"));

        println!("{:#?}", data.head(Some(3)));

        let uniques = 
            data
            .clone()
            .lazy()
            .group_by([col("variety")])
            .agg([
                col("variety").count().alias("count")
            ])
            .collect()
            .unwrap();

        let _single_obs_within_cluster = 
            uniques
            .clone()
            .lazy()
            .filter(col("count").lt_eq(1))
            .collect()
            .unwrap()
            .shape().0 > 0;

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
                ["variety"],
                ["variety"],
                JoinArgs::new(JoinType::Left)
            )
            .unwrap();

        println!("{:#?}", data);

        let uniques = data.select(["tag"]).unwrap().to_ndarray::<Int32Type>(IndexOrder::C).unwrap().into_shape((data.shape().0, )).unwrap();
        println!("{:#?}", uniques);
    }
}