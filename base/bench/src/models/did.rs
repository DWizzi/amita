
#[cfg(test)]
mod tests {
    use amita_models::did::twfe::TWFE;

    use crate::datasets::banks;


    #[test]
    fn test_twfe() {
        let data = banks();
        let twfe = TWFE::new(&data, "bib", "treat", "post", None);
        let results = twfe.fit();
        println!("{:#?}", results);
    }

}