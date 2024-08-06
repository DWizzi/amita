use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    1. / ( 1. + E.powf(-x) )
}