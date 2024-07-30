use std::f64::consts::E;

use amita_error::AmitaError;
use linfa_linalg::qr::QR;
use ndarray::{ArrayBase, Dim, Ix1, Ix2, OwnedRepr, ViewRepr};


pub type ArrayFloat64Dim1 = ArrayBase<OwnedRepr<f64>, Ix1>;
pub type ArrayFloat64Dim2 = ArrayBase<OwnedRepr<f64>, Ix2>;

pub struct LogitResults {

}

pub struct LogitSolver {
    y: ArrayFloat64Dim2,
    x: ArrayFloat64Dim2,

    results: LogitResults,
}

impl LogitSolver {
    fn sigmoid(x: f64) -> f64 {
        1. / (1. + E.powf(-x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        println!("{}, {}", 0, LogitSolver::sigmoid(0.));
        println!("{}, {}", 2, LogitSolver::sigmoid(2.));
        println!("{}, {}", "inf", LogitSolver::sigmoid(f64::INFINITY));
        println!("{}, {}", "-inf", LogitSolver::sigmoid(f64::NEG_INFINITY));
    }
}