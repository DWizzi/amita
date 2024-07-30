
#[derive(Debug, Clone, thiserror::Error)]
pub enum AmitaError {
    // Matrix
    #[error("{matrix_name:?} is expected to be a single-column, vector-like array")]
    ExpectedVector { matrix_name: String },
    #[error("Matrices consist of a different number of observations")]
    NotSameObservations,
    #[error("{matrix_name:?} is not QR-decomposable")]
    NotQRDecomposable { matrix_name: String },
    #[error("{matrix_name:?} is not inversable")]
    NotInversable { matrix_name: String },

    // Solver
    #[error("Solver is not solved")]
    NotSolved,

    // Model
    // #[error("Model is not fitted")]
    // NotFittedModel,
}