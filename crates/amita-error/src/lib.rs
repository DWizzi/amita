
#[derive(Debug, Clone, thiserror::Error)]
pub enum AmitaError {
    // Matrix
    #[error("Matrices consist of a different number of observations")]
    NotSameObservations,
    #[error("{matrix_name:?} is not QR-decomposable")]
    NotQRDecomposable { matrix_name: String },
    #[error("{matrix_name:?} is not inversable")]
    NotInversable { matrix_name: String },
    #[error("Elements in {matrix_name:?} are non-binary")]
    NonBinary { matrix_name: String },

    // Solver
    #[error("Solver is not solved")]
    NotSolved,

    // Model
    #[error("Model is not fitted")]
    NotFittedModel,
    #[error("Column {column:?} not found")]
    ColumnNotFound { column: String },
}