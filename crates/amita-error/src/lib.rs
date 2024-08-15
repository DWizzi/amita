
#[derive(Debug, Clone, thiserror::Error)]
pub enum AmitaError {
    // Data Inputs
    #[error("Matrices consist of a different number of observations")]
    NotSameObservations,
    #[error("{matrix_name:?} is not QR-decomposable")]
    NotQRDecomposable { matrix_name: String },
    #[error("{matrix_name:?} is not invertible")]
    NotInvertible { matrix_name: String },
    #[error("Elements in {matrix_name:?} are non-binary")]
    NonBinary { matrix_name: String },
    #[error("Cluster {cluster:?} contains only 1 observation")]
    SingleObservationWithinCluster { cluster: String },

    // Solver
    #[error("Solver is not solved")]
    NotSolved,

    // Model
    #[error("Model is not fitted")]
    NotFittedModel,

    // DataFrame
    #[error("Column {column:?} not found")]
    ColumnNotFound { column: String },
    #[error("Column {column:?} contains unexpected datatype. 
Expected {expected:?}, found {found:?}.")]
    ColumnDataTypeError { column: String, expected: String, found: String },
}