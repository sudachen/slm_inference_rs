use std::os::raw::c_int;
use std::string::FromUtf8Error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    #[error("Decode Error: NoKvCacheSlot")]
    NoKvCacheSlot,
    #[error("Decode Error: no tokens")]
    NTokensZero,
    #[error("Decode Error:  {0}")]
    Unknown(c_int),
}

impl From<i32> for DecodeError {
    fn from(value: i32) -> Self {
        match value {
            1 => DecodeError::NoKvCacheSlot,
            -1 => DecodeError::NTokensZero,
            i => DecodeError::Unknown(i),
        }
    }
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("Sampling Error: {0}")]
    Error(String),
    #[error("Stop")]
    Stop,
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenToStringError {
    #[error("Unknown Token Type")]
    UnknownTokenType,
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("Invalid Lstrip")]
    InvalidLstrip,
    #[error("FfiError {0}")]
    FfiError(#[from] FfiError),
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StringToTokenError {
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("FfiError {0}")]
    FfiError(#[from] FfiError),
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum InferenceError {
    #[error("Error {0}")]
    Error(String),
    #[error("Batch Error {0}")]
    BatchError(#[from] BatchError),
    #[error("Ffi Error {0}")]
    FfiError(#[from] FfiError),
    #[error("StringToTokenError {0}")]
    StringToTokenError(#[from] StringToTokenError),
    #[error("TokenToStringError {0}")]
    TokenToStringError(#[from] TokenToStringError),
    #[error("DecodeError {0}")]
    DecodeError(#[from] DecodeError),
    #[error("SamplingError {0}")]
    SamplingError(#[from] SamplingError),
    #[error("SnapshotError {0}")]
    ContextError(#[from] ContextError),
    #[error("Invalid Role")]
    InvalidRole,
    #[error("Unsupported Feature")]
    Unsupported,
    #[error("Empty Batch")]
    EmptyBatch,
    #[error("Incomplete Answer")]
    IncompleteAnswer,
    #[error("Invalid JSON")]
    InvalidJson,
    #[error("Invalid JSON Schema")]
    InvalidJsonSchema(String),
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GgufLoaderError {
    #[error("Invalid Path")]
    InvalidPath,
    #[error("Bad model")]
    BadModel,
    #[error("Ffi Error {0}")]
    FfiError(#[from] FfiError),
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchError {
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    #[error("Empty buffer")]
    EmptyBuffer,
    #[error("n_tokens {0} is too large for a batch")]
    NtokTooLarge(usize),
    #[error("n_seq_max {0} is too large for a batch")]
    NseqTooLarge(usize),
    #[error("Internal error {0}")]
    InternalError(String),
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum FfiError {
    #[error("Null Ptr")]
    NullPtr,
    #[error("Cstring Allocation Error")]
    CstAllocationError,
    #[error("C_int Conversion Error")]
    CintConversionError,
    #[error("Error {0}")]
    Error(String),
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ContextBuilderError {
    #[error("Ffi Error {0}")]
    FfiError(#[from] FfiError),
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ContextError {
    #[error("Error {0}")]
    Error(String),
    #[error("Ffi Error {0}")]
    FfiError(#[from] FfiError),
    #[error("Position Not Found")]
    PosNotFound,
    #[error("Unsupported Feature")]
    Unsupported,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ModelFormatterError {
    #[error("Unknown model formatter {0}")]
    UnknownModelFormatter(String),
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum GuidanceError {
    #[error("Error {0}")]
    Error(String),
    #[error("Ffi Error {0}")]
    FfiError(#[from] FfiError),
}