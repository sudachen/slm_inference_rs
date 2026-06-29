//! Low-level utilities shared across backend implementations.
//!
//! Currently exposes [`shared_ptr`], which provides RAII smart-pointer wrappers
//! (`UniquePtr`, `SharedPtr`) for raw C pointers obtained through FFI.
pub mod shared_ptr;
pub use shared_ptr::*;
