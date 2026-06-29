//! Small Language Model (SLM) inference library.
//!
//! This crate provides a model-agnostic abstraction layer for running autoregressive
//! text generation with small language models. It includes:
//!
//! - [`slm`] - Core inference primitives (Context, Vocab, Formatter, Assistant)
//! - [`models`] - Model-specific chat-template formatters
//! - [`core`] - Low-level utilities (RAII smart pointers for FFI)

pub mod slm;
pub mod models;
pub mod core;