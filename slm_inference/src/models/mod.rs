//! Model-specific chat-template formatters.
//!
//! This module provides [`Formatter`] implementations for various language model families:
//!
//! - [`GemmaFormatter`] for Google Gemma 4 models
//! - [`Llama3Formatter`] for Meta Llama 3 models
//! - [`MistralFormatter`] for Mistral models
//! - [`Phi4Formatter`] for Microsoft Phi-4 models
//! - [`Qwen25Formatter`] for Alibaba Qwen 2.5 models

mod gemma4;
mod llama3;
mod mistal;
mod phi4;
mod qwen25;

pub use gemma4::{Gemma4Flavor, GemmaFormatter};
pub use llama3::Llama3Formatter;
pub use mistal::{MistralFlavor, MistralFormatter};
pub use phi4::Phi4Formatter;
pub use qwen25::Qwen25Formatter;
