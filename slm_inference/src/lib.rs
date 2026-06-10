pub mod hf;
pub use hf::SlmHfModel;
pub mod core;
pub mod errors;
pub mod inference;
mod chat;
mod answer;
mod forkable;
mod formatter;
pub mod models;

use std::path::Path;
use std::result::Result;

use errors::*;
pub use inference::{SlmInference, SlmSimpleInference};
pub use forkable::{SlmForkableInference, SlmParallelInference};
pub use chat::{SlmChat, SlmSimpleChat};
pub use answer::{SlmAnswer, SlmBrake, SlmBrakeFilter};
pub use formatter::SlmFormatter;
pub use models::SlmDynamicFormatter;

pub trait SlmToken: Copy {
    fn as_i32(&self) -> i32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlmRole {
    System,
    User,
    Assistant,
    Tool(String),
}

impl SlmRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            SlmRole::System => "system",
            SlmRole::User => "user",
            SlmRole::Assistant => "assistant",
            SlmRole::Tool(_) => "tool",
        }
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, SlmRole::Tool(_))
    }

    pub fn tool_name(&self) -> Option<&str> {
        match self {
            SlmRole::Tool(name) => Some(name.as_str()),
            _ => None,
        }
    }

    pub fn tool(name: &str) -> SlmRole {
        SlmRole::Tool(name.to_string())
    }
}

pub trait SlmBatch<T: SlmToken> {
    fn add(
        &mut self,
        token: T,
        pos: usize,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchError>;
    fn clear(&mut self);
    fn n_tokens(&self) -> usize;
    fn n_max(&self) -> usize;
}

pub trait SlmModelConfig {
    type Context: SlmContext;
    type Model: SlmModel<Context = Self::Context>;
    fn load_gguf(self, path: impl AsRef<Path>) -> Result<Self::Model, GgufLoaderError>;
}

pub trait SlmModel {
    type Context: SlmContext;
    fn context(&self) -> impl SlmContextBuilder<Self::Context>;
}

pub trait SlmContextBuilder<T> {
    fn build(self) -> Result<T, ContextBuilderError>;
    fn with_sampler(self, temperature: f32, top_k: i32, top_p: f32) -> Self;
}

pub trait SlmContext {
    type Token: SlmToken;
    type Batch: SlmBatch<Self::Token>;
    type Snapshot;
    fn new_batch(&self, tokens: usize, sequences: usize) -> Result<Self::Batch, BatchError>;
    fn max_batch_len(&self) -> usize;
    fn decode(&mut self, batch: &mut Self::Batch) -> Result<(), DecodeError>;
    fn sample(&mut self, logit_idx: usize) -> Result<Option<Self::Token>, SamplingError>;
    fn token_to_bytes(
        &self,
        token: Self::Token,
        buffer_size: usize,
        special: bool,
        lstrip: Option<usize>,
    ) -> Result<Vec<u8>, TokenToStringError>;
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Self::Token>, StringToTokenError>;
    fn save(&mut self, n_pos: usize, seq_id: Option<i32>) -> Result<Self::Snapshot, ContextError>;
    fn rollback(&mut self, snapshot: &Self::Snapshot) -> Result<usize, ContextError>;
    fn clear(&mut self) -> Result<usize, ContextError>;
    fn format(&self, parts: &[(SlmRole, &str)], ask: bool) -> Result<String, ContextError>;
}

pub trait SlmRollback {
    fn save(&mut self) -> Result<(),InferenceError>;
    fn rollback(&mut self) -> Result<(),InferenceError>;
}

