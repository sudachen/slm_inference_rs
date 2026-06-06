pub mod hf;
pub use hf::HfModelInfo;
pub mod core;
pub mod errors;
pub mod inference;

use std::path::Path;
use std::result::Result;

pub use crate::errors::*;
pub use crate::inference::SlmInference;

pub trait SlmToken: Copy {
    fn as_i32(&self) -> i32;
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
}
