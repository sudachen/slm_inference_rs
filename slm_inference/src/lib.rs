pub mod hf;
pub use hf::SlmHfModel;
mod answer;
pub mod core;
pub mod errors;
mod formatter;
pub mod inference;
pub mod models;
mod oracle;
mod simple_oracle;
mod llg_lark;

pub use answer::SlmAnswer;
use errors::*;
pub use formatter::SlmFormatter;
pub use inference::{SlmBoxedBrakeFn, SlmBrake, SlmInference, SlmSimpleInference};
pub use models::SlmDynamicFormatter;
pub use oracle::{SlmOracle, SlmJsonOracle};
pub use simple_oracle::SlmSimpleOracle;
pub use llg_lark::SlmSimpleTokEnv;
use std::path::Path;
use std::result::Result;
use llguidance::toktrie;

pub trait SlmToken: Copy {
    fn as_i32(&self) -> i32;
    fn from_i32(i: i32) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlmRole {
    System,
    User,
    Assistant,
}

impl SlmRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            SlmRole::System => "system",
            SlmRole::User => "user",
            SlmRole::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlmPos {
    pub token_pos: usize,
    pub fork_id: usize,
}

#[allow(dead_code)]
impl SlmPos {
    fn fork_id(&self) -> usize {
        self.fork_id
    }
    fn token_pos(&self) -> usize {
        self.token_pos
    }
    pub fn new(token_pos: usize, fork_id: usize) -> SlmPos {
        Self { token_pos, fork_id }
    }
}

pub trait SlmBatch<Token: SlmToken> {
    fn add(&mut self, token: Token, pos: SlmPos, logits: bool) -> Result<(), BatchError>;
    fn clear(&mut self);
    fn n_tokens(&self) -> usize;
    fn n_max(&self) -> usize;
}

pub trait SlmModelConfig {
    type Context: SlmContext;
    type Model: SlmModel<Context = Self::Context>;
    fn load_gguf(self, path: impl AsRef<Path>) -> Result<Self::Model, GgufLoaderError>;
}

pub type SlmTokEnv = toktrie::TokEnv;

pub trait SlmVocab {
    type Token: SlmToken;
    fn vocab_size(&self) -> usize;
    fn token_to_bytes(
        &self,
        token: Self::Token,
        special: bool,
        left_strip: Option<usize>,
    ) -> Result<Vec<u8>, TokenToStringError>;
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Self::Token>, StringToTokenError>;
    fn tok_env(&self) -> &SlmTokEnv;
}

pub trait SlmModel {
    type Context: SlmContext;
    fn context(&self) -> impl SlmContextBuilder<Self::Context>;
}

pub enum SlmKvType {
    Q4,
    Q5,
    Q6,
    Q8,
    RawQ8,
    F16,
    F32,
}

pub trait SlmContextBuilder<T> {
    fn build(self) -> Result<T, ContextBuilderError>;
    fn with_sampler(self, temperature: f32, top_k: i32, top_p: f32) -> Self;
    fn with_n_ctx(self, n_ctx: usize) -> Self;
    fn with_gen_type_kv(self, k: SlmKvType, v: SlmKvType) -> Self;
    fn with_n_batch(self, n_batch: usize) -> Self;
    fn with_flash_attn(self, enable: bool) -> Self;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd)]
pub enum SlmEditLevel {
    #[default]
    DumpRestore = 0,
    Cut = 1,
    Truncate = 2,
}

#[derive(Debug, Clone)]
pub enum SlmConstraintStep {
    FastForward(Vec<i32>),
    Forward,
    Stop,
}

pub trait SlmConstraint {
    fn mask(&mut self, logits: &mut [f32]) -> Result<bool, SamplingError>;
    fn forward(&mut self, token_id: i32) -> Result<SlmConstraintStep, SamplingError>;
    fn prefill(&mut self, text: &str) -> Result<(), SamplingError>;
}

/// Core inference context that owns a KV cache and provides all token-level operations
/// required for autoregressive text generation.
///
/// A context is obtained from [`SlmContextBuilder::build`] and is used by inference
/// helpers such as [`SlmSimpleInference`].  Each context is associated with a single
/// model and maintains mutable state (KV cache, sampler state) across calls.
pub trait SlmContext {
    /// The token type produced and consumed by this context.
    type Token: SlmToken;
    /// The batch type used to submit tokens for decoding.
    type Batch: SlmBatch<Self::Token>;
    type Vocab: SlmVocab<Token = Self::Token>;
    fn vocab(&self) -> &Self::Vocab;

    /// Allocates a new batch capable of holding up to `tokens` token slots across
    /// up to `sequences` parallel sequences.
    fn new_batch(&self, tokens: usize, sequences: usize) -> Result<Self::Batch, BatchError>;

    /// Returns the maximum number of token slots the context can process in a single
    /// [`decode`](Self::decode) call.
    fn max_batch_len(&self) -> usize;

    /// Runs the model forward pass for all tokens currently queued in `batch`,
    /// updating the KV cache and computing logits for every slot that requested them.
    fn decode(&mut self, batch: &mut Self::Batch) -> Result<(), DecodeError>;

    /// Samples the next token from the logits stored at slot `logit_idx` of the most
    /// recently decoded batch.  Returns `None` when the model signals end-of-sequence
    /// via an EOS token.
    fn sample_with_constraint(
        &mut self,
        logit_idx: usize,
        constraint: Option<&mut dyn SlmConstraint>,
    ) -> Result<Option<Self::Token>, SamplingError>;

    /// Converts `token` to its raw byte representation.
    ///
    /// - `buffer_size` – internal scratch buffer size; increase if tokens can produce
    ///   many bytes.
    /// - `special` – when `true`, special tokens (BOS, EOS, …) are rendered as their
    ///   text representation rather than being skipped.
    /// - `lstrip` – if `Some(n)`, strip up to `n` leading space bytes from the result.
    fn token_to_bytes(
        &self,
        token: Self::Token,
        special: bool,
        left_strip: Option<usize>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        self.vocab().token_to_bytes(token, special, left_strip)
    }

    /// Tokenizes `str` into a sequence of model tokens.
    ///
    /// - `add_special` – prepend/append BOS/EOS markers as required by the model.
    /// - `parse_special` – treat special-token text representations (e.g. `<|im_start|>`)
    ///   as their corresponding token IDs rather than as plain text.
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Self::Token>, StringToTokenError> {
        self.vocab().str_to_tokens(str, add_special, parse_special)
    }

    /// Resets the context to an empty state, discarding the entire KV cache.
    fn clear(&mut self) -> Result<(), ContextError>;

    /// Discards all KV cache entries belonging to the sequence identified by
    /// `fork_id`, freeing the associated cache slots without affecting other
    /// sequences.  Equivalent to [`clear`](Self::clear) when only a single
    /// sequence is in use.
    fn drop(&mut self, fork_id: usize) -> Result<(), ContextError>;

    /// Removes all tokens that were added from the `pos`, effectively rolling the
    /// KV cache back to that position.  Returns the [`SlmPos`] at which the next
    /// token should be inserted.
    fn truncate(&mut self, pos: &SlmPos) -> Result<SlmPos, ContextError>;

    /// Removes the token range `[start_pos, end_pos)` from the middle of the KV
    /// cache, shifting subsequent tokens down.  `end_pos` is the position
    /// *immediately after* the last token to remove.  Returns the new tail
    /// [`SlmPos`] at which the next token should be inserted.
    fn cut(&mut self, start_pos: &SlmPos, end_pos: &SlmPos) -> Result<SlmPos, ContextError>;

    /// Serialises the full context state (KV cache, sampler state, etc.) to a byte
    /// buffer so it can be persisted or transferred.
    fn dump(&mut self) -> Result<Vec<u8>, ContextError>;

    /// Restores a context state previously produced by [`dump`](Self::dump).
    fn restore(&mut self, data: Vec<u8>) -> Result<(), ContextError>;

    fn edit_level(&self) -> SlmEditLevel;
}
