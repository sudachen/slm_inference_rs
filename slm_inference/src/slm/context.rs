use super::{
    BatchError, BoxedVocab, Constraint, ContextBuilderError, ContextError, DecodeError, Pos,
    SamplingError
};

/// Hardware zone where inference computation is performed.
#[derive(Copy,Clone,Debug)]
pub enum ComputationZone {
    /// Computation on CPU.
    CPU,
    /// Computation on GPU.
    GPU,
}

/// A submission buffer that accumulates tokens before they are decoded in bulk.
///
/// Batching amortises the cost of the model's forward pass: instead of invoking
/// the backend once per token, callers fill the batch and call [`Context::decode`] once.
pub trait Batch {
    /// Enqueue a single token at the given KV-cache position.
    ///
    /// Set `logits` to `true` for the last token in a prefill chunk, or for every
    /// token during generation, so that [`Context::sample_with_constraint`] can
    /// read the computed logits.
    fn add(&mut self, token: i32, pos: Pos, logits: bool) -> Result<(), BatchError>;
    /// Discard all queued tokens, resetting the batch for re-use.
    fn clear(&mut self);
    /// Number of token slots currently queued in this batch.
    fn n_tokens(&self) -> usize;
    /// Maximum number of token slots this batch can hold.
    fn n_max(&self) -> usize;
}

/// Declares which KV-cache editing operations an [`Context`] implementation supports.
///
/// Used by [`SimpleInference`] to select the most efficient rollback strategy.
/// Variants are ordered by capability: higher values include the capabilities of all
/// lower values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd)]
pub enum EditLevel {
    /// Only dump/restore is supported.  Every rollback requires a full
    /// serialise–deserialise round-trip via [`Context::dump`] / [`Context::restore`].
    #[default]
    DumpRestore = 0,
    /// [`Context::truncate`] is available: the tail of the KV cache can be
    /// removed efficiently without a full dump/restore cycle.
    Cut = 1,
    /// Both [`Context::truncate`] and [`Context::cut`] are available:
    /// arbitrary token ranges can be excised from the KV cache in-place.
    Truncate = 2,
}

/// Core inference context that owns a KV cache and provides all token-level operations
/// required for autoregressive text generation.
///
/// A context is obtained from [`ContextBuilder::build`] and is used by inference
/// helpers such as [`SimpleInference`].  Each context is associated with a single
/// model and maintains mutable state (KV cache, sampler state) across calls.
pub trait Context {
    /// The batch type used to submit tokens for decoding.
    type Batch: Batch + Send;
    /// Returns a reference to this context's vocabulary for token encoding/decoding.
    fn vocab(&self) -> &BoxedVocab;
    fn zone(&self) -> ComputationZone;

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
        constraint: Option<&mut dyn Constraint>,
    ) -> Result<Option<i32>, SamplingError>;

    /// Resets the context to an empty state, discarding the entire KV cache.
    fn clear(&mut self) -> Result<(), ContextError>;

    /// Discards all KV cache entries belonging to the sequence identified by
    /// `fork_id`, freeing the associated cache slots without affecting other
    /// sequences.  Equivalent to [`clear`](Self::clear) when only a single
    /// sequence is in use.
    fn drop(&mut self, fork_id: usize) -> Result<(), ContextError>;

    /// Removes all tokens that were added from the `pos`, effectively rolling the
    /// KV cache back to that position.  Returns the [`Pos`] at which the next
    /// token should be inserted.
    fn truncate(&mut self, pos: &Pos) -> Result<Pos, ContextError>;

    /// Removes the token range `[start_pos, end_pos)` from the middle of the KV
    /// cache, shifting subsequent tokens down.  `end_pos` is the position
    /// *immediately after* the last token to remove.  Returns the new tail
    /// [`Pos`] at which the next token should be inserted.
    fn cut(&mut self, start_pos: &Pos, end_pos: &Pos) -> Result<Pos, ContextError>;

    /// Serialises the full context state (KV cache, sampler state, etc.) to a byte
    /// buffer so it can be persisted or transferred.
    fn dump(&mut self) -> Result<Vec<u8>, ContextError>;

    /// Restores a context state previously produced by [`dump`](Self::dump).
    fn restore(&mut self, data: Vec<u8>) -> Result<(), ContextError>;

    /// Reports which in-place editing operations this context supports.
    ///
    /// The value is used by [`SimpleInference`] to pick the cheapest rollback
    /// strategy.  See [`EditLevel`] for the available levels.
    fn edit_level(&self) -> EditLevel;
}

/// Builder for configuring and instantiating an [`Context`].
///
/// Obtained from [`Model::context`].  All `with_*` methods consume `self`
/// and return `Self` for method chaining.  Call [`build`](Self::build) to
/// produce the configured context.
pub trait ContextBuilder<T: Context + Send> {
    /// Consume the builder and create the inference context.
    fn build(self) -> Result<T, ContextBuilderError>;
    /// Configure sampler parameters.
    ///
    /// `temperature` – softmax temperature (`0.0` → greedy, `1.0` → unmodified distribution).
    /// `top_k` – keep only the top-k highest-probability candidates; `≤0` disables.
    /// `top_p` – nucleus-sampling threshold `(0.0–1.0)`; `1.0` disables.
    fn with_sampler(self, temperature: f32, top_k: i32, top_p: f32) -> Self;
    /// Set the maximum context length in tokens.
    fn with_n_ctx(self, n_ctx: usize) -> Self;
    /// Override the quantization format for KV-cache key and value tensors.
    fn with_gen_type_kv(self, k: KvType, v: KvType) -> Self;
    /// Set the maximum batch size (tokens processed in a single forward pass).
    fn with_n_batch(self, n_batch: usize) -> Self;
    /// Enable or disable Flash Attention for this context.
    fn with_flash_attn(self, enable: bool) -> Self;
}

/// Quantization format used for KV-cache tensors.
///
/// Lower-precision formats reduce VRAM usage at the cost of a small accuracy
/// penalty.  `F16`/`F32` are lossless; the `Q*` variants trade precision for memory.
pub enum KvType {
    /// 4-bit quantised KV cache.
    Q4,
    /// 5-bit quantised KV cache.
    Q5,
    /// 6-bit quantised KV cache.
    Q6,
    /// 8-bit quantised KV cache.
    Q8,
    /// Raw (non-llama.cpp-quantised) 8-bit KV cache.
    RawQ8,
    /// Half-precision (16-bit float) KV cache.
    F16,
    /// Single-precision (32-bit float) KV cache.
    F32,
}
