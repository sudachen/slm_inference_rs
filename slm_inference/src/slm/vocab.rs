use std::any::TypeId;
use std::sync::Arc;
use super::{SamplingError, StringToTokenError, TokenToStringError};

/// Type alias for a boxed [`Vocab`] trait object with thread-safety guarantees.
pub type BoxedVocab = Arc<dyn Vocab + Send + Sync>;

/// Instruction returned by a [`Constraint`] after a sampled token is committed.
///
/// The generation loop in [`SimpleInference`] inspects this value to decide
/// whether to run a normal forward pass, inject deterministic tokens, or halt.
#[derive(Debug, Clone)]
pub enum ConstraintStep {
    /// Skip the normal sampling step and inject this sequence of token IDs directly
    /// into the KV cache.  Used when the grammar uniquely determines the next tokens
    /// (e.g. forced punctuation in a JSON schema).
    FastForward(Vec<i32>),
    /// Run the normal forward pass for the next token.
    Forward,
    /// Terminate generation immediately; the constraint has been fully satisfied.
    Stop,
}


/// Token-level filter applied during autoregressive sampling.
///
/// Implementations can enforce structural constraints on model output, such as
/// JSON-schema validation ([`LarkConstraint`]) or simple token allow-lists.
///
/// The constraint is invoked once per generated token in two phases:
/// 1. [`mask`](Self::mask) adjusts logits *before* sampling.
/// 2. [`forward`](Self::forward) is called *after* sampling to advance the
///    constraint's internal state machine.
///
/// [`prefill`](Self::prefill) synchronises the constraint with text that was
/// already committed to the KV cache before generation begins.
pub trait Constraint {
    /// Mask logits in-place by setting forbidden token logits to `f32::NEG_INFINITY`.
    ///
    /// Returns `true` to continue generation, or `false` to stop immediately.
    fn mask(&mut self, logits: &mut [f32]) -> Result<bool, SamplingError>;
    /// Advance the constraint's state machine after `token_id` was sampled.
    ///
    /// Returns a [`ConstraintStep`] telling the generation loop what to do next.
    fn forward(&mut self, token_id: i32) -> Result<ConstraintStep, SamplingError>;
    /// Synchronise the constraint state with `text` that is already present in the
    /// KV cache (e.g. a reasoning trigger prefix).
    fn prefill(&mut self, text: &str) -> Result<(), SamplingError>;
}

/// Type alias for a boxed [`Constraint`] trait object.
pub type BoxedConstraint = Box<dyn Constraint + Send>;

/// No-op constraint that allows all tokens.
///
/// Used as a default when no structural constraints are needed.
pub struct Unconstrained;
impl Constraint for Unconstrained {
    fn mask(&mut self, _logits: &mut [f32]) -> Result<bool, SamplingError> {
        Ok(true)
    }

    fn forward(&mut self, _token_id: i32) -> Result<ConstraintStep, SamplingError> {
        Ok(ConstraintStep::Forward)
    }

    fn prefill(&mut self, _text: &str) -> Result<(), SamplingError> {
        Ok(())
    }
}


/// Vocabulary operations: encoding strings to tokens and decoding tokens to bytes.
///
/// A vocabulary is accessible through [`Context::vocab`].  The method
/// signatures mirror the llama.cpp tokeniser API, and [`tok_env`](Self::tok_env)
/// exposes the underlying `TokEnv` required by the `llguidance` constrained
/// generation library.
pub trait Vocab {
    /// Convert a token to its raw byte representation.
    ///
    /// `special` – render special tokens as text instead of skipping them.
    /// `left_strip` – if `Some(n)`, strip up to `n` leading space bytes from the result.
    fn token_to_bytes(
        &self,
        token: i32,
        special: bool,
    ) -> Result<Vec<u8>, TokenToStringError>;
    /// Tokenise a UTF-8 string into a sequence of model token IDs.
    ///
    /// `add_special` – prepend/append BOS/EOS markers as required by the model.
    /// `parse_special` – treat special-token text sequences (e.g. `<|im_start|>`) as
    /// their corresponding token IDs rather than plain text.
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, StringToTokenError>;
    fn json_constraint(
        &self,
        _type_id: TypeId,
        _json_schema: &dyn Fn() -> Result<(serde_json::Value,Option<(String,String)>), SamplingError>,

    ) -> Result<BoxedConstraint, SamplingError> {
        Ok(Box::new(Unconstrained))
    }
    fn enum_constraint(
        &self,
        _type_id: TypeId,
        _variants: &dyn Fn() -> Result<(Vec<String>,Option<(String,String)>), SamplingError>,
    ) -> Result<BoxedConstraint, SamplingError> {
        Ok(Box::new(Unconstrained))
    }
}

/// No-op vocabulary implementation for testing or placeholder use.
///
/// All operations return `Unsupported` errors. Used by [`NullInference`]
/// when a real vocabulary is not available.
pub struct NullVocab;

impl NullVocab {
    /// Create a new boxed [`NullVocab`] instance.
    pub fn new() -> BoxedVocab {
        Arc::new(Self)
    }
}

impl Vocab for NullVocab {
    fn token_to_bytes(&self, _token: i32, _special: bool) -> Result<Vec<u8>, TokenToStringError> {
        Err(TokenToStringError::Unsupported)
    }

    fn str_to_tokens(&self, _str: &str, _add_special: bool, _parse_special: bool) -> Result<Vec<i32>, StringToTokenError> {
        Err(StringToTokenError::Unsupported)
    }
}