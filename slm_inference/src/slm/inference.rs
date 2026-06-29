use super::{
    Answer, BoxedConstraint, BoxedInference, BoxedVocab, ComputationZone, InferenceError, NullVocab,
};

/// Type alias for an optional callback that controls generation flow.
///
/// The callback receives the current answer text, last token, token count,
/// and fork ID, and returns an [`Action`] indicating whether to continue,
/// pause, or stop generation.
pub type BoxedAction = Option<
    Box<
        dyn FnMut(
                /*answer*/ &str,
                /*last_token*/ &str,
                /*n_tokens*/ usize,
                /*fork_id*/ usize,
            ) -> Action
            + Send
            + 'static,
    >,
>;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Action {
    /// Stop generation and return the accumulated text as a [`Answer::Complete`](crate::slm::Answer::Complete).
    Finish,
    /// Stop generation and return the accumulated text as a [`Answer::Incomplete`](crate::slm::Answer::Incomplete).
    Stop,
    /// Enqueue the sampled token for a future decode but pause generation now,
    /// returning a [`Answer::Partial`](crate::slm::Answer::Partial).
    /// Any subsequent prompt call will resume and eventually terminate the sequence.
    Delay,
    /// Continue generation normally; this token is accepted.
    Continue,
    /// Accept the token without counting it as a stopping condition (used internally
    /// in multi-callback chains to signal "not my business").
    Next,
}

impl Action {
    /// Returns a callback that signals [`Finish`](Self::Finish) once `max_tokens` tokens
    /// have been generated.
    pub fn token_limit(max_tokens: usize) -> BoxedAction {
        Some(Box::new(move |_, _, n, _| match n >= max_tokens {
            true => Action::Finish,
            false => Action::Continue,
        }))
    }

    /// Returns a callback that prints each new token to stdout and continues generation.
    pub fn print_token() -> BoxedAction {
        Some(Box::new(move |_, token, _, _| {
            print!("{token}");
            Action::Next
        }))
    }

    /// Returns `true` if this action stops or pauses generation
    /// (`Finish`, `Stop`, or `Delay`).
    pub fn brake(&self) -> bool {
        matches!(self, Action::Finish | Action::Stop | Action::Delay)
    }

    /// Poll a slice of optional callbacks with the current generation state, returning
    /// the first non-`Next` action found, or `Continue` if all callbacks return `Next`.
    pub fn brake_on(a: &str, b: &str, n: usize, fork_id: usize, lf: &mut [BoxedAction]) -> Self {
        lf.iter_mut()
            .flatten()
            .map(|f| f(a, b, n, fork_id))
            .find(|b| *b != Action::Next)
            .unwrap_or(Action::Continue)
    }
}

/// Trait for autoregressive text generation engines.
///
/// Implementations handle tokenization, KV-cache management, sampling,
/// and constraint enforcement. The trait is designed to be backend-agnostic,
/// allowing different inference backends (CPU, GPU, etc.) to be used
/// interchangeably.
pub trait Inference {
    /// Tokenise `prompt` and append the tokens to the pending prefill buffer.
    ///
    /// Returns the number of tokens added.  The tokens are not decoded until
    /// [`generate_until`](Self::generate_until) is called.
    fn prefill(&mut self, prompt: &str) -> Result<usize, InferenceError>;
    /// Run the autoregressive generation loop until a callback in `f` signals a
    /// stop condition or EOS is reached.
    ///
    /// `f` is a mutable slice of optional [`BoxedAction`] callbacks polled
    /// after every token.  `c` is an optional [`Constraint`] applied at each
    /// sampling step.
    fn generate_until(
        &mut self,
        f: &mut [BoxedAction],
        c: Option<BoxedConstraint>,
    ) -> Result<Answer<String>, InferenceError>;
    /// Clear the KV cache and all pending tokens, resetting to an empty state.
    fn clear(&mut self) -> Result<(), InferenceError>;
    /// Roll the KV cache back to `pos`, discarding all tokens added since then.
    fn rollback(&mut self, pos: usize) -> Result<(), InferenceError>;
    fn pos(&self) -> usize;
    /// Returns the llguidance token environment for this inference engine.
    fn vocab(&self) -> &BoxedVocab;
    fn zone(&self) -> ComputationZone;
}

/// No-op implementation of [`Inference`] for testing or placeholder use.
///
/// All operations return `InferenceError::Unsupported`. Used internally
/// by async inference to temporarily swap out the real inference engine.
pub struct NullInference {
    vocab: BoxedVocab,
}

impl NullInference {
    /// Create a new boxed [`NullInference`] instance.
    pub fn new() -> BoxedInference {
        let this = Self {
            vocab: NullVocab::new(),
        };
        Box::new(this)
    }
}

impl Inference for NullInference {
    fn prefill(&mut self, _prompt: &str) -> Result<usize, InferenceError> {
        Err(InferenceError::Unsupported)
    }

    fn generate_until(
        &mut self,
        _f: &mut [BoxedAction],
        _c: Option<BoxedConstraint>,
    ) -> Result<Answer<String>, InferenceError> {
        Err(InferenceError::Unsupported)
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        Err(InferenceError::Unsupported)
    }

    fn rollback(&mut self, _pos: usize) -> Result<(), InferenceError> {
        Err(InferenceError::Unsupported)
    }

    fn pos(&self) -> usize {
        0
    }

    fn vocab(&self) -> &BoxedVocab {
        &self.vocab
    }

    fn zone(&self) -> ComputationZone {
        ComputationZone::CPU
    }
}
