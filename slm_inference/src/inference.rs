use crate::errors::InferenceError;
use crate::{SlmAnswer, SlmBatch, SlmConstraint, SlmConstraintStep, SlmContext, SlmEditLevel, SlmPos, SlmTokEnv, SlmToken, SlmVocab};
use tracing::error;

/// A heap-allocated, callable early-stop callback passed to [`SlmInference::generate_until`].
///
/// The closure receives the full accumulated answer so far (`&str`), the last
/// decoded token as text (`&str`), the total number of generated tokens (`usize`),
/// and the active fork/sequence ID (`usize`).
/// It returns an [`SlmAction`] that tells the generation loop how to proceed.
pub type SlmBoxedAction = Box<
    dyn FnMut(
            /*answer*/ &str,
            /*last_token*/ &str,
            /*n_tokens*/ usize,
            /*fork_id*/ usize,
        ) -> SlmAction
        + Send
        + 'static,
>;

/// Control-flow signal returned by an [`SlmBoxedAction`] callback after each generated token.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum SlmAction {
    /// Stop generation and return the accumulated text as a [`SlmAnswer::Complete`](crate::SlmAnswer::Complete).
    Finish,
    /// Stop generation and return the accumulated text as a [`SlmAnswer::Incomplete`](crate::SlmAnswer::Incomplete).
    Stop,
    /// Enqueue the sampled token for a future decode but pause generation now,
    /// returning a [`SlmAnswer::Partial`](crate::SlmAnswer::Partial).
    /// Any subsequent prompt call will resume and eventually terminate the sequence.
    /// Not applicable to `ask_for`-style calls.
    Delay,
    /// Continue generation normally; this token is accepted.
    Continue,
    /// Accept the token without counting it as a stopping condition (used internally
    /// in multi-callback chains to signal "not my business").
    Next,
}

impl SlmAction {
    /// Returns a callback that signals [`Finish`](Self::Finish) once `max_tokens` tokens
    /// have been generated.
    pub fn token_limit(max_tokens: usize) -> SlmBoxedAction {
        Box::new(move |_, _, n, _| match n >= max_tokens {
            true => SlmAction::Finish,
            false => SlmAction::Continue,
        })
    }

    /// Returns a callback that prints each new token to stdout and continues generation.
    pub fn print_token() -> SlmBoxedAction {
        Box::new(move |_, token, _, _| {
            print!("{token}");
            SlmAction::Next
        })
    }

    /// Returns `true` if this action stops or pauses generation
    /// (`Finish`, `Stop`, or `Delay`).
    pub fn brake(&self) -> bool {
        matches!(self, SlmAction::Finish | SlmAction::Stop | SlmAction::Delay)
    }

    /// Poll a slice of optional callbacks with the current generation state, returning
    /// the first non-`Next` action found, or `Continue` if all callbacks return `Next`.
    pub fn brake_on(
        a: &str,
        b: &str,
        n: usize,
        fork_id: usize,
        lf: &mut [Option<SlmBoxedAction>],
    ) -> Self {
        lf.iter_mut()
            .flatten()
            .map(|f| f(a, b, n, fork_id))
            .find(|b| *b != SlmAction::Next)
            .unwrap_or(SlmAction::Continue)
    }
}

/// Low-level generation engine used by [`SlmSimpleOracle`](crate::SlmSimpleOracle).
///
/// The typical call sequence is:
/// 1. [`prefill`](Self::prefill) one or more formatted prompt fragments.
/// 2. [`generate_until`](Self::generate_until) to run the autoregressive loop.
/// 3. [`rollback`](Self::rollback) (or [`clear`](Self::clear)) to discard generated tokens.
pub trait SlmInference {
    /// Tokenise `prompt` and append the tokens to the pending prefill buffer.
    ///
    /// Returns the number of tokens added.  The tokens are not decoded until
    /// [`generate_until`](Self::generate_until) is called.
    fn prefill(&mut self, prompt: &str) -> Result<usize, InferenceError>;
    /// Run the autoregressive generation loop until a callback in `f` signals a
    /// stop condition or EOS is reached.
    ///
    /// `f` is a mutable slice of optional [`SlmBoxedAction`] callbacks polled
    /// after every token.  `c` is an optional [`SlmConstraint`] applied at each
    /// sampling step.
    fn generate_until(
        &mut self,
        f: &mut [Option<SlmBoxedAction>],
        c: Option<&mut dyn SlmConstraint>,
    ) -> Result<SlmAnswer, InferenceError>;
    /// Clear the KV cache and all pending tokens, resetting to an empty state.
    fn clear(&mut self) -> Result<(), InferenceError>;
    /// Record the current token position so it can be restored later.
    fn save(&mut self) -> Result<SlmPos, InferenceError>;
    /// Roll the KV cache back to `pos`, discarding all tokens added since then.
    fn rollback(&mut self, pos: &SlmPos) -> Result<(), InferenceError>;
    /// Serialise the full generation state to bytes.
    fn dump(&mut self) -> Result<Vec<u8>, InferenceError>;
    /// Restore a state previously produced by [`dump`](Self::dump).
    fn restore(&mut self, data: Vec<u8>) -> Result<(), InferenceError>;
    /// Returns the total number of tokens currently in the sequence (prefill + generated).
    fn tokens_n(&self) -> usize;
    /// Returns the llguidance token environment for this inference engine.
    fn tok_env(&self) -> &SlmTokEnv;
}

/// A straightforward single-sequence implementation of [`SlmInference`] backed by
/// any [`SlmContext`].
///
/// Manages an internal `tokens` buffer and a reusable `batch`, draining the buffer
/// in chunks of up to [`SlmContext::max_batch_len`] tokens per forward pass.
pub struct SlmSimpleInference<C: SlmContext> {
    context: C,
    n_cur: usize,
    batch: C::Batch,
    tokens: Vec<C::Token>,
}

impl<C: SlmContext> SlmSimpleInference<C> {
    pub fn new(context: C) -> Result<Self, InferenceError> {
        let n_batch = context.max_batch_len();
        let batch = context.new_batch(n_batch, 1)?;
        Ok(Self {
            context,
            n_cur: 0,
            batch,
            tokens: Vec::new(),
        })
    }
}

impl<C: SlmContext> SlmInference for SlmSimpleInference<C> {
    fn prefill(&mut self, prompt: &str) -> Result<usize, InferenceError> {
        let tokens_list = self.context.str_to_tokens(prompt, false, true)?;
        if tokens_list.is_empty() {
            return Ok(tokens_list.len());
        }
        self.tokens.extend_from_slice(&tokens_list);
        Ok(tokens_list.len())
    }

    fn generate_until(
        &mut self,
        filter: &mut [Option<SlmBoxedAction>],
        mut constraint: Option<&mut dyn SlmConstraint>,
    ) -> Result<SlmAnswer, InferenceError> {
        let mut response_str = String::with_capacity(4096);
        let mut brake = SlmAction::Continue;
        let mut n_tokens = 0usize;
        self.internal_prefill(true)?;
        if self.batch.n_tokens() == 0 {
            return Err(InferenceError::EmptyBatch);
        }
        while !brake.brake() {
            let k = constraint.as_mut().map(|x| &mut **x as &mut dyn SlmConstraint);
            let token = match self
                .context
                .sample_with_constraint(self.batch.n_tokens() - 1, k)?
            {
                Some(t) => t,
                None => {
                    self.batch.clear();
                    return Ok(SlmAnswer::Complete(response_str, 0, None));
                }
            };
            n_tokens += 1;
            match self.context.token_to_bytes(token, false, None) {
                Ok(bytes) => {
                    let last_token = String::from_utf8_lossy(&bytes);
                    brake = SlmAction::brake_on(&response_str, &last_token, n_tokens, 0, filter);
                    response_str.push_str(&last_token);
                }
                Err(e) => {
                    error!("Failed to extract token bytes: {:?}", e);
                    return Err(e.into());
                }
            }

            self.batch.clear();
            if brake == SlmAction::Continue || brake == SlmAction::Delay {
                let r = constraint.as_deref_mut().map_or(Ok(None),|x| x.forward(token.as_i32()).map(Some))?;
                match r {
                    Some(SlmConstraintStep::FastForward(ff_tokens)) => {
                        self.batch.add(token, SlmPos::new(self.n_cur, 0), false)?;
                        let last_token = ff_tokens.len() - 1;
                        for (i, t) in ff_tokens.iter().enumerate() {
                            self.n_cur += 1;
                            self.batch.add(
                                C::Token::from_i32(*t),
                                SlmPos::new(self.n_cur, 0),
                                i == last_token,
                            )?;
                        }
                    }
                    Some(SlmConstraintStep::Forward) | None => {
                        self.batch.add(token, SlmPos::new(self.n_cur, 0), true)?
                    }
                    Some(SlmConstraintStep::Stop) => {
                        return Ok(SlmAnswer::Complete(response_str, 0, None));
                    }
                }
            }
            if brake == SlmAction::Continue {
                self.context.decode(&mut self.batch)?;
            }
            self.tokens.push(token);
            self.n_cur += 1;
        }
        Ok(SlmAnswer::Partial(response_str, 0))
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        self.batch.clear();
        self.context.clear()?;
        Ok(())
    }

    fn save(&mut self) -> Result<SlmPos, InferenceError> {
        Ok(SlmPos::new(self.tokens.len(), 0))
    }

    fn rollback(&mut self, pos: &SlmPos) -> Result<(), InferenceError> {
        // speculative save
        if self.tokens.len() > pos.token_pos {
            self.tokens.truncate(pos.token_pos);
        }
        if self.n_cur > pos.token_pos {
            if self.context.edit_level() >= SlmEditLevel::Cut {
                self.n_cur = self.context.truncate(pos)?.token_pos;
            } else {
                // non-cuttable models with SST/Mamba arch
                self.context.drop(pos.fork_id)?;
                self.n_cur = 0;
            }
        }
        Ok(())
    }

    fn dump(&mut self) -> Result<Vec<u8>, InferenceError> {
        todo!()
    }

    fn restore(&mut self, _data: Vec<u8>) -> Result<(), InferenceError> {
        todo!()
    }

    fn tokens_n(&self) -> usize {
        self.tokens.len()
    }

    fn tok_env(&self) -> &SlmTokEnv {
        self.context.vocab().tok_env()
    }
}
