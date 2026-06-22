use crate::errors::{InferenceError, SamplingError};
use crate::{SlmAnswer, SlmBoxedAction, SlmConstraint, SlmConstraintStep, SlmPos, SlmRole};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::any::TypeId;

/// Default upper bound on generated tokens per `generate` / `ask` call.
pub(crate) const DEFAULT_MAX_ANSWER_TOKENS: usize = 1024;

/// A snapshot of the oracle's conversation state at a given point in time.
///
/// Returned by [`SlmOracle::save`] and consumed by [`SlmOracle::rollback`]
/// to restore the context to an earlier turn boundary.
pub struct SlmOracleState {
    pub(crate) pos: SlmPos,
    pub(crate) role: Option<SlmRole>,
}

impl SlmOracleState {
    pub fn new(pos: SlmPos, role: Option<SlmRole>) -> Self {
        Self { pos, role }
    }
}

/// High-level interface for interacting with a language model in a conversational manner.
///
/// `SlmOracle` manages turn-taking, context formatting, and answer generation.
/// Implementors are responsible for maintaining the conversation state (active role,
/// fresh-context flag, etc.) and delegating to an underlying [`SlmInference`] engine.
pub trait SlmOracle {
    /// Append a pre-formatted turn to the context without generating a response.
    /// Use this to replay history or inject system/user/assistant messages verbatim.
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<usize, InferenceError>;
    /// Append the user/tool turn, then generate the model's response.
    ///
    /// The context is saved before generation and automatically rolled back
    /// afterwards, so each call is stateless with respect to the KV cache.
    ///
    /// - `role` — must be [`SlmRole::User`] or [`SlmRole::Tool`].
    /// - `think` — when `true`, the reasoning trigger prefix is injected to
    ///   activate chain-of-thought (requires a compatible formatter).
    /// - `action` — optional early-stop callback; combined with the default
    ///   token-limit brake.
    fn generate(
        &mut self,
        /*User/Tool*/ role: &SlmRole,
        text: &str,
        think: bool,
        reset: bool,
        action: Option<SlmBoxedAction>,
        constraint: Option<&mut dyn SlmConstraint>,
    ) -> Result<SlmAnswer, InferenceError>;

    /// Reset the conversation: clear the KV cache and forget all turn state.
    fn clear(&mut self) -> Result<(), InferenceError>;

    /// Convenience wrapper: append a system-role turn to the context.
    fn system(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&SlmRole::System, text)
    }
    /// Convenience wrapper: append a user-role turn to the context.
    fn user(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&SlmRole::User, text)
    }
    /// Convenience wrapper: append an assistant-role turn to the context.
    fn assistant(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&SlmRole::Assistant, text)
    }

    /// Generate an answer to `text` without retaining the exchange in the context.
    /// Equivalent to `generate(User, text, think=false, brake)`.
    fn ask(&mut self, think: bool, text: &str, action: Option<SlmBoxedAction>) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::User, text, think, true, action, None)
    }

    /// Append a user turn and generate a response, *retaining* the exchange in the
    /// context (unlike [`ask`](Self::ask) which discards it).
    fn turn(&mut self, text: &str, think: bool, action: Option<SlmBoxedAction>) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::User, text, think, false, action, None)
    }

    /// Roll the conversation back to a previously saved state.
    fn rollback(&mut self, state: &SlmOracleState) -> Result<(), InferenceError>;
    /// Save the current conversation state so it can be restored later.
    fn save(&mut self) -> Result<SlmOracleState, InferenceError>;
    /// Returns the number of tokens currently in the context.
    fn tokens_n(&self) -> usize;
    /// Override the per-call token generation limit (default: [`DEFAULT_MAX_ANSWER_TOKENS`]).
    fn set_max_answer_tokens(&mut self, max_answer_tokens: usize);

    /// Build a [`SlmConstraint`] that enforces the JSON schema of type `T`.
    ///
    /// The default implementation returns an [`Unconstrained`] no-op constraint.
    /// [`SlmSimpleOracle`](crate::SlmSimpleOracle) overrides this with a real
    /// Lark-grammar constraint backed by `llguidance`.
    fn json_constraint(
        &mut self,
        _type_id: TypeId,
        _json_schema: &dyn Fn() -> Result<serde_json::Value, InferenceError>,
    ) -> Result<Box<dyn SlmConstraint>, InferenceError> {
        Ok(Box::new(Unconstrained))
    }
}

/// Extension of [`SlmOracle`] that generates structured JSON output validated against
/// the compile-time schema of a `serde`/`schemars` type `T`.
///
/// Blanket-implemented for every `SlmOracle`, delegating constraint construction
/// to [`SlmOracle::json_constraint`].
pub trait SlmJsonOracle {
    /// Ask the model a question and parse the response as `Vec<T>`, enforcing the
    /// JSON schema of `T` via constrained decoding.
    fn json_ask<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: Option<SlmBoxedAction>
    ) -> Result<Vec<T>, InferenceError>;
}

impl<Oracle: SlmOracle + ?Sized> SlmJsonOracle for Oracle {
    fn json_ask<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: Option<SlmBoxedAction>
    ) -> Result<Vec<T>, InferenceError> {
        let mut constraint = self.json_constraint(TypeId::of::<T>(), &|| {
            let schema = schemars::schema_for!(T);
            serde_json::to_value(schema)
                .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")))
        })?;
        let answer = self.generate(
            &SlmRole::User,
            text,
            think,
            true,
            action,
            Some(constraint.as_mut()),
        )?;

        serde_json::from_str(answer.as_str())
            .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")))
    }
}

/// A no-op [`SlmConstraint`] that places no restrictions on generated tokens.
///
/// Used as the default when no grammar or schema constraint is provided.
pub struct Unconstrained;
impl SlmConstraint for Unconstrained {
    fn mask(&mut self, _logits: &mut [f32]) -> Result<bool, SamplingError> {
        Ok(true)
    }

    fn forward(&mut self, _token_id: i32) -> Result<SlmConstraintStep, SamplingError> {
        Ok(SlmConstraintStep::Forward)
    }

    fn prefill(&mut self, _text: &str) -> Result<(), SamplingError> {
        Ok(())
    }
}
