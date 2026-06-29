use super::{Answer, BoxedAction, BoxedVocab, Constraint, Pos, Role, InferenceError, SamplingError, Formatter};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::any::TypeId;
use std::fmt::Display;
use std::str::FromStr;
use strum::VariantNames;

/// Default upper bound on generated tokens per `generate` / `ask` call.
pub(crate) const DEFAULT_MAX_ANSWER_TOKENS: usize = 1024;

pub struct OracleState {
    pub(crate) pos: Pos,
    pub(crate) role: Option<Role>,
}

impl OracleState {
    pub fn new(pos: Pos, role: Option<Role>) -> Self {
        Self { pos, role }
    }
}

pub trait Oracle {
    /// Append a pre-formatted turn to the context without generating a response.
    /// Use this to replay history or inject system/user/assistant messages verbatim.
    fn prompt(&mut self, role: &Role, text: &str) -> Result<usize, InferenceError>;
    /// Append the user/tool turn, then generate the model's response.
    ///
    /// The context is saved before generation and automatically rolled back
    /// afterwards, so each call is stateless with respect to the KV cache.
    ///
    /// - `role` — must be [`Role::User`] or [`Role::Tool`].
    /// - `think` — when `true`, the reasoning trigger prefix is injected to
    ///   activate chain-of-thought (requires a compatible formatter).
    /// - `action` — optional early-stop callback; combined with the default
    ///   token-limit brake.
    fn generate(
        &mut self,
        /*User/Tool*/ role: &Role,
        text: &str,
        think: bool,
        reset: bool,
        action: BoxedAction,
        constraint: Option<&mut dyn Constraint>,
    ) -> Result<Answer<String>, InferenceError>;

    /// Reset the conversation: clear the KV cache and forget all turn state.
    fn clear(&mut self) -> Result<(), InferenceError>;

    /// Convenience wrapper: append a system-role turn to the context.
    fn system(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::System, text)
    }
    /// Convenience wrapper: append a user-role turn to the context.
    fn user(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::User, text)
    }
    /// Convenience wrapper: append an assistant-role turn to the context.
    fn assistant(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::Assistant, text)
    }

    /// Generate an answer to `text` without retaining the exchange in the context.
    /// Equivalent to `generate(User, text, think=false, brake)`.
    fn ask(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<String>, InferenceError> {
        self.generate(&Role::User, text, think, true, action, None)
    }

    /// Append a user turn and generate a response, *retaining* the exchange in the
    /// context (unlike [`ask`](Self::ask) which discards it).
    fn turn(
        &mut self,
        text: &str,
        think: bool,
        action: BoxedAction,
    ) -> Result<Answer<String>, InferenceError> {
        self.generate(&Role::User, text, think, false, action, None)
    }

    /// Roll the conversation back to a previously saved state.
    fn rollback(&mut self, state: &OracleState) -> Result<(), InferenceError>;
    /// Save the current conversation state so it can be restored later.
    fn save(&mut self) -> Result<OracleState, InferenceError>;
    /// Returns the number of tokens currently in the context.
    fn tokens_n(&self) -> usize;
    /// Override the per-call token generation limit (default: [`crate::oracle::DEFAULT_MAX_ANSWER_TOKENS`]).
    fn set_max_answer_tokens(&mut self, max_answer_tokens: usize);
    fn vocab(&self) -> &BoxedVocab;
    fn formatter(&self) -> &dyn Formatter;
}

pub type BoxedOracle = Box<dyn Oracle>;

pub trait JsonOracleExt {
    /// Ask the model a question and parse the response as `Vec<T>`, enforcing the
    /// JSON schema of `T` via constrained decoding.
    fn json_ask<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<Vec<T>>, InferenceError>;

    fn ask_values<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Vec<T>, InferenceError> {
        let answer = self.json_ask(think, text, action)?;
        match answer {
            Answer::Complete(answer, _) => Ok(answer),
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }
}

impl<Q: Oracle + ?Sized> JsonOracleExt for Q {
    fn json_ask<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<Vec<T>>, InferenceError> {
        let mut constraint = self.vocab().json_constraint(TypeId::of::<T>(), &|| {
            let schema = schemars::schema_for!(T);
            let value = serde_json::to_value(schema)
                .map_err(|e| SamplingError::Error(format!("serde_json error: {e}")))?;
            let bounds = self.formatter().reasoning_bounds().map(|(a,b)| (a.to_string(), b.to_string()));
            Ok((value, bounds))
        })?;
        let answer = self.generate(
            &Role::User,
            text,
            think,
            true,
            action,
            Some(constraint.as_mut()),
        )?;

        match &answer {
            Answer::Complete(str, thoughts) => {
                let e = serde_json::from_str(str)
                    .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")))?;
                Ok(Answer::Complete(e, thoughts.clone()))
            }
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }
}

pub trait EnumOracleExt {
    /// Ask the model a question and parse the response as `Vec<T>`, enforcing the
    /// JSON schema of `T` via constrained decoding.
    fn choose<T>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<T>, InferenceError>
    where
        T: VariantNames + FromStr + 'static,
        <T as FromStr>::Err: Display;

    fn choose_value<T>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<T, InferenceError>
    where
        T: VariantNames + FromStr + 'static,
        <T as FromStr>::Err: Display,
    {
        let answer = self.choose(think, text, action)?;
        match answer {
            Answer::Complete(answer, _) => Ok(answer),
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }
}

impl<Q: Oracle + ?Sized> EnumOracleExt for Q {
    fn choose<T>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<T>, InferenceError>
    where
        T: VariantNames + FromStr + 'static,
        <T as FromStr>::Err: Display,
    {
        let mut constraint = self.vocab().enum_constraint(TypeId::of::<T>(), &|| {
            let value = T::VARIANTS
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>();
            let bounds = self.formatter().reasoning_bounds().map(|(a,b)| (a.to_string(), b.to_string()));
            Ok((value, bounds))
        })?;
        let answer = self.generate(
            &Role::User,
            text,
            think,
            true,
            action,
            Some(constraint.as_mut()),
        )?;

        match &answer {
            Answer::Complete(str, thoughts) => {
                let val: Result<T, _> = T::from_str(str.as_str())
                    .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")));
                Ok(Answer::Complete(val?, thoughts.clone()))
            }
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }
}
