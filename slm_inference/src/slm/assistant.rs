use super::{
    Action, Answer, BoxedAction, BoxedConstraint, BoxedVocab, Context, Formatter, Inference,
    InferenceError, Role, SamplingError, SimpleInference, ToolStyle,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::any::TypeId;
use std::fmt::Display;
use std::str::FromStr;
use strum::VariantNames;

/// Default upper bound on generated tokens per `generate` / `ask` call.
pub const DEFAULT_MAX_ANSWER_TOKENS: usize = 1024;

/// Type alias for a boxed [`Inference`] trait object.
pub type BoxedInference = Box<dyn Inference + Send>;
/// Type alias for a boxed [`Formatter`] trait object.
pub type BoxedFormatter = Box<dyn Formatter + Send>;

/// Snapshot of an [`Assistant`]'s state for rollback/restore operations.
///
/// Captures the KV-cache position and active conversation role at a point in time.
pub struct State {
    pub(crate) pos: usize,
    pub(crate) role: Option<Role>,
}

impl State {
    /// Create a new state snapshot.
    pub fn new(pos: usize, role: Option<Role>) -> Self {
        Self { pos, role }
    }
}

/// High-level conversational inference interface.
///
/// Combines an [`Inference`] engine with a [`Formatter`] to provide turn-based
/// chat functionality with automatic chat-template management. Supports
/// system/user/assistant roles, tool calling, and structured output generation.
pub struct Assistant {
    pub(crate) inference: BoxedInference,
    pub(crate) formatter: BoxedFormatter,
    pub(crate) max_answer_tokens: usize,
    pub(crate) is_fresh_context: bool,
    pub(crate) active_turn: Option<Role>,
    pub(crate) reset_point: Option<State>,
}

/// RAII guard that automatically rolls back an [`Assistant`] to its saved state on drop.
///
/// Used internally by [`Assistant::generate`] to implement the `reset` parameter.
pub struct SavePoint<'a>(pub &'a mut Assistant);

impl Drop for SavePoint<'_> {
    fn drop(&mut self) {
        if let Some(sp) = &self.0.reset_point {
            _ = self.0.inference.rollback(sp.pos);
            self.0.active_turn = sp.role.clone();
            self.0.reset_point = None;
        }
    }
}

impl Assistant {
    /// Create a new [`Assistant`] from a context and formatter.
    pub fn new<C: Context + Send + 'static, F: Formatter + Send + 'static>(
        context: C,
        formatter: F,
    ) -> Result<Self, InferenceError> {
        let inference = SimpleInference::new(context)?;
        Ok(Self {
            inference: Box::new(inference),
            formatter: Box::new(formatter),
            max_answer_tokens: DEFAULT_MAX_ANSWER_TOKENS,
            is_fresh_context: true,
            active_turn: None,
            reset_point: None,
        })
    }

    /// Convenience wrapper: append a system-role turn to the context.
    pub fn system(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::System, text)
    }
    /// Convenience wrapper: append a user-role turn to the context.
    pub fn user(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::User, text)
    }
    /// Convenience wrapper: append an assistant-role turn to the context.
    pub fn assistant(&mut self, text: &str) -> Result<usize, InferenceError> {
        self.prompt(&Role::Assistant, text)
    }

    /// Generate an answer to `text` without retaining the exchange in the context.
    /// Equivalent to `generate(User, text, think=false, brake)`.
    pub fn ask(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<String>, InferenceError> {
        self.generate(&Role::User, text, think, true, action, None)
    }

    /// Append a user turn and generate a response, *retaining* the exchange in the
    /// context (unlike [`ask`](Self::ask) which discards it).
    pub fn turn(
        &mut self,
        text: &str,
        think: bool,
        action: BoxedAction,
    ) -> Result<Answer<String>, InferenceError> {
        self.generate(&Role::User, text, think, false, action, None)
    }

    /// Append the BOS (beginning of sequence) token if this is a fresh context.
    pub fn bos(&mut self, s: &mut String) {
        if self.is_fresh_context {
            if let Some(bos) = self.formatter.bos() {
                s.push_str(bos);
            }
            self.is_fresh_context = false;
        }
    }
    fn prepare_prompt(
        &mut self,
        role: &Role,
        text: &str,
        fragment: &mut String,
    ) -> Result<(), InferenceError> {
        self.bos(fragment);
        match self.formatter.tool_style() {
            ToolStyle::Inline => {
                match role {
                    Role::System | Role::User => {
                        if self.active_turn == Some(Role::Assistant) {
                            fragment.push_str(&self.formatter.turn_end(&Role::Assistant));
                        }

                        let role_clone = Some(role.clone());
                        if self.active_turn != role_clone {
                            if let Some(active_role) = &self.active_turn {
                                fragment.push_str(&self.formatter.turn_end(active_role));
                            }
                            self.active_turn = role_clone;
                            fragment.push_str(&self.formatter.turn_start(role));
                        }
                        fragment.push_str(text);
                    }
                    Role::Assistant => {
                        // if assistant turn is not active, start it
                        if self.active_turn != Some(Role::Assistant) {
                            fragment.push_str(&self.formatter.turn_start(&Role::Assistant));
                            self.active_turn = Some(Role::Assistant);
                        }
                        fragment.push_str(text);
                    }
                }
            }
            ToolStyle::SeparateTurn => {
                if let Some(active_role) = &self.active_turn
                    && active_role != role
                {
                    fragment.push_str(&self.formatter.turn_end(&active_role));
                    fragment.push_str(&self.formatter.turn_start(&active_role));
                }
                fragment.push_str(text);
                self.active_turn = Some(role.clone());
            }
        }
        Ok(())
    }

    /// Append a turn with the given role and text to the context.
    pub fn prompt(&mut self, role: &Role, text: &str) -> Result<usize, InferenceError> {
        let mut fragment = String::new();
        self.prepare_prompt(role, text, &mut fragment)?;
        self.inference.prefill(&fragment)
    }

    /// Generate a response for the given role and text.
    ///
    /// If `reset` is true, the context is rolled back after generation.
    /// If `think` is true, reasoning tags are added for models that support them.
    pub fn generate(
        &mut self,
        role: &Role,
        text: &str,
        think: bool,
        reset: bool,
        action: BoxedAction,
        mut constraint: Option<BoxedConstraint>,
    ) -> Result<Answer<String>, InferenceError> {
        let fragment = self.generate_fragment(role, text, think, &mut constraint)?;
        let _ = SavePoint(self);
        self.inference.prefill(&fragment)?;
        let answer = self.inference.generate_until(
            &mut [action, Action::token_limit(self.max_answer_tokens)],
            constraint,
        )?;
        self.generate_answer(answer, think, reset)
    }

    /// Build the prompt fragment for generation without actually running inference.
    pub fn generate_fragment(
        &mut self,
        role: &Role,
        text: &str,
        think: bool,
        constraint: &mut Option<BoxedConstraint>,
    ) -> Result<String, InferenceError> {
        let mut fragment = String::new();

        if role == &Role::Assistant || role == &Role::System {
            return Err(InferenceError::InvalidRole);
        }

        self.prepare_prompt(role, text, &mut fragment)?;

        if self.active_turn != Some(Role::Assistant) {
            fragment.push_str(&self.formatter.turn_end(role));
            fragment.push_str(&self.formatter.turn_start(&Role::Assistant));
            self.active_turn = Some(Role::Assistant);
        }

        if think && let Some(trigger) = self.formatter.reasoning_trigger() {
            fragment.push_str(trigger);
            if let Some(tk) = constraint.as_deref_mut() {
                tk.prefill(trigger)?;
            }
        } else if self.formatter.reasoning_trigger().is_some() {
            //fragment.push_str(&self.formatter.wrap_reasoning(""))
        }

        self.reset_point = Some(self.save()?);
        Ok(fragment)
    }

    /// Post-process a generated answer, handling reasoning extraction and state management.
    pub fn generate_answer(
        &mut self,
        mut answer: Answer<String>,
        think: bool,
        reset: bool,
    ) -> Result<Answer<String>, InferenceError> {
        if think {
            answer =
                answer.map(|s| self.formatter.reasoning_trigger().unwrap_or("").to_string() + &s);
        }

        if !reset && !think && answer.is_complete() {
            self.reset_point = None;
            self.active_turn = None;
        }

        Ok(answer.split_thought(self.formatter.as_ref()))
    }

    /// Clear the entire context, resetting to an empty state.
    pub fn clear(&mut self) -> Result<(), InferenceError> {
        self.is_fresh_context = true;
        self.active_turn = None;
        self.inference.clear()
    }

    /// Roll back the context to a previously saved state.
    pub fn rollback(&mut self, state: &State) -> Result<(), InferenceError> {
        self.inference.rollback(state.pos)?;
        self.active_turn = state.role.clone();
        if let Some(p) = self.reset_point.as_ref()
            && p.pos > state.pos
        {
            self.reset_point = None;
        }
        Ok(())
    }

    /// Save the current state for potential rollback.
    pub fn save(&mut self) -> Result<State, InferenceError> {
        Ok(State::new(self.inference.pos(), self.active_turn.clone()))
    }

    /// Return the total number of tokens in the context.
    pub fn tokens_n(&self) -> usize {
        self.inference.pos()
    }

    /// Set the maximum number of tokens to generate per answer.
    pub fn set_max_answer_tokens(&mut self, max_answer_tokens: usize) {
        self.max_answer_tokens = max_answer_tokens;
    }

    /// Return a reference to the vocabulary.
    pub fn vocab(&self) -> &BoxedVocab {
        self.inference.vocab()
    }

    /// Return a reference to the formatter.
    pub fn formatter(&self) -> &dyn Formatter {
        self.formatter.as_ref()
    }

    /// Generate a JSON array of type `T` using schema-based constrained generation.
    pub fn json_ask<T: DeserializeOwned + JsonSchema + 'static>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<Vec<T>>, InferenceError> {
        let constraint = self.vocab().json_constraint(TypeId::of::<T>(), &|| {
            let schema = schemars::schema_for!(T);
            let value = serde_json::to_value(schema)
                .map_err(|e| SamplingError::Error(format!("serde_json error: {e}")))?;
            let bounds = self
                .formatter()
                .reasoning_bounds()
                .map(|(a, b)| (a.to_string(), b.to_string()));
            Ok((value, bounds))
        })?;
        let answer = self.generate(&Role::User, text, think, true, action, Some(constraint))?;

        match &answer {
            Answer::Complete(str, thoughts) => {
                let e = serde_json::from_str(str)
                    .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")))?;
                Ok(Answer::Complete(e, thoughts.clone()))
            }
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }

    /// Convenience wrapper for [`json_ask`] that returns the values directly.
    pub fn ask_values<T: DeserializeOwned + JsonSchema + 'static>(
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
    /// Choose a single value from an enum using constrained generation.
    pub fn choose<T>(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<T>, InferenceError>
    where
        T: VariantNames + FromStr + 'static,
        <T as FromStr>::Err: Display,
    {
        let constraint = self.vocab().enum_constraint(TypeId::of::<T>(), &|| {
            let value = T::VARIANTS
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>();
            let bounds = self
                .formatter()
                .reasoning_bounds()
                .map(|(a, b)| (a.to_string(), b.to_string()));
            Ok((value, bounds))
        })?;
        let answer = self.generate(&Role::User, text, think, true, action, Some(constraint))?;

        match &answer {
            Answer::Complete(str, thoughts) => {
                let val: Result<T, _> = T::from_str(str.as_str())
                    .map_err(|e| InferenceError::Error(format!("serde_json error: {e}")));
                Ok(Answer::Complete(val?, thoughts.clone()))
            }
            _ => Err(InferenceError::IncompleteAnswer),
        }
    }

    /// Convenience wrapper for [`choose`] that returns the value directly.
    pub fn choose_value<T>(
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
