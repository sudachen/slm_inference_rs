use crate::errors::InferenceError;
use crate::formatter::{SlmFormatter, SlmToolStyle};
use crate::{
    SlmAnswer, SlmBoxedBrakeFn, SlmBrake, SlmContext, SlmInference, SlmRole, SlmSimpleInference,
};

const DEFAULT_MAX_ANSWER_TOKENS: usize = 1024;

/// High-level interface for interacting with a language model in a conversational manner.
///
/// `SlmOracle` manages turn-taking, context formatting, and answer generation.
/// Implementors are responsible for maintaining the conversation state (active role,
/// fresh-context flag, etc.) and delegating to an underlying [`SlmInference`] engine.
pub trait SlmOracle {
    /// Append a pre-formatted turn to the context without generating a response.
    /// Use this to replay history or inject system/user/assistant messages verbatim.
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<(), InferenceError>;
    /// Append the user/tool turn, then generate the model's response.
    ///
    /// The context is saved before generation and automatically rolled back
    /// afterwards, so each call is stateless with respect to the KV cache.
    ///
    /// - `role` — must be [`SlmRole::User`] or [`SlmRole::Tool`].
    /// - `think` — when `true`, the reasoning trigger prefix is injected to
    ///   activate chain-of-thought (requires a compatible formatter).
    /// - `brake` — optional early-stop callback; combined with the default
    ///   token-limit brake.
    fn generate(
        &mut self,
        /*User/Tool*/ role: &SlmRole,
        text: &str,
        think: bool,
        brake: Option<SlmBoxedBrakeFn>,
    ) -> Result<SlmAnswer, InferenceError>;

    /// Reset the conversation: clear the KV cache and forget all turn state.
    fn clear(&mut self) -> Result<(), InferenceError>;

    /// Convenience wrapper: append a system-role turn to the context.
    fn system(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::System, text)
    }
    /// Convenience wrapper: append a user-role turn to the context.
    fn user(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::User, text)
    }
    /// Convenience wrapper: append an assistant-role turn to the context.
    fn assistant(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::Assistant, text)
    }
    /// Convenience wrapper: append a tool-response turn to the context.
    fn tool(&mut self, tool_name: &str, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::tool(tool_name), text)
    }

    /// Generate an answer to `text` without retaining the exchange in the context.
    /// Equivalent to `generate(User, text, think=false, brake)`.
    fn ask(
        &mut self,
        text: &str,
        brake: Option<SlmBoxedBrakeFn>,
    ) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::User, text, false, brake)
    }

    /// Generate a reasoned answer to `text` without retaining the exchange in the context.
    /// Injects the reasoning trigger prefix so the model produces chain-of-thought output.
    /// Equivalent to `generate(User, text, think=true, brake)`.
    fn think(
        &mut self,
        text: &str,
        brake: Option<SlmBoxedBrakeFn>,
    ) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::User, text, true, brake)
    }
}

/// Standard [`SlmOracle`] implementation backed by any [`SlmInference`] engine
/// and any [`SlmFormatter`].
///
/// Tracks the currently active role and whether the KV cache is empty so it
/// can emit the correct opening BOS token and role delimiters.
pub struct SlmSimpleOracle<I: SlmInference, F: SlmFormatter> {
    inference: I,
    formatter: F,
    max_answer_tokens: usize,
    is_fresh_context: bool,
    active_turn: Option<SlmRole>,
}

/// RAII guard that rolls back the inference KV cache to the last save point on drop.
///
/// Used inside [`SlmSimpleOracle::generate`] to ensure the generated tokens are
/// never committed to the persistent context.
struct SavePoint<'a>(&'a mut dyn SlmInference);

impl Drop for SavePoint<'_> {
    fn drop(&mut self) {
        self.0.rollback().unwrap();
    }
}

impl<C: SlmContext, F: SlmFormatter> SlmSimpleOracle<SlmSimpleInference<C>, F> {
    /// Create a new `SlmSimpleChat` wrapping a raw [`SlmContext`].
    pub fn new(context: C, formatter: F) -> Result<Self, InferenceError> {
        let inference = SlmSimpleInference::new(context)?;
        Ok(Self {
            inference,
            formatter,
            max_answer_tokens: DEFAULT_MAX_ANSWER_TOKENS,
            is_fresh_context: true,
            active_turn: None,
        })
    }
}

impl<I: SlmInference, F: SlmFormatter> SlmSimpleOracle<I, F> {
    /// Append the BOS token to `s` the very first time a prompt is built,
    /// then mark the context as no longer fresh.
    fn bos(&mut self, s: &mut String) {
        if self.is_fresh_context {
            if let Some(bos) = self.formatter.bos() {
                s.push_str(bos);
            }
            self.is_fresh_context = false;
        }
    }
}

impl<I: SlmInference, F: SlmFormatter> SlmSimpleOracle<I, F> {
    /// Build the formatted prompt fragment for `role`/`text` into `fragment`.
    ///
    /// Handles BOS injection, role-delimiter open/close sequencing, and the
    /// two tool-embedding strategies ([`SlmToolStyle::Inline`] vs
    /// [`SlmToolStyle::SeparateTurn`]).
    fn prepare_prompt(
        &mut self,
        role: &SlmRole,
        text: &str,
        fragment: &mut String,
    ) -> Result<(), InferenceError> {
        self.bos(fragment);
        match self.formatter.tool_style() {
            SlmToolStyle::Inline => {
                match role {
                    SlmRole::System | SlmRole::User => {
                        if self.active_turn == Some(SlmRole::Assistant) {
                            fragment.push_str(&self.formatter.turn_end(&SlmRole::Assistant));
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
                    SlmRole::Assistant => {
                        // if assistant turn is not active, start it
                        if self.active_turn != Some(SlmRole::Assistant) {
                            fragment.push_str(&self.formatter.turn_start(&SlmRole::Assistant));
                            self.active_turn = Some(SlmRole::Assistant);
                        }
                        fragment.push_str(text);
                    }
                    SlmRole::Tool(tool_name) => {
                        // restoring tool response from history
                        // mast be in the assistant turn
                        if self.active_turn != Some(SlmRole::Assistant) {
                            fragment.push_str(&self.formatter.turn_start(&SlmRole::Assistant));
                            self.active_turn = Some(SlmRole::Assistant);
                        }
                        fragment.push_str(&self.formatter.format_tool_response(tool_name, text));
                    }
                }
            }
            SlmToolStyle::SeparateTurn => {
                if let Some(active_role) = &self.active_turn
                    && active_role != role
                {
                    fragment.push_str(&self.formatter.turn_end(&active_role));
                    fragment.push_str(&self.formatter.turn_start(&active_role));
                }
                if let SlmRole::Tool(tool_name) = role {
                    fragment.push_str(&self.formatter.format_tool_response(tool_name, text));
                } else {
                    fragment.push_str(text);
                }
                self.active_turn = Some(role.clone());
            }
        }
        Ok(())
    }
}

impl<I: SlmInference, F: SlmFormatter> SlmOracle for SlmSimpleOracle<I, F> {
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<(), InferenceError> {
        let mut fragment = String::new();
        self.prepare_prompt(role, text, &mut fragment)?;
        self.inference.prefill(&fragment)
    }

    fn generate(
        &mut self,
        role: &SlmRole,
        text: &str,
        think: bool,
        brake: Option<SlmBoxedBrakeFn>,
    ) -> Result<SlmAnswer, InferenceError> {
        let mut fragment = String::new();

        if role == &SlmRole::Assistant || role == &SlmRole::System {
            return Err(InferenceError::InvalidRole);
        }

        self.prepare_prompt(role, text, &mut fragment)?;

        if self.active_turn != Some(SlmRole::Assistant) {
            fragment.push_str(&self.formatter.turn_end(role));
            fragment.push_str(&self.formatter.turn_start(&SlmRole::Assistant));
            self.active_turn = Some(SlmRole::Assistant);
        }

        if think {
            fragment.push_str(self.formatter.reasoning_trigger().unwrap_or(""));
        }

        self.inference.save()?;
        let _ = SavePoint(&mut self.inference);
        self.inference.prefill(&fragment)?;
        let mut answer = self
            .inference
            .generate_until(&mut [brake, Some(SlmBrake::token_limit(self.max_answer_tokens))])?;
        if think {
            answer =
                answer.map(|s| self.formatter.reasoning_trigger().unwrap_or("").to_string() + &s);
        }
        Ok(answer.split_thought(&self.formatter))
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        self.is_fresh_context = true;
        self.active_turn = None;
        self.inference.clear()
    }
}
