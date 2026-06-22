use crate::errors::InferenceError;
use crate::formatter::SlmToolStyle;
use crate::oracle::{SlmOracleState};
use crate::{SlmAnswer, SlmBoxedBrakeFn, SlmBrake, SlmConstraint, SlmContext, SlmFormatter, SlmInference, SlmOracle, SlmRole, SlmSimpleInference};
use llguidance::api::TopLevelGrammar;
use llguidance::TokenParser;
use std::any::TypeId;
use crate::llg_lark::{json_schema_to_lark, LarkConstraint, ParserRegistry};

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
    reset_point: Option<SlmOracleState>,
    registry: Option<ParserRegistry>,
}

/// RAII guard that rolls back the inference KV cache to the last save point on drop.
///
/// Used inside [`SlmSimpleOracle::generate`] to ensure the generated tokens are
/// never committed to the persistent context.
struct SavePoint<'a, I: SlmInference, F: SlmFormatter>(&'a mut SlmSimpleOracle<I, F>);

impl<I: SlmInference, F: SlmFormatter> Drop for SavePoint<'_, I, F> {
    fn drop(&mut self) {
        if let Some(sp) = &self.0.reset_point {
            _ = self.0.inference.rollback(&sp.pos);
            self.0.active_turn = sp.role.clone();
            self.0.reset_point = None;
        }
    }
}

impl<C: SlmContext, F: SlmFormatter> SlmSimpleOracle<SlmSimpleInference<C>, F> {
    /// Create a new `SlmSimpleChat` wrapping a raw [`SlmContext`].
    pub fn new(context: C, formatter: F) -> Result<Self, InferenceError> {
        let inference = SlmSimpleInference::new(context)?;
        Ok(Self {
            inference,
            formatter,
            max_answer_tokens: crate::oracle::DEFAULT_MAX_ANSWER_TOKENS,
            is_fresh_context: true,
            active_turn: None,
            reset_point: None,
            registry: None,
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
                }
            }
            SlmToolStyle::SeparateTurn => {
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

    fn parser(
        &mut self,
        type_id: TypeId,
        grammar: Option<TopLevelGrammar>,
    ) -> Result<Option<TokenParser>, InferenceError> {
        let q = self
            .registry
            .get_or_insert_with(|| ParserRegistry::new(&self.inference.tok_env()));
        q.parser(type_id, grammar)
    }

}

impl<I: SlmInference, F: SlmFormatter> SlmOracle for SlmSimpleOracle<I, F> {
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<usize, InferenceError> {
        let mut fragment = String::new();
        self.prepare_prompt(role, text, &mut fragment)?;
        self.inference.prefill(&fragment)
    }

    fn generate(
        &mut self,
        role: &SlmRole,
        text: &str,
        think: bool,
        reset: bool,
        brake: Option<SlmBoxedBrakeFn>,
        mut constraint: Option<&mut dyn SlmConstraint>,
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

        if think && let Some(trigger) = self.formatter.reasoning_trigger(){
            fragment.push_str(trigger);
            if let Some(tk) = constraint.as_deref_mut() {
                tk.prefill(trigger)?;
            }
        } else if self.formatter.reasoning_trigger().is_some() {
            //fragment.push_str(&self.formatter.wrap_reasoning(""))
        }

        self.reset_point = Some(self.save()?);
        let _ = SavePoint(self);

        self.inference.prefill(&fragment)?;
        let mut answer = self.inference.generate_until(
            &mut [brake, Some(SlmBrake::token_limit(self.max_answer_tokens))],
            constraint,
        )?;

        if think {
            answer =
                answer.map(|s| self.formatter.reasoning_trigger().unwrap_or("").to_string() + &s);
        }

        if !reset && !think && answer.is_complete() {
            self.reset_point = None;
            self.active_turn = None;
        }

        Ok(answer.split_thought(&self.formatter))
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        self.is_fresh_context = true;
        self.active_turn = None;
        self.inference.clear()
    }

    fn rollback(&mut self, state: &SlmOracleState) -> Result<(), InferenceError> {
        self.inference.rollback(&state.pos)?;
        self.active_turn = state.role.clone();
        Ok(())
    }

    fn save(&mut self) -> Result<SlmOracleState, InferenceError> {
        Ok(SlmOracleState::new(
            self.inference.save()?,
            self.active_turn.clone(),
        ))
    }

    fn tokens_n(&self) -> usize {
        self.inference.tokens_n()
    }

    fn set_max_answer_tokens(&mut self, max_answer_tokens: usize) {
        self.max_answer_tokens = max_answer_tokens;
    }

    fn json_constraint(&mut self, type_id: TypeId,
                       json_schema: &dyn Fn() -> Result<serde_json::Value, InferenceError>) -> Result<Box<dyn SlmConstraint>, InferenceError> {
        if let Some(parser) = self.parser(type_id, None)? {
            return Ok(Box::new(LarkConstraint::new(parser)));
        }
        let thinking = self.formatter.reasoning_bounds().map(|(s,e)| (s.trim(), e.trim()));
        let lark = json_schema_to_lark(json_schema()?, thinking).map_err(|s| InferenceError::InvalidJsonSchema(s.to_string()))?;
        let grammar = TopLevelGrammar::from_lark(lark);
        let parser = self
            .parser(type_id, Some(grammar))?
            .ok_or(InferenceError::Error("parser not found".to_string()))?;
        Ok(Box::new(LarkConstraint::new(parser)))
    }
}

