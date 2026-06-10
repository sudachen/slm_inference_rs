use crate::errors::InferenceError;
use crate::formatter::{SlmFormatter, SlmToolStyle};
use crate::{
    SlmAnswer, SlmBrake, SlmBrakeFilter, SlmContext, SlmInference, SlmRole, SlmRollback,
    SlmSimpleInference,
};

const DEFAULT_MAX_ANSWER_TOKENS: usize = 1024;

pub trait SlmChat: SlmRollback {
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<(), InferenceError>;
    fn generate(
        &mut self,
        /*User/Tool*/ role: &SlmRole,
        text: &str,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError>;
    fn continue_answer(
        &mut self,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError>;

    fn clear(&mut self) -> Result<(), InferenceError>;
    //fn ask_for(&mut self, text: &[String], brake: Option<&SlmBrakeFilter>) -> Result<Vec<SlmAnswer>,InferenceError>;

    fn system(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::System, text)
    }
    fn user(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::User, text)
    }
    fn assistant(&mut self, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::Assistant, text)
    }
    fn tool(&mut self, tool_name: &str, text: &str) -> Result<(), InferenceError> {
        self.prompt(&SlmRole::tool(tool_name), text)
    }
    fn user_ask(
        &mut self,
        text: &str,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::User, text, brake)
    }
    fn tool_ask(
        &mut self,
        tool_name: &str,
        text: &str,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError> {
        self.generate(&SlmRole::tool(tool_name), text, brake)
    }
}

pub struct SlmSimpleChat<I: SlmInference, F: SlmFormatter> {
    inference: I,
    formatter: F,
    max_answer_tokens: usize,
    is_fresh_context: bool,
    active_turn: Option<SlmRole>,
}

impl<C: SlmContext, F: SlmFormatter> SlmSimpleChat<SlmSimpleInference<C>, F> {
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

impl<I: SlmInference, F: SlmFormatter> SlmSimpleChat<I, F> {
    fn bos(&mut self, s: &mut String) {
        if self.is_fresh_context {
            if let Some(bos) = self.formatter.bos() {
                s.push_str(bos);
            }
            self.is_fresh_context = false;
        }
    }
}

impl<I: SlmInference, F: SlmFormatter> SlmRollback for SlmSimpleChat<I, F> {
    fn save(&mut self) -> Result<(), InferenceError> {
        self.inference.save()
    }

    fn rollback(&mut self) -> Result<(), InferenceError> {
        self.inference.rollback()
    }
}

impl<I: SlmInference, F: SlmFormatter> SlmSimpleChat<I, F> {
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
                            self.active_turn = None;
                        }

                        fragment.push_str(&self.formatter.turn_start(role));
                        fragment.push_str(text);
                        fragment.push_str(&self.formatter.turn_end(role));
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
                fragment.push_str(&self.formatter.turn_start(role));
                if let SlmRole::Tool(tool_name) = role {
                    fragment.push_str(&self.formatter.format_tool_response(tool_name, text));
                } else {
                    fragment.push_str(text);
                }
                fragment.push_str(&self.formatter.turn_end(role));
                self.active_turn = None;
            }
        }
        Ok(())
    }
}
impl<I: SlmInference, F: SlmFormatter> SlmChat for SlmSimpleChat<I, F> {
    fn prompt(&mut self, role: &SlmRole, text: &str) -> Result<(), InferenceError> {
        let mut fragment = String::new();
        self.prepare_prompt(role, text, &mut fragment)?;
        self.inference.prefill(&fragment, false)
    }

    fn generate(
        &mut self,
        role: &SlmRole,
        text: &str,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError> {
        let mut fragment = String::new();

        if role == &SlmRole::Assistant || role == &SlmRole::System {
            return Err(InferenceError::InvalidRole);
        }

        self.bos(&mut fragment);
        if role == &SlmRole::User || role.is_tool() {
            self.prepare_prompt(role, text, &mut fragment)?;
        }

        if self.active_turn != Some(SlmRole::Assistant) {
            fragment.push_str(&self.formatter.turn_start(&SlmRole::Assistant));
            self.active_turn = Some(SlmRole::Assistant);
        }

        self.inference.prefill(&fragment, true)?;
        let answer = self
            .inference
            .generate_until(brake.unwrap_or(&SlmBrake::token_limit(self.max_answer_tokens)))?;

        if self.formatter.tool_style() == SlmToolStyle::SeparateTurn && answer.is_complete() {
            self.inference
                .prefill(&self.formatter.turn_end(&SlmRole::Assistant), false)?;
            self.active_turn = None;
        }

        Ok(answer.map(|s| self.formatter.clean(&s)))
    }

    fn continue_answer(
        &mut self,
        brake: Option<&SlmBrakeFilter>,
    ) -> Result<SlmAnswer, InferenceError> {
        let answer = self
            .inference
            .generate_until(brake.unwrap_or(&SlmBrake::token_limit(self.max_answer_tokens)))?;

        if self.formatter.tool_style() == SlmToolStyle::SeparateTurn && answer.is_complete() {
            self.inference
                .prefill(&self.formatter.turn_end(&SlmRole::Assistant), false)?;
            self.active_turn = None;
        }

        Ok(answer.map(|s| self.formatter.clean(&s)))
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        self.is_fresh_context = true;
        self.active_turn = None;
        self.inference.clear()
    }
}
