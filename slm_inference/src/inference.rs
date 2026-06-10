use std::cmp::PartialEq;
use crate::errors::{ContextError, InferenceError};
use crate::{SlmContext,SlmBatch, SlmRollback, SlmBrakeFilter, SlmBrake, SlmAnswer, SlmRole};
use tracing::error;

pub trait SlmInference : SlmRollback {
    fn prefill(&mut self, prompt: &str, logits: bool) -> Result<(), InferenceError>;
    fn generate_until(&mut self, f: &SlmBrakeFilter) -> Result<SlmAnswer, InferenceError>;
    fn clear(&mut self) -> Result<(), InferenceError>;
    fn format(&self, parts: &[(SlmRole,&str)], ask: bool) -> Result<String, InferenceError>;

    fn generate(&mut self, max_tokens: usize) -> Result<SlmAnswer, InferenceError> {
        self.generate_until(&SlmBrake::token_limit(max_tokens))
    }
}

pub struct SlmSimpleInference<C: SlmContext> {
    context: C,
    n_cur: usize,
    batch: C::Batch,
    snapshot: Option<C::Snapshot>,
}

impl<C: SlmContext> SlmSimpleInference<C> {
    pub fn new(context: C) -> Result<Self,InferenceError> {
        let n_batch = context.max_batch_len();
        let batch = context.new_batch(n_batch, 1)?;
        Ok(Self {
            context,
            n_cur: 0,
            batch,
            snapshot: None
        })
    }
}

impl<C: SlmContext> SlmInference for SlmSimpleInference<C> {
    fn prefill(&mut self, prompt: &str, logits: bool) -> Result<(), InferenceError> {
        self.batch.clear();
        let tokens_list = self.context.str_to_tokens(prompt, true, true)?;
        if tokens_list.is_empty() {
            return Ok(());
        }
        let last_index = tokens_list.len() - 1;
        let n_batch = self.batch.n_max();
        let base_pos = self.n_cur;

        for (i, token) in tokens_list.iter().enumerate() {
            let is_last = i == last_index;
            self.batch.add(*token, base_pos + i, &[0], is_last && logits)?;
            if self.batch.n_tokens() >= n_batch || is_last {
                self.n_cur += self.batch.n_tokens();
                self.context.decode(&mut self.batch)?;
                if !is_last {
                    self.batch.clear();
                }
            }
        }

        Ok(())
    }
    fn generate_until(&mut self, filter: &SlmBrakeFilter) -> Result<SlmAnswer, InferenceError> {
        let mut response_str = String::with_capacity(4096);
        let mut brake = SlmBrake::Continue;
        let mut n_tokens = 0usize;
        while !brake.brake() {
            let token = match self.context.sample(self.batch.n_tokens() - 1)? {
                Some(t) => t,
                None => {
                    self.batch.clear();
                    return Ok(SlmAnswer::Complete(response_str, 0))
                },
            };
            n_tokens += 1;
            match self.context.token_to_bytes(token, 64, false, None) {
                Ok(bytes) => {
                    let last_token = String::from_utf8_lossy(&bytes);
                    brake = filter(&response_str, &last_token, n_tokens);
                    response_str.push_str(&last_token);
                }
                Err(e) => {
                    error!("Failed to extract token bytes: {:?}", e);
                    return Err(e.into())
                }
            }
            self.batch.clear();
            if brake == SlmBrake::Continue || brake == SlmBrake::Delay {
                self.batch.add(token, self.n_cur, &[0], true)?;
                self.n_cur += 1;
            }
            if brake == SlmBrake::Continue {
                self.context.decode(&mut self.batch)?;
            }
        }
        Ok(SlmAnswer::Partial(response_str,0))
    }
    fn clear(&mut self) -> Result<(), InferenceError> {
        self.batch.clear();
        self.context.clear()?;
        Ok(())
    }
    fn format(&self, parts: &[(SlmRole,&str)], ask: bool) -> Result<String, InferenceError> {
        let prompt = self.context.format(parts, ask)?;
        Ok(prompt)
    }
}

impl<C: SlmContext> SlmRollback for SlmSimpleInference<C> {
    fn save(&mut self) -> Result<(), InferenceError> {
        self.snapshot = Some(self.context.save(self.n_cur,None)?);
        Ok(())
    }
    fn rollback(&mut self) -> Result<(), InferenceError> {
        match self.snapshot.as_ref() {
            Some(s) => {
                self.n_cur = self.context.rollback(s)?
            },
            None => return Err(ContextError::SnapshotNotFound.into())
        }
        Ok(())
    }
}

