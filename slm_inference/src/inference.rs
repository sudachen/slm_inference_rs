use crate::errors::{ContextError, InferenceError};
use crate::{SlmAnswer, SlmBatch, SlmContext, SlmEditLevel, SlmPos};
use tracing::error;

pub type SlmBoxedBrakeFn = Box<
    dyn FnMut(
            /*answer*/ &str,
            /*last_token*/ &str,
            /*n_tokens*/ usize,
            /*fork_id*/ usize,
        ) -> SlmBrake
        + Send
        + 'static,
>;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum SlmBrake {
    // prevent following generation and returns Complete Answer
    Finish,
    // prevent following generation and returns Incomplete Answer
    Stop,
    // puts sampled token to the batch for continuation and returns Partial Answer
    // any following prompt will terminate generation
    // is not applicable to aks_for
    Delay,
    Continue,
    Next,
}

impl SlmBrake {
    pub fn token_limit(max_tokens: usize) -> SlmBoxedBrakeFn {
        Box::new(move |_, _, n, _| match n >= max_tokens {
            true => SlmBrake::Finish,
            false => SlmBrake::Continue,
        })
    }

    pub fn brake(&self) -> bool {
        matches!(self, SlmBrake::Finish | SlmBrake::Stop | SlmBrake::Delay)
    }

    pub fn brake_on(
        a: &str,
        b: &str,
        n: usize,
        fork_id: usize,
        lf: &mut [Option<SlmBoxedBrakeFn>],
    ) -> Self {
        lf.iter_mut()
            .flatten()
            .map(|f| f(a, b, n, fork_id))
            .find(|b| *b != SlmBrake::Next)
            .unwrap_or(SlmBrake::Continue)
    }
}

pub trait SlmInference {
    fn prefill(&mut self, prompt: &str) -> Result<(), InferenceError>;
    fn generate_until(
        &mut self,
        f: &mut [Option<SlmBoxedBrakeFn>],
    ) -> Result<SlmAnswer, InferenceError>;
    fn generate(&mut self, max_tokens: usize) -> Result<SlmAnswer, InferenceError> {
        self.generate_until(&mut [Some(SlmBrake::token_limit(max_tokens))])
    }
    fn clear(&mut self) -> Result<(), InferenceError>;
    fn save(&mut self) -> Result<(), InferenceError>;
    fn rollback(&mut self) -> Result<(), InferenceError>;
    fn dump(&mut self) -> Result<Vec<u8>, InferenceError>;
    fn restore(&mut self, data: Vec<u8>) -> Result<(), InferenceError>;
}

pub struct SlmSimpleInference<C: SlmContext> {
    context: C,
    n_cur: usize,
    batch: C::Batch,
    save_point: Option<SlmPos>,
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
            save_point: None,
            tokens: Vec::new(),
        })
    }
}

impl<C: SlmContext> SlmSimpleInference<C> {
    fn internal_prefill(&mut self, logits: bool) -> Result<(), InferenceError> {
        if self.n_cur < self.tokens.len() {
            let last_index = self.tokens.len() - 1;
            let n_batch = self.batch.n_max();
            let base_pos = self.n_cur;
            self.batch.clear();
            for (i, token) in self.tokens[base_pos..].iter().enumerate() {
                let pos = base_pos + i;
                let is_last = pos == last_index;
                self.batch
                    .add(*token, SlmPos::new(pos, 0), is_last && logits)?;
                if self.batch.n_tokens() >= n_batch || is_last {
                    self.n_cur += self.batch.n_tokens();
                    self.context.decode(&mut self.batch)?;
                    if !is_last {
                        self.batch.clear();
                    }
                }
            }
        }
        Ok(())
    }
}

impl<C: SlmContext> SlmInference for SlmSimpleInference<C> {
    fn prefill(&mut self, prompt: &str) -> Result<(), InferenceError> {
        let tokens_list = self.context.str_to_tokens(prompt, false, true)?;
        if tokens_list.is_empty() {
            return Ok(());
        }
        self.tokens.extend_from_slice(&tokens_list);
        Ok(())
    }
    fn generate_until(
        &mut self,
        filter: &mut [Option<SlmBoxedBrakeFn>],
    ) -> Result<SlmAnswer, InferenceError> {
        let mut response_str = String::with_capacity(4096);
        let mut brake = SlmBrake::Continue;
        let mut n_tokens = 0usize;
        self.internal_prefill(true)?;
        if self.batch.n_tokens() == 0 {
            return Err(InferenceError::EmptyBatch);
        }
        while !brake.brake() {
            let token = match self.context.sample(self.batch.n_tokens() - 1)? {
                Some(t) => t,
                None => {
                    self.batch.clear();
                    return Ok(SlmAnswer::Complete(response_str, 0, None));
                }
            };
            n_tokens += 1;
            match self.context.token_to_bytes(token, 64, false, None) {
                Ok(bytes) => {
                    let last_token = String::from_utf8_lossy(&bytes);
                    brake = SlmBrake::brake_on(&response_str, &last_token, n_tokens, 0, filter);
                    response_str.push_str(&last_token);
                }
                Err(e) => {
                    error!("Failed to extract token bytes: {:?}", e);
                    return Err(e.into());
                }
            }
            self.batch.clear();
            if brake == SlmBrake::Continue || brake == SlmBrake::Delay {
                self.batch.add(token, SlmPos::new(self.n_cur, 0), true)?;
                self.tokens.push(token);
                self.n_cur += 1;
            }
            if brake == SlmBrake::Continue {
                self.context.decode(&mut self.batch)?;
            }
        }
        Ok(SlmAnswer::Partial(response_str, 0))
    }
    fn clear(&mut self) -> Result<(), InferenceError> {
        self.batch.clear();
        self.context.clear()?;
        Ok(())
    }

    fn save(&mut self) -> Result<(), InferenceError> {
        //self.save_point = Some(SlmPos::new(self.n_cur,0));
        // speculative save
        self.save_point = Some(SlmPos::new(self.tokens.len(), 0));
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), InferenceError> {
        match self.save_point.as_ref() {
            Some(s) => {
                //self.n_cur = self.context.truncate(s)?.token_pos;
                // speculative save
                if self.tokens.len() > s.token_pos {
                    self.tokens.truncate(s.token_pos);
                }
                if self.n_cur > s.token_pos {
                    if self.context.edit_level() >= SlmEditLevel::Cut {
                        self.n_cur = self.context.truncate(s)?.token_pos;
                    } else {
                        // non-cuttable models with SST/Mamba arch
                        self.context.drop(s.fork_id)?;
                        self.n_cur = 0;
                    }
                }
            }
            None => return Err(ContextError::PosNotFound.into()),
        }
        Ok(())
    }

    fn dump(&mut self) -> Result<Vec<u8>, InferenceError> {
        todo!()
    }

    fn restore(&mut self, _data: Vec<u8>) -> Result<(), InferenceError> {
        todo!()
    }
}
