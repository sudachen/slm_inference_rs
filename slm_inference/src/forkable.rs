use crate::{SlmAnswer, SlmBrake, SlmBrakeFilter, SlmContext, SlmInference};
use crate::errors::InferenceError;

pub trait SlmForkableInference : SlmInference {
    fn prefill_fork(&mut self, fork_id: usize, prompt: &str, logits: bool) -> Result<(), InferenceError>;
    fn generate_all_until(&mut self, f: &SlmBrakeFilter) -> Result<Vec<SlmAnswer>, InferenceError>;
    fn fork_context(&mut self) -> Result<usize, InferenceError>;
    fn drop_fork(&mut self, fork_id: usize) -> Result<(), InferenceError>;
    fn drop_all_forks(&mut self) -> Result<(), InferenceError>;

    fn generate_all(&mut self, max_tokens: usize) -> Result<Vec<SlmAnswer>, InferenceError> {
        self.generate_all_until(&SlmBrake::token_limit(max_tokens))
    }
}

pub struct SlmParallelInference<C: SlmContext> {
    context: C,
    n_cur: usize,
    batch: C::Batch,
    snapshot: Option<C::Snapshot>,
}

