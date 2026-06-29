use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use super::{Action, Answer, Batch, BoxedAction, BoxedVocab, BoxedConstraint, Constraint, ConstraintStep, Context, EditLevel, Inference, InferenceError, Pos, ComputationZone, DecodeError};
use tracing::error;

#[derive(Clone,Debug,Default)]
struct ContextState {
    id: u64,
    state: Vec<u8>,
    pos: Pos,
    n_cur: usize,
    seq_id: usize,
}

struct InferenceCore<C: Context + Send> {
    context: C,
    batch: C::Batch,
    id_counter: u64,
    state: ContextState,
    store: HashMap<u64, ContextState>,
}

impl<C: Context + Send> InferenceCore<C> {
    pub fn new(context: C) -> Result<Arc<Mutex<Self>>, InferenceError> {
        let n_batch = context.max_batch_len();
        let batch = context.new_batch(n_batch, 1)?;
        Ok(Arc::new(Mutex::new(Self {
            context,
            batch,
            id_counter: 1,
            state: ContextState::default(),
            store: HashMap::new(),
        })))
    }

    pub fn allocate(&mut self) -> u64 {
        let id = self.id_counter;
        self.id_counter += 1;
        let state = ContextState {
            id,
            state: vec![],
            pos: Pos::new(0, 0),
            n_cur: 0,
            seq_id: 0,
        };
        self.store.insert(id, state);
        id
    }

    pub fn clear(&mut self, state_id: u64) {
        todo!()
    }

    pub fn set_current_state(&mut self, state_id: u64) ->Result<(), InferenceError> {
        // TODO: dump/save
        Ok(())
    }

    fn internal_prefill(&mut self, tokens: &[i32], logits: bool, state_id: u64, n_cur: usize) -> Result<usize, InferenceError> {
        self.set_current_state(state_id)?;
        if self.state.n_cur > n_cur {
            if self.context.edit_level() >= EditLevel::Cut {
                self.state.n_cur = self.context.truncate(&Pos::new(n_cur, self.state.seq_id))?.token_pos;
            } else {
                // non-cuttable models with SST/Mamba arch
                self.context.drop(self.state.seq_id)?;
                self.state.n_cur = 0;
            }
        }
        self.batch.clear();
        if self.state.n_cur < tokens.len() {
            let last_index = tokens.len() - 1;
            let n_batch = self.batch.n_max();
            let base_pos = self.state.n_cur;
            for (i, token) in tokens[base_pos..].iter().enumerate() {
                let pos = base_pos + i;
                let is_last = pos == last_index;
                self.batch
                    .add(*token, Pos::new(pos, 0), is_last && logits)?;
                if self.batch.n_tokens() >= n_batch || is_last {
                    self.context.decode(&mut self.batch)?;
                    self.state.n_cur += self.batch.n_tokens();
                    if !is_last {
                        self.batch.clear();
                    }
                }
            }
        }
        Ok(self.state.n_cur)
    }

    fn internal_decode(&mut self) -> Result<(), DecodeError> {
        self.context.decode(&mut self.batch)?;
        self.state.n_cur += self.batch.n_tokens();
        Ok(())
    }
}

pub struct SimpleInference<C: Context + Send> {
    state_id: u64,
    context: Arc<Mutex<InferenceCore<C>>>,
    n_cur: usize,
    tokens: Vec<i32>,
    vocab: BoxedVocab,
    zone: ComputationZone,
}

impl<C: Context + Send> SimpleInference<C> {
    pub fn new(context: C) -> Result<Self, InferenceError> {
        let vocab = context.vocab().clone();
        let zone = context.zone();
        Ok(Self {
            state_id: 0,
            context: InferenceCore::new(context)?,
            n_cur: 0,
            tokens: Vec::new(),
            vocab,
            zone,
        })
    }
    pub fn share(&self) -> SimpleInference<C> {
        let context = self.context.clone();
        let state_id = context.lock().unwrap().allocate();
        Self {
            state_id,
            context,
            n_cur: 0,
            tokens: Vec::new(),
            vocab: self.vocab.clone(),
            zone: self.zone,
        }
    }
}

impl<C: Context + Send> Inference for SimpleInference<C> {
    fn prefill(&mut self, prompt: &str) -> Result<usize, InferenceError> {
        let tokens_list = self.vocab.str_to_tokens(prompt, false, true)?;
        if tokens_list.is_empty() {
            return Ok(tokens_list.len());
        }
        self.tokens.extend_from_slice(&tokens_list);
        Ok(tokens_list.len())
    }

    fn generate_until(
        &mut self,
        filter: &mut [BoxedAction],
        mut constraint: Option<BoxedConstraint>,
    ) -> Result<Answer<String>, InferenceError> {
        let mut core = self.context.lock().unwrap();
        let mut response_str = String::with_capacity(4096);
        let mut brake = Action::Continue;
        let mut n_tokens = 0usize;
        self.n_cur = core.internal_prefill(&self.tokens, true, self.state_id, self.n_cur)?;
        if core.batch.n_tokens() == 0 {
            return Err(InferenceError::EmptyBatch);
        }
        while !brake.brake() {
            let k = constraint.as_mut().map(|x| x.as_mut() as &mut dyn Constraint);
            let logit_idx = core.batch.n_tokens() - 1;
            let token = match core
                .context
                .sample_with_constraint(logit_idx, k)?
            {
                Some(t) => t,
                None => {
                    core.batch.clear();
                    return Ok(Answer::Complete(response_str, None));
                }
            };
            n_tokens += 1;
            match self.vocab.token_to_bytes(token, false) {
                Ok(bytes) => {
                    let last_token = String::from_utf8_lossy(&bytes);
                    brake = Action::brake_on(&response_str, &last_token, n_tokens, 0, filter);
                    response_str.push_str(&last_token);
                }
                Err(e) => {
                    error!("Failed to extract token bytes: {:?}", e);
                    return Err(e.into());
                }
            }

            core.batch.clear();
            if brake == Action::Continue || brake == Action::Delay {
                let r = constraint
                    .as_deref_mut()
                    .map_or(Ok(None), |x| x.forward(token).map(Some))?;
                match r {
                    Some(ConstraintStep::FastForward(ff_tokens)) => {
                        core.batch.add(token, Pos::new(self.n_cur, 0), false)?;
                        let last_token = ff_tokens.len() - 1;
                        for (i, t) in ff_tokens.iter().enumerate() {
                            self.n_cur += 1;
                            core.batch
                                .add(*t, Pos::new(self.n_cur, 0), i == last_token)?;
                        }
                    }
                    Some(ConstraintStep::Forward) | None => {
                        core.batch.add(token, Pos::new(self.n_cur, 0), true)?
                    }
                    Some(ConstraintStep::Stop) => {
                        return Ok(Answer::Complete(response_str, None));
                    }
                }
            }
            if brake == Action::Continue {
                core.internal_decode()?;
            }
            self.tokens.push(token);
            self.n_cur += 1;
        }
        Ok(Answer::Partial(response_str))
    }

    fn clear(&mut self) -> Result<(), InferenceError> {
        self.tokens.clear();
        self.n_cur = 0;
        Ok(())
    }

    fn rollback(&mut self, pos: usize) -> Result<(), InferenceError> {
        if self.tokens.len() > pos {
            self.tokens.truncate(pos)
        }
        if self.n_cur > pos {
            self.n_cur = pos
        }
        Ok(())
    }

    fn pos(&self) -> usize {
        self.tokens.len()
    }

    fn vocab(&self) -> &BoxedVocab {
        &self.vocab
    }

    fn zone(&self) -> ComputationZone {
        self.zone
    }
}
