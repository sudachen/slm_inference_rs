use super::{Batch, Model, vocab::Tokenizer};
use slm_inference::slm;
use slm_inference::slm::ComputationZone;
use std::sync::Arc;

unsafe impl Send for Context {}

pub struct Context {
    vocab_ptr: *const slm_ikllama_sys::llama_vocab,
    n_batch: u32,
    n_vocab: usize,
    ctx: super::LlamaContextPtr,
    edit_level: slm::EditLevel,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    #[allow(dead_code)]
    model: Model,
    vocab: slm::BoxedVocab,
}

impl slm::Context for Context {
    type Batch = Batch;
    fn vocab(&self) -> &slm::BoxedVocab {
        &self.vocab
    }

    fn zone(&self) -> ComputationZone {
        self.model.zone
    }

    fn new_batch(&self, tokens: usize, sequences: usize) -> Result<Batch, slm::BatchError> {
        Batch::new(tokens, sequences)
    }

    fn max_batch_len(&self) -> usize {
        self.n_batch as usize
    }

    #[inline(never)]
    fn decode(&mut self, batch: &mut Batch) -> Result<(), slm::DecodeError> {
        let result =
            unsafe { slm_ikllama_sys::llama_decode(self.ctx.get_ptr(), batch.llama_batch) };

        if result != 0 {
            return Err(slm::DecodeError::from(result));
        }

        Ok(())
    }

    #[inline(never)]
    fn sample_with_constraint(
        &mut self,
        logit_idx: usize,
        constraint: Option<&mut dyn slm::Constraint>,
    ) -> Result<Option<i32>, slm::SamplingError> {
        unsafe {
            let ctx = self.ctx.get_ptr();
            // TODO: validate logit_idx
            let logits_ptr = slm_ikllama_sys::llama_get_logits_ith(ctx, logit_idx as i32);
            let logits = std::slice::from_raw_parts_mut(logits_ptr, self.n_vocab);
            if let Some(c) = constraint {
                if !c.mask(logits)? {
                    return Ok(None);
                }
            }
            let mut candidates_vec: Vec<slm_ikllama_sys::llama_token_data> = (0..self.n_vocab)
                .map(|id| slm_ikllama_sys::llama_token_data {
                    id: id as slm_ikllama_sys::llama_token,
                    logit: logits[id],
                    p: 0.0,
                })
                .collect();

            let mut candidates_array = slm_ikllama_sys::llama_token_data_array {
                data: candidates_vec.as_mut_ptr(),
                size: self.n_vocab,
                selected: 0,
                sorted: false,
            };

            let token = if self.temperature <= 0.0 {
                slm_ikllama_sys::llama_sample_token_greedy(ctx, &mut candidates_array)
            } else {
                slm_ikllama_sys::llama_sample_top_k(ctx, &mut candidates_array, self.top_k, 1);
                slm_ikllama_sys::llama_sample_temp(ctx, &mut candidates_array, self.temperature);
                slm_ikllama_sys::llama_sample_softmax(ctx, &mut candidates_array);
                slm_ikllama_sys::llama_sample_top_p(ctx, &mut candidates_array, self.top_p, 1);
                slm_ikllama_sys::llama_sample_token(ctx, &mut candidates_array)
            };

            if slm_ikllama_sys::llama_vocab_is_eog(self.vocab_ptr, token) {
                Ok(None)
            } else {
                Ok(Some(token.into()))
            }
        }
    }

    #[inline(never)]
    fn clear(&mut self) -> Result<(), slm::ContextError> {
        let ctx = self.ctx.get_ptr();
        unsafe { slm_ikllama_sys::llama_kv_cache_clear(ctx) };
        Ok(())
    }

    fn drop(&mut self, fork_id: usize) -> Result<(), slm::ContextError> {
        let ctx = self.ctx.get_ptr();
        unsafe { slm_ikllama_sys::llama_kv_cache_seq_rm(ctx, fork_id as i32, -1, -1) };
        Ok(())
    }

    #[inline(never)]
    fn truncate(&mut self, pos: &slm::Pos) -> Result<slm::Pos, slm::ContextError> {
        let slm::Pos { token_pos, fork_id } = *pos;
        let ctx = self.ctx.get_ptr();
        unsafe {
            slm_ikllama_sys::llama_kv_cache_seq_rm(ctx, fork_id as i32, token_pos as i32, -1)
        };
        Ok(slm::Pos::new(token_pos, fork_id))
    }

    #[inline(never)]
    fn cut(
        &mut self,
        start_pos: &slm::Pos,
        end_pos: &slm::Pos,
    ) -> Result<slm::Pos, slm::ContextError> {
        if start_pos.fork_id != end_pos.fork_id {
            return Err(slm::ContextError::Error(
                "positions must have the same fork_id".to_string(),
            ));
        }
        if start_pos.token_pos < end_pos.token_pos {
            return Err(slm::ContextError::Error(
                "start_pos must be before end_pos".to_string(),
            ));
        }
        let ctx = self.ctx.get_ptr();
        unsafe {
            slm_ikllama_sys::llama_kv_cache_seq_rm(
                ctx,
                start_pos.fork_id as i32,
                start_pos.token_pos as i32,
                end_pos.token_pos as i32 - 1,
            );
        }
        let pos_n = end_pos.token_pos - start_pos.token_pos;
        unsafe {
            slm_ikllama_sys::llama_kv_cache_seq_add(
                ctx,
                start_pos.fork_id as i32,
                end_pos.token_pos as i32,
                -1,
                pos_n as i32,
            );
        }
        let next_pos = unsafe {
            slm_ikllama_sys::llama_kv_cache_seq_pos_max(ctx, start_pos.fork_id as i32) + 1
        };
        Ok(slm::Pos::new(next_pos as usize, start_pos.fork_id))
    }

    fn dump(&mut self, fork_id: usize) -> Result<Vec<u8>, slm::ContextError> {
        let ctx = self.ctx.get_ptr();
        let flags = 0; // copy to memory
        let size = unsafe { slm_ikllama_sys::llama_state_seq_get_size(ctx, fork_id as i32, flags) };
        if size == 0 {
            return Ok(vec![]);
        }
        let mut data = vec![0u8; size];
        unsafe {
            slm_ikllama_sys::llama_state_seq_get_data(
                ctx,
                data.as_mut_ptr(),
                size,
                fork_id as i32,
                flags,
            )
        };
        Ok(data)
    }

    fn restore(&mut self, fork_id: usize, data: &[u8]) -> Result<(), slm::ContextError> {
        let ctx = self.ctx.get_ptr();
        let flags = 0; // copy to memory
        unsafe { slm_ikllama_sys::llama_kv_cache_seq_rm(ctx, fork_id as i32, -1, -1) };
        unsafe {
            slm_ikllama_sys::llama_state_seq_set_data(
                ctx,
                data.as_ptr(),
                data.len(),
                fork_id as i32,
                flags,
            )
        };
        Ok(())
    }

    fn edit_level(&self) -> slm::EditLevel {
        self.edit_level
    }
}

pub struct Builder {
    model: Model,
    params: slm_ikllama_sys::llama_context_params,
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

#[repr(u32)]
#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KVType {
    Q4_0 = slm_ikllama_sys::GGML_TYPE_Q4_0,
    Q5_0 = slm_ikllama_sys::GGML_TYPE_Q5_0,
    Q6_0 = slm_ikllama_sys::GGML_TYPE_Q6_0,
    Q8_0 = slm_ikllama_sys::GGML_TYPE_Q8_0,
    F16 = slm_ikllama_sys::GGML_TYPE_F16,
    F32 = slm_ikllama_sys::GGML_TYPE_F32,
}

impl KVType {
    pub fn from(t: slm::KvType) -> Option<(KVType, bool)> {
        match t {
            slm::KvType::Q4 => Some((KVType::Q4_0, true)),
            slm::KvType::Q5 => Some((KVType::Q5_0, true)),
            slm::KvType::Q6 => Some((KVType::Q6_0, true)),
            slm::KvType::Q8 => Some((KVType::Q8_0, true)),
            slm::KvType::RawQ8 => Some((KVType::Q8_0, false)),
            slm::KvType::F16 => Some((KVType::F16, false)),
            slm::KvType::F32 => Some((KVType::F32, false)),
        }
    }
}
impl Builder {
    #[inline(never)]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            params: unsafe { slm_ikllama_sys::llama_context_default_params() },
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
        }
    }
    #[allow(dead_code)]
    pub fn with_flash_attn(mut self) -> Self {
        self.params.flash_attn = true;
        self
    }

    #[allow(dead_code)]
    pub fn with_kv_hadamard(mut self, k: bool, v: bool) -> Self {
        self.params.flash_attn = true;
        self.params.k_cache_hadamard = k;
        self.params.v_cache_hadamard = v;
        self
    }
    #[allow(dead_code)]
    pub fn with_type_kv(mut self, type_k: KVType, type_v: KVType) -> Self {
        self.params.flash_attn = true;
        if type_k == KVType::Q4_0 || type_k == KVType::Q5_0 {
            self.params.k_cache_hadamard = true;
        }
        if type_k == KVType::Q4_0 || type_k == KVType::Q5_0 {
            self.params.v_cache_hadamard = true;
        }
        self.params.type_k = type_k as u32;
        self.params.type_v = type_v as u32;
        self
    }
}

impl slm::ContextBuilder<Context> for Builder {
    #[inline(never)]
    fn build(mut self) -> Result<Context, slm::ContextBuilderError> {
        let ctx =
            unsafe { slm_ikllama_sys::llama_init_from_model(self.model.get_ptr()?, self.params) };

        let model_ptr = self.model.get_const_ptr()?;
        let vocab_ptr = unsafe { slm_ikllama_sys::llama_model_get_vocab(model_ptr) };
        let n_vocab = unsafe { slm_ikllama_sys::llama_n_vocab(model_ptr) } as usize;

        let ctx = super::LlamaContextPtr::new(ctx);
        let vocab = Arc::new(slm::SimpleVocab::new(Tokenizer::new(
            ctx.clone(),
            n_vocab,
            model_ptr,
        )));
        // TODO: decide by arch from model metadata
        let edit_level = slm::EditLevel::Cut;

        Ok(Context {
            ctx,
            vocab_ptr,
            n_batch: self.params.n_batch,
            n_vocab,
            edit_level,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            model: self.model,
            vocab,
        })
    }

    #[inline(never)]
    fn with_sampler(mut self, temperature: f32, top_k: i32, top_p: f32) -> Self {
        self.temperature = temperature;
        self.top_k = top_k;
        self.top_p = top_p;
        self
    }

    #[inline(never)]
    fn with_n_ctx(mut self, n_ctx: usize) -> Self {
        self.params.n_ctx = n_ctx as u32;
        self
    }

    #[inline(never)]
    fn with_n_batch(mut self, n_batch: usize) -> Self {
        self.params.n_batch = n_batch as u32;
        self
    }

    #[inline(never)]
    fn with_gen_type_kv(mut self, k: slm::KvType, v: slm::KvType) -> Self {
        let (k, kh) = KVType::from(k).unwrap();
        let (v, vh) = KVType::from(v).unwrap();
        self.params.flash_attn = true;
        self.params.type_k = k as u32;
        self.params.k_cache_hadamard = kh;
        self.params.type_v = v as u32;
        self.params.v_cache_hadamard = vh;
        self
    }

    #[inline(never)]
    fn with_flash_attn(mut self, enable: bool) -> Self {
        self.params.flash_attn = enable;
        self
    }
}
