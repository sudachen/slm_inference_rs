use crate::batch::Batch;
use crate::model::Model;
use crate::vocab::Tokenizer;
use crate::{LlamaContextPtr, LlamaSamplerPtr};
use slm_inference::slm;
use slm_inference::slm::ComputationZone;
use std::sync::Arc;

unsafe impl Send for Context {}

#[derive(Clone)]
pub struct Context {
    vocab_ptr: *const llama_cpp_sys_2::llama_vocab,
    n_batch: u32,
    n_vocab: usize,
    sampler: LlamaSamplerPtr,
    ctx: LlamaContextPtr,
    #[allow(dead_code)]
    model: Model,
    edit_level: slm::EditLevel,
    vocab: slm::BoxedVocab,
}

impl slm::Context for Context {
    type Batch = Batch;

    fn vocab(&self) -> &slm::BoxedVocab {
        &self.vocab
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
            unsafe { llama_cpp_sys_2::llama_decode(self.ctx.get_ptr(), batch.llama_batch) };

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
        if let Some(c) = constraint {
            let logits_ptr = unsafe {
                llama_cpp_sys_2::llama_get_logits_ith(self.ctx.get_ptr(), logit_idx as i32)
            };
            let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, self.n_vocab) };
            c.mask(logits)?;
        }
        let token = unsafe {
            llama_cpp_sys_2::llama_sampler_sample(
                self.sampler.get_ptr(),
                self.ctx.get_ptr(),
                logit_idx as i32,
            )
        };
        //let is_control = unsafe { llama_cpp_sys_2::llama_vocab_is_control(self.vocab_ptr, token) };
        let is_eog = unsafe { llama_cpp_sys_2::llama_vocab_is_eog(self.vocab_ptr, token) };
        if is_eog {
            return Ok(None);
        }
        Ok(Some(token.into()))
    }

    #[inline(never)]
    fn clear(&mut self) -> Result<(), slm::ContextError> {
        let memory = unsafe { llama_cpp_sys_2::llama_get_memory(self.ctx.get_ptr()) };
        if memory.is_null() {
            return Err(slm::FfiError::NullPtr.into());
        }
        unsafe { llama_cpp_sys_2::llama_memory_clear(memory, true) };
        Ok(())
    }

    #[inline(never)]
    fn truncate(&mut self, pos: &slm::Pos) -> Result<slm::Pos, slm::ContextError> {
        let memory = unsafe { llama_cpp_sys_2::llama_get_memory(self.ctx.get_ptr()) };
        if memory.is_null() {
            return Err(slm::FfiError::NullPtr.into());
        }
        let slm::Pos { token_pos, fork_id } = *pos;
        unsafe {
            llama_cpp_sys_2::llama_memory_seq_rm(memory, fork_id as i32, token_pos as i32, -1)
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
        let memory = unsafe { llama_cpp_sys_2::llama_get_memory(self.ctx.get_ptr()) };
        if memory.is_null() {
            return Err(slm::FfiError::NullPtr.into());
        }
        unsafe {
            llama_cpp_sys_2::llama_memory_seq_rm(
                memory,
                start_pos.fork_id as i32,
                start_pos.token_pos as i32,
                end_pos.token_pos as i32 - 1,
            );
        }
        let pos_n = end_pos.token_pos - start_pos.token_pos;
        unsafe {
            llama_cpp_sys_2::llama_memory_seq_add(
                memory,
                start_pos.fork_id as i32,
                end_pos.token_pos as i32,
                -1,
                pos_n as i32,
            );
        }
        let next_pos = unsafe {
            llama_cpp_sys_2::llama_memory_seq_pos_max(memory, start_pos.fork_id as i32) + 1
        };
        Ok(slm::Pos::new(next_pos as usize, start_pos.fork_id))
    }

    fn drop(&mut self, fork_id: usize) -> Result<(), slm::ContextError> {
        let memory = unsafe { llama_cpp_sys_2::llama_get_memory(self.ctx.get_ptr()) };
        if memory.is_null() {
            return Err(slm::FfiError::NullPtr.into());
        }
        unsafe { llama_cpp_sys_2::llama_memory_seq_rm(memory, fork_id as i32, -1, -1) };
        Ok(())
    }

    fn dump(&mut self) -> Result<Vec<u8>, slm::ContextError> {
        todo!()
    }

    fn restore(&mut self, _data: Vec<u8>) -> Result<(), slm::ContextError> {
        todo!()
    }

    fn edit_level(&self) -> slm::EditLevel {
        self.edit_level
    }

    fn zone(&self) -> ComputationZone {
        self.model.zone
    }
}

pub struct Builder {
    model: Model,
    params: llama_cpp_sys_2::llama_context_params,
    temperature: f32,
    top_k: i32,
    top_p: f32,
}

#[repr(u32)]
#[allow(dead_code)]
pub enum KVType {
    Q4_0 = llama_cpp_sys_2::GGML_TYPE_Q4_0,
    Q5_0 = llama_cpp_sys_2::GGML_TYPE_Q5_0,
    Q8_0 = llama_cpp_sys_2::GGML_TYPE_Q8_0,
    F16 = llama_cpp_sys_2::GGML_TYPE_F16,
    F32 = llama_cpp_sys_2::GGML_TYPE_F32,
}

impl KVType {
    pub fn from(t: slm::KvType) -> Option<KVType> {
        match t {
            slm::KvType::Q4 => Some(KVType::Q4_0),
            slm::KvType::Q5 => Some(KVType::Q5_0),
            slm::KvType::Q6 => Some(KVType::Q8_0),
            slm::KvType::Q8 => Some(KVType::Q8_0),
            slm::KvType::RawQ8 => Some(KVType::Q8_0),
            slm::KvType::F16 => Some(KVType::F16),
            slm::KvType::F32 => Some(KVType::F32),
        }
    }
}

impl Builder {
    #[inline(never)]
    pub fn new(model: Model) -> Self {
        let mut params = unsafe { llama_cpp_sys_2::llama_context_default_params() };
        params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        Self {
            model,
            params,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
        }
    }

    #[allow(dead_code)]
    pub fn with_type_kv(mut self, type_k: KVType, type_v: KVType) -> Self {
        self.params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.params.type_k = type_k as u32;
        self.params.type_v = type_v as u32;
        self
    }
}

impl slm::ContextBuilder<Context> for Builder {
    #[allow(dead_code)]
    fn with_n_ctx(mut self, n_ctx: usize) -> Self {
        self.params.n_ctx = n_ctx as u32;
        self
    }

    #[allow(dead_code)]
    fn with_n_batch(mut self, n_batch: usize) -> Self {
        self.params.n_batch = n_batch as u32;
        self
    }

    #[inline(never)]
    fn with_gen_type_kv(self, type_k: slm::KvType, type_v: slm::KvType) -> Self {
        let type_k = KVType::from(type_k).unwrap_or(KVType::Q8_0);
        let type_v = KVType::from(type_v).unwrap_or(KVType::Q8_0);
        self.with_type_kv(type_k, type_v)
    }

    #[inline(never)]
    fn build(mut self) -> Result<Context, slm::ContextBuilderError> {
        let ctx =
            unsafe { llama_cpp_sys_2::llama_init_from_model(self.model.get_ptr()?, self.params) };

        let model_ptr = self.model.get_const_ptr()?;
        let vocab_ptr = unsafe { llama_cpp_sys_2::llama_model_get_vocab(model_ptr) };
        let n_vocab = unsafe { llama_cpp_sys_2::llama_vocab_n_tokens(vocab_ptr) as usize };
        let sampler = unsafe {
            let sampler = llama_cpp_sys_2::llama_sampler_chain_init(
                llama_cpp_sys_2::llama_sampler_chain_default_params(),
            );
            if self.temperature <= 0.0 {
                llama_cpp_sys_2::llama_sampler_chain_add(
                    sampler,
                    llama_cpp_sys_2::llama_sampler_init_greedy(),
                );
            } else {
                llama_cpp_sys_2::llama_sampler_chain_add(
                    sampler,
                    llama_cpp_sys_2::llama_sampler_init_top_k(self.top_k),
                );
                llama_cpp_sys_2::llama_sampler_chain_add(
                    sampler,
                    llama_cpp_sys_2::llama_sampler_init_top_p(self.top_p, 1),
                );
                llama_cpp_sys_2::llama_sampler_chain_add(
                    sampler,
                    llama_cpp_sys_2::llama_sampler_init_temp(self.temperature),
                );
                llama_cpp_sys_2::llama_sampler_chain_add(
                    sampler,
                    llama_cpp_sys_2::llama_sampler_init_dist(42),
                );
            }
            LlamaSamplerPtr::new(sampler)
        };

        let ctx = LlamaContextPtr::new(ctx);
        let edit_level = slm::EditLevel::Cut;
        let vocab = Arc::new(slm::SimpleVocab::new(Tokenizer::new(
            ctx.clone(),
            n_vocab,
            vocab_ptr,
        )));

        Ok(Context {
            ctx,
            vocab_ptr,
            n_batch: self.params.n_batch,
            n_vocab,
            model: self.model,
            sampler,
            edit_level,
            vocab,
        })
    }

    fn with_sampler(mut self, temperature: f32, top_k: i32, top_p: f32) -> Self {
        self.temperature = temperature;
        self.top_k = top_k;
        self.top_p = top_p;
        self
    }

    #[allow(dead_code)]
    fn with_flash_attn(mut self, enable: bool) -> Self {
        if enable {
            self.params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else {
            self.params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        self
    }
}
