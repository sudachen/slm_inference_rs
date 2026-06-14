use crate::batch::{Batch, Token};
use crate::model::Model;
use std::ffi::CString;
use std::os::raw::c_int;

use slm_inference::core::shared_ptr::{Free, SharedPtr};
use slm_inference::errors::{
    BatchError, ContextBuilderError, ContextError, DecodeError, FfiError, SamplingError,
    StringToTokenError, TokenToStringError,
};
use slm_inference::{SlmContext, SlmContextBuilder, SlmEditLevel, SlmKvType, SlmPos, SlmToken};

#[derive(Clone)]
struct LlamaContextFree;
impl Free<slm_ikllama_sys::llama_context> for LlamaContextFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut slm_ikllama_sys::llama_context) {
        unsafe { slm_ikllama_sys::llama_free(ptr) };
    }
}

#[derive(Clone)]
pub struct Context {
    vocab_ptr: *const slm_ikllama_sys::llama_vocab,
    n_batch: u32,
    n_vocab: usize,
    ctx: SharedPtr<slm_ikllama_sys::llama_context, LlamaContextFree>,
    edit_level: SlmEditLevel,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    #[allow(dead_code)]
    model: Model,
}

impl SlmContext for Context {
    type Token = Token;
    type Batch = Batch;

    fn new_batch(&self, tokens: usize, sequences: usize) -> Result<Batch, BatchError> {
        Batch::new(tokens, sequences)
    }

    fn max_batch_len(&self) -> usize {
        self.n_batch as usize
    }

    #[inline(never)]
    fn decode(&mut self, batch: &mut Batch) -> Result<(), DecodeError> {
        let result =
            unsafe { slm_ikllama_sys::llama_decode(self.ctx.get_ptr(), batch.llama_batch) };

        if result != 0 {
            return Err(DecodeError::from(result));
        }

        Ok(())
    }

    #[inline(never)]
    fn sample(&mut self, logit_idx: usize) -> Result<Option<Self::Token>, SamplingError> {
        unsafe {
            let ctx = self.ctx.get_ptr();
            // TODO: validate logit_idx
            let logits_ptr = slm_ikllama_sys::llama_get_logits_ith(ctx, logit_idx as i32);
            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab);
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
                slm_ikllama_sys::llama_sample_top_p(ctx, &mut candidates_array, self.top_p, 1);
                slm_ikllama_sys::llama_sample_temp(ctx, &mut candidates_array, self.temperature);
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
    fn token_to_bytes(
        &self,
        token: Self::Token,
        buffer_size: usize,
        special: bool,
        lstrip: Option<usize>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip
            .map_or(Ok(0), i32::try_from)
            .map_err(|_| TokenToStringError::InvalidLstrip)?;
        let size = unsafe {
            slm_ikllama_sys::llama_token_to_piece(
                self.model.get_const_ptr()?,
                token.as_i32() as slm_ikllama_sys::llama_token,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Ok(vec![]),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }

    #[inline(never)]
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Self::Token>, StringToTokenError> {
        let add_bos = match add_special {
            true => 1,
            _ => 0,
        };
        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + add_bos);
        let mut buffer: Vec<Self::Token> = Vec::with_capacity(tokens_estimation);
        let c_string = CString::new(str).map_err(|_| FfiError::CstAllocationError)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).map_err(|_| FfiError::CintConversionError)?;

        let text_len = c_int::try_from(c_string.as_bytes().len())
            .map_err(|_| FfiError::CintConversionError)?;

        let size = unsafe {
            slm_ikllama_sys::llama_tokenize(
                self.model.get_const_ptr()?,
                c_string.as_ptr(),
                text_len,
                buffer.as_mut_ptr().cast::<slm_ikllama_sys::llama_token>(),
                buffer_capacity,
                add_special,
                parse_special,
            )
        };

        // if we fail the first time we can resize the vector to the correct size and try again. This should never fail.
        // as a result - size is guaranteed to be positive here.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                slm_ikllama_sys::llama_tokenize(
                    self.model.get_const_ptr()?,
                    c_string.as_ptr(),
                    text_len,
                    buffer.as_mut_ptr().cast::<slm_ikllama_sys::llama_token>(),
                    -size,
                    add_special,
                    parse_special,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }
        Ok(buffer)
    }

    #[inline(never)]
    fn clear(&mut self) -> Result<(), ContextError> {
        let ctx = self.ctx.get_ptr();
        unsafe { slm_ikllama_sys::llama_kv_cache_clear(ctx) };
        Ok(())
    }

    #[inline(never)]
    fn truncate(&mut self, pos: &SlmPos) -> Result<SlmPos, ContextError> {
        let SlmPos { token_pos, fork_id } = *pos;
        let ctx = self.ctx.get_ptr();
        unsafe {
            slm_ikllama_sys::llama_kv_cache_seq_rm(ctx, fork_id as i32, token_pos as i32, -1)
        };
        Ok(SlmPos::new(token_pos, fork_id))
    }

    #[inline(never)]
    fn cut(&mut self, start_pos: &SlmPos, end_pos: &SlmPos) -> Result<SlmPos, ContextError> {
        if start_pos.fork_id != end_pos.fork_id {
            return Err(ContextError::Error(
                "positions must have the same fork_id".to_string(),
            ));
        }
        if start_pos.token_pos < end_pos.token_pos {
            return Err(ContextError::Error(
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
        Ok(SlmPos::new(next_pos as usize, start_pos.fork_id))
    }

    fn drop(&mut self, fork_id: usize) -> Result<(), ContextError> {
        let ctx = self.ctx.get_ptr();
        unsafe { slm_ikllama_sys::llama_kv_cache_seq_rm(ctx, fork_id as i32, -1, -1) };
        Ok(())
    }

    fn dump(&mut self) -> Result<Vec<u8>, ContextError> {
        todo!()
    }

    fn restore(&mut self, _data: Vec<u8>) -> Result<(), ContextError> {
        todo!()
    }

    fn edit_level(&self) -> SlmEditLevel {
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
    pub fn from(t: SlmKvType) -> Option<(KVType, bool)> {
        match t {
            SlmKvType::Q4 => Some((KVType::Q4_0, true)),
            SlmKvType::Q5 => Some((KVType::Q5_0, true)),
            SlmKvType::Q6 => Some((KVType::Q6_0, true)),
            SlmKvType::Q8 => Some((KVType::Q8_0, true)),
            SlmKvType::RawQ8 => Some((KVType::Q8_0, false)),
            SlmKvType::F16 => Some((KVType::F16, false)),
            SlmKvType::F32 => Some((KVType::F32, false)),
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

impl SlmContextBuilder<Context> for Builder {
    #[inline(never)]
    fn build(mut self) -> Result<Context, ContextBuilderError> {
        let ctx =
            unsafe { slm_ikllama_sys::llama_init_from_model(self.model.get_ptr()?, self.params) };

        let model_ptr = self.model.get_const_ptr()?;
        let vocab_ptr = unsafe { slm_ikllama_sys::llama_model_get_vocab(model_ptr) };
        let n_vocab = unsafe { slm_ikllama_sys::llama_n_vocab(model_ptr) } as usize;

        // TODO: decide by arch from model metadata
        let edit_level = SlmEditLevel::Cut;

        Ok(Context {
            ctx: SharedPtr::new(ctx),
            vocab_ptr,
            n_batch: self.params.n_batch,
            n_vocab,
            edit_level,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            model: self.model,
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
    fn with_gen_type_kv(mut self, k: SlmKvType, v: SlmKvType) -> Self {
        let (k, kh) = KVType::from(k).unwrap();
        let (v, vh) = KVType::from(v).unwrap();
        self.params.type_k = k as u32;
        self.params.k_cache_hadamard = kh;
        self.params.type_v = v as u32;
        self.params.v_cache_hadamard = vh;
        self
    }
}
