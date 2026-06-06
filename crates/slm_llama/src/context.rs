use std::ffi::CString;
use std::os::raw::c_int;

use super::model::Model;
use crate::batch::{Batch, Token};

use slm_inference::core::shared_ptr::{Free, SharedPtr};
use slm_inference::errors::{BatchError, DecodeError, TokenToStringError};
use slm_inference::{ContextBuilderError, FfiError, SamplingError, SlmToken, StringToTokenError};

#[derive(Clone)]
struct LlamaSamplerFree;
impl Free<llama_cpp_sys_2::llama_sampler> for LlamaSamplerFree {
    #[inline]
    unsafe fn free(ptr: *mut llama_cpp_sys_2::llama_sampler) {
        unsafe { llama_cpp_sys_2::llama_sampler_free(ptr) };
    }
}

#[derive(Clone)]
struct LlamaContextFree;
impl Free<llama_cpp_sys_2::llama_context> for LlamaContextFree {
    #[inline]
    unsafe fn free(ptr: *mut llama_cpp_sys_2::llama_context) {
        unsafe { llama_cpp_sys_2::llama_free(ptr) };
    }
}

#[derive(Clone)]
pub struct Context {
    //ctx: *mut llama_cpp_sys_2::llama_context,
    //model_ptr: *const llama_cpp_sys_2::llama_model,
    vocab_ptr: *const llama_cpp_sys_2::llama_vocab,
    n_batch: u32,
    sampler: SharedPtr<llama_cpp_sys_2::llama_sampler, LlamaSamplerFree>,
    ctx: SharedPtr<llama_cpp_sys_2::llama_context, LlamaContextFree>,
    #[allow(dead_code)]
    model: Model,
}

impl slm_inference::SlmContext for Context {
    type Token = Token;
    type Batch = Batch;

    fn new_batch(&self, tokens: usize, sequences: usize) -> Result<Batch, BatchError> {
        Batch::new(tokens, sequences)
    }

    fn max_batch_len(&self) -> usize {
        self.n_batch as usize
    }

    fn decode(&mut self, batch: &mut Batch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_decode(self.ctx.get_ptr(), batch.llama_batch) };

        if result != 0 {
            return Err(DecodeError::from(result));
        }

        Ok(())
    }

    fn sample(&mut self, logit_idx: usize) -> Result<Option<Self::Token>, SamplingError> {
        let token = unsafe {
            llama_cpp_sys_2::llama_sampler_sample(
                self.sampler.get_ptr(),
                self.ctx.get_ptr(),
                logit_idx as i32,
            )
        };
        match unsafe { llama_cpp_sys_2::llama_vocab_is_eog(self.vocab_ptr, token) } {
            false => Ok(Some(token.into())),
            _ => Ok(None),
        }
    }

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
            llama_cpp_sys_2::llama_token_to_piece(
                self.vocab_ptr,
                token.as_i32() as llama_cpp_sys_2::llama_token,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
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
            llama_cpp_sys_2::llama_tokenize(
                self.vocab_ptr,
                c_string.as_ptr(),
                text_len,
                buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
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
                llama_cpp_sys_2::llama_tokenize(
                    self.vocab_ptr,
                    c_string.as_ptr(),
                    text_len,
                    buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
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

impl Builder {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            params: unsafe { llama_cpp_sys_2::llama_context_default_params() },
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
        }
    }

    #[allow(dead_code)]
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.params.n_ctx = n_ctx;
        self
    }

    #[allow(dead_code)]
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.params.n_batch = n_batch;
        self
    }

    #[allow(dead_code)]
    pub fn with_flash_attn(mut self) -> Self {
        self.params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self
    }

    #[allow(dead_code)]
    pub fn with_type_kv(mut self, type_k: KVType, type_v: KVType) -> Self {
        self.params.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.params.type_k = type_k as u32;
        self.params.type_v = type_v as u32;
        self
    }
}

impl slm_inference::SlmContextBuilder<Context> for Builder {
    fn build(mut self) -> Result<Context, ContextBuilderError> {
        let ctx =
            unsafe { llama_cpp_sys_2::llama_init_from_model(self.model.get_ptr()?, self.params) };

        let model_ptr = self.model.get_const_ptr()?;
        let vocab_ptr = unsafe { llama_cpp_sys_2::llama_model_get_vocab(model_ptr) };
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
            }
            SharedPtr::new(sampler)
        };

        Ok(Context {
            ctx: SharedPtr::new(ctx),
            vocab_ptr,
            n_batch: self.params.n_batch,
            model: self.model,
            sampler,
        })
    }

    fn with_sampler(mut self, temperature: f32, top_k: i32, top_p: f32) -> Self {
        self.temperature = temperature;
        self.top_k = top_k;
        self.top_p = top_p;
        self
    }
}
