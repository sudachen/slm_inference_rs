use slm_inference::slm::{VobTokenizer, SimpleTokEnv, FfiError, StringToTokenError, TokenToStringError};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::{Arc, OnceLock};
use toktrie::TokEnv;
use crate::LlamaContextPtr;

unsafe impl Send for Tokenizer{}
unsafe impl Sync for Tokenizer{}

pub struct Tokenizer {
    tok_env: OnceLock<toktrie::TokEnv>,
    vocab_size: usize,
    vocab_ptr: *const llama_cpp_sys_2::llama_vocab,
    #[allow(dead_code)]
    ctx_holder: LlamaContextPtr,
}

impl Tokenizer {
    pub fn new(ctx_holder: LlamaContextPtr, vocab_size: usize, vocab_ptr: *const llama_cpp_sys_2::llama_vocab ) -> Self {
        Self {
            tok_env: OnceLock::new(),
            vocab_size,
            vocab_ptr,
            ctx_holder,
        }
    }
}

impl VobTokenizer for Tokenizer {

    #[inline(never)]
    fn token_to_bytes(
        &self,
        token: i32,
        special: bool,
    ) -> Result<Vec<u8>, TokenToStringError> {
        let mut buf: Vec<u8> = vec![0u8; 128];
        let len = buf.len() as c_int;
        let size = unsafe {
            llama_cpp_sys_2::llama_token_to_piece(
                self.vocab_ptr,
                token as llama_cpp_sys_2::llama_token,
                buf.as_mut_ptr() as *mut c_char,
                len,
                0,
                special,
            )
        };

        match size {
            0 => Ok(vec![]),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                buf.truncate(len);
                Ok(buf)
            }
        }
    }

    #[inline(never)]
    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, StringToTokenError> {
        let add_bos = match add_special {
            true => 1,
            _ => 0,
        };
        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + add_bos);
        let mut buffer: Vec<i32> = Vec::with_capacity(tokens_estimation);
        let c_string = CString::new(str).map_err(|_| FfiError::CstAllocationError)?;
        let buf_capacity = buffer.capacity() as c_int;
        let text_len = c_string.as_bytes().len() as c_int;

        let buf_ptr = buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>();

        let size = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab_ptr,
                c_string.as_ptr(),
                text_len,
                buf_ptr,
                buf_capacity,
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

    #[inline(never)]
    fn tok_env(&self) -> &TokEnv {
        self.tok_env.get_or_init(|| {
            let vocab_size = self.vocab_size;
            let mut last_used = 0;
            let tok_eos = unsafe { llama_cpp_sys_2::llama_vocab_eos(self.vocab_ptr) } as u32;
            let mut words = Vec::with_capacity(vocab_size);
            for i in 0..vocab_size {
                let k = self.token_to_bytes(i as i32,true).unwrap_or_else(|_|vec![]);
                if k.starts_with(b"<unused") {
                    words.push(vec![]);
                } else {
                    words.push(k);
                    last_used = i;
                }
            }
            words.truncate(last_used+1);
            Arc::new(SimpleTokEnv::new(tok_eos, &words))
        })

    }
}
