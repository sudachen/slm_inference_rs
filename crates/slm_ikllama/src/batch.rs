use slm_inference::{SlmPos, SlmToken};
use slm_inference::errors::BatchError;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Token(slm_ikllama_sys::llama_token);

impl SlmToken for Token {
    fn as_i32(&self) -> i32 {
        self.0
    }
    fn from_i32(i: i32) -> Self {
        Self(i)
    }
}

impl Token {
    #[allow(dead_code)]
    pub(crate) fn token(&self) -> slm_ikllama_sys::llama_token {
        self.0
    }
}


impl From<i32> for Token {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

#[derive(Debug)]
pub struct Batch {
    pub allocated: usize,
    pub llama_batch: slm_ikllama_sys::llama_batch,
}

impl Batch {
    #[inline(never)]
    pub fn new(n_tokens: usize, n_seq_max: usize) -> Result<Self, BatchError> {
        let n_tokens_i32 =
            i32::try_from(n_tokens).map_err(|_| BatchError::NtokTooLarge(n_tokens))?;
        let n_seq_max =
            i32::try_from(n_seq_max).map_err(|_| BatchError::NseqTooLarge(n_seq_max))?;
        let batch = unsafe { slm_ikllama_sys::llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        Ok(Batch {
            allocated: n_tokens,
            llama_batch: batch,
        })
    }
}
impl slm_inference::SlmBatch<Token> for Batch {
    #[inline(never)]
    fn add(&mut self, token: Token, pos: SlmPos, logits: bool) -> Result<(), BatchError> {
        let last_pos = usize::try_from(self.n_tokens())
            .map_err(|_| BatchError::InternalError("n_tokens overflow".to_string()))?;
        if self.allocated < last_pos + 1 {
            return Err(BatchError::InsufficientSpace(self.allocated));
        }
        let token = token.0;
        let offset = self.llama_batch.n_tokens;
        let token_pos = i32::try_from(pos.token_pos)
            .map_err(|_| BatchError::NtokTooLarge(pos.token_pos))?
            as slm_ikllama_sys::llama_pos;
        let fork_id =
            i32::try_from(pos.fork_id).map_err(|_| BatchError::NseqTooLarge(pos.fork_id))?;
        let offset_usize = usize::try_from(offset)
            .map_err(|_| BatchError::InternalError("buffer offest is negative".to_string()))?;
        unsafe {
            self.llama_batch.token.add(offset_usize).write(token);
            self.llama_batch.pos.add(offset_usize).write(token_pos);
            self.llama_batch.n_seq_id.add(offset_usize).write(1);
            (*self.llama_batch.seq_id.add(offset_usize))
                .add(0)
                .write(fork_id);
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }
        self.llama_batch.n_tokens += 1;
        Ok(())
    }

    fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
    }

    fn n_tokens(&self) -> usize {
        self.llama_batch.n_tokens as usize
    }

    fn n_max(&self) -> usize {
        self.allocated
    }
}

impl<'a> Drop for Batch {
    #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                slm_ikllama_sys::llama_batch_free(self.llama_batch);
            }
        }
    }
}
