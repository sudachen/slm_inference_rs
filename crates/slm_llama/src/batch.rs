use slm_inference::errors::BatchError;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Token(llama_cpp_sys_2::llama_token);

impl slm_inference::SlmToken for Token {
    fn as_i32(&self) -> i32 {
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
    pub initialized_logits: Vec<llama_cpp_sys_2::llama_token>,
    pub llama_batch: llama_cpp_sys_2::llama_batch,
}

impl Batch {
    pub fn new(n_tokens: usize, n_seq_max: usize) -> Result<Self, BatchError> {
        let n_tokens_i32 =
            i32::try_from(n_tokens).map_err(|_| BatchError::NtokTooLarge(n_tokens))?;
        let n_seq_max =
            i32::try_from(n_seq_max).map_err(|_| BatchError::NseqTooLarge(n_seq_max))?;
        let batch = unsafe { llama_cpp_sys_2::llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        Ok(Batch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
        })
    }
}

impl slm_inference::SlmBatch<Token> for Batch {
    fn add(
        &mut self,
        token: Token,
        pos: usize,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchError::InsufficientSpace(self.allocated));
        }
        let token = token.0;
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            self.llama_batch.token.add(offset_usize).write(token);
            self.llama_batch
                .pos
                .add(offset_usize)
                .write(pos as llama_cpp_sys_2::llama_pos);
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_cpp_sys_2::llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }
        self.llama_batch.n_tokens += 1;
        Ok(())
    }

    fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    fn n_tokens(&self) -> usize {
        self.llama_batch.n_tokens as usize
    }
}

impl<'a> Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_cpp_sys_2::llama_batch_free(self.llama_batch);
            }
        }
    }
}
