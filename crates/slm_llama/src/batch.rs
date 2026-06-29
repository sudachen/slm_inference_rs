use slm_inference::slm;

unsafe impl Send for Batch {}

#[derive(Debug)]
pub struct Batch {
    pub allocated: usize,
    pub llama_batch: llama_cpp_sys_2::llama_batch,
}

impl Batch {
    #[inline(never)]
    pub fn new(n_tokens: usize, n_seq_max: usize) -> Result<Self, slm::BatchError> {
        let n_tokens_i32 =
            i32::try_from(n_tokens).map_err(|_| slm::BatchError::NtokTooLarge(n_tokens))?;
        let n_seq_max =
            i32::try_from(n_seq_max).map_err(|_| slm::BatchError::NseqTooLarge(n_seq_max))?;
        let batch = unsafe { llama_cpp_sys_2::llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        Ok(Batch {
            allocated: n_tokens,
            llama_batch: batch,
        })
    }
}
impl slm::Batch for Batch {
    #[inline(never)]
    fn add(&mut self, token: i32, pos: slm::Pos, logits: bool) -> Result<(), slm::BatchError> {
        let last_pos = usize::try_from(self.n_tokens())
            .map_err(|_| slm::BatchError::InternalError("n_tokens overflow".to_string()))?;
        if self.allocated < last_pos + 1 {
            return Err(slm::BatchError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let token_pos = i32::try_from(pos.token_pos)
            .map_err(|_| slm::BatchError::NtokTooLarge(pos.token_pos))?
            as llama_cpp_sys_2::llama_pos;
        let fork_id =
            i32::try_from(pos.fork_id).map_err(|_| slm::BatchError::NseqTooLarge(pos.fork_id))?;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            self.llama_batch.token.add(offset_usize).write(token);
            self.llama_batch.pos.add(offset_usize).write(token_pos);
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_cpp_sys_2::llama_seq_id::try_from(1)
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
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
                llama_cpp_sys_2::llama_batch_free(self.llama_batch);
            }
        }
    }
}
