use crate::errors::InferenceError;
use crate::{SlmBatch, SlmContext};
use tracing::error;

pub trait SlmInference {
    fn inference(&mut self, prompt: &str, max_tokens: u32) -> Result<String, InferenceError>;
}

impl<C: SlmContext> SlmInference for C {
    fn inference(&mut self, prompt: &str, max_tokens: u32) -> Result<String, InferenceError> {
        let (mut batch, mut n_cur) = {
            let tokens_list = self.str_to_tokens(prompt, true, true)?;
            if tokens_list.is_empty() {
                return Ok(String::new());
            }
            let last_index = tokens_list.len() - 1;

            let n_batch = self.max_batch_len();
            let mut batch = self.new_batch(n_batch, 1)?;
            for (chunk_idx, chunk) in tokens_list.chunks(n_batch).enumerate() {
                batch.clear();
                for (token_idx, token) in chunk.iter().enumerate() {
                    let absolute_pos = chunk_idx * n_batch + token_idx;
                    let is_last_token = absolute_pos == last_index;
                    batch.add(*token, absolute_pos, &[0], is_last_token)?;
                }
                self.decode(&mut batch)?;
            }
            (batch, tokens_list.len())
        };

        let mut response_bytes: Vec<u8> = Vec::new();

        for _ in 0..max_tokens {
            let token = match self.sample(batch.n_tokens() - 1)? {
                Some(t) => t,
                None => break,
            };

            match self.token_to_bytes(token, 64, false, None) {
                Ok(bytes) => {
                    response_bytes.extend(&bytes);
                }
                Err(e) => {
                    error!("Failed to extract token bytes: {:?}", e);
                    break;
                }
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            self.decode(&mut batch)?;
            n_cur += 1;
        }

        Ok(String::from_utf8_lossy(&response_bytes).into_owned())
    }
}
