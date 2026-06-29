use super::{
    Action, Answer, BoxedAction, BoxedConstraint, ComputationZone, InferenceError,
    NullInference, Assistant, Role, SavePoint,
};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// Semaphore for CPU-bound inference (one concurrent operation).
static CPU_ZONE: Semaphore = Semaphore::const_new(1);
/// Semaphore for GPU-bound inference (one concurrent operation).
static GPU_ZONE: Semaphore = Semaphore::const_new(1);

impl Assistant {
    /// Async version of [`Assistant::generate`] that offloads blocking inference to a thread pool.
    ///
    /// Uses semaphores to limit concurrent CPU/GPU operations to one per zone.
    pub async fn async_generate(
        &mut self,
        role: &Role,
        text: &str,
        think: bool,
        reset: bool,
        action: BoxedAction,
        mut constraint: Option<BoxedConstraint>,
    ) -> Result<Answer<String>, InferenceError> {
        let fragment = self.generate_fragment(role, text, think, &mut constraint)?;
        let inference = Arc::new(Mutex::new(NullInference::new()));
        let _ = SavePoint(self);
        self.inference.prefill(&fragment)?;

        let zone = match self.inference.zone() {
            ComputationZone::CPU => CPU_ZONE.acquire().await,
            ComputationZone::GPU => GPU_ZONE.acquire().await,
        }
        .map_err(|e| InferenceError::Error(format!("tokio task error: {e}")))?;
        // --- computation zone ---
        std::mem::swap(&mut *inference.lock().unwrap(), &mut self.inference);
        let inference_clone = inference.clone();
        let max_answer_tokens = self.max_answer_tokens;
        let handle = tokio::task::spawn_blocking(move || {
            let mut inference = inference_clone.lock().unwrap();
            inference.generate_until(
                &mut [action, Action::token_limit(max_answer_tokens)],
                constraint,
            )
        });
        let answer = handle.await;
        std::mem::swap(&mut *inference.lock().unwrap(), &mut self.inference);
        // --- end computation zone ---
        drop(zone);

        let answer =
            answer.map_err(|e| InferenceError::Error(format!("tokio task error: {e}")))??;
        self.generate_answer(answer, think, reset)
    }

    /// Async version of [`Assistant::ask`] that offloads blocking inference to a thread pool.
    pub async fn async_ask(
        &mut self,
        think: bool,
        text: &str,
        action: BoxedAction,
    ) -> Result<Answer<String>, InferenceError> {
        self.async_generate(&Role::User, text, think, true, action, None)
            .await
    }
}
