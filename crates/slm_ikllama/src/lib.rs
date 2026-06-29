mod backend;
mod batch;
mod context;
mod model;
mod vocab;

pub use batch::{Batch};
pub use context::{Builder, Context, KVType};
pub use model::{Model, ModelConfig};

use slm_inference::core::{Free,SharedPtr};

#[derive(Clone)]
pub struct LlamaContextFree;
impl Free<slm_ikllama_sys::llama_context> for LlamaContextFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut slm_ikllama_sys::llama_context) {
        unsafe { slm_ikllama_sys::llama_free(ptr) };
    }
}

pub type LlamaContextPtr = SharedPtr<slm_ikllama_sys::llama_context, LlamaContextFree>;

#[derive(Clone)]
pub struct LlamaModelFree;
impl Free<slm_ikllama_sys::llama_model> for LlamaModelFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut slm_ikllama_sys::llama_model) {
        unsafe { slm_ikllama_sys::llama_free_model(ptr) };
    }
}

pub type LlamaModelPtr = SharedPtr<slm_ikllama_sys::llama_model, LlamaModelFree>;

