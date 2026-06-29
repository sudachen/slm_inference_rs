mod backend;
mod batch;
mod context;
mod model;
mod vocab;

pub use batch::Batch;
pub use context::{Builder, Context, KVType};
pub use model::{Model, ModelConfig};
use slm_inference::core::{Free, SharedPtr};


#[derive(Clone)]
struct LlamaSamplerFree;
impl Free<llama_cpp_sys_2::llama_sampler> for LlamaSamplerFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut llama_cpp_sys_2::llama_sampler) {
        unsafe { llama_cpp_sys_2::llama_sampler_free(ptr) };
    }
}

type LlamaSamplerPtr = slm_inference::core::SharedPtr<llama_cpp_sys_2::llama_sampler, LlamaSamplerFree>;

#[derive(Clone)]
struct LlamaContextFree;
impl Free<llama_cpp_sys_2::llama_context> for LlamaContextFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut llama_cpp_sys_2::llama_context) {
        unsafe { llama_cpp_sys_2::llama_free(ptr) };
    }
}

type LlamaContextPtr = slm_inference::core::SharedPtr<llama_cpp_sys_2::llama_context, LlamaContextFree>;

#[derive(Clone)]
struct LlamaModelFree;
impl Free<llama_cpp_sys_2::llama_model> for LlamaModelFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut llama_cpp_sys_2::llama_model) {
        unsafe { llama_cpp_sys_2::llama_free_model(ptr) };
    }
}

type LlamaModelPtr = SharedPtr<llama_cpp_sys_2::llama_model, LlamaModelFree>;

