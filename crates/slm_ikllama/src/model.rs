use crate::context::Builder;
use slm_inference::core::shared_ptr::{Free, SharedPtr};
use slm_inference::errors::{FfiError, GgufLoaderError};
use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::sync::Arc;
use slm_inference::SlmModel;
use crate::vocab::Vocab;

#[derive(Clone)]
struct LlamaModelFree;
impl Free<slm_ikllama_sys::llama_model> for LlamaModelFree {
    #[inline(never)]
    unsafe fn free(ptr: *mut slm_ikllama_sys::llama_model) {
        unsafe { slm_ikllama_sys::llama_free_model(ptr) };
    }
}

type ModelPtr = SharedPtr<slm_ikllama_sys::llama_model, LlamaModelFree>;

#[derive(Clone)]
pub struct Model {
    ptr: ModelPtr,
    vocab: Arc<Vocab>,
}

impl Model {
    #[allow(dead_code)]
    pub fn get_ptr(&mut self) -> Result<*mut slm_ikllama_sys::llama_model, FfiError> {
        if self.ptr.is_null() {
            return Err(FfiError::NullPtr);
        }
        Ok(self.ptr.get_ptr())
    }
    #[allow(dead_code)]
    pub fn get_const_ptr(&self) -> Result<*const slm_ikllama_sys::llama_model, FfiError> {
        if self.ptr.is_null() {
            return Err(FfiError::NullPtr);
        }
        Ok(self.ptr.get_const_ptr())
    }
    #[allow(dead_code)]
    pub fn raw_ptr(&self) -> *mut slm_ikllama_sys::llama_model {
        self.ptr.get_ptr()
    }

    #[allow(refining_impl_trait)]
    pub(crate) fn vocab(&self) -> &Vocab {
        self.vocab.as_ref()
    }
}

impl SlmModel for Model {
    type Context = crate::context::Context;

    #[allow(refining_impl_trait)]
    fn context(&self) -> Builder {
        Builder::new(self.clone())
    }
}

pub struct ModelConfig {
    pub params: slm_ikllama_sys::llama_model_params,
}

impl Debug for ModelConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelConfig")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.params.use_mmap)
            .field("use_mlock", &self.params.use_mlock)
            .finish()
    }
}

impl Default for ModelConfig {
    #[inline(never)]
    fn default() -> Self {
        Self {
            params: unsafe { slm_ikllama_sys::llama_model_default_params() },
        }
    }
}

impl ModelConfig {
    #[allow(dead_code)]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }
    #[allow(dead_code)]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    #[allow(dead_code)]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    #[allow(dead_code)]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    #[allow(dead_code)]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    #[allow(dead_code)]
    pub fn with_split_mode(mut self, split_mode: SplitMode) -> Self {
        self.params.split_mode = split_mode as slm_ikllama_sys::llama_split_mode;
        self
    }
}

#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SplitMode {
    /// Single GPU
    None = slm_ikllama_sys::LLAMA_SPLIT_MODE_NONE as i8,
    /// Split layers and KV across GPUs
    Layer = slm_ikllama_sys::LLAMA_SPLIT_MODE_LAYER as i8,
    /// Split layers and KV across GPUs, use tensor parallelism if supported
    Row = slm_ikllama_sys::LLAMA_SPLIT_MODE_ATTN as i8,
    /// Experimental tensor parallelism across GPUs
    Tensor = slm_ikllama_sys::LLAMA_SPLIT_MODE_GRAPH as i8,
}

impl slm_inference::SlmModelConfig for ModelConfig {
    type Context = crate::context::Context;
    type Model = Model;

    fn load_gguf(self, path: impl AsRef<Path>) -> Result<Model, GgufLoaderError> {
        super::backend::init();
        let path_ref = path.as_ref();
        let path = path_ref.to_str().ok_or(GgufLoaderError::InvalidPath)?;
        let model = self.load_llama_model(path)?;
        let vcab_ptr = unsafe { slm_ikllama_sys::llama_model_get_vocab(model) };
        let vocab = Arc::new(Vocab::new(model, vcab_ptr));
        Ok(Model{
            ptr: ModelPtr::new(model),
            vocab,
        })
    }
}

impl ModelConfig {
    #[inline(never)]
    pub fn load_llama_model(
        &self,
        path: &str,
    ) -> Result<*mut slm_ikllama_sys::llama_model, GgufLoaderError> {
        let cstr = CString::new(path)
            .map_err(|_| FfiError::Error("path string allocation".to_string()))?;
        let llama_model =
            unsafe { slm_ikllama_sys::llama_model_load_from_file(cstr.as_ptr(), self.params) };
        if llama_model.is_null() {
            return Err(GgufLoaderError::BadModel);
        }
        Ok(llama_model)
    }
}
