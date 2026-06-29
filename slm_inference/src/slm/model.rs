use super::{Context, ContextBuilder, GgufLoaderError};
use std::path::Path;

/// A loaded model weight set from which inference contexts can be created.
///
/// The model itself is stateless; all mutable state (KV cache, sampler, etc.)
/// lives in the [`Context`] produced by the builder returned from
/// [`context`](Self::context).
pub trait Model {
    type Context: Context + Send;
    /// Return a builder for creating a new inference context backed by this model.
    fn context(&self) -> impl ContextBuilder<Self::Context>;
}

/// Factory that describes how a specific model variant should be loaded.
///
/// Callers typically construct a concrete config (e.g. with builder methods),
/// then call [`load_gguf`](Self::load_gguf) to obtain a usable [`Model`].
pub trait ModelConfig {
    //type Context: Context;
    //type Model: Model<Context=Self::Context>;
    /// Load model weights from the given GGUF file and return a model handle.
    fn load_gguf(self, path: impl AsRef<Path>) -> Result<impl Model, GgufLoaderError>;
}
