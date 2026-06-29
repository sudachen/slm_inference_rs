use std::path::PathBuf;
use hf_hub::api::sync::ApiBuilder;
use tracing::debug;

use anyhow::Context as _;
use super::{ModelConfig,Model};

#[derive(Copy,Clone)]
pub struct HfModel {
    /// HuggingFace repository identifier (e.g. `"bartowski/Llama-3.2-3B-Instruct-GGUF"`).
    pub repo: &'static str,
    /// Filename within the repository (e.g. `"Llama-3.2-3B-Instruct-Q8_0.gguf"`).
    pub filename: &'static str,
    /// Formatter name string accepted by [`SlmDynamicFormatter`](crate::SlmDynamicFormatter)
    /// (e.g. `"llama3"`, `"qwen25"`).
    pub formatter: &'static str,
}

impl HfModel {
    /// Resolve the model's local cache path, downloading from HuggingFace if absent.
    pub fn get_or_download(&self) -> anyhow::Result<PathBuf> {
        get_or_download_model(self.repo, self.filename)
    }

    #[allow(dead_code)]
    pub fn load<Config>(self, cfg: Config) -> anyhow::Result<impl Model>
    where
        Config: ModelConfig,
    {
        let path = self.get_or_download()?;
        cfg.load_gguf(path).context("Failed to load model")
    }
}

/// Resolve `filename` in `repo` to a local path, downloading from HuggingFace Hub
/// into the default cache directory (`~/.cache/huggingface/hub`) if not already present.
pub fn get_or_download_model(repo: &str, filename: &str) -> anyhow::Result<PathBuf> {
    // This will use default HF cache directory (~/.cache/huggingface/hub)
    let api = ApiBuilder::new()
        .with_progress(true) // Enable progress bars automatically
        .build()?;

    let api_repo = api.model(repo.to_string());

    debug!("Checking for model: {} in repo: {}...", filename, repo);

    // If it's already in the cache, it returns the local path instantly.
    // If not, it starts the download.
    let model_path = api_repo.get(filename)?;

    debug!("Model ready at: {:?}", model_path);
    Ok(model_path)
}
