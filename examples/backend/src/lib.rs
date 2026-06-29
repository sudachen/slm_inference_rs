use anyhow::Result;
use clap::ValueEnum;
use slm_inference::{slm, slm::ContextBuilder, slm::Model};
use strum::Display;

#[allow(unused)]
pub fn setup_backend(
    config: impl slm::ModelConfig + 'static,
    model_info: slm::HfModel,
) -> Result<slm::Assistant> {
    let model = model_info.load(config)?;
    let mut builder = model.context().with_n_ctx(20000).with_sampler(0.3, 20, 0.9);
    if model_info.formatter != "phi4" {
        builder = builder.with_gen_type_kv(slm::KvType::Q6, slm::KvType::Q6)
    }
    let context = builder.build()?;
    Ok(slm::Assistant::new(
        context,
        slm::DynamicFormatter::try_from(model_info.formatter)?,
    )?)
}

pub fn select_model(model: ModelId) -> slm::HfModel {
    match model {
        ModelId::Gemma4eb => slm::HfModel {
            repo: "unsloth/gemma-4-E4B-it-GGUF",
            filename: "gemma-4-E4B-it-IQ4_XS.gguf",
            formatter: "gemma4",
        },
        ModelId::Gemma12b => slm::HfModel {
            repo: "unsloth/gemma-4-12B-it-qat-GGUF",
            filename: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
            formatter: "gemma4",
        },
        ModelId::Phi4 => slm::HfModel {
            repo: "bartowski/microsoft_Phi-4-mini-reasoning-GGUF",
            filename: "microsoft_Phi-4-mini-reasoning-IQ4_XS.gguf",
            formatter: "phi4",
        },
        ModelId::Qwen25 => slm::HfModel {
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
            filename: "Qwen2.5-7B-Instruct-IQ4_XS.gguf",
            formatter: "qwen25",
        },
    }
}

pub fn selector(model: ModelId, backend: BackendId, cpu: bool) -> Result<slm::Assistant> {
    #[allow(unused)]
    let gpu_layers = if cpu { 0 } else { 199 };
    #[allow(unused)]
    let model_info = select_model(model);
    #[allow(unreachable_code)]
    match backend {
        #[cfg(feature = "llama")]
        BackendId::Llama => setup_backend(
            slm_llama::ModelConfig::default().with_n_gpu_layers(gpu_layers),
            model_info,
        ),
        #[cfg(feature = "ikllama")]
        BackendId::Ikllama => setup_backend(
            slm_ikllama::ModelConfig::default().with_n_gpu_layers(gpu_layers),
            model_info,
        ),
        #[allow(unused)]
        _ => Err(anyhow::anyhow!("Unsupported backend")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, ValueEnum, Display)]
#[strum(serialize_all = "lowercase")]
pub enum BackendId {
    #[cfg(feature = "llama")]
    Llama,
    #[cfg(feature = "ikllama")]
    Ikllama,
}

impl Default for BackendId {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "ikllama")]
        return BackendId::Ikllama;
        #[cfg(feature = "llama")]
        return BackendId::Llama;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, ValueEnum, Display)]
#[strum(serialize_all = "lowercase")]
pub enum ModelId {
    Gemma4eb,
    #[default]
    Gemma12b,
    Phi4,
    Qwen25,
}
