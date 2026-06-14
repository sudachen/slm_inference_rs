use clap::ValueEnum;
use slm_inference::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, ValueEnum)]
pub enum BackendId {
    #[default]
    Llama,
    Ikllama,
    Unknown,
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value().unwrap().get_name().fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, ValueEnum)]
pub enum ModelId {
    #[default]
    Gemma4eb,
    Gemma12b,
    Phi4,
    Qwen25,
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value().unwrap().get_name().fmt(f)
    }
}

/*
const MODEL: SlmHfModel = SlmHfModel {
    repo: "unsloth/gemma-4-E4B-it-GGUF",
    filename: "gemma-4-E4B-it-IQ4_XS.gguf",
    formatter: "gemma4",
    //repo: "unsloth/gemma-4-12B-it-qat-GGUF",
    //filename: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
    //formatter: "gemma4",
    //repo: "google/gemma-4-12B-it-qat-q4_0-gguf",
    //filename: "gemma-4-12b-it-qat-q4_0.gguf",
    //repo: "bartowski/gemma-4-12B-it-GGUF",
    //filename: "gemma-4-12B-it-IQ4_XS.gguf",
    //filename: "gemma-4-12B-it-Q6_K.gguf",
    //repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    //filename: "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
    //formatter: "llama3",
    //repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    //filename: "microsoft_Phi-4-mini-instruct-Q8_0.gguf",
    //formatter: "phi4",
    //repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
    //filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    //formatter: "qwen25",
    //repo: "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    //filename: "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
    //formatter: "mistral",
};
*/

pub fn setup_backend(config: impl SlmModelConfig + 'static, model_info: SlmHfModel) -> anyhow::Result<Box<dyn SlmOracle>> {
    let model = model_info.load(config)?;
    let context =
        model
            .context()
            .with_n_ctx(20000)
            .with_gen_type_kv(SlmKvType::Q6,SlmKvType::Q6)
            .build()?;
    let oracle = SlmSimpleOracle::new(context, SlmDynamicFormatter::try_from(model_info.formatter)?)?;
    Ok(Box::new(oracle))
}

pub fn select_model(model: ModelId) -> SlmHfModel {
    match model {
        ModelId::Gemma4eb => {
            SlmHfModel {
                repo: "unsloth/gemma-4-E4B-it-GGUF",
                filename: "gemma-4-E4B-it-IQ4_XS.gguf",
                formatter: "gemma4"
            }
        }
        ModelId::Gemma12b => {
            SlmHfModel {
                repo: "unsloth/gemma-4-12B-it-qat-GGUF",
                filename: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
                formatter: "gemma4"
            }
        }
        ModelId::Phi4 => {
            SlmHfModel {
                repo: "unsloth/Phi-4-mini-reasoning-GGUF",
                filename: "Phi-4-mini-reasoning-Q4_K_M.gguf",
                formatter: "phi4"
            }
        }
        ModelId::Qwen25 => {
            SlmHfModel {
                repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
                filename: "Qwen2.5-7B-Instruct-IQ4_XS.gguf",
                formatter: "qwen25"
            }
        }
    }
}

pub fn selector(model: ModelId, backend: BackendId,cpu: bool) -> anyhow::Result<Box<dyn SlmOracle>> {
    let gpu_layers = if cpu { 0 } else { 199 };
    let model_info = select_model(model);
    match backend {
        BackendId::Llama =>
            setup_backend(slm_llama::ModelConfig::default().with_n_gpu_layers(gpu_layers),model_info),
        BackendId::Ikllama =>
            setup_backend(slm_ikllama::ModelConfig::default().with_n_gpu_layers(gpu_layers),model_info),
        _ => Err(anyhow::anyhow!("Unsupported model"))
    }
}

