use clap::ValueEnum;
use slm_inference::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, ValueEnum)]
pub enum BackendId {
    #[default]
    Llama,
    Bitnet,
    Camelid,
    Ikllama,
    Mistral,
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value().unwrap().get_name().fmt(f)
    }
}

const MODEL: SlmHfModel = SlmHfModel {
    repo: "bartowski/gemma-4-12B-it-GGUF",
    filename: "gemma-4-12B-it-Q6_K.gguf",
    formatter: "gemma4",
    //repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    //filename: "Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
    //formatter: "llama3",
    //repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    //filename: "microsoft_Phi-4-mini-instruct-Q8_0.gguf",
    //repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
    //filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    //repo: "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    //filename: "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
};

pub fn backend_selector(model: BackendId) -> anyhow::Result<Box<dyn SlmChat>> {
    match model {
        BackendId::Llama => {
            let config = slm_llama::ModelConfig::default().with_n_gpu_layers(199);
            let model = MODEL.load(config)?;
            let context =
                model
                    .context()
                    .with_n_ctx(20000)
                    .with_type_kv(slm_llama::KVType::Q8_0,slm_llama::KVType::Q8_0)
                    .build()?;
            Ok(Box::new(SlmSimpleChat::new(context, SlmDynamicFormatter::try_from(MODEL.formatter)?)?))
        }
        _ => Err(anyhow::anyhow!("Unsupported model"))
    }
}

