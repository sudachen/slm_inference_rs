use clap::Parser;
use slm_inference::SlmBrake;
use crate::backend::{BackendId, ModelId, selector};


#[derive(Parser, Debug)]
pub struct SayHiArgs {
    #[arg(short, long, default_value_t = BackendId::default())]
    pub backend: BackendId,
    #[arg(short, long, default_value_t = ModelId::default())]
    pub model: ModelId,
}

impl SayHiArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        let mut oracle = selector(self.model, self.backend, true)?;
        oracle.system("You are a precise QA tool. Answer the user's question with exactly one word: \"Hi\"")?;
        let answer = oracle.ask("Say Hi", Some(SlmBrake::token_limit(30)))?;
        println!("\n{}", answer);
        Ok(())
    }
}
