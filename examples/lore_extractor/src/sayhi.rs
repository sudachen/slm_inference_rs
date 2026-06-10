use clap::Parser;
use crate::backend::{BackendId,backend_selector};


#[derive(Parser, Debug)]
pub struct SayHiArgs {
    #[arg(short, long, default_value_t = BackendId::default())]
    pub backend: BackendId,
}

impl SayHiArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        let mut chat = backend_selector(self.backend)?;
        chat.system("You are a precise QA tool. Answer the user's question with exactly one word: \"Hi\"")?;
        let answer = chat.user_ask("Say Hi", None)?;
        println!("\n{}", answer);
        Ok(())
    }
}
