use clap::Parser;
use slm_inference::slm;

#[derive(Parser, Debug)]
pub struct SayHi;

impl SayHi {
    pub fn run(&self, assistant: &mut slm::Assistant) -> anyhow::Result<()> {
        assistant.system(
            "You are a precise QA tool. Answer the user's question with exactly one word: \"Hi\"",
        )?;
        assistant.set_max_answer_tokens(30);
        let answer = assistant.ask(false, "Say Hi", None)?;
        println!("\n{}", answer);
        Ok(())
    }
}
