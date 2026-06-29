use clap::Parser;
use slm_inference::slm;

#[derive(Parser, Debug)]
pub struct SayHiArgs;

impl SayHiArgs {
    pub fn run(&self, oracle: &mut slm::Oracle) -> anyhow::Result<()> {
        oracle.system(
            "You are a precise QA tool. Answer the user's question with exactly one word: \"Hi\"",
        )?;
        oracle.set_max_answer_tokens(30);
        let answer = oracle.ask(false,"Say Hi", None)?;
        println!("\n{}", answer);
        Ok(())
    }
}
