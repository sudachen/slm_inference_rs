use clap::Parser;
use slm_inference::SlmOracle;

#[derive(Parser, Debug)]
pub struct SayHiArgs;

impl SayHiArgs {
    pub fn run(&self, oracle: &mut dyn SlmOracle) -> anyhow::Result<()> {
        oracle.system(
            "You are a precise QA tool. Answer the user's question with exactly one word: \"Hi\"",
        )?;
        oracle.set_max_answer_tokens(30);
        let answer = oracle.ask(false,"Say Hi")?;
        println!("\n{}", answer);
        Ok(())
    }
}
