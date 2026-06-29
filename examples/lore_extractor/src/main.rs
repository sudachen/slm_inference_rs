use backend::{BackendId, ModelId, selector};
use clap::{Parser, Subcommand};

//mod yesno;
mod ents;
mod sayhi;
mod yesno;

fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // Явно указываем stderr (хотя это дефолт)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let mut assistant = selector(cli.model, cli.backend, cli.cpu)?;

    match cli.command {
        Commands::YesNo(cmd) => cmd.run(&mut assistant),
        Commands::SayHi(cmd) => cmd.run(&mut assistant),
        Commands::Ents(cmd) => cmd.run(&mut assistant),
    }
}

#[derive(Parser, Debug)]
#[command(name = "lab16")]
#[command(about = "Lab16 CLI tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    #[arg(short, long, default_value_t = BackendId::default(), global=true)]
    pub backend: BackendId,
    #[arg(short, long, default_value_t = ModelId::default(), global=true)]
    pub model: ModelId,
    #[arg(long, global = true)]
    pub cpu: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Extract knowledge from a text
    #[command(name = "yesno")]
    YesNo(yesno::YesNo),
    #[command(name = "sayhi")]
    SayHi(sayhi::SayHi),
    #[command(name = "ents")]
    Ents(ents::Ents),
}
