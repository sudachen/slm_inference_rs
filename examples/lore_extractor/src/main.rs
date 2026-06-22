use clap::{Parser, Subcommand};
use backend::{selector, BackendId, ModelId};

//mod yesno;
mod sayhi;
mod yesno;
mod ents;

fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // Явно указываем stderr (хотя это дефолт)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let mut oracle = selector(cli.model, cli.backend, cli.cpu)?;

    match cli.command {
        Commands::YesNo(args) => args.run(oracle.as_mut()),
        Commands::SayHi(args) => args.run(oracle.as_mut()),
        Commands::Ents(args) => args.run(oracle.as_mut()),
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
    #[arg(long, global=true)]
    pub cpu: bool,

}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Extract knowledge from a text
    #[command(name = "yesno")]
    YesNo(yesno::YesNoArgs),
    #[command(name = "sayhi")]
    SayHi(sayhi::SayHiArgs),
    #[command(name = "ents")]
    Ents(ents::EntsArgs),
}
