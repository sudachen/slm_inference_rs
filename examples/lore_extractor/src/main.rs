use clap::{Parser, Subcommand};

//mod yesno;
mod backend;
mod sayhi;
mod yesno;

fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // Явно указываем stderr (хотя это дефолт)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::YesNo(args) => args.run(),
        Commands::SayHi(args) => args.run(),
    }
}

#[derive(Parser, Debug)]
#[command(name = "lab16")]
#[command(about = "Lab16 CLI tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Extract knowledge from a text
    #[command(name = "yesno")]
    YesNo(yesno::YesNoArgs),
    #[command(name = "sayhi")]
    SayHi(sayhi::SayHiArgs),
}
