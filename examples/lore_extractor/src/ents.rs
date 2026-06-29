use std::path::PathBuf;
use clap::Parser;
use serde::Deserialize;
use epubscan::EpubScan;
use slm_inference::slm;

#[derive(Parser, Debug)]
pub struct Ents {
    #[arg(short, long)]
    pub think: bool,
    #[arg(num_args(1))]
    pub input: Vec<PathBuf>,
}

#[derive(Deserialize, Debug, schemars::JsonSchema)]
#[allow(unused)]
pub struct EntityCard {
    pub term: String,
    pub category: String,
    pub clue: String,
}


impl Ents {
    pub fn run(&self, assistant: &mut slm::Assistant) -> anyhow::Result<()> {
        let input = self.input[1].clone();
        println!("Input: {:?}", input);
        let epub = EpubScan::from_file(&input)?;
        assistant.set_max_answer_tokens(4096);
        assistant.system(SYSTEM_PROMPT)?;
        let pos = assistant.save()?;
        for section in epub.sections()[1 .. 2].iter() {
            println!("Section: {}", section.title().unwrap_or(""));
            assistant.user(&section.text())?;
            let cards: Vec<EntityCard> = assistant.ask_values(self.think, ENTITY_PROMPT, slm::Action::print_token())?;
            println!("{cards:?}");
            assistant.rollback(&pos)?;
        }
        Ok(())
    }
}

const SYSTEM_PROMPT: &str = r#"
You are an advanced, zero-knowledge ontology engineer and literary analyst. Your task is to process the provided English book excerpt and extract structured information based on the specific analytical focus requested at the end of the prompt.

CRITICAL LANGUAGE RULES:
1. Everything you extract (including terms, names, factor names, descriptions, categories, and clues) MUST be written strictly in ENGLISH.

CRITICAL FORMATTING RULES:
1. Rely ONLY on the clear facts directly mentioned in the provided text. Do not assume or extrapolate.
2. If the text does not contain data matching the requested focus, return an empty JSON array: [].
3. Output MUST be a strictly valid JSON array of objects. Do not wrap the JSON in markdown blocks like ```json ... ```. Do not add any conversational text, introductions, or postscripts. Output raw JSON code only."#;


const ENTITY_PROMPT: &str = r#"
Scan the English book text above and extract all unique characters, organizations, species, locations, and universe-specific neologisms/jargon.

Rules for this pass:
1. Extract "term" in ENGLISH. Normalize names to their canonical full form and terms to their singular/base form.
2. Write the "clue" strictly in ENGLISH. Provide a brief sentence explaining who or what this entity is based ONLY on the current chunk.

Output format:
[
  {
    "term": "Canonical English name or base term",
    "category": "Character / Neologism / Organization / Location",
    "clue": "Explanation written strictly in ENGLISH"
  }
]
"#;
