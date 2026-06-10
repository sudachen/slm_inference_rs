#![allow(unused)]

use std::path::PathBuf;
use clap::{Parser,ValueEnum};
use anyhow::Result;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use serde::{Deserialize, Serialize};
use slm_inference::*;
use epubscan::EpubScan;
use itertools::Itertools;
use tracing::{debug, error};
use crate::backend::{BackendId,backend_selector};

#[derive(Parser, Debug)]
pub struct YesNoArgs {
    #[arg(short, long, default_value_t = BackendId::default())]
    pub backend: BackendId,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(num_args(1))]
    pub questions: Vec<PathBuf>,
    #[arg(num_args(1..))]
    pub input: Vec<PathBuf>,
}

impl YesNoArgs {
    pub fn run(&self) -> Result<()> {
        println!("Answer Yes/No for questions over {} file(s):",
                 self.input.len());
        let mut outfile = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.output.clone().unwrap_or("entities.json".into()))?;
        let mut chat = backend_selector(self.backend)?;
        chat.system("You are a precise tool that answers only \"Yes\" or \"No\" without any other symbols based on the text:")?;
        for (index, file) in self.input.iter().enumerate() {
            println!("  File {}: {:?}", index + 1, file);
            let epub = EpubScan::from_file(file)?;
            for section in epub.sections().iter().take(2) {
                println!("    Section: {}", section.title().unwrap_or(""));
                chat.user(&section.text());
            }
        }
        let questions: Vec<YesNoQuest> = self.questions.iter()
            .map(|p| -> Result<Vec<YesNoQuest>> {
                let f = std::fs::File::open(p)?;
                BufReader::new(f).lines()
                    .filter_map(|l| l.ok())
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| Ok(serde_json::from_str(&l)?))
                    .collect()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        chat.save()?;
        for (no, q) in questions.into_iter().enumerate() {
            let question = q.question;
            let no = no + 1;
            println!("Question {no}: {question}");
            let answer = chat.user_ask(&question, None)?;
            println!("E {question} -> {answer} ?= {}", q.answer);
            if answer.trim().to_lowercase() != q.answer.trim().to_lowercase() {
                error!("failed to answer question {no} : {question}");
            }
            chat.rollback()?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct YesNoQuest {
    question: String,
    answer: String,
}
