use super::{BoxedConstraint, SamplingError, StringToTokenError, TokenToStringError, Vocab};
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use serde_json::Value;
use std::any::TypeId;
use std::sync::{Mutex,OnceLock};
use llguidance::api::TopLevelGrammar;
use super::llg_lark::{ParserRegistry, LarkConstraint, json_schema_to_lark, variants_to_lark};

pub trait VobTokenizer {
    fn token_to_bytes(&self, token: i32, special: bool) -> Result<Vec<u8>, TokenToStringError>;
    fn str_to_tokens(&self, str: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>, StringToTokenError>;
    fn tok_env(&self) -> &TokEnv;
}

pub struct SimpleVocab<T: VobTokenizer + Send + Sync> {
    tokenizer: T,
    registry: OnceLock<Mutex<ParserRegistry>>
}

impl<T: VobTokenizer + Send + Sync> SimpleVocab<T> {
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            registry: OnceLock::new()
        }
    }

    pub fn registry(&self) -> &Mutex<ParserRegistry> {
        self.registry.get_or_init(|| {
            let tok_env = self.tokenizer.tok_env();
            Mutex::new(ParserRegistry::new(tok_env))
        })
    }
}

impl<T: VobTokenizer + Send + Sync> Vocab for SimpleVocab<T> {
    fn token_to_bytes(
        &self,
        token: i32,
        special: bool,
    ) -> Result<Vec<u8>, TokenToStringError> {
        self.tokenizer.token_to_bytes(token, special)
    }

    fn str_to_tokens(
        &self,
        str: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, StringToTokenError> {
        self.tokenizer.str_to_tokens(str, add_special, parse_special)
    }

    fn json_constraint(
        &self,
        type_id: TypeId,
        json_schema: &dyn Fn() -> Result<(Value,Option<(String,String)>), SamplingError>,
    ) -> Result<BoxedConstraint, SamplingError> {
        let mut registry = self.registry().lock().unwrap();
        if let Some(parser) = registry.parser(type_id, None)? {
            return Ok(Box::new(LarkConstraint::new(parser)));
        }
        let (schema, bounds) = json_schema()?;
        let lark = json_schema_to_lark(schema, bounds)
            .map_err(|s| SamplingError::Error(format!("Invalid constraint/json scheme:{s}")))?;
        let grammar = TopLevelGrammar::from_lark(lark);
        let parser = registry
            .parser(type_id, Some(grammar))?
            .ok_or(SamplingError::Error("parser not found".to_string()))?;
        Ok(Box::new(LarkConstraint::new(parser)))
    }

    fn enum_constraint(
        &self,
        type_id: TypeId,
        variants: &dyn Fn() -> Result<(Vec<String>,Option<(String,String)>), SamplingError>,
    ) -> Result<BoxedConstraint, SamplingError> {
        let mut registry = self.registry().lock().unwrap();
        if let Some(parser) = registry.parser(type_id, None)? {
            return Ok(Box::new(LarkConstraint::new(parser)));
        }
        let (schema, bounds) = variants()?;
        let lark = variants_to_lark(schema, bounds)
            .map_err(|s| SamplingError::Error(format!("Invalid constraint/enum variants:{s}")))?;
        let grammar = TopLevelGrammar::from_lark(lark);
        let parser = registry
            .parser(type_id, Some(grammar))?
            .ok_or(SamplingError::Error("parser not found".to_string()))?;
        Ok(Box::new(LarkConstraint::new(parser)))
    }
}

pub struct SimpleTokEnv {
    pub trie: TokTrie,
    pub vocab_size: usize,
}

impl SimpleTokEnv {
    /// Build a `SimpleTokEnv` from a vocabulary described as a flat list of byte
    /// sequences, one per token ID.  `tok_eos` is the end-of-sequence token ID.
    pub fn new(tok_eos: u32, words: &[Vec<u8>]) -> Self {
        let vocab_size = words.len();
        let rx_info = TokRxInfo {
            vocab_size: vocab_size as u32,
            tok_eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        Self {
            trie: TokTrie::from(&rx_info, &words),
            vocab_size,
        }
    }
}

impl TokenizerEnv for SimpleTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}
