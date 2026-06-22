use std::any::TypeId;
use std::collections::HashMap;
use llguidance::{Constraint, ParserFactory, TokenParser};
use llguidance::api::TopLevelGrammar;
use llguidance::earley::SlicedBiasComputer;
use llguidance::toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use serde_json::Value;
use crate::errors::{InferenceError, SamplingError};
use crate::{SlmConstraint, SlmConstraintStep};

pub struct SlmSimpleTokEnv {
    trie: TokTrie,
}

impl SlmSimpleTokEnv {
    pub fn new(tok_eos: u32, words: &[Vec<u8>]) -> Self {
        let rx_info = TokRxInfo {
            vocab_size: words.len() as u32,
            tok_eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        Self {
            trie: TokTrie::from(&rx_info, &words),
        }
    }
}

impl TokenizerEnv for SlmSimpleTokEnv {
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

pub struct LarkConstraint {
    constraint: Constraint,
    tokenv: TokEnv,
}

impl LarkConstraint {
    pub fn new(parser: TokenParser) -> Self {
        let tokenv = parser.token_env.clone();
        Self {
            constraint: Constraint::new(parser),
            tokenv,
        }
    }
}

impl SlmConstraint for LarkConstraint {
    fn mask(&mut self, logits: &mut [f32]) -> Result<bool, SamplingError> {
        let step = self
            .constraint
            .compute_mask()
            .map_err(|e| SamplingError::Error(e.to_string()))?;
        if step.is_stop() {
            return Ok(false);
        }
        if let Some(mask) = step.sample_mask.as_ref() {
            mask.iter_unset_entries(|i| {
                logits[i] = f32::NEG_INFINITY;
            });
        }
        Ok(true)
    }

    fn forward(&mut self, token: i32) -> Result<SlmConstraintStep, SamplingError> {
        let res = self
            .constraint
            .commit_token(Some(token as u32))
            .map_err(|e| SamplingError::Error(e.to_string()))?;
        if res.stop {
            return Ok(SlmConstraintStep::Stop);
        }
        // if res.ff_tokens.len() > 1 {
        //     Ok(SlmConstraintStep::FastForward(
        //         res.ff_tokens.iter().map(|x| *x as i32).collect(),
        //     ))
        // } else {
        //     Ok(SlmConstraintStep::Forward)
        // }
        Ok(SlmConstraintStep::Forward)
    }

    fn prefill(&mut self, text: &str) -> Result<(), SamplingError> {
        let tokens = self.tokenv.tokenize_special(text);
        for t in tokens.into_iter() {
            self.constraint.compute_mask().
                map_err(|e| SamplingError::Error(e.to_string()))?;
            let _s = self.constraint.commit_token(Some(t)).
                map_err(|e| SamplingError::Error(e.to_string()))?;
        }
        Ok(())
    }
}


/// Converts a single JSON object schema into a Lark grammar
/// designed to parse a JSON array of these objects.
pub fn json_schema_to_lark(schema: Value, reasoning_bounds: Option<(&str,&str)>) -> Result<String, &'static str> {
    // Validate that the provided schema represents a JSON object (struct element)
    if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
        return Err("The input schema must be of type 'object' (the array element definition)");
    }

    let mut lark_rules = Vec::new();
    if let Some((start, end)) = reasoning_bounds {
        lark_rules.push("start: [thinking] json_array".to_string());
        lark_rules.push(format!("thinking: \"{}\" /(?s).*?/ \"{}\"", start, end));
    } else {
        lark_rules.push("start: free_text json_array".to_string());
        lark_rules.push("free_text: /([^{\\[]*)/".to_string());
    }

    lark_rules.push("WS: /[ \\t\\n\\r]+/".to_string());

    // Define the root array rule that wraps the cards
    lark_rules.push("json_array: \"[\" WS? card (WS? \",\" WS? card)* WS? \"]\"".to_string());

    // Extract the properties of the object
    let properties = schema.get("properties")
        .and_then(|p| p.as_object())
        .ok_or("Missing or invalid 'properties' block in the schema")?;

    let mut field_rules = Vec::new();

    // Iterate through fields and generate Lark terminals based on their JSON types
    for (field_name, field_info) in properties {
        let rule_name = format!("{}_field", field_name);

        // Basic type mapping for the value regex
        let val_rule = match field_info.get("type").and_then(|t| t.as_str()) {
            Some("integer") => "/[0-9]+/",
            Some("number")  => "/-?[0-9]+(\\.[0-9]+)?/",
            Some("boolean") => "\"true\" | \"false\"",
            _               => "/\"[^\"\\\\]*\"/",
        };

        // Format: "field_name" WS? ":" WS? value_regex
        lark_rules.push(format!(
            "{}: \"\\\"{}\\\"\" WS? \":\" WS? {}",
            rule_name, field_name, val_rule
        ));

        field_rules.push(rule_name);
    }

    if field_rules.is_empty() {
        return Err("The object schema must have at least one property defined");
    }

    // Combine all fields into a single object definition separated by commas
    let object_body = field_rules.join(" \",\" WS? ");
    lark_rules.push(format!("card: \"{{\" WS? {} WS? \"}}\"", object_body));

    Ok(lark_rules.join("\n"))
}

pub struct ParserRegistry {
    tok_env: TokEnv,
    factory: Option<ParserFactory>,
    parsers: HashMap<TypeId, TokenParser>,
    caps: InferenceCapabilities,
}

impl ParserRegistry {
    pub fn new(tk: &TokEnv) -> Self {
        let canonical = tk.tokenize_is_canonical();
        Self {
            tok_env: tk.clone(),
            factory: None,
            parsers: HashMap::new(),
            caps: InferenceCapabilities {
                ff_tokens: canonical,
                conditional_ff_tokens: false,
                backtrack: false,
                fork: false,
            },
        }
    }

    pub fn factory(&mut self) -> Result<&ParserFactory, InferenceError> {
        if self.factory.is_none() {
            self.factory = Some(
                ParserFactory::new(&self.tok_env, self.caps.clone(), &SlicedBiasComputer::general_slices())
                    .map_err(|x| InferenceError::Error(format!("llguidance factory error: {x}")))?,
            );
        }
        Ok(self.factory.as_ref().unwrap())
    }

    pub fn parser(
        &mut self,
        type_id: TypeId,
        grammar: Option<TopLevelGrammar>,
    ) -> Result<Option<TokenParser>, InferenceError> {
        if !self.parsers.contains_key(&type_id) {
            if grammar.is_none() {
                return Ok(None);
            }
            let factory = self.factory()?;
            let parser = factory
                .create_parser(grammar.unwrap())
                .map_err(|e| InferenceError::Error(format!("parser error: {e}")))?;
            self.parsers.insert(type_id, parser);
        }
        Ok(self.parsers.get(&type_id).cloned())
    }
}