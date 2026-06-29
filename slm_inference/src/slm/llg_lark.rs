//! Integration with the `llguidance` library for constrained generation.
//!
//! This module provides:
//!
//! - [`LarkConstraint`] - Wrapper around llguidance's [`Constraint`] for enforcing Lark grammars
//! - [`variants_to_lark`] - Converts enum variant lists to Lark grammars
//! - [`json_schema_to_lark`] - Converts JSON schemas to Lark grammars
//! - [`ParserRegistry`] - Caches compiled parsers keyed by Rust type ID

use std::any::TypeId;
use std::collections::HashMap;
use llguidance::{Constraint, ParserFactory, TokenParser};
use llguidance::api::TopLevelGrammar;
use llguidance::earley::SlicedBiasComputer;
use llguidance::toktrie::{InferenceCapabilities, TokEnv};
use serde_json::Value;
use super::{ConstraintStep, Constraint as SlmConstraint, SamplingError};

/// A [`Constraint`] that enforces a Lark grammar during token sampling.
///
/// Wraps a `llguidance` [`Constraint`] compiled from a [`TopLevelGrammar`].
/// Logit masking is performed by the underlying `Constraint::compute_mask` call.
#[derive(Clone)]
pub struct LarkConstraint {
    constraint: Constraint,
    tok_env: TokEnv,
}

impl LarkConstraint {
    /// Wrap a compiled [`TokenParser`] in a `LarkConstraint`.
    pub fn new(parser: TokenParser) -> Self {
        let tok_env = parser.token_env.clone();
        Self {
            constraint: Constraint::new(parser),
            tok_env,
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

    fn forward(&mut self, token: i32) -> Result<ConstraintStep, SamplingError> {
        let res = self
            .constraint
            .commit_token(Some(token as u32))
            .map_err(|e| SamplingError::Error(e.to_string()))?;
        if res.stop {
            return Ok(ConstraintStep::Stop);
        }
        // if res.ff_tokens.len() > 1 {
        //     Ok(SlmConstraintStep::FastForward(
        //         res.ff_tokens.iter().map(|x| *x as i32).collect(),
        //     ))
        // } else {
        //     Ok(SlmConstraintStep::Forward)
        // }
        Ok(ConstraintStep::Forward)
    }

    fn prefill(&mut self, text: &str) -> Result<(), SamplingError> {
        let tokens = self.tok_env.tokenize_special(text);
        for t in tokens.into_iter() {
            self.constraint.compute_mask().
                map_err(|e| SamplingError::Error(e.to_string()))?;
            let _s = self.constraint.commit_token(Some(t)).
                map_err(|e| SamplingError::Error(e.to_string()))?;
        }
        Ok(())
    }
}

/// Convert a list of enum variant strings into a Lark grammar for constrained generation.
///
/// The grammar enforces that the model output must match one of the provided variants.
/// If `reasoning_bounds` is provided, the grammar allows optional reasoning content.
pub fn variants_to_lark(variants: Vec<String>, reasoning_bounds: Option<(String,String)>) -> Result<String, &'static str> {
    if variants.is_empty() {
        return Err("The variants list must have at least one variant defined");
    }

    let mut lark_rules = Vec::new();

    if let Some((start, end)) = &reasoning_bounds.as_ref().map(|(a,b)| (a.trim(), b.trim())) {
        lark_rules.push("start: [thinking] WS? enum_value WS?".to_string());
        lark_rules.push(format!("thinking: \"{}\" /(?s).*?/ \"{}\"", start, end));
    } else {
        lark_rules.push("start: WS? enum_value WS?".to_string());
    }

    lark_rules.push("WS: /[ \\t\\n\\r]+/".to_string());

    let mut variant_rules = Vec::new();
    for variant in variants {
        let escaped_variant = variant.replace('"', "\\\"");
        variant_rules.push(format!("\"{}\"", escaped_variant));
    }

    let enum_body = variant_rules.join(" | ");
    lark_rules.push(format!("enum_value: {}", enum_body));

    Ok(lark_rules.join("\n"))
}


/// Convert a JSON object schema into a Lark grammar for constrained generation.
///
/// The grammar enforces that the model output must be a JSON array of objects
/// matching the provided schema. If `reasoning_bounds` is provided, the grammar
/// allows optional reasoning content before the JSON array.
pub fn json_schema_to_lark(schema: Value, reasoning_bounds: Option<(String,String)>) -> Result<String, &'static str> {
    // Validate that the provided schema represents a JSON object (struct element)
    if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
        return Err("The input schema must be of type 'object' (the array element definition)");
    }

    let mut lark_rules = Vec::new();
    if let Some((start, end)) = reasoning_bounds.as_ref().map(|(a,b)| (a.trim(), b.trim())) {
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

/// Cache of compiled `llguidance` parsers, keyed by the Rust [`TypeId`] of the
/// target schema type.
///
/// The [`ParserFactory`] is initialised lazily on the first call to
/// [`parser`](Self::parser) that requires compilation.
pub struct ParserRegistry {
    tok_env: TokEnv,
    factory: Option<ParserFactory>,
    parsers: HashMap<TypeId, TokenParser>,
    caps: InferenceCapabilities,
}

impl ParserRegistry {
    /// Create a new empty registry using `tk` as the shared token environment.
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

    /// Return the [`ParserFactory`], initialising it the first time this is called.
    pub fn factory(&mut self) -> Result<&ParserFactory, SamplingError> {
        if self.factory.is_none() {
            self.factory = Some(
                ParserFactory::new(&self.tok_env, self.caps.clone(), &SlicedBiasComputer::general_slices())
                    .map_err(|x| SamplingError::Error(format!("llguidance factory error: {x}")))?,
            );
        }
        Ok(self.factory.as_ref().unwrap())
    }

    /// Look up or compile a [`TokenParser`] for `type_id`.
    ///
    /// If `grammar` is `Some` and no parser for `type_id` is cached yet, one is
    /// compiled from the grammar and stored.  Returns `None` when `grammar` is
    /// `None` and no cached entry exists.
    pub fn parser(
        &mut self,
        type_id: TypeId,
        grammar: Option<TopLevelGrammar>,
    ) -> Result<Option<TokenParser>, SamplingError> {
        if !self.parsers.contains_key(&type_id) {
            if grammar.is_none() {
                return Ok(None);
            }
            let factory = self.factory()?;
            let parser = factory
                .create_parser(grammar.unwrap())
                .map_err(|e| SamplingError::Error(format!("parser error: {e}")))?;
            self.parsers.insert(type_id, parser);
        }
        Ok(self.parsers.get(&type_id).cloned())
    }
}