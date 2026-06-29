use crate::models::{Gemma4Flavor, GemmaFormatter, Llama3Formatter, MistralFlavor, MistralFormatter, Phi4Formatter, Qwen25Formatter};
use super::{Role, ModelFormatterError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStyle {
    /// Tool calls and responses are embedded as special markup blocks *inside* the
    /// current assistant or user turn (e.g. Gemma 4, Mistral, Qwen 2.5).
    Inline,
    /// Tool calls and responses occupy their own dedicated conversation turns
    /// (e.g. Llama 3 `ipython` role).
    SeparateTurn,
}

/// Chat-template renderer for a specific model family.
///
/// Each model family wraps messages in its own delimiter scheme (ChatML, Llama 3
/// header tokens, Mistral `[INST]`, etc.).  Implementing this trait allows
/// [`SimpleOracle`](SimpleOracle) to stay model-agnostic.
///
/// See [`DynamicFormatter`](DynamicFormatter) for a runtime-selectable
/// dispatcher over the built-in formatters.
pub trait Formatter {
    /// Optional byte-order mark / BOS token prepended before the very first turn.
    /// Returns `None` for models that start directly with the first role delimiter.
    fn bos(&self) -> Option<&str>;
    /// Returns the opening delimiter for a turn with the given role.
    fn turn_start(&self, role: &Role) -> String;
    /// Returns the closing delimiter for a turn with the given role.
    fn turn_end(&self, role: &Role) -> String;

    // --- Reasoning Control ---

    /// Tags that the model itself uses to highlight thoughts during generation.
    /// Needed so that your streaming parser can separate on the fly
    /// reasoning_content from the final text (for example, for display to the user).
    fn reasoning_bounds(&self) -> Option<(&str, &str)>; // Example: Some(("<think>", "</think>"))

    /// How to wrap the text of thoughts if we forcibly push ready thoughts into the cache
    /// (for example, restoring context from a database)
    fn wrap_reasoning(&self, content: &str) -> String;

    fn reasoning_trigger(&self) -> Option<&str>;

    // --- Tool Control ---

    /// Which strategy the model uses for working with tools
    fn tool_style(&self) -> ToolStyle;

    /// Formats a tool call by the model (when the model writes: "calling the calculator")
    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String;

    /// Formats a tool response.
    /// For the Inline style this will be just an inner block,
    /// for SeparateTurn — a full body inside turn_start(Tool) and turn_end(Tool).
    fn format_tool_response(&self, tool_name: &str, content: &str) -> String;

    /// Strip all model-specific markup tags from `text`, leaving only content.
    fn strip_tags(&self, text: &str) -> String;

    /// Remove reasoning blocks and all markup tags from `text`, returning clean content.
    fn clean(&self, text: &str) -> String {
        let mut cleaned = text.to_string();
        if let Some((start, end)) = self.reasoning_bounds() {
            while let Some(start_idx) = cleaned.find(start) {
                if let Some(end_idx) = cleaned[start_idx..].find(end) {
                    let absolute_end_idx = start_idx + end_idx;
                    cleaned.drain(start_idx..absolute_end_idx);
                } else {
                    cleaned.drain(start_idx..);
                    break;
                }
            }
        }
        self.strip_tags(&cleaned).trim().to_string()
    }

    /// Split `text` into `(clean_content, Option<thinking>)` by extracting any
    /// reasoning block demarcated by [`reasoning_bounds`](Self::reasoning_bounds).
    fn strip_thought(&self, text: &str) -> (String, Option<String>) {
        let mut cleaned = text.to_string();
        let mut thinking = String::new();
        let mut idx = 0;
        if let Some((start, end)) = self.reasoning_bounds() {
            while let Some(start_idx) = cleaned[idx..].find(start) {
                idx += start_idx;
                cleaned.drain(idx..idx + start.len());
                if let Some(end_idx) = cleaned[idx..].find(end) {
                    thinking.extend(cleaned.drain(idx..idx + end_idx));
                    cleaned.drain(idx..idx + end.len());
                } else {
                    thinking.extend(cleaned.drain(idx..));
                    break;
                }
            }
        }
        let thinking = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };
        (self.strip_tags(&cleaned).trim().to_string(), thinking)
    }
}

pub enum DynamicFormatter {
    Gemma(GemmaFormatter),
    Llama3(Llama3Formatter),
    Mistral(MistralFormatter),
    Qwen25(Qwen25Formatter),
    Phi4(Phi4Formatter),
}

impl TryFrom<&str> for DynamicFormatter {
    type Error = ModelFormatterError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "llama3" => Ok(Self::Llama3(Llama3Formatter)),
            "gemma4" => Ok(Self::Gemma(GemmaFormatter::new(
                Gemma4Flavor::Vanilla,
                true,
            ))),
            "gemma4-google" => Ok(Self::Gemma(GemmaFormatter::new(
                Gemma4Flavor::GoogleOfficial,
                true,
            ))),
            "gemma4-unsloth" => Ok(Self::Gemma(GemmaFormatter::new(
                Gemma4Flavor::UnslothFixed,
                true,
            ))),
            "mistral" => Ok(Self::Mistral(MistralFormatter::new(
                MistralFlavor::V3Tekken,
                true,
            ))),
            "mistral-legacy" => Ok(Self::Mistral(MistralFormatter::new(
                MistralFlavor::Legacy,
                true,
            ))),
            "qwen25" => Ok(Self::Qwen25(Qwen25Formatter::new(true))),
            "phi4" => Ok(Self::Phi4(Phi4Formatter::new(true))),
            _ => Err(ModelFormatterError::UnknownModelFormatter(
                value.to_string(),
            )),
        }
    }
}

impl DynamicFormatter {
    fn deref(&self) -> &dyn Formatter {
        match self {
            Self::Llama3(f) => f,
            Self::Gemma(f) => f,
            Self::Mistral(f) => f,
            Self::Qwen25(f) => f,
            Self::Phi4(f) => f,
        }
    }
}

impl Formatter for DynamicFormatter {
    fn bos(&self) -> Option<&str> {
        self.deref().bos()
    }

    fn turn_start(&self, role: &Role) -> String {
        self.deref().turn_start(role)
    }

    fn turn_end(&self, role: &Role) -> String {
        self.deref().turn_end(role)
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        self.deref().reasoning_bounds()
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        self.deref().wrap_reasoning(content)
    }

    fn reasoning_trigger(&self) -> Option<&str> {
        self.deref().reasoning_trigger()
    }

    fn tool_style(&self) -> ToolStyle {
        self.deref().tool_style()
    }

    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String {
        self.deref().format_tool_call(name, arguments_json)
    }

    fn format_tool_response(&self, tool_name: &str, response_content: &str) -> String {
        self.deref()
            .format_tool_response(tool_name, response_content)
    }

    fn clean(&self, text: &str) -> String {
        self.deref().clean(text)
    }

    fn strip_tags(&self, text: &str) -> String {
        self.deref().strip_tags(text)
    }
}
