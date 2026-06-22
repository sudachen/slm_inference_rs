mod gemma4;
mod llama3;
mod mistal;
mod phi4;
mod qwen25;

pub use gemma4::{Gemma4Flavor, GemmaFormatter};
pub use llama3::Llama3Formatter;
pub use mistal::{MistralFlavor, MistralFormatter};
pub use phi4::Phi4Formatter;
pub use qwen25::Qwen25Formatter;

use crate::errors::ModelFormatterError;
use crate::formatter::SlmToolStyle;
use crate::{SlmFormatter, SlmRole};

/// Runtime-selectable chat-template formatter that dispatches to one of the
/// built-in model-family formatters.
///
/// Construct via `TryFrom<&str>` using the formatter name keys below:
///
/// | Key | Formatter | Notes |
/// |-----|-----------|-------|
/// | `"llama3"` | [`Llama3Formatter`] | Meta Llama 3 header-token style |
/// | `"gemma4"` | [`GemmaFormatter`] (Vanilla) | Google Gemma 4 turn/channel style |
/// | `"gemma4-google"` | [`GemmaFormatter`] (GoogleOfficial) | Official Google template |
/// | `"gemma4-unsloth"` | [`GemmaFormatter`] (UnslothFixed) | Unsloth-patched template |
/// | `"mistral"` | [`MistralFormatter`] (V3Tekken) | Mistral v3 / Tekken tokenizer |
/// | `"mistral-legacy"` | [`MistralFormatter`] (Legacy) | Mistral v1/v2 legacy template |
/// | `"qwen25"` | [`Qwen25Formatter`] | Qwen 2.5 ChatML style |
/// | `"phi4"` | [`Phi4Formatter`] | Microsoft Phi-4 style |
pub enum SlmDynamicFormatter {
    Gemma(GemmaFormatter),
    Llama3(Llama3Formatter),
    Mistral(MistralFormatter),
    Qwen25(Qwen25Formatter),
    Phi4(Phi4Formatter),
}

impl TryFrom<&str> for SlmDynamicFormatter {
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

impl SlmDynamicFormatter {
    fn deref(&self) -> &dyn SlmFormatter {
        match self {
            Self::Llama3(f) => f,
            Self::Gemma(f) => f,
            Self::Mistral(f) => f,
            Self::Qwen25(f) => f,
            Self::Phi4(f) => f,
        }
    }
}

impl SlmFormatter for SlmDynamicFormatter {
    fn bos(&self) -> Option<&str> {
        self.deref().bos()
    }

    fn turn_start(&self, role: &SlmRole) -> String {
        self.deref().turn_start(role)
    }

    fn turn_end(&self, role: &SlmRole) -> String {
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

    fn tool_style(&self) -> SlmToolStyle {
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
