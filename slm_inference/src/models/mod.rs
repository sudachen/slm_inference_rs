mod gemma4;
mod llama3;

pub use gemma4::GemmaFormatter;
pub use llama3::Llama3Formatter;
use crate::formatter::SlmToolStyle;
use crate::{SlmFormatter, SlmRole};
use crate::errors::ModelFormatterError;

pub enum SlmDynamicFormatter {
    Gemma(GemmaFormatter),
    Llama3(Llama3Formatter),
}

impl TryFrom<&str> for SlmDynamicFormatter {
    type Error = ModelFormatterError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "llama3" => Ok(Self::Llama3(Llama3Formatter)),
            "gemma4" => Ok(Self::Gemma(GemmaFormatter)),
            _ => Err(ModelFormatterError::UnknownModelFormatter(value.to_string())),
        }
    }
}

impl SlmDynamicFormatter {
    fn deref(&self) -> &dyn SlmFormatter {
        match self {
            Self::Llama3(f) => f,
            Self::Gemma(f) => f
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

    fn tool_style(&self) -> SlmToolStyle {
        self.deref().tool_style()
    }

    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String {
        self.deref().format_tool_call(name, arguments_json)
    }

    fn format_tool_response(&self, tool_name: &str, response_content: &str) -> String {
        self.deref().format_tool_response(tool_name, response_content)
    }

    fn clean(&self, text: &str) -> String {
        self.deref().clean(text)
    }

    fn strip_tags(&self, text: &str) -> String {
        self.deref().strip_tags(text)
    }
}