use crate::SlmRole;
use crate::formatter::{SlmFormatter, SlmToolStyle};

pub struct Llama3Formatter;

impl SlmFormatter for Llama3Formatter {
    fn bos(&self) -> Option<&str> {
        Some("<|begin_of_text|>")
    }
    fn tool_style(&self) -> SlmToolStyle {
        SlmToolStyle::SeparateTurn
    }

    fn turn_start(&self, role: &SlmRole) -> String {
        match role {
            SlmRole::System => "<|start_header_id|>system<|end_header_id|>\n\n".to_string(),
            SlmRole::User => "<|start_header_id|>user<|end_header_id|>\n\n".to_string(),
            SlmRole::Assistant => "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string(),
            // "<|start_header_id|>ipython<|end_header_id|>\n\n".to_string(),
        }
    }

    fn turn_end(&self, _role: &SlmRole) -> String {
        "<|eot_id|>".to_string()
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        Some(("<think>\n", "\n</think>\n"))
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        format!("<think>\n{}\n</think>\n", content.trim())
    }

    fn reasoning_trigger(&self) -> Option<&str> {
        Some("\n<think>\n")
    }

    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String {
        format!(
            "{{\"name\": \"{}\", \"parameters\": {}}}",
            name,
            arguments_json.trim()
        )
    }

    fn format_tool_response(&self, _name: &str, content: &str) -> String {
        format!("{}", content.trim())
    }

    fn strip_tags(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        let full_headers = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "<|eot_id|>",
            "<|eom_id|>",
        ];

        for tag in full_headers {
            cleaned = cleaned.replace(tag, "");
        }

        cleaned
    }
}
