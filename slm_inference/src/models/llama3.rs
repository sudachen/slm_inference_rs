use crate::slm::{Formatter, Role, ToolStyle};

/// [`Formatter`] for Meta Llama 3 models.
///
/// Uses Llama 3's `<|start_header_id|>` / `<|end_header_id|>` header tokens
/// and `<|eot_id|>` end-of-turn marker. Supports native reasoning with
/// `<|thinking|>` / `<|eot_id|>` tags.
pub struct Llama3Formatter;

impl Formatter for Llama3Formatter {
    fn bos(&self) -> Option<&str> {
        Some("<|begin_of_text|>")
    }
    fn tool_style(&self) -> ToolStyle {
        ToolStyle::SeparateTurn
    }

    fn turn_start(&self, role: &Role) -> String {
        match role {
            Role::System => "<|start_header_id|>system<|end_header_id|>\n\n".to_string(),
            Role::User => "<|start_header_id|>user<|end_header_id|>\n\n".to_string(),
            Role::Assistant => "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string(),
            // "<|start_header_id|>ipython<|end_header_id|>\n\n".to_string(),
        }
    }

    fn turn_end(&self, _role: &Role) -> String {
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
