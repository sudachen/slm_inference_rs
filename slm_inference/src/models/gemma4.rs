use crate::formatter::{SlmFormatter, SlmToolStyle};
use crate::SlmRole;

pub struct GemmaFormatter;

impl SlmFormatter for GemmaFormatter {
    fn bos(&self) -> Option<&str> { Some("<|begin_of_text|>") }
    fn turn_start(&self, role: &SlmRole) -> String {
        match role {
            SlmRole::System => "<|turn>system\n".to_string(),
            SlmRole::User => "<|turn>user\n".to_string(),
            SlmRole::Assistant => "<|turn>model\n".to_string(),
            SlmRole::Tool(_) => String::new(), // У Gemma инструменты внутри model turn
        }
    }

    fn turn_end(&self, role: &SlmRole) -> String {
        match role {
            SlmRole::Tool(_) => String::new(),
            _ => "<turn|>\n".to_string(),
        }
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        Some(("<|channel>thought\n", "\n<channel|>"))
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        format!("<|channel>thought\n{}\n<channel|>", content.trim())
    }

    fn tool_style(&self) -> SlmToolStyle { SlmToolStyle::Inline }

    fn format_tool_call(&self, name: &str, arguments: &str) -> String {
        format!("<|tool_call>call:{}{{{}}}<tool_call|>", name, arguments.trim())
    }

    fn format_tool_response(&self, name: &str, content: &str) -> String {
        format!("<|tool_response>response:{}{{value:{}}}<tool_response|>", name, content.trim())
    }

    fn strip_tags(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        let gemma_structural_tags = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|turn>user\n",
            "<|turn>model\n",
            "<|turn>system\n",
            "<turn|>\n",
            "<turn|>",
        ];

        for tag in gemma_structural_tags {
            cleaned = cleaned.replace(tag, "");
        }

        let gemma_channels = [
            "<|channel>thought\n",
            "<|channel>code\n",
            "<|channel>custom\n",
            "<channel|>\n",
            "<channel|>",
        ];

        for tag in gemma_channels {
            cleaned = cleaned.replace(tag, "");
        }

        while let Some(start_idx) = cleaned.find("<|tool_call>") {
            if let Some(end_idx) = cleaned[start_idx..].find("<tool_call|>") {
                let absolute_end_idx = start_idx + end_idx + "<tool_call|>".len();
                cleaned.drain(start_idx..absolute_end_idx);
            } else {
                cleaned.drain(start_idx..);
                break;
            }
        }

        cleaned
    }
}
