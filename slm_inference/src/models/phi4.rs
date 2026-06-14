use crate::formatter::{SlmFormatter, SlmToolStyle};
use crate::SlmRole;

pub struct Phi4Formatter {
    thinking: bool,
}

impl Phi4Formatter {
    pub fn new(thinking: bool) -> Self {
        Self { thinking }
    }
}

impl SlmFormatter for Phi4Formatter {
    // The official Phi-4 template has no explicit BOS token (<s>).
    // Generation and context start directly with the special token of the first role.
    fn bos(&self) -> Option<&str> {
        None
    }

    fn turn_start(&self, role: &SlmRole) -> String {
        // As in ChatML, a \n newline is strictly required after the role token!
        match role {
            SlmRole::System => "<|system|>\n".to_string(),
            SlmRole::User => "<|user|>\n".to_string(),
            SlmRole::Assistant => "<|assistant|>\n".to_string(),
            SlmRole::Tool(_) => String::new(), // Tools are embedded inside the assistant turn
        }
    }

    fn turn_end(&self, role: &SlmRole) -> String {
        match role {
            SlmRole::Tool(_) => String::new(),
            // Microsoft's philosophy: every turn, regardless of role, is closed with a single <|end|> token
            _ => "<|end|>\n".to_string(),
        }
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        if self.thinking {
            // Base Phi-4-Instruct has no native hidden reasoning channel,
            // but the DeepSeek-R1-Distill-Phi-4-8B distillation is extremely popular.
            // For it we force the standard XML reasoning tags.
            Some(("<think>\n", "\n</think>"))
        } else {
            None
        }
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        if self.thinking {
            format!("<think>\n{}\n</think>", content.trim())
        } else {
            content.to_string()
        }
    }

    fn reasoning_trigger(&self) -> Option<&str> {
        if self.thinking {
            Some("<think>\n")
        } else {
            None
        }
    }

    fn tool_style(&self) -> SlmToolStyle {
        SlmToolStyle::Inline
    }

    fn format_tool_call(&self, name: &str, arguments: &str) -> String {
        // Advanced Phi-4 function-calling fine-tunes expect plain JSON
        format!(r#"{{"name": "{}", "arguments": {}}}"#, name, arguments.trim())
    }

    fn format_tool_response(&self, _name: &str, content: &str) -> String {
        // Microsoft wraps tool responses in the plain-text <|tool_response|> tag
        format!("<|tool_response|>\n{}\n<|end|>\n", content.trim())
    }

    fn strip_tags(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        let phi4_structural_tags = [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<|tool_response|>",
        ];

        for tag in phi4_structural_tags {
            cleaned = cleaned.replace(tag, "");
        }

        let phi4_channels = [
            "<think>",
            "</think>",
        ];

        for tag in phi4_channels {
            cleaned = cleaned.replace(tag, "");
        }

        cleaned.trim().to_string()
    }
}