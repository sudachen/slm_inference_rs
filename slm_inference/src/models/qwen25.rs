use crate::formatter::{SlmFormatter, SlmToolStyle};
use crate::SlmRole;

pub struct Qwen25Formatter {
    thinking: bool,
}

impl Qwen25Formatter {
    pub fn new(thinking: bool) -> Self {
        Self { thinking }
    }
}

impl SlmFormatter for Qwen25Formatter {
    // The official Qwen 2.5 template has no dedicated BOS token (it is null).
    // The model always starts directly with the first markup tag.
    fn bos(&self) -> Option<&str> {
        None
    }

    fn turn_start(&self, role: &SlmRole) -> String {
        // ChatML requires a strict \n newline immediately after the role name!
        match role {
            SlmRole::System => "<|im_start|>system\n".to_string(),
            SlmRole::User => "<|im_start|>user\n".to_string(),
            SlmRole::Assistant => "<|im_start|>assistant\n".to_string(),
            // Tools use Inline style, so no prefix here
            SlmRole::Tool(_) => String::new(),
        }
    }

    fn turn_end(&self, role: &SlmRole) -> String {
        match role {
            SlmRole::Tool(_) => String::new(),
            // Every ChatML container closes the same way, plus \n to keep history clean
            _ => "<|im_end|>\n".to_string(),
        }
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        if self.thinking {
            // Vanilla Qwen 2.5-Instruct does not reason on its own, but it is the base
            // for top reasoning models: QwQ-32B and DeepSeek-R1-Distill-Qwen.
            // All of them use plain-text <think> tags
            Some(("<think>\n", "</think>"))
        } else {
            None
        }
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        if self.thinking {
            format!("<think>\n{}</think>", content.trim())
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
        // Standard Qwen 2.5 expects a function call as a plain JSON object
        // inside a regular assistant response.
        format!(r#"{{"name": "{}", "arguments": {}}}"#, name, arguments.trim())
    }

    fn format_tool_response(&self, _name: &str, content: &str) -> String {
        // Qwen has an official native role for tool responses — "tool".
        // Format it as a full ChatML container inline within the response stream.
        format!("<|im_start|>tool\n{}<|im_end|>\n", content.trim())
    }

    fn strip_tags(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        let qwen_structural_tags = [
            "<|im_start|>",
            "<|im_end|>",
            "system\n",
            "user\n",
            "assistant\n",
            "tool\n",
        ];

        for tag in qwen_structural_tags {
            cleaned = cleaned.replace(tag, "");
        }

        let qwen_channels = [
            "<think>",
            "</think>",
        ];

        for tag in qwen_channels {
            cleaned = cleaned.replace(tag, "");
        }

        // Strip any leftover tool response containers
        while let Some(start_idx) = cleaned.find("<|im_start|>tool") {
            if let Some(end_idx) = cleaned[start_idx..].find("<|im_end|>") {
                let absolute_end_idx = start_idx + end_idx + "<|im_end|>\n".len();
                cleaned.drain(start_idx..absolute_end_idx);
            } else {
                cleaned.drain(start_idx..);
                break;
            }
        }

        cleaned.trim().to_string()
    }
}