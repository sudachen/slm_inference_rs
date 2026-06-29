use crate::slm::{Formatter, Role, ToolStyle};

/// [`Formatter`] for Mistral models.
///
/// Supports both modern Mistral v3 (Tekken tokenizer) and legacy v1/v2 templates.
/// Uses `[INST]` / `[/INST]` delimiters and `[SYSTEM_PROMPT]` for system messages.
/// Tool calls use `[TOOL_CALLS]` / `[/TOOL_CALLS]` markup.
pub struct MistralFormatter {
    flavor: MistralFlavor,
    thinking: bool,
}

pub enum MistralFlavor {
    /// For Mistral-Nemo, Mistral Large 2, and 7B v0.3 (Tekken / v3 tokenizer)
    /// Has native tokens for system and tools.
    V3Tekken,
    /// For legacy Mistral 7B v0.1 / v0.2 models (v1/v2 template)
    Legacy,
}

impl MistralFormatter {
    pub fn new(flavor: MistralFlavor, thinking: bool) -> Self {
        Self { flavor, thinking }
    }
}

impl Formatter for MistralFormatter {
    // Native BOS for all Mistral models
    fn bos(&self) -> Option<&str> {
        Some("<s>")
    }

    fn turn_start(&self, role: &Role) -> String {
        match self.flavor {
            MistralFlavor::V3Tekken => match role {
                Role::System => "[SYSTEM_PROMPT]".to_string(),
                Role::User => "[INST]".to_string(),
                // In Mistral, the model response follows IMMEDIATELY after [/INST] with no prefix
                Role::Assistant => String::new(),
            },
            MistralFlavor::Legacy => match role {
                // Legacy models had no system token; system text was packed inside [INST]
                Role::System => "[INST] ".to_string(),
                Role::User => "[INST]".to_string(),
                Role::Assistant => String::new(),
            },
        }
    }

    fn turn_end(&self, role: &Role) -> String {
        match self.flavor {
            MistralFlavor::V3Tekken => match role {
                Role::System => " [/SYSTEM_PROMPT]\n".to_string(),
                Role::User => " [/INST]\n".to_string(),
                // Each assistant turn in Mistral is closed with the classic EOS
                Role::Assistant => "</s>".to_string(),
            },
            MistralFlavor::Legacy => match role {
                Role::System => "\n\n".to_string(), // Separate system from user with a newline
                Role::User => "[/INST]".to_string(),
                Role::Assistant => "</s>".to_string(),
            },
        }
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        if self.thinking {
            // Vanilla Mistral-Nemo has no native hidden reasoning,
            // but for its popular R1 distillations (DeepSeek-R1-Distill-Mistral)
            // the standard is plain-text <think> tags
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

    fn tool_style(&self) -> ToolStyle {
        ToolStyle::Inline
    }

    fn format_tool_call(&self, name: &str, arguments: &str) -> String {
        let args = arguments.trim();
        // The official Mistral v3 spec requires function calls to be
        // a JSON array of objects strictly wrapped in [TOOL_CALLS] tokens
        format!(
            r#"[TOOL_CALLS][{{"name": "{}", "arguments": {}}}][\/TOOL_CALLS]"#,
            name, args
        )
    }

    fn format_tool_response(&self, _name: &str, content: &str) -> String {
        // Tool result is wrapped in the dedicated tool-results token
        format!("[TOOL_RESULTS]{}[/TOOL_RESULTS]", content.trim())
    }

    fn strip_tags(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        let mistral_structural_tags = [
            "<s>",
            "</s>",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "[INST]",
            "[/INST]",
        ];

        for tag in mistral_structural_tags {
            cleaned = cleaned.replace(tag, "");
        }

        let mistral_channels = [
            "[TOOL_CALLS]",
            "[/TOOL_CALLS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "<think>",
            "</think>",
        ];

        for tag in mistral_channels {
            cleaned = cleaned.replace(tag, "");
        }

        // Strip tool call blocks (greedy parsing)
        while let Some(start_idx) = cleaned.find("[TOOL_CALLS]") {
            if let Some(end_idx) = cleaned[start_idx..].find("[/TOOL_CALLS]") {
                let absolute_end_idx = start_idx + end_idx + "[/TOOL_CALLS]".len();
                cleaned.drain(start_idx..absolute_end_idx);
            } else {
                cleaned.drain(start_idx..);
                break;
            }
        }

        // Strip tool result blocks
        while let Some(start_idx) = cleaned.find("[TOOL_RESULTS]") {
            if let Some(end_idx) = cleaned[start_idx..].find("[/TOOL_RESULTS]") {
                let absolute_end_idx = start_idx + end_idx + "[/TOOL_RESULTS]".len();
                cleaned.drain(start_idx..absolute_end_idx);
            } else {
                cleaned.drain(start_idx..);
                break;
            }
        }

        cleaned.trim().to_string()
    }
}
