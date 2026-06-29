use crate::slm::{Formatter, Role, ToolStyle};

/// [`Formatter`] for Google Gemma 4 models.
///
/// Uses Gemma's `<|turn>` / `<turn|>` turn delimiters and `<|channel>thought` /
/// `<channel|>` reasoning tags.  The exact format varies by [`Gemma4Flavor`].
pub struct GemmaFormatter {
    flavor: Gemma4Flavor,
    thinking: bool,
}

/// Selects the exact Gemma 4 chat-template variant to use.
///
/// Different fine-tuned releases of Gemma 4 use slightly different token sequences
/// for tool calls, necessitating per-flavor formatting.
pub enum Gemma4Flavor {
    /// Official Google template as shipped with the base Gemma 4 checkpoint.
    GoogleOfficial,
    /// Template used by Unsloth fine-tunes with a corrected tool-call format.
    UnslothFixed,
    /// Simplified vanilla format suitable for most community fine-tunes.
    Vanilla,
}

impl GemmaFormatter {
    /// Create a new Gemma formatter with the specified flavor and reasoning support.
    pub fn new(flavor: Gemma4Flavor, thinking: bool) -> Self {
        Self { flavor, thinking }
    }
}

impl Formatter for GemmaFormatter {
    fn bos(&self) -> Option<&str> {
        Some("<bos>")
    }
    fn turn_start(&self, role: &Role) -> String {
        match role {
            Role::System => "<|turn>system\n".to_string(),
            Role::User => "<|turn>user\n".to_string(),
            Role::Assistant => "<|turn>model\n".to_string(),
        }
    }

    fn turn_end(&self, role: &Role) -> String {
        match role {
            _ => "<turn|>\n".to_string(),
        }
    }

    fn reasoning_bounds(&self) -> Option<(&str, &str)> {
        if self.thinking {
            Some(("<|channel>thought\n", "<channel|>"))
        } else {
            None
        }
    }

    fn wrap_reasoning(&self, content: &str) -> String {
        if self.thinking {
            format!("<|channel>thought\n{}<channel|>", content.trim())
        } else {
            content.to_string()
        }
    }

    fn reasoning_trigger(&self) -> Option<&str> {
        if self.thinking {
            Some("<|channel>thought\n")
        } else {
            None
        }
    }

    fn tool_style(&self) -> ToolStyle {
        ToolStyle::Inline
    }

    fn format_tool_call(&self, name: &str, arguments: &str) -> String {
        let args = arguments.trim();
        match self.flavor {
            Gemma4Flavor::GoogleOfficial => {
                format!("<|tool_call>call:{}{{{{{}}}}}<tool_call|>", name, args)
            }
            _ => format!("<|tool_call>call:{}{{{}}}<tool_call|>", name, args),
        }
    }

    fn format_tool_response(&self, name: &str, content: &str) -> String {
        format!(
            "<|tool_response>response:{}{{value:{}}}<tool_response|>",
            name,
            content.trim()
        )
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

        while let Some(start_idx) = cleaned.find("<|tool_response>") {
            if let Some(end_idx) = cleaned[start_idx..].find("<tool_response|>") {
                let absolute_end_idx = start_idx + end_idx + "<tool_response|>".len();
                cleaned.drain(start_idx..absolute_end_idx);
            } else {
                cleaned.drain(start_idx..);
                break;
            }
        }

        cleaned.trim().to_string()
    }
}
