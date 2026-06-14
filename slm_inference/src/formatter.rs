use crate::SlmRole;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlmToolStyle {
    Inline,
    SeparateTurn,
}

pub trait SlmFormatter {
    fn bos(&self) -> Option<&str>;
    fn turn_start(&self, role: &SlmRole) -> String;
    fn turn_end(&self, role: &SlmRole) -> String;

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
    fn tool_style(&self) -> SlmToolStyle;

    /// Formats a tool call by the model (when the model writes: "calling the calculator")
    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String;

    /// Formats a tool response.
    /// For the Inline style this will be just an inner block,
    /// for SeparateTurn — a full body inside turn_start(Tool) and turn_end(Tool).
    fn format_tool_response(&self, tool_name: &str, content: &str) -> String;

    fn strip_tags(&self, text: &str) -> String;

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
        let thinking = if thinking.is_empty() { None } else { Some(thinking) };
        (self.strip_tags(&cleaned).trim().to_string(), thinking)
    }
}

