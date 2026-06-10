# slm_inference

Backend-agnostic trait layer for running Small Language Model (SLM) inference in Rust.

## Idea

This crate defines a set of composable traits that abstract over the full inference pipeline — from loading a GGUF model file to producing structured chat — without being tied to any specific backend (llama.cpp, bitnet, etc.).

```
SlmModelConfig  →  load_gguf()  →  SlmModel
                                        ↓
                               SlmContextBuilder  →  SlmContext
                                                          ↓
                                                    SlmInference  +  SlmFormatter
                                                          ↓
                                                    SlmSimpleChat  (implements SlmChat)
```

## Core Traits

- **`SlmModelConfig`** — knows how to load a GGUF file and produce a `SlmModel`.
- **`SlmModel`** — owns the loaded weights and creates a `SlmContextBuilder`.
- **`SlmContextBuilder`** — configures sampling (temperature, top-k, top-p) and builds a `SlmContext`.
- **`SlmContext`** — the stateful session: tokenizes input, runs batched decode, and samples tokens.
- **`SlmBatch`** / **`SlmToken`** — low-level primitives for feeding tokens to the context.
- **`SlmInference`** — higher-level prefill/generate loop over a `SlmContext`.
- **`SlmRollback`** — save/rollback the KV-cache state (for branching conversations).
- **`SlmHfModel`** — thin helper that downloads (or returns a cached) GGUF file from Hugging Face Hub.

Concrete backends (e.g. `slm_llama`, `slm_bitnet`) implement these traits against their own FFI layers.

## Chat Layer

`SlmSimpleChat<I, F>` wraps an `SlmInference` and an `SlmFormatter` to provide a turn-aware chat interface:

```rust
let context = /* build SlmContext from backend */;
let formatter = SlmDynamicFormatter::try_from("llama3")?;
let mut chat = SlmSimpleChat::new(context, formatter)?;

chat.system("You are a helpful assistant.")?;
let answer = chat.user_ask("What is 2+2?", None)?;
println!("{}", answer);
```

### `SlmChat` methods

| Method | Description |
|---|---|
| `system(text)` | Prefill a system turn |
| `user(text)` | Prefill a user turn without generating |
| `assistant(text)` | Prefill an assistant turn (history injection) |
| `tool(name, text)` | Prefill a tool-response turn without generating |
| `user_ask(text, brake)` | Prefill user turn and generate assistant reply |
| `tool_ask(name, text, brake)` | Prefill tool turn and generate assistant reply |
| `continue_answer(brake)` | Continue generating from current position |
| `clear()` | Reset context and turn state |
| `save()` / `rollback()` | KV-cache snapshot for branching |

## Formatter Layer

`SlmFormatter` abstracts chat-template formatting per model family.

```rust
pub trait SlmFormatter {
    fn bos(&self) -> Option<&str>;
    fn turn_start(&self, role: &SlmRole) -> String;
    fn turn_end(&self, role: &SlmRole) -> String;
    fn reasoning_bounds(&self) -> Option<(&str, &str)>;
    fn wrap_reasoning(&self, content: &str) -> String;
    fn tool_style(&self) -> SlmToolStyle;
    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String;
    fn format_tool_response(&self, tool_name: &str, content: &str) -> String;
    fn strip_tags(&self, text: &str) -> String;
    fn clean(&self, text: &str) -> String { /* strips reasoning blocks + tags */ }
}
```

### Tool styles

- **`SlmToolStyle::Inline`** — tool calls and responses are embedded inside the assistant turn (e.g. Gemma 4).
- **`SlmToolStyle::SeparateTurn`** — tool responses occupy a dedicated turn with their own `turn_start`/`turn_end` (e.g. Llama 3).

### Built-in formatters (`slm_inference::models`)

| Key | Type | Tool style |
|---|---|---|
| `"llama3"` | `Llama3Formatter` | `SeparateTurn` |
| `"gemma4"` | `GemmaFormatter` | `Inline` |

Use `SlmDynamicFormatter::try_from("llama3")` to select at runtime by name.

## Roles

```rust
pub enum SlmRole {
    System,
    User,
    Assistant,
    Tool(String),   // carries the tool name
}
```

Helper constructors: `SlmRole::tool("calculator")`, `role.tool_name()`, `role.is_tool()`.

## Generation Control

`SlmBrake` / `SlmBrakeFilter` control when generation stops:

```rust
SlmBrake::token_limit(512)            // stop after N tokens
SlmBrake::on_str("<|eot_id|>")        // stop on a specific string
```

`SlmAnswer` wraps the generated text and exposes whether the answer is `Complete` or `Partial` (hit the brake mid-stream).
