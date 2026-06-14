# slm_inference

Backend-agnostic trait layer for running Small Language Model (SLM) inference in Rust.

## Idea

This crate defines a set of composable traits that abstract over the full inference pipeline — from loading a GGUF model file to producing structured chat — without being tied to any specific backend (llama.cpp, ik_llama.cpp, etc.).

```
SlmModelConfig  →  load_gguf()  →  SlmModel
                                        ↓
                               SlmContextBuilder  →  SlmContext
                                                          ↓
                                                    SlmInference  +  SlmFormatter
                                                          ↓
                                                    SlmSimpleOracle  (implements SlmOracle)
```

## Core Traits

- **`SlmModelConfig`** — knows how to load a GGUF file and produce a `SlmModel`.
- **`SlmModel`** — owns the loaded weights and creates a `SlmContextBuilder`.
- **`SlmContextBuilder`** — configures sampling (temperature, top-k, top-p) and builds a `SlmContext`.
- **`SlmContext`** — the stateful session: tokenizes input, runs batched decode, and samples tokens.
- **`SlmBatch`** / **`SlmToken`** — low-level primitives for feeding tokens to the context.
- **`SlmInference`** — higher-level prefill/generate loop over a `SlmContext`; includes `save`/`rollback` for KV-cache branching.
- **`SlmHfModel`** — thin helper that downloads (or returns a cached) GGUF file from Hugging Face Hub.

Concrete backends (e.g. `slm_llama`, `slm_ikllama`) implement these traits against their own FFI layers.

## Oracle Layer

`SlmSimpleOracle<I, F>` wraps an `SlmInference` and an `SlmFormatter` to provide a turn-aware
conversational interface. Each `ask`/`think` call saves the KV-cache beforehand and rolls it back
after generation, so the context (system prompt + injected history) is never contaminated by the answer.

```rust
let context = /* build SlmContext from backend */;
let formatter = SlmDynamicFormatter::try_from("gemma4")?;
let mut oracle = SlmSimpleOracle::new(context, formatter)?;

oracle.system("You are a precise QA tool.")?;
oracle.user("Some background text...")?;    // inject context without generating

let answer = oracle.ask("What is X?", None)?;   // plain generation
let answer = oracle.think("Reason about X", None)?; // chain-of-thought

println!("{}", answer);                     // final answer text
println!("{:?}", answer.thought());         // Option<&str> — reasoning trace
```

### `SlmOracle` methods

| Method | Description |
|---|---|
| `system(text)` | Prefill a system turn |
| `user(text)` | Prefill a user turn without generating |
| `assistant(text)` | Prefill an assistant turn (history injection) |
| `tool(name, text)` | Prefill a tool-response turn without generating |
| `ask(text, brake)` | Generate a reply to `text`; context rolls back after |
| `think(text, brake)` | Like `ask`, but injects the reasoning trigger so the model produces chain-of-thought |
| `generate(role, text, think, brake)` | Low-level entry point for the above |
| `clear()` | Reset context and turn state |

## Formatter Layer

`SlmFormatter` abstracts chat-template formatting per model family.

```rust
pub trait SlmFormatter {
    fn bos(&self) -> Option<&str>;
    fn turn_start(&self, role: &SlmRole) -> String;
    fn turn_end(&self, role: &SlmRole) -> String;
    fn reasoning_bounds(&self) -> Option<(&str, &str)>;  // e.g. Some(("<think>", "</think>"))
    fn reasoning_trigger(&self) -> Option<&str>;          // prefix injected to activate CoT
    fn wrap_reasoning(&self, content: &str) -> String;
    fn tool_style(&self) -> SlmToolStyle;
    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String;
    fn format_tool_response(&self, tool_name: &str, content: &str) -> String;
    fn strip_tags(&self, text: &str) -> String;
    fn clean(&self, text: &str) -> String;        // strips reasoning blocks + tags
    fn strip_thought(&self, text: &str) -> (String, Option<String>); // separates answer from CoT
}
```

### Tool styles

- **`SlmToolStyle::Inline`** — tool calls and responses are embedded inside the assistant turn (e.g. Gemma 4).
- **`SlmToolStyle::SeparateTurn`** — tool responses occupy a dedicated turn with their own `turn_start`/`turn_end` (e.g. Llama 3).

### Built-in formatters (`slm_inference::models`)

| Key | Type | Thinking | Tool style |
|---|---|---|---|
| `"llama3"` | `Llama3Formatter` | — | `SeparateTurn` |
| `"gemma4"` | `GemmaFormatter` | ✓ | `Inline` |
| `"gemma4-google"` | `GemmaFormatter` (Google template) | ✓ | `Inline` |
| `"gemma4-unsloth"` | `GemmaFormatter` (unsloth fixed) | ✓ | `Inline` |
| `"mistral"` | `MistralFormatter` (v3 Tekken) | ✓ | `SeparateTurn` |
| `"mistral-legacy"` | `MistralFormatter` (legacy) | ✓ | `SeparateTurn` |
| `"qwen25"` | `Qwen25Formatter` | ✓ | `SeparateTurn` |
| `"phi4"` | `Phi4Formatter` | ✓ | `SeparateTurn` |

Use `SlmDynamicFormatter::try_from("gemma4")` to select at runtime by name.

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

`SlmBrake` controls when generation stops. Brake functions have the signature:

```rust
FnMut(answer: &str, last_token: &str, n_tokens: usize, fork_id: usize) -> SlmBrake
```

| Variant | Effect |
|---|---|
| `Continue` | Keep generating |
| `Finish` | Stop and return `SlmAnswer::Complete` |
| `Stop` | Stop and return `SlmAnswer::Incomplete` |
| `Delay` | Emit current token as `SlmAnswer::Partial`, pause |
| `Next` | Defers decision to the next brake in the chain |

Built-in factory:

```rust
SlmBrake::token_limit(512)   // stop after N tokens
```

## Answer

`SlmAnswer` wraps the generated text with its completion state and optional reasoning trace:

```rust
pub enum SlmAnswer {
    Complete(String, fork_id, Option<String>),  // answer + CoT thought
    Partial(String, fork_id),
    Incomplete(String, fork_id),
}
```

| Method | Returns |
|---|---|
| `answer.as_str()` / `Deref` | Final answer text |
| `answer.thought()` | `Option<&str>` — chain-of-thought content |
| `answer.is_complete()` | `true` if generation finished normally |
| `answer.fork_id()` | Sequence ID in the KV cache |
