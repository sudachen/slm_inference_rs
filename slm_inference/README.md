# slm_inference

Backend-agnostic trait layer for running Small Language Model (SLM) inference in Rust.

## Idea

This crate defines a set of composable traits that abstract over the full inference pipeline —
from loading a GGUF model file to producing structured chat — without being tied to any specific
backend (llama.cpp, ik_llama.cpp, etc.).

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
- **`SlmContext`** — the stateful session: tokenises input, runs batched decode, and samples tokens.
- **`SlmBatch`** / **`SlmToken`** — low-level primitives for feeding tokens to the context.
- **`SlmInference`** — higher-level prefill/generate loop over a `SlmContext`; includes `save`/`rollback` for KV-cache branching.
- **`SlmHfModel`** — thin helper that downloads (or returns a cached) GGUF file from Hugging Face Hub.

Concrete backends (e.g. `slm_ikllama`) implement these traits against their own FFI layers.

---

## Usage Examples

### Setup

```rust
use slm_inference::{
    SlmDynamicFormatter, SlmOracle, SlmJsonOracle,
    SlmSimpleOracle, SlmAction,
};

// A backend crate (e.g. slm_ikllama) implements SlmModelConfig / SlmModel / SlmContext.
// Build a context from it, then wrap it in an oracle:
let formatter = SlmDynamicFormatter::try_from("qwen25")?;
let mut oracle = SlmSimpleOracle::new(context, formatter)?;
```

---

### Simple Q&A with `ask`

`ask` generates a reply and **discards the exchange from the KV cache** afterwards, so the
persistent context (system prompt, injected history) is always preserved unchanged.

```rust
oracle.system("You are a precise assistant. Answer in one sentence.")?;

// Plain generation — no chain-of-thought, no custom action
let answer = oracle.ask(false, "What is the capital of France?", None)?;
println!("{answer}");
// → "The capital of France is Paris."

// With chain-of-thought  (think = true)
let answer = oracle.ask(true, "Explain why the sky is blue.", None)?;
println!("Answer:  {}", answer.as_str());
println!("Thought: {:?}", answer.thought()); // Option<&str>

// Hard token limit via SlmAction
let answer = oracle.ask(false, "Tell me a short joke.", Some(SlmAction::token_limit(64)))?;
println!("{answer}");
```

`think = true` injects the formatter's `reasoning_trigger` prefix before generation starts,
prompting models that support chain-of-thought (QwQ, DeepSeek-R1 distillations, etc.) to reason
before answering.  The reasoning block is separated from the final answer by `split_thought` and
accessible via `answer.thought()`.

---

### Streaming output with `SlmAction::print_token`

`SlmAction::print_token()` returns an `SlmBoxedAction` callback that prints each decoded token
to stdout as it is generated, then signals `Next` so that other actions in the chain still run.
The full answer text is returned normally at the end.

```rust
oracle.system("You are a helpful assistant.")?;

// Tokens are printed to stdout one by one while the call blocks:
let answer = oracle.ask(
    false,
    "Write a haiku about Rust.",
    Some(SlmAction::print_token()),
)?;
println!("\n--- {} chars total ---", answer.as_str().len());
```

Combine with a token limit by passing a custom action that chains both behaviours:

```rust
use slm_inference::SlmAction;

let mut actions = [
    Some(SlmAction::print_token()),
    Some(SlmAction::token_limit(256)),
];
// Pass &mut actions to SlmInference::generate_until directly, or build a
// single boxed closure that calls both in sequence.
```

---

### Structured JSON extraction with `json_ask`

`json_ask` constrains the model's output to a `serde` / `schemars` type using
`llguidance` grammar-gated sampling.  The grammar is compiled from the JSON schema once per
type and cached; subsequent calls reuse the compiled parser.  The result is parsed and returned
as `Vec<T>` — one element per JSON object in the model's response array.

```rust
use serde::Deserialize;
use slm_inference::SlmJsonOracle;

#[derive(Deserialize, Debug, schemars::JsonSchema)]
struct EntityCard {
    term:     String,   // canonical name
    category: String,   // Character / Location / Neologism / ...
    clue:     String,   // one-sentence description
}

oracle.system(
    "You are an ontology extractor. Output raw JSON only — \
     no markdown, no commentary.",
)?;

// Inject background text without generating
oracle.user("Alice chased the White Rabbit down the rabbit hole.")?;

// Generate a JSON array of EntityCard objects, streaming tokens while running
let cards: Vec<EntityCard> = oracle.json_ask(
    false,                            // think
    "Extract all named entities.",    // prompt
    Some(SlmAction::print_token()),   // action — stream tokens to stdout
)?;

println!(); // newline after streamed tokens
for card in &cards {
    println!("{}: {} — {}", card.term, card.category, card.clue);
}
// → "Alice: Character — The protagonist who follows the White Rabbit."
// → "White Rabbit: Character — A rabbit that Alice chases underground."
```

> **Tip:** Use `oracle.set_max_answer_tokens(n)` before `json_ask` to increase the budget for
> long extractions (default is 1 024 tokens).

---

### Multi-turn conversation with `turn`

Unlike `ask`, `turn` **retains the exchange in the context**, building up a persistent
conversation history.  Use `save` / `rollback` to branch without losing earlier state.

```rust
oracle.system("You are a helpful assistant.")?;

let a1 = oracle.turn("What is Rust?", false, None)?;
println!("{a1}");

// The model sees the full history including a1
let a2 = oracle.turn("Give me a one-line code example.", false, None)?;
println!("{a2}");

// Snapshot current context, explore a branch, then restore
let state = oracle.save()?;
let branch = oracle.turn("Explain ownership in three words.", false, None)?;
println!("Branch: {branch}");
oracle.rollback(&state)?; // back to after a2
```

---

## Oracle Layer

`SlmSimpleOracle<I, F>` wraps an `SlmInference` and an `SlmFormatter` to provide a turn-aware
conversational interface.  Each `ask` call saves the KV-cache beforehand and rolls it back after
generation, so the context (system prompt + injected history) is never contaminated by the answer.

### `SlmOracle` methods

| Method | Retains context? | Description |
|---|---|---|
| `system(text)` | ✓ | Prefill a system turn |
| `user(text)` | ✓ | Prefill a user turn without generating |
| `assistant(text)` | ✓ | Prefill an assistant turn (history injection) |
| `ask(think, text, action)` | ✗ | Generate a reply; context rolls back after |
| `turn(text, think, action)` | ✓ | Generate a reply; exchange kept in context |
| `generate(role, text, think, reset, action, constraint)` | configurable | Low-level entry point |
| `save()` → `SlmOracleState` | — | Snapshot current turn position |
| `rollback(state)` | — | Restore to a previous snapshot |
| `clear()` | — | Reset context and turn state |
| `set_max_answer_tokens(n)` | — | Override the per-call token budget (default 1 024) |

---

## Formatter Layer

`SlmFormatter` abstracts chat-template rendering per model family.

```rust
pub trait SlmFormatter {
    fn bos(&self) -> Option<&str>;
    fn turn_start(&self, role: &SlmRole) -> String;
    fn turn_end(&self, role: &SlmRole) -> String;
    fn reasoning_bounds(&self) -> Option<(&str, &str)>;   // e.g. Some(("<think>", "</think>"))
    fn reasoning_trigger(&self) -> Option<&str>;           // prefix injected to activate CoT
    fn wrap_reasoning(&self, content: &str) -> String;
    fn tool_style(&self) -> SlmToolStyle;
    fn format_tool_call(&self, name: &str, arguments_json: &str) -> String;
    fn format_tool_response(&self, tool_name: &str, content: &str) -> String;
    fn strip_tags(&self, text: &str) -> String;
    fn clean(&self, text: &str) -> String;                 // strips reasoning blocks + tags
    fn strip_thought(&self, text: &str) -> (String, Option<String>);
}
```

### Tool styles

- **`SlmToolStyle::Inline`** — tool calls/responses are embedded inside the assistant turn (e.g. Gemma 4, Mistral, Qwen 2.5).
- **`SlmToolStyle::SeparateTurn`** — tool responses occupy a dedicated turn (e.g. Llama 3 `ipython` role).

### Built-in formatters (`slm_inference::models`)

| Key | Type | Chain-of-thought | Tool style |
|---|---|---|---|
| `"llama3"` | `Llama3Formatter` | ✓ (`<think>`) | `SeparateTurn` |
| `"gemma4"` | `GemmaFormatter` (Vanilla) | ✓ (`<\|channel>thought`) | `Inline` |
| `"gemma4-google"` | `GemmaFormatter` (GoogleOfficial) | ✓ | `Inline` |
| `"gemma4-unsloth"` | `GemmaFormatter` (UnslothFixed) | ✓ | `Inline` |
| `"mistral"` | `MistralFormatter` (V3Tekken) | ✓ (`<think>`) | `Inline` |
| `"mistral-legacy"` | `MistralFormatter` (Legacy) | ✓ (`<think>`) | `Inline` |
| `"qwen25"` | `Qwen25Formatter` | ✓ (`<think>`) | `Inline` |
| `"phi4"` | `Phi4Formatter` | ✓ (`<think>`) | `Inline` |

Select at runtime with `SlmDynamicFormatter::try_from("qwen25")?`.

---

## Roles

```rust
pub enum SlmRole {
    System,
    User,
    Assistant,
}
```

---

## Generation Control (`SlmAction`)

`SlmAction` is the control-flow signal returned by an `SlmBoxedAction` callback.
Callbacks have the signature:

```rust
FnMut(answer: &str, last_token: &str, n_tokens: usize, fork_id: usize) -> SlmAction
```

| Variant | Effect |
|---|---|
| `Continue` | Keep generating |
| `Finish` | Stop and return `SlmAnswer::Complete` |
| `Stop` | Stop and return `SlmAnswer::Incomplete` |
| `Delay` | Emit `SlmAnswer::Partial` and pause (future prompt resumes) |
| `Next` | Defer to the next callback in the chain |

Built-in factories:

```rust
SlmAction::token_limit(512)   // Finish after N tokens
SlmAction::print_token()      // Print each token to stdout, then Next
```

---

## Answer (`SlmAnswer`)

`SlmAnswer` wraps the generated text with its completion state and an optional reasoning trace:

```rust
pub enum SlmAnswer {
    Complete(String, usize, Option<String>),  // text, fork_id, thinking
    Partial(String, usize),
    Incomplete(String, usize),
}
```

| Method | Returns |
|---|---|
| `answer.as_str()` / `Deref` | Generated text |
| `answer.thought()` | `Option<&str>` — chain-of-thought content (after `split_thought`) |
| `answer.is_complete()` | `true` if generation ended naturally (EOS or constraint stop) |
| `answer.is_partial()` | `true` if paused by a `Delay` action |
| `answer.fork_id()` | Sequence ID in the KV cache |
| `answer.map(f)` | Transform the text string, preserving variant and metadata |
