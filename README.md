# slm_inference_rs

A Rust workspace for running small language models (SLMs) locally via GGUF backends.
Provides an abstract trait layer over different llama.cpp-derived runtimes, a chat-template
formatting system, and a high-level conversational oracle API with built-in chain-of-thought
and structured JSON output support.

## Workspace layout

```
slm_inference_rs/
├── slm_inference/          # Core abstract trait layer (the public API)
├── crates/
│   ├── slm_llama/          # Backend: llama.cpp via llama-cpp-sys-2
│   ├── slm_ikllama/        # Backend: ik_llama.cpp (performance fork)
│   ├── slm_ikllama_sys/    # FFI bindings for ik_llama.cpp (git submodule)
│   ├── epubscan/           # EPUB reader utility
│   └── fb2scan/            # FB2 reader utility
└── examples/
    ├── backend/            # Shared helper: model/backend selection for examples
    └── lore_extractor/     # Example: entity extraction and Q&A from books
```

## Crates

### `slm_inference`

The core crate. Defines all abstract traits and the high-level oracle API that the rest of the
workspace builds on. No runtime dependencies — backends are chosen by the consuming binary.

**Key traits and types:**

| Item | Description |
|---|---|
| `Assistant` | High-level conversational interface |
| `State` | Snapshot returned by `save()` for context branching |
| `Formatter` | Chat-template trait (BOS, turn delimiters, tool calling, reasoning bounds) |
| `DynamicFormatter` | Runtime formatter selected by name string |
| `Answer` | Typed response; carries optional chain-of-thought `.thought()` |
| `Context` | Low-level KV-cache / decode / sample interface |
| `ContextBuilder` | Builder for configuring and instantiating a `Context` |
| `KvType` | KV-cache quantization format (Q4, Q5, Q6, Q8, F16, F32, …) |
| `Inference` / `SimpleInference` | Autoregressive token-generation loop |
| `Action` / `BoxedAction` | Generation control callbacks (token limit, streaming, …) |
| `Constraint` | Token-level filter for constrained decoding (e.g. JSON grammar) |
| `EditLevel` | Declares which KV-cache editing operations a backend supports |
| `HfModel` | Hugging Face model descriptor (repo, filename, formatter name) |

**`Assistant` quick example:**

```rust
assistant.system("You are a helpful assistant.")?;
assistant.user("Some context...")?;

let answer = assistant.ask(false, "What is X?", None)?;   // plain generation
let answer = assistant.ask(true,  "Reason about X", None)?; // chain-of-thought

println!("{}", answer);                                    // final answer
println!("{:?}", answer.thought());                        // reasoning trace
```

**`Assistant` methods:**

| Method | Retains context? | Description |
|---|---|---|
| `system(text)` | ✓ | Prefill a system turn |
| `user(text)` | ✓ | Prefill a user turn without generating |
| `assistant(text)` | ✓ | Prefill an assistant turn (history injection) |
| `ask(think, text, action)` | ✗ | Generate a reply; context rolls back after |
| `turn(text, think, action)` | ✓ | Generate a reply; exchange kept in context |
| `json_ask(think, text, action)` | ✗ | Constrained JSON generation; returns `Answer<Vec<T>>` |
| `ask_values(think, text, action)` | ✗ | Constrained JSON generation; returns `Vec<T>` directly |
| `choose(think, text, action)` | ✗ | Constrained enum selection; returns `Answer<T>` |
| `choose_value(think, text, action)` | ✗ | Constrained enum selection; returns `T` directly |
| `generate(role, text, think, reset, action, constraint)` | configurable | Low-level entry point |
| `save()` → `State` | — | Snapshot current turn position |
| `rollback(state)` | — | Restore to a previous snapshot |
| `clear()` | — | Reset context and turn state |
| `set_max_answer_tokens(n)` | — | Override the per-call token budget (default 1 024) |
| `tokens_n()` | — | Total number of tokens currently in the context |
| `vocab()` | — | Reference to the active `Vocab` |
| `formatter()` | — | Reference to the active `Formatter` |

**`Assistant::json_ask` — structured JSON extraction:**

```rust
#[derive(Deserialize, schemars::JsonSchema)]
struct EntityCard { term: String, category: String, clue: String }

let cards: Vec<EntityCard> = assistant.ask_values(
    false,
    "Extract all named entities.",
    Action::print_token(),
)?;
```

**Supported chat templates (`DynamicFormatter`):**

| Key | Model family |
|---|---|
| `gemma4` | Gemma 4 (unsloth variant, thinking enabled) |
| `gemma4-google` | Gemma 4 (Google official template) |
| `gemma4-unsloth` | Gemma 4 (unsloth fixed template) |
| `llama3` | Llama 3 / Llama 3.1 |
| `mistral` | Mistral v3 Tekken |
| `mistral-legacy` | Mistral legacy |
| `qwen25` | Qwen 2.5 / QwQ / DeepSeek-R1-Distill-Qwen |
| `phi4` | Phi-4 mini |

---

### `slm_llama`

Implements the `slm_inference` trait layer using [llama.cpp](https://github.com/ggerganov/llama.cpp)
via the `llama-cpp-sys-2` crate.

| `slm_inference` trait | This crate's type |
|---|---|
| `ModelConfig` | `ModelConfig` |
| `Model` | `Model` |
| `ContextBuilder` | `Builder` |
| `Context` | `Context` |
| `Batch` | `Batch` |

**Accelerator features:**

| Feature | Accelerator |
|---|---|
| *(default)* | CPU only |
| `vulkan` | Vulkan |
| `metal` | Apple Metal |

---

### `slm_ikllama` / `slm_ikllama_sys`

Implements the `slm_inference` trait layer using [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp),
a performance-oriented fork of llama.cpp with additional quantization and CUDA kernel optimizations.

`slm_ikllama_sys` contains the raw FFI bindings generated by `bindgen` and the CMake build script
for the `ik_llama.cpp` submodule.

**Accelerator features:**

| Feature | Description |
|---|---|
| `cuda` | NVIDIA CUDA (default) |
| `native` | CPU with native arch tuning (default) |

---

### `epubscan`

Lightweight EPUB reader that iterates over sections and exposes their plain text for feeding
into an oracle. Used by the `lore_extractor` example.

---

### `fb2scan`

FB2 book format reader, analogous to `epubscan`.

---

## Example: `backend`

A shared library crate (`examples/backend`) used by all example binaries. Provides:

- `selector(model, backend, cpu)` — instantiates the requested model on the requested backend and returns an `slm::Assistant`
- `select_model(model_id)` — maps a `ModelId` variant to an `HfModel` descriptor
- `ModelId` / `BackendId` — `clap`-compatible enums for CLI argument parsing

**Supported models (`ModelId`):**

| Variant | Repo | File |
|---|---|---|
| `gemma4eb` | `unsloth/gemma-4-E4B-it-GGUF` | `gemma-4-E4B-it-IQ4_XS.gguf` |
| `gemma12b` *(default)* | `unsloth/gemma-4-12B-it-qat-GGUF` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` |
| `phi4` | `bartowski/microsoft_Phi-4-mini-reasoning-GGUF` | `microsoft_Phi-4-mini-reasoning-IQ4_XS.gguf` |
| `qwen25` | `bartowski/Qwen2.5-7B-Instruct-GGUF` | `Qwen2.5-7B-Instruct-IQ4_XS.gguf` |

**Cargo features:**

| Feature | Backend enabled |
|---|---|
| `ikllama` *(default)* | `slm_ikllama` with CUDA + native |
| `llama` | `slm_llama` with Vulkan |

---

## Example: `lore_extractor`

A command-line tool that reads EPUB books and applies language-model analysis — Yes/No
fact-checking and structured entity extraction.

**Subcommands:**

- `say-hi` — sanity check: loads the model and asks it to say "Hi"
- `yes-no` — reads EPUB sections into the context, then answers a set of Yes/No questions
  from a JSON file; supports optional chain-of-thought via `--think`
- `ents` — reads EPUB sections and extracts named entities (characters, locations,
  organizations, neologisms) as structured JSON using `Assistant::ask_values`

**CLI flags (global):**

| Flag | Description |
|---|---|
| `--model` | Model to use (see `ModelId` variants above; default `gemma12b`) |
| `--backend` | Backend to use (`llama`, `ikllama`; default `ikllama`) |
| `--cpu` | Disable GPU offloading (sets `n_gpu_layers = 0`) |

Models are downloaded automatically from Hugging Face Hub on first run.

```
cargo run -p lore_extractor -- yes-no \
    --model gemma12b --backend ikllama \
    --think \
    --input book.epub \
    --questions yesno.json

cargo run -p lore_extractor -- ents \
    --model gemma12b --backend ikllama \
    book.epub
```

## License

Licensed under either of [MIT](LICENSE-MIT.md) or [Apache-2.0](LICENSE-APACHE.md) at your option.
