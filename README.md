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
| `SlmOracle` | High-level conversational interface |
| `SlmJsonOracle` | Extension of `SlmOracle` for structured JSON generation |
| `SlmSimpleOracle<I, F>` | Standard `SlmOracle` implementation |
| `SlmOracleState` | Snapshot returned by `save()` for context branching |
| `SlmFormatter` | Chat-template trait (BOS, turn delimiters, tool calling, reasoning bounds) |
| `SlmDynamicFormatter` | Runtime formatter selected by name string |
| `SlmAnswer` | Typed response; carries optional chain-of-thought `.thought()` |
| `SlmContext` | Low-level KV-cache / decode / sample interface |
| `SlmContextBuilder` | Builder for configuring and instantiating an `SlmContext` |
| `SlmKvType` | KV-cache quantization format (Q4, Q5, Q6, Q8, F16, F32, …) |
| `SlmInference` / `SlmSimpleInference` | Autoregressive token-generation loop |
| `SlmAction` / `SlmBoxedAction` | Generation control callbacks (token limit, streaming, …) |
| `SlmConstraint` | Token-level filter for constrained decoding (e.g. JSON grammar) |
| `SlmEditLevel` | Declares which KV-cache editing operations a backend supports |
| `SlmHfModel` | Hugging Face model descriptor (repo, filename, formatter name) |

**`SlmOracle` quick example:**

```rust
oracle.system("You are a helpful assistant.")?;
oracle.user("Some context...")?;

let answer = oracle.ask(false, "What is X?", None)?;   // plain generation
let answer = oracle.ask(true,  "Reason about X", None)?; // chain-of-thought

println!("{}", answer);                                  // final answer
println!("{:?}", answer.thought());                      // reasoning trace
```

**`SlmOracle` methods:**

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

**`SlmJsonOracle::json_ask` — structured JSON extraction:**

```rust
#[derive(Deserialize, schemars::JsonSchema)]
struct EntityCard { term: String, category: String, clue: String }

let cards: Vec<EntityCard> = oracle.json_ask(
    false,
    "Extract all named entities.",
    Some(SlmAction::print_token()),
)?;
```

**Supported chat templates (`SlmDynamicFormatter`):**

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
| `SlmModelConfig` | `ModelConfig` |
| `SlmModel` | `Model` |
| `SlmContextBuilder` | `Builder` |
| `SlmContext` | `Context` |
| `SlmBatch` / `SlmToken` | `Batch` / `Token` |

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

- `selector(model, backend, cpu)` — instantiates the requested model on the requested backend and returns a `Box<dyn SlmOracle>`
- `select_model(model_id)` — maps a `ModelId` variant to an `SlmHfModel` descriptor
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
  organizations, neologisms) as structured JSON using `SlmJsonOracle::json_ask`

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
