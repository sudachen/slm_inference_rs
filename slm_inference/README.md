# slm_inference

Backend-agnostic trait layer for running Small Language Model (SLM) inference in Rust.

## Idea

This crate defines a set of composable traits that abstract over the full inference pipeline — from loading a GGUF model file to producing text — without being tied to any specific backend (llama.cpp, bitnet, etc.).

```
SlmModelConfig  →  load_gguf()  →  SlmModel
                                        ↓
                               SlmContextBuilder  →  SlmContext
                                                          ↓
                                                    SlmInference::inference()
```

- **`SlmModelConfig`** — knows how to load a GGUF file and produce a `SlmModel`.
- **`SlmModel`** — owns the loaded weights and creates a `SlmContextBuilder`.
- **`SlmContextBuilder`** — configures sampling (temperature, top-k, top-p) and builds a `SlmContext`.
- **`SlmContext`** — the stateful session: tokenizes input, runs batched decode, and samples tokens.
- **`SlmBatch`** / **`SlmToken`** — low-level primitives for feeding tokens to the context.
- **`SlmInference`** — a blanket `inference(prompt, max_tokens) -> String` impl provided automatically for any `SlmContext`.
- **`HfModelInfo`** — thin helper that downloads (or returns a cached) GGUF file from Hugging Face Hub.

Concrete backends (e.g. `slm_llama`, `slm_bitnet`) implement these traits against their own FFI layers. 
