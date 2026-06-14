# slm_ikllama

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)-backed implementation of the
[`slm_inference`](../../slm_inference) trait layer.

ik_llama.cpp is a performance-oriented fork of llama.cpp with improved CPU/GPU kernels,
new quantization types, first-class DeepSeek/MLA support, and fused MoE operations.

## Idea

This crate wires the `slm_inference` abstract traits to the ik_llama.cpp runtime via the
[`slm_ikllama_sys`](../slm_ikllama_sys) FFI bindings.

| `slm_inference` trait | This crate's type |
|---|---|
| `SlmModelConfig` | `ModelConfig` |
| `SlmModel` | `Model` |
| `SlmContextBuilder<Context>` | `Builder` |
| `SlmContext` | `Context` |
| `SlmBatch` / `SlmToken` | `Batch` / `Token` |

## Key details

- **`ModelConfig`** wraps `llama_model_params` and exposes builder methods for GPU layers,
  main GPU selection, mmap, mlock, and multi-GPU split mode.
- **`Model`** holds a ref-counted pointer to the loaded `llama_model`; it is cheaply
  `Clone`able for sharing across contexts.
- **`Builder`** configures `llama_context_params` (context size, batch size, KV cache
  quantization with optional Hadamard transform, flash attention) and builds the sampler:
  greedy when `temperature ≤ 0`, otherwise top-k → top-p → temperature.
- **`Context`** implements the full `SlmContext` protocol: batched decode via `llama_decode`,
  manual sampling from logits, tokenization via `llama_tokenize`, piece extraction via
  `llama_token_to_piece`, and KV cache editing (`truncate`, `cut`, `drop`).
- The ik_llama.cpp backend is initialized exactly once (guarded by an `AtomicBool`) and
  its log output is routed to `tracing`.
- `edit_level` is `SlmEditLevel::Cut` — the KV cache supports mid-sequence removal,
  enabling efficient context management without full resets.

## KV cache quantization

`Builder::with_gen_type_kv(k, v)` accepts `SlmKvType` values and maps them to ik_llama.cpp
quantization types with automatic Hadamard-transform activation:

| `SlmKvType` | GGML type | Hadamard |
|---|---|---|
| `Q4` | `Q4_0` | ✓ |
| `Q5` | `Q5_0` | ✓ |
| `Q6` | `Q6_0` | ✓ |
| `Q8` | `Q8_0` | ✓ |
| `RawQ8` | `Q8_0` | — |
| `F16` | `F16` | — |
| `F32` | `F32` | — |

## Multi-GPU split modes

`ModelConfig::with_split_mode(SplitMode)` controls how the model is distributed across GPUs:

| Variant | Description |
|---|---|
| `None` | Single GPU |
| `Layer` | Split layers and KV across GPUs |
| `Row` | Split with attention-layer tensor parallelism |
| `Tensor` | Experimental full tensor parallelism |

## Features

| Feature | Description |
|---|---|
| `cuda` *(default)* | NVIDIA CUDA via `slm_ikllama_sys/cuda` |
| `native` *(default)* | CPU compiled for the host architecture via `slm_ikllama_sys/native` |
