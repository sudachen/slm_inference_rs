# slm_llama

`llama-cpp-sys-2`-backed implementation of the [`slm_inference`] trait layer.

## Idea

This crate wires the `slm_inference` abstract traits to the [llama.cpp](https://github.com/ggerganov/llama.cpp) runtime via the `llama-cpp-sys-2` FFI bindings.

| `slm_inference` trait | This crate's type |
|---|---|
| `SlmModelConfig` | `ModelConfig` |
| `SlmModel` | `Model` |
| `SlmContextBuilder<Context>` | `Builder` |
| `SlmContext` | `Context` |
| `SlmBatch` / `SlmToken` | `Batch` / `Token` |

**Key details:**

- `ModelConfig` wraps `llama_model_params` and exposes builder methods for GPU layers, mmap, mlock, and multi-GPU split mode.
- `Model` holds a ref-counted pointer to the loaded `llama_model`; it is cheaply `Clone`able for sharing across contexts.
- `Builder` configures `llama_context_params` (context size, batch size, flash-attention, KV cache quantization) and builds the sampler chain: greedy when `temperature ≤ 0`, otherwise top-k → top-p → temperature.
- `Context` implements the full `SlmContext` protocol: batched decode via `llama_decode`, sampling via `llama_sampler_sample`, tokenization via `llama_tokenize`, and piece extraction via `llama_token_to_piece`.
- The llama.cpp backend is initialized exactly once (guarded by an `AtomicBool`) and its log output is routed to `tracing`.

## Features

| Feature | Accelerator |
|---|---|
| *(default)* | CPU only |
| `cuda` | NVIDIA CUDA |
| `vulkan` | Vulkan |
| `metal` | Apple Metal |
