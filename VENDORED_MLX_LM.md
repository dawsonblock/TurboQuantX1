# Vendored upstream boundary

This file records the intentional patch boundary against upstream `mlx_lm`.

Upstream source: `mlx_lm`
Version/Commit: v0.29.1

## Modified files

- `mlx_lm/models/llama.py`: Wired to `turboquant_streaming_attention`. Required for supported runtime.
- `mlx_lm/models/gemma.py`: Wired to `turboquant_streaming_attention`. Required for supported runtime.
- `mlx_lm/generate.py`: Hooked `upgrade_cache_list` to optionally inject TurboQuant. Required for supported runtime.
- `mlx_lm/models/base.py`: Modified SDPA dispatch. Included for broader structural tests and scaffolding, though only Llama/Gemma are officially supported.
