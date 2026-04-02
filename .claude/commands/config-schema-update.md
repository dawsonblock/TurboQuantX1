---
name: config-schema-update
description: Workflow command scaffold for config-schema-update in TurboQuantX1.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /config-schema-update

Use this workflow when working on **config-schema-update** in `TurboQuantX1`.

## Goal

Updates to the configuration schema, often involving renaming fields, changing config formats, or improving config dataclass usage. These changes are frequently accompanied by updates to integration adapters and related tests to ensure compatibility.

## Common Files

- `turboquant/config.py`
- `integrations/mlx/cache_adapter.py`
- `integrations/mlx/upgrade.py`
- `tests/unit/test_kv_interface.py`
- `benchmarks/exploratory/bench_dense_vs_turboquant.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Modify turboquant/config.py to update config schema or fields
- Update integration adapters (e.g., integrations/mlx/cache_adapter.py, integrations/mlx/upgrade.py) to use new config fields
- Update or fix tests and benchmarks that rely on config (e.g., tests/unit/test_kv_interface.py, benchmarks/exploratory/bench_dense_vs_turboquant.py)
- Update documentation or comments if necessary

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.