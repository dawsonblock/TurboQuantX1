---
name: benchmark-script-update
description: Workflow command scaffold for benchmark-script-update in TurboQuantX1.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /benchmark-script-update

Use this workflow when working on **benchmark-script-update** in `TurboQuantX1`.

## Goal

Refactors or updates benchmarking scripts, often to improve measurement accuracy, add new metrics, or integrate new features such as cache changes.

## Common Files

- `benchmarks/exploratory/bench_decode_step.py`
- `benchmarks/exploratory/bench_k_encode.py`
- `benchmarks/exploratory/bench_memory.py`
- `scripts/run_benchmarks.sh`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add scripts in benchmarks/exploratory/ or scripts/ (e.g., bench_decode_step.py, run_benchmarks.sh)
- Update related shell scripts or pipeline scripts
- Optionally update related core modules if benchmarked functionality changes

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.