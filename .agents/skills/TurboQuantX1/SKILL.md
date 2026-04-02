```markdown
# TurboQuantX1 Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns, coding conventions, and common workflows for contributing to the TurboQuantX1 Python codebase. TurboQuantX1 is a high-performance quantization and runtime library, with a focus on modularity, benchmarking, and integration flexibility. The repository emphasizes clear configuration management, robust benchmarking, and comprehensive testing. This guide will help you understand how to make effective contributions, maintain consistency, and follow established workflows.

## Coding Conventions

**File Naming:**  
- Use `snake_case` for Python files and modules.
  - Example: `cache_adapter.py`, `bench_decode_step.py`

**Import Style:**  
- Prefer **relative imports** within modules.
  - Example:
    ```python
    from .core import pipeline
    from ..config import Config
    ```

**Export Style:**  
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["Pipeline", "QJL", "ResidualCodec"]
    ```

**Commit Message Patterns:**  
- Prefixes: `fix`, `chore`, `refactor`
- Example:  
  ```
  fix: handle edge case in cache_adapter for empty config
  refactor: rename QJL class for clarity
  chore: update benchmarks for new metrics
  ```

## Workflows

### Config Schema Update
**Trigger:** When you need to change or refactor the configuration structure or naming conventions.  
**Command:** `/update-config-schema`

1. Modify `turboquant/config.py` to update the config schema or fields.
2. Update integration adapters to use new config fields:
   - `integrations/mlx/cache_adapter.py`
   - `integrations/mlx/upgrade.py`
3. Update or fix tests and benchmarks that rely on config:
   - `tests/unit/test_kv_interface.py`
   - `benchmarks/exploratory/bench_dense_vs_turboquant.py`
4. Update documentation or comments if necessary.

**Example:**
```python
# turboquant/config.py
@dataclass
class Config:
    cache_size: int
    enable_quant: bool = True  # New field added
```

---

### Benchmark Script Update
**Trigger:** When you want to improve, refactor, or extend the benchmarking capabilities.  
**Command:** `/update-benchmarks`

1. Edit or add scripts in `benchmarks/exploratory/` or `scripts/` (e.g., `bench_decode_step.py`, `run_benchmarks.sh`).
2. Update related shell scripts or pipeline scripts.
3. Optionally update related core modules if benchmarked functionality changes.

**Example:**
```python
# benchmarks/exploratory/bench_decode_step.py
def benchmark_decode_step():
    # Add new metric collection
    start = time.time()
    ...
    print("Latency:", time.time() - start)
```

---

### Test Suite Expansion or Fix
**Trigger:** When adding new features, changing config, or fixing bugs that require new or updated tests.  
**Command:** `/expand-tests`

1. Add or update test files in `tests/integration_mlx/` or `tests/unit/`.
2. Ensure tests cover new or changed functionality.
3. Run tests to verify passing status.

**Example:**
```python
# tests/unit/test_kv_interface.py
def test_new_config_field():
    config = Config(cache_size=128, enable_quant=False)
    assert not config.enable_quant
```

---

### Core Module Refactor
**Trigger:** When you want to improve code quality, rename classes, or refactor core logic.  
**Command:** `/refactor-core`

1. Refactor or rename classes/functions in `turboquant/core/` or `turboquant/runtime/`.
2. Update all references in benchmarks, integration adapters, and `__init__.py` files.
3. Update related tests and documentation for consistency.

**Example:**
```python
# turboquant/core/qjl.py
class QJLBlock:  # Renamed from QJL
    ...
```
Update usage:
```python
from turboquant.core.qjl import QJLBlock
```

## Testing Patterns

- **Framework:** Unknown (likely pytest or unittest, but not specified)
- **Test File Pattern:** Tests are in `tests/integration_mlx/` and `tests/unit/`, using `test_*.py` naming.
- **Test Example:**
  ```python
  # tests/unit/test_attention_score_block.py
  def test_attention_block_output_shape():
      block = AttentionScoreBlock(...)
      output = block.forward(...)
      assert output.shape == (batch_size, seq_len, hidden_dim)
  ```
- **Note:** Some legacy or exploratory tests may use `.test.ts` pattern, but Python is the main language.

## Commands

| Command                | Purpose                                                      |
|------------------------|--------------------------------------------------------------|
| /update-config-schema  | Update or refactor the configuration schema and adapters     |
| /update-benchmarks     | Refactor or extend benchmarking scripts and metrics          |
| /expand-tests          | Add or update tests for new features or bugfixes             |
| /refactor-core         | Refactor or rename core modules and update all references    |
```
