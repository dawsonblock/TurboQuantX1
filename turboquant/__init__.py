"""
TurboQuant — research-stage KV-cache compression for selected MLX/Apple-Silicon RLM paths.

This package exposes the supported public surface for the current TurboQuant prototype.
Do not treat the package-level API as production-certified unless the corresponding
runtime-certification artifacts have been generated on Apple Silicon.

Public API
+--------
TurboQuantConfig          — runtime-immutable configuration
TurboQuantPipeline        — low-level encode/decode pipeline
KVCompressor  # Compatibility alias              — compatibility alias to TurboQuantKVCache
calibrate                 — calibration pass over representative data
"""

from turboquant._deps import (
    check_mlx_version,
    has_mlx,
    is_apple_silicon,
    require_mlx,
)
from turboquant.config import TurboQuantConfig

# Validate MLX version bounds at import time (no-op if MLX is absent)
check_mlx_version()

# Lazy imports for MLX-dependent runtime symbols
def __getattr__(name: str):
    if name == "calibrate":
        require_mlx("calibrate()")
        from turboquant.calibration.fit_quantizer import calibrate
        return calibrate
    elif name == "TurboQuantPipeline":
        require_mlx("TurboQuantPipeline")
        from turboquant.core.pipeline import TurboQuantPipeline
        return TurboQuantPipeline
    elif name == "KVCompressor  # Compatibility alias":
        require_mlx("KVCompressor  # Compatibility alias")
        from turboquant.runtime.kv_interface import TurboQuantKVCache as KVCompressor  # Compatibility alias
        return KVCompressor  # Compatibility alias
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TurboQuantConfig",
    "TurboQuantPipeline",
    "KVCompressor  # Compatibility alias",
    "calibrate",
    "check_mlx_version",
    "has_mlx",
    "is_apple_silicon",
    "require_mlx",
]

__version__ = "0.2.2"
