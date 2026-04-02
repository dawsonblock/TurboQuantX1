# Runtime Certification

The runtime certification script proves the following claim **only**:
For supported model families (Llama/Gemma) on Apple Silicon with MLX, the TurboQuant cache path is active, generation succeeds, and evidence artifacts are produced for memory, latency, and quality.

Generic CI does not certify the runtime.

Artifacts included:
- model family
- model identifier
- dense baseline vs TurboQuant delta
- memory / latency output
- experimental kernel flag status
- environment stamp
