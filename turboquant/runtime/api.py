from turboquant.config import TurboQuantConfig


class TurboQuantRuntime:
    def __init__(self, config: TurboQuantConfig):
        raise NotImplementedError("TurboQuantRuntime is not yet a stable public wrapper; use integrations.mlx.cache_adapter.TurboQuantKCache")

    def step(self, keys, values):
        pass

    def attention(self, queries, state):
        pass
