with open("integrations/mlx/cache_adapter.py", "r") as f:
    text = f.read()

text = text.replace("return block, values", """
        from turboquant.runtime.kv_interface import TurboQuantKeysView
        return TurboQuantKeysView(self, self._offset - keys.shape[-2], self._offset), values
""")

with open("integrations/mlx/cache_adapter.py", "w") as f:
    f.write(text)
