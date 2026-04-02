import re
with open("turboquant/eval/compare.py", "r") as f:
    text = f.read()

pattern = r"legacy_cfg = TurboQuantConfig\.from_legacy_kwargs\([\s\S]*?turboquant_rotation=True,\n\s*\)"
replacement = """legacy_cfg = TurboQuantConfig(
            k_bits=getattr(self._config, "k_bits", 3),
            group_size=getattr(self._config, "k_group_size", 64)
        )"""

text = re.sub(pattern, replacement, text)

with open("turboquant/eval/compare.py", "w") as f:
    f.write(text)
