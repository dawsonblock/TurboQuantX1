with open("turboquant/runtime/attention.py", "r") as f:
    text = f.read()

text = text.replace("def turboquant_streaming_attention(queries, keys_view, scale=1.0):",
"def turboquant_streaming_attention(queries, keys_view, scale=1.0, mask=None):")

text = text.replace("scores = mx.concatenate(scores, axis=-1)",
"""scores = mx.concatenate(scores, axis=-1)
    if mask is not None:
        scores = scores + mask""")

with open("turboquant/runtime/attention.py", "w") as f:
    f.write(text)

with open("mlx_lm/models/llama.py", "r") as f:
    text2 = f.read()

text2 = text2.replace("""output = turboquant_streaming_attention(
                queries,
                keys,
                scale=self.scale,
            )""", """output = turboquant_streaming_attention(
                queries,
                keys,
                scale=self.scale,
                mask=mask,
            )""")

with open("mlx_lm/models/llama.py", "w") as f:
    f.write(text2)
