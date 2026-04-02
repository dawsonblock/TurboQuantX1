with open("turboquant/runtime/attention.py", "r") as f:
    text = f.read()

text = text.replace("def turboquant_streaming_attention(queries, cache, scale=1.0):",
"""def turboquant_streaming_attention(queries, keys_view, scale=1.0):
    cache = keys_view.cache"""
)

with open("turboquant/runtime/attention.py", "w") as f:
    f.write(text)
