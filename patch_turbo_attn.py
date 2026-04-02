with open("turboquant/runtime/attention.py", "r") as f:
    text = f.read()

text = text.replace("attn = mx.softmax(scores, axis=-1)\n    return attn @ vals", """
    attn = mx.softmax(scores, axis=-1)
    if queries.shape[-3] != vals.shape[-3]:
        n_rep = queries.shape[-3] // vals.shape[-3]
        vals = mx.repeat(vals, n_rep, axis=-3)
    return attn @ vals
""")

with open("turboquant/runtime/attention.py", "w") as f:
    f.write(text)
