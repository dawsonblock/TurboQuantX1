import re
with open("turboquant/core/qjl.py", "r") as f:
    text = f.read()

text = text.replace("return scores * (norm_scale[..., None, :] / q_norm[..., :, None])", """
        if q_norm.shape[-2] != norm_scale.shape[-2]:
            n_rep = q_norm.shape[-2] // norm_scale.shape[-2]
            norm_scale = mx.repeat(norm_scale, n_rep, axis=-2)
        
        return scores * (norm_scale[..., None, :] / q_norm[..., :, None])
""")

with open("turboquant/core/qjl.py", "w") as f:
    f.write(text)
