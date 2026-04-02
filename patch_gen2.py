import sys

with open('mlx_lm/generate.py', 'r') as f:
    text = f.read()

old_call = """    _cfg = _TQConfig(
        k_bits=turboquant_main_bits,
        k_group_size=turboquant_group_size,
        rotation=turboquant_rotation,
        residual_topk=turboquant_residual_topk,
        v_bits=turboquant_v_bits,
        v_group_size=turboquant_v_group_size,
        v_enabled=turboquant_v_enabled,
        block_tokens=turboquant_block_tokens,
    )"""

new_call = """    _cfg = _TQConfig.from_legacy_kwargs(
        main_bits=turboquant_main_bits,
        group_size=turboquant_group_size,
        rotation_mode=turboquant_rotation,
        residual_topk=turboquant_residual_topk,
        v_bits=turboquant_v_bits,
        v_group_size=turboquant_v_group_size,
        v_enabled=turboquant_v_enabled,
        block_tokens=turboquant_block_tokens,
    )"""

new_text = text.replace(old_call, new_call)
if new_text != text:
    with open('mlx_lm/generate.py', 'w') as f:
        f.write(new_text)
    print("Updated mlx_lm/generate.py")
else:
    print("Failed to update mlx_lm/generate.py")
