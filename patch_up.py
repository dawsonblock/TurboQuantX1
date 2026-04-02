import sys

with open('integrations/mlx/upgrade.py', 'r') as f:
    content = f.read()

old_code = """        legacy_cfg = TurboQuantConfig(
            k_bits=config.k_bits,  # type: ignore
            group_size=config.k_group_size,  # type: ignore
            rotation=config.rotation,  # type: ignore
            return_mode="view",  # type: ignore
            scale_dtype=config.scale_dtype,  # type: ignore
            resid_scale_bits=8,  # legacy fallback  # type: ignore
            residual_topk=config.residual_topk,  # type: ignore
            v_bits=config.v_bits,  # type: ignore
            v_group_size=config.v_group_size,  # type: ignore
            v_scale_dtype=config.v_scale_dtype,  # type: ignore
            v_enabled=config.v_enabled,  # type: ignore
            block_tokens=config.block_tokens,  # type: ignore
        )"""

new_code = """        legacy_cfg = TurboQuantConfig(
            k_bits=getattr(config, 'k_bits', 3),
            group_size=getattr(config, 'k_group_size', 32),
            rotation_mode=getattr(config, 'rotation_mode', getattr(config, 'rotation', 'hadamard')),
            rotation_pad_to_pow2=getattr(config, 'rotation_pad_to_pow2', True),
            residual_mode=getattr(config, 'residual_mode', 'qjl' if getattr(config, 'residual_topk', 0) == 0 else 'topk'),
            residual_topk=getattr(config, 'residual_topk', 0),
            resid_scale_bits=8,
            qjl_proj_dim=getattr(config, 'qjl_proj_dim', 64),
            qjl_seed=getattr(config, 'qjl_seed', 42),
            qjl_bits=getattr(config, 'qjl_bits', 1),
            return_mode="view",
            v_bits=getattr(config, 'v_bits', 4),
            v_group_size=getattr(config, 'v_group_size', 64),
            v_scale_dtype=getattr(config, 'v_scale_dtype', "float16"),
            v_enabled=getattr(config, 'v_enabled', True),
            block_tokens=getattr(config, 'block_tokens', 256),
        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('integrations/mlx/upgrade.py', 'w') as f:
        f.write(content)
    print("Patched upgrade.py successfully.")
else:
    print("WARNING: Could not find old_code to replace in upgrade.py")
