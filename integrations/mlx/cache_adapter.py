import warnings
from dataclasses import dataclass
from typing import Optional, Any
import mlx.core as mx

from mlx_lm.models.cache import _BaseCache, create_attention_mask
from turboquant.config import TurboQuantConfig as _ProdTurboQuantConfig
from turboquant.runtime.kv_interface import TurboQuantKVCache


@dataclass
class TurboQuantConfig:
    k_bits: int = 3
    group_size: int = 32
    rotation_mode: str = "hadamard"
    rotation_pad_to_pow2: bool = True
    residual_mode: str = "qjl"
    residual_topk: int = 0
    resid_scale_bits: int = 8
    qjl_proj_dim: int = 64
    qjl_seed: int = 42
    qjl_bits: int = 1
    return_mode: str = "view"
    block_tokens: int = 256
    v_bits: int = 4
    v_group_size: int = 64
    v_scale_dtype: str = "float16"
    v_enabled: bool = True


def _to_prod_config(cfg: TurboQuantConfig) -> _ProdTurboQuantConfig:
    return _ProdTurboQuantConfig.from_legacy_kwargs(
        k_bits=cfg.k_bits,
        group_size=cfg.group_size,
        rotation_mode=cfg.rotation_mode,
        rotation_pad_to_pow2=cfg.rotation_pad_to_pow2,
        residual_mode=cfg.residual_mode,
        residual_topk=cfg.residual_topk,
        resid_scale_bits=cfg.resid_scale_bits,
        qjl_proj_dim=cfg.qjl_proj_dim,
        qjl_seed=cfg.qjl_seed,
        qjl_bits=cfg.qjl_bits
    )

def dummy_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.group_size), dtype=mx.float32)

def dummy_dequantize_main(packed, scales, *, config):
    return packed


class TurboQuantKCache(_BaseCache):
    step = 512

    def __init__(self, config: Optional[TurboQuantConfig] = None) -> None:
        self.config = config or TurboQuantConfig()
        self._impl = TurboQuantKVCache(
            config=_to_prod_config(self.config),
            quantize_main=dummy_quantize_main,
            dequantize_main=dummy_dequantize_main
        )
        self.v_cache = []
        self._offset = 0

    def size(self) -> int:
        return self._offset

    def __len__(self) -> int:
        return self._offset

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, v: int) -> None:
        self._offset = v

    def update_and_fetch(self, keys, values):
        block = self._impl.append_keys(keys)
        self.v_cache.append(values)
        self._offset += keys.shape[2]
        
        from turboquant.runtime.kv_interface import TurboQuantKeysView
        return TurboQuantKeysView(self, self._offset - keys.shape[-2], self._offset), values


    @property
    def state(self):
        return (self._impl.state(), self.v_cache)

    def byte_size(self):
        return self._impl.byte_size() + sum(v.nbytes for v in self.v_cache)

    @state.setter
    def state(self, v):
        k_state, v_cache = v
        self._impl = TurboQuantKVCache.from_state(
            k_state, quantize_main=dummy_quantize_main, dequantize_main=dummy_dequantize_main
        )
        self.v_cache = v_cache

    @property
    def meta_state(self):
        return (
            str(self._offset),
            str(self.config.k_bits),
            str(self.config.group_size),
            str(self.config.rotation_mode),
            str(self.config.residual_topk),
            str(self.config.resid_scale_bits),
            str(self.config.return_mode),
            str(self.config.block_tokens),
            str(self.config.v_bits),
            str(self.config.v_group_size),
            str(self.config.v_scale_dtype),
            str(self.config.v_enabled),
            str(self.config.rotation_pad_to_pow2),
            str(self.config.residual_mode),
            str(self.config.qjl_proj_dim),
            str(self.config.qjl_seed),
            str(self.config.qjl_bits),
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) >= 17:
            (
                offset,
                mb,
                gs,
                rot,
                rt,
                rs,
                rm,
                bt,
                vb,
                vg,
                vsd,
                ve,
                rpp,
                r_mode,
                qpd,
                qs,
                qb,
            ) = v[:17]
            self.config = TurboQuantConfig(
                k_bits=int(mb),
                group_size=int(gs),
                rotation_mode=rot,
                residual_topk=int(rt),
                resid_scale_bits=int(rs),
                return_mode=rm,
                block_tokens=int(bt),
                v_bits=int(vb),
                v_group_size=int(vg),
                v_scale_dtype=vsd,
                v_enabled=(ve == "True"),
                rotation_pad_to_pow2=(rpp == "True"),
                residual_mode=r_mode,
                qjl_proj_dim=int(qpd),
                qjl_seed=int(qs),
                qjl_bits=int(qb),
            )
            self._offset = int(offset)
        elif len(v) == 12:
            (
                offset,
                mb,
                gs,
                rot,
                rt,
                rs,
                rm,
                bt,
                vb,
                vg,
                vsd,
                ve,
            ) = v
            self.config = TurboQuantConfig(
                k_bits=int(mb),
                group_size=int(gs),
                rotation_mode=rot,
                residual_topk=int(rt),
                resid_scale_bits=int(rs),
                return_mode=rm,
                block_tokens=int(bt),
                v_bits=int(vb),
                v_group_size=int(vg),
                v_scale_dtype=vsd,
                v_enabled=(ve == "True"),
            )
            self._offset = int(offset)
        else:
            self._offset = int(v[0]) if v else 0
