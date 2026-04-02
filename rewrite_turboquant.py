import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

write_file('turboquant/config.py', """from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


RotationMode = Literal["hadamard"]
ResidualMode = Literal["none", "topk", "qjl"]


@dataclass(slots=True)
class TurboQuantConfig:
    # First-stage quantizer
    k_bits: int = 3
    group_size: int = 32

    # Rotation
    rotation_mode: RotationMode = "hadamard"
    rotation_pad_to_pow2: bool = True

    # Second-stage residual codec
    residual_mode: ResidualMode = "qjl"

    # Legacy sparse-residual fields
    residual_topk: int = 0
    resid_scale_bits: int = 8

    # QJL fields
    qjl_proj_dim: int = 64
    qjl_seed: int = 42
    qjl_bits: int = 1

    # Runtime / migration flags
    paper_faithful_mode: bool = False
    return_mode: str = "reconstruct"

    def validate(self) -> None:
        if self.k_bits <= 0 or self.k_bits > 8:
            raise ValueError(f"k_bits must be in [1, 8], got {self.k_bits}")

        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")

        if self.rotation_mode != "hadamard":
            raise ValueError(f"Unsupported rotation_mode: {self.rotation_mode}")

        if self.residual_mode not in {"none", "topk", "qjl"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        if self.residual_mode == "topk" and self.residual_topk <= 0:
            raise ValueError(
                "residual_topk must be > 0 when residual_mode='topk'"
            )

        if self.residual_mode == "qjl":
            if self.qjl_bits != 1:
                raise ValueError(
                    f"Only 1-bit QJL is currently supported, got {self.qjl_bits}"
                )
            if self.qjl_proj_dim <= 0:
                raise ValueError(
                    f"qjl_proj_dim must be > 0, got {self.qjl_proj_dim}"
                )

    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> "TurboQuantConfig":
        \"\"\"
        Thin migration shim for older callers.
        \"\"\"
        cfg = cls(
            k_bits=kwargs.get("k_bits", 3),
            group_size=kwargs.get("group_size", 32),
            return_mode=kwargs.get("return_mode", "reconstruct"),
            residual_topk=kwargs.get("residual_topk", kwargs.get("residual", 0)),
            resid_scale_bits=kwargs.get("resid_scale_bits", 8),
        )

        if "rotation_pad_to_pow2" in kwargs:
            cfg.rotation_pad_to_pow2 = bool(kwargs["rotation_pad_to_pow2"])

        if "residual_mode" in kwargs:
            cfg.residual_mode = kwargs["residual_mode"]
        else:
            # Preserve old behavior, but only as a compatibility default.
            cfg.residual_mode = "topk" if cfg.residual_topk > 0 else "none"

        if "qjl_proj_dim" in kwargs:
            cfg.qjl_proj_dim = int(kwargs["qjl_proj_dim"])
        if "qjl_seed" in kwargs:
            cfg.qjl_seed = int(kwargs["qjl_seed"])
        if "qjl_bits" in kwargs:
            cfg.qjl_bits = int(kwargs["qjl_bits"])
        if "paper_faithful_mode" in kwargs:
            cfg.paper_faithful_mode = bool(kwargs["paper_faithful_mode"])

        cfg.validate()
        return cfg
""")

write_file('turboquant/core/residual_codec.py', """from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import mlx.core as mx

from turboquant.config import TurboQuantConfig

from .residual import encode_topk_residual, decode_topk_residual
from .qjl import QJLProjector


ResidualMode = Literal["none", "topk", "qjl"]


@dataclass(slots=True)
class ResidualPayload:
    mode: ResidualMode
    data: dict[str, Any]


@runtime_checkable
class ResidualCodec(Protocol):
    mode: ResidualMode

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        ...

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array | None:
        ...

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        raise NotImplementedError


class NoResidualCodec:
    mode: ResidualMode = "none"

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        return ResidualPayload(mode="none", data={})

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array | None:
        return None

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        batch_shape = q.shape[:-1]
        return mx.zeros(batch_shape, dtype=q.dtype)


class TopKResidualCodec:
    mode: ResidualMode = "topk"

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        vals, idx = encode_topk_residual(
            residual,
            k=config.residual_topk,
            scale_bits=config.resid_scale_bits,
        )
        return ResidualPayload(
            mode="topk",
            data={
                "vals": vals,
                "idx": idx,
                "shape": tuple(residual.shape),
            },
        )

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return decode_topk_residual(
            payload.data["vals"],
            payload.data["idx"],
            shape=payload.data["shape"],
        )

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        resid = self.decode(payload, config=config)
        return (q * resid).sum(axis=-1)


class QJLResidualCodec:
    mode: ResidualMode = "qjl"

    def __init__(self, *, proj_dim: int, seed: int):
        self.projector = QJLProjector(proj_dim=proj_dim, seed=seed)

    def encode(
        self,
        residual: mx.array,
        *,
        config: TurboQuantConfig,
    ) -> ResidualPayload:
        bits, norms, meta = self.projector.encode(residual)
        return ResidualPayload(
            mode="qjl",
            data={
                "bits": bits,
                "norms": norms,
                "meta": meta.to_dict(),
            },
        )

    def decode(
        self,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return self.projector.decode(
            payload.data["bits"],
            payload.data["norms"],
            payload.data["meta"],
        )

    def dot_estimate(
        self,
        q: mx.array,
        payload: ResidualPayload,
        *,
        config: TurboQuantConfig,
    ) -> mx.array:
        return self.projector.dot_estimate(
            q,
            payload.data["bits"],
            payload.data["norms"],
            payload.data["meta"],
        )


def build_residual_codec(config: TurboQuantConfig) -> ResidualCodec:
    config.validate()

    if config.residual_mode == "none":
        return NoResidualCodec()

    if config.residual_mode == "topk":
        return TopKResidualCodec()

    if config.residual_mode == "qjl":
        return QJLResidualCodec(
            proj_dim=config.qjl_proj_dim,
            seed=config.qjl_seed,
        )

    raise ValueError(f"Unknown residual_mode: {config.residual_mode}")
""")

write_file('turboquant/core/qjl.py', """from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx


@dataclass(slots=True)
class QJLMeta:
    input_dim: int
    proj_dim: int
    seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QJLMeta":
        return cls(
            input_dim=int(data["input_dim"]),
            proj_dim=int(data["proj_dim"]),
            seed=int(data["seed"]),
        )


def _ensure_float(x: mx.array) -> mx.array:
    if x.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return x
    return x.astype(mx.float32)


def pack_sign_bits(signs: mx.array) -> mx.array:
    \"\"\"
    Minimal placeholder packer.

    Stores 0/1 as uint8 for correctness-first wiring.
    Replace with real bit packing later.
    \"\"\"
    return signs.astype(mx.uint8)


def unpack_sign_bits(bits: mx.array) -> mx.array:
    \"\"\"
    Map uint8 {0,1} -> float {-1,+1}
    \"\"\"
    return bits.astype(mx.float32) * 2.0 - 1.0


class QJLProjector:
    def __init__(self, *, proj_dim: int, seed: int):
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {proj_dim}")
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self._proj_cache: dict[tuple[int, int, int], mx.array] = {}

    def _projection(self, input_dim: int) -> mx.array:
        key = (int(input_dim), self.proj_dim, self.seed)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached

        rng = mx.random.key(self.seed)
        proj = mx.random.normal(
            shape=(input_dim, self.proj_dim),
            key=rng,
        ).astype(mx.float32)

        proj = proj / mx.sqrt(mx.array(float(self.proj_dim), dtype=mx.float32))
        self._proj_cache[key] = proj
        return proj

    def encode(self, residual: mx.array) -> tuple[mx.array, mx.array, QJLMeta]:
        residual = _ensure_float(residual)
        input_dim = int(residual.shape[-1])

        proj = self._projection(input_dim)
        sketch = residual @ proj

        norms = mx.linalg.norm(residual, axis=-1, keepdims=True)
        bits = pack_sign_bits(sketch >= 0)

        meta = QJLMeta(
            input_dim=input_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
        )
        return bits, norms, meta

    def decode(
        self,
        bits: mx.array,
        norms: mx.array,
        meta: QJLMeta | dict[str, Any],
    ) -> mx.array:
        \"\"\"
        Crude proxy reconstruction for debug/fallback paths.

        This is not the final paper-grade residual recovery procedure.
        It only gives the runtime something shaped correctly while
        the direct dot-estimator path is integrated.
        \"\"\"
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        proj = self._projection(meta.input_dim)
        signed = unpack_sign_bits(bits)

        proxy = signed @ proj.T
        proxy_norm = mx.linalg.norm(proxy, axis=-1, keepdims=True)
        proxy_norm = mx.maximum(
            proxy_norm,
            mx.array(1e-8, dtype=proxy.dtype),
        )
        proxy = proxy * (norms / proxy_norm)
        return proxy

    def dot_estimate(
        self,
        q: mx.array,
        bits: mx.array,
        norms: mx.array,
        meta: QJLMeta | dict[str, Any],
    ) -> mx.array:
        \"\"\"
        Estimate q · residual for all query/key pairs.

        q:
            [..., q_len, d]
        bits:
            [..., k_len, proj_dim]
        norms:
            [..., k_len, 1]

        returns:
            [..., q_len, k_len]
        \"\"\"
        if isinstance(meta, dict):
            meta = QJLMeta.from_dict(meta)

        q = _ensure_float(q)
        proj = self._projection(meta.input_dim)

        q_proj = q @ proj                      # [..., q_len, proj_dim]
        signed = unpack_sign_bits(bits)        # [..., k_len, proj_dim]

        scores = q_proj @ mx.swapaxes(signed, -1, -2)   # [..., q_len, k_len]

        norm_scale = norms.squeeze(-1)                  # [..., k_len]
        q_norm = mx.linalg.norm(q, axis=-1)             # [..., q_len]
        q_norm = mx.maximum(q_norm, mx.array(1e-8, dtype=q_norm.dtype))

        return scores * (norm_scale[..., None, :] / q_norm[..., :, None])
""")

write_file('turboquant/core/pipeline.py', """from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from .residual_codec import ResidualPayload, build_residual_codec


@dataclass(slots=True)
class EncodedKeyBlock:
    packed_main: mx.array
    scales: mx.array
    residual: ResidualPayload
    d_head: int
    d_rot: int
    d_quant: int


def pad_last_dim(x: mx.array, multiple: int) -> tuple[mx.array, int]:
    d = int(x.shape[-1])
    d2 = ((d + multiple - 1) // multiple) * multiple
    if d2 == d:
        return x, d2

    pad = d2 - d
    zeros = mx.zeros((*x.shape[:-1], pad), dtype=x.dtype)
    return mx.concatenate([x, zeros], axis=-1), d2


def encode_k_block(
    k_rot: mx.array,
    *,
    config: TurboQuantConfig,
    quantize_main,
    dequantize_main,
) -> EncodedKeyBlock:
    \"\"\"
    Transitional version:
    expects already-rotated K until rotation.py is patched.
    \"\"\"
    config.validate()

    d_head = int(k_rot.shape[-1])
    d_rot = d_head

    k_quant_in, d_quant = pad_last_dim(k_rot, config.group_size)

    packed_main, scales = quantize_main(k_quant_in, config=config)
    main_hat = dequantize_main(packed_main, scales, config=config)

    residual = k_quant_in - main_hat
    codec = build_residual_codec(config)
    residual_payload = codec.encode(residual, config=config)

    return EncodedKeyBlock(
        packed_main=packed_main,
        scales=scales,
        residual=residual_payload,
        d_head=d_head,
        d_rot=d_rot,
        d_quant=d_quant,
    )


def decode_k_block(
    block: EncodedKeyBlock,
    *,
    config: TurboQuantConfig,
    dequantize_main,
) -> mx.array:
    config.validate()

    main_hat = dequantize_main(block.packed_main, block.scales, config=config)

    codec = build_residual_codec(config)
    resid_hat = codec.decode(block.residual, config=config)

    if resid_hat is None:
        k_quant_hat = main_hat
    else:
        k_quant_hat = main_hat + resid_hat

    return k_quant_hat[..., : block.d_rot]
""")

write_file('tests/unit/test_residual_codec.py', """import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.residual_codec import build_residual_codec


def test_codec_dispatch_none():
    cfg = TurboQuantConfig(residual_mode="none")
    codec = build_residual_codec(cfg)
    assert codec.mode == "none"


def test_codec_dispatch_topk():
    cfg = TurboQuantConfig(residual_mode="topk", residual_topk=8)
    codec = build_residual_codec(cfg)
    assert codec.mode == "topk"


def test_codec_dispatch_qjl():
    cfg = TurboQuantConfig(residual_mode="qjl", qjl_proj_dim=32, qjl_seed=7)
    codec = build_residual_codec(cfg)
    assert codec.mode == "qjl"


def test_qjl_roundtrip_shape():
    cfg = TurboQuantConfig(residual_mode="qjl", qjl_proj_dim=32, qjl_seed=7)
    codec = build_residual_codec(cfg)

    x = mx.random.normal(shape=(2, 4, 96), key=mx.random.key(0))
    payload = codec.encode(x, config=cfg)
    xhat = codec.decode(payload, config=cfg)

    assert xhat.shape == x.shape
""")

write_file('tests/unit/test_qjl.py', """import mlx.core as mx

from turboquant.core.qjl import QJLProjector


def test_qjl_encode_shapes():
    qjl = QJLProjector(proj_dim=64, seed=123)
    x = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(0))

    bits, norms, meta = qjl.encode(x)

    assert bits.shape == (3, 5, 64)
    assert norms.shape == (3, 5, 1)
    assert meta.input_dim == 96
    assert meta.proj_dim == 64
    assert meta.seed == 123


def test_qjl_decode_shape():
    qjl = QJLProjector(proj_dim=64, seed=123)
    x = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(0))

    bits, norms, meta = qjl.encode(x)
    xhat = qjl.decode(bits, norms, meta)

    assert xhat.shape == x.shape


def test_qjl_dot_estimate_shape():
    qjl = QJLProjector(proj_dim=64, seed=123)
    q = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(1))
    r = mx.random.normal(shape=(3, 5, 96), key=mx.random.key(2))

    bits, norms, meta = qjl.encode(r)
    est = qjl.dot_estimate(q, bits, norms, meta)

    assert est.shape == (3, 5)
""")

write_file('turboquant/runtime/kv_interface.py', """from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import (
    EncodedKeyBlock,
    decode_k_block,
    encode_k_block,
)


@dataclass(slots=True)
class KVCacheState:
    blocks: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "config": self.config,
        }


class TurboQuantKVCache:
    \"\"\"
    Transitional generic cache.

    Stores one EncodedKeyBlock per appended chunk.
    This removes any assumption that residuals are represented
    as sparse vals/idx tensors.
    \"\"\"

    def __init__(
        self,
        *,
        config: TurboQuantConfig,
        quantize_main,
        dequantize_main,
    ):
        config.validate()
        self.config = config
        self.quantize_main = quantize_main
        self.dequantize_main = dequantize_main
        self._blocks: list[EncodedKeyBlock] = []

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()

    def append_keys(self, k: mx.array) -> EncodedKeyBlock:
        \"\"\"
        Encode and append one key block.

        Expected input shape:
            [..., seq, d_head] or [seq, d_head]
        depending on caller convention.
        \"\"\"
        block = encode_k_block(
            k,
            config=self.config,
            quantize_main=self.quantize_main,
            dequantize_main=self.dequantize_main,
        )
        self._blocks.append(block)
        return block

    def append_encoded_block(self, block: EncodedKeyBlock) -> None:
        self._blocks.append(block)

    def block(self, index: int) -> EncodedKeyBlock:
        return self._blocks[index]

    def iter_blocks(self):
        yield from self._blocks

    def decode_block_full(self, index: int) -> mx.array:
        return decode_k_block(
            self._blocks[index],
            config=self.config,
            dequantize_main=self.dequantize_main,
        )

    def state(self) -> KVCacheState:
        return KVCacheState(
            blocks=[b.to_dict() for b in self._blocks],
            config={
                "k_bits": self.config.k_bits,
                "group_size": self.config.group_size,
                "rotation_mode": self.config.rotation_mode,
                "rotation_pad_to_pow2": self.config.rotation_pad_to_pow2,
                "residual_mode": self.config.residual_mode,
                "residual_topk": self.config.residual_topk,
                "resid_scale_bits": self.config.resid_scale_bits,
                "qjl_proj_dim": self.config.qjl_proj_dim,
                "qjl_seed": self.config.qjl_seed,
                "qjl_bits": self.config.qjl_bits,
                "paper_faithful_mode": self.config.paper_faithful_mode,
                "return_mode": self.config.return_mode,
            },
        )

    @classmethod
    def from_state(
        cls,
        state: KVCacheState | dict[str, Any],
        *,
        quantize_main,
        dequantize_main,
    ) -> "TurboQuantKVCache":
        if isinstance(state, dict):
            raw = state
        else:
            raw = state.to_dict()

        config = TurboQuantConfig(
            k_bits=int(raw["config"]["k_bits"]),
            group_size=int(raw["config"]["group_size"]),
            rotation_mode=raw["config"]["rotation_mode"],
            rotation_pad_to_pow2=bool(raw["config"]["rotation_pad_to_pow2"]),
            residual_mode=raw["config"]["residual_mode"],
            residual_topk=int(raw["config"]["residual_topk"]),
            resid_scale_bits=int(raw["config"]["resid_scale_bits"]),
            qjl_proj_dim=int(raw["config"]["qjl_proj_dim"]),
            qjl_seed=int(raw["config"]["qjl_seed"]),
            qjl_bits=int(raw["config"]["qjl_bits"]),
            paper_faithful_mode=bool(raw["config"]["paper_faithful_mode"]),
            return_mode=raw["config"]["return_mode"],
        )
        config.validate()

        cache = cls(
            config=config,
            quantize_main=quantize_main,
            dequantize_main=dequantize_main,
        )
        cache._blocks = [EncodedKeyBlock.from_dict(b) for b in raw["blocks"]]
        return cache
""")

write_file('turboquant/runtime/attention.py', """from __future__ import annotations

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import EncodedKeyBlock
from turboquant.core.residual_codec import build_residual_codec


def score_block(
    q_rot: mx.array,
    block: EncodedKeyBlock,
    *,
    config: TurboQuantConfig,
    dequantize_main,
) -> mx.array:
    \"\"\"
    Compute attention scores against one encoded key block.

    q_rot:
        [..., q_len, d_rot]

    Returns:
        [..., q_len, k_len]
    \"\"\"
    config.validate()

    main_hat = dequantize_main(block.packed_main, block.scales, config=config)
    main_rot = main_hat[..., : block.d_rot]

    if int(q_rot.shape[-1]) != int(main_rot.shape[-1]):
        raise ValueError(
            f"q_rot dim {int(q_rot.shape[-1])} != main_rot dim {int(main_rot.shape[-1])}"
        )

    main_scores = q_rot @ mx.swapaxes(main_rot, -1, -2)

    codec = build_residual_codec(config)

    if block.residual.mode == "none":
        return main_scores

    if block.residual.mode == "topk":
        resid_hat = codec.decode(block.residual, config=config)
        resid_rot = resid_hat[..., : block.d_rot]
        resid_scores = q_rot @ mx.swapaxes(resid_rot, -1, -2)
        return main_scores + resid_scores

    if block.residual.mode == "qjl":
        \"\"\"
        QJL path:
        dot_estimate() is expected to return residual score contribution
        in [..., q_len, k_len] form.
        \"\"\"
        resid_scores = codec.dot_estimate(q_rot, block.residual, config=config)
        if tuple(resid_scores.shape) != tuple(main_scores.shape):
            raise ValueError(
                f"QJL residual score shape {tuple(resid_scores.shape)} "
                f"!= main score shape {tuple(main_scores.shape)}"
            )
        return main_scores + resid_scores

    raise ValueError(f"Unsupported residual mode: {block.residual.mode}")

def streaming_scores(
    q_rot: mx.array,
    *,
    cache,
    config: TurboQuantConfig,
    dequantize_main,
) -> list[mx.array]:
    \"\"\"
    Produce per-block score tensors.
    \"\"\"
    out: list[mx.array] = []
    for block in cache.iter_blocks():
        scores = score_block(
            q_rot,
            block,
            config=config,
            dequantize_main=dequantize_main,
        )
        out.append(scores)
    return out
""")

write_file('tests/unit/test_kv_interface.py', """import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import TurboQuantKVCache


def fake_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.group_size), dtype=mx.float32)


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_cache_stores_generic_blocks_qjl():
    cfg = TurboQuantConfig(
        k_bits=3,
        group_size=32,
        residual_mode="qjl",
        qjl_proj_dim=64,
        qjl_seed=7,
        rotation_pad_to_pow2=True,
    )

    cache = TurboQuantKVCache(
        config=cfg,
        quantize_main=fake_quantize_main,
        dequantize_main=fake_dequantize_main,
    )

    k = mx.random.normal(shape=(2, 5, 96), key=mx.random.key(0))
    block = cache.append_keys(k)

    assert cache.num_blocks == 1
    assert block.residual.mode == "qjl"
    assert "bits" in block.residual.data
    assert "norms" in block.residual.data
""")

write_file('tests/unit/test_attention_score_block.py', """import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import encode_k_block
from turboquant.runtime.attention import score_block


def fake_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.group_size), dtype=mx.float32)


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_score_block_none_mode_shape():
    cfg = TurboQuantConfig(
        k_bits=3,
        group_size=32,
        residual_mode="none",
        rotation_pad_to_pow2=True,
    )

    k = mx.random.normal(shape=(1, 6, 96), key=mx.random.key(0))
    block = encode_k_block(
        k,
        config=cfg,
        quantize_main=fake_quantize_main,
        dequantize_main=fake_dequantize_main,
    )

    q_rot = mx.random.normal(shape=(1, 4, block.d_rot), key=mx.random.key(1))
    scores = score_block(
        q_rot,
        block,
        config=cfg,
        dequantize_main=fake_dequantize_main,
    )

    assert scores.shape == (1, 4, 6)
""")

write_file('tests/unit/test_attention_score_block_qjl.py', """import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import encode_k_block
from turboquant.runtime.attention import score_block


def fake_quantize_main(x, *, config):
    return x, mx.ones((*x.shape[:-1], x.shape[-1] // config.group_size), dtype=mx.float32)


def fake_dequantize_main(packed, scales, *, config):
    return packed


def test_score_block_qjl_shape():
    cfg = TurboQuantConfig(
        k_bits=3,
        group_size=32,
        residual_mode="qjl",
        qjl_proj_dim=64,
        qjl_seed=7,
        rotation_pad_to_pow2=True,
    )

    k = mx.random.normal(shape=(2, 5, 96), key=mx.random.key(0))
    block = encode_k_block(
        k,
        config=cfg,
        quantize_main=fake_quantize_main,
        dequantize_main=fake_dequantize_main,
    )

    q_rot = mx.random.normal(shape=(2, 3, block.d_rot), key=mx.random.key(1))
    scores = score_block(
        q_rot,
        block,
        config=cfg,
        dequantize_main=fake_dequantize_main,
    )

    assert scores.shape == (2, 3, 5)
""")

