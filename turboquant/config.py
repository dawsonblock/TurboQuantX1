from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


RotationMode = Literal["hadamard"]
ResidualMode = Literal["none", "topk", "qjl"]


@dataclass(slots=True)
class TurboQuantConfig:
    # First-stage quantizer
    main_bits: int = 3
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
        if self.main_bits <= 0 or self.main_bits > 8:
            raise ValueError(f"main_bits must be in [1, 8], got {self.main_bits}")

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
        """
        Thin migration shim for older callers.
        """
        cfg = cls(
            main_bits=kwargs.get("main_bits", 3),
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
