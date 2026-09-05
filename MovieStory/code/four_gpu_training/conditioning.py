from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Type

import torch
import torch.nn as nn


class MQToT5Mapper(nn.Module):
    """Residual bottleneck mapping from MQ features to the Wan T5 manifold.

    MQ and UMT5 happen to have the same output width (4096), but equal width is
    not evidence of a shared representation space.  This mapper is initialized
    close to the identity so mode 1 starts from the already useful MQ features,
    while the denoising objective can learn the required semantic correction.
    Parameters remain FP32 master weights; the output is restored to the input
    dtype for Wan's BF16 cross attention.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        bottleneck_size: int = 1024,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("mapper dimensions must be positive")
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")

        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.input_norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.down = nn.Linear(self.hidden_size, self.bottleneck_size)
        self.activation = nn.SiLU()
        self.up = nn.Linear(self.bottleneck_size, self.hidden_size)
        self.residual_logit = nn.Parameter(
            torch.tensor(float(math.atanh(min(residual_scale, 0.999))))
        )

        # A near-identity start avoids destroying an already trained Connector.
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or int(features.shape[-1]) != self.hidden_size:
            raise ValueError(
                "MQ mapper expects [B, N, "
                f"{self.hidden_size}], got {tuple(features.shape)}"
            )
        output_dtype = features.dtype
        compute_dtype = self.down.weight.dtype
        x = features.to(dtype=compute_dtype)
        delta = self.up(self.activation(self.down(self.input_norm(x))))
        mapped = x + self.residual_logit.tanh() * delta
        return mapped.to(dtype=output_dtype)


def _pad_feature_sequences(
    sequences: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not sequences:
        raise ValueError("T5 returned no feature sequences")
    width = int(sequences[0].shape[-1])
    max_tokens = max(int(sequence.shape[0]) for sequence in sequences)
    padded = []
    for index, sequence in enumerate(sequences):
        if sequence.ndim != 2 or int(sequence.shape[-1]) != width:
            raise ValueError(
                f"T5 sequence {index} has invalid shape {tuple(sequence.shape)}"
            )
        missing = max_tokens - int(sequence.shape[0])
        if missing > 0:
            sequence = torch.cat(
                [sequence, sequence.new_zeros((missing, width))], dim=0
            )
        padded.append(sequence)
    return torch.stack(padded, dim=0)


def build_dual_mode_encoder_class(
    base_encoder_class: Type[nn.Module],
    *,
    conditioning_mode: int,
    mapper_bottleneck_size: int = 1024,
    mapper_residual_scale: float = 0.1,
    mapper_rms_match: bool = True,
) -> Type[nn.Module]:
    """Add the requested mode-0/mode-1 context contract to an MQ encoder.

    Mode 0 returns MQ only.  Mode 1 maps MQ to T5 space and concatenates the
    prompt's frozen T5 tokens in ``[mapped MQ, T5]`` order.
    """

    mode = int(conditioning_mode)
    if mode not in (0, 1):
        raise ValueError(f"conditioning_mode must be 0 or 1, got {mode}")

    class DualModeMetaQueryEncoder(base_encoder_class):  # type: ignore[misc, valid-type]
        moviestory_conditioning_mode = mode

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mq_to_t5_mapper: Optional[MQToT5Mapper]
            if mode == 1:
                self.mq_to_t5_mapper = MQToT5Mapper(
                    hidden_size=int(self.wan_text_dim),
                    bottleneck_size=int(mapper_bottleneck_size),
                    residual_scale=float(mapper_residual_scale),
                ).to(device=self.device, dtype=torch.float32)
            else:
                self.mq_to_t5_mapper = None
            # Keep the bound trainer method out of nn.Module registration and
            # checkpoint serialization.
            object.__setattr__(self, "_moviestory_t5_provider", None)
            self.mapper_rms_match = bool(mapper_rms_match)
            self.last_context_audit: dict[str, object] = {}

        def bind_t5_provider(
            self,
            provider: Callable[[Sequence[str]], list[torch.Tensor]],
        ) -> None:
            object.__setattr__(self, "_moviestory_t5_provider", provider)

        @staticmethod
        def _rms(value: torch.Tensor) -> torch.Tensor:
            return value.float().square().mean().clamp_min(1e-12).sqrt()

        def _compose_mode_one(
            self,
            captions: Any,
            mq_features: torch.Tensor,
        ) -> torch.Tensor:
            provider = getattr(self, "_moviestory_t5_provider", None)
            if not callable(provider):
                raise RuntimeError("mode 1 requires a bound frozen-T5 provider")
            if self.mq_to_t5_mapper is None:
                raise RuntimeError("mode 1 mapper was not initialized")

            mapped = self.mq_to_t5_mapper(mq_features)
            caption_rows = [captions] if isinstance(captions, str) else list(captions)
            t5_rows = provider(caption_rows)
            t5 = _pad_feature_sequences(t5_rows).to(
                device=mapped.device,
                dtype=mapped.dtype,
            )
            if int(t5.shape[0]) != int(mapped.shape[0]):
                raise RuntimeError(
                    "MQ/T5 batch mismatch: "
                    f"{int(mapped.shape[0])} != {int(t5.shape[0])}"
                )
            if int(t5.shape[-1]) != int(mapped.shape[-1]):
                raise RuntimeError(
                    "MQ/T5 hidden mismatch: "
                    f"{int(mapped.shape[-1])} != {int(t5.shape[-1])}"
                )

            match_scale = mapped.new_tensor(1.0)
            if self.mapper_rms_match:
                # Scale only MQ.  Raw T5 tokens must remain bit-for-bit the
                # frozen encoder output supplied to Wan.
                with torch.no_grad():
                    match_scale = (
                        self._rms(t5) / self._rms(mapped)
                    ).clamp(0.25, 4.0).to(mapped.dtype)
                mapped = mapped * match_scale

            context = torch.cat([mapped, t5], dim=1)
            self.last_context_audit = {
                "mode": 1,
                "mq_tokens": int(mapped.shape[1]),
                "t5_tokens": int(t5.shape[1]),
                "context_tokens": int(context.shape[1]),
                "hidden_size": int(context.shape[2]),
                "mq_rms": float(self._rms(mapped).detach().item()),
                "t5_rms": float(self._rms(t5).detach().item()),
                "mq_rms_match_scale": float(match_scale.detach().float().item()),
                "mapper_trainable": True,
            }
            return context

        def forward(self, captions: Any, input_images: Any = None) -> torch.Tensor:
            mq_features = super().forward(captions, input_images)
            if mode == 0:
                self.last_context_audit = {
                    "mode": 0,
                    "mq_tokens": int(mq_features.shape[1]),
                    "t5_tokens": 0,
                    "context_tokens": int(mq_features.shape[1]),
                    "hidden_size": int(mq_features.shape[2]),
                    "mapper_trainable": False,
                }
                return mq_features
            return self._compose_mode_one(captions, mq_features)

        def get_conditioning_metadata(self) -> dict[str, object]:
            return {
                "conditioning_mode": mode,
                "context_order": "mq_only" if mode == 0 else "mapped_mq_then_t5",
                "mapper": None
                if self.mq_to_t5_mapper is None
                else {
                    "type": "residual_bottleneck_mq_to_t5",
                    "hidden_size": self.mq_to_t5_mapper.hidden_size,
                    "bottleneck_size": self.mq_to_t5_mapper.bottleneck_size,
                    "fp32_master_weights": True,
                    "rms_match_to_frozen_t5": self.mapper_rms_match,
                },
            }

    DualModeMetaQueryEncoder.__name__ = "DualModeMetaQueryEncoderForWan"
    DualModeMetaQueryEncoder.__qualname__ = "DualModeMetaQueryEncoderForWan"
    return DualModeMetaQueryEncoder
