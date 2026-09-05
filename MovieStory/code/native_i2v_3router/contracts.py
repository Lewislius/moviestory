"""Small, testable contracts for MovieStory's native Wan I2V-A14B path.

Nothing in this file patches or subclasses ``WanModel``.  In particular, the
20-channel I2V condition is built byte-for-byte from the operations used by
``wan.image2video.WanI2V.generate``: four mask channels followed by the VAE
encoding of ``[first_frame, zeros...]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass(frozen=True)
class ThreeRouterConfig:
    hidden_size: int = 2048
    role_tokens: int = 96
    action_tokens: int = 96
    global_tokens: int = 64

    def __post_init__(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "role_tokens": self.role_tokens,
            "action_tokens": self.action_tokens,
            "global_tokens": self.global_tokens,
        }
        invalid = {name: value for name, value in values.items() if value <= 0}
        if invalid:
            raise ValueError(f"three-router dimensions must be positive: {invalid}")

    @property
    def total_tokens(self) -> int:
        return self.role_tokens + self.action_tokens + self.global_tokens

    @property
    def route_slices(self) -> Dict[str, Tuple[int, int]]:
        role_end = self.role_tokens
        action_end = role_end + self.action_tokens
        return {
            "role": (0, role_end),
            "action": (role_end, action_end),
            "global": (action_end, self.total_tokens),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "role_tokens": self.role_tokens,
            "action_tokens": self.action_tokens,
            "global_tokens": self.global_tokens,
            "total_tokens": self.total_tokens,
            "route_slices": self.route_slices,
        }


@dataclass
class ThreeRouterOutput:
    tokens: torch.Tensor
    role: torch.Tensor
    action: torch.Tensor
    global_route: torch.Tensor

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        def pooled(value: torch.Tensor) -> torch.Tensor:
            return F.normalize(value.float().mean(dim=1), dim=-1)

        role = pooled(self.role)
        action = pooled(self.action)
        global_route = pooled(self.global_route)
        return {
            "role_action_cosine": (role * action).sum(dim=-1),
            "role_global_cosine": (role * global_route).sum(dim=-1),
            "action_global_cosine": (action * global_route).sum(dim=-1),
            "role_rms": self.role.float().square().mean(dim=(1, 2)).sqrt(),
            "action_rms": self.action.float().square().mean(dim=(1, 2)).sqrt(),
            "global_rms": self.global_route.float().square().mean(dim=(1, 2)).sqrt(),
        }


class ThreeRouterPlanner(nn.Module):
    """Parameter-free identity split of ordered Qwen MetaQuery states."""

    def __init__(self, config: ThreeRouterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ThreeRouterConfig()

    def forward(self, tokens: torch.Tensor) -> ThreeRouterOutput:
        expected = (self.config.total_tokens, self.config.hidden_size)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"router expects [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(tokens.shape)}"
            )
        slices = self.config.route_slices
        role = tokens[:, slices["role"][0] : slices["role"][1]]
        action = tokens[:, slices["action"][0] : slices["action"][1]]
        global_route = tokens[:, slices["global"][0] : slices["global"][1]]
        return ThreeRouterOutput(tokens, role, action, global_route)


@dataclass
class NativeI2VCondition:
    y: torch.Tensor
    mask: torch.Tensor
    image_latent: torch.Tensor
    conditioning_video: torch.Tensor


def _as_rgb_pil(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"I2V reference must be PIL.Image, got {type(image).__name__}")
    return image.convert("RGB")


@torch.no_grad()
def build_native_i2v_condition(
    *,
    vae,
    first_frame: Image.Image,
    frame_num: int,
    latent_shape: Sequence[int],
    vae_stride: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> NativeI2VCondition:
    """Build Wan I2V's native ``y`` without adding any new injection path.

    ``latent_shape`` is the target video's ``[C,T,H,W]`` shape.  Training uses
    that spatial size so ``y`` and the noised target are exactly compatible.
    """

    if frame_num <= 0 or (frame_num - 1) % int(vae_stride[0]) != 0:
        raise ValueError(
            f"frame_num must satisfy 4n+1 for stride {vae_stride[0]}, got {frame_num}"
        )
    if len(latent_shape) != 4:
        raise ValueError(f"latent_shape must be [C,T,H,W], got {tuple(latent_shape)}")
    z_channels, latent_frames, latent_h, latent_w = map(int, latent_shape)
    expected_latent_frames = (frame_num - 1) // int(vae_stride[0]) + 1
    if latent_frames != expected_latent_frames:
        raise ValueError(
            f"target latent has T={latent_frames}; expected {expected_latent_frames} "
            f"from frame_num={frame_num}"
        )

    target_h = latent_h * int(vae_stride[1])
    target_w = latent_w * int(vae_stride[2])
    image = TF.to_tensor(_as_rgb_pil(first_frame)).sub_(0.5).div_(0.5).to(device)

    # This is deliberately the same interpolation/transposition sequence as
    # WanI2V.generate, including CPU interpolation and the default align policy.
    first = F.interpolate(
        image[None].cpu(),
        size=(target_h, target_w),
        mode="bicubic",
    ).transpose(0, 1)
    conditioning_video = torch.concat(
        [first, torch.zeros(3, frame_num - 1, target_h, target_w)],
        dim=1,
    ).to(device=device)
    # Wan2_1_VAE defaults to FP32 and native WanI2V.generate passes FP32 here.
    # Keep ``dtype`` only as an explicit assertion guard for callers that audit
    # the contract; silently converting this input would diverge from native I2V.
    if dtype != torch.float32:
        raise ValueError("native Wan I2V conditioning_video must remain float32")
    image_latent = vae.encode([conditioning_video])[0]
    if tuple(image_latent.shape) != (z_channels, latent_frames, latent_h, latent_w):
        raise RuntimeError(
            "native I2V VAE condition shape mismatch: "
            f"expected {(z_channels, latent_frames, latent_h, latent_w)}, "
            f"got {tuple(image_latent.shape)}"
        )

    mask = torch.ones(1, frame_num, latent_h, latent_w, device=device)
    mask[:, 1:] = 0
    mask = torch.concat(
        [torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]],
        dim=1,
    )
    mask = mask.view(1, mask.shape[1] // 4, 4, latent_h, latent_w)
    mask = mask.transpose(1, 2)[0]
    y = torch.concat([mask.to(dtype=image_latent.dtype), image_latent], dim=0)
    if y.shape[0] != z_channels + 4:
        raise RuntimeError(f"native I2V y must have {z_channels + 4} channels, got {y.shape[0]}")
    return NativeI2VCondition(y, mask, image_latent, conditioning_video)


def native_flow_matching_pair(
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    normalized_timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wan training pair: ``x_t=(1-t)x0+t*eps``, target ``eps-x0``."""

    if clean_latent.shape != noise.shape:
        raise ValueError(
            f"clean/noise shape mismatch: {tuple(clean_latent.shape)} != {tuple(noise.shape)}"
        )
    if normalized_timestep.numel() != 1:
        raise ValueError("one scalar timestep is required per native I2V sample")
    t = normalized_timestep.float().reshape(1, 1, 1, 1)
    clean = clean_latent.float()
    eps = noise.float()
    return (1.0 - t) * clean + t * eps, eps - clean
