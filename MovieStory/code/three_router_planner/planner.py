from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ThreeRouterConfig


@dataclass
class ThreeRouterOutput:
    tokens: torch.Tensor
    role: torch.Tensor
    action: torch.Tensor
    global_route: torch.Tensor

    def pooled(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        values = {
            "role": self.role.mean(dim=1),
            "action": self.action.mean(dim=1),
            "global": self.global_route.mean(dim=1),
        }
        if normalize:
            values = {key: F.normalize(value.float(), dim=-1) for key, value in values.items()}
        return values

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        pooled = self.pooled(normalize=True)
        return {
            "role_action_cosine": (pooled["role"] * pooled["action"]).sum(dim=-1),
            "role_global_cosine": (pooled["role"] * pooled["global"]).sum(dim=-1),
            "action_global_cosine": (pooled["action"] * pooled["global"]).sum(dim=-1),
            "role_rms": self.role.float().square().mean(dim=(1, 2)).sqrt(),
            "action_rms": self.action.float().square().mean(dim=(1, 2)).sqrt(),
            "global_rms": self.global_route.float().square().mean(dim=(1, 2)).sqrt(),
        }


class ThreeRouterPlanner(nn.Module):
    """
    Validate and expose ordered Qwen/MetaQuery states as role/action/global routes.

    The planner is intentionally parameter-free: the three trainable MetaQuery
    ParameterDict entries are injected before their isolated Qwen forwards.  Their
    resulting hidden states must reach the one shared Connector unchanged.
    """

    def __init__(self, config: Optional[ThreeRouterConfig] = None) -> None:
        super().__init__()
        self.config = config or ThreeRouterConfig()

    def _validate_seed(self, seed_tokens: torch.Tensor) -> None:
        if seed_tokens.ndim != 3:
            raise ValueError(
                "seed_tokens must have shape "
                f"[batch, {self.config.total_tokens}, hidden], "
                f"got {tuple(seed_tokens.shape)}"
            )
        expected = (self.config.total_tokens, self.config.hidden_size)
        actual = tuple(seed_tokens.shape[1:])
        if actual != expected:
            raise ValueError(
                f"seed token shape mismatch: expected [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(seed_tokens.shape)}"
            )

    def split(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._validate_seed(tokens)
        slices = self.config.route_slices
        return {
            name: tokens[:, start:end]
            for name, (start, end) in slices.items()
        }

    def forward(
        self,
        seed_tokens: torch.Tensor,
    ) -> ThreeRouterOutput:
        self._validate_seed(seed_tokens)
        routes = self.split(seed_tokens)
        return ThreeRouterOutput(
            tokens=seed_tokens,
            role=routes["role"],
            action=routes["action"],
            global_route=routes["global"],
        )
