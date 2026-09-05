from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ThreeRouterConfig:
    """Configuration for the isolated 3-router planner increment."""

    hidden_size: int = 2048
    role_tokens: int = 96
    action_tokens: int = 96
    global_tokens: int = 64

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "role_tokens": self.role_tokens,
            "action_tokens": self.action_tokens,
            "global_tokens": self.global_tokens,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"3-router dimensions must be positive: {invalid}")

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
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        payload["route_slices"] = self.route_slices
        return payload
