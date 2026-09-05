import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A lightweight LoRA wrapper for nn.Linear used by Wan DiT training/inference."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.scaling = float(alpha) / float(rank)
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        base_device = base.weight.device
        base_dtype = base.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=base_device,
                dtype=base_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=base_device,
                dtype=base_dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA mode keeps the base projection frozen.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.rank <= 0:
            return base_out

        lora_x = x
        if self.dropout_p > 0.0:
            lora_x = F.dropout(lora_x, p=self.dropout_p, training=self.training)
        lora_x = lora_x.to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(dtype=base_out.dtype)


def _replace_child_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    parent._modules[str(name)] = module


def _normalize_target_types(targets: Sequence[str] | str | None) -> Tuple[str, ...]:
    if targets is None:
        return ("self_attn", "cross_attn", "ffn")
    if isinstance(targets, str):
        raw = [x.strip().lower() for x in targets.split(",") if x.strip()]
    else:
        raw = [str(x).strip().lower() for x in targets if str(x).strip()]
    if not raw:
        return ("self_attn", "cross_attn", "ffn")
    allowed = {"self_attn", "cross_attn", "ffn"}
    out = []
    for item in raw:
        if item not in allowed:
            raise ValueError(
                f"Unknown LoRA target type: {item}. Allowed: {sorted(allowed)}"
            )
        if item not in out:
            out.append(item)
    return tuple(out)


def _match_wan_lora_linear(full_name: str, target_types: Sequence[str]) -> bool:
    name = str(full_name)
    if not name.startswith("blocks."):
        return False
    if ".self_attn." in name:
        return "self_attn" in target_types and any(
            name.endswith(suffix) for suffix in (".q", ".k", ".v", ".o")
        )
    if ".cross_attn." in name:
        return "cross_attn" in target_types and any(
            name.endswith(suffix) for suffix in (".q", ".k", ".v", ".o")
        )
    if ".ffn." in name:
        return "ffn" in target_types and any(
            name.endswith(suffix) for suffix in (".0", ".2")
        )
    return False


def apply_lora_to_wan_model(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_types: Sequence[str] | str | None = None,
) -> List[str]:
    if model is None or not isinstance(model, nn.Module):
        raise TypeError("Wan model must be an nn.Module for LoRA injection")

    normalized_targets = _normalize_target_types(target_types)
    replaced: List[str] = []
    for full_name, child in list(model.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not _match_wan_lora_linear(full_name, normalized_targets):
            continue
        if isinstance(child, LoRALinear):
            replaced.append(full_name)
            continue

        if "." not in full_name:
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        _replace_child_module(parent, child_name, wrapped)
        replaced.append(full_name)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> List[str]:
    names: List[str] = []
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            param.requires_grad_(True)
            names.append(name)
    return names


def collect_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".lora_A" in name or ".lora_B" in name:
            if torch.is_tensor(tensor):
                out[str(name)] = tensor.detach().cpu().contiguous()
    return out


def infer_lora_targets_from_state_keys(keys: Iterable[str]) -> Tuple[str, ...]:
    found: List[str] = []
    for key in keys:
        key = str(key)
        if ".self_attn." in key and "self_attn" not in found:
            found.append("self_attn")
        if ".cross_attn." in key and "cross_attn" not in found:
            found.append("cross_attn")
        if ".ffn." in key and "ffn" not in found:
            found.append("ffn")
    return tuple(found) if found else ("self_attn", "cross_attn", "ffn")


def infer_lora_rank_from_state(state: Dict[str, torch.Tensor], default: int = 16) -> int:
    for key, tensor in state.items():
        if str(key).endswith(".lora_A") and torch.is_tensor(tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    return int(default)


def build_lora_config_dict(
    enabled: bool,
    rank: int,
    alpha: float,
    dropout: float,
    targets: Sequence[str] | str | None,
    module_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "targets": list(_normalize_target_types(targets)),
        "module_count": int(len(module_names or [])),
        "module_preview": list(module_names or [])[:64],
    }
