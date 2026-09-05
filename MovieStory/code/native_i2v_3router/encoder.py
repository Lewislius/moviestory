"""MovieStory Qwen/MetaQuery three-router adapter and Wan context composition."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Type

import torch
import torch.nn as nn

from .contracts import ThreeRouterConfig, ThreeRouterOutput, ThreeRouterPlanner


class MQToT5Mapper(nn.Module):
    """Near-identity residual mapper from MQ features to frozen T5 space."""

    def __init__(
        self,
        hidden_size: int = 4096,
        bottleneck_size: int = 1024,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("mapper dimensions must be positive")
        if not math.isfinite(residual_scale) or not 0.0 < residual_scale < 1.0:
            raise ValueError("residual_scale must be finite and within (0, 1)")
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.input_norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.down = nn.Linear(self.hidden_size, self.bottleneck_size)
        self.activation = nn.SiLU()
        self.up = nn.Linear(self.bottleneck_size, self.hidden_size)
        self.residual_logit = nn.Parameter(torch.tensor(math.atanh(residual_scale)))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or int(features.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"MQ mapper expects [B,N,{self.hidden_size}], got {tuple(features.shape)}"
            )
        output_dtype = features.dtype
        x = features.to(dtype=self.down.weight.dtype)
        delta = self.up(self.activation(self.down(self.input_norm(x))))
        return (x + self.residual_logit.tanh() * delta).to(dtype=output_dtype)


def _pad_feature_sequences(sequences: Sequence[torch.Tensor]) -> torch.Tensor:
    if not sequences:
        raise ValueError("T5 returned no feature sequences")
    width = int(sequences[0].shape[-1])
    max_tokens = max(int(row.shape[0]) for row in sequences)
    rows = []
    for index, row in enumerate(sequences):
        if row.ndim != 2 or int(row.shape[-1]) != width:
            raise ValueError(f"invalid T5 row {index}: {tuple(row.shape)}")
        if row.shape[0] < max_tokens:
            row = torch.cat(
                [row, row.new_zeros(max_tokens - row.shape[0], width)], dim=0
            )
        rows.append(row)
    return torch.stack(rows, dim=0)


def build_dual_mode_encoder_class(
    base_encoder_class: Type[nn.Module],
    *,
    conditioning_mode: int,
    mapper_bottleneck_size: int = 1024,
    mapper_residual_scale: float = 0.1,
    mapper_rms_match: bool = True,
) -> Type[nn.Module]:
    """Return MQ-only (mode 0) or ``[mapped MQ, frozen T5]`` (mode 1)."""

    mode = int(conditioning_mode)
    if mode not in (0, 1):
        raise ValueError(f"conditioning_mode must be 0 or 1, got {mode}")

    class DualModeMetaQueryEncoder(base_encoder_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mq_to_t5_mapper: Optional[MQToT5Mapper] = None
            if mode == 1:
                self.mq_to_t5_mapper = MQToT5Mapper(
                    hidden_size=int(self.wan_text_dim),
                    bottleneck_size=int(mapper_bottleneck_size),
                    residual_scale=float(mapper_residual_scale),
                ).to(device=self.device, dtype=torch.float32)
            object.__setattr__(self, "_native_i2v_t5_provider", None)
            self.mapper_rms_match = bool(mapper_rms_match)
            self.last_context_audit: Dict[str, object] = {}

        def bind_t5_provider(
            self, provider: Callable[[Sequence[str]], list[torch.Tensor]]
        ) -> None:
            object.__setattr__(self, "_native_i2v_t5_provider", provider)

        @staticmethod
        def _rms(value: torch.Tensor) -> torch.Tensor:
            return value.float().square().mean().clamp_min(1e-12).sqrt()

        def forward(self, captions: Any, input_images: Any = None) -> torch.Tensor:
            mq = super().forward(captions, input_images)
            if mode == 0:
                self.last_context_audit = {
                    "mode": 0,
                    "mq_tokens": int(mq.shape[1]),
                    "t5_tokens": 0,
                    "context_tokens": int(mq.shape[1]),
                    "hidden_size": int(mq.shape[2]),
                }
                return mq

            provider = getattr(self, "_native_i2v_t5_provider", None)
            if not callable(provider) or self.mq_to_t5_mapper is None:
                raise RuntimeError("conditioning mode 1 requires a bound frozen-T5 provider")
            caption_rows = [captions] if isinstance(captions, str) else list(captions)
            mapped = self.mq_to_t5_mapper(mq)
            t5 = _pad_feature_sequences(provider(caption_rows)).to(
                device=mapped.device, dtype=mapped.dtype
            )
            if mapped.shape[0] != t5.shape[0] or mapped.shape[2] != t5.shape[2]:
                raise RuntimeError(
                    f"MQ/T5 context mismatch: {tuple(mapped.shape)} vs {tuple(t5.shape)}"
                )
            scale = mapped.new_tensor(1.0)
            if self.mapper_rms_match:
                with torch.no_grad():
                    scale = (self._rms(t5) / self._rms(mapped)).clamp(0.25, 4.0)
                mapped = mapped * scale.to(dtype=mapped.dtype)
            context = torch.cat([mapped, t5], dim=1)
            self.last_context_audit = {
                "mode": 1,
                "mq_tokens": int(mapped.shape[1]),
                "t5_tokens": int(t5.shape[1]),
                "context_tokens": int(context.shape[1]),
                "hidden_size": int(context.shape[2]),
                "mq_rms": float(self._rms(mapped).detach().item()),
                "t5_rms": float(self._rms(t5).detach().item()),
                "mq_rms_match_scale": float(scale.detach().float().item()),
            }
            return context

        def get_conditioning_metadata(self) -> Dict[str, object]:
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

    DualModeMetaQueryEncoder.__name__ = "DualModeMetaQueryEncoderForNativeWanI2V"
    return DualModeMetaQueryEncoder


def build_three_router_encoder_class(
    base_encoder_class: Type[nn.Module],
    router_config: ThreeRouterConfig,
    *,
    enabled: bool = True,
) -> Type[nn.Module]:
    """Specialize the existing MetaQuery encoder without changing Qwen or Wan.

    Role uses image only, action uses caption only, and global uses image plus
    caption.  Each route has an isolated Qwen forward.  Ordered raw Qwen states
    are concatenated and passed through the one shared MetaQuery Connector.
    """

    class ThreeRouterMetaQueryEncoder(base_encoder_class):  # type: ignore[misc, valid-type]
        three_router_enabled = bool(enabled)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            requested = int(kwargs.get("num_metaqueries", router_config.total_tokens))
            if requested != router_config.total_tokens:
                raise ValueError(
                    f"three-router needs {router_config.total_tokens} tokens, got {requested}"
                )
            super().__init__(*args, **kwargs)
            qwen_width = int(self.mllm_model.mllm_hidden_size)
            if qwen_width != router_config.hidden_size:
                raise ValueError(
                    f"router hidden_size={router_config.hidden_size} but Qwen width={qwen_width}"
                )
            self.router_planner = ThreeRouterPlanner(router_config).to(
                device=self.device, dtype=self.dtype
            )
            self.last_router_output: Optional[ThreeRouterOutput] = None
            self.last_router_diagnostics: Dict[str, torch.Tensor] = {}
            self.last_route_embedding_grad_rms: Dict[str, torch.Tensor] = {}
            self.last_route_input_audit: Dict[str, Dict[str, object]] = {}
            self.last_joint_connector_audit: Dict[str, object] = {}
            if self.three_router_enabled:
                self._initialize_route_token_ids()
                self._initialize_route_parameters()
                self._register_route_gradient_hooks()

        def _initialize_route_token_ids(self) -> None:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            route_ids: Dict[str, tuple[int, ...]] = {}
            for name, (start, end) in router_config.route_slices.items():
                ids = tuple(
                    int(tokenizer.convert_tokens_to_ids(f"<img{index}>"))
                    for index in range(start, end)
                )
                if len(set(ids)) != end - start:
                    raise RuntimeError(f"{name}: MetaQuery token IDs are not unique")
                route_ids[name] = ids
            all_ids = tuple(
                token_id
                for name in ("role", "action", "global")
                for token_id in route_ids[name]
            )
            if len(set(all_ids)) != router_config.total_tokens:
                raise RuntimeError("MetaQuery token IDs overlap across routes")
            self._route_token_ids = route_ids
            self._all_route_token_ids = all_ids

        def _initialize_route_parameters(self) -> None:
            weight = self.mllm_model.mllm_backbone.get_input_embeddings().weight
            parameters = {}
            with torch.no_grad():
                for name, ids in self._route_token_ids.items():
                    index = torch.tensor(ids, device=weight.device, dtype=torch.long)
                    parameters[name] = nn.Parameter(
                        weight.index_select(0, index).detach().clone().float()
                    )
            self.route_metaquery_embeddings = nn.ParameterDict(parameters)

        def _register_route_gradient_hooks(self) -> None:
            def capture(name: str, grad: torch.Tensor) -> torch.Tensor:
                self.last_route_embedding_grad_rms[
                    f"{name}_mq_embedding_grad_rms"
                ] = grad.detach().float().square().mean().sqrt()
                return grad

            self._route_gradient_handles = [
                parameter.register_hook(
                    lambda grad, name=name: capture(name, grad)
                )
                for name, parameter in self.route_metaquery_embeddings.items()
            ]

        def _route_input_embeddings(
            self, input_ids: torch.Tensor, route_name: str
        ) -> torch.Tensor:
            embedding = self.mllm_model.mllm_backbone.get_input_embeddings()
            inputs = embedding(input_ids).detach()
            ids = torch.tensor(
                self._route_token_ids[route_name],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            positions = (input_ids.unsqueeze(-1) == ids).any(dim=-1)
            expected = input_ids.shape[0] * ids.numel()
            if int(positions.sum().item()) != expected:
                raise RuntimeError(
                    f"{route_name}: expected {expected} MQ positions, "
                    f"got {int(positions.sum().item())}"
                )
            values = (
                self.route_metaquery_embeddings[route_name]
                .to(dtype=inputs.dtype)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1, -1)
                .reshape(-1, inputs.shape[-1])
            )
            inputs = inputs.clone()
            inputs[positions] = values
            return inputs

        @staticmethod
        def _qwen3vl_position_ids(
            backbone: nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_grid_thw: Optional[torch.Tensor],
        ) -> torch.Tensor:
            model = getattr(backbone, "model", None)
            get_rope_index = getattr(model, "get_rope_index", None)
            if image_grid_thw is not None and callable(get_rope_index):
                position_ids, _ = get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
                return position_ids.to(device=input_ids.device)
            positions = attention_mask.long().cumsum(dim=-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 0)
            return positions.unsqueeze(0).expand(3, -1, -1)

        def _tokenize_inputs(self, captions: Any, images: Any):
            if images is None:
                input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
                pixel_values = image_sizes = None
            else:
                input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
                    self.tokenizer, captions, images
                )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device, self.dtype)
                if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
                    pixel_values = pixel_values.squeeze(0)
            if image_sizes is not None:
                image_sizes = image_sizes.to(self.device)
            return input_ids, attention_mask, pixel_values, image_sizes

        def _keep_only_route_tokens(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            route_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            any_route = torch.zeros_like(input_ids, dtype=torch.bool)
            selected = torch.zeros_like(input_ids, dtype=torch.bool)
            selected_ids = set(self._route_token_ids[route_name])
            for token_id in self._all_route_token_ids:
                match = input_ids == token_id
                any_route |= match
                if token_id in selected_ids:
                    selected |= match
            keep = ~(any_route & ~selected)
            lengths = keep.sum(dim=1)
            if not torch.equal(lengths, lengths[:1].expand_as(lengths)):
                raise RuntimeError(f"{route_name}: filtered prompt lengths differ")
            removed = input_ids.shape[1] - int(lengths[0].item())
            expected = router_config.total_tokens - len(self._route_token_ids[route_name])
            if removed != expected:
                raise RuntimeError(
                    f"{route_name}: expected to remove {expected} MQ tokens, removed {removed}"
                )
            batch = input_ids.shape[0]
            return (
                input_ids[keep].reshape(batch, -1),
                attention_mask[keep].reshape(batch, -1),
            )

        def _raw_route_states(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: Optional[torch.Tensor],
            image_sizes: Optional[torch.Tensor],
            *,
            route_name: str,
            expected_tokens: int,
        ) -> torch.Tensor:
            backbone = self.mllm_model.mllm_backbone
            common: Dict[str, Any] = {
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
            }
            if self.three_router_enabled and route_name != "baseline":
                common["input_ids"] = None
                common["inputs_embeds"] = self._route_input_embeddings(
                    input_ids, route_name
                )
                if self.mllm_model.mllm_type == "qwen3vl":
                    common["position_ids"] = self._qwen3vl_position_ids(
                        backbone, input_ids, attention_mask, image_sizes
                    )
            else:
                common["input_ids"] = input_ids

            if self.mllm_model.mllm_type in ("qwen3vl", "qwenvl"):
                outputs = backbone(
                    **common,
                    pixel_values=pixel_values,
                    image_grid_thw=image_sizes,
                )
            elif self.mllm_model.mllm_type == "llavaov":
                outputs = backbone(
                    **common, pixel_values=pixel_values, image_sizes=image_sizes
                )
            else:
                outputs = backbone(**common)

            hidden = outputs.logits
            boi_id = int(self.mllm_model.boi_token_id)
            eoi_id = int(self.mllm_model.eoi_token_id)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            selected_rows = []
            for row in range(input_ids.shape[0]):
                boi = torch.where(input_ids[row] == boi_id)[0]
                eoi = torch.where(input_ids[row] == eoi_id)[0]
                if boi.numel() != 1 or eoi.numel() != 1:
                    raise RuntimeError(
                        f"sample {row}: expected one BOI/EOI, got {boi.numel()}/{eoi.numel()}"
                    )
                mask = (positions > boi.item()) & (positions < eoi.item())
                states = hidden[row, mask]
                if states.shape[0] != expected_tokens:
                    raise RuntimeError(
                        f"sample {row} {route_name}: expected {expected_tokens} states, "
                        f"got {states.shape[0]}"
                    )
                selected_rows.append(states)
            return torch.stack(selected_rows, dim=0)

        @staticmethod
        def _empty_captions(captions: Any):
            if isinstance(captions, str):
                return ""
            return ["" for _ in captions]

        def _one_route(self, captions: Any, images: Any, route_name: str):
            input_ids, attention_mask, pixel_values, image_sizes = self._tokenize_inputs(
                captions, images
            )
            input_ids, attention_mask = self._keep_only_route_tokens(
                input_ids, attention_mask, route_name
            )
            start, end = router_config.route_slices[route_name]
            states = self._raw_route_states(
                input_ids,
                attention_mask,
                pixel_values,
                image_sizes,
                route_name=route_name,
                expected_tokens=end - start,
            )
            self.last_route_input_audit[route_name] = {
                "caption_nonempty": bool(
                    any(str(value).strip() for value in ([captions] if isinstance(captions, str) else captions))
                ),
                "image_input_supplied": images is not None,
                "pixel_values_present": pixel_values is not None,
                "input_ids_shape": tuple(input_ids.shape),
                "qwen_output_shape": tuple(states.shape),
            }
            return states

        def forward(self, captions: Any, input_images: Any = None) -> torch.Tensor:
            self.last_route_input_audit = {}
            if self.three_router_enabled:
                role = self._one_route(self._empty_captions(captions), input_images, "role")
                action = self._one_route(captions, None, "action")
                global_route = self._one_route(captions, input_images, "global")
                seeds = torch.cat([role, action, global_route], dim=1)
                planned = self.router_planner(seeds)
                self.last_router_output = ThreeRouterOutput(
                    planned.tokens.detach(),
                    planned.role.detach(),
                    planned.action.detach(),
                    planned.global_route.detach(),
                )
                self.last_router_diagnostics = {
                    key: value.detach() for key, value in planned.diagnostics().items()
                }
                features = self.mllm_model.connector(planned.tokens)
                self.last_joint_connector_audit = {
                    "call_count": 1,
                    "input_shape": tuple(planned.tokens.shape),
                    "output_shape": tuple(features.shape),
                }
            else:
                input_ids, attention_mask, pixel_values, image_sizes = self._tokenize_inputs(
                    captions, input_images
                )
                seeds = self._raw_route_states(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_sizes,
                    route_name="baseline",
                    expected_tokens=router_config.total_tokens,
                )
                features = self.mllm_model.connector(seeds)
                self.last_router_output = None
                self.last_router_diagnostics = {}
                self.last_joint_connector_audit = {
                    "call_count": 1,
                    "input_shape": tuple(seeds.shape),
                    "output_shape": tuple(features.shape),
                }
            if not self._printed_forward_stats:
                print(
                    "[Native-I2V-3Router] "
                    f"enabled={int(self.three_router_enabled)} "
                    "role=image action=text global=image+text "
                    f"shared_connector=1 seed={tuple(seeds.shape)} "
                    f"features={tuple(features.shape)}",
                    flush=True,
                )
                self._printed_forward_stats = True
            return features

        def get_three_router_metadata(self) -> Dict[str, object]:
            return {
                "enabled": self.three_router_enabled,
                "route_modalities": {
                    "role": "reference_image",
                    "action": "caption",
                    "global": "reference_image+caption",
                },
                "isolated_qwen_forwards": True,
                "joint_shared_connector_forward": True,
                "post_qwen_transform": "identity_split_only",
                **router_config.to_dict(),
            }

    ThreeRouterMetaQueryEncoder.__name__ = "ThreeRouterMetaQueryEncoderForNativeWanI2V"
    return ThreeRouterMetaQueryEncoder
