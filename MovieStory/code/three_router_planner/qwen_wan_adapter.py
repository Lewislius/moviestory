from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from .config import ThreeRouterConfig
from .planner import ThreeRouterPlanner, ThreeRouterOutput


def build_three_router_encoder_class(
    base_encoder_class: Type[torch.nn.Module],
    router_config: ThreeRouterConfig,
    *,
    enabled: bool = True,
) -> Type[torch.nn.Module]:
    """
    Build a drop-in replacement for Wan's active MetaQueryEncoderForWan.

    The base class still owns Qwen3-VL loading, tokenization, the Qwen2 Connector,
    dtype/device policy and checkpoint compatibility. Only its forward path is
    specialized. In 3-router mode, role/action/global use separate modality inputs
    and disjoint MetaQuery token IDs. Their Qwen outputs are concatenated before one
    shared Connector forward, matching the single 256-token Wan conditioning path.
    """

    class ThreeRouterMetaQueryEncoderForWan(base_encoder_class):  # type: ignore[misc, valid-type]
        three_router_enabled = bool(enabled)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            requested = int(kwargs.get("num_metaqueries", router_config.total_tokens))
            if requested != router_config.total_tokens:
                raise ValueError(
                    "3-router requires exactly "
                    f"{router_config.total_tokens} MetaQuery tokens, got {requested}"
                )
            super().__init__(*args, **kwargs)
            if int(getattr(self, "num_metaqueries", requested)) != router_config.total_tokens:
                raise ValueError("Base encoder changed num_metaqueries unexpectedly")
            actual_hidden_size = int(self.mllm_model.mllm_hidden_size)
            if actual_hidden_size != router_config.hidden_size:
                raise ValueError(
                    "router hidden size must match the loaded Qwen text width: "
                    f"config={router_config.hidden_size}, qwen={actual_hidden_size}"
                )
            self.router_planner = ThreeRouterPlanner(router_config).to(
                device=self.device,
                dtype=self.dtype,
            )
            if self.three_router_enabled:
                self._initialize_route_token_ids()
                self._initialize_route_metaquery_parameters()
                self._register_route_embedding_grad_hook()
            else:
                self.router_planner.requires_grad_(False)
            self.last_router_output: Optional[ThreeRouterOutput] = None
            self.last_router_diagnostics: Dict[str, torch.Tensor] = {}
            self.last_route_embedding_grad_rms: Dict[str, torch.Tensor] = {}
            self.last_route_input_audit: Dict[str, Dict[str, object]] = {}
            self.last_joint_connector_audit: Dict[str, object] = {}

        def _initialize_route_token_ids(self) -> None:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            route_token_ids: Dict[str, tuple[int, ...]] = {}
            for route_name, (start, end) in router_config.route_slices.items():
                token_ids = tuple(
                    int(tokenizer.convert_tokens_to_ids(f"<img{index}>"))
                    for index in range(start, end)
                )
                if len(set(token_ids)) != end - start:
                    raise RuntimeError(
                        f"{route_name}: MetaQuery token IDs are not unique"
                    )
                route_token_ids[route_name] = token_ids
            all_ids = tuple(
                token_id
                for route_name in ("role", "action", "global")
                for token_id in route_token_ids[route_name]
            )
            if len(set(all_ids)) != router_config.total_tokens:
                raise RuntimeError("MetaQuery token IDs overlap across routes")
            self._route_token_ids = route_token_ids
            self._all_route_token_ids = all_ids

        def _initialize_route_metaquery_parameters(self) -> None:
            embedding = self.mllm_model.mllm_backbone.get_input_embeddings()
            weight = getattr(embedding, "weight", None)
            if weight is None:
                raise RuntimeError("Qwen input embedding weight is unavailable")
            route_parameters = {}
            with torch.no_grad():
                for route_name, token_ids in self._route_token_ids.items():
                    indices = torch.tensor(
                        token_ids,
                        device=weight.device,
                        dtype=torch.long,
                    )
                    route_parameters[route_name] = nn.Parameter(
                        # Keep these small trainable tables in FP32. With a 1e-5
                        # learning rate, direct BF16 optimizer updates can round
                        # to zero even when a route has a valid gradient.
                        weight.index_select(0, indices).detach().clone().float()
                    )
            self.route_metaquery_embeddings = nn.ParameterDict(route_parameters)

        def _register_route_embedding_grad_hook(self) -> None:
            def capture_route_gradient(
                route_name: str,
                grad: torch.Tensor,
            ) -> torch.Tensor:
                self.last_route_embedding_grad_rms[
                    f"{route_name}_mq_embedding_grad_rms"
                ] = grad.detach().float().square().mean().sqrt()
                return grad

            self._route_embedding_grad_hook_handles = [
                parameter.register_hook(
                    lambda grad, route_name=route_name: capture_route_gradient(
                        route_name,
                        grad,
                    )
                )
                for route_name, parameter in self.route_metaquery_embeddings.items()
            ]

        def _route_input_embeddings(
            self,
            input_ids: torch.Tensor,
            route_name: str,
        ) -> torch.Tensor:
            if route_name not in self.route_metaquery_embeddings:
                raise ValueError(f"unknown route: {route_name}")
            embedding = self.mllm_model.mllm_backbone.get_input_embeddings()
            inputs_embeds = embedding(input_ids).detach()
            token_ids = torch.tensor(
                self._route_token_ids[route_name],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            route_positions = (input_ids.unsqueeze(-1) == token_ids).any(dim=-1)
            expected = input_ids.shape[0] * token_ids.numel()
            if int(route_positions.sum().item()) != expected:
                raise RuntimeError(
                    f"{route_name}: expected {expected} MetaQuery positions, "
                    f"got {int(route_positions.sum().item())}"
                )
            route_values = (
                self.route_metaquery_embeddings[route_name]
                .to(dtype=inputs_embeds.dtype)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1, -1)
                .reshape(-1, inputs_embeds.shape[-1])
            )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[route_positions] = route_values
            return inputs_embeds

        @staticmethod
        def _qwen3vl_position_ids(
            backbone: torch.nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_grid_thw: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """
            Preserve Qwen3-VL's multimodal RoPE when using ``inputs_embeds``.

            Qwen3-VL can recover image placeholder masks from embeddings, but its
            automatic 3D-position path needs input IDs.  We cannot pass input IDs
            and custom embeddings together, so compute the positions before the
            forward call and pass them explicitly.
            """
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

            # Text-only Qwen3-VL still expects the three M-RoPE axes.  They are
            # identical for ordinary text positions.
            positions = attention_mask.long().cumsum(dim=-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 0)
            return positions.unsqueeze(0).expand(3, -1, -1)

        def _tokenize_inputs(self, captions: Any, input_images: Any):
            if input_images is not None:
                input_ids, attention_mask, pixel_values, image_sizes = self.tokenize(
                    self.tokenizer, captions, input_images
                )
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device, self.dtype)
                    if self.mllm_model.mllm_type in ("qwenvl", "qwen3vl"):
                        pixel_values = pixel_values.squeeze(0)
                if image_sizes is not None:
                    image_sizes = image_sizes.to(self.device)
            else:
                input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                pixel_values = None
                image_sizes = None
            return input_ids, attention_mask, pixel_values, image_sizes

        def _keep_only_route_tokens(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            route_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if route_name not in self._route_token_ids:
                raise ValueError(f"unknown route: {route_name}")

            is_any_route_token = torch.zeros_like(input_ids, dtype=torch.bool)
            is_selected_route_token = torch.zeros_like(input_ids, dtype=torch.bool)
            selected_ids = set(self._route_token_ids[route_name])
            for token_id in self._all_route_token_ids:
                matches = input_ids == token_id
                is_any_route_token |= matches
                if token_id in selected_ids:
                    is_selected_route_token |= matches

            keep = ~(is_any_route_token & ~is_selected_route_token)
            kept_lengths = keep.sum(dim=1)
            if not torch.equal(kept_lengths, kept_lengths[:1].expand_as(kept_lengths)):
                raise RuntimeError(
                    f"{route_name}: filtered prompt lengths differ across the batch"
                )
            expected_removed = (
                router_config.total_tokens
                - len(self._route_token_ids[route_name])
            )
            removed = input_ids.shape[1] - int(kept_lengths[0].item())
            if removed != expected_removed:
                raise RuntimeError(
                    f"{route_name}: expected to remove {expected_removed} MetaQuery "
                    f"tokens, removed {removed}"
                )
            batch = input_ids.shape[0]
            return (
                input_ids[keep].reshape(batch, -1),
                attention_mask[keep].reshape(batch, -1),
            )

        def _tokenize_route_inputs(
            self,
            captions: Any,
            input_images: Any,
            route_name: str,
        ):
            input_ids, attention_mask, pixel_values, image_sizes = (
                self._tokenize_inputs(captions, input_images)
            )
            input_ids, attention_mask = self._keep_only_route_tokens(
                input_ids,
                attention_mask,
                route_name,
            )
            return input_ids, attention_mask, pixel_values, image_sizes

        def _raw_metaquery_states(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: Optional[torch.Tensor],
            image_sizes: Optional[torch.Tensor],
            *,
            expected_tokens: int,
            route_name: str,
        ) -> torch.Tensor:
            backbone = self.mllm_model.mllm_backbone
            common = {
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
            }
            if self.three_router_enabled and route_name != "baseline":
                common["input_ids"] = None
                common["inputs_embeds"] = self._route_input_embeddings(
                    input_ids,
                    route_name,
                )
                if self.mllm_model.mllm_type == "qwen3vl":
                    common["position_ids"] = self._qwen3vl_position_ids(
                        backbone,
                        input_ids,
                        attention_mask,
                        image_sizes,
                    )
            else:
                common["input_ids"] = input_ids
            mllm_type = self.mllm_model.mllm_type
            if mllm_type in ("qwen3vl", "qwenvl"):
                outputs = backbone(
                    **common,
                    pixel_values=pixel_values,
                    image_grid_thw=image_sizes,
                )
            elif mllm_type == "llavaov":
                outputs = backbone(
                    **common,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                )
            else:
                outputs = backbone(**common)

            # MLLMInContext replaces lm_head with Identity, so logits are hidden states.
            hidden = outputs.logits
            boi_id = int(self.mllm_model.boi_token_id)
            eoi_id = int(self.mllm_model.eoi_token_id)
            batch, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)
            selected = []
            for row in range(batch):
                boi = torch.where(input_ids[row] == boi_id)[0]
                eoi = torch.where(input_ids[row] == eoi_id)[0]
                if boi.numel() != 1 or eoi.numel() != 1:
                    raise RuntimeError(
                        f"sample {row}: expected one BOI and EOI, got {boi.numel()}/{eoi.numel()}"
                    )
                mask = (indices > boi.item()) & (indices < eoi.item())
                row_states = hidden[row, mask]
                if row_states.shape[0] != expected_tokens:
                    raise RuntimeError(
                        f"sample {row} {route_name}: expected {expected_tokens} "
                        "route states, "
                        f"got {row_states.shape[0]}"
                    )
                selected.append(row_states)
            return torch.stack(selected, dim=0)

        @staticmethod
        def _empty_captions_like(captions: Any):
            if isinstance(captions, str):
                return ""
            try:
                return ["" for _ in captions]
            except TypeError:
                return ""

        def _isolated_route_seeds(
            self,
            captions: Any,
            input_images: Any,
        ) -> Dict[str, torch.Tensor]:
            route_inputs = {
                "role": (self._empty_captions_like(captions), input_images),
                "action": (captions, None),
                "global": (captions, input_images),
            }
            route_seeds: Dict[str, torch.Tensor] = {}
            route_audit: Dict[str, Dict[str, object]] = {}
            for route_name in ("role", "action", "global"):
                route_captions, route_images = route_inputs[route_name]
                input_ids, attention_mask, pixel_values, image_sizes = (
                    self._tokenize_route_inputs(
                        route_captions,
                        route_images,
                        route_name,
                    )
                )
                start, end = router_config.route_slices[route_name]
                route_seeds[route_name] = self._raw_metaquery_states(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_sizes,
                    expected_tokens=end - start,
                    route_name=route_name,
                )
                if isinstance(route_captions, str):
                    caption_values = [route_captions]
                else:
                    try:
                        caption_values = list(route_captions)
                    except TypeError:
                        caption_values = [route_captions]
                route_audit[route_name] = {
                    "caption_nonempty": [
                        bool(str(value).strip()) for value in caption_values
                    ],
                    "image_input_supplied": route_images is not None,
                    "pixel_values_present": pixel_values is not None,
                    "image_grid_present": image_sizes is not None,
                    "input_ids_shape": tuple(input_ids.shape),
                    "qwen_output_shape": tuple(route_seeds[route_name].shape),
                    "expected_metaquery_tokens": end - start,
                }
            self.last_route_input_audit = route_audit
            return route_seeds

        def _connect_joint_routes(
            self,
            router_output: ThreeRouterOutput,
        ) -> torch.Tensor:
            features = self.mllm_model.connector(router_output.tokens)
            previous_count = int(
                self.last_joint_connector_audit.get("call_count", 0)
            )
            self.last_joint_connector_audit = {
                "call_count": previous_count + 1,
                "input_shape": tuple(router_output.tokens.shape),
                "output_shape": tuple(features.shape),
            }
            return features

        def forward(
            self,
            captions: Any,
            input_images: Any = None,
        ) -> torch.Tensor:
            self.last_route_input_audit = {}
            self.last_joint_connector_audit = {"call_count": 0}
            if self.three_router_enabled:
                route_seeds = self._isolated_route_seeds(captions, input_images)
                route_seed = torch.cat(
                    [
                        route_seeds["role"],
                        route_seeds["action"],
                        route_seeds["global"],
                    ],
                    dim=1,
                )
            else:
                input_ids, attention_mask, pixel_values, image_sizes = (
                    self._tokenize_inputs(captions, input_images)
                )
                route_seed = self._raw_metaquery_states(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_sizes,
                    expected_tokens=router_config.total_tokens,
                    route_name="baseline",
                )
            if self.three_router_enabled:
                router_output = self.router_planner(route_seed)
                self.last_router_output = ThreeRouterOutput(
                    tokens=router_output.tokens.detach(),
                    role=router_output.role.detach(),
                    action=router_output.action.detach(),
                    global_route=router_output.global_route.detach(),
                )
                self.last_router_diagnostics = {
                    key: value.detach()
                    for key, value in router_output.diagnostics().items()
                }
                features = self._connect_joint_routes(router_output)
            else:
                planned = route_seed
                self.last_router_output = None
                self.last_router_diagnostics = {}
                features = self.mllm_model.connector(planned)

            if not self._printed_forward_stats:
                print(
                    "[ThreeRouterMetaQueryEncoderForWan] "
                    f"enabled={int(self.three_router_enabled)} "
                    "modalities=role:image,action:text,global:image+text "
                    "connector=shared_single_joint_forward "
                    f"seed={tuple(route_seed.shape)} features={tuple(features.shape)} "
                    f"dtype={features.dtype}"
                )
                self._printed_forward_stats = True
            return features

        def get_three_router_metadata(self) -> Dict[str, object]:
            return {
                "enabled": self.three_router_enabled,
                "routing_mode": "isolated_modalities_v1",
                "route_modalities": {
                    "role": "reference_image",
                    "action": "text",
                    "global": "reference_image+text",
                },
                "shared_route_connector": True,
                "separate_connector_forwards": False,
                "joint_connector_forward": True,
                "post_qwen_router_transform": "identity_split_only",
                "qwen_outputs_direct_to_connector": True,
                "trainable_route_metaquery_parameters": self.three_router_enabled,
                **router_config.to_dict(),
            }

    ThreeRouterMetaQueryEncoderForWan.__name__ = "ThreeRouterMetaQueryEncoderForWan"
    ThreeRouterMetaQueryEncoderForWan.__qualname__ = "ThreeRouterMetaQueryEncoderForWan"
    return ThreeRouterMetaQueryEncoderForWan
