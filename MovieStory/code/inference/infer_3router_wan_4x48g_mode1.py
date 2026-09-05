#!/usr/bin/env python3
"""Inference for the 4 x 48 GiB MovieStory three-router mode-1 checkpoint.

The training contract is:

    three routed Qwen streams -> one shared Connector -> MQ-to-T5 mapper
    -> mapper-only RMS matching -> frozen UMT5 prompt tokens -> Wan DiT

The resulting Wan context is ``[mapped MQ (256), actual frozen-T5 tokens]``;
Wan pads that variable-length sequence to its configured 768-token capacity.
The clean first-frame latent replaces target slot zero, is kept fixed at
timestep zero throughout sampling, and remains in the sequence passed to VAE
decoding, matching ``train_openvid4000_3router_4x48g.sh 1``.
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image

INFERENCE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = INFERENCE_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
HOME_ROOT = PROJECT_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import infer_3router_planner_wan as core  # noqa: E402
from four_gpu_training import build_dual_mode_encoder_class  # noqa: E402
from three_router_planner import (  # noqa: E402
    ThreeRouterConfig,
    build_three_router_encoder_class,
)


DEFAULT_CHECKPOINT = (
    CODE_ROOT
    / "checkpoint"
    / "three_router_mapped-mq-plus-t5_strongbind_openvid4000_4x48g_steps150"
    / "checkpoint-final"
)
DEFAULT_OUTPUT = (
    CODE_ROOT
    / "inference_outputs"
    / "openvid4000_3router_4x48g_mode1_strongbind"
    / "mode1_result.mp4"
)
DEFAULT_WAN_CHECKPOINT = HOME_ROOT / "model" / "Wan2.2" / "Wan2.2-TI2V-5B"
DEFAULT_QWEN_MODEL = (
    HOME_ROOT / "model" / "Qwen3-VL-main" / "Qwen3-VL-2B-Thinking"
)
MODE_ONE_NAME = "mapped_mq_then_frozen_t5_prompt_tokens"
TARGET_CHECKPOINT_STEP = 150
TARGET_FRAME_NUM = 49
TARGET_MAX_AREA = 262144
MAPPER_STATE_KEYS = {
    "mq_to_t5_mapper.input_norm.weight": (4096,),
    "mq_to_t5_mapper.down.weight": (1024, 4096),
    "mq_to_t5_mapper.down.bias": (1024,),
    "mq_to_t5_mapper.up.weight": (4096, 1024),
    "mq_to_t5_mapper.up.bias": (4096,),
    "mq_to_t5_mapper.residual_logit": (),
}


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def parse_args(argv: Sequence[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(raw_argv, "--checkpoint_dir"):
        raw_argv = ["--checkpoint_dir", str(DEFAULT_CHECKPOINT), *raw_argv]
    if not _has_option(raw_argv, "--output_path"):
        raw_argv = ["--output_path", str(DEFAULT_OUTPUT), *raw_argv]
    if not _has_option(raw_argv, "--wan_checkpoint_dir"):
        raw_argv = [
            "--wan_checkpoint_dir",
            str(DEFAULT_WAN_CHECKPOINT),
            *raw_argv,
        ]
    if not _has_option(raw_argv, "--qwen3vl_model_id"):
        raw_argv = ["--qwen3vl_model_id", str(DEFAULT_QWEN_MODEL), *raw_argv]
    if not _has_option(raw_argv, "--encoder_device"):
        raw_argv = ["--encoder_device", "0", *raw_argv]
    if not _has_option(raw_argv, "--dit_device"):
        raw_argv = ["--dit_device", "0", *raw_argv]
    args = core.parse_args(raw_argv)
    if int(args.encoder_device) != int(args.dit_device):
        raise ValueError(
            "Mode-1 inference uses one GPU; --encoder_device and "
            "--dit_device must be identical"
        )
    if args.first_frame_mode != "preserved":
        raise ValueError(
            "Mode-1 strongbind inference requires "
            "--first_frame_mode preserved"
        )
    if args.cfg_uncond_mode != "empty_mq":
        raise ValueError(
            "Only --cfg_uncond_mode empty_mq matches this run's joint-null "
            "training branch"
        )
    return args


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def validate_checkpoint_bundle(path_value: str | Path) -> core.CheckpointBundle:
    """Validate and describe exactly the checkpoint produced by mode-1 training."""
    checkpoint_dir = core._resolve_checkpoint_dir(path_value)
    required_json = (
        "config.json",
        "four_gpu_training_config.json",
        "training_args.json",
        "trainer_state.json",
        "metrics_tail.json",
    )
    missing = [
        name for name in required_json if not (checkpoint_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Mode-1 checkpoint is missing required metadata files: {missing}"
        )

    config = core._read_json(checkpoint_dir / "config.json")
    four_gpu = core._read_json(
        checkpoint_dir / "four_gpu_training_config.json"
    )
    training_args = core._read_json(checkpoint_dir / "training_args.json")
    trainer_state = core._read_json(checkpoint_dir / "trainer_state.json")
    metrics_tail = core._read_json(checkpoint_dir / "metrics_tail.json")
    conditioning = _as_mapping(four_gpu.get("conditioning"))
    distributed = _as_mapping(four_gpu.get("distributed"))
    reference = _as_mapping(four_gpu.get("reference_conditioning"))
    router_metadata = _as_mapping(four_gpu.get("router"))

    router_config = ThreeRouterConfig(
        hidden_size=int(router_metadata.get("hidden_size", 0)),
        role_tokens=int(router_metadata.get("role_tokens", 0)),
        action_tokens=int(router_metadata.get("action_tokens", 0)),
        global_tokens=int(router_metadata.get("global_tokens", 0)),
    )
    expected_context_tokens = router_config.total_tokens + 512
    exact_checks = {
        "four_gpu_format": (
            four_gpu.get("format")
            == "moviestory_4x48g_first_frame_strongbind_v1"
        ),
        "conditioning_mode_one": (
            int(conditioning.get("mode", -1)) == 1
            and int(training_args.get("conditioning_mode", -1)) == 1
        ),
        "mode_one_contract": (
            conditioning.get("active_contract") == MODE_ONE_NAME
        ),
        "mapped_mq_then_t5": conditioning.get("mq_to_t5_mapper") is True,
        "mapper_bottleneck": (
            int(conditioning.get("mapper_bottleneck_size", -1)) == 1024
            and int(training_args.get("mapper_bottleneck_size", -2)) == 1024
        ),
        "mapper_rms_match": (
            conditioning.get("mapper_rms_match") is True
            and training_args.get("disable_mapper_rms_match") is False
        ),
        "context_is_256_plus_512": (
            router_config.total_tokens == 256
            and int(conditioning.get("wan_text_len", -1))
            == expected_context_tokens
            and int(training_args.get("moviestory_context_text_len", -1))
            == expected_context_tokens
        ),
        "three_router_enabled": (
            four_gpu.get("three_router_enabled") is True
            and training_args.get("three_router_enabled") is True
        ),
        "router_layout": (
            router_config.hidden_size == 2048
            and router_config.role_tokens == 96
            and router_config.action_tokens == 96
            and router_config.global_tokens == 64
        ),
        "connector_contract": (
            int(config.get("connector_num_hidden_layers", -1)) == 24
            and int(config.get("wan_text_dim", -1)) == 4096
        ),
        "wan_context_mode": (
            training_args.get("dit_condition_mode") == "mq_only"
            and config.get("wan_train_mode") == "cond_only"
        ),
        "first_frame_reference_training": (
            training_args.get("moviestory_first_frame_reference") is True
            and reference.get("sampling") == "target_video_first_frame"
        ),
        "strongbind_first_frame_contract": (
            training_args.get("wan_first_frame_strong_bind") is True
            and training_args.get("moviestory_wan_first_frame_strong_bind")
            is True
            and training_args.get("enable_ti2v_first_frame_condition") is True
            and training_args.get("train_video_conditioning_mode")
            == "wan_animate_slot"
            and int(training_args.get("train_animate_ref_frames", -1)) == 1
            and int(training_args.get("train_animate_temporal_frames", -1)) == 0
            and int(training_args.get("train_animate_conditional_frames", -1))
            == 0
            and training_args.get("train_animate_preserve_timestep_zero") is True
            and training_args.get("train_animate_drop_prefix_loss") is True
            and reference.get("wan_injection")
            == "strongbind_clean_preserved_reference_prefix"
            and reference.get("strong_bind") is True
            and int(reference.get("reference_timestep", -1)) == 0
            and reference.get("reference_prefix_in_loss") is False
            and reference.get("target_first_latent_removed") is True
            and reference.get("original_first_target_latent_in_loss") is False
            and reference.get("temporal_binding") == "exact_first_frame"
            and reference.get("latent_prefix_binding") == "hard_clean"
            and reference.get("mq_image_dropout_preserves_wan_reference") is True
        ),
        "no_soft_anchor": (
            training_args.get("train_ref_anchor_mode") == "none"
            and float(training_args.get("train_ref_anchor_alpha0", math.nan))
            == 0.0
            and float(
                training_args.get("train_ref_anchor_warmup_ratio", math.nan)
            )
            == 0.0
        ),
        "joint_null_branch": (
            0.0 < float(reference.get("joint_null_prob", -1.0)) <= 1.0
            and float(training_args.get("joint_null_prob", -2.0))
            == float(reference.get("joint_null_prob", -1.0))
        ),
        "four_rank_training": (
            int(distributed.get("world_size", -1)) == 4
            and training_args.get("moviestory_hardware_contract") == "4x48g"
        ),
        "video_velocity_loss_only": (
            four_gpu.get("loss_contract")
            == "video_ground_truth_velocity_mse_only"
            and training_args.get("moviestory_loss_contract")
            == "video_ground_truth_velocity_mse_only"
            and training_args.get("enable_t5_alignment") is False
            and float(training_args.get("lambda_t5_align_l2", math.nan)) == 0.0
            and float(training_args.get("lambda_t5_align_cos", math.nan)) == 0.0
            and float(training_args.get("lambda_t5_align_stats", math.nan))
            == 0.0
            and training_args.get("enable_mq_image_preserve") is False
            and float(
                training_args.get("lambda_mq_image_preserve", math.nan)
            )
            == 0.0
            and training_args.get("enable_wan_func_distill") is False
            and float(training_args.get("lambda_wan_func_distill", math.nan))
            == 0.0
        ),
        "training_completed": (
            int(trainer_state.get("global_step", -1))
            == TARGET_CHECKPOINT_STEP
            and int(config.get("checkpoint_step", -1))
            == TARGET_CHECKPOINT_STEP
            and int(config.get("num_train_steps", -1))
            == TARGET_CHECKPOINT_STEP
            and int(training_args.get("num_train_steps", -1))
            == TARGET_CHECKPOINT_STEP
            and int(distributed.get("optimizer_steps", -1))
            == TARGET_CHECKPOINT_STEP
        ),
        "training_geometry": (
            int(config.get("frame_num", -1)) == TARGET_FRAME_NUM
            and int(training_args.get("frame_num", -1))
            == TARGET_FRAME_NUM
            and int(config.get("max_area", -1)) == TARGET_MAX_AREA
            and int(training_args.get("max_area", -1))
            == TARGET_MAX_AREA
        ),
        "wan_update_saved": config.get("has_wan_dit_trainable_pt") is True,
    }
    failures = [name for name, passed in exact_checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Checkpoint is incompatible with 4x48G mode-1 inference: "
            f"{failures}"
        )

    mq_state_path = core._pick_tensor_file(
        checkpoint_dir, "mq_encoder_trainable"
    )
    wan_state_path = core._pick_tensor_file(checkpoint_dir, "wan_dit_trainable")
    assert mq_state_path is not None
    assert wan_state_path is not None
    full_mq_path = next(
        (
            checkpoint_dir / name
            for name in ("model.safetensors", "mq_encoder_full.pt")
            if (checkpoint_dir / name).is_file()
        ),
        None,
    )
    if full_mq_path is None:
        raise FileNotFoundError(
            "Checkpoint lacks model.safetensors or mq_encoder_full.pt"
        )

    mq_shapes = core._inspect_tensor_file(mq_state_path)
    wan_shapes = core._inspect_tensor_file(wan_state_path)
    full_shapes = core._inspect_tensor_file(full_mq_path)
    tensor_failures: list[str] = []
    expected_route_shapes = {
        "role": (router_config.role_tokens, router_config.hidden_size),
        "action": (router_config.action_tokens, router_config.hidden_size),
        "global": (router_config.global_tokens, router_config.hidden_size),
    }
    for route_name, expected_shape in expected_route_shapes.items():
        key = core.ROUTE_STATE_KEYS[route_name]
        if mq_shapes.get(key) != expected_shape:
            tensor_failures.append(
                f"{key}: expected {expected_shape}, got {mq_shapes.get(key)}"
            )
    for key, expected_shape in MAPPER_STATE_KEYS.items():
        if mq_shapes.get(key) != expected_shape:
            tensor_failures.append(
                f"{key}: expected {expected_shape}, got {mq_shapes.get(key)}"
            )
    connector_keys = [
        key for key in mq_shapes if key.startswith("mllm_model.connector.")
    ]
    if not connector_keys:
        tensor_failures.append("no shared Connector tensors were saved")
    expected_wan_count = int(
        _as_mapping(config.get("extra_info")).get(
            "wan_trainable_tensor_count", 0
        )
    )
    if expected_wan_count <= 0 or len(wan_shapes) != expected_wan_count:
        tensor_failures.append(
            "Wan tensor count mismatch: "
            f"metadata={expected_wan_count}, file={len(wan_shapes)}"
        )
    if any("_flat_param" in key for key in wan_shapes):
        tensor_failures.append("Wan state contains non-portable FSDP flat keys")

    embedding_keys = [
        key
        for key in full_shapes
        if key.endswith("language_model.embed_tokens.weight")
    ]
    if len(embedding_keys) != 1:
        tensor_failures.append(
            f"expected one Qwen input embedding tensor, got {embedding_keys}"
        )
        full_embedding_key = ""
    else:
        full_embedding_key = embedding_keys[0]
    base_embedding_rows = int(config.get("mllm_embed_rows_base", 0))
    total_embedding_rows = int(config.get("mllm_embed_rows_total", 0))
    expected_total_rows = base_embedding_rows + router_config.total_tokens + 2
    if (
        not full_embedding_key
        or base_embedding_rows <= 0
        or total_embedding_rows != expected_total_rows
        or full_shapes.get(full_embedding_key)
        != (total_embedding_rows, router_config.hidden_size)
    ):
        tensor_failures.append(
            "Qwen embedding layout mismatch: "
            f"base={base_embedding_rows}, total={total_embedding_rows}, "
            f"shape={full_shapes.get(full_embedding_key)}"
        )
    if tensor_failures:
        raise RuntimeError(
            "Mode-1 checkpoint tensor validation failed: "
            + "; ".join(tensor_failures)
        )

    metric_records = metrics_tail.get("records")
    if not isinstance(metric_records, list) or not metric_records:
        raise RuntimeError("metrics_tail.json has no training records")
    final_metrics = max(
        (row for row in metric_records if isinstance(row, dict)),
        key=lambda row: int(row.get("train/step", -1)),
    )
    metric_step = int(final_metrics.get("train/step", -1))
    if metric_step != int(trainer_state["global_step"]):
        raise RuntimeError(
            "Final metric step does not match trainer state: "
            f"metrics={metric_step}, trainer={trainer_state['global_step']}"
        )
    skip_counts = {
        name: int(final_metrics.get(f"train/{name}", -1))
        for name in ("skipped_step_count", "oom_skip_count", "error_skip_count")
    }
    if any(value != 0 for value in skip_counts.values()):
        raise RuntimeError(f"Training contains skipped steps: {skip_counts}")
    final_loss = _require_finite_float(
        final_metrics.get("train/loss_denoise"), "final denoising loss"
    )

    synthetic_router = {
        "format": four_gpu["format"],
        "enabled": True,
        "hidden_size": router_config.hidden_size,
        "role_tokens": router_config.role_tokens,
        "action_tokens": router_config.action_tokens,
        "global_tokens": router_config.global_tokens,
        "wan_first_frame_conditioning": {
            "enabled": True,
            "mode": "clean_preserved_latent_slot",
            "source": "ref_image",
            "latent_slots": 1,
            "timestep_zero": True,
            "shares_video_random_timestep": False,
            "remains_noisy_denoising_input": False,
            "original_first_target_slot_removed": True,
            "original_first_target_slot_retained": False,
            "excluded_from_denoising_loss": True,
            "soft_anchor_mode": "none",
            "soft_anchor_alpha0": 0.0,
            "soft_anchor_warmup_ratio": 0.0,
        },
    }
    report = {
        "status": "pass",
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": int(trainer_state["global_step"]),
        "checks": exact_checks,
        "files": {
            "mq_trainable": mq_state_path,
            "wan_trainable": wan_state_path,
            "full_mq": full_mq_path,
        },
        "tensor_layout": {
            "mq_tensor_count": len(mq_shapes),
            "connector_tensor_count": len(connector_keys),
            "mapper_shapes": {
                key: mq_shapes[key] for key in MAPPER_STATE_KEYS
            },
            "route_shapes": {
                route: mq_shapes[core.ROUTE_STATE_KEYS[route]]
                for route in core.ROUTE_NAMES
            },
            "wan_tensor_count": len(wan_shapes),
            "qwen_full_embedding_shape": full_shapes[full_embedding_key],
        },
        "conditioning_contract": {
            "mode": 1,
            "order": "mapped_mq_then_frozen_t5_prompt_tokens",
            "mq_tokens": router_config.total_tokens,
            "t5_token_capacity": 512,
            "dit_text_len_capacity": expected_context_tokens,
            "actual_t5_tokens": "prompt_dependent",
            "actual_context_tokens": "256 + actual_t5_tokens",
            "mapper_bottleneck_size": 1024,
            "mapper_rms_match_clip": [0.25, 4.0],
            "route_modalities": core.EXPECTED_MODALITIES,
            "wan_reference": "clean_preserved_latent_slot",
            "reference_timestep": 0,
            "reference_replaces_target_slot_zero": True,
            "original_target_slot_zero_removed_during_training": True,
            "reference_prefix_retained_for_vae_decode": True,
            "cfg_null_branch": "MQ('', None) + frozen T5('')",
            "cfg_wan_reference_preserved": True,
        },
        "training_evidence": {
            "final_step": metric_step,
            "final_denoising_loss": final_loss,
            "parameter_sample_l2_delta": _require_finite_float(
                final_metrics.get("train/param_sample_l2_delta"),
                "parameter sample L2 delta",
            ),
            "skip_counts": skip_counts,
        },
    }
    return core.CheckpointBundle(
        directory=checkpoint_dir,
        config=config,
        router=synthetic_router,
        training_args=training_args,
        trainer_state=trainer_state,
        mq_state_path=mq_state_path,
        wan_state_path=wan_state_path,
        full_mq_path=full_mq_path,
        full_embedding_key=full_embedding_key,
        base_embedding_rows=base_embedding_rows,
        total_embedding_rows=total_embedding_rows,
        router_config=router_config,
        report=report,
    )


class ModeOneThreeRouterWanInference(core.ThreeRouterWanInference):
    """Restore and run the exact mode-1 context composition used in training."""

    def __init__(self, args: Any, bundle: core.CheckpointBundle) -> None:
        self.context_tokens = int(
            bundle.training_args["moviestory_context_text_len"]
        )
        super().__init__(args, bundle)

    def _dit_context_token_count(self) -> int:
        return self.context_tokens

    def _validate_dit_context_token_count(
        self,
        actual_tokens: int,
        configured_tokens: int,
    ) -> None:
        mq_tokens = self.bundle.router_config.total_tokens
        if not mq_tokens < actual_tokens <= configured_tokens:
            raise RuntimeError(
                "Mode-1 context must contain 256 mapped-MQ tokens followed by "
                "1..512 actual T5 tokens: "
                f"context={actual_tokens}, Wan text_len={configured_tokens}"
            )

    def _extra_reference_prefix_slots(self) -> int:
        # Strongbind training replaces the removed target slot zero with the
        # clean reference. It is part of the decoded output, not an extra slot.
        return 0

    def _assess_output_quality(
        self,
        video: torch.Tensor,
        reference_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        report = super()._assess_output_quality(video, reference_tensor)
        report["strongbind_contract"] = {
            "reference_is_target_first_frame": True,
            "first_output_frame_must_match_reference": True,
            "reference_prefix_retained_for_vae_decode": True,
            "spatial_high_frequency_audit_enabled": True,
        }
        return report

    def _encode_t5_for_encoder(
        self, texts: Sequence[str]
    ) -> list[torch.Tensor]:
        """Match the trainer's frozen-T5 provider, targeting the encoder GPU."""
        if not texts:
            raise ValueError("Mode 1 requires at least one caption for T5")
        t5_device = torch.device("cpu") if self.args.t5_cpu else self.encoder_device
        if not self.args.t5_cpu:
            self.wan.text_encoder.model.to(t5_device)
        with torch.inference_mode():
            values = self.wan.text_encoder(list(texts), t5_device)
        if len(values) != len(texts):
            raise RuntimeError(
                f"T5 returned {len(values)} rows for {len(texts)} captions"
            )
        result = [
            value.to(self.encoder_device, dtype=torch.bfloat16)
            for value in values
        ]
        if not self.args.t5_cpu and self.args.offload_model:
            self.wan.text_encoder.model.cpu()
            torch.cuda.empty_cache()
        return result

    def _load_mq_encoder(self) -> None:
        import train_connector_for_wan as connector_module

        routed_class = build_three_router_encoder_class(
            connector_module.MetaQueryEncoderForWan,
            self.bundle.router_config,
            enabled=True,
        )
        encoder_class = build_dual_mode_encoder_class(
            routed_class,
            conditioning_mode=1,
            mapper_bottleneck_size=int(
                self.bundle.training_args["mapper_bottleneck_size"]
            ),
            mapper_residual_scale=float(
                self.bundle.training_args["mapper_residual_scale"]
            ),
            mapper_rms_match=not bool(
                self.bundle.training_args["disable_mapper_rms_match"]
            ),
        )
        encoder = encoder_class(
            qwen3vl_model_id=self.args.qwen3vl_model_id,
            num_metaqueries=self.bundle.router_config.total_tokens,
            connector_num_hidden_layers=int(
                self.bundle.config["connector_num_hidden_layers"]
            ),
            gradient_checkpointing=False,
            train_input_embeddings=False,
            connector_norm_init_scale=float(
                self.bundle.config.get("mq_connector_norm_init_scale", 1.0)
            ),
            dtype=torch.bfloat16,
            device=str(self.encoder_device),
        )
        self.wan.text_encoder.model.eval().requires_grad_(False)
        encoder.bind_t5_provider(self._encode_t5_for_encoder)

        embedding = encoder.mllm_model.mllm_backbone.get_input_embeddings()
        embedding_weight = embedding.weight
        if int(encoder.mllm_model.num_embeddings) != self.bundle.base_embedding_rows:
            raise RuntimeError(
                "Fresh Qwen base embedding rows differ from training: "
                f"model={encoder.mllm_model.num_embeddings}, "
                f"checkpoint={self.bundle.base_embedding_rows}"
            )
        if int(embedding_weight.shape[0]) != self.bundle.total_embedding_rows:
            raise RuntimeError(
                "Fresh Qwen total embedding rows differ from training: "
                f"model={embedding_weight.shape[0]}, "
                f"checkpoint={self.bundle.total_embedding_rows}"
            )

        checkpoint_special_rows = core._load_tensor_rows(
            self.bundle.full_mq_path,
            self.bundle.full_embedding_key,
            self.bundle.base_embedding_rows,
        )
        expected_special_shape = (
            self.bundle.router_config.total_tokens + 2,
            self.bundle.router_config.hidden_size,
        )
        if tuple(checkpoint_special_rows.shape) != expected_special_shape:
            raise RuntimeError(
                "Frozen Qwen special-token shape mismatch: "
                f"expected={expected_special_shape}, "
                f"checkpoint={tuple(checkpoint_special_rows.shape)}"
            )
        with torch.no_grad():
            embedding_weight[self.bundle.base_embedding_rows :].copy_(
                checkpoint_special_rows.to(
                    device=embedding_weight.device,
                    dtype=embedding_weight.dtype,
                )
            )
        restored_special_rows = embedding_weight[
            self.bundle.base_embedding_rows :
        ].detach()
        if not torch.equal(
            restored_special_rows,
            checkpoint_special_rows.to(
                device=restored_special_rows.device,
                dtype=restored_special_rows.dtype,
            ),
        ):
            raise RuntimeError("Frozen Qwen special-token rows were not restored")

        state, load_report = core._load_subset_strict(
            encoder,
            self.bundle.mq_state_path,
            label="mode-1 MQ/Connector/mapper",
        )
        restored: Dict[str, Any] = {}
        model_state = encoder.state_dict()
        for key in (*core.ROUTE_STATE_KEYS.values(), *MAPPER_STATE_KEYS):
            current = model_state[key].detach().cpu()
            expected = state[key].to(dtype=current.dtype)
            if not torch.equal(current, expected):
                raise RuntimeError(f"Loaded tensor does not equal checkpoint: {key}")
            restored[key] = {
                "shape": tuple(current.shape),
                "dtype": str(current.dtype),
                "checkpoint_exact_match": True,
                "rms": core._tensor_rms(current),
            }

        encoder.eval().requires_grad_(False)
        self.mq_encoder = encoder
        load_report.update(
            {
                "conditioning_mode": 1,
                "context_order": "mapped_mq_then_t5",
                "wan_text_len_capacity": self.context_tokens,
                "restored_route_and_mapper_tensors": restored,
                "frozen_qwen_special_token_rows": {
                    "source": self.bundle.full_mq_path,
                    "embedding_key": self.bundle.full_embedding_key,
                    "shape": expected_special_shape,
                    "checkpoint_exact_match": True,
                },
                "qwen_frozen": not any(
                    parameter.requires_grad
                    for parameter in encoder.mllm_model.mllm_backbone.parameters()
                ),
                "t5_frozen": not any(
                    parameter.requires_grad
                    for parameter in self.wan.text_encoder.model.parameters()
                ),
            }
        )
        self.report["model_loading"]["mq_encoder_mode_one"] = load_report
        del state, model_state, checkpoint_special_rows, restored_special_rows
        gc.collect()

    def _encode_mode_one_context(
        self,
        prompt: str,
        image: Image.Image | None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        images = [[image]] if image is not None else None
        with torch.inference_mode():
            context = self.mq_encoder([prompt], images)
        mq_tokens = self.bundle.router_config.total_tokens
        hidden_size = int(self.bundle.config["wan_text_dim"])
        if (
            context.ndim != 3
            or int(context.shape[0]) != 1
            or int(context.shape[2]) != hidden_size
        ):
            raise RuntimeError(
                "Mode-1 context must be [1, variable_tokens, "
                f"{hidden_size}], got {tuple(context.shape)}"
            )
        actual_context_tokens = int(context.shape[1])
        self._validate_dit_context_token_count(
            actual_context_tokens,
            self.context_tokens,
        )
        actual_t5_tokens = actual_context_tokens - mq_tokens
        if not bool(torch.isfinite(context).all()) or core._tensor_rms(context) <= 0:
            raise RuntimeError("Mode-1 context is zero or contains NaN/Inf")
        audit = dict(getattr(self.mq_encoder, "last_context_audit", {}))
        expected_audit = {
            "mode": 1,
            "mq_tokens": mq_tokens,
            "t5_tokens": actual_t5_tokens,
            "context_tokens": actual_context_tokens,
            "hidden_size": hidden_size,
            "mapper_trainable": True,
        }
        mismatches = {
            key: {"actual": audit.get(key), "expected": value}
            for key, value in expected_audit.items()
            if audit.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Mode-1 context audit mismatch: {mismatches}")
        for key in ("mq_rms", "t5_rms", "mq_rms_match_scale"):
            _require_finite_float(audit.get(key), f"context audit {key}")
        scale = float(audit["mq_rms_match_scale"])
        if not 0.25 <= scale <= 4.0:
            raise RuntimeError(f"Mapper RMS match scale is out of range: {scale}")
        return context, audit

    def _build_contexts(
        self,
        prompt: str,
        negative_prompt: str,
        ref_image: Image.Image,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        positive, positive_audit = self._encode_mode_one_context(prompt, ref_image)
        self._verify_positive_route_audit()
        positive_routes = self._snapshot_routes(self.mq_encoder)
        no_image: torch.Tensor | None = None
        ablation_report: Dict[str, Any] | None = None

        if self.args.runtime_audit == "full":
            no_image_batch, no_image_audit = self._encode_mode_one_context(
                prompt, None
            )
            no_image_routes = self._snapshot_routes(self.mq_encoder)
            image_ablation = self._route_ablation_stats(
                positive_routes,
                no_image_routes,
                unchanged_routes={"action"},
                tag="image ablation",
            )
            no_text_batch, no_text_audit = self._encode_mode_one_context(
                "", ref_image
            )
            no_text_routes = self._snapshot_routes(self.mq_encoder)
            text_ablation = self._route_ablation_stats(
                positive_routes,
                no_text_routes,
                unchanged_routes={"role"},
                tag="text ablation",
            )
            mq_tokens = self.bundle.router_config.total_tokens
            t5_suffix_delta = core._tensor_difference_stats(
                positive[0, mq_tokens:],
                no_image_batch[0, mq_tokens:],
                label="T5 suffix after image ablation",
            )
            if float(t5_suffix_delta["diff_rms"]) != 0.0:
                raise RuntimeError(
                    "Removing the image changed frozen T5 prompt tokens"
                )
            final_image_ablation = core._tensor_difference_stats(
                positive[0],
                no_image_batch[0],
                label="image ablation in final mode-1 context",
            )
            if float(final_image_ablation["diff_rms"]) <= self.args.audit_epsilon:
                raise RuntimeError(
                    "Image conditioning collapsed in the final mode-1 context"
                )
            no_image = no_image_batch[0].to(
                device=self.dit_device,
                dtype=torch.bfloat16,
            )
            ablation_report = {
                "status": "pass",
                "routes_image_removed": image_ablation,
                "routes_text_removed": text_ablation,
                "final_image_ablation": final_image_ablation,
                "t5_suffix_image_invariant": t5_suffix_delta,
                "no_image_context_audit": no_image_audit,
                "no_text_context_audit": no_text_audit,
            }

        positive_context = positive[0].to(
            device=self.dit_device,
            dtype=torch.bfloat16,
        )
        unconditional_context: torch.Tensor | None = None
        unconditional_audit: Dict[str, Any] | None = None
        use_cfg = self.args.guide_scale > 1.0
        if use_cfg:
            unconditional_batch, unconditional_audit = (
                self._encode_mode_one_context("", None)
            )
            unconditional_context = unconditional_batch[0].to(
                device=self.dit_device,
                dtype=torch.bfloat16,
            )

        hidden_size = int(self.bundle.config["wan_text_dim"])
        for label, value in (
            ("positive", positive_context),
            ("unconditional", unconditional_context),
            ("no-image", no_image),
        ):
            if value is None:
                continue
            actual_tokens = int(value.shape[0]) if value.ndim == 2 else -1
            if (
                value.ndim != 2
                or int(value.shape[1]) != hidden_size
                or not self.bundle.router_config.total_tokens
                < actual_tokens
                <= self.context_tokens
            ):
                raise RuntimeError(
                    f"{label} context must be [257..{self.context_tokens}, "
                    f"{hidden_size}], got {tuple(value.shape)}"
                )

        self.report["runtime"]["mode_one_context"] = {
            "status": "pass",
            "order": "mapped_mq_then_frozen_t5",
            "positive": positive_audit,
            "unconditional": unconditional_audit,
            "mq_tokens_sent_to_dit": self.bundle.router_config.total_tokens,
            "positive_t5_tokens_sent_to_dit": int(positive_audit["t5_tokens"]),
            "positive_total_tokens_sent_to_dit": int(
                positive_audit["context_tokens"]
            ),
            "wan_text_len_capacity": self.context_tokens,
        }
        if ablation_report is not None:
            self.report["runtime"]["route_isolation_ablation"] = ablation_report
        self.report["runtime"]["cfg"] = {
            "enabled": use_cfg,
            "guide_scale": float(self.args.guide_scale),
            "unconditional_mode": "empty_mq" if use_cfg else "not_computed",
            "unconditional_contract": (
                "MQ('', None) + frozen T5('')" if use_cfg else None
            ),
            "negative_prompt_ignored": bool(negative_prompt.strip()),
            "training_aligned": True,
        }
        return (
            [positive_context],
            [unconditional_context]
            if unconditional_context is not None
            else None,
            no_image,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.parse_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "entrypoint": "openvid4000_3router_4x48g_mode1_strongbind",
                    "args": vars(args),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    bundle = validate_checkpoint_bundle(args.checkpoint_dir)
    if args.check_only:
        print(
            json.dumps(
                core._jsonable(bundle.report),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    ref_path = Path(args.ref_image).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not args.prompt.strip():
        raise ValueError("--prompt must be non-empty for full inference")
    output_path = Path(args.output_path).expanduser().resolve()
    report_path = (
        Path(args.verify_report_path).expanduser().resolve()
        if args.verify_report_path
        else Path(f"{output_path}.verify.json").resolve()
    )
    if output_path == report_path:
        raise ValueError("--output_path and --verify_report_path must differ")

    pipeline: ModeOneThreeRouterWanInference | None = None
    try:
        pipeline = ModeOneThreeRouterWanInference(args, bundle)
        pipeline.report["input"] = {
            "reference_image": ref_path,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "frame_num": int(args.frame_num),
            "max_area": int(args.max_area),
            "seed": int(args.seed),
        }
        with Image.open(ref_path) as handle:
            reference = handle.convert("RGB")
        video = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            ref_image=reference,
        )
        expected_size = tuple(
            int(value)
            for value in pipeline.report["runtime"]["generation"]["output_size"]
        )
        output_metadata = core._save_video(
            video,
            output_path,
            args.fps,
            expected_frame_num=args.frame_num,
            expected_size=expected_size,
        )
        pipeline.report["runtime"]["generation"]["status"] = "pass"
        pipeline.report["output"] = {
            "status": "pass",
            "video": output_path,
            "verify_report": report_path,
            "metadata": output_metadata,
        }
        pipeline.report["status"] = "pass"
        print(f"[DONE] video={output_path}")
        print(f"[DONE] verify_report={report_path}")
    except Exception as exc:
        if pipeline is not None:
            pipeline.report["status"] = "fail"
            pipeline.report["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        raise
    finally:
        if pipeline is not None:
            core._write_json(report_path, pipeline.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
