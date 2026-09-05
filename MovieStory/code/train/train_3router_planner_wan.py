#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
HOME_ROOT = PROJECT_ROOT.parents[1]
WAN_TRAIN_ROOT = (
    HOME_ROOT / "model" / "Wan2.2" / "scripts-metaquery-single" / "train"
)
WAN_ROOT = HOME_ROOT / "model" / "Wan2.2"
METAQUERY_ROOT = HOME_ROOT / "model" / "Qwen3-VL-main" / "metaquery-main"

for path in (
    str(CODE_ROOT),
    str(WAN_TRAIN_ROOT),
    str(WAN_ROOT),
    str(METAQUERY_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from three_router_planner import (  # noqa: E402
    StrongFirstFrameTrainingMixin,
    ThreeRouterConfig,
    ThreeRouterPlanner,
    build_three_router_encoder_class,
    configure_wan_first_frame_strong_binding,
)

ROUTER_DIAGNOSTIC_METRICS = {
    "role_action_cosine": "train/router_role_action_cosine",
    "role_global_cosine": "train/router_role_global_cosine",
    "action_global_cosine": "train/router_action_global_cosine",
    "role_rms": "train/router_role_rms",
    "action_rms": "train/router_action_rms",
    "global_rms": "train/router_global_rms",
    "role_mq_embedding_grad_rms": "train/router_role_mq_embedding_grad_rms",
    "action_mq_embedding_grad_rms": "train/router_action_mq_embedding_grad_rms",
    "global_mq_embedding_grad_rms": "train/router_global_mq_embedding_grad_rms",
}

ROUTE_NAMES = ("role", "action", "global")
VIDEO_GROUND_TRUTH_ONLY_LOSS = "video_ground_truth_velocity_mse_only"
for _route_name in ROUTE_NAMES:
    for _metric_suffix in (
        "param_rms",
        "step_grad_rms",
        "step_update_rms",
        "step_update_relative",
        "step_update_max_abs",
        "step_changed_fraction",
        "initial_delta_rms",
        "optimizer_lr",
        "update_expected",
        "update_applied",
        "stale_steps",
        "no_grad_steps",
    ):
        _metric_name = f"{_route_name}_{_metric_suffix}"
        ROUTER_DIAGNOSTIC_METRICS[_metric_name] = f"train/router_{_metric_name}"
ROUTER_DIAGNOSTIC_METRICS.update(
    {
        "optimizer_step": "train/router_optimizer_step",
        "all_updates_applied": "train/router_all_updates_applied",
    }
)


def build_router_wandb_config(
    base_config: Mapping[str, object],
    router_config: ThreeRouterConfig,
    *,
    enabled: bool,
    wan_first_frame_strong_bind: bool,
    joint_null_prob: float = 0.0,
) -> dict[str, object]:
    """Add MovieStory-specific settings to the inherited W&B run config."""
    config = dict(base_config)
    config.update(
        {
            "three_router_enabled": bool(enabled),
            "router_hidden_size": int(router_config.hidden_size),
            "router_role_tokens": int(router_config.role_tokens),
            "router_action_tokens": int(router_config.action_tokens),
            "router_global_tokens": int(router_config.global_tokens),
            "router_total_tokens": int(router_config.total_tokens),
            "router_routing_mode": "isolated_modalities_v1",
            "router_shared_connector": True,
            "wan_first_frame_strong_bind": bool(
                wan_first_frame_strong_bind
            ),
            "joint_null_prob": float(joint_null_prob),
            "loss_contract": VIDEO_GROUND_TRUTH_ONLY_LOSS,
        }
    )
    return config


def configure_video_ground_truth_only_loss(args: argparse.Namespace) -> None:
    """Make the 3-router loss contract explicit and fail-safe.

    T5 RMS probing/matching remains a conditioning normalization operation. It
    is not a loss. Every auxiliary loss switch and coefficient is disabled so
    the optimized scalar is only the generated-video versus ground-truth-video
    velocity MSE produced by the inherited trainer.
    """
    args.enable_t5_alignment = False
    args.lambda_t5_align_l2 = 0.0
    args.lambda_t5_align_cos = 0.0
    args.lambda_t5_align_stats = 0.0
    args.enable_mq_image_preserve = False
    args.lambda_mq_image_preserve = 0.0
    args.enable_wan_func_distill = False
    args.lambda_wan_func_distill = 0.0
    args.moviestory_loss_contract = VIDEO_GROUND_TRUTH_ONLY_LOSS


def build_joint_null_dataset_class(
    base_dataset_class,
    joint_null_prob: float,
):
    """Wrap the active Wan dataset with an explicit CFG-null distribution."""
    probability = float(joint_null_prob)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"--joint_null_prob must be finite and within [0, 1], got {probability}"
        )

    class JointNullWanVideoDataset(base_dataset_class):
        moviestory_joint_null_prob = probability

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            # Never mutate the base dataset's cached last-good sample.
            result = dict(sample)
            apply_joint_null = bool(
                probability > 0.0 and random.random() < probability
            )
            result["moviestory_joint_null"] = apply_joint_null
            if not apply_joint_null:
                return result
            result["caption"] = ""
            result["mq_ref_image"] = None
            # ref_image deliberately remains present: it is the independently
            # preserved Wan first-frame condition, matching inference CFG.
            return result

    JointNullWanVideoDataset.__name__ = (
        f"JointNull{base_dataset_class.__name__}"
    )
    return JointNullWanVideoDataset


def configure_wandb_metrics(run) -> None:
    """Use optimizer step as the W&B x-axis and retain useful summaries."""
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("train/loss_step", summary="min")
    run.define_metric("train/loss_ema", summary="min")
    run.define_metric("train/grad_norm", summary="max")
    for route_name in ROUTE_NAMES:
        run.define_metric(
            f"train/router_{route_name}_initial_delta_rms",
            summary="max",
        )
    run.define_metric("train/router_all_updates_applied", summary="mean")


class RouterParameterUpdateTracker:
    """Measure real optimizer updates for each route MetaQuery table."""

    def __init__(
        self,
        parameters: Mapping[str, nn.Parameter],
        *,
        stale_update_patience: int = 5,
    ) -> None:
        missing = [name for name in ROUTE_NAMES if name not in parameters]
        if missing:
            raise ValueError(f"missing route parameters: {missing}")
        self.parameters = {name: parameters[name] for name in ROUTE_NAMES}
        self.stale_update_patience = max(int(stale_update_patience), 0)
        self.optimizer_step = 0
        self._initial = {
            name: parameter.detach().float().clone()
            for name, parameter in self.parameters.items()
        }
        self._before: dict[str, torch.Tensor] = {}
        self._step_grad_rms: dict[str, float] = {}
        self._step_lr: dict[str, float] = {}
        self._stale_steps = {name: 0 for name in ROUTE_NAMES}
        self._no_grad_steps = {name: 0 for name in ROUTE_NAMES}
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _rms(value: torch.Tensor) -> float:
        return float(value.float().square().mean().sqrt().item())

    @staticmethod
    def _parameter_group(
        optimizer: torch.optim.Optimizer,
        parameter: nn.Parameter,
    ) -> dict:
        groups = [
            group
            for group in optimizer.param_groups
            if any(candidate is parameter for candidate in group["params"])
        ]
        if len(groups) != 1:
            raise RuntimeError(
                "each 3-router parameter must occur in exactly one optimizer group; "
                f"found {len(groups)}"
            )
        return groups[0]

    def before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *_,
    ) -> None:
        self._before = {}
        self._step_grad_rms = {}
        self._step_lr = {}
        for name, parameter in self.parameters.items():
            self._before[name] = parameter.detach().float().clone()
            grad = parameter.grad
            if grad is not None and not torch.isfinite(grad).all():
                raise FloatingPointError(
                    f"3-router gradient '{name}' became non-finite"
                )
            self._step_grad_rms[name] = (
                self._rms(grad.detach()) if grad is not None else 0.0
            )
            group = self._parameter_group(optimizer, parameter)
            self._step_lr[name] = float(group["lr"])

    def after_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *_,
    ) -> None:
        del optimizer
        if not self._before:
            raise RuntimeError("3-router optimizer post-hook ran without a pre-hook")

        self.optimizer_step += 1
        metrics: dict[str, float] = {
            "train/router_optimizer_step": float(self.optimizer_step),
        }
        all_positive_lr_routes_verified = True
        any_positive_lr = False
        for name, parameter in self.parameters.items():
            current = parameter.detach().float()
            if not torch.isfinite(current).all():
                raise FloatingPointError(
                    f"3-router parameter '{name}' became non-finite"
                )
            before = self._before[name]
            update = current - before
            initial_delta = current - self._initial[name]
            param_rms = self._rms(current)
            update_rms = self._rms(update)
            update_max_abs = float(update.abs().max().item())
            changed_fraction = float(
                (update != 0).float().mean().item()
            )
            grad_rms = self._step_grad_rms[name]
            lr = self._step_lr[name]
            update_expected = bool(lr > 0.0 and grad_rms > 0.0)
            update_applied = bool(update_max_abs > 0.0)
            any_positive_lr |= lr > 0.0

            if update_expected and not update_applied:
                self._stale_steps[name] += 1
            else:
                self._stale_steps[name] = 0
            if lr > 0.0 and grad_rms <= 0.0:
                self._no_grad_steps[name] += 1
            else:
                self._no_grad_steps[name] = 0
            if lr > 0.0 and (grad_rms <= 0.0 or not update_applied):
                all_positive_lr_routes_verified = False

            prefix = f"train/router_{name}_"
            metrics.update(
                {
                    prefix + "param_rms": param_rms,
                    prefix + "step_grad_rms": grad_rms,
                    prefix + "step_update_rms": update_rms,
                    prefix + "step_update_relative": (
                        update_rms / max(self._rms(before), 1e-30)
                    ),
                    prefix + "step_update_max_abs": update_max_abs,
                    prefix + "step_changed_fraction": changed_fraction,
                    prefix + "initial_delta_rms": self._rms(initial_delta),
                    prefix + "optimizer_lr": lr,
                    prefix + "update_expected": float(update_expected),
                    prefix + "update_applied": float(update_applied),
                    prefix + "stale_steps": float(self._stale_steps[name]),
                    prefix + "no_grad_steps": float(
                        self._no_grad_steps[name]
                    ),
                }
            )

            if (
                self.stale_update_patience > 0
                and self._no_grad_steps[name] >= self.stale_update_patience
            ):
                raise RuntimeError(
                    "[3-ROUTER][NO-GRAD] "
                    f"{name} had lr={lr:.3e} but no gradient for "
                    f"{self._no_grad_steps[name]} consecutive optimizer steps"
                )
            if (
                self.stale_update_patience > 0
                and self._stale_steps[name] >= self.stale_update_patience
            ):
                raise RuntimeError(
                    "[3-ROUTER][STALE] "
                    f"{name} had a finite non-zero gradient and lr={lr:.3e}, "
                    f"but its parameter did not change for "
                    f"{self._stale_steps[name]} consecutive optimizer steps"
                )

        metrics["train/router_all_updates_applied"] = float(
            any_positive_lr and all_positive_lr_routes_verified
        )
        self.last_metrics = metrics


def router_diagnostics_to_metrics(
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Convert the latest per-sample router diagnostics to batch-mean scalars."""
    metrics: dict[str, float] = {}
    with torch.no_grad():
        for diagnostic_name, metric_name in ROUTER_DIAGNOSTIC_METRICS.items():
            value = diagnostics.get(diagnostic_name)
            if value is None or not torch.is_tensor(value):
                continue
            finite_values = value.detach().float().flatten()
            finite_values = finite_values[torch.isfinite(finite_values)]
            if finite_values.numel() > 0:
                metrics[metric_name] = float(finite_values.mean().item())
    return metrics


def move_parameters_to_zero_weight_decay_group(
    optimizer,
    scheduler,
    parameters,
    *,
    group_name: str,
) -> bool:
    """Move selected parameters into one scheduler-compatible AdamW group."""
    selected_parameters = list(parameters)
    selected_ids = {id(parameter) for parameter in selected_parameters}
    if not selected_ids:
        return False

    for group_index, group in enumerate(optimizer.param_groups):
        params = list(group["params"])
        selected = [
            parameter for parameter in params if id(parameter) in selected_ids
        ]
        if not selected:
            continue
        remaining = [
            parameter for parameter in params if id(parameter) not in selected_ids
        ]
        if not remaining:
            group["weight_decay"] = 0.0
            group["name"] = group_name
            return True

        group["params"] = remaining
        new_group = {key: value for key, value in group.items() if key != "params"}
        new_group.update(
            {
                "name": group_name,
                "params": selected,
                "weight_decay": 0.0,
            }
        )
        optimizer.add_param_group(new_group)

        # LambdaLR was created before this group split.
        scheduler.base_lrs.append(
            float(new_group.get("initial_lr", new_group["lr"]))
        )
        if hasattr(scheduler, "lr_lambdas"):
            scheduler.lr_lambdas.append(scheduler.lr_lambdas[group_index])
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr.append(float(new_group["lr"]))
        return True
    return False


def parse_router_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--router_hidden_size", type=int, default=2048)
    parser.add_argument("--router_role_tokens", type=int, default=96)
    parser.add_argument("--router_action_tokens", type=int, default=96)
    parser.add_argument("--router_global_tokens", type=int, default=64)
    parser.add_argument("--disable_3router", action="store_true")
    parser.add_argument(
        "--router_log_steps",
        type=int,
        default=1,
        help="Print 3-router forward/gradient/update evidence every N optimizer steps.",
    )
    parser.add_argument(
        "--router_stale_update_patience",
        type=int,
        default=5,
        help=(
            "Abort after N consecutive steps with non-zero route gradient/LR but "
            "no parameter change; 0 disables the guard."
        ),
    )
    first_frame_group = parser.add_mutually_exclusive_group()
    first_frame_group.add_argument(
        "--wan_first_frame_strong_bind",
        dest="wan_first_frame_strong_bind",
        action="store_true",
        default=True,
        help=(
            "Use ref_image as a clean timestep-zero Wan latent prefix, remove "
            "the duplicate first target slot, and exclude the prefix from loss "
            "(default; flow-consistent I2V conditioning)."
        ),
    )
    first_frame_group.add_argument(
        "--disable_wan_first_frame_strong_bind",
        dest="wan_first_frame_strong_bind",
        action="store_false",
        help=(
            "Ablation only: use ordinary legacy_t2v flow matching with no "
            "first-frame latent anchor. The inconsistent soft anchor is disabled."
        ),
    )
    parser.add_argument(
        "--joint_null_prob",
        type=float,
        default=0.1,
        help=(
            "Probability of forcing caption='' and mq_ref_image=None together. "
            "This trains the inference CFG null branch without adding a loss."
        ),
    )
    parser.add_argument(
        "--router_check_only",
        action="store_true",
        help="Run a CPU shape/backprop check without loading Qwen or Wan.",
    )
    parser.add_argument(
        "--router_parse_only",
        action="store_true",
        help="Validate planner and inherited Wan CLI arguments without loading models.",
    )
    return parser.parse_known_args(list(argv))


def build_config(router_args: argparse.Namespace) -> ThreeRouterConfig:
    return ThreeRouterConfig(
        hidden_size=router_args.router_hidden_size,
        role_tokens=router_args.router_role_tokens,
        action_tokens=router_args.router_action_tokens,
        global_tokens=router_args.router_global_tokens,
    )


def run_check_only(config: ThreeRouterConfig) -> None:
    torch.manual_seed(7)
    planner = ThreeRouterPlanner(config)
    route_metaqueries = nn.ParameterDict(
        {
            route_name: nn.Parameter(
                torch.randn(end - start, config.hidden_size)
            )
            for route_name, (start, end) in config.route_slices.items()
        }
    )
    seed = torch.cat(
        [route_metaqueries[name] for name in ROUTE_NAMES],
        dim=0,
    ).unsqueeze(0).expand(2, -1, -1)
    result = planner(seed)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "route_metaquery_embeddings",
                "params": list(route_metaqueries.parameters()),
                "lr": 1e-5,
                "weight_decay": 0.0,
            }
        ]
    )
    tracker = RouterParameterUpdateTracker(route_metaqueries)
    optimizer.register_step_pre_hook(tracker.before_optimizer_step)
    optimizer.register_step_post_hook(tracker.after_optimizer_step)
    loss = result.tokens.square().mean()
    loss.backward()
    gradients_finite = {
        route_name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
        )
        for route_name, parameter in route_metaqueries.items()
    }
    optimizer.step()
    update_evidence = {
        route_name: {
            "grad_rms": tracker.last_metrics[
                f"train/router_{route_name}_step_grad_rms"
            ],
            "update_rms": tracker.last_metrics[
                f"train/router_{route_name}_step_update_rms"
            ],
            "changed_fraction": tracker.last_metrics[
                f"train/router_{route_name}_step_changed_fraction"
            ],
            "updated": bool(
                tracker.last_metrics[
                    f"train/router_{route_name}_update_applied"
                ]
            ),
        }
        for route_name in ROUTE_NAMES
    }
    report = {
        "status": (
            "ok"
            if all(item["updated"] for item in update_evidence.values())
            else "failed"
        ),
        "config": config.to_dict(),
        "output_shape": list(result.tokens.shape),
        "route_shapes": {
            "role": list(result.role.shape),
            "action": list(result.action.shape),
            "global": list(result.global_route.shape),
        },
        "route_gradients_finite": gradients_finite,
        "route_optimizer_update_evidence": update_evidence,
        "all_route_updates_applied": bool(
            tracker.last_metrics["train/router_all_updates_applied"]
        ),
        "planner_trainable_parameters": sum(
            parameter.numel() for parameter in planner.parameters()
        ),
        "route_metaquery_trainable_parameters": sum(
            parameter.numel() for parameter in route_metaqueries.parameters()
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise RuntimeError("3-router optimizer update self-check failed")


def _write_router_metadata(
    checkpoint_path: Path,
    config: ThreeRouterConfig,
    enabled: bool,
    trainable_metaquery_embeddings: bool,
    wan_first_frame_strong_bind: bool,
    router_log_steps: int,
    router_stale_update_patience: int,
    training_args: argparse.Namespace,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    no_first_frame_anchor = not bool(wan_first_frame_strong_bind)
    payload = {
        "format": "moviestory_three_router_planner_v6_flow_consistent",
        "enabled": bool(enabled),
        "routing_mode": "isolated_modalities_v1",
        "route_modalities": {
            "role": "reference_image",
            "action": "text",
            "global": "reference_image+text",
        },
        "shared_route_connector": True,
        "separate_connector_forwards": False,
        "joint_connector_forward": bool(enabled),
        "post_qwen_router_transform": "identity_split_only",
        "qwen_outputs_direct_to_connector": bool(enabled),
        "trainable_route_metaquery_parameters": bool(enabled),
        "route_metaquery_master_dtype": "float32" if enabled else None,
        "runtime_update_verification": {
            "enabled": bool(enabled),
            "console_log_steps": int(router_log_steps),
            "stale_update_patience": int(router_stale_update_patience),
            "checks": [
                "optimizer_membership",
                "requires_grad",
                "finite_gradient",
                "finite_parameter",
                "step_parameter_delta",
                "cumulative_parameter_delta",
            ],
        },
        "qwen_input_embedding_table_trainable": bool(
            trainable_metaquery_embeddings
        ),
        "connector": {
            "type": "Qwen2Encoder",
            "num_hidden_layers": int(
                training_args.connector_num_hidden_layers
            ),
        },
        "mq_t5_rms": {
            "probe_enabled": bool(training_args.mq_norm_probe_with_t5),
            "match_enabled": bool(training_args.mq_norm_match_t5),
        },
        "conditioning_dropout": {
            "image": float(training_args.null_image_prob),
            "text": float(training_args.null_caption_prob),
            "joint_null": float(training_args.joint_null_prob),
            "joint_null_contract": {
                "caption": "",
                "mq_ref_image": None,
                "wan_ref_image_preserved": bool(wan_first_frame_strong_bind),
            },
        },
        "loss_contract": {
            "name": VIDEO_GROUND_TRUTH_ONLY_LOSS,
            "optimized_terms": ["video_ground_truth_velocity_mse"],
            "t5_alignment_loss": False,
            "mq_image_preserve_loss": False,
            "wan_function_distillation_loss": False,
            "preserved_reference_prefix_in_loss": False,
        },
        "wan_first_frame_conditioning": {
            "enabled": bool(training_args.enable_ti2v_first_frame_condition),
            "mode": (
                "clean_preserved_latent_slot"
                if wan_first_frame_strong_bind
                else "legacy_t2v_no_anchor"
            ),
            "source": (
                "ref_image"
                if wan_first_frame_strong_bind
                else "none"
            ),
            "latent_slots": 1 if wan_first_frame_strong_bind else 0,
            "timestep_zero": bool(wan_first_frame_strong_bind),
            "shares_video_random_timestep": bool(no_first_frame_anchor),
            "remains_noisy_denoising_input": False,
            "soft_anchor_mode": "none",
            "soft_anchor_alpha0": 0.0,
            "soft_anchor_warmup_ratio": 0.0,
            "excluded_from_denoising_loss": bool(
                wan_first_frame_strong_bind
            ),
            "original_first_target_slot_removed": bool(
                wan_first_frame_strong_bind
            ),
            "original_first_target_slot_retained": bool(no_first_frame_anchor),
        },
        **config.to_dict(),
    }
    target = checkpoint_path / "three_router_config.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_training(
    base_argv: Sequence[str],
    router_args: argparse.Namespace,
    config: ThreeRouterConfig,
) -> None:
    if not WAN_TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Wan training root not found: {WAN_TRAIN_ROOT}")
    if not METAQUERY_ROOT.exists():
        raise FileNotFoundError(f"MetaQuery root not found: {METAQUERY_ROOT}")

    import train_connector_for_wan as connector_module
    import train_metaquery_wan as base_train

    enabled = not router_args.disable_3router
    patched_encoder = build_three_router_encoder_class(
        connector_module.MetaQueryEncoderForWan,
        config,
        enabled=enabled,
    )
    connector_module.MetaQueryEncoderForWan = patched_encoder

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *base_argv]
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    if int(args.num_metaqueries) != config.total_tokens:
        raise ValueError(
            f"--num_metaqueries must be {config.total_tokens} for 3-router, "
            f"got {args.num_metaqueries}"
        )
    if int(args.connector_num_hidden_layers) != 24:
        raise ValueError(
            "--connector_num_hidden_layers must be 24 for the baseline-aligned "
            f"3-router Connector, got {args.connector_num_hidden_layers}"
        )
    if int(config.hidden_size) <= 0:
        raise ValueError("router_hidden_size must be positive")
    if (
        not math.isfinite(float(router_args.joint_null_prob))
        or not 0.0 <= float(router_args.joint_null_prob) <= 1.0
    ):
        raise ValueError("--joint_null_prob must be finite and within [0, 1]")

    # Three independent FP32 route tables replace the MetaQuery rows at
    # runtime.  The original Qwen embedding table is an immutable semantic
    # reference and must never be added to the optimizer in 3-router mode.
    if enabled:
        args.train_mq_input_embeddings = False

    # Persist planner settings through the existing training_args checkpoint files.
    for key, value in vars(router_args).items():
        setattr(args, key, value)
    args.router_config = config.to_dict()
    args.three_router_enabled = enabled
    configure_video_ground_truth_only_loss(args)
    configure_wan_first_frame_strong_binding(
        args,
        enabled=router_args.wan_first_frame_strong_bind,
    )
    if router_args.router_parse_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "three_router_enabled": enabled,
                    "router_config": config.to_dict(),
                    "num_metaqueries": int(args.num_metaqueries),
                    "connector_num_hidden_layers": int(
                        args.connector_num_hidden_layers
                    ),
                    "output_dir": str(args.output_dir),
                    "local_openvid_limit": args.local_openvid_limit,
                    "wan_train_mode": str(args.wan_train_mode),
                    "qwen_input_embedding_table_trainable": bool(
                        args.train_mq_input_embeddings
                    ),
                    "mq_t5_rms": {
                        "probe_enabled": bool(args.mq_norm_probe_with_t5),
                        "match_enabled": bool(args.mq_norm_match_t5),
                    },
                    "conditioning_dropout": {
                        "image": float(args.null_image_prob),
                        "text": float(args.null_caption_prob),
                        "joint_null": float(args.joint_null_prob),
                    },
                    "loss_contract": str(args.moviestory_loss_contract),
                    "wan_first_frame_conditioning": {
                        "strong_bind": bool(
                            args.moviestory_wan_first_frame_strong_bind
                        ),
                        "mode": str(args.train_video_conditioning_mode),
                        "reference_frames": int(
                            args.train_animate_ref_frames
                        ),
                        "temporal_frames": int(
                            args.train_animate_temporal_frames
                        ),
                        "conditional_frames": int(
                            args.train_animate_conditional_frames
                        ),
                        "preserve_timestep_zero": bool(
                            args.train_animate_preserve_timestep_zero
                        ),
                        "drop_prefix_loss": bool(
                            args.train_animate_drop_prefix_loss
                        ),
                        "soft_anchor_mode": str(
                            args.train_ref_anchor_mode
                        ),
                        "soft_anchor_alpha0": float(
                            args.train_ref_anchor_alpha0
                        ),
                        "soft_anchor_warmup_ratio": float(
                            args.train_ref_anchor_warmup_ratio
                        ),
                        "original_first_latent_retained": not bool(
                            args.moviestory_wan_first_frame_strong_bind
                        ),
                        "shares_video_random_timestep": not bool(
                            args.moviestory_wan_first_frame_strong_bind
                        ),
                        "included_in_denoising_loss": not bool(
                            args.moviestory_wan_first_frame_strong_bind
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if base_train.WanDatasetClass is None:
        raise RuntimeError("WanVideoDataset is unavailable for 3-router training")
    base_train.WanDatasetClass = build_joint_null_dataset_class(
        base_train.WanDatasetClass,
        float(args.joint_null_prob),
    )
    base_trainer_class = base_train.MetaQueryWanTrainer

    class ThreeRouterWanTrainer(
        StrongFirstFrameTrainingMixin,
        base_trainer_class,
    ):
        def _wandb_config(self):
            return build_router_wandb_config(
                super()._wandb_config(),
                config,
                enabled=enabled,
                wan_first_frame_strong_bind=bool(
                    self.args.moviestory_wan_first_frame_strong_bind
                ),
                joint_null_prob=float(self.args.joint_null_prob),
            )

        def _compute_loss(self, batch):
            loss = super()._compute_loss(batch)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError(
                    "3-router loss must be one scalar ground-truth video loss"
                )
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError("3-router video denoising loss is non-finite")

            denoise_value = float(getattr(self, "_last_loss_denoise", math.nan))
            loss_value = float(loss.detach().item())
            if (
                not math.isfinite(denoise_value)
                or not math.isclose(
                    loss_value,
                    denoise_value,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
            ):
                raise RuntimeError(
                    "Loss contract violation: total loss differs from the "
                    "ground-truth video denoising loss"
                )
            auxiliary_metrics = (
                "_last_loss_aux_align_total",
                "_last_loss_aux_t5_l2",
                "_last_loss_aux_t5_cos",
                "_last_loss_aux_t5_stats",
                "_last_loss_aux_t5_gram",
                "_last_loss_aux_t5_cka",
                "_last_loss_aux_t5_ot",
                "_last_loss_aux_image_preserve",
                "_last_loss_aux_wan_func",
            )
            nonzero_auxiliary = {
                name: float(getattr(self, name, 0.0))
                for name in auxiliary_metrics
                if float(getattr(self, name, 0.0)) != 0.0
            }
            if nonzero_auxiliary:
                raise RuntimeError(
                    "Loss contract violation: auxiliary losses are non-zero: "
                    f"{nonzero_auxiliary}"
                )
            return loss

        def _init_wandb(self):
            super()._init_wandb()
            if self.wandb_run is not None:
                configure_wandb_metrics(self.wandb_run)
                print(
                    "[W&B] optimizer-step metric axis and 3-router summaries "
                    "configured",
                    flush=True,
                )

        def _setup_optimizer(self):
            super()._setup_optimizer()
            self._router_update_tracker = None
            self._router_optimizer_hook_handles = []
            module = self._mq_encoder_module()
            route_metaqueries = getattr(
                module,
                "route_metaquery_embeddings",
                None,
            )
            if route_metaqueries is None:
                return
            route_parameters = list(route_metaqueries.parameters())
            moved = move_parameters_to_zero_weight_decay_group(
                self.optimizer,
                self.scheduler,
                route_parameters,
                group_name="route_metaquery_embeddings",
            )
            if moved and self.is_main_process:
                print(
                    "[3-ROUTER][OPT] Route MetaQuery parameters use an isolated "
                    "optimizer group with weight_decay=0",
                    flush=True,
                )
            self._audit_and_track_route_parameters(route_metaqueries)

        def _audit_and_track_route_parameters(self, route_metaqueries):
            for route_name in ROUTE_NAMES:
                parameter = route_metaqueries[route_name]
                locations = [
                    group
                    for group in self.optimizer.param_groups
                    if any(candidate is parameter for candidate in group["params"])
                ]
                if not parameter.requires_grad:
                    raise RuntimeError(
                        f"3-router parameter '{route_name}' is frozen"
                    )
                if parameter.dtype != torch.float32:
                    raise RuntimeError(
                        f"3-router parameter '{route_name}' must use FP32 master "
                        f"weights, got {parameter.dtype}"
                    )
                if len(locations) != 1:
                    raise RuntimeError(
                        f"3-router parameter '{route_name}' occurs in "
                        f"{len(locations)} optimizer groups"
                    )
                group = locations[0]
                if float(group.get("weight_decay", 0.0)) != 0.0:
                    raise RuntimeError(
                        f"3-router parameter '{route_name}' must use weight_decay=0"
                    )
                if self.is_main_process:
                    print(
                        "[3-ROUTER][AUDIT] "
                        f"route={route_name} shape={tuple(parameter.shape)} "
                        f"dtype={parameter.dtype} requires_grad=1 "
                        f"optimizer_group={group.get('name', '<unnamed>')} "
                        f"lr={float(group['lr']):.3e} "
                        f"weight_decay={float(group.get('weight_decay', 0.0)):.1f}",
                        flush=True,
                    )

            self._router_update_tracker = RouterParameterUpdateTracker(
                route_metaqueries,
                stale_update_patience=self.args.router_stale_update_patience,
            )
            self._router_optimizer_hook_handles = [
                self.optimizer.register_step_pre_hook(
                    self._router_update_tracker.before_optimizer_step
                ),
                self.optimizer.register_step_post_hook(
                    self._router_update_tracker.after_optimizer_step
                ),
            ]
            if self.is_main_process:
                print(
                    "[3-ROUTER][AUDIT][PASS] all routes are trainable FP32 "
                    "parameters in exactly one optimizer group; live step-delta "
                    "tracking is active",
                    flush=True,
                )

        def _collect_router_metrics(self) -> dict[str, float]:
            module = self._mq_encoder_module()
            diagnostics = {}
            forward_diagnostics = getattr(module, "last_router_diagnostics", {})
            if isinstance(forward_diagnostics, dict):
                diagnostics.update(forward_diagnostics)
            gradient_diagnostics = getattr(
                module,
                "last_route_embedding_grad_rms",
                {},
            )
            if isinstance(gradient_diagnostics, dict):
                diagnostics.update(gradient_diagnostics)
            metrics = router_diagnostics_to_metrics(diagnostics)
            tracker = getattr(self, "_router_update_tracker", None)
            if tracker is not None:
                metrics.update(tracker.last_metrics)
            return metrics

        def _collect_trainability_metrics(self):
            metrics = super()._collect_trainability_metrics()
            metrics.update(self._collect_router_metrics())
            return metrics

        def _record_metrics(self, metrics):
            super()._record_metrics(metrics)
            router_metrics = {
                key: value
                for key, value in metrics.items()
                if key in ROUTER_DIAGNOSTIC_METRICS.values()
            }
            if self._metrics_history:
                self._metrics_history[-1].update(router_metrics)

            step = int(metrics.get("train/step", 0))
            router_log_steps = max(int(self.args.router_log_steps), 1)
            should_log = bool(
                step > 0 and step % router_log_steps == 0
            )
            console_metric_names = (
                "train/router_role_action_cosine",
                "train/router_role_global_cosine",
                "train/router_action_global_cosine",
                "train/router_role_rms",
                "train/router_action_rms",
                "train/router_global_rms",
            )
            complete = all(name in router_metrics for name in console_metric_names)
            if complete and self.is_main_process and should_log:
                grad_metrics = (
                    router_metrics.get(
                        "train/router_role_mq_embedding_grad_rms",
                        float("nan"),
                    ),
                    router_metrics.get(
                        "train/router_action_mq_embedding_grad_rms",
                        float("nan"),
                    ),
                    router_metrics.get(
                        "train/router_global_mq_embedding_grad_rms",
                        float("nan"),
                    ),
                )
                print(
                    "[3-ROUTER][DIAG] "
                    f"step={step} "
                    f"cos(role,action)={router_metrics['train/router_role_action_cosine']:.4f} "
                    f"cos(role,global)={router_metrics['train/router_role_global_cosine']:.4f} "
                    f"cos(action,global)={router_metrics['train/router_action_global_cosine']:.4f} "
                    f"rms=({router_metrics['train/router_role_rms']:.4f},"
                    f"{router_metrics['train/router_action_rms']:.4f},"
                    f"{router_metrics['train/router_global_rms']:.4f}) "
                    f"mq_emb_grad=({grad_metrics[0]:.3e},"
                    f"{grad_metrics[1]:.3e},{grad_metrics[2]:.3e})",
                    flush=True,
                )
                tracker = getattr(self, "_router_update_tracker", None)
                if tracker is not None and tracker.last_metrics:
                    update_rows = []
                    all_updated = True
                    any_positive_lr = False
                    all_have_grad = True
                    for route_name in ROUTE_NAMES:
                        prefix = f"train/router_{route_name}_"
                        lr = router_metrics[prefix + "optimizer_lr"]
                        grad = router_metrics[prefix + "step_grad_rms"]
                        expected = bool(
                            router_metrics[prefix + "update_expected"]
                        )
                        applied = bool(
                            router_metrics[prefix + "update_applied"]
                        )
                        any_positive_lr |= lr > 0.0
                        all_have_grad &= (lr <= 0.0 or grad > 0.0)
                        all_updated &= (not expected or applied)
                        update_rows.append(
                            f"{route_name}:"
                            f"p={router_metrics[prefix + 'param_rms']:.3e},"
                            f"g={router_metrics[prefix + 'step_grad_rms']:.3e},"
                            f"d={router_metrics[prefix + 'step_update_rms']:.3e},"
                            f"rel={router_metrics[prefix + 'step_update_relative']:.3e},"
                            f"changed={router_metrics[prefix + 'step_changed_fraction']:.1%},"
                            f"d_init={router_metrics[prefix + 'initial_delta_rms']:.3e},"
                            f"lr={router_metrics[prefix + 'optimizer_lr']:.2e}"
                        )
                    status = (
                        "PASS"
                        if any_positive_lr and all_have_grad and all_updated
                        else "WAIT_LR"
                        if not any_positive_lr
                        else "FAIL_NO_GRAD"
                        if not all_have_grad
                        else "FAIL_STALE"
                    )
                    print(
                        f"[3-ROUTER][UPDATE] step={step} status={status} "
                        + " | ".join(update_rows),
                        flush=True,
                    )

        def _build_metrics_summary(self, step):
            summary = super()._build_metrics_summary(step)
            for metric_name in ROUTER_DIAGNOSTIC_METRICS.values():
                values = [
                    float(row[metric_name])
                    for row in self._metrics_history
                    if metric_name in row
                ]
                if not values:
                    continue
                summary_name = metric_name.removeprefix("train/")
                summary[f"{summary_name}_last"] = values[-1]
                summary[f"{summary_name}_avg"] = sum(values) / len(values)
            return summary

        def _save_checkpoint(self, path, step, extra_info=None):
            result = super()._save_checkpoint(path, step, extra_info=extra_info)
            if getattr(self, "is_main_process", True):
                _write_router_metadata(
                    Path(path).expanduser().resolve(),
                    config,
                    enabled,
                    bool(self.args.train_mq_input_embeddings),
                    bool(self.args.moviestory_wan_first_frame_strong_bind),
                    int(self.args.router_log_steps),
                    int(self.args.router_stale_update_patience),
                    self.args,
                )
            return result

    print(
        "[3-ROUTER] "
        f"enabled={int(enabled)} layout=({config.role_tokens},"
        f"{config.action_tokens},{config.global_tokens}) "
        f"hidden={config.hidden_size} output_dir={args.output_dir}"
    )
    print(
        "[WAN-FIRST-FRAME] "
        f"strong_bind={int(args.moviestory_wan_first_frame_strong_bind)} "
        f"source={'ref_image' if args.moviestory_wan_first_frame_strong_bind else 'mq_ref_image'} "
        f"mode={args.train_video_conditioning_mode} "
        f"ref_slots={args.train_animate_ref_frames} "
        f"timestep_zero={int(args.train_animate_preserve_timestep_zero)} "
        f"drop_prefix_loss={int(args.train_animate_drop_prefix_loss)} "
        f"soft_anchor={args.train_ref_anchor_mode}"
    )
    print(
        "[3-ROUTER][LOSS] "
        f"contract={args.moviestory_loss_contract} "
        f"joint_null_prob={float(args.joint_null_prob):.3f} "
        "auxiliary_losses=0",
        flush=True,
    )
    trainer = ThreeRouterWanTrainer(args)
    trainer.train()


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    router_args, base_argv = parse_router_args(argv)
    config = build_config(router_args)
    if router_args.router_check_only:
        run_check_only(config)
        return
    run_training(base_argv, router_args, config)


if __name__ == "__main__":
    main()
