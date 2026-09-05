#!/usr/bin/env python3
"""MovieStory 3-router training for exactly four 48 GiB GPUs.

The legacy 2 x 96 GiB entrypoint remains untouched.  This implementation uses
four data-parallel ranks, FULL_SHARD for Wan, DDP for MQ/Connector, an equivalent
global batch stream, two conditioning modes, and strong-bound first-frame
references.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Type

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


TRAIN_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TRAIN_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
HOME_ROOT = PROJECT_ROOT.parents[1]
WAN_TRAIN_ROOT = (
    HOME_ROOT / "model" / "Wan2.2" / "scripts-metaquery-single" / "train"
)
WAN_ROOT = HOME_ROOT / "model" / "Wan2.2"
METAQUERY_ROOT = HOME_ROOT / "model" / "Qwen3-VL-main" / "metaquery-main"

for _path in (TRAIN_ROOT, CODE_ROOT, WAN_TRAIN_ROOT, WAN_ROOT, METAQUERY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from four_gpu_training import (  # noqa: E402
    GlobalBatchEquivalentSampler,
    MQToT5Mapper,
    build_dual_mode_encoder_class,
    clip_grad_norm_mixed_sharded_,
    install_equivalent_dataloader,
    install_wan_fsdp_sharder,
)
from three_router_planner import (  # noqa: E402
    StrongFirstFrameTrainingMixin,
    ThreeRouterConfig,
    build_three_router_encoder_class,
    configure_wan_first_frame_strong_binding,
    video_start_frame_to_reference_image,
)
from train_3router_planner_wan import (  # noqa: E402
    RouterParameterUpdateTracker,
    configure_video_ground_truth_only_loss,
    move_parameters_to_zero_weight_decay_group,
    router_diagnostics_to_metrics,
)


FORMAT_NAME = "moviestory_4x48g_first_frame_strongbind_v1"


class DistributedTrainingAbort(BaseException):
    """Bypass the legacy trainer's skip-and-continue exception handler."""


def parse_four_gpu_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conditioning_mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expected_world_size", type=int, default=4)
    parser.add_argument("--global_effective_batch", type=int, default=8)
    parser.add_argument("--expected_train_samples", type=int, default=4000)
    parser.add_argument("--minimum_gpu_memory_gib", type=float, default=44.0)
    parser.add_argument("--minimum_free_gpu_memory_gib", type=float, default=42.0)
    parser.add_argument("--mapper_bottleneck_size", type=int, default=1024)
    parser.add_argument("--mapper_residual_scale", type=float, default=0.1)
    parser.add_argument(
        "--disable_mapper_rms_match", action="store_true", default=False
    )
    parser.add_argument("--router_hidden_size", type=int, default=2048)
    parser.add_argument("--router_role_tokens", type=int, default=96)
    parser.add_argument("--router_action_tokens", type=int, default=96)
    parser.add_argument("--router_global_tokens", type=int, default=64)
    parser.add_argument("--disable_3router", action="store_true")
    parser.add_argument("--router_log_steps", type=int, default=1)
    parser.add_argument("--router_stale_update_patience", type=int, default=5)
    parser.add_argument("--joint_null_prob", type=float, default=0.1)
    strong_bind_group = parser.add_mutually_exclusive_group()
    strong_bind_group.add_argument(
        "--wan_first_frame_strong_bind",
        dest="wan_first_frame_strong_bind",
        action="store_true",
        default=True,
    )
    strong_bind_group.add_argument(
        "--disable_wan_first_frame_strong_bind",
        dest="wan_first_frame_strong_bind",
        action="store_false",
    )
    parser.add_argument(
        "--disable_wan_activation_checkpointing", action="store_true", default=False
    )
    parser.add_argument("--four_gpu_check_only", action="store_true")
    parser.add_argument("--four_gpu_parse_only", action="store_true")
    return parser.parse_known_args(list(argv))


def _stable_uint64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def _stable_uniform(*parts: object) -> float:
    return _stable_uint64(*parts) / float(2**64)


def build_first_frame_reference_dataset_class(
    base_dataset_class: Type,
    *,
    joint_null_prob: float,
    reference_seed: int,
) -> Type:
    """Keep the decoded target's first frame as both Wan and MQ reference.

    Wan's direct ``ref_image`` is never dropped. Caption/MQ-image dropout is
    derived from the topology-neutral global draw id so splitting one global
    batch over four ranks does not change its conditioning distribution.
    """

    joint_probability = float(joint_null_prob)
    if (
        not math.isfinite(joint_probability)
        or not 0.0 <= joint_probability <= 1.0
    ):
        raise ValueError("joint_null_prob must be finite and within [0, 1]")

    class FirstFrameReferenceDataset(base_dataset_class):
        moviestory_first_frame_reference = True
        moviestory_joint_null_prob = joint_probability

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._moviestory_caption_null_prob = float(
                kwargs.get("null_caption_prob", 0.0)
            )
            self._moviestory_image_null_prob = float(
                kwargs.get("null_image_prob", 0.0)
            )
            for name, probability in (
                ("null_caption_prob", self._moviestory_caption_null_prob),
                ("null_image_prob", self._moviestory_image_null_prob),
            ):
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{name} must be finite and within [0, 1]")

            # Apply dropout below with a stable sample key. The direct Wan
            # reference remains present even for the joint CFG-null branch.
            kwargs["null_caption_prob"] = 0.0
            kwargs["null_image_prob"] = 0.0
            super().__init__(*args, **kwargs)
            self._moviestory_reference_seed = int(reference_seed)
            self._moviestory_printed_first_frame = False

        def __getitem__(self, index):
            global_draw_id = int(getattr(index, "global_draw_id", int(index)))
            result = dict(super().__getitem__(index))
            video = result.get("video")
            if not torch.is_tensor(video) or video.ndim != 4:
                raise ValueError(
                    "first-frame dataset expects video [C,T,H,W], got "
                    f"{getattr(video, 'shape', type(video).__name__)}"
                )

            reference = video_start_frame_to_reference_image(video)
            video_path = str(Path(result["video_path"]).expanduser().resolve())
            key = (
                f"seed={self._moviestory_reference_seed}|path={video_path}|"
                f"global_draw={global_draw_id}"
            )
            joint_null = _stable_uniform(key, "joint-null") < joint_probability
            caption_null = joint_null or (
                _stable_uniform(key, "caption-null")
                < self._moviestory_caption_null_prob
            )
            image_null = joint_null or (
                _stable_uniform(key, "image-null")
                < self._moviestory_image_null_prob
            )

            result["caption"] = "" if caption_null else result["caption"]
            result["ref_image"] = reference
            result["mq_ref_image"] = None if image_null else reference
            result["reference_frame_index"] = 0
            result["reference_total_frames"] = int(video.shape[1])
            result["reference_frame_ratio"] = 0.0
            result["moviestory_joint_null"] = bool(joint_null)
            result["moviestory_global_draw_id"] = int(global_draw_id)
            result["moviestory_sample_seed"] = int(
                _stable_uint64(key, "training-rng") % (2**63 - 1)
            )
            result["moviestory_reference_is_target_first_frame"] = True

            if not self._moviestory_printed_first_frame:
                print(
                    "[FIRST-FRAME-REFERENCE] "
                    f"path={video_path} frame=0/{int(video.shape[1])} "
                    f"target_frames={int(video.shape[1])} "
                    f"joint_null={int(joint_null)} mq_drop={int(image_null)}",
                    flush=True,
                )
                self._moviestory_printed_first_frame = True
            return result

    FirstFrameReferenceDataset.__name__ = (
        f"FirstFrameReference{base_dataset_class.__name__}"
    )
    return FirstFrameReferenceDataset


@contextlib.contextmanager
def topology_neutral_sample_rng(
    trainer: Any,
    batch: Mapping[str, Any],
) -> Iterator[None]:
    """Derive model-side randomness from the global sample draw."""
    seeds = batch.get("moviestory_sample_seed")
    if seeds is None or len(seeds) != 1:
        raise ValueError(
            "4-GPU training requires one moviestory_sample_seed per microbatch"
        )
    seed = int(seeds[0])
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = []
    if torch.cuda.is_available():
        for attribute in ("dev_dit", "dev_enc"):
            device = getattr(trainer, attribute, None)
            if isinstance(device, torch.device) and device.type == "cuda":
                index = (
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index
                )
                if index not in cuda_devices:
                    cuda_devices.append(index)
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_router_config(args: argparse.Namespace) -> ThreeRouterConfig:
    return ThreeRouterConfig(
        hidden_size=int(args.router_hidden_size),
        role_tokens=int(args.router_role_tokens),
        action_tokens=int(args.router_action_tokens),
        global_tokens=int(args.router_global_tokens),
    )


def run_check_only(args: argparse.Namespace) -> None:
    """CPU-only validation for the mapper and exact global sampler contract."""
    torch.manual_seed(11)
    mapper = MQToT5Mapper(
        hidden_size=32,
        bottleneck_size=8,
        residual_scale=float(args.mapper_residual_scale),
    )
    features = torch.randn(2, 7, 32, requires_grad=True)
    mapped = mapper(features)
    mapped.square().mean().backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise RuntimeError("mapper backward self-check failed")

    world_size = int(args.expected_world_size)
    samplers = [
        GlobalBatchEquivalentSampler(
            int(args.expected_train_samples),
            rank=rank,
            world_size=world_size,
            global_batch_size=int(args.global_effective_batch),
            optimizer_steps=(
                int(args.expected_train_samples)
                // int(args.global_effective_batch)
            ),
            seed=42,
        )
        for rank in range(world_size)
    ]
    local_rows = [list(sampler) for sampler in samplers]
    flattened = [index for row in local_rows for index in row]
    reconstructed_draw_ids = []
    local_microbatches = int(args.global_effective_batch) // world_size
    optimizer_steps = int(args.expected_train_samples) // int(
        args.global_effective_batch
    )
    for step in range(optimizer_steps):
        for rank in range(world_size):
            start = step * local_microbatches
            reconstructed_draw_ids.extend(
                int(value.global_draw_id)
                for value in local_rows[rank][
                    start : start + local_microbatches
                ]
            )
    status = {
        "status": "ok",
        "mapper_output_shape": list(mapped.shape),
        "mapper_gradient_finite": True,
        "world_size": world_size,
        "global_effective_batch": int(args.global_effective_batch),
        "local_microbatches": int(args.global_effective_batch) // world_size,
        "draws_per_rank": [len(row) for row in local_rows],
        "total_draws": len(flattened),
        "unique_draws": len(set(flattened)),
        "global_draw_ids_contiguous": reconstructed_draw_ids
        == list(range(int(args.expected_train_samples))),
    }
    if len(flattened) != int(args.expected_train_samples):
        raise RuntimeError("global sampler draw-count self-check failed")
    if len(set(flattened)) != int(args.expected_train_samples):
        raise RuntimeError("single-epoch sampler uniqueness self-check failed")
    if reconstructed_draw_ids != list(range(int(args.expected_train_samples))):
        raise RuntimeError("global draw-id reconstruction self-check failed")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def init_distributed(expected_world_size: int) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != int(expected_world_size):
        raise RuntimeError(
            "this entrypoint requires exactly "
            f"{expected_world_size} torchrun ranks, got {world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4 x 48 GiB training")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, visible GPUs={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout = timedelta(
            seconds=max(int(os.environ.get("MOVIESTORY_DIST_TIMEOUT_SEC", "3600")), 60)
        )
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timeout,
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timeout,
            )
    warmup = torch.zeros(1, device=f"cuda:{local_rank}")
    dist.all_reduce(warmup)
    return rank, local_rank


def audit_gpu_memory(
    local_rank: int,
    minimum_gib: float,
    minimum_free_gib: float,
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(local_rank)
    total_gib = float(properties.total_memory) / float(1024**3)
    if total_gib + 1e-6 < float(minimum_gib):
        raise RuntimeError(
            f"GPU {local_rank} has {total_gib:.2f} GiB; "
            f"at least {minimum_gib:.2f} GiB is required"
        )
    free_bytes, _ = torch.cuda.mem_get_info(local_rank)
    free_gib = float(free_bytes) / float(1024**3)
    if free_gib + 1e-6 < float(minimum_free_gib):
        raise RuntimeError(
            f"GPU {local_rank} has only {free_gib:.2f} GiB free before model "
            f"loading; at least {minimum_free_gib:.2f} GiB is required"
        )
    payload = {
        "rank": int(dist.get_rank()),
        "local_rank": int(local_rank),
        "name": str(properties.name),
        "total_gib": total_gib,
        "free_gib_before_load": free_gib,
    }
    print(f"[GPU-AUDIT] {json.dumps(payload, ensure_ascii=False)}", flush=True)
    return payload


def _write_metadata(
    checkpoint_path: Path,
    *,
    args: argparse.Namespace,
    router_config: ThreeRouterConfig,
    enabled: bool,
    world_size: int,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    mode = int(args.conditioning_mode)
    payload = {
        "format": FORMAT_NAME,
        "conditioning": {
            "mode": mode,
            "mode_0": "processed_mq_replaces_t5",
            "mode_1": "mapped_mq_then_frozen_t5_prompt_tokens",
            "active_contract": (
                "processed_mq_replaces_t5"
                if mode == 0
                else "mapped_mq_then_frozen_t5_prompt_tokens"
            ),
            "wan_text_len": int(args.moviestory_context_text_len),
            "mq_to_t5_mapper": bool(mode == 1),
            "mapper_bottleneck_size": (
                int(args.mapper_bottleneck_size) if mode == 1 else None
            ),
            "mapper_rms_match": bool(
                mode == 1 and not args.disable_mapper_rms_match
            ),
        },
        "distributed": {
            "world_size": int(world_size),
            "per_rank_batch_size": 1,
            "per_rank_gradient_accumulation": int(
                args.gradient_accumulation_steps
            ),
            "global_effective_batch": int(args.global_effective_batch),
            "minimum_total_gpu_gib": float(args.minimum_gpu_memory_gib),
            "minimum_free_gpu_gib_before_load": float(
                args.minimum_free_gpu_memory_gib
            ),
            "optimizer_steps": int(args.num_train_steps),
            "total_sample_draws": int(
                args.num_train_steps * args.global_effective_batch
            ),
            "wan_fsdp": "FULL_SHARD_use_orig_params",
            "mq_parallel": "DDP",
            "t5_device": "cpu",
            "wan_activation_checkpointing": bool(
                not args.disable_wan_activation_checkpointing
            ),
            "gradient_clip": "global_l2_over_ddp_mq_and_fsdp_wan",
        },
        "reference_conditioning": {
            "sampling": "target_video_first_frame",
            "strong_bind": bool(args.moviestory_wan_first_frame_strong_bind),
            "wan_injection": (
                "strongbind_clean_preserved_reference_prefix"
                if args.moviestory_wan_first_frame_strong_bind
                else "none"
            ),
            "reference_timestep": (
                0 if args.moviestory_wan_first_frame_strong_bind else None
            ),
            "reference_prefix_in_loss": False,
            "target_first_latent_removed": bool(
                args.moviestory_wan_first_frame_strong_bind
            ),
            "original_first_target_latent_in_loss": not bool(
                args.moviestory_wan_first_frame_strong_bind
            ),
            "temporal_binding": (
                "exact_first_frame"
                if args.moviestory_wan_first_frame_strong_bind
                else "none"
            ),
            "latent_prefix_binding": (
                "hard_clean"
                if args.moviestory_wan_first_frame_strong_bind
                else "none"
            ),
            "joint_null_prob": float(args.joint_null_prob),
            "mq_image_dropout_preserves_wan_reference": True,
            "stochastic_seed": "stable_hash(global_draw_id, video_path)",
        },
        "three_router_enabled": bool(enabled),
        "router": router_config.to_dict(),
        "loss_contract": "video_ground_truth_velocity_mse_only",
    }
    (checkpoint_path / "four_gpu_training_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def configure_base_args(
    args: argparse.Namespace,
    custom: argparse.Namespace,
    *,
    local_rank: int,
    world_size: int,
    router_config: ThreeRouterConfig,
) -> None:
    if int(args.num_metaqueries) != int(router_config.total_tokens):
        raise ValueError(
            f"--num_metaqueries must be {router_config.total_tokens}, "
            f"got {args.num_metaqueries}"
        )
    if int(args.connector_num_hidden_layers) != 24:
        raise ValueError("4x48G equivalence requires connector_num_hidden_layers=24")
    if int(args.batch_size) != 1:
        raise ValueError("4x48G equivalence requires --batch_size 1")
    # 下面这个是检测训练时候所用的数据条数必须等于加载的数据集的总条数的
    # if int(args.num_train_steps) * int(custom.global_effective_batch) != int(
    #     custom.expected_train_samples
    # ):
    #     raise ValueError(
    #         "num_train_steps * global_effective_batch must equal "
    #         f"expected_train_samples ({custom.expected_train_samples})"
    #     )
    if int(custom.global_effective_batch) % int(world_size) != 0:
        raise ValueError("global_effective_batch must be divisible by world_size")

    local_accumulation = int(custom.global_effective_batch) // int(world_size)
    # The old job used batch=1, accumulation=8, world=1.  Four ranks use two
    # local microbatches and DDP/FSDP averaging, preserving an effective batch 8.
    args.gradient_accumulation_steps = local_accumulation
    args.dit_device = int(local_rank)
    args.encoder_device = int(local_rank)
    args.dit_fsdp = True
    args.t5_fsdp = False
    args.t5_cpu = True
    args.use_sp = False
    args.train_mq_input_embeddings = False
    args.mq_gradient_checkpointing = True
    args.dataloader_num_workers = 0
    args.aggressive_empty_cache = True
    args.log_cuda_memory = True
    args.wan_train_mode = "cond_only"

    configure_video_ground_truth_only_loss(args)
    configure_wan_first_frame_strong_binding(
        args,
        enabled=bool(custom.wan_first_frame_strong_bind),
    )
    for key, value in vars(custom).items():
        setattr(args, key, value)
    args.three_router_enabled = not bool(custom.disable_3router)
    args.router_config = router_config.to_dict()
    args.moviestory_first_frame_reference = True
    args.moviestory_hardware_contract = "4x48g"
    args.moviestory_context_text_len = int(
        router_config.total_tokens + (512 if custom.conditioning_mode == 1 else 0)
    )
    if int(custom.conditioning_mode) == 1:
        # The dual-mode encoder performs one explicit T5 call and scales only
        # mapped MQ.  Disable the inherited whole-context probe/matcher to avoid
        # a second T5 forward and accidental rescaling of raw T5 tokens.
        args.mq_norm_probe_with_t5 = False
        args.mq_norm_match_t5 = False


def build_trainer_class(base_train, custom, router_config, enabled):
    class FourGPUThreeRouterTrainer(
        StrongFirstFrameTrainingMixin,
        base_train.MetaQueryWanTrainer,
    ):
        def __init__(self, args):
            super().__init__(args)
            module = self._mq_encoder_module()
            module.bind_t5_provider(self._encode_text)
            self._aug_text_len = int(args.moviestory_context_text_len)

        def _wandb_config(self):
            config = dict(super()._wandb_config())
            config.update(
                {
                    "moviestory_format": FORMAT_NAME,
                    "conditioning_mode": int(self.args.conditioning_mode),
                    "global_effective_batch": int(
                        self.args.global_effective_batch
                    ),
                    "world_size": int(dist.get_world_size()),
                    "first_frame_reference": True,
                    "wan_first_frame_strong_bind": bool(
                        self.args.moviestory_wan_first_frame_strong_bind
                    ),
                    "first_target_latent_removed": bool(
                        self.args.moviestory_wan_first_frame_strong_bind
                    ),
                    "mq_to_t5_mapper": bool(
                        int(self.args.conditioning_mode) == 1
                    ),
                }
            )
            return config

        def _setup_optimizer(self):
            super()._setup_optimizer()
            self._router_update_tracker = None
            module = self._mq_encoder_module()
            route_tables = getattr(module, "route_metaquery_embeddings", None)
            if route_tables is None:
                return
            moved = move_parameters_to_zero_weight_decay_group(
                self.optimizer,
                self.scheduler,
                route_tables.parameters(),
                group_name="route_metaquery_embeddings",
            )
            if not moved:
                raise RuntimeError("route MetaQuery parameters are absent from optimizer")
            self._router_update_tracker = RouterParameterUpdateTracker(
                route_tables,
                stale_update_patience=int(self.args.router_stale_update_patience),
            )
            self.optimizer.register_step_pre_hook(
                self._router_update_tracker.before_optimizer_step
            )
            self.optimizer.register_step_post_hook(
                self._router_update_tracker.after_optimizer_step
            )

            if int(self.args.conditioning_mode) == 1:
                mapper = getattr(module, "mq_to_t5_mapper", None)
                if mapper is None:
                    raise RuntimeError("mode 1 has no MQ-to-T5 mapper")
                mapper_ids = {id(parameter) for parameter in mapper.parameters()}
                optimizer_ids = {
                    id(parameter)
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                }
                if not mapper_ids or not mapper_ids.issubset(optimizer_ids):
                    raise RuntimeError("not all mapper parameters are in optimizer")

        def _verify_train_context_injection_once(self, mq_feat, aug_feat):
            if self._printed_context_inject_check:
                return
            if not torch.allclose(
                mq_feat.float(), aug_feat.float(), atol=1e-3, rtol=1e-3
            ):
                raise RuntimeError("composed context changed before Wan injection")
            module = self._mq_encoder_module()
            audit = dict(getattr(module, "last_context_audit", {}))
            expected_mode = int(self.args.conditioning_mode)
            if int(audit.get("mode", -1)) != expected_mode:
                raise RuntimeError(f"context audit mode mismatch: {audit}")
            if int(audit.get("context_tokens", -1)) != int(aug_feat.shape[0]):
                raise RuntimeError(f"context length audit mismatch: {audit}")
            if int(aug_feat.shape[0]) > int(self._aug_text_len):
                raise RuntimeError(
                    f"context={int(aug_feat.shape[0])} exceeds text_len={self._aug_text_len}"
                )
            print(
                f"[CONTEXT-VERIFY] {json.dumps(audit, ensure_ascii=False)}",
                flush=True,
            )
            self._printed_context_inject_check = True

        def _compute_loss(self, batch):
            try:
                with topology_neutral_sample_rng(self, batch):
                    loss = super()._compute_loss(batch)
                if not torch.is_tensor(loss) or loss.ndim != 0:
                    raise RuntimeError("training loss must be one scalar")
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise FloatingPointError("training loss is non-finite")
                denoise = float(getattr(self, "_last_loss_denoise", math.nan))
                if (
                    not math.isfinite(denoise)
                    or not math.isclose(
                        float(loss.detach().item()),
                        denoise,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    )
                ):
                    raise RuntimeError(
                        "loss differs from ground-truth denoising MSE"
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
                        "auxiliary losses must remain zero: "
                        f"{nonzero_auxiliary}"
                    )
                return loss
            except DistributedTrainingAbort:
                raise
            except Exception as error:
                # The inherited loop intentionally skips failed microbatches.
                # That changes the 4000-sample contract and can deadlock other
                # collective ranks, so this entrypoint always aborts instead.
                draw_ids = batch.get("moviestory_global_draw_id", ["unknown"])
                raise DistributedTrainingAbort(
                    "4x48G equivalence violation at global_draw_id="
                    f"{draw_ids[0]}: {type(error).__name__}: {error}"
                ) from error

        def _collect_trainability_metrics(self):
            metrics = super()._collect_trainability_metrics()
            module = self._mq_encoder_module()
            diagnostics = dict(getattr(module, "last_router_diagnostics", {}))
            diagnostics.update(
                dict(getattr(module, "last_route_embedding_grad_rms", {}))
            )
            metrics.update(router_diagnostics_to_metrics(diagnostics))
            tracker = getattr(self, "_router_update_tracker", None)
            if tracker is not None:
                metrics.update(tracker.last_metrics)
            audit = dict(getattr(module, "last_context_audit", {}))
            for key in ("mq_rms", "t5_rms", "mq_rms_match_scale"):
                if key in audit:
                    metrics[f"train/context_{key}"] = float(audit[key])
            if hasattr(self, "_moviestory_global_grad_norm"):
                metrics["train/global_grad_norm_preclip"] = float(
                    self._moviestory_global_grad_norm
                )
            return metrics

        def _save_checkpoint(self, path, step, extra_info=None):
            result = super()._save_checkpoint(path, step, extra_info=extra_info)
            if self.is_main_process:
                _write_metadata(
                    Path(path).expanduser().resolve(),
                    args=self.args,
                    router_config=router_config,
                    enabled=enabled,
                    world_size=dist.get_world_size(),
                )
            return result

        def train(self):
            original_clip_grad_norm = torch.nn.utils.clip_grad_norm_
            replicated = list(self._mq_trainable_params())
            sharded = list(self._wan_trainable_params())
            expected_ids = {id(parameter) for parameter in (*replicated, *sharded)}

            def distributed_clip_grad_norm_(
                parameters,
                max_norm,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=None,
            ):
                parameter_rows = list(parameters)
                actual_ids = {id(parameter) for parameter in parameter_rows}
                if actual_ids != expected_ids:
                    return original_clip_grad_norm(
                        parameter_rows,
                        max_norm,
                        norm_type=norm_type,
                        error_if_nonfinite=error_if_nonfinite,
                        foreach=foreach,
                    )
                norm = clip_grad_norm_mixed_sharded_(
                    replicated_parameters=replicated,
                    sharded_parameters=sharded,
                    max_norm=float(max_norm),
                    norm_type=float(norm_type),
                )
                self._moviestory_global_grad_norm = float(norm.detach().item())
                return norm

            # The inherited loop calls this public function once immediately
            # before optimizer.step.  Replace it only for this trainer/process.
            torch.nn.utils.clip_grad_norm_ = distributed_clip_grad_norm_
            try:
                result = super().train()
            finally:
                torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm
            if int(getattr(self, "_skipped_step_count", 0)) != 0:
                raise RuntimeError(
                    "a step was skipped and the run is invalid for equivalence: "
                    f"skipped={self._skipped_step_count}, "
                    f"oom={self._oom_skip_count}, errors={self._error_skip_count}"
                )
            return result

    FourGPUThreeRouterTrainer.__name__ = "FourGPUThreeRouterTrainer"
    return FourGPUThreeRouterTrainer


def run_training(
    custom: argparse.Namespace,
    base_argv: Sequence[str],
    router_config: ThreeRouterConfig,
) -> None:
    rank, local_rank = init_distributed(int(custom.expected_world_size))
    world_size = dist.get_world_size()
    audit_gpu_memory(
        local_rank,
        float(custom.minimum_gpu_memory_gib),
        float(custom.minimum_free_gpu_memory_gib),
    )

    # Identical initialization on every rank; DDP then broadcasts rank zero.
    base_seed = 42
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    import train_connector_for_wan as connector_module
    import train_metaquery_wan as base_train

    enabled = not bool(custom.disable_3router)
    routed_class = build_three_router_encoder_class(
        connector_module.MetaQueryEncoderForWan,
        router_config,
        enabled=enabled,
    )
    dual_mode_class = build_dual_mode_encoder_class(
        routed_class,
        conditioning_mode=int(custom.conditioning_mode),
        mapper_bottleneck_size=int(custom.mapper_bottleneck_size),
        mapper_residual_scale=float(custom.mapper_residual_scale),
        mapper_rms_match=not bool(custom.disable_mapper_rms_match),
    )
    connector_module.MetaQueryEncoderForWan = dual_mode_class

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *base_argv]
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv
    configure_base_args(
        args,
        custom,
        local_rank=local_rank,
        world_size=world_size,
        router_config=router_config,
    )

    if custom.four_gpu_parse_only:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "conditioning_mode": int(args.conditioning_mode),
                        "world_size": int(world_size),
                        "local_gradient_accumulation": int(
                            args.gradient_accumulation_steps
                        ),
                        "global_effective_batch": int(args.global_effective_batch),
                        "context_text_len": int(args.moviestory_context_text_len),
                        "wan_fsdp": bool(args.dit_fsdp),
                        "t5_cpu": bool(args.t5_cpu),
                        "first_frame_reference": True,
                        "wan_first_frame_strong_bind": bool(
                            args.moviestory_wan_first_frame_strong_bind
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    if base_train.WanDatasetClass is None:
        raise RuntimeError("WanVideoDataset is unavailable")
    base_train.WanDatasetClass = build_first_frame_reference_dataset_class(
        base_train.WanDatasetClass,
        joint_null_prob=float(custom.joint_null_prob),
        reference_seed=int(args.seed),
    )
    install_equivalent_dataloader(
        base_train,
        rank=rank,
        world_size=world_size,
        global_batch_size=int(custom.global_effective_batch),
        optimizer_steps=int(args.num_train_steps),
        seed=int(args.seed),
        expected_dataset_size=int(custom.expected_train_samples),
    )
    install_wan_fsdp_sharder(
        context_text_len=int(args.moviestory_context_text_len),
        activation_checkpointing=not bool(
            custom.disable_wan_activation_checkpointing
        ),
    )

    trainer_class = build_trainer_class(
        base_train,
        custom,
        router_config,
        enabled,
    )
    trainer = trainer_class(args)
    trainer.mq_encoder = DDP(
        trainer.mq_encoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=25,
    )
    trainer.post_wrap_ddp_audit()

    # Keep the dataset seed identical on every rank so a sampler index always
    # resolves to the same video. Model-side randomness is derived from each
    # topology-neutral global draw id.
    print(
        "[4X48G] "
        f"rank={rank}/{world_size} gpu={local_rank} mode={args.conditioning_mode} "
        f"local_accum={args.gradient_accumulation_steps} "
        f"global_batch={args.global_effective_batch} "
        f"context_text_len={args.moviestory_context_text_len} "
        f"first_frame_reference=1 "
        f"strong_bind={int(args.moviestory_wan_first_frame_strong_bind)}",
        flush=True,
    )
    trainer.train()
    dist.barrier(device_ids=[local_rank])


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    custom, base_argv = parse_four_gpu_args(argv)
    router_config = build_router_config(custom)
    if custom.four_gpu_check_only:
        run_check_only(custom)
        return
    try:
        run_training(custom, base_argv, router_config)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
