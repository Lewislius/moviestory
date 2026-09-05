#!/usr/bin/env python3
"""Train 3-router MetaQuery with untouched native Wan2.2 I2V-A14B.

The only Wan-side inputs are the stock ``context`` and I2V ``y`` arguments.
No Wan block, attention layer, latent prefix, or timestep layout is replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


TRAIN_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TRAIN_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
HOME_ROOT = PROJECT_ROOT.parents[1]
WAN_TRAIN_ROOT = HOME_ROOT / "model" / "Wan2.2" / "scripts-metaquery-single" / "train"
WAN_ROOT = HOME_ROOT / "model" / "Wan2.2"
MODEL_ROOT = HOME_ROOT / "model"
METAQUERY_ROOT = MODEL_ROOT / "Qwen3-VL-main" / "metaquery-main"
for _path in (TRAIN_ROOT, CODE_ROOT, WAN_TRAIN_ROOT, WAN_ROOT, METAQUERY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from native_i2v_3router import (  # noqa: E402
    MQToT5Mapper,
    MetaQueryQwenWanI2VA14B,
    ThreeRouterConfig,
    build_dual_mode_encoder_class,
    build_three_router_encoder_class,
)
from native_i2v_3router.distributed import (  # noqa: E402
    GlobalBatchSampler,
    broadcast_replicated_parameters_,
    clip_grad_norm_mixed_sharded_,
    full_fsdp_connector_state_dict,
    install_native_i2v_fsdp,
    shard_mq_connector_fsdp,
    sync_replicated_gradients_,
)


FORMAT_NAME = "wan_i2v_a14b_native_3router_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native Wan2.2 I2V-A14B + Qwen/MetaQuery 3-router training"
    )
    parser.add_argument(
        "--wan_checkpoint_dir",
        default=str(WAN_ROOT / "Wan2.2-I2V-A14B"),
    )
    parser.add_argument(
        "--qwen3vl_model_id",
        default=str(MODEL_ROOT / "Qwen3-VL-main" / "Qwen3-VL-2B-Thinking"),
    )
    parser.add_argument("--output_dir", required=False, default="./i2v_3router_output")
    parser.add_argument("--local_openvid_video_root", default=None)
    parser.add_argument("--local_openvid_csv_path", default=None)
    parser.add_argument("--local_openvid_limit", type=int, default=4000)
    parser.add_argument("--local_video_cache_dir", default=None)
    parser.add_argument("--caption_tokenizer_path", default="google/umt5-xxl")

    parser.add_argument("--expected_world_size", type=int, default=8)
    parser.add_argument("--global_effective_batch", type=int, default=8)
    parser.add_argument("--expected_train_samples", type=int, default=4000)
    parser.add_argument("--num_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--wan_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=25)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--max_area", type=int, default=512 * 512)
    parser.add_argument("--max_caption_tokens", type=int, default=512)
    parser.add_argument("--min_duration_sec", type=float, default=0.5)
    parser.add_argument("--max_duration_sec", type=float, default=20.0)
    parser.add_argument("--null_caption_prob", type=float, default=0.1)
    parser.add_argument("--null_image_prob", type=float, default=0.1)

    parser.add_argument("--conditioning_mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--num_metaqueries", type=int, default=256)
    parser.add_argument("--connector_num_hidden_layers", type=int, default=24)
    parser.add_argument("--mapper_bottleneck_size", type=int, default=1024)
    parser.add_argument("--mapper_residual_scale", type=float, default=0.1)
    parser.add_argument("--disable_mapper_rms_match", action="store_true")
    parser.add_argument("--router_hidden_size", type=int, default=2048)
    parser.add_argument("--router_role_tokens", type=int, default=96)
    parser.add_argument("--router_action_tokens", type=int, default=96)
    parser.add_argument("--router_global_tokens", type=int, default=64)
    parser.add_argument("--disable_3router", action="store_true")
    parser.add_argument("--disable_mq_gradient_checkpointing", action="store_true")
    parser.add_argument("--train_mq_input_embeddings", action="store_true")
    parser.add_argument("--connector_norm_init_scale", type=float, default=1.0)

    parser.add_argument(
        "--wan_train_mode",
        choices=("frozen", "cond_only", "full"),
        default="cond_only",
        help="cond_only is the default 8x48G mode; all modes keep Wan architecture unchanged",
    )
    parser.add_argument("--wan_cond_name_pattern", default="")
    wan_checkpointing = parser.add_mutually_exclusive_group()
    wan_checkpointing.add_argument(
        "--enable_wan_activation_checkpointing",
        dest="enable_wan_activation_checkpointing",
        action="store_true",
        help="checkpoint native Wan blocks; enabled by default for the 8x48G cond-only path",
    )
    wan_checkpointing.add_argument(
        "--disable_wan_activation_checkpointing",
        dest="enable_wan_activation_checkpointing",
        action="store_false",
        help="disable native Wan block activation checkpointing",
    )
    parser.set_defaults(enable_wan_activation_checkpointing=True)
    parser.add_argument("--minimum_gpu_memory_gib", type=float, default=44.0)
    parser.add_argument("--minimum_free_gpu_memory_gib", type=float, default=40.0)
    parser.add_argument("--strict_dataset_size", action="store_true")

    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", default="wan-i2v-a14b-3router")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--parse_only", action="store_true")
    return parser.parse_args(argv)


def router_config_from_args(args: argparse.Namespace) -> ThreeRouterConfig:
    return ThreeRouterConfig(
        hidden_size=args.router_hidden_size,
        role_tokens=args.router_role_tokens,
        action_tokens=args.router_action_tokens,
        global_tokens=args.router_global_tokens,
    )


def validate_args(args: argparse.Namespace, router: ThreeRouterConfig) -> None:
    if args.num_metaqueries != router.total_tokens:
        raise ValueError(
            f"num_metaqueries={args.num_metaqueries}, router total={router.total_tokens}"
        )
    if args.connector_num_hidden_layers != 24:
        raise ValueError("the MQ reference architecture requires a 24-layer Connector")
    if args.frame_num <= 0 or (args.frame_num - 1) % 4 != 0:
        raise ValueError("native I2V frame_num must be 4n+1")
    if args.global_effective_batch % args.expected_world_size != 0:
        raise ValueError("global_effective_batch must be divisible by world size")
    if args.num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    for name in ("null_caption_prob", "null_image_prob"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    for path_name in ("wan_checkpoint_dir", "qwen3vl_model_id"):
        path = Path(getattr(args, path_name)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path_name} does not exist: {path}")


def run_check_only(args: argparse.Namespace, router: ThreeRouterConfig) -> None:
    validate_args(args, router)
    mapper = MQToT5Mapper(hidden_size=32, bottleneck_size=8)
    inputs = torch.randn(2, 7, 32, requires_grad=True)
    mapper(inputs).square().mean().backward()
    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("MQ mapper backward check failed")
    samplers = [
        GlobalBatchSampler(
            args.expected_train_samples,
            rank=rank,
            world_size=args.expected_world_size,
            global_batch_size=args.global_effective_batch,
            optimizer_steps=args.num_train_steps,
            seed=args.seed,
        )
        for rank in range(args.expected_world_size)
    ]
    local_rows = [list(sampler) for sampler in samplers]
    print(
        json.dumps(
            {
                "status": "ok",
                "format": FORMAT_NAME,
                "router": router.to_dict(),
                "conditioning_mode": args.conditioning_mode,
                "context_text_len": router.total_tokens
                + (512 if args.conditioning_mode == 1 else 0),
                "draws_per_rank": [len(rows) for rows in local_rows],
                "total_draws": sum(map(len, local_rows)),
                "native_i2v_contract": {
                    "image": "y=concat(mask4,vae(first_frame+zeros)16)",
                    "flow": "x_t=(1-t)x0+t*noise",
                    "target": "noise-x0",
                    "loss": "MSE(pred,target)",
                    "dual_dit_boundary": 900,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected {args.expected_world_size} torchrun ranks, got {world_size}"
        )
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA local rank {local_rank} unavailable; visible={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout = timedelta(seconds=max(int(os.getenv("WAN_I2V_DIST_TIMEOUT_SEC", "300")), 60))
        try:
            dist.init_process_group(
                "nccl",
                init_method="env://",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timeout,
            )
        except TypeError:
            dist.init_process_group("nccl", init_method="env://", timeout=timeout)
    probe = torch.zeros(1, device=local_rank)
    dist.all_reduce(probe)
    return rank, local_rank, world_size


def audit_gpu(args: argparse.Namespace, local_rank: int) -> Dict[str, object]:
    properties = torch.cuda.get_device_properties(local_rank)
    total = properties.total_memory / 1024**3
    free, _ = torch.cuda.mem_get_info(local_rank)
    free_gib = free / 1024**3
    if total < args.minimum_gpu_memory_gib or free_gib < args.minimum_free_gpu_memory_gib:
        raise RuntimeError(
            f"GPU {local_rank}: total={total:.2f}GiB free={free_gib:.2f}GiB; "
            f"required total/free={args.minimum_gpu_memory_gib}/{args.minimum_free_gpu_memory_gib}"
        )
    payload = {
        "local_rank": local_rank,
        "gpu": properties.name,
        "total_gib": total,
        "free_gib": free_gib,
    }
    print(f"[GPU] {json.dumps(payload, ensure_ascii=False)}", flush=True)
    return payload


def build_dataset(args: argparse.Namespace):
    from train_connector_for_wan import WanVideoDataset
    from native_i2v_3router.data import build_timeout_video_dataset_class

    dataset_class = build_timeout_video_dataset_class(WanVideoDataset)
    requested_limit = args.local_openvid_limit
    preclean_enabled = os.environ.get("WAN_DATA_PRECLEAN", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )
    dataset_limit = requested_limit
    if preclean_enabled and requested_limit is not None:
        pool_factor = max(
            1.0, float(os.environ.get("WAN_DATA_PRECLEAN_POOL_FACTOR", "1.25"))
        )
        dataset_limit = int(math.ceil(requested_limit * pool_factor))

    dataset = dataset_class(
        frame_num=args.frame_num,
        max_area=args.max_area,
        null_caption_prob=args.null_caption_prob,
        null_image_prob=args.null_image_prob,
        max_caption_tokens=args.max_caption_tokens,
        caption_tokenizer_path=args.caption_tokenizer_path,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        seed=args.seed,
        local_video_cache_dir=args.local_video_cache_dir,
        local_openvid_video_root=args.local_openvid_video_root,
        local_openvid_csv_path=args.local_openvid_csv_path,
        local_openvid_limit=dataset_limit,
    )
    if requested_limit is not None:
        if len(dataset) < requested_limit:
            raise RuntimeError(
                f"bounded preclean retained {len(dataset)} videos, fewer than the requested "
                f"{requested_limit}; increase WAN_DATA_PRECLEAN_POOL_FACTOR"
            )
        if len(dataset) > requested_limit:
            dataset.samples = dataset.samples[:requested_limit]
    if args.strict_dataset_size and len(dataset) != args.expected_train_samples:
        raise RuntimeError(
            f"strict dataset size requires {args.expected_train_samples}, got {len(dataset)}"
        )
    return dataset


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def unique_parameters(parameters: Iterable[torch.nn.Parameter]):
    output = []
    seen = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            output.append(parameter)
    return output


def build_optimizer(model: MetaQueryQwenWanI2VA14B, args: argparse.Namespace):
    encoder = model.mq_encoder.module if hasattr(model.mq_encoder, "module") else model.mq_encoder
    connector = encoder.mllm_model.connector
    connector_parameters = unique_parameters(connector.parameters())
    connector_ids = {id(parameter) for parameter in connector_parameters}
    route_module = getattr(encoder, "route_metaquery_embeddings", None)
    route_ids = set()
    route_parameters = []
    if route_module is not None:
        route_parameters = unique_parameters(route_module.parameters())
        route_ids = {id(parameter) for parameter in route_parameters}
    other_mq_parameters = [
        parameter
        for parameter in unique_parameters(model.mq_trainable_parameters())
        if id(parameter) not in route_ids and id(parameter) not in connector_ids
    ]
    groups = []
    if connector_parameters:
        groups.append(
            {
                "params": connector_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "mq_connector_fsdp",
            }
        )
    if other_mq_parameters:
        groups.append(
            {
                "params": other_mq_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "mq_replicated",
            }
        )
    if route_parameters:
        groups.append(
            {
                "params": route_parameters,
                "lr": args.learning_rate,
                "weight_decay": 0.0,
                "name": "route_metaquery_embeddings",
            }
        )
    wan_parameters = unique_parameters(model.wan_trainable_parameters())
    if wan_parameters:
        groups.append(
            {
                "params": wan_parameters,
                "lr": args.learning_rate * args.wan_lr_ratio,
                "weight_decay": args.weight_decay,
                "name": "wan_native_parameters",
            }
        )
    if not groups:
        raise RuntimeError("no trainable parameters")
    # foreach=False avoids allocating parameter-sized tensor lists at AdamW.step.
    # It preserves the same per-parameter AdamW equations and hyperparameters.
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), foreach=False)
    replicated_mq = other_mq_parameters + route_parameters
    sharded_parameters = connector_parameters + wan_parameters
    return optimizer, replicated_mq, sharded_parameters


def encoder_diagnostics(model: MetaQueryQwenWanI2VA14B) -> Dict[str, float]:
    encoder = model.mq_encoder.module if hasattr(model.mq_encoder, "module") else model.mq_encoder
    values: Dict[str, float] = {}
    diagnostics = dict(getattr(encoder, "last_router_diagnostics", {}))
    diagnostics.update(dict(getattr(encoder, "last_route_embedding_grad_rms", {})))
    for name, value in diagnostics.items():
        if torch.is_tensor(value):
            values[f"router/{name}"] = float(value.detach().float().mean().item())
    audit = getattr(encoder, "last_context_audit", {})
    for name in ("mq_rms", "t5_rms", "mq_rms_match_scale"):
        if name in audit:
            values[f"context/{name}"] = float(audit[name])
    return values


def connector_fsdp_metadata(connector: torch.nn.Module) -> Dict[str, object]:
    return {
        "strategy": "FSDP_FULL_SHARD",
        "wrapped_layers": int(
            getattr(connector, "_moviestory_connector_layers", 0)
        ),
        "full_parameter_numel": getattr(
            connector, "_moviestory_connector_full_numel", None
        ),
        "parameter_shard_numel_per_rank": getattr(
            connector, "_moviestory_connector_shard_numel", None
        ),
        "frozen_qwen_replicated": True,
        "replicated_mq_initialization": "rank0_broadcast_like_ddp",
        "replicated_mq_grad_sync": "explicit_mean_all_reduce",
        "optimizer_foreach": False,
    }


def enforce_no_before_training_checkpoint(output_dir: Path, rank: int) -> None:
    """Remove the unused step-0 checkpoint while preserving normal checkpoints."""

    keep = os.environ.get(
        "WAN_KEEP_BEFORE_TRAINING_CHECKPOINT", "0"
    ).strip().lower() in ("1", "true", "on", "yes")
    if keep:
        return
    target = output_dir / "checkpoint-before-training"
    if rank == 0 and os.path.lexists(target):
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        print(f"[CHECKPOINT] removed disabled before-training checkpoint: {target}", flush=True)
    elif rank != 0:
        deadline = time.monotonic() + max(
            60, int(os.environ.get("WAN_I2V_DIST_TIMEOUT_SEC", "300"))
        )
        while os.path.lexists(target) and time.monotonic() < deadline:
            time.sleep(0.2)
        if os.path.lexists(target):
            raise TimeoutError(
                f"rank {rank} timed out waiting for removal of {target}"
            )


def save_checkpoint(
    *,
    model: MetaQueryQwenWanI2VA14B,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    step: int,
    output_dir: Path,
    rank: int,
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    encoder = model.mq_encoder.module if hasattr(model.mq_encoder, "module") else model.mq_encoder
    connector = encoder.mllm_model.connector
    # This collective gathers only the sharded Connector (not frozen Qwen) and
    # returns its original, wrapper-free state-dict keys on rank 0.
    connector_state = full_fsdp_connector_state_dict(connector)
    if rank == 0:
        connector_parameter_ids = {id(parameter) for parameter in connector.parameters()}
        trainable_state = {}
        for name, parameter in encoder.named_parameters():
            if parameter.requires_grad and id(parameter) not in connector_parameter_ids:
                trainable_state[name] = parameter.detach().cpu()
        for name, tensor in connector_state.items():
            if "_fsdp_wrapped_module" in name:
                raise RuntimeError(f"non-portable FSDP key in Connector state: {name}")
            trainable_state[f"mllm_model.connector.{name}"] = tensor.detach().cpu()
        torch.save(trainable_state, checkpoint / "mq_qwen_connector_trainable.pt")
        architecture = model.architecture_metadata()
        architecture["mq_connector_distributed"] = connector_fsdp_metadata(connector)
        architecture["wan_activation_checkpointing"] = bool(
            args.enable_wan_activation_checkpointing
        )
        (checkpoint / "architecture.json").write_text(
            json.dumps(architecture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (checkpoint / "training_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Wan and Connector optimizer states are local FSDP shards.  Saving every
    # rank avoids gathering either 14B branch; the portable full MQ/Connector
    # model state remains available separately on rank 0.
    local_wan = {
        name: parameter.detach().cpu()
        for name, parameter in zip(
            model._wan_trainable_names, model.wan_trainable_parameters()
        )
    }
    rank_state = {
        "format": FORMAT_NAME,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "step": step,
        "wan_train_mode": args.wan_train_mode,
        "mq_connector_sharding": "FULL_SHARD",
        "wan_local_trainable_shards": local_wan,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(rank_state, checkpoint / f"training_rank{rank:05d}.pt")
    dist.barrier()
    if rank == 0:
        latest = output_dir / "latest"
        latest.write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
        print(f"[CHECKPOINT] {checkpoint}", flush=True)


def maybe_init_wandb(args: argparse.Namespace, rank: int, config: Dict[str, object]):
    if rank != 0 or not args.wandb_enabled or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[W&B] package unavailable; continuing without W&B", flush=True)
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"native-i2v-3router-{int(time.time())}",
        mode=args.wandb_mode,
        config=config,
    )


def train(args: argparse.Namespace, router: ThreeRouterConfig) -> None:
    rank, local_rank, world_size = init_distributed(args)
    audit_gpu(args, local_rank)
    context_text_len = router.total_tokens + (512 if args.conditioning_mode == 1 else 0)
    install_native_i2v_fsdp(
        context_text_len=context_text_len,
        activation_checkpointing=args.enable_wan_activation_checkpointing,
        wan_train_mode=args.wan_train_mode,
        wan_cond_name_pattern=args.wan_cond_name_pattern,
    )

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    from train_connector_for_wan import MetaQueryEncoderForWan

    routed_class = build_three_router_encoder_class(
        MetaQueryEncoderForWan,
        router,
        enabled=not args.disable_3router,
    )
    encoder_class = build_dual_mode_encoder_class(
        routed_class,
        conditioning_mode=args.conditioning_mode,
        mapper_bottleneck_size=args.mapper_bottleneck_size,
        mapper_residual_scale=args.mapper_residual_scale,
        mapper_rms_match=not args.disable_mapper_rms_match,
    )
    model = MetaQueryQwenWanI2VA14B(
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        qwen3vl_model_id=args.qwen3vl_model_id,
        metaquery_encoder_class=encoder_class,
        num_metaqueries=args.num_metaqueries,
        connector_num_hidden_layers=args.connector_num_hidden_layers,
        mq_gradient_checkpointing=not args.disable_mq_gradient_checkpointing,
        train_mq_input_embeddings=args.train_mq_input_embeddings,
        connector_norm_init_scale=args.connector_norm_init_scale,
        device_id=local_rank,
        rank=rank,
        dit_fsdp=True,
        t5_cpu=True,
        context_text_len=context_text_len,
        wan_train_mode=args.wan_train_mode,
        wan_cond_name_pattern=args.wan_cond_name_pattern,
    )
    connector_fsdp = shard_mq_connector_fsdp(
        model.mq_encoder,
        device_id=local_rank,
    )
    optimizer, replicated_mq_parameters, sharded_parameters = build_optimizer(model, args)
    # DDP used to perform this rank-0 broadcast at construction time.  Keep it
    # explicitly for route embeddings and the optional mode-1 mapper, which are
    # deliberately replicated instead of being part of Connector FSDP.
    broadcast_replicated_parameters_(replicated_mq_parameters, src=0)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, args.num_train_steps)

    dataset = build_dataset(args)
    sampler = GlobalBatchSampler(
        len(dataset),
        rank=rank,
        world_size=world_size,
        global_batch_size=args.global_effective_batch,
        optimizer_steps=args.num_train_steps,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    accumulation = args.global_effective_batch // world_size
    if len(dataloader) != args.num_train_steps * accumulation:
        raise RuntimeError("rank-local dataloader length violates the global batch contract")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    enforce_no_before_training_checkpoint(output_dir, rank)
    metadata = model.architecture_metadata()
    metadata.update(
        {
            "format": FORMAT_NAME,
            "world_size": world_size,
            "global_effective_batch": args.global_effective_batch,
            "rank_local_accumulation": accumulation,
            "dataset_size": len(dataset),
            "before_training_checkpoint": "disabled",
            "wan_activation_checkpointing": bool(
                args.enable_wan_activation_checkpointing
            ),
            "mq_connector_distributed": connector_fsdp_metadata(connector_fsdp),
        }
    )
    wandb_run = maybe_init_wandb(args, rank, metadata)
    if rank == 0:
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(dataloader)
    branch_counts = {"low_noise": 0, "high_noise": 0}
    for step in range(1, args.num_train_steps + 1):
        started = time.perf_counter()
        loss_sum = 0.0
        last_output = None
        for micro_step in range(accumulation):
            batch = next(iterator)
            output = model(batch)
            loss = output.loss / accumulation
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError(f"non-finite loss at step={step}")
            # Let Connector FSDP reduce-scatter every micro-step.  FSDP
            # no_sync() would retain full, unsharded Connector gradients during
            # accumulation and defeat the memory objective on smaller worlds.
            loss.backward()
            loss_sum += float(loss.detach().item())
            branch_counts[output.branch] += 1
            last_output = output

        # Route MQ tables and the optional mapper are intentionally replicated;
        # match DDP semantics by averaging their accumulated gradients once.
        sync_replicated_gradients_(replicated_mq_parameters)
        grad_norm = clip_grad_norm_mixed_sharded_(
            replicated_parameters=replicated_mq_parameters,
            sharded_parameters=sharded_parameters,
            max_norm=args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - started, 1e-6)
        metrics: Dict[str, float | int | str] = {
            "train/step": step,
            "train/loss": loss_sum,
            "train/grad_norm": float(grad_norm.detach().item()),
            "train/lr": float(scheduler.get_last_lr()[0]),
            "train/step_sec": elapsed,
            "train/samples_per_sec_global": args.global_effective_batch / elapsed,
            "train/branch": last_output.branch if last_output is not None else "unknown",
            "train/low_noise_microbatches": branch_counts["low_noise"],
            "train/high_noise_microbatches": branch_counts["high_noise"],
        }
        metrics.update(encoder_diagnostics(model))
        if torch.cuda.is_available():
            metrics["train/cuda_alloc_gib"] = torch.cuda.memory_allocated(local_rank) / 1024**3
            metrics["train/cuda_peak_gib"] = torch.cuda.max_memory_allocated(local_rank) / 1024**3
        if rank == 0 and step % args.log_steps == 0:
            print(f"[STEP] {json.dumps(metrics, ensure_ascii=False)}", flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
        if step % args.save_steps == 0 or step == args.num_train_steps:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                step=step,
                output_dir=output_dir,
                rank=rank,
            )
    if wandb_run is not None:
        wandb_run.finish()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    router = router_config_from_args(args)
    validate_args(args, router)
    if args.check_only:
        run_check_only(args, router)
        return
    if args.parse_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return
    try:
        train(args, router)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
