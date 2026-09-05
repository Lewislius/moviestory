"""MovieStory distributed helpers for two untouched native Wan I2V-A14B DiTs."""

from __future__ import annotations

from functools import partial
from typing import Dict, Iterable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Sampler


class SampleDrawIndex(int):
    def __new__(cls, dataset_index: int, global_draw_id: int):
        value = int.__new__(cls, int(dataset_index))
        value.global_draw_id = int(global_draw_id)
        return value


class GlobalBatchSampler(Sampler[int]):
    """One deterministic global stream split into rank-local microbatches."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        global_batch_size: int,
        optimizer_steps: int,
        seed: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_batch_size = int(global_batch_size)
        self.optimizer_steps = int(optimizer_steps)
        self.seed = int(seed)
        if self.dataset_size <= 0 or self.optimizer_steps <= 0:
            raise ValueError("dataset_size and optimizer_steps must be positive")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid rank/world_size")
        if self.global_batch_size % self.world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        self.local_microbatches = self.global_batch_size // self.world_size

    def _stream(self) -> torch.Tensor:
        required = self.optimizer_steps * self.global_batch_size
        chunks = []
        produced = 0
        epoch = 0
        while produced < required:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            permutation = torch.randperm(self.dataset_size, generator=generator)
            take = min(required - produced, int(permutation.numel()))
            chunks.append(permutation[:take])
            produced += take
            epoch += 1
        return torch.cat(chunks)

    def __iter__(self) -> Iterator[int]:
        blocks = self._stream().reshape(self.optimizer_steps, self.global_batch_size)
        start = self.rank * self.local_microbatches
        stop = start + self.local_microbatches
        local = []
        for step in range(self.optimizer_steps):
            for offset in range(start, stop):
                draw_id = step * self.global_batch_size + offset
                local.append(SampleDrawIndex(int(blocks[step, offset]), draw_id))
        return iter(local)

    def __len__(self) -> int:
        return self.optimizer_steps * self.local_microbatches


def install_native_i2v_fsdp(
    *,
    context_text_len: int,
    activation_checkpointing: bool = False,
    wan_train_mode: str = "frozen",
    wan_cond_name_pattern: str = "",
) -> None:
    """Replace only WanI2V's wrapping function, never ``WanModel`` itself."""

    if context_text_len <= 0:
        raise ValueError("context_text_len must be positive")
    train_mode = str(wan_train_mode).strip().lower()
    if train_mode not in ("frozen", "cond_only", "full"):
        raise ValueError(f"unsupported wan_train_mode: {wan_train_mode}")
    cond_keywords = tuple(
        item.strip().lower()
        for item in wan_cond_name_pattern.split(",")
        if item.strip()
    ) or (
        "cross_attn",
        "cross-attn",
        "crossattention",
        "cross_attention",
        "text_embedding",
        "time_projection",
        "modulation",
        "cross_attn_norm",
        "norm3",
    )
    import wan.image2video as wan_i2v_module
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    def training_shard_model(
        model: nn.Module,
        device_id: int,
        param_dtype: torch.dtype = torch.bfloat16,
        reduce_dtype: torch.dtype = torch.float32,
        buffer_dtype: torch.dtype = torch.float32,
        process_group=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states: bool = True,
        use_lora: bool = False,
    ) -> FSDP:
        del use_lora
        model.text_len = int(context_text_len)
        # WanI2V freezes the model immediately before invoking shard_model.
        # Select trainability here so FSDP sees the final requires_grad layout
        # at construction time.  This changes no module or parameter shape.
        for name, parameter in model.named_parameters():
            selected = train_mode == "full" or (
                train_mode == "cond_only"
                and any(keyword in name.lower() for keyword in cond_keywords)
            )
            parameter.requires_grad_(selected)
        if not hasattr(model, "blocks") or len(model.blocks) == 0:
            raise RuntimeError("Wan I2V model has no transformer blocks")
        if activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )

            for index, block in enumerate(list(model.blocks)):
                if not getattr(block, "_native_i2v_activation_checkpoint", False):
                    wrapped = checkpoint_wrapper(
                        block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    )
                    wrapped._native_i2v_activation_checkpoint = True
                    model.blocks[index] = wrapped
        wrapped_blocks = tuple(model.blocks)
        auto_wrap = partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module in wrapped_blocks,
        )
        wrapped = FSDP(
            module=model,
            process_group=process_group,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap,
            mixed_precision=MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=buffer_dtype,
            ),
            device_id=device_id,
            sync_module_states=sync_module_states,
            use_orig_params=True,
            limit_all_gathers=True,
            forward_prefetch=False,
        )
        wrapped._native_i2v_context_text_len = int(context_text_len)
        wrapped._native_i2v_activation_checkpointing = bool(activation_checkpointing)
        return wrapped

    # WanI2V imported shard_model into this module namespace.  Patching this
    # local binding affects construction only; native WanModel.forward is intact.
    wan_i2v_module.shard_model = training_shard_model


def shard_mq_connector_fsdp(
    encoder: nn.Module,
    *,
    device_id: int,
    process_group=None,
    sync_module_states: bool = True,
) -> nn.Module:
    """FULL_SHARD the shared Connector while leaving frozen Qwen replicated.

    The Connector keeps its original BF16 parameters and forward implementation.
    Each of its 24 Qwen2 encoder layers becomes an FSDP unit, with the Sequential
    Connector itself as the FSDP root so its projection head is sharded as well.
    Route MQ tables and the optional FP32 MQ-to-T5 mapper remain replicated and
    are synchronized explicitly by :func:`sync_replicated_gradients_`.
    """

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    mllm_model = getattr(encoder, "mllm_model", None)
    connector = getattr(mllm_model, "connector", None)
    if not isinstance(connector, nn.Module):
        raise RuntimeError("MQ encoder has no shared Connector module")
    if isinstance(connector, FSDP):
        raise RuntimeError("MQ Connector is already FSDP wrapped")
    if not isinstance(connector, nn.Sequential) or len(connector) == 0:
        raise RuntimeError("MQ Connector must be the expected Sequential module")
    connector_encoder = connector[0]
    layers = getattr(connector_encoder, "layers", None)
    if layers is None or len(layers) != 24:
        raise RuntimeError(
            "MQ Connector FULL_SHARD requires the reference 24-layer encoder"
        )
    wrapped_layers = tuple(layers)
    full_parameter_numel = sum(parameter.numel() for parameter in connector.parameters())
    auto_wrap = partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: module in wrapped_layers,
    )
    wrapped = FSDP(
        module=connector,
        process_group=process_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        device_id=device_id,
        sync_module_states=bool(sync_module_states),
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=False,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
    )
    wrapped._moviestory_connector_fsdp = True
    wrapped._moviestory_connector_layers = len(wrapped_layers)
    wrapped._moviestory_connector_sharding = "FULL_SHARD"
    wrapped._moviestory_connector_full_numel = int(full_parameter_numel)
    local_parameter_numel = sum(parameter.numel() for parameter in wrapped.parameters())
    shard_counts = torch.tensor(
        [local_parameter_numel],
        device=torch.device(f"cuda:{device_id}")
        if isinstance(device_id, int)
        else torch.device(device_id),
        dtype=torch.int64,
    )
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    gathered_counts = [torch.zeros_like(shard_counts) for _ in range(world_size)]
    if len(gathered_counts) > 1:
        torch.distributed.all_gather(gathered_counts, shard_counts)
    else:
        gathered_counts[0].copy_(shard_counts)
    counts = tuple(int(value.item()) for value in gathered_counts)
    if sum(counts) != full_parameter_numel:
        raise RuntimeError(
            "Connector FSDP shard audit failed: "
            f"full={full_parameter_numel}, local_counts={counts}"
        )
    if len(counts) > 1 and max(counts) >= full_parameter_numel:
        raise RuntimeError(f"Connector was not actually sharded: local_counts={counts}")
    wrapped._moviestory_connector_shard_numel = counts
    mllm_model.connector = wrapped
    return wrapped


def full_fsdp_connector_state_dict(connector: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect a portable full Connector state on rank 0, empty on other ranks."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not isinstance(connector, FSDP) or not getattr(
        connector, "_moviestory_connector_fsdp", False
    ):
        raise TypeError("expected MovieStory's FSDP-wrapped MQ Connector")
    state = get_model_state_dict(
        connector,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    return dict(state)


@torch.no_grad()
def broadcast_replicated_parameters_(
    parameters: Iterable[nn.Parameter], *, src: int = 0
) -> None:
    """Match DDP construction semantics for MQ parameters outside FSDP.

    DDP broadcasts parameters from rank 0 when it is constructed.  The route
    tables and optional mode-1 mapper intentionally stay replicated, so they
    need the same one-time broadcast after the Connector moves to FSDP.
    """

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.distributed.get_world_size() <= 1:
        return
    for parameter in dict.fromkeys(parameters):
        torch.distributed.broadcast(parameter.detach(), src=src)


@torch.no_grad()
def sync_replicated_gradients_(parameters: Iterable[nn.Parameter]) -> None:
    """Average replicated MQ/mapper gradients exactly once across all ranks."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return
    for parameter in dict.fromkeys(parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise RuntimeError("replicated MQ gradients must be dense")
        torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
        gradient.div_(world_size)


def clip_grad_norm_mixed_sharded_(
    *,
    replicated_parameters: Iterable[nn.Parameter],
    sharded_parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Global L2 clipping for replicated MQ params plus Connector/Wan shards."""

    replicated = list(dict.fromkeys(replicated_parameters))
    sharded = list(dict.fromkeys(sharded_parameters))
    replicated_ids = {id(parameter) for parameter in replicated}
    if any(id(parameter) in replicated_ids for parameter in sharded):
        raise ValueError("replicated and sharded parameter groups overlap")
    gradients = [
        parameter.grad
        for parameter in (*replicated, *sharded)
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise RuntimeError(f"gradients span devices: {devices}")
    device = next(iter(devices))

    def squared(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
        value = torch.zeros((), device=device, dtype=torch.float32)
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            value += torch.linalg.vector_norm(
                gradient.detach(), ord=2, dtype=torch.float32
            ).square()
        return value

    replicated_sq = squared(replicated)
    sharded_sq = squared(sharded)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sharded_sq, op=torch.distributed.ReduceOp.SUM)
    norm = (replicated_sq + sharded_sq).sqrt()
    if not bool(torch.isfinite(norm).item()):
        raise FloatingPointError(f"non-finite gradient norm: {norm.item()}")
    coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return norm
