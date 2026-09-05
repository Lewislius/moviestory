from __future__ import annotations

from functools import partial
from typing import Iterable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Sampler


class SampleDrawIndex(int):
    """Dataset index carrying its position in the topology-neutral draw stream."""

    def __new__(cls, dataset_index: int, global_draw_id: int):
        value = int.__new__(cls, int(dataset_index))
        value.global_draw_id = int(global_draw_id)
        return value


class GlobalBatchEquivalentSampler(Sampler[int]):
    """Split one deterministic global sample stream across data-parallel ranks.

    For the production settings, every optimizer step consumes one global block
    of eight indices.  Rank 0 receives the first two, rank 1 the next two, and so
    on.  Therefore 4 ranks x 2 accumulation steps is the same batch contract as
    1 rank x 8 accumulation steps.
    """

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
        if self.dataset_size <= 0:
            raise ValueError("dataset must be non-empty")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank/world_size")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.global_batch_size % self.world_size != 0:
            raise ValueError(
                "global_batch_size must be divisible by world_size: "
                f"{self.global_batch_size} % {self.world_size} != 0"
            )
        if self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        self.local_microbatches = self.global_batch_size // self.world_size

    def _global_stream(self) -> torch.Tensor:
        needed = self.optimizer_steps * self.global_batch_size
        chunks = []
        produced = 0
        epoch = 0
        while produced < needed:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            permutation = torch.randperm(
                self.dataset_size,
                generator=generator,
            )
            take = min(int(permutation.numel()), needed - produced)
            chunks.append(permutation[:take])
            produced += take
            epoch += 1
        return torch.cat(chunks, dim=0)

    def __iter__(self) -> Iterator[int]:
        blocks = self._global_stream().reshape(
            self.optimizer_steps,
            self.global_batch_size,
        )
        start = self.rank * self.local_microbatches
        end = start + self.local_microbatches
        local_rows = []
        for optimizer_step in range(self.optimizer_steps):
            for offset in range(start, end):
                global_draw_id = optimizer_step * self.global_batch_size + offset
                local_rows.append(
                    SampleDrawIndex(
                        int(blocks[optimizer_step, offset].item()),
                        global_draw_id,
                    )
                )
        return iter(local_rows)

    def __len__(self) -> int:
        return self.optimizer_steps * self.local_microbatches


def install_equivalent_dataloader(
    base_training_module,
    *,
    rank: int,
    world_size: int,
    global_batch_size: int,
    optimizer_steps: int,
    seed: int,
    expected_dataset_size: int | None = None,
) -> None:
    """Patch the inherited trainer's sole DataLoader construction safely."""

    original_dataloader = base_training_module.DataLoader

    def equivalent_dataloader(dataset, *args, **kwargs):
        if expected_dataset_size is not None and len(dataset) != int(
            expected_dataset_size
        ):
            raise RuntimeError(
                "4-GPU equivalence requires exactly "
                f"{int(expected_dataset_size)} decoded dataset rows, got "
                f"{len(dataset)}. Re-run data preparation instead of silently "
                "repeating or dropping samples."
            )
        batch_size = int(kwargs.get("batch_size", args[0] if args else 1))
        if batch_size != 1:
            raise ValueError(
                "MovieStory 4-GPU equivalence requires per-rank batch_size=1"
            )
        if kwargs.get("sampler") is not None:
            raise ValueError("an unexpected sampler was already configured")
        sampler = GlobalBatchEquivalentSampler(
            len(dataset),
            rank=rank,
            world_size=world_size,
            global_batch_size=global_batch_size,
            optimizer_steps=optimizer_steps,
            seed=seed,
        )
        kwargs["shuffle"] = False
        kwargs["sampler"] = sampler
        loader = original_dataloader(dataset, *args, **kwargs)
        print(
            "[GLOBAL-BATCH-SAMPLER] "
            f"rank={rank}/{world_size} dataset={len(dataset)} "
            f"local_draws={len(sampler)} global_batch={global_batch_size} "
            f"optimizer_steps={optimizer_steps}",
            flush=True,
        )
        return loader

    base_training_module.DataLoader = equivalent_dataloader


def clip_grad_norm_mixed_sharded_(
    *,
    replicated_parameters: Iterable[nn.Parameter],
    sharded_parameters: Iterable[nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    process_group=None,
) -> torch.Tensor:
    """Clip one DDP parameter set plus one FSDP FULL_SHARD parameter set.

    The MQ/Connector gradients are replicated after DDP reduction and must be
    counted once.  Wan gradients are disjoint FULL_SHARD pieces and must first
    be summed across ranks.  Treating both groups as ordinary local parameters
    gives each rank a different norm and is not equivalent to clipping the
    original unsharded model.
    """

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if not torch.isfinite(torch.tensor(max_norm)) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    if norm_type != 2.0:
        raise ValueError("MovieStory distributed clipping supports only L2 norm")

    replicated = list(dict.fromkeys(replicated_parameters))
    sharded = list(dict.fromkeys(sharded_parameters))
    replicated_ids = {id(parameter) for parameter in replicated}
    overlap = [parameter for parameter in sharded if id(parameter) in replicated_ids]
    if overlap:
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
        raise RuntimeError(f"all gradients must share one device, got {devices}")
    device = next(iter(devices))

    def squared_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            value = torch.linalg.vector_norm(
                gradient.detach(),
                ord=2,
                dtype=torch.float32,
            )
            total = total + value.square()
        return total

    replicated_sq = squared_norm(replicated)
    sharded_sq = squared_norm(sharded)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            sharded_sq,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
    total_norm = (replicated_sq + sharded_sq).sqrt()
    if not bool(torch.isfinite(total_norm).item()):
        raise FloatingPointError(
            f"non-finite distributed gradient norm: {float(total_norm.item())}"
        )
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


def install_wan_fsdp_sharder(
    *,
    context_text_len: int,
    activation_checkpointing: bool = True,
) -> None:
    """Install the train-safe FULL_SHARD implementation used by WanTI2V.

    Wan's stock helper uses flattened parameters.  The inherited trainer selects
    ``cond_only`` parameters *after* wrapping, so ``use_orig_params=True`` is
    required to retain their names and individual ``requires_grad`` flags.
    """

    if context_text_len <= 0:
        raise ValueError("context_text_len must be positive")

    import wan.textimage2video as wan_ti2v_module
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
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("Wan model has no transformer blocks to shard")

        if activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )

            for index, block in enumerate(list(blocks)):
                if not bool(
                    getattr(block, "_moviestory_activation_checkpoint", False)
                ):
                    wrapped = checkpoint_wrapper(
                        block,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    )
                    wrapped._moviestory_activation_checkpoint = True
                    blocks[index] = wrapped

        wrapped_blocks = tuple(model.blocks)
        auto_wrap = partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda module: module in wrapped_blocks,
        )
        sharded = FSDP(
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
        sharded._moviestory_context_text_len = int(context_text_len)
        sharded._moviestory_activation_checkpointing = bool(
            activation_checkpointing
        )
        return sharded

    wan_ti2v_module.shard_model = training_shard_model
