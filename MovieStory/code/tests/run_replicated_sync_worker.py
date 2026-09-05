"""Two-rank smoke test for replicated parameters kept outside Connector FSDP."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from native_i2v_3router.distributed import (
    broadcast_replicated_parameters_,
    shard_mq_connector_fsdp,
    sync_replicated_gradients_,
)


class TinyConnectorEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(24)])


class TinyMQEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mllm_model = nn.Module()
        self.mllm_model.connector = nn.Sequential(
            TinyConnectorEncoder(), nn.Linear(4, 4)
        )


def main() -> None:
    dist.init_process_group("gloo", init_method="env://")
    rank = dist.get_rank()
    parameter = torch.nn.Parameter(torch.tensor([float(rank + 1)]))
    broadcast_replicated_parameters_([parameter], src=0)
    if parameter.item() != 1.0:
        raise AssertionError(f"rank {rank}: broadcast produced {parameter.item()}")

    parameter.grad = torch.tensor([float(rank + 1)])
    sync_replicated_gradients_([parameter])
    if parameter.grad.item() != 1.5:
        raise AssertionError(f"rank {rank}: gradient mean produced {parameter.grad.item()}")

    encoder = TinyMQEncoder()
    connector = shard_mq_connector_fsdp(
        encoder,
        device_id=torch.device("cpu"),
        sync_module_states=False,
    )
    counts = connector._moviestory_connector_shard_numel
    full_numel = connector._moviestory_connector_full_numel
    if len(counts) != dist.get_world_size() or sum(counts) != full_numel:
        raise AssertionError(f"rank {rank}: invalid FSDP shard audit {counts}/{full_numel}")
    if max(counts) >= full_numel:
        raise AssertionError(f"rank {rank}: Connector is still fully replicated")
    if rank == 0:
        print(
            "24-layer Connector FULL_SHARD + replicated parameter sync: ok",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
