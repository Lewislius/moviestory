"""Four-GPU MovieStory training components.

This package is deliberately separate from the legacy two-GPU entrypoint so a
working 2 x 96 GiB run is never changed in place.
"""

from .conditioning import MQToT5Mapper, build_dual_mode_encoder_class
from .data import build_random_reference_dataset_class
from .distributed import (
    GlobalBatchEquivalentSampler,
    SampleDrawIndex,
    clip_grad_norm_mixed_sharded_,
    install_equivalent_dataloader,
    install_wan_fsdp_sharder,
)
from .random_reference import RandomReferenceTrainingMixin

__all__ = [
    "GlobalBatchEquivalentSampler",
    "MQToT5Mapper",
    "RandomReferenceTrainingMixin",
    "SampleDrawIndex",
    "build_dual_mode_encoder_class",
    "build_random_reference_dataset_class",
    "clip_grad_norm_mixed_sharded_",
    "install_equivalent_dataloader",
    "install_wan_fsdp_sharder",
]
