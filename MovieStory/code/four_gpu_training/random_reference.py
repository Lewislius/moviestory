from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


def bind_clean_reference_prefix(
    x_inputs: Sequence[torch.Tensor],
    reference_latents: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if len(x_inputs) != len(reference_latents):
        raise ValueError(
            "Wan/reference batch mismatch: "
            f"{len(x_inputs)} != {len(reference_latents)}"
        )
    outputs: list[torch.Tensor] = []
    for index, (x_input, reference) in enumerate(
        zip(x_inputs, reference_latents)
    ):
        if x_input.ndim != 4 or reference.ndim != 4:
            raise ValueError(
                f"sample {index}: expected [C,T,H,W], got "
                f"{tuple(x_input.shape)} and {tuple(reference.shape)}"
            )
        if x_input.shape[0] != reference.shape[0] or x_input.shape[2:] != reference.shape[2:]:
            raise ValueError(
                f"sample {index}: incompatible reference {tuple(reference.shape)} "
                f"for Wan input {tuple(x_input.shape)}"
            )
        prefix = int(reference.shape[1])
        if prefix <= 0 or prefix >= int(x_input.shape[1]):
            raise ValueError(
                f"sample {index}: invalid prefix={prefix} for {tuple(x_input.shape)}"
            )
        clean = reference.to(device=x_input.device, dtype=x_input.dtype)
        outputs.append(torch.cat([clean, x_input[:, prefix:]], dim=1))
    return outputs


def resolve_direct_random_references(batch: Mapping[str, Any]) -> list[Any]:
    references = batch.get("ref_image")
    videos = batch.get("video")
    if videos is None:
        raise KeyError("random-reference conditioning requires batch['video']")
    batch_size = len(videos)
    if references is None or len(references) != batch_size:
        raise ValueError(
            "random-reference conditioning requires one ref_image per video"
        )
    resolved = []
    for index, reference in enumerate(references):
        if reference is None:
            raise ValueError(f"sample {index}: random ref_image is missing")
        resolved.append(
            reference.convert("RGB")
            if isinstance(reference, Image.Image)
            else reference
        )
    return resolved


class RandomReferenceTrainingMixin:
    """Inject a random whole-video frame as a clean Wan reference prefix.

    Unlike the legacy first-frame mixin, this class does *not* remove target
    latent slot zero: a random reference is not a duplicate of the target's
    first frame.  The reference prefix is timestep zero and excluded from loss
    by the inherited ``wan_animate_slot`` implementation.
    """

    def _encode_ref_image_to_latent(
        self,
        ref_img,
        latent_h: int,
        latent_w: int,
        z_channels: int,
    ):
        if getattr(self, "_moviestory_random_reference_active", False):
            ref_index = int(getattr(self, "_moviestory_ref_encode_index", 0))
            references = getattr(self, "_moviestory_wan_references", ())
            if ref_index >= len(references):
                raise RuntimeError("random reference queue was exhausted")
            ref_img = references[ref_index]
            self._moviestory_ref_encode_index = ref_index + 1

        latent = super()._encode_ref_image_to_latent(
            ref_img,
            latent_h,
            latent_w,
            z_channels,
        )
        if getattr(self, "_moviestory_random_reference_active", False):
            self._moviestory_reference_latents.append(latent)
        return latent

    def _moviestory_random_ref_forward_pre_hook(self, module, args, kwargs):
        del module
        if not getattr(self, "_moviestory_random_reference_active", False):
            return None
        if not self._moviestory_reference_latents:
            raise RuntimeError("Wan forward occurred before reference VAE encoding")
        if args:
            bound = bind_clean_reference_prefix(
                args[0],
                self._moviestory_reference_latents,
            )
            return (bound, *args[1:]), kwargs
        if "x" not in kwargs:
            raise RuntimeError("unable to locate Wan x input for reference binding")
        updated = dict(kwargs)
        updated["x"] = bind_clean_reference_prefix(
            kwargs["x"],
            self._moviestory_reference_latents,
        )
        return args, updated

    @contextlib.contextmanager
    def _moviestory_sample_rng(self, batch: Mapping[str, Any]) -> Iterator[None]:
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
                device = getattr(self, attribute, None)
                if isinstance(device, torch.device) and device.type == "cuda":
                    index = torch.cuda.current_device() if device.index is None else device.index
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

    def _compute_loss(self, batch):
        references = resolve_direct_random_references(batch)
        self._moviestory_random_reference_active = True
        self._moviestory_wan_references = references
        self._moviestory_ref_encode_index = 0
        self._moviestory_reference_latents: list[torch.Tensor] = []
        hook = self.wan.model.register_forward_pre_hook(
            self._moviestory_random_ref_forward_pre_hook,
            with_kwargs=True,
        )
        try:
            # Batch size is deliberately fixed at one.  Deriving all model-side
            # randomness from the sampler's global draw id keeps Qwen dropout,
            # flow timestep and diffusion noise independent of rank placement.
            with self._moviestory_sample_rng(batch):
                loss = super()._compute_loss(batch)
            if self._moviestory_ref_encode_index != len(references):
                raise RuntimeError(
                    "not every random reference was VAE encoded: "
                    f"{self._moviestory_ref_encode_index}/{len(references)}"
                )
            if len(self._moviestory_reference_latents) != len(references):
                raise RuntimeError("reference latent count does not match batch")
            return loss
        finally:
            hook.remove()
            self._moviestory_random_reference_active = False
            self._moviestory_wan_references = []
            self._moviestory_reference_latents = []
            self._moviestory_ref_encode_index = 0
