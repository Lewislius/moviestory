from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image


def configure_wan_first_frame_strong_binding(
    args: Any,
    *,
    enabled: bool,
) -> Any:
    """
    Configure a flow-consistent first-frame conditioning contract.

    Strong binding prepends a clean reference slot, gives it timestep zero,
    removes the duplicate first target latent, and excludes the clean
    conditioning prefix from the ground-truth video denoising loss.

    Disabling strong binding is retained only as a flow-consistent ablation: it
    uses ordinary T2V flow matching with no first-frame latent anchor.  The old
    ``animate_like`` path is deliberately not configured because changing
    ``x_t`` after constructing it while retaining the original velocity target
    is mathematically inconsistent.
    """
    args.moviestory_wan_first_frame_strong_bind = bool(enabled)
    if not enabled:
        args.enable_ti2v_first_frame_condition = False
        args.train_video_conditioning_mode = "legacy_t2v"
        args.train_animate_ref_frames = 0
        args.train_animate_temporal_frames = 0
        args.train_animate_conditional_frames = 0
        args.train_animate_preserve_timestep_zero = False
        args.train_animate_drop_prefix_loss = False
        args.train_ref_anchor_mode = "none"
        args.train_ref_anchor_alpha0 = 0.0
        args.train_ref_anchor_warmup_ratio = 0.0
        return args

    args.enable_ti2v_first_frame_condition = True
    args.train_video_conditioning_mode = "wan_animate_slot"
    args.train_animate_ref_frames = 1
    args.train_animate_temporal_frames = 0
    args.train_animate_conditional_frames = 0
    args.train_animate_preserve_timestep_zero = True
    args.train_animate_drop_prefix_loss = True
    args.train_ref_anchor_mode = "none"
    args.train_ref_anchor_alpha0 = 0.0
    args.train_ref_anchor_warmup_ratio = 0.0
    return args


def remove_first_target_latent_slot(
    latents: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Remove the target video's first latent slot.

    The clean VAE-encoded reference is prepended separately, so retaining the
    target's original first slot would duplicate the same frame and increase
    the temporal sequence length by one.
    """
    trimmed: list[torch.Tensor] = []
    for index, latent in enumerate(latents):
        if latent.ndim != 4:
            raise ValueError(
                "video latent must have shape [C, T, H, W], "
                f"got sample {index}: {tuple(latent.shape)}"
            )
        if latent.shape[1] <= 1:
            raise ValueError(
                "strong first-frame binding needs at least two video latent "
                f"slots, got sample {index}: {tuple(latent.shape)}"
            )
        trimmed.append(latent[:, 1:].contiguous())
    return trimmed


def bind_clean_reference_prefix(
    x_inputs: Sequence[torch.Tensor],
    reference_latents: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Replace each Wan input prefix with its clean reference latent.

    A new tensor is returned for every sample; the caller's noisy input is not
    mutated.  The reference may contain one or more prefix slots, although the
    current MovieStory configuration always uses exactly one.
    """
    if len(x_inputs) != len(reference_latents):
        raise ValueError(
            "Wan inputs/reference latents batch mismatch: "
            f"{len(x_inputs)} != {len(reference_latents)}"
        )

    bound_inputs: list[torch.Tensor] = []
    for index, (x_input, reference) in enumerate(
        zip(x_inputs, reference_latents)
    ):
        if x_input.ndim != 4 or reference.ndim != 4:
            raise ValueError(
                "Wan input and reference latent must have shape [C, T, H, W], "
                f"got sample {index}: {tuple(x_input.shape)} / "
                f"{tuple(reference.shape)}"
            )
        if (
            x_input.shape[0] != reference.shape[0]
            or x_input.shape[2:] != reference.shape[2:]
        ):
            raise ValueError(
                f"sample {index} reference shape is incompatible with Wan input: "
                f"{tuple(reference.shape)} vs {tuple(x_input.shape)}"
            )
        prefix_slots = int(reference.shape[1])
        if prefix_slots <= 0 or prefix_slots >= int(x_input.shape[1]):
            raise ValueError(
                f"sample {index} invalid reference prefix length "
                f"{prefix_slots} for Wan input {tuple(x_input.shape)}"
            )

        clean_reference = reference.to(
            device=x_input.device,
            dtype=x_input.dtype,
        )
        bound_inputs.append(
            torch.cat([clean_reference, x_input[:, prefix_slots:]], dim=1)
        )
    return bound_inputs


def video_start_frame_to_reference_image(
    video_tensor: torch.Tensor,
) -> Image.Image:
    """Convert a normalized ``[C, T, H, W]`` video's first frame to RGB PIL."""
    if not torch.is_tensor(video_tensor) or video_tensor.ndim != 4:
        shape = (
            tuple(video_tensor.shape)
            if torch.is_tensor(video_tensor)
            else type(video_tensor).__name__
        )
        raise ValueError(
            "video must be a [C, T, H, W] tensor to derive a reference image, "
            f"got {shape}"
        )
    if video_tensor.shape[0] != 3 or video_tensor.shape[1] < 1:
        raise ValueError(
            "video must contain at least one RGB frame, "
            f"got {tuple(video_tensor.shape)}"
        )

    frame = video_tensor[:, 0].detach().float().cpu()
    if not torch.isfinite(frame).all():
        raise ValueError("video first frame contains non-finite values")
    frame_min = float(frame.min().item())
    frame_max = float(frame.max().item())
    if frame_min >= -1.01 and frame_max <= 1.01:
        frame = (frame + 1.0) * 127.5
    frame = frame.clamp(0.0, 255.0).round().to(torch.uint8)
    return Image.fromarray(frame.permute(1, 2, 0).numpy(), mode="RGB")


def resolve_wan_reference_images(
    batch: Mapping[str, Any],
) -> list[Any]:
    """
    Resolve one Wan reference per video.

    Prefer the direct ``ref_image``, then the MetaQuery reference.  If neither
    exists for a sample, reconstruct the reference from video frame zero.
    """
    videos = batch.get("video")
    if videos is None:
        raise KeyError(
            "strong first-frame binding requires batch['video'] when a "
            "reference image is missing"
        )
    videos = list(videos)
    direct_refs = batch.get("ref_image")
    mq_refs = batch.get("mq_ref_image")
    direct_refs = [None] * len(videos) if direct_refs is None else list(direct_refs)
    mq_refs = [None] * len(videos) if mq_refs is None else list(mq_refs)
    if len(direct_refs) != len(videos) or len(mq_refs) != len(videos):
        raise ValueError(
            "video/reference batch size mismatch: "
            f"video={len(videos)} ref_image={len(direct_refs)} "
            f"mq_ref_image={len(mq_refs)}"
        )

    resolved: list[Any] = []
    for video, direct_ref, mq_ref in zip(videos, direct_refs, mq_refs):
        reference = direct_ref if direct_ref is not None else mq_ref
        if reference is None:
            reference = video_start_frame_to_reference_image(video)
        resolved.append(
            reference.convert("RGB")
            if isinstance(reference, Image.Image)
            else reference
        )
    return resolved


class StrongFirstFrameTrainingMixin:
    """
    Add exact Wan-side first-frame conditioning to the inherited trainer.

    During one loss computation this mixin:
      1. uses ``batch["ref_image"]`` as the Wan reference independently of MQ
         image dropout;
      2. removes the first target-video latent slot;
      3. captures the separately VAE-encoded reference latent; and
      4. replaces the noised prefix with that clean latent immediately before
         every Wan forward.

    The inherited ``wan_animate_slot`` path supplies timestep=0 and drops the
    prefix from the denoising loss.
    """

    def _moviestory_strong_binding_enabled(self) -> bool:
        return bool(
            getattr(
                self.args,
                "moviestory_wan_first_frame_strong_bind",
                False,
            )
        )

    def _encode_video(self, video_tensors):
        latents = super()._encode_video(video_tensors)
        if (
            not self._moviestory_strong_binding_enabled()
            or not getattr(self, "_moviestory_binding_active", False)
        ):
            return latents
        return remove_first_target_latent_slot(latents)

    def _encode_ref_image_to_latent(
        self,
        ref_img,
        latent_h: int,
        latent_w: int,
        z_channels: int,
    ):
        if getattr(self, "_moviestory_binding_active", False):
            ref_index = int(getattr(self, "_moviestory_ref_encode_index", 0))
            wan_ref_images = getattr(self, "_moviestory_wan_ref_images", ())
            if ref_index >= len(wan_ref_images):
                raise RuntimeError(
                    "Wan reference-image queue was exhausted during strong binding"
                )
            ref_img = wan_ref_images[ref_index]
            self._moviestory_ref_encode_index = ref_index + 1

        reference = super()._encode_ref_image_to_latent(
            ref_img,
            latent_h,
            latent_w,
            z_channels,
        )
        if getattr(self, "_moviestory_binding_active", False):
            self._moviestory_reference_latents.append(reference)
        return reference

    def _moviestory_bind_forward_pre_hook(self, module, args, kwargs):
        del module
        if not getattr(self, "_moviestory_binding_active", False):
            return None
        if not self._moviestory_reference_latents:
            raise RuntimeError(
                "Wan forward reached before the reference latent was encoded"
            )

        if args:
            x_inputs = args[0]
            remaining_args = args[1:]
            bound = bind_clean_reference_prefix(
                x_inputs,
                self._moviestory_reference_latents,
            )
            return (bound, *remaining_args), kwargs

        if "x" not in kwargs:
            raise RuntimeError("Unable to locate Wan x inputs for strong binding")
        updated_kwargs = dict(kwargs)
        updated_kwargs["x"] = bind_clean_reference_prefix(
            kwargs["x"],
            self._moviestory_reference_latents,
        )
        return args, updated_kwargs

    def _compute_loss(self, batch):
        if not self._moviestory_strong_binding_enabled():
            return super()._compute_loss(batch)

        wan_ref_images = resolve_wan_reference_images(batch)
        working_batch = batch
        if batch.get("mq_ref_image") is None:
            # Keep explicit per-sample MQ image dropout intact.  This only
            # supplies the derived first frame when the entire field is absent.
            working_batch = dict(batch)
            working_batch["mq_ref_image"] = wan_ref_images
        if working_batch.get("ref_image") is None:
            if working_batch is batch:
                working_batch = dict(batch)
            working_batch["ref_image"] = wan_ref_images

        self._moviestory_binding_active = True
        self._moviestory_wan_ref_images = wan_ref_images
        self._moviestory_ref_encode_index = 0
        self._moviestory_reference_latents: list[torch.Tensor] = []
        hook = self.wan.model.register_forward_pre_hook(
            self._moviestory_bind_forward_pre_hook,
            with_kwargs=True,
        )
        try:
            return super()._compute_loss(working_batch)
        finally:
            hook.remove()
            self._moviestory_binding_active = False
            self._moviestory_wan_ref_images = []
            self._moviestory_reference_latents = []
            self._moviestory_ref_encode_index = 0
