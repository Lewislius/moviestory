from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Type

from PIL import Image


def _stable_uint64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def _stable_uniform(*parts: object) -> float:
    return _stable_uint64(*parts) / float(2**64)


def build_random_reference_dataset_class(
    base_dataset_class: Type,
    *,
    joint_null_prob: float = 0.1,
    strict_random_reference: bool = True,
    reference_seed: int | None = None,
) -> Type:
    """Wrap WanVideoDataset with whole-video random reference sampling.

    The target clip is intentionally left unchanged.  Only ``ref_image`` and
    the non-dropped ``mq_ref_image`` are replaced, so the requested augmentation
    cannot silently alter the video supervision frames.
    """

    joint_probability = float(joint_null_prob)
    if not math.isfinite(joint_probability) or not 0.0 <= joint_probability <= 1.0:
        raise ValueError("joint_null_prob must be within [0, 1]")

    class WholeVideoRandomReferenceDataset(base_dataset_class):
        moviestory_random_reference = True
        moviestory_joint_null_prob = joint_probability

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._moviestory_caption_null_prob = float(
                kwargs.get("null_caption_prob", 0.0)
            )
            self._moviestory_image_null_prob = float(
                kwargs.get("null_image_prob", 0.0)
            )
            if not 0.0 <= self._moviestory_caption_null_prob <= 1.0:
                raise ValueError("null_caption_prob must be within [0, 1]")
            if not 0.0 <= self._moviestory_image_null_prob <= 1.0:
                raise ValueError("null_image_prob must be within [0, 1]")

            # Disable the base class's process-global RNG dropout.  Applying it
            # below from a stable sample key keeps results invariant when the
            # same global batch is split over a different GPU topology.
            kwargs["null_caption_prob"] = 0.0
            kwargs["null_image_prob"] = 0.0
            super().__init__(*args, **kwargs)
            self._moviestory_reference_seed = int(
                reference_seed
                if reference_seed is not None
                else kwargs.get("seed", getattr(self, "seed", 42))
            )
            self._moviestory_printed_reference = False

        @staticmethod
        def _resize_rgb(frame_bgr, *, height: int, width: int) -> Image.Image:
            import cv2

            resized = cv2.resize(
                frame_bgr,
                (int(width), int(height)),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb, mode="RGB")

        def _reservoir_reference(
            self,
            video_path: str,
            *,
            key: str,
            height: int,
            width: int,
        ) -> tuple[Image.Image, int, int]:
            """Fallback for containers whose frame-count metadata is missing."""
            import cv2

            capture = cv2.VideoCapture(video_path)
            selected = None
            selected_index = -1
            count = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    count += 1
                    # Deterministic reservoir sampling over the entire stream.
                    replace = _stable_uint64(key, "reservoir", count) % count == 0
                    if replace:
                        selected = frame
                        selected_index = count - 1
            finally:
                capture.release()
            if selected is None or count <= 0:
                raise RuntimeError(f"unable to decode random reference: {video_path}")
            return (
                self._resize_rgb(selected, height=height, width=width),
                selected_index,
                count,
            )

        def _random_reference(
            self,
            video_path: str,
            *,
            key: str,
            height: int,
            width: int,
        ) -> tuple[Image.Image, int, int]:
            import cv2

            capture = cv2.VideoCapture(video_path)
            total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            if total <= 0:
                capture.release()
                return self._reservoir_reference(
                    video_path,
                    key=key,
                    height=height,
                    width=width,
                )

            first_index = _stable_uint64(key, "reference-index") % total
            frame = None
            selected_index = -1
            # Some codecs cannot seek to every predicted frame.  Retry stable,
            # widely separated positions without falling back to frame zero.
            try:
                for retry in range(min(total, 8)):
                    candidate = int(
                        (first_index + retry * max(total // 7, 1)) % total
                    )
                    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                    ok, decoded = capture.read()
                    if ok and decoded is not None:
                        frame = decoded
                        selected_index = candidate
                        break
            finally:
                capture.release()

            if frame is None:
                if strict_random_reference:
                    raise RuntimeError(
                        "whole-video random seek failed for "
                        f"{video_path} (reported_frames={total})"
                    )
                return self._reservoir_reference(
                    video_path,
                    key=key,
                    height=height,
                    width=width,
                )
            return (
                self._resize_rgb(frame, height=height, width=width),
                selected_index,
                total,
            )

        def __getitem__(self, index):
            global_draw_id = int(
                getattr(index, "global_draw_id", int(index))
            )
            base_result = super().__getitem__(index)
            result = dict(base_result)
            video_path = str(Path(result["video_path"]).expanduser().resolve())
            video = result["video"]
            if getattr(video, "ndim", None) != 4:
                raise ValueError(
                    "random-reference dataset expects video [C,T,H,W], got "
                    f"{getattr(video, 'shape', type(video).__name__)}"
                )
            height, width = int(video.shape[-2]), int(video.shape[-1])
            key = (
                f"seed={self._moviestory_reference_seed}|path={video_path}|"
                f"global_draw={global_draw_id}"
            )
            reference, frame_index, total_frames = self._random_reference(
                video_path,
                key=key,
                height=height,
                width=width,
            )

            joint_null = (
                _stable_uniform(key, "joint-null") < joint_probability
            )
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
            result["reference_frame_index"] = int(frame_index)
            result["reference_total_frames"] = int(total_frames)
            result["reference_frame_ratio"] = float(
                frame_index / max(total_frames - 1, 1)
            )
            result["moviestory_joint_null"] = bool(joint_null)
            result["moviestory_global_draw_id"] = int(global_draw_id)
            result["moviestory_sample_seed"] = int(
                _stable_uint64(key, "training-rng") % (2**63 - 1)
            )
            result["moviestory_reference_is_target_first_frame"] = bool(
                frame_index == 0
            )

            if not self._moviestory_printed_reference:
                print(
                    "[RANDOM-REFERENCE] "
                    f"path={video_path} frame={frame_index}/{total_frames} "
                    f"ratio={result['reference_frame_ratio']:.4f} "
                    f"target_frames={int(video.shape[1])} "
                    f"joint_null={int(joint_null)} mq_drop={int(image_null)}",
                    flush=True,
                )
                self._moviestory_printed_reference = True
            return result

    WholeVideoRandomReferenceDataset.__name__ = (
        f"WholeVideoRandomReference{base_dataset_class.__name__}"
    )
    return WholeVideoRandomReferenceDataset
